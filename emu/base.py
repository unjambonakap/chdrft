import re
import sys
import subprocess as sp
import struct
import mmap
import array
import binascii
import tempfile
from inspect import signature
import chdrft.utils.misc as cmisc
from contextlib import ExitStack
from chdrft.tools.xxd import xxd
from chdrft.struct.base import Intervals

from chdrft.utils.misc import Attributize, lowercase_norm, Arch, struct_helper, csv_list


# contract
class MemReader:

  def __call__(self, ptr, sz):
    assert 0

x86_eflags_def = '''
eflags_t: {
  atom_size: 1,
  fields: [
    {name: 'cf', bitsize: 1},
    {name: 'fixed1', bitsize: 1},
    {name: 'pf', bitsize: 1},
    {name: 'res1', bitsize: 1},
    {name: 'af', bitsize: 1},
    {name: 'res2', bitsize: 1},
    {name: 'zf', bitsize: 1},
    {name: 'sf', bitsize: 1},
    {name: 'tf', bitsize: 1},
    {name: 'if', bitsize: 1},
    {name: 'df', bitsize: 1},
    {name: 'of', bitsize: 1},
    {name: 'iopl', bitsize: 2},
    {name: 'of', bitsize: 1},
  ],
}
''' #finish this?

eflags = cmisc.BitMapper(
    cmisc.to_list('cf fixed1 pf res1 af res2 zf sf tf if df of iopl0 iopl1 of')
)


class BitFieldWrapper:

  def __init__(self, bmap, getter=None, setter=None, gettersetter=None):
    if gettersetter is not None:
      getter = gettersetter.getter
      setter = gettersetter.setter

    self.__dict__.update(bmap=bmap, getter=getter, setter=setter)

  def __str__(self):
    return self.bmap.from_value(self.getter())

  def __setattr__(self, name, val):
    if val:
      self.setter(self.getter() | (1 << self.bmap.field_to_pos[name]))
    else:
      self.setter(self.getter() & ~(1 << self.bmap.field_to_pos[name]))

  def __getattr__(self, name):
    return self.getter() >> self.bmap.field_to_pos[name] & 1


class FieldGetSet:

  def __init__(self, obj, field):
    self.obj = obj
    self.field = field

  def setter(self, v):
    setattr(self.obj, self.field, v)

  def getter(self):
    return getattr(self.obj, self.field)


class BufferMemReader(MemReader):

  def __init__(self, buf, base_addr=0):
    self.buf = buf
    self.base_addr = base_addr

  def __call__(self, ptr, sz):
    assert ptr >= self.base_addr and ptr + sz <= self.base_addr + len(
        self.buf
    ), 'Bad shit ptr=%x, base=%x, sz=%x, len=%x' % (ptr, self.base_addr, sz, len(self.buf))
    ptr -= self.base_addr
    return self.buf[ptr:ptr + sz]


class RegExtractor(Attributize):

  def __init__(self, const_store, prefix):
    super().__init__(key_norm=lowercase_norm)
    super(dict, self).__setattr__('_inv', {})
    for k, v in const_store.__dict__.items():
      if not k.startswith(prefix):
        continue
      name = k[len(prefix):]
      self._inv[v] = lowercase_norm(name)
      self[name] = v

  def get_name(self, v):
    return self._inv[v]


class Regs:

  def __init__(self, arch, do_normalize=1, **kwargs):
    redirect = dict(ins_pointer=arch.reg_pc, stack_pointer=arch.reg_stack)
    if 'redirect' in arch: redirect.update(arch.redirect)

    if 'flags_desc' in arch:
      desc, field = arch.flags_desc
      gs = FieldGetSet(self, field)
      self.__dict__.update(flags=BitFieldWrapper(desc, gettersetter=gs))
    self.__dict__.update(arch=arch, do_normalize=do_normalize, redirect=redirect, **kwargs)

  def reg_list(self):
    return self.arch.regs

  def __getattr__(self, name):
    return self.get(name)

  def __setattr__(self, name, val):
    self.set(name, val)

  def __contains__(self, name):
    name = self.normalize(name)
    return self._has(name)

  def set(self, name, val):
    name = self.normalize(name)
    val = self.normalize_val(name, val)
    self.set_register(name, val)

  def get(self, name):
    name = self.normalize(name)
    return self.get_register(name)

  def normalize(self, name):
    if self.arch.get('do_normalize', 1):
      name = lowercase_norm(name)
    if name in self.redirect:
      name = self.redirect[name]
    return name

  def normalize_val(self, name, val):
    if isinstance(val, bytes):
      val = struct_helper.get(val, self.reg_size(name))
    return val

  def __getitem__(self, name):
    return self.get(name)

  def __setitem__(self, name, val):
    self.__setattr__(name, val)

  def reg_size(self, name):
    return self.arch.reg_size

  def get_register(self, name):
    assert 0

  def set_register(self, name, val):
    assert 0

  def _has(self, name):
    assert 0

  def snapshot_all_but_pc(self):
    res = Attributize()
    for a in self.arch.regs:
      if a == self.arch.reg_pc: continue
      res[a] = self[a]
    return res

  def snapshot_all(self):
    return self.snapshot(self.arch.regs)

  def snapshot(self, want, failsafe=0):
    res = {}
    for x  in want:
      try:
        res[x] = self[x]
      except:
        if not failsafe:
          raise
    return cmisc.Attr(res)

  def context(self, lst):
    return RegContext(self, lst)


class RegContext(ExitStack):

  def __init__(self, regs, lst):
    super().__init__()
    self.lst = cmisc.to_list(lst)
    self.vals = {}
    self.regs = regs

  def __enter__(self):
    super().__enter__()
    self.callback(self.restore)
    for k in self.lst:
      self.vals[k] = self.regs[k]
    return self

  def restore(self):
    for k, v in self.vals.items():
      self.regs[k] = v


class Memory:

  def __init__(self, reader=None, writer=None, arch=None, minv=None, maxv=None, le=True, **kwargs):
    assert arch is not None
    if isinstance(reader, bytes) or isinstance(reader, bytearray):
      reader = BufferMemReader(reader)

    from chdrft.emu.binary import arch_data
    if isinstance(arch, Arch): arch = arch_data[arch]
    self.reader = reader
    self.writer = writer
    self.arch = arch
    self.minv = minv
    self.maxv = maxv
    types = []
    types.append('Q 64')
    types.append('I 32')
    types.append('H 16')
    types.append('B 8')
    self.le_sgn = '><' [le]
    self.le = le
    #size to char
    self.s2c = {}

    self.read_funcs = []
    self.write_funcs = []

    self.add_rw_funcs('f32', 'f')
    self.add_rw_funcs('f64', 'd')

    for x in types:
      l, val = x.split(' ')
      val = int(val)
      self.s2c[val // 8] = self.le_sgn + l

      for lx in (l + 'u', l.lower() + 's'):
        self.add_rw_funcs('%c%d' % (lx[1], val), lx[0])

    self.ptr_size = self.arch.reg_size
    if self.ptr_size == 8:
      suf = 'u64'
    elif self.ptr_size == 4:
      suf = 'u32'
      self.read_ptr = self.read_u32
      self.write_ptr = self.write_u32
    elif self.ptr_size == 2:
      suf = 'u16'
    else:
      assert 0
    for typ in csv_list('read_,read_n,write_,write_n'):
      fname = typ + 'ptr'
      setattr(self, fname, getattr(self, typ + suf))

    self.read_funcs += csv_list('read,read2,unpack,read_u')
    self.write_funcs += csv_list('write,write_u')

  def add_read_func(self, name, func):
    self.read_funcs.append(name)
    setattr(self, name, func)

  def add_write_func(self, name, func):
    self.write_funcs.append(name)
    setattr(self, name, func)

  def get_ins_str(self, addr):
    return self.arch.mc.ins_str(self.get_ins(addr))

  def get_ins(self, addr):
    assert self.arch.ins_size != -1
    data = self.read(addr, self.arch.ins_size)
    return self.arch.mc.get_one_ins(data, addr)

  def add_rw_funcs(self, prefix, fmt):
    g, s, ng, ns = self.make_funcs(fmt)
    read_name = 'read_%s' % (prefix)
    read_n_name = 'read_n%s' % (prefix)
    write_n_name = 'write_n%s' % (prefix)
    write_name = 'write_%s' % (prefix)
    self.add_read_func(read_name, g)
    self.add_read_func(read_n_name, ng)
    self.add_write_func(write_name, s)
    self.add_write_func(write_n_name, ns)

  def make_funcs(self, fmt):
    fmt_size = struct.calcsize(fmt)

    def g(addr, ):
      y = self.reader(addr, fmt_size)
      #if y is not None: y = self.reader(addr, fmt_size * n)
      return struct.unpack(self.le_sgn + '%c' % fmt, y)[0]

    def ng(addr=None, n=None, sz=None, y=None):
      if sz is not None: n = sz // fmt_size
      if y is None: y = self.reader(addr, fmt_size * n)
      return struct.unpack(self.le_sgn + '%d%c' % (n, fmt), y)

    def s(addr, val):
      packed_val = struct.pack(self.le_sgn + '%c' % fmt, val)
      self.writer(addr, packed_val)

    def ns(addr, tb):
      packed_val = struct.pack(self.le_sgn + '%d%c' % (len(tb), fmt), *tb)
      self.writer(addr, packed_val)

    return g, s, ng, ns

  def xxd(self, addr, sz, word_size=None, endian='little', num_cols=4):
    if word_size is None:
      word_size = self.arch.reg_size
    return xxd.FromBuf(
        self.read(addr, sz),
        cmisc.Attributize(
            reverse=0,
            num_cols=num_cols,
            head=-1,
            skip_head=0,
            word_size=word_size,
            endian=endian,
            offset=addr,
        )
    )

  # read by size, automatically cast to number
  def read_u(self, addr, size):
    assert size in self.s2c
    res = self.reader(addr, size)
    return struct.unpack(self.s2c[size], res)[0]

  # read by size, automatically cast to number
  def read_n_u(self, addr, size, count):
    for i in range(count):
      yield self.read_u(addr+size*i, size)

  def write_u(self, v, size):
    x = struct.pack(self.s2c[size], v)
    self.reader(x, size)

  def read2(self, addr, n):
    if n in self.s2c:
      return self.read_u(addr, n)
    return self.read(addr, n)

  def read(self, addr, n):
    return self.reader(addr, n)

  def write(self, addr, buf):
    if isinstance(buf, bytearray):
      buf = bytes(buf)
    return self.writer(addr, buf)

  def unpack(self, addr, fmt):
    size = struct.calcsize(fmt)
    cur = self.read(addr, size)
    res = struct.unpack(cur, fmt)
    return res

  def bzero(self, addr, sz):
    self.write(addr, b'\x00' * sz)

  def get_str(self, addr, n=0x100):
    s = b''
    while True:
      print('OOON ', addr, n)
      x = self.read(addr, n)
      for j in range(len(x)):
        if x[j] == 0:
          s += x[:j]
          return s
      s += x
      addr += len(x)
    return s


class AtomMem(Memory):

  def __init__(self, atom_size=4, use_int=1, read_atom=None, write_atom=None, **kwargs):
    super().__init__(reader=self._read_impl, writer=self._write_impl, **kwargs)
    self.use_int=use_int
    self.atom_size = atom_size
    self.read_atom = read_atom
    self.write_atom = write_atom

  def _read_impl(self, addr, sz):
    res = bytearray()
    prefix = addr % self.atom_size
    for i in range(0, prefix + sz, self.atom_size):
      res += self.to_buf(self.read_atom(addr + i))
    return res[prefix:prefix + sz]


  def _write_impl(self, addr, content):
    assert len(content) % self.atom_size == 0
    assert addr % self.atom_size == 0
    for i in range(0, len(content), self.atom_size):
      self.write_atom(addr + i, self.from_buf(content[i:i + self.atom_size]))

  def to_buf(self, v):
    if isinstance(v, int): return struct_helper.set(v, self.atom_size)
    return self
  def from_buf(self, v):
    if self.use_int: return struct_helper.get(v, self.atom_size)
    return v

class BufMem(Memory):

  def __init__(self, buf, base_addr=0, **kwargs):
    self.buf = buf
    self.base_addr = base_addr
    super().__init__(reader=self._read, writer=self._write, **kwargs)

  def to_hex(self):
    res = []
    for i in range(0, len(self.buf), self.ptr_size):
      res.append(self.read_ptr(i))
    return '\n'.join(['{:08x}'.format(x) for x in res])

  def _read(self, ptr, sz):
    assert ptr >= self.base_addr and ptr + sz <= self.base_addr + len(
        self.buf
    ), 'Bad shit ptr=%x, base=%x, sz=%x, len=%x' % (ptr, self.base_addr, sz, len(self.buf))
    ptr -= self.base_addr
    return self.buf[ptr:ptr + sz]

  def _write(self, ptr, content):
    ptr -= self.base_addr
    self.buf[ptr:ptr + len(content)] = content

class StreamRW(BufMem):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._pos = 0

  def _read(self, ptr, sz):
    res = super()._read(ptr+self._pos, sz)
    self._pos += len(res)
    return res

  def _write(self, ptr, content):
    super()._write(ptr+self._pos, content)
    self._pos += len(content)

class NonRandomMemory(Memory):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    def tmp(old):
      return lambda *_args: old(0, *_args)

    for x in self.read_funcs + self.write_funcs:
      old = getattr(self, x)
      setattr(self, x, tmp(old))


class Stack:

  def __init__(self, regs, mem, arch, obj=None, fixed_sp=None):
    self.regs = regs
    self.fixed_sp = fixed_sp
    self.mem = mem
    self.arch = arch
    self.obj = obj

  def _write_one(self, addr, e):
    self.mem.write_ptr(addr, e)

  def _read_one(self, addr):
    return self.mem.read_ptr(addr)

  @property
  def sp(self):
    if self.fixed_sp: return self.fixed_sp
    return self.regs.stack_pointer

  @sp.setter
  def sp(self, v):
    if self.fixed_sp:
      self.fixed_sp = v
    else:
      self.regs.stack_pointer = v

  def get_addr(self, off=0):
    addr = self.sp
    if self.obj and self.obj.real_mode:
      addr = self.obj.stack_lin(addr)
    addr += off * self.arch.reg_size
    return addr

  def push(self, e):
    if isinstance(e, int):
      self.sp -= self.arch.reg_size
      self.set(0, e)
    else:
      self.sp -= len(e)
      self.mem.write(self.get_addr(), e)

  def pop(self):
    res = self.get(0)
    self.sp += self.arch.reg_size
    return res

  def popn(self, n):
    res = []
    for i in range(n):
      res.append(self.pop())
    return res

  def get(self, off):
    return self._read_one(self.get_addr(off))

  def set(self, off, v):
    return self._write_one(self.get_addr(off), v)


def binary_re(*arg):
  fmt = arg[0]
  order = '@'
  if fmt[0] in '@=<>!':
    order = fmt[0]
    fmt = fmt[1:]

  num = None
  cur = b''
  pos = 1
  for i in fmt:
    v = ord(i) - ord('0')
    if v >= 0 and v <= 9:
      if num is None:
        num = 0
      num = num * 10 + v
    else:
      if num is None:
        num = 1
      if i == b'x':
        cur += b'.{%d}' % num
      else:
        #print(arg, num, pos, arg[pos:pos + num])
        cur += re.escape(struct.pack('%c%d%c' % (order, num, i), *arg[pos:pos + num]))
        pos += num
      num = None
  return cur


class SimpleAllocator(ExitStack):

  def __init__(self, alloc=None, free=None, bufcl=None):
    super().__init__()
    self.alloc = alloc
    self.free = free
    self.bufcl = bufcl

  def __call__(self, sz):
    addr = self.alloc(sz)
    self.callback(self.free, addr)
    if self.bufcl is None:
      return addr
    return addr, self.bufcl(addr, sz, allocator=self)


class DummyAllocator(SimpleAllocator):

  def __init__(self, addr, sz, bufcl=None):
    super().__init__(alloc=self.alloc_func, free=self.free_func, bufcl=bufcl)
    self.addr = addr
    self.sz = sz
    self.pos = 0

  def alloc_func(self, sz):
    ret = self.addr + self.pos
    self.pos += (sz + 16) & ~15
    self.pos += 16
    assert self.pos <= self.sz
    return ret

  def free_func(self, addr):
    print('KAPPA freeing ', addr)


class SegmentedMemory(Memory):

  def __init__(self, mem_segs, **kwargs):
    super().__init__(reader=self._read, writer=self._write, **kwargs)
    self.mem_segs = mem_segs

  def _read(self, ptr, sz):
    res =  self.mem_segs.query_data_do(ptr, lambda obj, pos: obj.read(pos, sz), fail_if_not=1)
    return res

  def _write(self, ptr, content):
    return  self.mem_segs.query_data_do(ptr, lambda obj, pos: obj.write(pos, content), fail_if_not=1)



def snapshot_mem(mem, segs):
  res = []
  for seg in segs:
    content = mem.read(seg.low, seg.n)
    res.append((seg, content))
  return cmisc.Attr(data=res, arch=mem.arch.typ)

def seg_mem_from_snapshot(snapshot):
  from chdrft.emu.binary import arch_data, Arch
  intervals = Intervals(use_range_data=1, merge=0, is_int=1)
  arch = arch_data[snapshot.arch]
  sm = SegmentedMemory(intervals, arch=arch)
  for seg, content in snapshot.data:
    intervals.add(seg.low,n=seg.n, data=BufMem(bytearray(content), arch=arch))
  return sm

