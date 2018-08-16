import re
import sys
import subprocess as sp
import struct
import mmap
import array
import binascii
import tempfile
from inspect import signature

from chdrft.utils.misc import Attributize, lowercase_norm, Arch, struct_helper, csv_list


# contract
class MemReader:

  def __call__(self, ptr, sz):
    assert 0


class BufferMemReader(MemReader):

  def __init__(self, buf, base_addr=0):
    self.buf = buf
    self.base_addr = base_addr

  def __call__(self, ptr, sz):
    assert ptr >= self.base_addr and ptr + sz <= self.base_addr + len(
        self.buf), 'Bad shit ptr=%x, base=%x, sz=%x, len=%x' % (
            ptr, self.base_addr, sz, len(self.buf))
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


class Regs(object):

  def __init__(self, arch, **kwargs):
    self.__dict__.update(arch=arch, **kwargs)

  def __getattr__(self, name):
    return self.get(name)

  def __setattr__(self, name, val):
    self.set(name, val)

  def set(self, name, val):
    name = self.normalize(name)
    val = self.normalize_val(name, val)
    self.set_register(name, val)

  def get(self, name):
    name = self.normalize(name)
    return self.get_register(name)

  def normalize(self, name):
    return lowercase_norm(name)

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

  def snapshot_all_but_pc(self):
    res = Attributize()
    for a in self.arch.regs:
      if a == self.arch.reg_pc: continue
      res[a] = self[a]
    return res
  def snapshot_all(self):
    return Attributize({x: self[x] for x in self.arch.regs})


class Memory:

  def __init__(self,
               reader=None,
               writer=None,
               arch=Arch.x86_64,
               minv=None,
               maxv=None,
               le=True
               ):
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
    self.le_sgn = '><'[le]
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
      self.read_ptr = self.read_u64
      self.write_ptr = self.write_u64
    else:
      self.read_ptr = self.read_u32
      self.write_ptr = self.write_u32
    self.read_funcs += csv_list('read,read2,unpack,read_u,read_ptr')
    self.write_funcs += csv_list('write,write_u,write_ptr')

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
    g, s, ng = self.make_funcs(fmt)
    read_name = 'read_%s' % (prefix)
    read_n_name = 'read_n%s' % (prefix)
    write_name = 'write_%s' % (prefix)
    self.add_read_func(read_name, g)
    self.add_read_func(read_n_name, ng)
    self.add_write_func(write_name, s)

  def make_funcs(self, fmt):
    fmt_size = struct.calcsize(fmt)

    def g(addr):
      y = self.reader(addr, fmt_size)
      return struct.unpack(self.le_sgn + '%c' % fmt, y)[0]

    def ng(addr, n):
      y = self.reader(addr, fmt_size*n)
      return struct.unpack(self.le_sgn + '%d%c' % (n,fmt), y)

    def s(addr, val):
      packed_val = struct.pack(self.le_sgn + '%c' % fmt, val)
      self.writer(addr, packed_val)

    return g, s, ng

  # read by size, automatically cast to number
  def read_u(self, addr, size):
    assert size in self.s2c
    res = self.reader(addr, size)
    return struct.unpack(self.s2c[size], res)[0]

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

class AtomMem(Memory):
  def __init__(self, atom_size=4, read_atom=None, write_atom=None, **kwargs):
    super().__init__(reader=self._read_impl, writer=self._write_impl, **kwargs)
    self.atom_size = atom_size
    self.read_atom = read_atom
    self.write_atom = write_atom

  def _read_impl(self, addr, sz):
    res = bytearray()
    prefix = addr % self.atom_size
    for i in range(0, prefix + sz, self.atom_size):
      res += self.read_atom(addr + i)
    return res[prefix:prefix+sz]

  def _write_impl(self, addr, content):
    assert len(content)% self.atom_size == 0
    assert addr % self.atom_size == 0
    for i in range(0, len(content), self.atom_size):
      self.write_atom(addr + i, content[i:i+self.atom_size])


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
        self.buf), 'Bad shit ptr=%x, base=%x, sz=%x, len=%x' % (
            ptr, self.base_addr, sz, len(self.buf))
    ptr -= self.base_addr
    return self.buf[ptr:ptr + sz]

  def _write(self, ptr, content):
    ptr -= self.base_addr
    self.buf[ptr:ptr+len(content)] = content

class NonRandomMemory(Memory):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    def tmp(old):
      return lambda *_args: old(0, *_args)

    for x in self.read_funcs + self.write_funcs:
      old = getattr(self, x)
      setattr(self, x, tmp(old))


class Stack:

  def __init__(self, regs, mem, ptr_size=8):
    self.regs = regs
    self.mem = mem
    self.ptr_size = ptr_size
    assert ptr_size == 8, 'Not supported'

  def write_one(self, addr, e):
    self.mem.write_u64(addr, e)

  def read_one(self, addr):
    return self.mem.read_u64(addr)

  def push(self, e):
    self.regs.rsp -= self.ptr_size
    self.write_one(self.regs.rsp, e)

  def pop(self):
    res = self.read_one(self.regs.rsp)
    self.regs.rsp += self.ptr_size
    return res

  def get(self, off):
    return self.read_one(self.regs.rsp + off * self.ptr_size)

  def set(self, off, v):
    return self.write_one(self.regs.rsp + off * self.ptr_size, v)


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
        print(arg, num, pos, arg[pos:pos + num])
        cur += re.escape(struct.pack('%c%d%c' % (order, num, i), *arg[pos:pos +
                                                                      num]))
        pos += num
      num = None
  return cur
