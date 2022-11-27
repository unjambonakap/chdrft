from chdrft.emu.base import Memory
from chdrft.gen.opa_clang import OpaIndex, InputOnlyFilter
from chdrft.gen.types import is_cursor_typ, OpaCoreType, OpaBaseType, OpaBaseField, types_helper_by_m32
from chdrft.gen.types import unsigned_primitives_types, set_cur_types_helper
from chdrft.struct.base import SparseList
from chdrft.utils.fmt import Format
from chdrft.utils.misc import Attributize, Arch, to_list, byte_norm, DictWithDefault, BitOps, NormHelper, struct_helper
from chdrft.utils.path import ProgDir
from clang.cindex import TypeKind
from clang.cindex import TypeKind, CursorKind
from collections import OrderedDict
from collections import defaultdict
from contextlib import ExitStack
import chdrft.gen.types as opa_types
import glog
import os
import struct
import yaml
import copy
from chdrft.algo.misc import LRUFifo

import chdrft.utils.misc as cmisc
from chdrft.emu.binary import arch_data


class Data:

  def __init__(self):
    self.types_helper = None
    self.is_m32 = False
    self.arch = None

  def set_m32(self, m32):
    glog.debug('SETTING m32 %s' % m32)
    self.is_m32 = m32
    set_cur_types_helper(m32)
    self.types_helper = None
    self.arch = None

    if m32 is not None:
      self.types_helper = types_helper_by_m32[m32]
      if m32:
        self.arch = arch_data[Arch.x86]
      else:
        self.arch = arch_data[Arch.x86_64]

  def set_arch(self, arch):
    self.arch = arch


g_data = Data()


class AccessorCtx:

  def __init__(self, st, pos, accessor):
    self.accessor = accessor
    self.pos = pos
    self.st = st

  def init(self, st, pos):
    # called by builders
    self.st = st
    self.pos = pos

  def get(self):
    # binary to view
    return self.accessor.get(self.st, self.get_next)

  def set(self, v):
    # view to binary
    return self.accessor.set(self.st, self.set_next, v)

  def get_next(self):
    return self.st.get(accessor_pos=self.pos + 1)

  def set_next(self, v):
    return self.st.set(v, accessor_pos=self.pos + 1)


class ScaleAccessor:

  def __init__(self, scale):
    self.scale = scale

  def get(self, st, get_next):
    return get_next() * self.scale

  def set(self, st, set_next, v):
    set_next(v / self.scale)


class InverterAccessor:

  def get(self, st, get_next):
    return 1. / get_next()

  def set(self, st, set_next, v):
    return set_next(1. / v)


class RemapAccessor:

  def __init__(self, remap_vec, field):
    self.remap_vec = remap_vec
    self.field = field
    self.iremap = {v: i for i, v in enumerate(remap_vec)}

  def get(self, st, get_next):
    return self.remap_vec[st.parent[self.field].get()]

  def set(self, st, get_next, v):
    st.parent[self.field].set(self.iremap[v])


class GetterFunc:

  def __init__(self, field, func):
    self.field = field
    self.func = func

  def get(self, st, get_next):
    v = st.parent[self.field].get()
    return self.func(v)

  def set(self, st, get_next, v):
    pass


accessors_ctx = dict(
    ScaleAccessor=ScaleAccessor,
    inverter=InverterAccessor(),
    RemapAccessor=RemapAccessor,
    GetterFunc=GetterFunc,
)


class StructExtractor:

  def keep_macro(self, location, name):
    assert 0

  def keep_typ(self, location, name):
    assert 0

  def get_includes(self):
    assert 0

  def add(self, e):
    pass

  def finish(self, builder):
    pass


class SimpleStructExtractor(StructExtractor):

  def __init__(self, filenames, prefixes, additional_includes=[]):
    super().__init__()
    self.filenames = to_list(filenames)
    self.prefixes = to_list(prefixes)
    self.additional_includes = additional_includes

  def get_includes(self):
    return ['#include<%s>' % filename for filename in self.filenames]

  def keep_macro(self, location, name):
    for prefix in self.prefixes:
      if name.startswith(prefix):
        return True
    return False

  def keep_typ(self, location, name, kind):
    for prefix in self.prefixes:
      if name.startswith(prefix):
        return True
    return False

  #not tested
  @staticmethod
  def CommonExtractor():
    return SimpleStructExtractor(os.path.join(ProgDir(), 'opa/common/inc/opa_common.h'))


class CategoryStructExtractor(SimpleStructExtractor):

  def __init__(self, name, filename, prefixes):
    super().__init__(filename, prefixes)
    self.added = []
    self.name = name

  def add(self, e):
    self.added.append(e)

  def finish(self, builder):
    cur = Attributize(key_norm=lambda x: byte_norm(x))
    builder.cats[self.name] = cur
    for prefix in self.prefixes:
      cur[prefix] = Attributize(key_norm=byte_norm)

    for e in self.added:
      for prefix in self.prefixes:
        if e.name.startswith(prefix):
          if isinstance(e, opa_types.OpaMacro):

            if isinstance(e.val, list):
              # expressino macro, skipping
              break
            cur[prefix][e.val] = e.name
          break
      else:
        assert 0, 'Should not have been added here'


class CodeStructExtractor(SimpleStructExtractor):

  def __init__(self, code, prefixes):
    super().__init__('', prefixes)
    self.code = code

  def get_includes(self):
    #bypass parent function
    return [self.code]


class StructBuilder:

  def __init__(self):
    self.extractors = []
    self.consts = Attributize(key_norm=byte_norm)
    self.typs = Attributize(key_norm=byte_norm)
    self.functions = Attributize(key_norm=byte_norm)
    self.vars = Attributize(key_norm=byte_norm)
    self.content = ['#include <stddef.h>']

    self.has_setup_includes = False
    self.cats = Attributize()

  def filter(self, cursor, loc):
    if cursor.kind == CursorKind.MACRO_DEFINITION:
      for extractor in self.extractors:
        if extractor.keep_macro(loc.location, cmisc.to_str(cursor.displayname)):
          return True, extractor

    elif is_cursor_typ(cursor.kind):
      for extractor in self.extractors:
        if extractor.keep_typ(loc.location, cmisc.to_str(cursor.displayname), cursor.kind):
          #print('keep ', cursor.displayname)
          return True, extractor

    return loc.is_input, None

  def add_extractor(self, extractor):
    self.extractors.append(extractor)

  def setup_includes(self):
    if self.has_setup_includes:
      return
    self.has_setup_includes = True
    for extractor in self.extractors:
      self.content += extractor.get_includes()

  def build(self, extra_args=[], want_sysincludes=True, cpp_mode=True):
    self.setup_includes()
    args = []
    if want_sysincludes:
      args.append('-isystem/usr/include')
    if g_data.is_m32:
      args.append('-m32')

    args.extend(extra_args)
    code_content = '\n'.join(self.content)
    assert len(self.content) > 0
    glog.debug('Extracting >> %s', code_content)
    self.res = OpaIndex.create_index(file_content=code_content,
                                     cpp_mode=cpp_mode,
                                     args=args,
                                     filter=self.filter)
    for macro in self.res.macros:
      self.consts[macro.name] = macro.val

    for typedef in self.res.typedefs:
      glog.debug('GOT TYP %s', typedef.name)
      self.typs[typedef.name] = typedef.typedef_typ

    for struct in self.res.structs:
      glog.debug('GOT struct %s', struct.name)
      self.typs[struct.name] = struct

    for fname, function in self.res.functions.items():
      glog.debug('GOT function %s', function.name)
      self.functions[fname] = function

    for var in self.res.vars:
      glog.debug('GOT var %s', var.name)
      self.vars[var.name] = var

    for extractor in self.extractors:
      extractor.finish(self)

  @staticmethod
  def opa_common_args():
    return ['-I%s' % os.path.join(ProgDir(), 'opa/common/common_base_hdr')]


class Structure(Attributize):

  def __init__(self,
               typ,
               off=0,
               backend=None,
               parent=None,
               fieldname='',
               default_backend=True,
               force_array_size=None,
               child_backend=None,
               byteoff=None):
    super().__init__(handler=self.item_handler, repr_blacklist_keys=['parent'])
    self._typ = typ
    self._base_typ = typ.get_base_typ()
    self._children = OrderedDict()
    if byteoff is not None:
      off = byteoff * 8
    self._off = off  # in bits
    self.parent = parent
    self._fieldname = fieldname
    self._bytesize = None
    self._fields = OrderedDict()
    self._force_array_size = force_array_size
    if backend is None and default_backend:
      backend = StructBackend(BufAccessor(self.byte_size))
    backend = self.norm_backend(backend)
    self._backend = backend

    if child_backend is None:
      child_backend = backend
    self._child_backend = child_backend

    if self._base_typ != self._typ:
      self._base_size = self._base_typ.size
      self._base_align = self._base_typ.align

      if self.is_array:
        pos = 0
        self.stride = BitOps.align(self._base_size, self._base_align * 8)
        assert self.stride != 0

        #for i in range(typ.array_size):
        #  self.add_child(self._base_typ, pos)
        #  pos += self._base_size
        #  pos = BitOps.align(pos, self._base_align)
      else:
        self._pointee = None
    else:
      if typ.is_primitive():
        pass
      elif typ.is_union:
        self._selected = 0
        for field in typ.ordered_fields:
          self.add_field(field)
      else:
        pos = 0
        for field in typ.ordered_fields:
          self.add_field(field)

    #self._mem = Memory(reader=self.get_buf, writer=self.set_buf, minv=0, maxv=self.bytesize)
    self._init_accessors()

  def clone(self):
    res = Structure(self._typ)
    res.set_raw(self.raw)
    return res

  def set_alloc_backend(self, allocator):
    self.set_backend(StructBackend(buf=BufAccessor(self.bytesize, allocator=allocator)))

  def _init_accessors(self):
    self._accessors = []
    for i, x in enumerate(self._typ.accessors):
      self._accessors.append(AccessorCtx(self, i, x))

  def get_child_backend(self):
    return self._child_backend

  def set_child_backend(self, v):
    self._child_backend = v

  def add_field(self, field):
    off = field.off
    self.add_field_internal(field.name, self.add_child(field.typ, off, field.name))

  def create_child(self, typ, off, fieldname):
    res = Structure(
        typ,
        self._off + off,
        self.get_child_backend(),
        parent=self,
        fieldname=fieldname,
        default_backend=0,
    )
    return res

  def add_child(self, typ, off, fieldname):
    res = self.create_child(typ, off, fieldname)
    self._children[len(self._children)] = res
    return res

  def add_field_internal(self, name, val):
    self._fields[name] = val
    self[name] = val

  def has_field(self, name):
    return name in self._fields

  def item_handler(self, name):
    if name.startswith('set_'):
      res = self._typ.choices.find_choice(name[4:])
      if res is not None:
        self.set(res)
        return (True, 0)
      glog.debug('cannot find here name=%s, res=%s, choices=%s', name, res, self._typ.choices)
    if name.startswith('is_'):
      res = self._typ.choices.find_choice(name[3:])
      if res is not None:
        return (self.get(choice=False) == res, 0)
      glog.debug('cannot find here name=%s, res=%s', name, res)

  def __getitem__(self, name):
    if self.is_array:
      return self.child(name)
    else:
      return super().__getitem__(name)

  @property
  def byte_size(self):
    return (7 + self._typ.size) // 8

  @property
  def bytesize(self):
    return self.byte_size

  @property
  def bitsize(self):
    return self._typ.size

  @property
  def offset(self):
    return self._off

  @property
  def mask(self):
    return (1 << 8 * self.byte_size) - 1

  @property
  def byte_offset(self):
    return self._off // 8

  @property
  def array_size(self):
    if self._force_array_size is not None:
      return self._force_array_size
    return self._typ.array_size

  @property
  def is_array(self):
    return self.array_size != -1

  @property
  def rel_offset(self):
    par_off = 0 if self.parent is None else self.parent.offset
    return self.offset - par_off

  @property
  def rel_byteoffset(self):
    return self.rel_offset // 8

  @property
  def is_primitive(self):
    return not self.is_array and self._typ.is_primitive()

  @property
  def is_pointer(self):
    return self._base_typ != self._typ and self._typ.ptr_count > 0

  @property
  def backend(self):
    return self._backend

  @property
  def raw(self):
    return self._get_raw()
  def norm_backend(self, backend):
    if backend is None: return None
    if backend.atom_size < self._typ.atom_size:
      backend = TransacBackend(backend.buf, self._typ.atom_size)
    return backend

  def set_backend(self, backend):
    backend  = self.norm_backend(backend)
    self._backend = backend
    self._child_backend = backend
    for child in self._children.values():
      child.set_backend(backend)
    return self

  #def set_buf(self, buf, offset=None):
  #  if offset is None:
  #    offset = self.offset
  #  self._backend.set_buf(buf, offset)
  #  for child in self_.children:
  #    child.set_buf(buf, offset)
  def reset(self):
    assert self.bitsize % 8 == 0
    self._backend.set(b'\x00' * self.byte_size, self.offset, self.bitsize)

  def get_ptr_as_str(self):
    res = []
    pos = self.get()
    cb = self.get_child_backend()
    read_batch = 128
    while True:
      c = cb.get((pos + len(res)) * 8, 8 * read_batch)
      assert len(c) == read_batch
      p = c.find(0)
      if p != -1:
        c = c[:p]
      res.append(c.decode())
      if p != -1:
        break

    return ''.join(res)

  def _set(self, typ_data, value):
    if typ_data.typ != 'float':
      value = round(value)
    value = self.val_to_buf(value, typ_data)
    self._set_raw(value)

  def get_for_call(self):
    content = self._get(self._typ.typ_data)
    if self.is_array:
      addr, buf = self.backend.buf.allocate(self.bytesize)
      buf.write(0, content)
      content = addr
    return content


  def val_to_buf(self, v, typ_data=None):
    if typ_data is None:
      typ_data = self._typ.typ_data
    return g_data.types_helper.pack(typ_data, v, le=self._typ.le)

  def _get(self, typ_data):
    res = self._get_raw()
    return g_data.types_helper.unpack(typ_data, res, le=self._typ.le)

  def _get_raw(self):
    return self._backend.get(self.offset, self.bitsize, le=self._typ.le)

  def _set_raw(self, val):
    self._backend.set(val, self.offset, self.bitsize, le=self._typ.le)

  def set_buf(self, byteoff, data):
    self._backend.set(data, self.offset + byteoff * 8, len(data) * 8, le=self._typ.le)

  def set_zero(self):
    self.set_buf(0, b'\x00' * self.bytesize)

  def get_buf(self, byteoff, n):
    return self._backend.get(self.offset + byteoff * 8, n * 8)

  def get(self, choice=True, accessor_pos=0):
    if accessor_pos is not None and accessor_pos < len(self._accessors) and choice:
      return self._accessors[accessor_pos].get()

    if self.is_primitive:
      val = self._get(self._typ.typ_data)
      #glog.debug('Req get %s: %s %s', val, self.offset, self.bitsize)
      if not choice:
        return val
      return self._typ.choices.choice_str(val)
    elif self.is_pointer:
      return self._get(self._typ.typ_data)
    else:
      return self._get_raw()

  def set(self, value, choice=True, accessor_pos=0):
    #glog.debug('Req set %s: %s %s', value, self.offset, self.bitsize)
    if accessor_pos is not None and accessor_pos < len(self._accessors) and choice:
      return self._accessors[accessor_pos].set(value)

    if not isinstance(value, bytes) and (self.is_primitive or
                                         (self.is_pointer and isinstance(value, int))):
      if isinstance(value, str):
        value = self._typ.choices.find_choice(value)
      return self._set(self._typ.typ_data, value)
    else:
      return self._set_raw(value)
      #    compose.append([0, struct.pack(self.get_prim_struct(), self._val)])

  def child(self, i):
    assert self.is_array
    assert self.stride != 0
    if not i in self._children:
      self._children[i] = self.create_child(self._base_typ, i * self.stride, 'elem_%d' % i)
    return self._children[i]

  def pretty_str(self, indent=0):
    prefix = '  ' * indent
    s = []
    if self.is_primitive or self.is_array or self.is_pointer:
      v = self.get()
      if isinstance(v, int):
        v = hex(v)
      s.append('%s - %s: %s' % (prefix, self._fieldname, v))
    else:
      s.append('%s %s' % (prefix, self._fieldname))
      for child in self._children.values():
        s.append(child.pretty_str(indent + 1))
    return '\n'.join(s)

  def pretty(self, indent=0):
    print(self.pretty_str(indent=indent))

  def __str__(self):
    return self.pretty_str()

  def to_attr(self):
    if self.is_primitive:
      return self.get()
    elif self.is_array:
      return [self.child(i).to_attr() for i in range(self.array_size)]
    else:
      res = Attributize()
      for x in self._children.values():
        res[x._fieldname] = x.to_attr()
    return res

  def cast_array(self, ntyp, n):
    ntyp = g_data.arch.typs.find_if_str(ntyp)
    pstruct = Structure(
        ntyp.make_array(n),
        off=self.get() * 8,
        backend=self.get_child_backend(),
    )
    return pstruct

  def deref(self, backend=None):
    if backend is None:
      backend = self.get_child_backend()
    pstruct = Structure(self._base_typ, off=self.get() * 8, backend=backend)
    return pstruct

  def get_pointee(self, array_len=1):
    pstruct = Structure(self._base_typ.make_array(array_len), default_backend=None)
    addr, nbackend = self.backend.allocate(pstruct.byte_size)
    pstruct.set_backend(nbackend)
    self.set(addr)
    return pstruct

  def smart_set(self, cur):
    if cur is None: return

    if self.is_pointer:
      if cur is None:
        self.set(0)
      elif isinstance(cur, int):
        self.set(cur)
      else:
        if not isinstance(cur, (list, bytes, bytearray, str)):
          cur = list([cur])
        array_len = len(cur)
        pointee = self.get_pointee(array_len=array_len)
        pointee.smart_set(cur)
    elif isinstance(cur, (bytes, bytearray)):
      self.set(Format.ToBytes(cur))
    elif isinstance(cur, str):
      res = self._typ.choices.find_choice(cur)
      if res is None:
        print(self._typ.choices)
        assert 0
        res = Format.ToBytes(cur)
      self.set(res)

    elif self.is_primitive:
      return self.set(cur)
    elif self.is_array:
      for i, v in enumerate(cur):
        self.child(i).smart_set(v)
    elif isinstance(cur, dict):
      for k, v in cur.items():
        self[k].smart_set(v)
    else:
      assert 0, type(cur)

  def from_args(self, **kwargs):
    return self.from_attr(kwargs)

  def from_attr(self, attr):
    if self.is_primitive:
      return self.smart_set(attr)
    elif self.is_array:
      for i, v in enumerate(attr):
        self.child(i).from_attr(v)
    else:
      for k, v in attr.items():
        self[k].from_attr(v)

  def set_scalar_backend(self, val):
    self.set_backend(StructBackend(BufAccessor(buf=self.val_to_buf(val))))

  #def get_prim_struct(self):
  #  return '<' + self._typ.typ_data.pack

  #def get_prim_val(self, data):
  #  assert self._typ.get_base_kind() in length_map, 'typ={}, kind={}'.format(
  #      self._typ, self._typ.get_base_kind())
  #  return struct.unpack(self.get_prim_struct(), data)[0]

  #def get_default_primitive(self):
  #  return self.get_prim_val(bytes(self._typ.size))

  #res = bytearray(self._typ.size)
  #compose = []

  #if self._base_typ != self._typ:

  #  if self._typ.array_size != -1:
  #    for i, x in enumerate(self._children):
  #      st = self._array_pos[i]
  #      compose.append([st, x.to_buf()])

  #  else:
  #    compose.append([0, struct.pack('<Q', self._val)])
  #else:
  #  if self._typ.is_primitive():
  #    compose.append([0, struct.pack(self.get_prim_struct(), self._val)])
  #  elif self._typ.is_union:
  #    compose.append([0, self[self._typ.ordered_fields[self._selected].name].to_buf()])
  #  else:
  #    for field in self._typ.ordered_fields:
  #      compose.append([field.off // 8, self[field.name].to_buf()])

  #for pos, buf in compose:
  #  res[pos:pos + len(buf)] = buf
  #return bytes(res)

  def from_buf(self, buf, reader):
    if isinstance(buf, int):
      buf = struct.pack('<Q', buf)
    assert self._typ.size <= len(buf)
    self._raw = buf

    base_typ = self._typ.get_base_typ()
    assert base_typ == self._typ, 'not handling pointers'

    if self._typ.is_primitive():
      self.set_raw(self.parse_primitive(buf))
    elif self._typ.is_union:
      for field in self._typ.ordered_fields:
        self[field.name].from_buf(buf, reader)
    else:
      #print('for structure ', typ, typ.get_loc())
      for field in self._typ.ordered_fields:
        off = field.off
        assert off % 8 == 0, 'do not handle bit structures'
        off = off // 8
        nv = buf[off:off + field.typ.bitsize]
        #print('on field ', field.name)
        self[field.name].from_buf(nv, reader)
      #print('ret')
    return self

  def dict(self):
    if self._typ.is_primitive():
      return self.get()
    else:
      res = {}
      for field in self._typ.ordered_fields:
        res[field.name] = self[field.name].dict()
      return res

  def parse_primitive(self):
    kind = typ.get_base_kind()
    if kind not in length_map:
      return None
    sz = typ.base_size
    assert len(val) >= sz
    return struct.unpack('<' + length_map[kind], val[:sz])[0]

  def parse_primitive(self, typ, val):
    kind = typ.get_base_kind()
    if kind not in length_map:
      return None
    sz = typ.base_size
    assert len(val) >= sz
    return struct.unpack('<' + length_map[kind], val[:sz])[0]


class BufAccessorBase:

  def __init__(self, allocator=None):
    self._allocator = allocator

  def read(self, pos, n=None):
    curn = n
    if n is None:
      curn = 1

    res = self._read(pos, curn)
    if n is None:
      return res[0]
    return res

  def write(self, pos, content):
    return self._write(pos, content)

  def _read(self, pos, n):
    assert 0

  def _write(self, pos, content):
    assert 0

  def read_atom(self, pos, n):
    return self.read(pos, n)

  def write_atom(self, pos, content):
    return self.write(pos, content)

  def allocate(self, sz):
    return self._allocator(sz)


class BufAccessor(BufAccessorBase):

  def __init__(self, size=None, buf=None, **kwargs):
    super().__init__(**kwargs)
    if buf is None:
      buf = bytearray([0] * size)
    if size is None:
      size = len(buf)
    self.buf = buf

  def _read(self, pos, n):
    assert pos + n <= len(self.buf), f'{pos} {n} {len(self.buf)} {type(self.buf)}'
    return self.buf[pos:pos + n]

  def _write(self, pos, content):
    self.buf[pos:pos + len(content)] = content


class MemBufAccessor(BufAccessorBase):

  def __init__(self, mem, **kwargs):
    super().__init__(**kwargs)
    self.mem = mem

  def _read(self, pos, n):
    return self.mem.read(pos, n)

  def _write(self, pos, content):
    return self.mem.write(pos, content)


class BaseBackend:

  def __init__(self, atom_size=1, **kwargs):
    self.buf = kwargs['buf']
    self.atom_size = atom_size
    # buffer start position is at @self.offset
    self.off = 0

  def set_buf(self, buf, off):
    self.buf = buf
    self.off = off

  def flush(self):
    assert 0

  def allocate(self, sz):
    nobj = copy.copy(self)
    addr, nobj.buf = self.buf.allocate(sz)
    return addr, nobj

  def get(self, off, size, le=True):
    # TODO: le ignored
    return self._get(off, size)

  def set(self, val, off, size, lazy=None, le=True):
    return self._set(val, off - self.off, size, lazy, le)

  def _get(self, off, size):
    assert 0

  def _set(self, val, off, size, lazy, le):
    assert 0


class StructBackend(BaseBackend):

  def __init__(self, buf=None, lazy=False, allocator=None, off_rebase=0, atom_size=1):
    super().__init__(buf=buf, atom_size=atom_size)
    self.allocator = allocator
    self.lazy = lazy
    self.ops = DictWithDefault(default=lambda: [])
    self.off_rebase = off_rebase

  def flush(self):
    for k, v in self.ops:
      pass

  def _get(self, off, size):
    off -= self.off_rebase * 8
    base = off // 8
    off %= 8
    nx = (off + size + 7) // 8
    buf = bytearray(self.buf.read(base, nx))
    if off != 0:
      buf = Format(buf).shiftr(off).v
    buf = Format(buf).resize_bits(size).v
    return buf

  def _set(self, val, off, size, lazy, le):
    off -= self.off_rebase * 8
    # TODO: le ignored here
    base = off // 8
    off %= 8
    val = bytearray(val)
    if len(val) == 0:
      return
    if lazy == False or not self.lazy:
      nx = (off + size + 7) // 8
      if le:
        val = val[:nx]
      else:
        val = val[len(val) - nx:]
      if len(val) != nx:
        val.append(0)
      val = Format(val).shiftl(off).v
      assert len(val) == nx, (val, nx)
      val = val[:nx]

      mask0 = BitOps.mask(off)
      maskn1 = BitOps.imask(BitOps.mod1(off + size, 8),
                            modpw=8)  # we dont want imask(0) (imask(8) needed)

      b0 = self.buf.read(base)
      mask_b0 = mask0
      if nx == 1:
        mask_b0 |= maskn1
      else:
        if maskn1 != 0:
          val[-1] |= self.buf.read(base + nx - 1) & maskn1

      if mask_b0 != 0:
        val[0] |= b0 & mask_b0

      self.buf.write(base, val)
    else:
      assert False


class AtomizedList:

  def __init__(self, atom_bitsize, reader=None, writer=None):
    self.atom_bitsize = atom_bitsize
    self.reader = reader
    self.writer = writer
    self.atom_data = defaultdict(lambda: [BitOps.mask(atom_bitsize), 0])
    self.lrufifo = LRUFifo()

  def add(self, pos, data):
    for i, v in enumerate(data):
      atom = (pos + i) // self.atom_bitsize
      in_atom = (pos + i) % self.atom_bitsize
      self.lrufifo.touch(atom)

      self.atom_data[atom][0] &= BitOps.ibit(in_atom)
      self.atom_data[atom][1] = (self.atom_data[atom][1] & BitOps.ibit(in_atom)) | v << in_atom

  def clear(self):
    self.lrufifo.clear()
    self.atom_data.clear()

  def proc(self):
    for k in self.lrufifo.get_all():
      retrieve_mask, val = self.atom_data[k]
      atom_addr = k * self.atom_bitsize // 8
      if retrieve_mask != 0:
        val |= self.reader(atom_addr) & retrieve_mask
      self.writer(atom_addr, val)
    self.clear()


class TransacBackend(BaseBackend):

  # Flushing in order of writes
  def __init__(self, buf, atom_size, deferred_write=True):
    super().__init__(buf=buf)
    self.atom_size = atom_size
    self.mask = ~(atom_size - 1)
    self.ops = AtomizedList(atom_size * 8,
                            reader=self.atom_list_reader,
                            writer=self.atom_list_writer)
    self.deferred_write = deferred_write
    self._enable_cache = None
    self.read_cache = None
    self.configure_flush_read_cache(enable=True)

  def read_cached(self, atom):
    val = None
    if atom in self.read_cache:
      val = self.read_cache[atom]
    else:
      val = self.buf.read_atom(atom, self.atom_size)

    if self._enable_cache:
      self.read_cache[atom] = val

    return val

  def atom_list_reader(self, atom):
    return struct_helper.get(self.read_cached(atom), size=self.atom_size)

  def atom_list_writer(self, atom, data):
    glog.debug('WRITE atom=%x, data=%s', atom, data)
    data = struct_helper.set(data, size=self.atom_size)
    self.buf.write_atom(atom, data)

  def enable_cache(self):
    self.configure_flush_read_cache(enable=True)

  def disable_cache(self):
    self.configure_flush_read_cache(enable=False)

  def configure_flush_read_cache(self, enable=False):
    self._enable_cache = enable
    self.read_cache = {}

  def flush(self):
    res = self.ops.proc()
    self.read_cache = {}

  def _get(self, off, size):
    base = off // 8
    off %= 8
    nx = (off + size + 7) // 8
    buf = bytearray(self.do_read(base, nx))
    if off != 0:
      buf = Format(buf).shiftr(off).v
    buf = Format(buf).resize_bits(size).v
    return buf

  def _set(self, val, off, size, lazy, le):
    self.ops.add(off, Format(val).bitlist(size, le).v)
    if not self.deferred_write:
      self.flush()

  def do_read(self, addr, n):

    base_addr = addr & self.mask
    nn = n + addr - base_addr
    nn += (-nn) & (self.atom_size - 1)
    assert n > 0
    assert nn > 0

    res = bytearray()
    for i in range(0, nn, self.atom_size):
      res += self.read_cached(base_addr + i)
    res = res[addr - base_addr:]
    return res[:n]


class TransacContext(ExitStack):

  def __init__(self, transa_backend):
    super().__init__()
    self.transa_backend = transa_backend

  def flush(self):
    self.transa_backend.flush()

  def __enter__(self):
    self.transa_backend.enable_cache()
    self.callback(self.transa_backend.disable_cache)
    self.callback(self.transa_backend.flush)
    return self


def yaml_get_size(desc, default=None):
  if 'size' in desc:
    return desc.size * 8
  elif 'bitsize' in desc:
    return desc.bitsize
  return default


class YamlType(OpaBaseType):

  def __init__(self, name, data, db):
    super().__init__()
    self.name = name
    self.atom_size = data.get('atom_size', 0)
    self.le = db.gbl.get('le', True)  # might be ignored in some cases. To fix

    last_field = None
    self.is_union = data.get('union', False)
    bitsize = 0
    if self.is_union:
      bitsize = data.get('size', 0) * 8
      if not bitsize:
        bitsize = data.get('bitsize', 0)

    size = 0
    for field_desc in data.fields:
      new_field = YamlField(field_desc, last_field, db, parent=self, bitsize=bitsize)
      self.add_field(new_field)
      if self.is_union:
        size = max(size, new_field.size)
      else:
        last_field = new_field

    if last_field is not None:
      size = last_field.off + last_field.size

    size = yaml_get_size(data, size)
    size=  cmisc.align(size, 8 * self.atom_size)
    self.size = size
    self._is_primitive = False
    self.base_size = self.size
    self.base_typ = None

  def as_struct(self, **kwargs):
    return Structure(self, **kwargs)


class YamlField(OpaBaseField):

  def __init__(self, data, last_field, db, parent, ctx=None, bitsize=0):
    super().__init__(parent)
    off = 0
    self.name = data.name

    if 'offset' in data:
      off = data.offset * 8
    elif 'bitoffset' in data:
      off = data.bitoffset
    elif last_field is not None:
      off = last_field.off + last_field.typ.size

    if 'align' in data:
      off = cmisc.align(off, 2**(data.align - 1 + 3))

    if 'fieldoff' in data:
      # used for views
      # struct {int a, b;}; a_l = {fieldoff:a, off:0, type:u16}, a_h = {fieldoff:a, bitoff:16...}

      assert parent is not None
      res = parent.get_field(data.fieldoff)
      off += res.off

    self.off = off

    if 'type' in data:
      if 'array' in data and data.array:
        typ = OpaBaseType()
        typ.base_typ = db.get_typ(data.type)
        if 'nelem' in data:
          typ.array_size = data.nelem
        else:
          print(data.size, typ.base_typ.size)
          assert data.size * 8 % typ.base_typ.size == 0
          typ.array_size = data.size * 8 // (typ.base_typ.size)
        typ.name = '%s[%d]' % (typ.base_typ.name, typ.array_size)
        typ.finalize()
      else:
        typ = db.get_typ(data.type)

      self.typ = typ
      self.size = self.typ.size

    elif 'fields' in data:
      struct_name = f'{parent.name}.{data.name}_t'
      #print('BUILDING STRUCT ', struct_name, data)
      self.typ = db.build_typ(struct_name, data)
      #print("RESULT >> >", struct_name, self.typ.gets())
      self.size = self.typ.size
    else:
      typ = OpaBaseType()
      typ._is_primitive = True

      if bitsize:
        typ.size = bitsize
      elif 'size' in data:
        typ.size = data.size * 8
      elif 'padalign' in data:
        next_off = cmisc.align(off, data.padalign * 8)
        typ.size = next_off - off
      else:
        typ.size = data.get('bitsize', 1)

      signed = data.get('signed', False)

      if 'scale' in data:
        scale = data.scale
        typ.add_accessor(ScaleAccessor(scale))

      for accessor in data.get('accessors', []):
        typ.add_accessor(eval(accessor, ctx))

      typ.base_typ = db.get_typ(g_data.types_helper.get_next_type(typ.size, signed=signed).name)
      typ.typ_data = typ.base_typ.typ_data
      typ.name = typ.base_typ.name
      typ.le = data.get('le', True)
      typ.choices.add_choices(getattr(data, 'vals', []))
      typ.finalize()

      self.typ = typ
      self.size = self.typ.size


def build_core_types():
  typs = Attributize(key_norm=byte_norm)
  for name, typ in g_data.types_helper.by_name.items():
    typs[name] = OpaCoreType(typ)
  return typs


class YamlStructBuilder:

  def __init__(self):
    self.typs = build_core_types()
    self.typs_to_build = Attributize(key_norm=byte_norm)
    self.gbl = {}

  def add_typ(self, name, typ):
    self.typs[name] = typ

  def add_yaml(self, s):
    res = yaml.load(s, Loader=yaml.FullLoader)
    res = Attributize.RecursiveImport(res)
    self.typs_to_build.update(res._elem)

  def build(self):
    for k, v in self.typs_to_build.items():
      if k == 'global':
        self.gbl = v
        continue
      self.build_typ(k, v)

  def get_typ(self, typ_name):
    if typ_name in self.typs:
      return self.typs[typ_name]
    if typ_name in self.typs_to_build:
      return self.build_typ(typ_name, self.typs_to_build[typ_name])

    assert typ_name.endswith('*') != -1, 'Bad typ %s' % typ_name
    print('qq', typ_name)
    base_type = self.get_typ(typ_name[:-1])
    ntype = base_type.make_ptr()
    self.add_typ(typ_name, ntype)
    return ntype

  def build_typ(self, name, desc):
    # prevent loops
    self.typs[name] = None
    res = YamlType(name, desc, self)
    self.add_typ(name, res)
    return res


if arch_data:
  g_data.set_m32(False)
  CORE_TYPES = YamlStructBuilder()
  arch_data[Arch.x86_64].typs = CORE_TYPES.typs
  g_data.set_m32(True)
  CORE_TYPES_U32 = YamlStructBuilder()
  arch_data[Arch.x86].typs = CORE_TYPES.typs
  g_data.set_m32(None)

##obsolete I believe
#class StructureReader:
#
#  def __init__(self, reader_func):
#    self.reader_func = reader_func
#
#  def parse(self, func, args):
#    lst = []
#    for i in range(len(args)):
#      s = self.from_ptr(args[i])
#      lst.append(Attributize(name=func.args[i].name, val=self.parse_one(func.args[i].typ, s)))
#
#    return lst
#
#  def get_ptr(self, v):
#    return struct.unpack('<Q', v)[0]
#
#  def from_ptr(self, v):
#    return struct.pack('<Q', v)
#
#  def ptr_size(self):
#    return 8
#
#  def has_reader(self):
#    return self.reader_func is not None
#
#  def parse_ptr(self, typ, ptr):
#    # while creating types on the fly is not available
#    val = self.reader_func(ptr, typ.size)
#    return self.parse_one(typ, val)
#
#  def parse_one(self, typ, val):
#    res = Attributize()
#    if isinstance(val, int):
#      val = struct.pack('<Q', val)
#    assert typ.size <= len(val)
#    res.raw = val
#
#    base_typ = typ.get_base_typ()
#    if base_typ != typ:
#      base_size = base_typ.size
#      base_align = base_typ.align
#
#      if typ.array_size != -1:
#        pos = 0
#        res.elems = []
#        for i in range(typ.array_size):
#          assert pos + base_size <= len(val)
#          res.elems.append(self.parse_one(base_typ, val[pos:pos + base_size]))
#          pos += base_size
#          pos = BitOps.align(pos, base_align)
#      else:
#        assert typ.ptr_count == 1
#        should_deref = True
#        ptr = self.get_ptr(val)
#        res._val = ptr
#        #print('for typ ', base_typ, ptr, typ.get_loc(), base_typ.kind, base_typ.size)
#
#        if ptr == 0 or base_typ.kind == TypeKind.FUNCTIONPROTO or not self.has_reader(
#        ) or base_typ.size < 0:
#          should_deref = False
#        if should_deref:
#          val = self.reader_func(ptr, base_size)
#          res.pointee = self.parse_one(base_typ, val)
#        else:
#          res.pointee = None
#
#    else:
#      res.typ = typ
#      if typ.is_primitive():
#        res._val = self.parse_primitive(typ, val)
#      elif typ.is_union:
#        for field in typ.ordered_fields:
#          res[field.name] = self.parse_one(field.typ, val)
#      else:
#        #print('for structure ', typ, typ.get_loc())
#        for field in typ.ordered_fields:
#          off = field.off
#          assert off % 8 == 0, 'do not handle bit structures'
#          off = off // 8
#          nv = val[off:off + field.typ.size]
#          #print('on field ', field.name)
#          res[field.name] = self.parse_one(field.typ, nv)
#        #print('ret')
#
#    return res
