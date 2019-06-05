#!/usr/bin/env python

from clang.cindex import TranslationUnit, Index, CursorKind, TypeKind, Cursor, Type, TokenKind
from clang.cindex import conf
from jsonpickle import handlers
import pprint as pp
from chdrft.utils.misc import Dict, Attributize, NormHelper, csv_list, BitOps
from chdrft.graph.base import OpaGraph
from collections import deque
import subprocess as sp
import re
import io
import sys
import hashlib
from chdrft.utils.fmt import Format
import struct
import glog
import collections
from asq.initiators import query as asq_query
def to_str(x):
  if isinstance(x, str): return x
  return x.decode()

types_list_base = [
    'bool:BOOL:1:?',
    'u8,unsigned char:UCHAR:1:B',
    'u16:USHORT:2:H',
    'u128:UINT128:16:',
    's8,signed char,char:SCHAR:1:b',
    's16:SHORT:2:H',
    's32,int:INT,LONG:4:I',
    's64:LONGLONG:8:Q',
    's128:INT128:16:',
    'float:FLOAT:4:f',
    'double:DOUBLE:8:d',
]
types_list_m32 = types_list_base + [
    'ptr,u32:UINT,ULONG,POINTER:4:I',
    'u64:ULONGLONG:8:Q',
    ]
types_list_m64 = types_list_base + [
    'u32:UINT,ULONG:4:I',
    'ptr,u64:ULONGLONG,POINTER:8:Q',
    ]

types_list= {False: types_list_m64, True: types_list_m32}


class TypesHelper:

  def __init__(self, m32=False):
    self.types = []
    self.by_clang_type = Attributize()
    self.by_name = Attributize()
    self.little_endian = '><'
    self.m32 = m32

    for typ in types_list[self.m32]:
      names, clang_types, size, pack = typ.split(':')
      size = int(size)
      names = names.split(',')
      name = names[0]
      typ = 'unsigned'
      if name in csv_list('float,double'):
        typ = 'float'
      elif name == 'bool':
        typ = 'bool'
      elif name[0] == 's':
        typ = 'signed'

      clang_types_list = [TypeKind.__dict__[e] for e in clang_types.split(',')]
      x = Attributize(name=name,
                      alias=names,
                      clang_types_list=clang_types_list,
                      size=size,
                      pack=pack,
                      typ=typ)
      self.types.append(x)
      for name in names:
        self.by_name[name] = x

      for v in clang_types_list:
        self.by_clang_type[v] = x


  def get_next_unsigned(self, bitsize):
    return self.get_next_type(bitsize, signed=False)

  def get_next_type(self, bitsize, signed=False):
    typ = 'signed' if signed else 'unsigned'
    return asq_query(self.types).where(
        lambda x: x.typ == typ and x.size * 8 >= bitsize).order_by(lambda x: x.size).first()

  def get_by_clang(self, kind):
    if not kind in self.by_clang_type:
      return None
    return self.by_clang_type[kind]

  def unpack_multiple(self, typ, buf, le=True):
    if typ is None: return buf

    if isinstance(typ, str): typ = self.by_name[typ]
    if le:
      buf = Format(buf).modpad(typ.size, 0).v
    else:
      buf = Format(buf).lmodpad(typ.size, 0).v

    nentries = len(buf) // typ.size
    pattern = f'{self.little_endian[le]}{nentries}{typ.pack}'
    return struct.unpack(pattern, buf)

  def unpack(self, typ, buf, le=True):
    if typ is None: return buf

    if isinstance(typ, str): typ = self.by_name[typ]
    if le:
      buf = Format(buf).pad(typ.size, 0).v
    else:
      buf = Format(buf).lpad(typ.size, 0).v
    pattern = self.little_endian[le] + typ.pack
    return struct.unpack(pattern, buf)[0]

  def pack(self, typ, v, le=True):
    if isinstance(typ, str): typ = self.by_name[typ]
    pattern = self.little_endian[le] + typ.pack
    return struct.pack(pattern, v)


types_helper = None
types_helper_by_m32 = {True: TypesHelper(m32=True), False: TypesHelper(m32=False), None: None}
g_types_helper = types_helper_by_m32[True]
g_types_helper32 = types_helper_by_m32[False]

def set_cur_types_helper(m32):
  global types_helper
  types_helper = types_helper_by_m32[m32]

primitive_types = [TypeKind.VOID,
                   TypeKind.BOOL,
                   TypeKind.CHAR_U,
                   TypeKind.UCHAR,
                   TypeKind.CHAR16,
                   TypeKind.CHAR32,
                   TypeKind.USHORT,
                   TypeKind.UINT,
                   TypeKind.ULONG,
                   TypeKind.ULONGLONG,
                   TypeKind.UINT128,
                   TypeKind.CHAR_S,
                   TypeKind.SCHAR,
                   TypeKind.WCHAR,
                   TypeKind.SHORT,
                   TypeKind.INT,
                   TypeKind.LONG,
                   TypeKind.LONGLONG,
                   TypeKind.INT128,
                   TypeKind.FLOAT,
                   TypeKind.DOUBLE,
                   TypeKind.LONGDOUBLE,
                   TypeKind.POINTER,
                   ]

unsigned_primitives_types = [
    TypeKind.UCHAR,
    TypeKind.USHORT,
    TypeKind.UINT,
    TypeKind.ULONG,
    TypeKind.ULONGLONG,
    TypeKind.UINT128,
]

types_db = {}
g_types_kinds = TypeKind


def is_cursor_typ(cursor_kind):
  return cursor_kind in (CursorKind.TYPEDEF_DECL, CursorKind.STRUCT_DECL, CursorKind.CLASS_DECL,
                         CursorKind.CLASS_TEMPLATE, CursorKind.FUNCTION_DECL, CursorKind.VAR_DECL)


def cursor_to_type(cursor):
  if cursor.kind == CursorKind.TRANSLATION_UNIT:
    return None
  return cursor.type.get_canonical()


def var_decl(name, typ):
  content_fmt = '{pre_typname} {fieldname}{post_typname}'
  return content_fmt.format(pre_typname=typ.field_typname_pre(),
                            fieldname=name,
                            post_typname=typ.field_typname_post(),)


class OpaBaseField:

  def __init__(self, parent_typ):
    self.name = None
    self.off = None
    self.typ = None
    self.size = None  # size in bits
    self.parent_typ = parent_typ

  def __str__(self):
    return 'Field: name={}, typ={}, off={}, size={}'.format(self.name, self.typ, self.off,
                                                            self.size)

  def __repr__(self):
    return str(self)


class OpaField(OpaBaseField):

  def __init__(self, index, cursor, parent_typ):
    super().__init__(parent_typ)
    self.index = index
    self.cursor = cursor
    self.name = to_str(cursor.displayname)
    self.off = parent_typ.internal_typ.get_offset(cursor.displayname)
    assert self.off != -1, f'Bad at name={self.name}, location={cursor.location}, displayname={cursor.displayname}'
    self.typ = index.get_typ(cursor)
    self.size = self.typ.size  # size in bits
    if cursor.is_bitfield():
      self.size = cursor.get_bitfield_width()

  @property
  def comment(self):
    return self.cursor.brief_comment

  def field_decl_str(self):
    return var_decl(self.name, self.typ)


class OpaTypedef:

  def __init__(self, index, cursor):
    self.index = index
    self.cursor = cursor
    self.name = to_str(cursor.displayname)
    assert isinstance(self.name, str)
    self.typedef_typ = self.index.get_typ(None, self.cursor.underlying_typedef_type)


class OpaFunction:

  def __init__(self, index, cursor):
    self.index = index
    self.cursor = cursor
    self.name = to_str(cursor.displayname)
    assert isinstance(self.name, str)
    self.shortname = to_str(cursor.spelling)

    self.typ = None
    self.args = []

  def build(self):
    self.typ = self.index.get_typ(self.cursor)
    for x in self.cursor.get_arguments():
      typ = self.index.get_typ(x)
      self.args.append(Attributize(name=x.spelling, typ=typ))
    return True


class OpaMacro:

  def conv_macro(self, val):
    val = to_str(val)
    assert len(val) > 0
    if val[0] == "'":
      #assert len(val)==3, val , might have comments after chars :( kappa
      return ord(val[1])
    elif val[0] == '"':
      return val[1:-1]
    else:
      try:
        p1 = val.find('/*')
        if p1 != -1:
          val = val[:p1]
        p1 = val.find('//')
        if p1 != -1:
          val = val[:p1]
        if len(val) > 0:
          if val[-1] == '#':  #hurray clang
            val = val[:-1]
        if val.startswith('0x'):
          return int(val, 16)
        else:
          return int(val, 10)
      except:
        # expression amcro, no special type here :(
        return [None, val]

  def __init__(self, index, cursor):
    self.index = index
    self.cursor = cursor
    self.tokens = list(self.cursor.get_tokens())
    self.mark = False
    self.name = None
    self.val = None

  def build(self):
    if len(self.tokens) < 2:
      return False
      #assert 0, f'location={self.cursor.location}, displayname={self.cursor.displayname}'

    self.name = to_str(self.tokens[0].spelling)
    self.val = ''.join([to_str(x.spelling) for x in self.tokens[1:]])
    self.val = self.conv_macro(self.val)

    if self.tokens[1].kind != TokenKind.LITERAL:
      #print(self.tokens[1].spelling)
      #print(self.tokens[0].spelling)
      #for j in self.tokens:
      #  print(j.kind)
      pass
    #print('got macro ', self.name, self.val)
    return True

  def macro_decl_str(self):
    macro_fmt = '#define {name} {val}'
    return macro_fmt.format(**self.__dict__)


class OpaVar:

  def __init__(self, index, cursor):
    self.index = index
    self.cursor = cursor

    if cursor.kind.is_unexposed():
      decl = cursor.type.get_declaration()
      #print('UNEXPOSED >> decl', decl.location)
      self.typ = self.index.get_typ(decl)
    else:
      self.typ = self.index.get_typ(cursor)

    self.name = to_str(self.cursor.spelling)
    self.val = None
    self.mark = False

  def build(self):
    children = list(self.cursor.get_children())
    if len(children) != 1:
      return True

    elem = children[0]
    tokens = list(elem.get_tokens())
    if len(tokens) == 0:
      return True
    token = tokens[0]
    val = token.spelling

    if elem.kind == CursorKind.STRING_LITERAL:
      self.val = val[1:-1]
    elif elem.kind == CursorKind.INTEGER_LITERAL:
      self.val = Format.ToInt(val)


    return True

  def var_decl_str(self):
    decl = var_decl(self.name, self.typ)
    return f'{decl}={self.val}'


class TypeBuilder:

  def __init__(self, cursor, typ):
    self.cursor = cursor
    self.typ = typ
    if not typ and not cursor:
      return

    if cursor and not typ:
      #print('got >> ', cursor)
      self.typ = cursor.type
    self.typ = self.typ.get_canonical()
    if not self.typ.kind == TypeKind.INVALID:
      self.cursor = self.typ.get_declaration()

  def get_key(self):
    if not self.cursor and not self.typ:
      return None
    return OpaType.get_key(self.cursor, self.typ)

  def get_type(self, index):
    return OpaType(index, self.cursor, self.typ)


class ChoiceHelper:

  def __init__(self, choices=[]):
    self.choice_count = 0
    self.mp = {}
    self.add_choices(choices)
    self.rmp = collections.defaultdict(lambda: [])

  def add_choice(self, choice, value=None, do_norm=True):
    if do_norm:
      choice = NormHelper.lowercase_norm(choice)
    if value is None:
      value = self.choice_count
    for v in choice.split('/'):
      self.mp[v] = value
      self.rmp[value].append(v)
    self.choice_count = value + 1

  def add_choices(self, choices):
    for choice in choices:
      self.add_choice(choice)

  def find_choice(self, name):
    return self.mp.get(name, None)

  def choice_str(self, val):
    if val not in self.rmp: return val
    return '/'.join(self.rmp[val])

  def __str__(self):
    return f'ChoiceHelper: {self.mp}'


class OpaBaseType(object):

  def __init__(self):
    self.typ = self
    self.base_typ = self
    self.name = None
    self.fields = Attributize(key_norm=NormHelper.byte_norm)
    self.ordered_fields = []
    self._is_primitive = False
    self.size = 0  #in bits
    self.align = -1  # in bytes
    self.base_size = 0
    self.array_size = -1
    self.is_union = False
    self.ptr_count = 0
    self.typ_data = None
    self.kind = ''
    self.atom_size = None
    self.choices = ChoiceHelper()
    self.is_enum = False
    self.enum_vals_long = Attributize(key_norm=NormHelper.byte_norm)
    self.enum_vals = Attributize(key_norm=NormHelper.byte_norm)
    self.ordered_enums = []
    self.le = True
    self.accessors = []

  def make_array(self, nelem):
    array_type = OpaBaseType()
    array_type.base_typ = self
    array_type.array_size = nelem
    array_type.name = f'{self.name}[{nelem}]'
    array_type.finalize()
    return array_type

  def make_ptr(self):
    ptr_type = OpaBaseType()
    ptr_type.typ_data = types_helper.by_name['ptr']
    ptr_type.base_typ = self.get_base_typ()
    ptr_type.ptr_count = self.ptr_count + 1
    ptr_type.name = f'*{self.name}'
    ptr_type.finalize()
    return ptr_type


  def add_accessor(self, accessor):
    self.accessors.append(accessor)

  def get_field(self, name):
    return self.fields[name]

  def finalize(self):
    self.base_size = self.get_base_typ().size
    if self.array_size >= 0:
      self.size = self.base_size * self.array_size
    elif self.ptr_count > 0:
      self.size = types_helper.by_name['ptr'].size * 8

  def add_field(self, field):
    self.fields[field.name] = field
    self.ordered_fields.append(field)
    return field

  def short_enum_name(self, name):
    prefix = self.name + '_'
    if name.startswith(prefix):
      name = name[len(prefix):]
    return name

  def add_enum_val(self, name, val):
    self.enum_vals_long[name] = val
    self.enum_vals[self.short_enum_name(name)] = val
    self.ordered_enums.append(name)

  def get_loc(self):
    return ''

  def get_base_typ(self):
    if self.base_typ:
      return self.base_typ
    return self

  def is_primitive(self):
    return self._is_primitive


  def check_range(self, v):
    v=int(v)
    return self.clamp(v) == v

  def clamp(self, v):
    assert self.is_primitive()
    assert self.typ_data.typ == 'unsigned'
    if v < 0: v = 0
    if v >= 2**self.size:
      v = 2**self.size - 1
    return int(v)

  def gets(self, indent=0, seen=set()):
    if len(seen) == 0:
      seen = set()
    prefix = '\t' * indent
    s = []
    s.append(
        'Struct: name={}, kind={}, size={}, align={}, ptr_count={} array_size={}, choices={}'.format(
            self.name, self.kind, self.size, self.align, self.ptr_count, self.array_size,
            self.choices))
    if self in seen:
      s.append('already seen')
    else:
      seen.add(self)
      s.append('        loc={}'.format(self.get_loc()))
      if self.get_base_typ() != self.typ:
        s.append('BaseType:\n{}'.format(self.get_base_typ().gets(indent + 1, seen)))
      if self.fields:
        s.append('Fields: (#{})'.format(len(self.ordered_fields)))
        for f in self.ordered_fields:
          glog.debug('got field %s', f)
          s.append('name={}, off={}, typ:\n{}'.format(f.name, f.off, f.typ.gets(indent + 1, seen)))

    s.append('=== DONE name={} ===='.format(self.name))

    return '\n'.join([prefix + x for x in s])

  def __str__(self):
    if isinstance(self.name, str):
      return 'Typ: ' + self.name
    return 'Typ: ' + self.name.decode()

  def struct(self, *args, **kwargs):
    from chdrft.emu.structures import Structure
    return Structure(self, *args, **kwargs)


class OpaCoreType(OpaBaseType):

  def __init__(self, typ, le=True):
    super().__init__()
    self._is_primitive = True
    self.name = typ.name
    self.size = typ.size * 8
    self.typ_data = typ
    self.le = le
    self.finalize()

  #incomplete
  #@staticmethod
  #def setup_core_types(db):
  #  core_types = Attributize()
  #  for e in types_helper.types:
  #    db[e.name] = e
  #    e.obj = OpaBaseType()


class OpaType(OpaBaseType):

  def __init__(self, index, cursor, typ):
    super().__init__()
    self.index = index
    self.kind = None
    self.cursor = cursor
    self.internal_typ = typ

    self.desc = OpaType.get_key(self.cursor, self.internal_typ)
    self.name = self.desc
    if self.name is not None:
      glog.debug(self.name)
      assert isinstance(self.name, str)

    if self.internal_typ:
      self.kind = self.internal_typ.kind

    self.deps = []
    self.children = []

    self.forward_decl = False
    self.decl = False
    self.node = None
    self.parent = None
    self.mark = 0
    self.templates = []
    self.template_base = None
    self.template_ref = None
    self.args = []  # for function proto
    self.hash = None

  def get_loc(self):
    return self.cursor.location

  def get_hash(self):
    if self.hash is None:
      s = '{name}:{location}:{size}:{kind}'.format(name=self.name,
                                                   location=repr(self.cursor.location),
                                                   size=self.size,
                                                   kind=self.kind.name)
      self.hash = hashlib.md5(s.encode()).hexdigest()

    return self.hash

  @staticmethod
  def get_key(cursor, typ=None):
    if not typ and (not cursor or cursor.kind == CursorKind.TRANSLATION_UNIT):
      return None

    cur = None
    if typ:
      return to_str(typ.spelling)
    else:
      if cursor.kind == CursorKind.FUNCTION_DECL:
        cur = to_str(cursor.spelling)
      else:
        return to_str(cursor.type.spelling)
    res = None

    if cursor:
      res = OpaType.get_key(cursor.semantic_parent)
    if not res:
      res = ''
    elif len(cur) > 0:
      res += '::'

    res += cur
    return res

  @property
  def comment(self):
    return self.cursor.brief_comment

  def build(self):
    if not self.cursor:
      return
    if self.cursor.kind is CursorKind.TRANSLATION_UNIT:
      # root node
      return
    if self.cursor.kind == CursorKind.UNION_DECL:
      self.is_union = True

    cur = self.internal_typ
    self.align = self.internal_typ.get_align()
    #print('got align >> ', self.align)
    base = conf.lib.clang_getSpecializedCursorTemplate(self.cursor)
    if base:
      tmp = base.get_definition()
      #print(tmp.location, tmp.kind, tmp.type.kind)

      self.template_base = self.index.get_typ(base)
      nt = self.internal_typ.get_num_template_arguments()
      for i in range(nt):
        tmp = self.internal_typ.get_template_argument_type(i)
        self.templates.append(self.index.get_typ(None, self.internal_typ))

    if cur.kind == TypeKind.POINTER:
      self.ptr_count += 1
      cur = cur.get_pointee()

    elif cur.kind == TypeKind.LVALUEREFERENCE:
      cur = cur.get_pointee()

    elif cur.kind == TypeKind.CONSTANTARRAY:
      assert self.array_size == -1
      self.array_size = cur.get_array_size()
      #print(cur.element_type.kind, cur.get_array_element_type().kind)
      cur = cur.element_type

    self.base_typ = None
    if cur != self.internal_typ:
      self.base_typ = self.index.get_typ(cur.get_declaration(), cur)
    else:
      if self.cursor.kind == CursorKind.CLASS_TEMPLATE or cur.kind == TypeKind.RECORD:
        self.init_record()
        self.setup_parent()
      elif cur.kind in primitive_types:
        pass
      elif cur.kind == TypeKind.FUNCTIONPROTO:
        self.init_funcproto()
      elif cur.kind == TypeKind.UNEXPOSED:
        children = list(self.cursor.get_children())
      elif cur.kind == TypeKind.CONSTANTARRAY:
        pass
      elif cur.kind == TypeKind.ENUM:
        self.init_enum()

      else:

        print(cur.get_declaration().displayname)
        print('got kind ', cur.kind, self.cursor.kind)
        print(cur.spelling)
        print(cur.get_pointee().kind)
        raise Exception('fail')

    self.size = self.internal_typ.get_size() * 8
    glog.debug('name=%s, base_kind=%s, typ=%s, base_type=%s, kind=%s', self.name, self.get_base_kind(),
              self.typ, self.base_typ, cur.kind)
    self._is_primitive = self.get_base_kind() in primitive_types
    self.typ_data = types_helper.get_by_clang(self.internal_typ.kind)
    #assert self.typ_data is not None, 'Bad shit for %s'%self.internal_typ.kind
    types_db[self.get_hash()] = self
    self.finalize()

  def setup_parent(self):
    par = self.cursor.semantic_parent
    while True:
      if par.kind == CursorKind.NAMESPACE or par.kind == CursorKind.UNEXPOSED_DECL:
        par = par.semantic_parent
      elif par.kind in (CursorKind.STRUCT_DECL, CursorKind.CLASS_DECL, CursorKind.UNION_DECL,
                        CursorKind.TRANSLATION_UNIT):
        self.parent = self.index.get_typ(par)
        self.parent.children.append(self)
        break
      else:
        glog.debug('%s >> %s', par.kind, par.location)
        assert False, "unhandled"

  def init_funcproto(self):
    cursor = self.cursor
    for x in self.typ.internal_typ.argument_types():
      self.deps.append(self.index.get_typ(None, x))

  def init_enum(self):
    self.is_enum = True
    self.base_typ = self.index.get_typ(None, self.cursor.enum_type)
    self.internal_typ = self.base_typ.internal_typ
    assert self.base_typ != None
    for u in self.cursor.get_children():
      elem_name = u.displayname
      self.add_enum_val(elem_name, u.enum_value)
      self.choices.add_choice(elem_name, u.enum_value, do_norm=False)

  def init_record(self):
    cursor = self.cursor
    self.name = to_str(cursor.displayname)
    tmp = cursor.canonical.type.get_canonical()
    glog.debug('adding record %s', self.name)

    for u in cursor.get_children():
      #print('adding record children >> ', u.kind, u.type.kind, cursor.location)
      if u.kind == CursorKind.TEMPLATE_NON_TYPE_PARAMETER:
        self.templates.append(u)
      elif u.kind == CursorKind.TEMPLATE_NON_TYPE_PARAMETER:
        self.templates.append(u)
      elif u.kind == CursorKind.FIELD_DECL:
        nfield = OpaField(self.index, u, self.typ)
        self.add_field(nfield)
      elif u.kind in (CursorKind.STRUCT_DECL, CursorKind.CLASS_DECL):
        self.index.get_typ(u)
    #print('DONE INNIT RECORD')

  def need_declare(self):
    if self.base_typ and self.base_typ.kind == TypeKind.FUNCTIONPROTO:
      return True
    return self.ptr_count == 0

  def dfs_mark(self):
    if self.mark:
      return
    self.mark = 1

    if self.parent:
      self.parent.dfs_mark()

    if self.base_typ:
      self.base_typ.dfs_mark()

    for k, x in self.fields.items():
      x.typ.dfs_mark()

  def dfs_build_info(self):
    if not self.mark:
      return None

    fwd_decl = [x for k, x in self.fields.items() if x.typ.forward_decl]
    decl = [x.rmp.dfs_build_info() for x in self.node.order]
    data = Dict(node=self, fwd=fwd_decl, decl=decl)
    return data

    def forward_decl_str(self):
      content = ''

    return content

  def decl_str(self):
    if self.cursor.kind == CursorKind.CLASS_DECL:
      s = 'class '
    elif self.cursor.kind == CursorKind.STRUCT_DECL:
      s = 'struct '
    else:
      assert False
    s += self.decl_typname()
    return s

  def full_typname(self):
    if not self.typ:
      return None

    tmp = OpaType.normalize_spelling(self.internal_typ).split('::')
    if not self.is_primitive() and self.index.modifier:
      tmp[-1] = self.index.modifier.name_mod(self, tmp[-1])
    return '::'.join(tmp)

  def field_typname_pre(self):
    full = self.full_typname()
    if self.get_base_typ().kind == TypeKind.FUNCTIONPROTO:
      return re.sub(r'\).*', '', full)
    else:
      return re.sub(r'\[.*\]', '', full)

  def field_typname_post(self):
    full = self.full_typname()
    if self.get_base_typ().kind == TypeKind.FUNCTIONPROTO:
      return re.sub(r'^[^)]*', '', full)
    else:
      m = re.search(r'\[.*\]', full)
      if not m:
        return ''
      return m.group(0)

  def decl_typname(self):
    if not self.typ:
      return None

    tmp = self.full_typname()
    tmp = tmp.split('::')[-1]
    return tmp

  def normalize_spelling(typ):
    name = typ.spelling
    name = re.sub(r'\s*(const|volatile)\s*', '', name)
    return name

  def get_parent(self):
    if self.base_typ:
      return self.base_typ.get_parent()
    return self.parent

  def get_base_kind(self):
    return self.get_base_typ().typ.kind

  def begin_def(self):
    if not self.typ:
      return ''
    assert self.cursor.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL)
    s = ''
    s += self.decl_str()
    s += '{'
    return s

  def end_def(self):
    if not self.typ:
      return ''
    s = '};'
    return s


class OpaTypeHandler(handlers.BaseHandler):

  def flatten(self, obj, data):
    data['data'] = obj.get_hash()
    return data

  def restore(self, data):
    return types_db[data['data']]

#class OpaTypeStructBuilder:
#
#  def __init__(self, name, fields=None, size=None, ctx=None):
#    self.fields = fields
#    self.typ = typ
#    self.name = name
#    assert size is not None
#    self.size = size
#    self.ctx = None
#
#  def build(self):
#    res = OpaBaseType()
#    res.name = name
#    res._is_primitive = False
#
#    for field in self.fields:
#      f = res.add_field(self.build_field(field))
#
#    res.size = self.size
#    res.base_size = self.size
#    return res
#
#  def build_field(self, field_desc):
#    res = OpaBaseField()
#    if isinstance(field_desc, str):
#      field_desc.sp
#
#
OpaTypeHandler.handles(OpaType)
