#!/usr/bin/env python

from clang.cindex import TranslationUnit, Index, CursorKind, TypeKind, Cursor, Type, TokenKind
import pprint as pp
from chdrft.utils.misc import Dict, Attributize, devnull
from chdrft.graph.base import OpaGraph
from collections import deque
import subprocess as sp
import re
import io
import sys
from chdrft.gen.types import *
import logging


class OpaModifier:

  def name_mod(self, node, typname):
    return typname

  def extra_mod(self, node):
    return ''

  def match(self, x):
    return True

  def match_var(self, x):
    return True

  def match_macro(self, x):
    return True

  def format_code(self, code):
    proc = sp.Popen(['clang-format'], stdin=sp.PIPE, stdout=sp.PIPE)
    out, err = proc.communicate(code.encode())
    assert proc.returncode == 0
    return out.decode()


class LocationWrapper:

  def __init__(self, location, inputfile):
    self.location = location
    self.is_input = False
    if location.file:
      self.is_input = location.file.name == inputfile

  def __str__(self):
    return 'LocationWrapper(location={location}, is_input={is_input})'.format(**self.__dict__)


class Filter:

  def __call__(self, cursor, filename):
    return True, self

  def add(self, e):
    pass


class EasyFilter(Filter):

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


InputOnlyFilter = EasyFilter(__call__=lambda self, cursor, x: [x.is_input, self])


class OpaIndex:

  def __init__(self, filter, inputfile):
    self.typs = {}
    self.vars = []
    self.macros = []
    self.typedefs = []
    self.modifier = None
    self.filter = filter
    self.inputfile = inputfile
    self.structs = []
    self.functions = {}

  def set_modifier(self, modifier):
    assert not modifier or isinstance(modifier, OpaModifier)
    self.modifier = modifier

  def get_typ(self, cursor, typ=None):
    builder = TypeBuilder(cursor, typ)
    key = builder.get_key()

    if not key in self.typs:
      elem = self.typs[key] = builder.get_type(self)
      elem.build()

    return self.typs[key]

  def add_typedef(self, cursor):
    tmp = OpaTypedef(self, cursor)
    self.typedefs.append(tmp)
    return tmp

  def add_var(self, cursor):
    tmp = OpaVar(self, cursor)
    if tmp.build():
      self.vars.append(tmp)
    return tmp

  def add_macro(self, cursor):
    tmp = OpaMacro(self, cursor)
    if tmp.build():
      self.macros.append(tmp)
    return tmp

  def add_hard_dep(self, src, dst, graph):
    dst = dst.get_base_typ()
    if dst.is_primitive():
      return
    graph.add_dep(src.node, dst.node)

  def add_function(self, cursor):
    tmp = OpaFunction(self, cursor)
    if tmp.build():
      self.functions[tmp.shortname] = tmp
    return tmp

  def build_graph(self):
    for x in self.typs.values():
      if x.is_primitive():
        continue
      x.node = OpaGraph.Node(x)
    if None not in self.typs:
      return

    root = self.typs[None]
    graph = OpaGraph(root.node)
    for x in self.typs.values():
      if x.is_primitive():
        continue
      x.node.children = [y.node for y in x.children]
      if x.parent:
        x.node.parent = x.parent.node

    for x in self.typs.values():
      for y in x.fields:
        if not y.typ.need_declare() and y.typ.get_parent() == x.get_parent():
          y.typ.get_base_typ().forward_decl = True
        else:
          self.add_hard_dep(x, y.typ, graph)

        # adding hard dependencies cause I dont know better
        for z in y.typ.get_base_typ().deps:
          self.add_hard_dep(x, z, graph)

    graph.solve_ordering()

  def reset_mark(self):
    for x in self.typs:
      x.mark = 0

  def get_build_info(self):
    res = self.typs[None].dfs_build_info()
    res.vars = [x for x in self.vars if x.mark]
    res.macros = [x for x in self.macros if x.mark]

    return res

  def build_info_to_decl(self, build_info):
    if not build_info:
      return ''
    node = build_info.node
    content = ''

    content += node.begin_def()

    for x in build_info.fwd:
      content += x.typ.decl_str() + ';'

    for x in build_info.decl:
      content += self.build_info_to_decl(x)

    for x in node.fields:
      content += x.field_decl_str() + ';'

    if node.index.modifier:
      content += self.modifier.extra_mod(node)

    for x in getattr(build_info, 'vars', []):
      content += x.var_decl_str() + ';'

    for x in getattr(build_info, 'macros', []):
      content += x.macro_decl_str() + '\n'

    content += node.end_def()

    return content

  def mark_matching(self):
    for x in self.typs.values():
      if x.cursor and self.modifier.match(x):
        x.dfs_mark()
    for x in self.vars:
      if self.modifier.match_var(x):
        x.mark = True

    for x in self.macros:
      if self.modifier.match_macro(x):
        x.mark = True

  def to_location(self, loc):
    return LocationWrapper(loc, self.inputfile)

  def get_code(self, modifier):
    self.set_modifier(modifier)
    self.mark_matching()
    self.build_graph()
    build_info = self.get_build_info()
    res = self.build_info_to_decl(build_info)
    self.set_modifier(None)
    return modifier.format_code(res)

  def fill(self, cursor):
    ok, adder = self.filter(cursor, self.to_location(cursor.location))
    if ok:

      e = None
      if cursor.kind == CursorKind.TYPEDEF_DECL:
        e = self.add_typedef(cursor)
      elif cursor.kind in (CursorKind.STRUCT_DECL, CursorKind.CLASS_DECL, CursorKind.CLASS_TEMPLATE):
        e = self.get_typ(cursor)
        self.structs.append(e)
      elif cursor.kind == CursorKind.VAR_DECL:
        e = self.add_var(cursor)
      elif cursor.kind == CursorKind.MACRO_DEFINITION:
        e = self.add_macro(cursor)
      elif cursor.kind == CursorKind.FUNCTION_DECL:
        e = self.add_function(cursor)

      if e:

        if adder:
          adder.add(e)

        if cursor.kind == CursorKind.TYPEDEF_DECL:
          return

    children = cursor.get_children()
    for c in children:
      self.fill(c)

  @staticmethod
  def create_index(filter=Filter(), args=None, cpp_mode=False, filename=None, file_content=None):
    #print('create >> ', args)
    if not args:
      args = []
    if cpp_mode:
      args.extend(['-std=c++11', '-Wall', '-Wextra'])
    for e in get_clang_includes():
      args.append('-isystem%s' % e)

    logging.info(f'Creating index with {args}')

    #print(args)

    args = [x.encode() for x in args]
    index = Index.create()
    options = 0
    options = TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    extra = {}

    if file_content:
      filename = 'dummy_RAWWWW.cpp'
      extra['unsaved_files'] = [(filename.encode(), file_content.encode())]
    else:
      assert filename

    tu = index.parse(filename.encode(), args, options=options, **extra)
    diags=list(tu.diagnostics)
    if len(diags)>0:
      logging.info('Clang diags: %d', len(diags))

    opa_index = OpaIndex(filter, filename)
    for x in tu.cursor.walk_preorder():
      if not filter(x, opa_index.to_location(x.location)):
        continue
      if 1: continue
      print(x.kind, x.type.kind, conf.lib.clang_Type_getNumTemplateArguments(x.type),
            x.get_num_children(), x.displayname, x.spelling, x.type.spelling)
      decl = x.type.get_declaration()

      tmp = conf.lib.clang_getSpecializedCursorTemplate(x)
      if tmp:
        print('Loc ', tmp.location)
      if x.kind == CursorKind.TYPE_REF:
        print('Typeref:')
      if x.kind == CursorKind.FIELD_DECL:
        print('FIELD >> ')
      if x.kind == CursorKind.VAR_DECL:
        a = x.type
        a = a.get_declaration()
        base = conf.lib.clang_getSpecializedCursorTemplate(a)
        if base:
          for k in base.walk_preorder():
            print('CHILDREN > ', k.displayname, k.kind, k.type.kind, k.get_num_children())

        tmp = x.canonical.type.get_canonical().get_declaration()

      if x.kind == CursorKind.FUNCTION_DECL:
        for a in x.get_arguments():
          print('args', a, a.displayname, a.type.kind)
        for a in x.get_children():
          print('children ', a.kind)
    if 0:
      print('\n\n\n\n')

    opa_index.fill(tu.cursor)
    opa_index.get_typ(None).mark = True

    return opa_index


def get_clang_includes():
  res = sp.check_output(['clang', '-E', '-x', 'c', '-', '-v'],
                        stdin=devnull,
                        stderr=sp.STDOUT).decode()
  st = res.find('search starts here')
  nd = res.find('End of search list')
  assert st != -1 and nd != -1
  res = res[st:nd]
  for entry in re.findall('/usr/[\S]+', res):
    yield entry


def test_main():
  print(list(get_clang_includes()))
  return
  # index=create_index(filename='./test.cpp')
  file_content = """
//#include <sys/mman.h>
#define PROT_READ 0x12
const int XX = PROT_READ;
//typedef double jambon;
//struct KAPA{int DEF, j; void *ptr;};
//struct tmp{ int (*f)(KAPA fu, int x, float y); };
"""

  class CurModifier(OpaModifier):

    def name_mod(self, node, typname):
      return typname

    def extra_mod(self, node):
      return ''

    def match(self, x):
      return x.cursor.kind == CursorKind.STRUCT_DECL

  print(file_content)
  args = ['-isystem/usr/lib/llvm-3.6/lib/clang/3.6.2/include']
  index = OpaIndex.create_index(file_content=file_content, cpp_mode=True, args=args)
  res = index.get_code(CurModifier())

  print(res)


if __name__ == '__main__':
  test_main()
