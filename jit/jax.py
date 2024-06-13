#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.types import *
from pydantic.v1 import Field
from chdrft.utils.path import FileFormatHelper
from jax.config import config
import jax
import jax.numpy as jnp
import enum
import re
import tempfile
import subprocess as sp
from contextlib import ExitStack
from pathlib import Path
import os

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class JaxModule(cmisc.PatchedModel):
  dump_dir: str
  base_name: str


class JaxDumpFileKind(enum.Enum):
  IR = 'ir'
  HLO = 'hlo'
  BUF_ASST = 'buf_asst'
  OBJ = 'obj'


class JaxDumpFileSpec(cmisc.PatchedModel):
  full_path: str
  name: str
  module: str
  extension: str
  func_name: str
  kind_str: str

  kind: JaxDumpFileKind
  ir_opt: bool = False
  ir_noconst: bool = False
  hlo_after_opt: bool = False

  @property
  def content(self) -> str:
    with open(self.full_path, 'r') as f:
      return f.read()

  def ir_with_funcname(self, funcname: str) -> str:
    res, cnt = re.subn('(define void @)(.*)\(', f'\\1{funcname}(', self.content)
    assert cnt == 1
    return res

  @classmethod
  def Make(cls, full_path) -> "JaxDumpFileSpec | None":
    name = os.path.basename(full_path)
    if not name.startswith('module_'): return None
    components = name.split('.')
    module, func_name, *kind_ext = components
    kind_str = ''

    kind = JaxDumpFileKind.IR
    if kind_ext[-1] == 'o':
      kind = JaxDumpFileKind.OBJ
      ext = kind_ext[0]
    else:
      kind_str, ext = kind_ext

    if kind_str.endswith('buffer-assignment'):
      kind = JaxDumpFileKind.BUF_ASST
    elif ext == 'txt':
      kind = JaxDumpFileKind.HLO

    res = JaxDumpFileSpec(
        full_path=full_path,
        name=name,
        module=module,
        func_name=func_name,
        kind_str=kind_str,
        kind=kind,
        extension=ext
    )
    res.hlo_after_opt = 'cpu_after_optimizations' in kind_str
    res.ir_opt = 'with-opt' in kind_str
    res.ir_noconst = 'noconst' in kind_str
    return res

  def as_buf_asst(self) -> "JaxBufAsst":
    return JaxBufAsst.Make(self)


class JaxDumpModule(cmisc.PatchedModel):
  entries: list[JaxDumpFileSpec]
  name: str

  def get_entry(self, **kwargs) -> JaxDumpFileSpec:
    return cmisc.asq_query(self.entries).where_like(**kwargs).single()

  @property
  def buf_asst(self) -> "JaxBufAsst":
    x = self.get_entry(kind=JaxDumpFileKind.BUF_ASST)
    return x.as_buf_asst()


class JaxIRExtractor(cmisc.PatchedModel):
  dump_dir: str
  name2modules: dict

  @property
  def last(self) -> JaxDumpModule:
    return list(self.name2modules.values())[-1]

  @classmethod
  def Make(cls, dump_dir) -> "JaxIRExtractor":
    files = cmisc.list_files_rec(dump_dir)
    specs = list(filter(None, map(JaxDumpFileSpec.Make, files)))

    name2modules = dict()
    for grp in cmisc.asq_query(specs).group_by(lambda x: x.module).order_by(lambda x: x.key):
      name2modules[grp.key] = JaxDumpModule(name=grp.key, entries=list(grp))

    return cls(dump_dir=dump_dir, name2modules=name2modules)


class JaxBufAsst(cmisc.PatchedModel):
  """JaxBufAsst(params=[{'num': 0, 'size': 64, 'parameter': 0, 'shape': ['|f32[4,4]| at ShapeIndex {}']}, 
    {'num': 1, 'size': 16, 'parameter': 1, 'shape': ['|f32[4]| at ShapeIndex {}']},
    {'num': 2, 'size': 4, 'output': ['shape is |f32[]|'], 'maybe-live-out': []}, 
    {'num': 3, 'size': 4, 'constant': []}, 
    {'num': 4, 'size': 16, 'preallocated-temp': []}]
    """

  params: list
  spec: JaxDumpFileSpec

  @classmethod
  def Make(cls, spec: JaxDumpFileSpec) -> "JaxBuxAsst":
    part = re.search(
        'BufferAssignment:(?P<data>.*?)Total bytes used', spec.content, re.MULTILINE | re.S
    ).groupdict()['data']
    lines = part.strip().splitlines()
    params = []
    for i in range(0, len(lines), 2):
      g1 = re.search('allocation (?P<num>\d+): (?P<rem>.*):', lines[i]).groupdict()
      num = g1['num']
      props = A(num=[num])
      params.append(props)
      for kv in g1['rem'].split(', '):
        k, *v = kv.split(' ', maxsplit=1)
        props[k] = v

      #
      for k in props.keys():
        v = props[k]
        if k in ('num', 'parameter', 'size'):
          v = int(v[0])
        props[k] = v

    return cls(params=params, spec=spec)

  @property
  def args(self) -> list[A]:
    return [x for x in self.params if 'parameter' in x]

  @property
  def output(self) -> A:
    return self.get_entry('output')

  @property
  def constant(self) -> A:
    return self.get_entry('constant')

  def get_entry(self, name) -> A:
    return cmisc.get_uniq(x for x in self.params if name in x)

  def generate_wrapper_code(self, wrapper_name: str, func_name: str) -> str:
    output_sym = 'output'

    extra_args = dict()
    extra_args[output_sym] = 'output'  # order matters !!
    extra_args.update(constant='constant', tmp='preallocated-temp')

    buffer_table_sym = 'buffer_table'
    extra_args_list = []
    for k, v in extra_args.items():
      e = self.get_entry(v)
      extra_args_list.append(f'static char *{k}[{e.size}];')
    extra_args_def = '\n'.join(extra_args_list)

    setup = []

    def add_buffer_entry(name):
      setup.append(f'{buffer_table_sym}[{len(setup)}] = (void*){name};')

    args = ','.join(f'void *param_{i}' for i in range(len(self.args)))
    for x in self.args:
      add_buffer_entry(f'param_{x.parameter}')
    for k in extra_args.keys():
      add_buffer_entry(k)

    buffer_setup = '\n'.join(setup)
    buffer_table_size = len(setup)

    res = cmisc.template_replace_safe(
        '''

extern "C" {
void ${func_name}(void *retval, void *run_options, void *params, void *buffer_table, void *status,
           void *prof_counters);
void *${wrapper_name}(${args});
}
${extra_args_def}

void *${wrapper_name}(${args}) {
    void *${buffer_table_sym}[${buffer_table_size}];
    ${buffer_setup}
    
    ${func_name}(nullptr, nullptr, nullptr, ${buffer_table_sym}, nullptr, nullptr);
    
    return ${output_sym};
}
        ''', **locals()
    )
    return res


class LLVMHelper(cmisc.PatchedModel):
  bin_dir: str
  ctx: ExitStack

  @property
  def llc_path(self) -> str:
    return os.path.join(self.bin_dir, 'llc')

  @property
  def opt_path(self) -> str:
    return os.path.join(self.bin_dir, 'opt')

  @property
  def clang_path(self) -> str:
    return os.path.join(self.bin_dir, 'clang')

  @property
  def llvm_link_path(self) -> str:
    return os.path.join(self.bin_dir, 'llvm-link')

  def cpp2ir(self, content) -> str:
    f = self.make_file(content, suffix='.cpp')
    ir = sp.check_output([self.clang_path, '-c', f, '-emit-llvm', '-S', '-o', '-']).decode()
    return ir

  def make_file(self, content: str | bytes, suffix: str) -> str:
    f = self.ctx.enter_context(tempfile.NamedTemporaryFile(suffix=suffix, mode='w+'))
    FileFormatHelper.Write(f.name, content)
    return Path(f.name)

  def read(self, p: Path) -> str:
    with open(p, 'r') as f:
      return f.read()

  def make_path(self, x: str | Path, suffix) -> Path:
    if isinstance(x, Path): return x
    return self.make_file(x, suffix)

  def merge_irs(self, ir1: str | Path, ir2: str | Path) -> str:
    ir1 = self.make_path(ir1, '.ll')
    ir2 = self.make_path(ir2, '.ll')
    ir = sp.check_output([self.llvm_link_path, ir1, ir2, '-S', '-o', '-']).decode()
    return ir

  def ir2obj(self, ir1: str | Path, options=[], opt=0) -> bytes:
    options = list(options)
    if opt: options.append('-O3')

    ir1 = self.make_path(ir1, '.ll')
    obj = sp.check_output([self.clang_path, ir1, '-c', '-o', '-'] + options)
    return obj

  def obj2so(self, obj: bytes | Path) -> bytes:
    options = []
    obj = self.make_path(obj, '.o')
    path = self.make_file(b'', '.so')
    obj = sp.check_output([self.clang_path, obj, '-shared', '-o', path] + options)
    return FileFormatHelper.Read(str(path))


class JaxCodeGen(cmisc.PatchedModel):
  buf_asst: JaxBufAsst
  ir_spec: JaxDumpFileSpec
  llvm_helper: LLVMHelper
  export_func: str = 'func'

  def generate(self) -> bytes:
    funcname = 'test1'
    ir = self.ir_spec.ir_with_funcname(funcname)
    ir_wrapper = self.llvm_helper.cpp2ir(
        self.buf_asst.generate_wrapper_code(self.export_func, funcname)
    )
    res_ir = self.llvm_helper.merge_irs(ir, ir_wrapper)
    res_obj = self.llvm_helper.ir2obj(res_ir)
    return res_obj


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
