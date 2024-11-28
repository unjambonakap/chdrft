#!/usr/bin/env python

#%%
import os

dump_dir = '/tmp/dump6'
os.environ[
    "XLA_FLAGS"
] = f'--xla_force_host_platform_device_count=16 --xla_embed_ir_in_executable --xla_dump_to={dump_dir}'

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import numpy as np
from chdrft.utils.opa_types import *
from chdrft.utils.path import FileFormatHelper
import jax
from chdrft.jit.ojax import *
from chdrft.emu.func_call import g_data, StructBuilder, CodeStructExtractor, create_ctypes_caller
import ctypes

global flags, cache
flags = None
cache = None



def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class JaxCompiler(cmisc.PatchedModel):
  existing: set[str] = None
  dump_dir: str = cmisc.pyd_f(lambda: app.env.jax_dump_dir.name)
  lh: LLVMHelper = cmisc.pyd_f(LLVMHelper)

  def list_modules(self):
    files = cmisc.list_files_rec(self.dump_dir)
    modules = [os.path.basename(x).split('.')[0] for x in files]
    return {x for x in modules if x.startswith('module_')}

  @classmethod
  def Make(cls, **kwargs):
    res = cls(**kwargs)
    res.existing = res.list_modules()
    return res

  def compile(self, f, args) -> JaxCodeGen:
    jax.clear_caches()
    a = jax.jit(f)
    b = a.lower(*args)
    b.compile()
    new_modules = self.list_modules()
    new = new_modules - self.existing
    self.existing = new_modules
    assert len(new) == 1
    module_name = list(new)[0]
    tg = cmisc.filter_glob_list(
        cmisc.list_files_rec(self.dump_dir), f'{self.dump_dir}/{module_name}.*'
    )
    ex = JaxIRExtractor.Make(tg)

    ba = ex.last.buf_asst

    wrap_code = ba.generate_wrapper_code('wrapper', 'mainx')
    cg = JaxCodeGen(
        buf_asst=ba,
        ir_spec=ex.last.get_entry(kind=JaxDumpFileKind.IR, ir_opt=True),
        llvm_helper=self.lh,
    )
    return cg




#%%

#%%


def test(ctx):

  def f(a, b):
    b = b + 3
    b = b + 3

    return [b @ a @ b + 0xabcdef]

  argsx = (np.identity(4), np.ones(4))

  jc = JaxCompiler.Make()
  cg = jc.compile(f, argsx)

  rso = cg.generate_so()

  with cmisc.tempfile.NamedTemporaryFile(suffix='.so') as fx:
    FileFormatHelper.Write(fx.name, rso)
    l1 = ctypes.cdll.LoadLibrary(fx.name)
    g_data.set_m32(False)

    test_code = f'''
    float *{cg.export_func}(float a[16], float b[4]);
    '''
    g_code = StructBuilder()
    g_code.add_extractor(CodeStructExtractor(test_code, ''))
    g_code.build(extra_args=[], want_sysincludes=False)

    fc = create_ctypes_caller(l1, g_code)
    a = getattr(fc, cg.export_func)([1] * 16, [1] * 4)
    a = getattr(fc, cg.export_func)([1] * 16, [1] * 4)
    a = getattr(fc, cg.export_func)([1] * 16, [1] * 4)
    print(a.deref())
    return cg


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
#%%
