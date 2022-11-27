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
from pydantic import Field
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
from chdrft.jit.jax import *
from chdrft.emu.func_call import g_data, StructBuilder, CodeStructExtractor, create_ctypes_caller
import ctypes

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  tg = '/home/benoit/programmation/projects/rockets/tests/dump6/'
  ex = JaxIRExtractor.Make(tg)

  ba = ex.last.buf_asst
  lh = LLVMHelper(bin_dir='/home/benoit/packets/llvm-git/src/_build/bin', ctx=app.global_context)
  wrap_code = ba.generate_wrapper_code('wrapper', 'mainx')
  cg = JaxCodeGen(
      buf_asst=ba,
      ir_spec=ex.last.get_entry(kind=JaxDumpFileKind.IR, ir_noconst=True, ir_opt=True),
      llvm_helper=lh,
  )
  obj = cg.generate()

  rso = lh.obj2so(obj)

  FileFormatHelper.Write('/tmp/test.so', rso)



  l1 = ctypes.cdll.LoadLibrary('/tmp/test.so')
  g_data.set_m32(False)

  test_code = f'''
  float *{cg.export_func}(float a[16], float b[4]);
  '''
  g_code = StructBuilder()
  g_code.add_extractor(CodeStructExtractor(test_code, ''))
  g_code.build(extra_args=[], want_sysincludes=False)




  fc =create_ctypes_caller(l1, g_code)
  a = getattr(fc, cg.export_func)([1]*16, [1]*4)
  print(a.deref())


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)

app()
