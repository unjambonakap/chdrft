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
import tempfile
from chdrft.utils.path import FileFormatHelper
import subprocess as sp

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)



class CXXEvaluator(cmisc.PatchedModel):
  headers: list[str] = Field(default_factory=list)
  def evaluate(self, expr):
    headers = '\n'.join(f'#include {x}' for x in self.headers)
    code = f'''

#include <iostream>
{headers}

int main(){{
std::cout<<({expr});
return 0;
}}
'''
    with  tempfile.TemporaryDirectory() as td:
      fname = f'{td}/in.cpp'
      fout = f'{td}/a.out'
      
      FileFormatHelper.Write(fname, code)
      sp.check_call(['g++', fname, '-o', fout], cwd=td)
      res = sp.check_output([fout], cwd=td)
      return res.decode()

def test(ctx):
  ev = CXXEvaluator(headers=[

'<linux/input.h>',
  ])
  print(ev.evaluate('EVIOCGID'))
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
