#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, cwdpath
import glog
import glob

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test).add(embed_files)
  ActionHandler.Prepare(parser, clist.lst)
  parser.add_argument('--base', type=cwdpath)
  parser.add_argument('--candidates', type=cwdpath, nargs='*')
  parser.add_argument('--glob', type=str, default='')



def test(ctx):
  print('on test')


def embed_files(ctx):
  base = ctx.base
  content = open(base, 'rb').read()

  cnds = []
  cnds += ctx.candidates
  if ctx.glob:
    cnds += glob.glob(ctx.glob)

  for other in cnds:
    print('trying file ', other)
    if other == base: continue
    try:
      cur = open(other, 'rb').read()
      matching = content.find(cur)
      if not matching != -1: continue
      print(f'{other} matches at {matching}')
    except:
      glog.error('Failing on %s', other)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
