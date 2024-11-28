#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import yaml
import pprint

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst)
  parser.add_argument('--file', nargs=2)


def start_diff(ctx, a, b):
  if isinstance(a, list) or isinstance(a, dict):
    if len(a) != len(b):
      ctx.diff = (a, b)
      print('=aaaaaaaaa')
      pprint.pprint(ctx.diff[0], depth=5)

      print('=++++++++++++++++++++++++++')
      pprint.pprint(ctx.diff[1], depth=5)
      print('\n\n\n')
      return
  else: return

  if isinstance(a, dict):
    a = a.values()
    b = b.values()

  for i, j in zip(a,b):
    start_diff(ctx, i,j)


def proc(ctx, file1, file2):
  a1 = yaml.load(open(file1, 'r'))
  a2 = yaml.load(open(file2, 'r'))

  try:
    start_diff(ctx, a1, a2)


  except:
    print('Got diff')

def test(ctx):
  proc(ctx, ctx.file[0], ctx.file[1])


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
