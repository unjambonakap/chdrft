#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, cwdpath
import glog
import glob
import hashlib, os, subprocess, sys

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)
  parser.add_argument('--base', type=cwdpath)
  parser.add_argument('--candidates', type=cwdpath, nargs='*')
  parser.add_argument('--glob', type=str, default='')
  parser.add_argument('--dir1')
  parser.add_argument('--dir2')


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


def diffdir(ctx):

  dir1 = os.path.realpath(ctx.dir1)
  dir2 = os.path.realpath(ctx.dir2)

  for root, dirs, files in os.walk(dir1):
    for f in files:
      if not f.endswith('.py'): continue
      f1 = '{}/{}'.format(root, f)
      f2 = f1.replace(dir1, dir2, 1)

      # Check if files are the same; in which case a diff is useless
      h1 = hashlib.sha256(open(f1, 'rb').read()).hexdigest()
      if not os.path.exists(f2):
        print('FILE NOT EXISTING ', f2)
        continue
      h2 = hashlib.sha256(open(f2, 'rb').read()).hexdigest()
      if h1 == h2: continue

      # Don't diff binary files
      if open(f1, 'rb').read().find(b'\000') >= 0: continue

      subprocess.call(['vimdiff', f1, f2])


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
