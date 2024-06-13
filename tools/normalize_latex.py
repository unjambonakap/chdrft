#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import chdrft.utils.Z as Z
import numpy as np
from pydantic.v1 import Field

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--infile')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  # for pandoc subfloat
  lines = []
  for x in Z.FileFormatHelper.Read(ctx.infile, mode='txt').splitlines():
    m = Z.re.match(r'\s*\\subfloat\s*(\[(?P<caption>[^\]]*)\])?\s*{(?P<content>.*)}\s*$', x)
    if m:
      u = m.groupdict()
      content = u['content']
      if u['caption']:
        content += r'\caption{%s}' % u['caption']
      x = r'\begin{subfigure}{\textwidth}' + content + r'\end{subfigure}'
    lines.append(x)


  res = '\n'.join(lines)
  print(res)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
