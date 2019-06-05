#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)
  parser.add_argument('--infile', type=str)
  parser.add_argument('--outfile', type=str)
  parser.add_argument('--intype', type=str)
  parser.add_argument('--outtype', type=str)


def conv(ctx):
  data = Z.DataFile(ctx.infile, typ=ctx.intype)
  data.to_ds().to_file(ctx.outfile, ctx.outtype)




def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
