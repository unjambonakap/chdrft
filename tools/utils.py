#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import numpy as np

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--infile')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def skeleton(ctx):
  Z.pprint(cmisc.Attr.Skeleton(Z.FileFormatHelper.Read(ctx.infile)))


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
