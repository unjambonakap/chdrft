#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import numpy as np
import pickle
import chdrft.utils.Z as Z

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)

class X:
  def __init__(self, abc):
    print('x', abc)

def test(ctx):
  a = cmisc.DictWithDefault(list)
  a['abc'].append(123)
  res = pickle.dumps(a)
  b =pickle.loads(res)


def fiji(ctx):
  import imagej
  ijx = imagej.init('/home/benoit/packets/Fiji.app/', headless=0)
  import jnius
  from jnius import autoclass, cast
  ijx.ui().showUI()

  try:
    while True:
      print('sleeping...')
      Z.time.sleep(1)
  except Exception as e:
    print('Got ', e)
    raise


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
