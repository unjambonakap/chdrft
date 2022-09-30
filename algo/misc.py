#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
import glog
import math, sys, os
import numpy as np
from chdrft.utils.types import *
from enum import Enum
from chdrft.utils.path import FileFormatHelper
from chdrft.utils.fmt import Format

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)



class LRUFifo:
  def __init__(self):
    self.clear()

  def touch(self, x):
    self.id += 1
    self.mp[x] = self.id
  def clear(self):
    self.mp = {}
    self.id = 0

  def get_all(self):
    tb = [(v,k) for k,v in self.mp.items()]
    tb.sort()
    for _,k in tb: yield k


def kmp(e):
  fail = list([None] * (len(e)+1))
  fail[0] = -1;
  fail[1] = 0
  p = 0
  for i in range(1, len(e)):
    while p != -1 and e[p] != e[i]: p = fail[p]
    p += 1
    fail[i+1] = p
  return fail

def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
