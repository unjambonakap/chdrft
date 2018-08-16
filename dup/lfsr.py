#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
from chdrft.utils.swig import swig
from chdrft.utils.fmt import Format

import glog

global flags, cache
flags = None
cache = None

m = swig.opa_math_common_swig
c = swig.opa_crypto_swig


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst)
  parser.add_argument('--n', type=int, default=10)
  parser.add_argument('--nv', type=int)
  parser.add_argument('--wordsize', type=int, default=32)


class LFSR:

  def __init__(self, coeffs, n, state=1):
    poly = 0
    for j in coeffs:
      poly |= 1 << j
    self.coeffs = coeffs
    self.poly = poly
    self.n = n
    self.state = state

  def next(self):
    b = self.state >> (self.n - 1)
    self.state = self.state << 1
    assert b == 0 or b == 1
    if b:
      self.state ^= self.poly
    return b


def test(ctx):
  n = ctx.n
  lfsr = c.LFSR_u32()
  lfsr.init_rand(n, m.cvar.GF2)
  nval = lfsr.max_period()
  if ctx.nv is not None: nval = ctx.nv * ctx.wordsize

  lst = [lfsr.get_next() for _ in range(nval)]
  s = '\n'.join([f"'{x}" for x in lst])
  return s


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
