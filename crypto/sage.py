#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import numpy as np
from chdrft.utils.swig import swig
from sage.all import *
m = swig.opa_math_common_swig
c = swig.opa_crypto_swig

global flags, cache
flags = None
cache = None

gf2 = m.cvar.GF2
pr_gf2 = m.cvar.PR_GF2


def find_galois_state(p, obs):
  n = len(p) - 1
  pr = GF(2)['x']
  qr = QuotientRing(pr, pr(p))
  x = qr([0, 1])
  m = matrix(GF(2), len(obs), n)
  assert len(obs) >= n
  b = []
  for i in range(len(obs)):
    v = x**(obs[i].pos - 1)
    for j in range(n):
      m[i, j] = (v * x**j)[n - 1]
    b.append(obs[i].val)
  print(m)
  return list(m.solve_right(vector(GF(2), b)))

def find_nongalois_lfsr(s, n):

  m = matrix(GF(2), len(s)-n, n)
  for i in range(len(s)-n):
    m[i] = s[i:i+n][::-1]
  r = s[n:]
  return [1] + list(m.solve_right(vector(GF(2), r)))


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  nx = 10
  ev = Z.np.random.choice(list(range(1, len(seq))), size=nx, replace=0)
  obs = [cmisc.Attr(pos=i + 1, val=seq[i]) for i in ev]
  print(len(obs))
  find_galois_state(p1, obs)

def test2(ctx):
  data = Z.FileFormatHelper.Read('/tmp/test.pickle')

  obs = []
  for i in range(len(data.seq)):
    obs.append(cmisc.A(pos=7+7*i, val=data.seq[i]))
  res = find_galois_state(data.poly, obs)
  print(res)
  print('done')

def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
