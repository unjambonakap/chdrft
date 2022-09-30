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
m = swig.opa_math_common_swig
c = swig.opa_crypto_swig

global flags, cache
flags = None
cache = None

gf2 = m.cvar.GF2
pr_gf2 = m.cvar.PR_GF2


def rand(n): return c.LFSR_GF2_small.rand(n)
def rand_normal(n): 
  res = c.LFSR_u32()
  res.init_rand(n, gf2)
  return res


def make_small(poly, state):
  if isinstance(poly, list):
    poly = Z.Format(poly).bit2num().v
  if isinstance(state, list):
    state = Z.Format(state).bit2num().v
  n = cmisc.highbit_log2(poly)
  res = c.LFSR_GF2_small()
  res.init(poly, n, state)
  return res


def to_small(lfsr):
  if isinstance(lfsr, dict):
    return make_small(lfsr.poly, lfsr.state)
  return make_small(poly2list(lfsr.get_poly()), poly2list(lfsr.get_state()))


def poly2list(p, sz=None): 
  if isinstance(p, int): return Z.Format(p).bitlist(sz).v
  if sz is None: return list(p.to_vec())
  return list(p.to_vec(sz))
def list2poly(e): return m.v_u32(list(map(int, e)))

def to_lfsr(lfsr):
  poly = Z.Format(lfsr.poly).bitlist().v
  state = Z.Format(lfsr.state).bitlist().v

  res = c.LFSR_u32()
  res.init(gf2, pr_gf2._import(list2poly(state)), pr_gf2._import(list2poly(poly)))

  return res


def lfsr_gen(lfsr=None, n=None, mx=-1, ng=0):
  if lfsr is None:
    lfsr = rand(n)

  while True:

    if not mx: break
    mx -= 1
    if ng: yield lfsr.get_next_non_galois()
    else: yield lfsr.get_next()
  return lfsr


def find_from_binary_seq(seq):

  sv = list2poly(seq)
  res = c.LFSR_u32()
  if res.init_from_seq(gf2, sv):
    res.advance(-len(seq))
    return res
  return None




def get_as_dict(lfsr, vec=0):
  if vec:
    if isinstance(lfsr, c.LFSR_u32):
      return cmisc.Attr(
          n=lfsr.size(),
          poly=poly2list(lfsr.get_poly()),
          state=poly2list(lfsr.get_state()),
      )
    return cmisc.Attr(
        n=lfsr.n,
        poly=Z.Format(lfsr.poly).bitlist(lfsr.n+1).v,
        state=Z.Format(lfsr.state).bitlist(lfsr.n).v,
    )

  else:
    d = get_as_dict(lfsr, vec=1)
    return cmisc.Attr(
        n=d.n,
        poly=Z.Format(d.poly).bit2num().v,
        state=Z.Format(d.state).bit2num().v,
    )


def test(ctx):
  print(get_as_dict(rand(20)))
  print(get_as_dict(rand(23)))
  return
  u = find_from_binary_seq([1, 1, 1, 1])
  print(u.get_next())
  print(u.get_next())
  print(u.get_next())

def test3(x):
  print('start')
  rand(30)

def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
