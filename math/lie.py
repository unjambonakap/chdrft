#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.opa_types import *
from pydantic.v1 import Field
from chdrft.utils.path import FileFormatHelper
import scipy.linalg

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def until_not_none(f):

  def wrap(*args, **kwargs):
    while True:
      res = f(*args, **kwargs)
      if res is not None: return res

  return wrap


class Lie:

  def g_e(self):
    raise Exception('123')

  def g_mul(self, a, b):
    raise Exception('123')

  def g_inv(self, a):
    raise Exception('123')

  def a_0(self):
    raise Exception('123')

  def a_add(self, a, b):
    raise Exception('123')

  def a_bracket(self, a, b):
    raise Exception('123')

  def a2g_exp(self, v):
    raise Exception('123')


def schur_tsf_2x2_blocks(mat, cb, one_blk):
  n = len(mat)
  t, z = scipy.linalg.schur(mat, output='real')
  db = t.diagonal(1)
  da = t.diagonal(0)[:-1]

  ti = np.abs(db) > 1e-6
  t1 = np.ones(n, dtype=bool)
  t1[:-1] &= ~ti
  t1[1:] &= ~ti

  b2 = list(cb(da[ti], db[ti]))
  blocks = []
  bpi = 0
  for i in range(n):
    if t1[i]:
      blocks.append([[one_blk]])
    elif i + 1 < n and ti[i]:
      blocks.append(b2[bpi])
      bpi += 1
  B = scipy.linalg.block_diag(*blocks)
  res = z @ B @ z.transpose()
  return res


class Lie_SO(Lie):

  def __init__(self, n):
    self.n = n

  def g_e(self):
    return np.identity(n)

  def g_mul(self, a, b):
    return a @ b

  def g_inv(self, a):
    return np.invert(a)

  def a_0(self):
    return np.zeros((n, n))

  def a_add(self, a, b):
    return a + b

  def a_bracket(self, a, b):
    return a @ b - b @ a

  def a2g_exp(self, v):

    def tsf_block(da, db):
      va = np.cos(db)
      vb = np.sin(db)
      return list(np.einsum('lij,lk->kij', [[[1, 0], [0, 1]], [[0, 1], [-1, 0]]], [va, vb]))

    return schur_tsf_2x2_blocks(v, tsf_block, 1)

  def g2a_log(self, v):

    def tsf_block(da, db):
      vx = np.arctan2(db, da)
      return np.einsum('ij,k->kij', [[0, 1], [-1, 0]], vx)

    return schur_tsf_2x2_blocks(v, tsf_block, 0)

    return scipy.linalg.logm(v)

  @until_not_none
  def g_gen(self):
    x = np.random.random((self.n, self.n)) - 0.5
    d = np.linalg.det(x)
    if d < 0.1: return None
    x = x / d**(1 / self.n)
    return np.linalg.qr(x)[0]

  @until_not_none
  def a_gen(self):
    x = np.random.random((self.n, self.n)) * 3
    x = np.tril(x)
    x = x - x.transpose()
    return x



  def da_pt_left(self, g, pt):
    v = g @ pt
    res = []
    for i in range(self.n):
      u = np.zeros(self.n)
      u[i,:] = pt
      u[:,i] = -pt
      u[i][i] = 0
      res.append(u)
    return res



def test(ctx):
  import jax
  expr =jax.make_jaxpr(lambda x,y: x[0]*x[0] + x[1] * y[2])(jax.numpy.array([1,2,3], dtype=float), jax.numpy.array([2,3], dtype=float))
  jax.grad(expr)
  print(expr)
  return
  l = Lie_SO(8)
  g = l.g_gen()
  a = l.a_gen()
  tg = [g]
  ta = []
  for i in range(4):
    ta.append(l.g2a_log(tg[-1]))
    tg.append(l.a2g_exp(ta[-1]))


  for x in (ta, tg):
    print('+++++')
    for ia, ib in cmisc.itertools.combinations(range(len(x)), 2):
      print(ia, ib, np.linalg.norm(x[ia] - x[ib]))

  return
  print(x)
  print()
  print(a)
  print()
  print(l.a2g_exp(a))
  tmp = (l.g2a_log(l.a2g_exp(a)))
  print(tmp)
  print(l.a2g_exp(tmp))
  t, z = scipy.linalg.schur(x, output='real')
  print(t)


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
