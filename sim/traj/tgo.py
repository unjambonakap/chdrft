#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.types import *
from pydantic import Field
from chdrft.utils.path import FileFormatHelper
import sage
from sage.all import var
import sage.all
from chdrft.sim.rb.base import Vec3, Transform, R
from astropy import constants as const

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class TGOSolver:

  def __init__(self, degs, dkp_end: np.ndarray = []):
    self.dkp_end = dkp_end
    self.data = self.build(degs)

  def dpfunc(self, xp: np.ndarray, vp: np.ndarray, tgo: float, integ):
    ev = self.eval(xp, vp, tgo)

    pxl = []
    for i, v in enumerate(ev):
      px = np.polynomial.Polynomial(v)
      csts = [vp[i], xp[i]][:integ]
      pxl.append(px.integ(m=integ, k=csts))

    def fx(t):
      return np.array(list(map(lambda x: x(t), pxl)))

    return fx

  def eval(self, xp, vp, tgo):

    def eval_entry(e, i):
      return float(e.subs(p_start=xp[i], tgo=tgo, v_start=vp[i]))

    dim = len(xp)
    entries = []
    for i in range(dim):
      entries.append([eval_entry(e, i) for e in self.data[i]])

    res = np.array(entries)
    return res

  def get(self, xp: np.ndarray, vp: np.ndarray, tgo: float):
    return self.eval(xp, vp, tgo)[:, 0]

  def dfunc(self, xp: np.ndarray, vp: np.ndarray, tgo: float):
    ev = self.eval(xp, vp, tgo)

    def fx(t):
      res = []
      for v in ev:
        s = 0
        for j, c in enumerate(v):
          if j > 0:
            s += c * j * (t**(j - 1))
        res.append(s)
      return np.array(res)

    return fx

  def func(self, xp: np.ndarray, vp: np.ndarray, tgo: float):
    ev = self.eval(xp, vp, tgo)

    def fx(t):
      res = []
      for v in ev:
        s = 0
        for j, c in enumerate(v):
          s += c * (t**j)
        res.append(s)
      return np.array(res)

    return fx

  def build(self, degs):
    dim = len(degs)
    t = var('t')
    dim2letter = 'xyz'
    tgo_v = var('tgo')
    lst = []
    for d in range(dim):
      x_s = var(f'p_start')
      v_s = var(f'v_start')

      nd = degs[d]
      assert nd >= 2
      vx = [var(f'p_{i}') for i in range(nd)]
      xdd = sum(vx[i] * (t**i) for i in range(nd))
      xd = xdd.integrate(t) + v_s
      x = xd.integrate(t) + x_s

      eqs = []
      cur = x
      for i in range(nd):
        tg = np.zeros(dim)
        if i < len(self.dkp_end):
          tg = self.dkp_end[i]

        eqs.append((cur.subs({t: tgo_v}) == tg[d]))
        cur = cur.derivative(t)

      res = sage.all.solve(eqs, vx)
      lst.append([e.rhs() for e in res[0]])
    return lst


class GravitySpec(cmisc.PatchedModel):
  center: Vec3
  mass: float

  def __call__(self, pos: Vec3 | np.ndarray) -> Vec3 | np.ndarray:
    if isinstance(pos, np.ndarray):
      assert len(pos.shape) == 2
      dim = pos.shape[0]
      diff = -(pos - self.center.vdata)
      norm = np.linalg.norm(diff, axis=1)
      k = const.G.value * self.mass / norm**3
      return diff * k.reshape((-1, 1))

    diff = self.center - pos
    norm = diff.norm
    if norm < 1e-6: return Vec3.Zero()
    k = const.G.value * self.mass / norm**3
    return diff * k


class TGOSnapshot(cmisc.PatchedModel):
  tgo: TGOSolver
  p: Vec3
  dp: Vec3
  tgo_t0: float
  t0: float

  def get_p(self, t: float) -> Vec3:
    f = self.tgo.func(self.p.vdata, self.dp.vdata, self.tgo_t0)
    res = Vec3(f(t - self.t0))
    return res

  def get_dp(self, t: float) -> Vec3:
    df = self.tgo.dfunc(self.p.vdata, self.dp.vdata, self.tgo_t0)
    return Vec3(df(t - self.t0))


def analyze_tgo(sx: TGOSolver, p0: Vec3, v0: Vec3, t_tgo: float, gspec: GravitySpec):
  tt = np.linspace(0, t_tgo)
  a = sx.dpfunc(p0.vdata, v0.vdata, t_tgo, 0)(tt).T
  p = sx.dpfunc(p0.vdata, v0.vdata, t_tgo, 2)(tt).T
  a = a - gspec(p)
  anorm = a / np.linalg.norm(a, axis=1, keepdims=True)
  dot = np.sum(anorm[1:] * anorm[:-1], axis=1)
  ang = np.arccos(np.clip(dot, -1, 1))
  return A(a=a, ang=ang)


def test(ctx):
  sx = TGOSolver([3, 2], [np.ones(2)])
  evx = sx.eval([10, 0], [0, 0], 10)
  fx = sx.func([10, 0], [0, 0], 10)
  print(evx)

  return

  dt = 1e-1
  tl0 = np.arange(0, 30, dt)
  state = [p0, v0.vdata * 0]
  tb = []
  tbv = []
  tba = []

  for t in tl0:

    fx = sx.func(xp=state[0], vp=state[1], tgo=t_tgo)
    cur = sx.eval(xp=state[0], vp=state[1], tgo=t_tgo)[0] * np.array([1, 0, 0])
    cur = fx(0)
    print(cur)
    tba.append(cur)
    state = state[0] + state[1] * dt + cur / 2 * dt**2, state[1] + cur * dt
    tb.append(state[0])
    tbv.append(state[1])
  tb = np.array(tb)
  tbv = np.array(tbv)
  tba = np.array(tba)


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
