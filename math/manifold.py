#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
from chdrft.struct.geo import DynamicKDTree
import itertools
from queue import SimpleQueue
from chdrft.struct.base import Range1D
from scipy.optimize import fsolve
import sympy
import numpy as np
from chdrft.utils.omath import *


global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  parser.add_argument('--delta', type=float, default=1e-1)
  parser.add_argument('--nsteps', type=int, default=100)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  pass


def make_norm(v):
  return v / np.linalg.norm(v)


def unit_vec(i, n):
  res = [0] * n
  res[i] = 1
  return res


class Parametrization:

  def __init__(self, kset, pset, syms):
    self.kset = tuple(kset)
    self.pset = tuple(pset)
    self.k = len(kset)
    self.p = len(pset)
    self.n = self.k + self.p
    self.syms = syms

  def __repr__(self):
    return f'Param(kset={self.kset}, pset={self.pset})'

  def get_set(self, p):
    return self.pset if p else self.kset

  def pset_id(self, p):
    for i, v in enumerate(self.pset):
      if v == p: return i
    assert 0

  def get_symset(self, p):
    return self.syms[list(self.get_set(p))]

  def compute_jacobian(self, m, pt_X, p):
    ss = self.get_symset(p)
    j = m.jacobian(ss)
    return j.subs(list(zip(self.syms, pt_X)))

  def remap_vec(self, v):
    res = [0] * self.n
    for i in range(self.k):
      res[self.kset[i]] = v[i]

    for i in range(self.p):
      res[self.pset[i]] = v[self.k + i]
    res = np.array(res, dtype=float)
    return res


class ParametrizationHelper:

  def __init__(self, n, pids=()):
    self.n = n
    self.pids = tuple(pids)

    self.syms = []
    for i in range(self.n):
      self.syms.append(sympy.Symbol(f'x{i}'))
    self.syms = np.array(self.syms)

  def list_params(self, p):
    k = self.n - p

    rem = []
    for i in range(self.n):
      if i not in self.pids:
        rem.append(i)
    for c in itertools.combinations(rem, k - len(self.pids)):
      pset = c + self.pids
      kset = []
      for i in range(self.n):
        if i not in pset: kset.append(i)
      yield Parametrization(kset, pset, self.syms)


def list_query_closest(pt, lst):
  if not lst: return None, None
  return min(((np.linalg.norm(pt - x), x) for x in lst), key=lambda x: x[0])


class ManifoldWalker:

  def __init__(self, f, coords, eps=1e-9, delta=1e-2):
    self.coords = coords
    self.f = f  # f: Rn x Rp -> Rn
    self.param_helper = f.get_param_helper(coords.dim)
    self.eps = eps
    self.delta = delta
    self.kdtree = DynamicKDTree()

  def froot(self, p, pad=1, **kwargs):

    res = self.f.froot(p, **kwargs)
    if pad: res += [0] * (self.param_helper.n - len(res))
    return res

  def froot_fix_p(self, block_ids, cur_sol):

    def f(p):
      pp = list(p)
      for idx in block_ids:
        pp[idx] = cur_sol[idx]
      return self.froot(np.array(pp))

    return f

  def compute_grad(self, pt_X, param, m):
    jx = param.compute_jacobian(m, pt_X, 0)
    jt = param.compute_jacobian(m, pt_X, 1)
    detv = abs(jx.det())
    if detv < self.eps: return None
    gx = -jx.inv() * jt
    vecs = []
    for i in range(param.p):
      v = cmisc.flatten(gx[:, i].tolist()) + unit_vec(i, param.p)
      vecs.append(param.remap_vec(v))
    return cmisc.Attr(vecs=vecs, det=detv)

  def do_fsolve(self, p, block_ids=None):
    if block_ids is not None:
      return fsolve(self.froot_fix_p(block_ids, p), p, factor=1)
    return fsolve(self.froot, p, factor=1)

  def pt_to_sol(self, p, fix_p=None):
    if fix_p is None: return self.do_fsolve(p)
    block_ids = []
    for i in range(self.param_helper.n):
      if i == fix_p.moveid:
        block_ids.append(i)

    res = self.do_fsolve(p, block_ids=block_ids)
    for idx in block_ids:
      res[idx] = p[idx]

    return res

    m = self.f(self.param_helper.syms)
    m = sympy.Matrix(m)
    j = m.jacobian(self.param_helper.syms)

    for i in range(self.param_helper.n):
      for ij in range(self.param_helper.k):
        j[ij, i] = sympy.Float(0)

    def fprime(X):
      res = np.zeros((self.param_helper.n, self.param_helper.n), dtype=float)
      res[:self.param_helper.k, :] = j.subs(list(zip(self.param_helper.syms, X)))
      return res

    return self.do_fsolve(p, fprime=fprime)

  def do_iter(self, pt, param, m):
    grad_data = self.compute_grad(pt, param, m)
    if grad_data is None: return None
    res = []
    for i in range(param.p):
      vnorm = np.linalg.norm(grad_data.vecs[i])
      vec = grad_data.vecs[i]
      res.append(cmisc.Attr(norm=vnorm, vec=vec, fix=param.pset[i]))
    return cmisc.Attr(ok=1, vecs=res, grad_data=grad_data, param=param)

  def get_best_tangent_space(self, p):
    m = self.froot(self.param_helper.syms, pad=0, pt=p)
    m = sympy.Matrix(m)

    tb = []
    for j, param in enumerate(self.param_helper.list_params(len(m))):
      u = self.do_iter(p, param, m)
      if u is None: continue
      tb.append(u)
    best = max(tb, key=lambda x: x.grad_data.det)
    return best

  def process(self, state):
    p = state.pos
    tangent_space = self.get_best_tangent_space(p)
    close_pts = self.kdtree.query_ball_point(p, self.delta * 2)
    print()
    print('\nProcessing point ', p, self.froot(p), len(close_pts))
    for vec in tangent_space.vecs:
      dv = make_norm(vec.vec) * self.delta
      print('ON VEc', vec, dv)
      for sgn in (1, -1):
        cnd = p + sgn * dv
        if close_pts:
          d1, _ = list_query_closest(cnd, close_pts)
          if d1 < self.delta * 0.3:
            print('SKIP1', d1, self.delta)
            continue

        npt = self.pt_to_sol(cnd, fix_p=cmisc.Attr(param=tangent_space.param, moveid=vec.fix))
        val = self.froot(npt)

        if np.any(np.array(val) >= self.eps): continue
        if close_pts:
          d3, _ = list_query_closest(npt, close_pts)
          if d3 < self.delta * 0.5:
            print('SKIP2', d3, self.delta)
            continue
        yield self.make_next_state(state, npt)

  def normalize(self, npt):
    return np.array(self.coords.normalize(npt))

  def make_next_state(self, prev, npt):
    npt = self.normalize(npt)
    if not self.coords.is_inside(npt): return None
    if not self.f.is_inside(npt): return None
    return cmisc.Attr(pos=npt)

  def go(self, startp, nsteps=-1):
    startp = self.normalize(startp)
    assert self.coords.is_inside(startp)
    print(startp)
    p = self.pt_to_sol(startp)

    q = SimpleQueue()
    self.kdtree.append(p)
    q.put(cmisc.Attr(pos=p))
    while not q.empty():
      if nsteps == 0: break
      nsteps -= 1
      state = q.get()
      for nxt in self.process(state):
        if nxt is None: continue
        self.kdtree.append(nxt.pos)
        q.put(nxt)
      print('End step ', nsteps)
    return np.array(self.kdtree.data)


class MainSpaceDesc:

  def __init__(self, coords):
    self.coords = coords
  @property
  def dim(self):
    return len(self.coords)

  def is_inside(self, X):
    return (all(c.is_inside(X[i]) for i, c in enumerate(self.coords)))

  def normalize(self, X):
    return list([c.normalize(X[i]) for i, c in enumerate(self.coords)])


class SegmentManifold:

  def __init__(self, range1d):
    self.range1d = range1d

  def normalize(self, X):
    return X

  def is_inside(self, X):
    return X in self.range1d


class CircleManifold:

  def __init__(self, range1d):
    self.range1d = range1d

  def normalize(self, X):
    return self.range1d.mod(X)


class LocusDesc:

  def __init__(self, neq=1, eps=1e-9):
    self.eps = eps
    self.neq = neq

  def froot_padded(self, X, n):
    res = self.froot(X)
    res.extend([0] * (n - len(res)))
    return res

  def froot(self, X, **kwargs):
    assert 0

  def is_inside(self, X):
    assert 0

  def get_param_helper(self, n):
    ph = ParametrizationHelper(n)
    return ph


class LocusEq(LocusDesc):

  def __init__(self, f, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.f = f

  def froot(self, X, pt=None):
    return [self.f(X, pt=pt)]

  def is_inside(self, X):
    return True


class LocusIneq(LocusDesc):

  def __init__(self, f, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.f = f

  def froot(self, X, pt=None):
    if pt is not None:
      if self.is_inside(pt): return []
      return [self.f(X, pt=pt)]

    if self.is_inside(X): return [0.]
    return [self.f(X)]

  def is_inside(self, X):
    v = self.f(X)
    return v <= 0


class LocusInter(LocusDesc):

  def __init__(self, sublocus_list, ineq_list=[], *args, **kwargs):
    neq = sum([x.neq for x in sublocus_list])
    self.ineq_list = ineq_list
    super().__init__(*args, neq=neq, **kwargs)
    self.sublocus_list = sublocus_list

  def froot(self, X, **kwargs):
    res = []
    for sl in self.sublocus_list:
      res.extend(sl.froot(X, **kwargs))
    return res

  def is_inside(self, X):
    return (all([sl.is_inside(X) for sl in self.sublocus_list]) and
          all([sl.is_inside(X) for sl in self.ineq_list]))

def test1(ctx):
  r1 = SegmentManifold(Range1D.All())
  r2 = MainSpaceDesc([r1,r1])

  def fcircle_r2(x, r=1):
    return np.dot(x, x) - r 
  circle = LocusEq(fcircle_r2)
  mw = ManifoldWalker(circle, r2, delta=ctx.delta)
  pts = mw.go((1,0), nsteps=ctx.nsteps)
  do_render(pts)

def test2(ctx):
  r1 = SegmentManifold(Range1D.All())
  r01 = SegmentManifold(Range1D(0,1))
  r2 = MainSpaceDesc([r1,r1, r01])

  def fcircle_r2(x, r=1):
    return np.dot(x[:2], x[:2]) - r
  circle = LocusEq(fcircle_r2)
  mw = ManifoldWalker(circle, r2, delta=ctx.delta)
  pts = mw.go((1,0, 0), nsteps=ctx.nsteps)
  do_render(pts)

def test3(ctx):
  r1 = SegmentManifold(Range1D.All())
  r3 = MainSpaceDesc([r1,r1, r1])

  def fcircle_r2(x, r=1):
    return np.dot(x[:2], x[:2]) - r
  def f_z0(x):
    return x[2]

  disk = LocusIneq(fcircle_r2)
  circle = LocusEq(fcircle_r2)
  l_z0 = LocusEq(f_z0)
  locus = LocusInter([disk, l_z0])
  mw = ManifoldWalker(locus, r3, delta=ctx.delta)
  pts = mw.go((1,0, 0), nsteps=ctx.nsteps)
  do_render(pts)

def test_torus(ctx):
  T = 1
  omega = 2*np.pi / T
  R = 3
  r1 = SegmentManifold(Range1D.All())
  tspace = SegmentManifold(Range1D(0, T))
  r4 = MainSpaceDesc([r1,r1, r1, tspace])
  space = r4

  def f_z0(x):
    return x[2]

  def center_pos(t):
    return (sympy.cos(t * omega) * R, sympy.sin(t*omega) * R, 0)

  def sphere_r3(x, r=1):
    dx = x[:3] - center_pos(x[3])
    return np.dot(dx, dx) - r

  def plan_r3(x):
    t = x[3]
    normal = (-sympy.sin(t * omega), sympy.cos(t*omega), 0)
    return np.dot(normal, x[:3])

  locus = LocusInter([LocusEq(sphere_r3), LocusEq(plan_r3)])
  mw = ManifoldWalker(locus, space, delta=ctx.delta)
  pts = mw.go((1,0, 0, 0), nsteps=ctx.nsteps)
  do_render(pts)

def test_mobius(ctx):
  T = 1
  omega = 2*np.pi / T
  R = 3
  r0 = 0.2
  r1 = SegmentManifold(Range1D.All())
  tspace = SegmentManifold(Range1D(0, T))
  r4 = MainSpaceDesc([r1,r1, r1, tspace])
  space = r4

  def center_pos(t):
    return (sympy.cos(t * omega) * R, sympy.sin(t*omega) * R, 0)

  def sphere_r3(x, pt=None, r=r0):
    t = x[3]
    dx = x[:3] - center_pos(x[3])
    return np.dot(dx, dx) - r

  def get_normal(t):
    px = center_pos(t)
    return (-px[1], px[0], 0)

  def plan1_r3(x, pt=None):
    t = x[3]
    normal = get_normal(t)
    return np.dot(normal, x[:3])

  def plan2_r3(x, pt=None):
    t = x[3]
    xv = get_normal(t)
    zv = (0, 0, 1)
    yv = np.cross(zv, xv)

    u = t * omega / 2
    uz = sympy.cos(u)
    uy = sympy.sin(u)
    normal = np.array(zv) * uz + np.array(yv) * uy
    target = np.dot(center_pos(t), normal)
    final= np.dot(normal, x[:3]) - target

    return final

  locus = LocusInter([LocusEq(plan1_r3), LocusEq(plan2_r3)], [LocusIneq(sphere_r3)])
  mw = ManifoldWalker(locus, space, delta=ctx.delta)
  p0 = np.array((R-r0,0, 0, 0))
  pts = mw.go(p0, nsteps=ctx.nsteps)
  if 0:
    p1 = np.array((R+r0,0, 0, 0))
    pts1 = mw.go(p1, nsteps=ctx.nsteps)
    print(pts.shape)
    print(pts1.shape)
    pts = np.vstack((pts, pts1))
  do_render(pts)
  return pts

def do_render(pts):
  import chdrft.utils.K as K
  pts=  np.array(pts)
  K.vispy_utils.render_for_meshes([cmisc.Attr(points=pts[:,:3])], cam=1)

def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
