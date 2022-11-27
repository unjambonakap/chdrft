#!/usr/bin/env python

from typing import Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.types import *
from pydantic import BaseModel, Field
from typing import Tuple
import xarray as xr
from typing import Callable, List
from scipy.spatial.transform import Rotation as R
from chdrft.display.base import TriangleActorBase
from chdrft.utils.math import MatHelper
import itertools
from enum import Enum
import functools
import sympy as sp
import jax.numpy as jnp
import jaxlib
import jax
import pygmo as pg
from chdrft.utils.fmt import Format
from chdrft.utils.path import FileFormatHelper

if TYPE_CHECKING:
  from chdrft.sim.rb.spatial_vectors import SpatialVector

global flags, cache
flags = None
cache = None

np_array_like = jnp.ndarray | np.ndarray


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def is_sympy(x):
  return isinstance(x, sp.Basic) or (isinstance(x, np.ndarray) and x.dtype.char == 'O')


def is_jnp(x):
  return isinstance(x, jnp.ndarray)


def make_identity(n, typ):
  if isinstance(typ, Vec3):
    typ = typ.vdata
  if is_jnp(typ):
    return jnp.identity(n)
  return np.identity(4, dtype=typ.dtype)


def make_array(x, type_hint=None):
  if isinstance(x, (np.ndarray, jnp.ndarray)): return x
  return g_oph.array(x, _oph_type_hint=type_hint)


def norm_pts(x):
  if len(x.shape) == 1:
    return x / x[3]
  return x / x[:, (3,)]


def as_pts3(x):
  if x.shape[-1] == 3:
    return x
  return norm_pts(x)[:, :3]


def as_pts4(x):
  if x.shape[-1] == 4:
    return x
  if len(x.shape) == 1: return Vec3.Vec(x).data
  return np.concatenate((x, np.zeros(x.shape[:-1] + (1,))), axis=len(x.shape) - 1)


@jax.custom_jvp
def jax_norm(x):
  return jnp.linalg.norm(x)


# f_jvp :: (a, T a) -> (b, T b)
def jax_norm_jvp(primals, tangents):
  x, = primals
  t, = tangents
  p = jax_norm(x)
  ot = jnp.dot(t, jnp.where(jnp.linalg.norm(x) == 0, jnp.ones_like(t), x / jnp.linalg.norm(x)))
  return p, ot


jax_norm.defjvp(jax_norm_jvp)


def raise_(x):
  raise x


ops_desc = dict(
    norm=(
        #lambda npx, x: npx.linalg.norm(x) if npx == np else jax_norm(x),
        lambda npx, x: npx.linalg.norm(x),
        lambda npx, x: sum(y**2 for y in x)**0.5,
    ),
    det=(
        #lambda npx, x: npx.linalg.norm(x) if npx == np else jax_norm(x),
        lambda npx, x: npx.linalg.det(x),
        lambda npx, x: raise_(NotImplemented()),
    ),

    # (1-cos x) / x^2
    cosc=(
        lambda npx, x: npx.where(
            x == 0,
            npx.sinc(x)**2 / (1 + npx.cos(x)),
            (1 - npx.cos(x)) / npx.where(x == 0, npx.ones_like(x), x)**2
        ), lambda npx, x: sp.sinc(x)**2 / (1 + sp.cos(x))
    ),
)


class SetterHelper:

  def __init__(self, obj, target):
    self.obj = obj
    self.target = target

  def __getitem__(self, slice):
    is_jnp = g_oph.is_jnp(self.obj) or g_oph.is_jnp(self.target)
    if is_jnp:
      if not g_oph.is_jnp(self.obj):
        self.obj = jnp.array(self.obj)
      return self.obj.at[slice].set(self.target)
    else:
      self.obj[slice] = self.target
      return self.obj


class OpHelper:

  def __init__(self):
    pass

  def is_jnp(self, v) -> bool:
    return isinstance(v, jnp.ndarray)

  def get_npx(self, *args, _oph_type_hint=None):
    if len(args) == 1 and cmisc.is_list(args[0]):
      args = args[0]
    for y in [_oph_type_hint] + list(args):
      if y is None: continue
      if is_sympy(y): return sp
      if self.is_jnp(y): return jnp
    return np

  def __getattr__(self, name):
    fs = ops_desc.get(name, None)
    return lambda *args, **kwargs: self.call(name, fs, *args, **kwargs)

  def call(self, name, fs, *args, _oph_type_hint=None, **kwargs):
    if fs is not None: fs = fs[is_sympy(args[0])]
    else: fs = lambda npx, *args, **kwargs: getattr(npx, name)(*args, **kwargs)

    return fs(self.get_npx(*args, _oph_type_hint=_oph_type_hint), *args, **kwargs)

  def inv(self, x):
    if not is_sympy(x): return self.get_npx(x).linalg.inv(x)
    m = sp.Matrix(x)
    return np.array(m.inv())

  def solve(self, x, y):
    if not is_sympy(x): return self.get_npx(x,y).linalg.solve(x, y)
    m = sp.Matrix(x)
    return np.array(list(m.solve(sp.Matrix(y))))

  def set(self, x, v):
    return SetterHelper(x, v)


g_oph = OpHelper()


class Vec3(cmisc.PatchedModel):
  data: np_array_like = None
  vec: bool = True

  def __repr__(self) -> str:
    return str(self)

  def __str__(self) -> str:
    return f'{"PV"[self.vec]}({self.vdata})'

  def __init__(self, *args, data=None, vec=True):
    super().__init__()
    if data is None:
      if len(args) == 1:
        data = args[0]
        if isinstance(data, Vec3):
          data = data.data
        elif isinstance(data, (np.ndarray, jnp.ndarray)):
          pass
        elif not cmisc.is_list(data):
          data = [data, data, data]
      elif len(args) == 0:
        data = [0, 0, 0]
      else:
        data = args

    if len(data) == 3:
      data = list(data) + [0 if vec else 1]

    self.data = make_array(data, type_hint=data[0])
    self.vec = vec
    assert self.data.shape == (4,)

  @property
  def as_vec(self) -> "Vec3":
    return Vec3(self.vdata, vec=True)

  @property
  def as_pt(self) -> "Vec3":
    return Vec3(self.vdata, vec=False)

  @staticmethod
  def Pt(x) -> "Vec3":
    return Vec3(x, vec=0)

  @staticmethod
  def Vec(x) -> "Vec3":
    return Vec3(x, vec=1)

  @staticmethod
  def Natif(v) -> np.ndarray:
    if isinstance(v, Vec3): return v.data
    return v

  @staticmethod
  def Make(v, **kwargs):
    if v is None: return v
    if isinstance(v, Vec3): return v
    return Vec3(v, **kwargs)

  @staticmethod
  def LinearComb(pt_w: "list[tuple[Vec3,float]]") -> "Vec3":
    res = np.zeros(3)
    for pt, w in pt_w:
      assert not pt.vec
      res = res + pt.vdata * w
    return Vec3(res, vec=0)

  @staticmethod
  def Zero(**kwargs):
    return Vec3(0, 0, 0, **kwargs)

  @staticmethod
  def ZeroPt():
    return Vec3(0, 0, 0, vec=0)

  @staticmethod
  def Z():
    return Vec3(0, 0, 1)

  @staticmethod
  def X():
    return Vec3(1, 0, 0)

  @staticmethod
  def Y():
    return Vec3(0, 1, 0)

  @staticmethod
  def Rand(**kwargs):
    return Vec3(np.random.rand(3), **kwargs)

  @property
  def skew_matrix(self) -> np.ndarray:
    return make_array(
        [
            [0, -self.data[2], self.data[1]],
            [self.data[2], 0, -self.data[0]],
            [-self.data[1], self.data[0], 0],
        ],
        type_hint=self.data
    )

  @property
  def skew_transform(self) -> "Transform":
    return Transform.From(rot=self.skew_matrix)

  def __neg__(a):
    return a.like(-a.vdata)

  def __sadd__(a, b):
    assert a.vec
    assert b.vec
    self.make_canon()
    self.data += b.canon.data
    return self

  def __add__(a, b):
    assert a.vec
    return a.like(a.canon.data + b.canon.data)

  def like(self, data):
    return Vec3(data, vec=self.vec)

  def cross(self, peer: "Vec3") -> "Vec3":
    return Vec3(g_oph.cross(self.vdata, peer.vdata))

  def __mul__(a, b):
    assert a.vec
    return a.like(a.data * Vec3.Natif(b))

  def __sub__(a, b):
    return Vec3(a.canon.data - b.canon.data, vec=True)

  def __truediv__(a, b):
    assert a.vec
    return a.like(a.data / Vec3.Natif(b))

  @property
  def canon(self) -> "Vec3":
    if self.vec: return self
    return Vec3(self.data / self.data[3])

  def make_canon(self):
    if self.vec: return
    self.data = self.data / data[3]

  def dot(self, vec: "Vec3 | np.ndarray") -> float:
    vec = Vec3.Make(vec)
    return g_oph.dot(self.vdata, vec.vdata)

  def proj(self, vec: "Vec3 | np.ndarray") -> "Vec3":
    vec = Vec3.Make(vec)
    return Vec3.Make(vec * g_oph.dot(self.data, vec.data))

  def orth_proj(self, vec: "Vec3 | np.ndarray") -> "Vec3":
    return self - self.proj(vec)

  @property
  def norm(self) -> float:
    return g_oph.norm(self.vdata)

  def make_norm(self) -> "Vec3":
    return self.norm_and_uvec[1]

  @property
  def uvec(self) -> "Vec3":
    return self.make_norm()

  @property
  def vdata(self) -> np.ndarray:
    return self.canon.data[:3]

  @property
  def norm_and_uvec(self) -> "Tuple[float, Vec3]":
    n = self.norm
    bad = n < 1e-5
    nv = g_oph.where(bad, self.data, (self / g_oph.where(bad, 1, n)).data)
    return n, Vec3(nv)

  def exp_rot_u_3(self, rot: float) -> np.ndarray:
    w1 = self.skew_matrix
    w2 = w1 @ w1
    res = np.identity(3) + g_oph.sin(rot) * w1 + (1 - g_oph.cos(rot)) * w2
    return res

  def exp_rot_u(self, rot: float, around: "Vec3" = None) -> "Transform":
    if around is None: around = Vec3.ZeroPt()
    assert not around.vec
    res = self.exp_rot_u_3(rot)
    tmp = around.vdata - res @ around.vdata
    ans = Transform.From(rot=res, pos=tmp)
    return ans

  def exp_rot_3(self) -> np.ndarray:
    rot, v = self.norm_and_uvec
    return v.exp_rot_u_3(rot)

  def exp_rot(self, around=None) -> "Transform":
    if 0:
      if around is None: around = Vec3.ZeroPt()
      w1 = self.skew_matrix
      w2 = w1 @ w1
      rot = self.norm

      res = np.identity(3) + g_oph.sinc(rot) * w1 + w2 * g_oph.cosc(rot)
      tmp = around.vdata - res @ around.vdata
      ans = Transform.From(rot=res, pos=tmp)
      return ans

    rot, v = self.norm_and_uvec
    return v.exp_rot_u(rot, around)


def get_axis_angle(r: np.ndarray) -> "Vec3":
  npx = g_oph.get_npx(r)
  w = r - np.identity(3)
  ang = npx.arccos(npx.clip((npx.trace(r) - 1) / 2, -1, 1))  # in [0, pi]

  sinv = npx.sin(ang)
  res = npx.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])

  res = res / npx.maximum(1e-9, npx.linalg.norm(res)) * ang

  is_zero = ang < 1e-8
  is_pi = ang > np.pi - 1e-8

  res = npx.where(is_zero, np.zeros(3), res)
  res = npx.where(is_pi, ((npx.diag(r) + 1) / 2)**0.5, res)

  p1 = Vec3.X().vdata
  p2 = Vec3.Z().vdata

  v1 = w @ p1
  v2 = w @ p2
  path1 = v1.dot(v1) > v2.dot(v2)

  other = npx.where(path1, v1, v2)
  vx = npx.cross(res, other)
  tsf = r @ other
  res = res * npx.where(tsf.dot(vx) < 0, -1, 1)
  res = npx.where(npx.max(npx.abs(w)) < 1e-8, np.zeros(3), res)
  return Vec3(res)


class Transform(cmisc.PatchedModel):
  data: np.ndarray | jnp.ndarray

  def __init__(self, data=None):
    if data is None:
      data = np.identity(4)
    assert data.shape == (4, 4)
    super().__init__(data=data)

  @property
  def T(self) -> "Transform":
    return Transform(np.array(self.data.T))

  @property
  def inv(self) -> "Transform":
    #return Transform(data=g_oph.inv(self.data))
    res = self.rot.T
    pos = -res @ self.pos
    return Transform.From(rot=res, pos=pos)

  @property
  def invT(self) -> "Transform":
    return self.inv.T

  def get_axis_angle(self) -> Vec3:
    return get_axis_angle(self.rot)

  @property
  def x(self) -> Vec3:
    return Vec3(self.data[:, 0])

  @property
  def y(self) -> Vec3:
    return Vec3(self.data[:, 1])

  @property
  def z(self) -> Vec3:
    return Vec3(self.data[:, 2])

  @property
  def pos_v(self) -> Vec3:
    return Vec3(self.pos, vec=0)

  @property
  def pos(self) -> np.ndarray:
    return self.data[:3, 3]

  @pos.setter
  def pos(self, v):
    self.data = g_oph.set(self.data, v[:3])[:3, 3]

  @pos_v.setter
  def pos_v(self, v):
    self.pos = v.vdata

  @property
  def rot(self) -> np.ndarray:
    return self.data[:3, :3]

  @rot.setter
  def rot(self, v):
    if isinstance(v, jnp.ndarray): self.data = jnp.array(self.data)
    if isinstance(self.data, jnp.ndarray): self.data = self.data.at[:3, :3].set(v)
    else: self.data[:3, :3] = v

  @property
  def tsf_rot(self) -> "Transform":
    return Transform.From(rot=self.rot)

  @property
  def tsf_xlt(self) -> "Transform":
    return Transform.From(pos=self.pos)

  @property
  def rot_R(self) -> R:
    return R.from_matrix(self.rot)

  def scale(self, v):
    res = self.data[:3, :3] * np.array(v)
    if isinstance(v, jnp.ndarray): self.data = jnp.array(self.data)
    if isinstance(self.data, jnp.ndarray): self.data = self.data.at[:3, :3].set(res)
    else: self.data[:3, :3] = res

  def apply(self, v: Vec3) -> Vec3:
    assert isinstance(v, Vec3)
    return v.like(self.map(v.data))

  def apply_vec(self, v: Vec3) -> Vec3:
    return Vec3(self.rot @ v.data)

  def map(self, v):
    if len(v) == 0: return np.zeros((0, 4))
    return (self.data @ v.T).T

  def imap(self, v):
    return self.inv.map(v)

  def __mul__(self, peer):
    return Transform(data=self.data @ peer.data)

  def to_sv(self) -> "SpatialVector":
    from chdrft.sim.spatial_vectors import SpatialVector
    return SpatialVector(dual=0, v=self.pos_v.as_vec, w=self.get_axis_angle())

  def __matmul__(self, peer):
    from chdrft.sim.spatial_vectors import SpatialVector
    if isinstance(peer, Vec3):
      return self.apply(peer)
    if isinstance(peer, SpatialVector):
      return peer.mat_tsf(self)
    if isinstance(peer, np.ndarray):
      return self.map(peer)
    if isinstance(peer, InertialTensor):
      return InertialTensor(data=self.rot @ peer.data)
    return Transform(data=self.data @ peer.data)

  @staticmethod
  def From(pos=None, rot=None, scale=None, data=None):

    if data is not None:
      return Transform(data=data)
    tsf = Transform()
    pos = Vec3.Make(pos)
    if scale is not None:
      tsf.scale(np.array(scale))
      scale = None
    elif pos is not None:
      tsf = Transform(data=make_identity(4, pos))
      tsf.pos_v = pos
      pos = None

    elif rot is not None:
      if isinstance(rot, R):
        rot = rot.as_matrix()
      tsf = Transform(data=make_identity(4, rot))
      tsf.rot = rot
      rot = None
    else:
      return tsf

    res = tsf * Transform.From(pos=pos, rot=rot, scale=scale)
    return res

  def clone(self) -> "Transform":
    return Transform.From(data=make_array(self.data))

def make_rot_ab(all_good=False, **kwargs) -> np.ndarray:
  #arg: x=a -> x = Ra
  return make_rot(all_good, T=True, **kwargs)



def make_rot(all_good=False, T=False, **kwargs) -> np.ndarray:
  #arg: x=a -> a = Rx
  mp = dict(x=0, y=1, z=2)
  vx = None
  tb: list[Tuple[int, Vec3]] = []

  def make_orth(v):
    v = Vec3(v)
    for _, e in tb:
      v = v.orth_proj(e)
    if not all_good and v.norm < 1e-5: return None
    v = v.make_norm()
    return v

  used = set()
  for k, v in kwargs.items():
    used.add(mp[k])
    tb.append((mp[k], make_orth(v)))

  nx = 3
  for i in range(nx):
    if i in used: continue
    while True:
      a = make_orth(Vec3.Rand(vec=1))
      if a is not None:
        tb.append((i, a))
        break

  mat = np.zeros((3, 3))
  for p, v in tb:
      mat[:, p] = v.data[:3]


  if T: mat = mat.T
  if np.linalg.det(mat) < 0:
    for i in range(nx):
      if i not in used:
        mat[:, i] *= -1
        break
  return mat


class AngularSpeed(cmisc.PatchedModel):
  dir: Vec3
  w: float

  @property
  def vec(self) -> Vec3:
    return self.dir * self.w


class AABB:

  def __str__(self) -> str:
    return f'AABB({self.low=}, {self.high=})'

  @staticmethod
  def FromPoints(pts):
    if len(pts) == 0: return AABB(empty=True)
    pts = as_pts3(pts)
    min = np.min(pts, axis=0)
    max = np.max(pts, axis=0)
    return AABB(min, max - min)

  def __init__(self, p=None, v=None, empty=False):
    self.empty = empty
    if self.empty: return
    self.p = np.array(p)
    self.n = len(p)
    self.v = np.array(v)
    assert len(self.p.shape) == 1
    assert len(self.v.shape) == 1

  def corner(self, m) -> Vec3:
    v = np.array(self.v)
    for i in range(self.n):
      v[i] *= (m >> i & 1)
    return Vec3.Pt(self.p + v)

  @property
  def points(self) -> np.ndarray:
    res = []
    if self.empty: return res
    for i in range(2**self.n):
      res.append(self.corner(i).data)
    return np.array(res)

  @property
  def surface_mesh(self):
    coords = [self.corner(i).vdata for i in range(8)]
    res = TriangleActorBase()
    ids = res.add_points(coords)
    res.push_quad(ids[:4], pow2=1)
    res.push_quad(ids[4:8], pow2=1)
    adjs = [0, 1, 3, 2, 0]
    for i in range(4):
      a = adjs[i]
      b = adjs[i + 1]

      res.push_quad(ids[[a, b, a + 4, b + 4]], pow2=1)
    res.build()
    return res

  @property
  def low(self):
    return self.p

  @property
  def center(self):
    return (self.low + self.high) / 2

  @property
  def high(self):
    return self.p + self.v


def loop(x):
  ix = iter(x)
  a = next(ix)
  yield a
  yield from ix
  yield a


class SurfaceMeshParams(BaseModel):
  points: int = None


class Particles(cmisc.PatchedModel):
  weights: np.ndarray = None
  p: np.ndarray = None
  v: np.ndarray = None
  n: int
  id_obj: np.ndarray = None

  @staticmethod
  def Default() -> "Particles":
    d0 = np.zeros((0, 4))
    return Particles(weights=d0, p=np.zeros((0, 4)), v=d0, id_obj=d0, n=0)

  def reduce(a: "Particles", b: "Particles") -> "Particles":
    if not a.n: return b.copy()
    if not b.n: return a.copy()
    return Particles(
        n=a.n + b.n,
        weights=np.concatenate((a.weights, b.weights)),
        p=np.vstack((a.p, b.p)),
        v=np.vstack((a.v, b.v)),
        id_obj=np.concatenate((a.id_obj, b.id_obj)),
    )

  @property
  def empty(self) -> bool:
    return self.n == 0

  @property
  def com(self) -> np.ndarray:
    return np.average(as_pts3(self.p), weights=self.weights, axis=0)

  def linear_momentum(self) -> np.ndarray:
    if self.empty: return np.zeros(3)
    return np.sum(self.v * self.weights.reshape((-1, 1)), axis=0)

  def angular_momentum(self) -> np.ndarray:
    if self.empty: return np.zeros(3)
    return np.sum(np.cross(as_pts3(self.p), self.v[:, :3]) * self.weights.reshape((-1, 1)), axis=0)

  def plot(self, fx=1, by_col=False):
    pts = as_pts3(self.p)
    lines = np.stack((pts, pts + self.v[:, :3] * fx), axis=2)
    lines = np.transpose(lines, (0, 2, 1))
    points_color = []
    from chdrft.display.service import g_plot_service, grid
    data = A(points=pts, lines=lines, conf=A(mode='3D'))
    if by_col:
      from chdrft.utils.colors import ColorMapper
      vals = set(self.id_obj)
      cm = ColorMapper(vals)
      data.points_color = [cm.get(x) for x in self.id_obj]
      data.lines = [A(polyline=x, color=cm.get(id)) for id, x in zip(self.id_obj, data.lines)]
    g_plot_service.plot(data)


def make_pts(x):
  return np.hstack((x, np.ones((len(x), 1))))


def rejection_sampling(n, bounds, checker, grp_size=1000):
  res = []
  while len(res) < n:
    cnds = np.random.uniform(low=bounds[0], high=bounds[1], size=(n, len(bounds[0])))
    cnds = cnds[checker(cnds)]
    res.extend(cnds)
  return make_pts(np.array(res[:n]))


def importance_sampling(n, bounds, weight_func, map=None):
  cnds = np.random.uniform(low=bounds[0], high=bounds[1], size=(n, len(bounds[0])))
  weights = weight_func(cnds)
  weights = weights / sum(weights)
  if map is not None: cnds = map(cnds)
  return make_pts(cnds), weights


class MeshDesc(cmisc.PatchedModel):

  @property
  def bounds(self) -> AABB:
    return AABB(empty=True)

  def rejection_sampling(self, n):
    return self.rejection_sampling0(n)

  def importance_sampling(self, n, wtarget=1):
    pts, w = self.importance_sampling0(n)
    w *= wtarget
    return pts, w

  def rejection_sampling0(self, n):
    return np.zeros((0, 4))

  def importance_sampling0(self, n):
    return np.array([]), np.zeros((0, 4))

  def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
    res = TriangleActorBase()
    res.build()
    return res


class TransformedMeshDesc(MeshDesc):
  mesh: MeshDesc
  transform: Transform

  @property
  def bounds(self) -> AABB:
    return AABB.FromPoints(self.transform.map(self.mesh.bounds.points))

  def rejection_sampling0(self, n):
    return self.transform.map(self.mesh.rejection_sampling0(n))

  def importance_sampling0(self, n):
    pts, w = self.mesh.importance_sampling0(n)
    return self.transform.map(pts), w

  def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
    res = TriangleActorBase()
    res.push_peer(self.mesh.surface_mesh(params))
    res.points = [self.transform.apply(Vec3.Pt(x)).vdata for x in res.points]
    res.build()
    return res


class ComposedMeshDesc(MeshDesc):
  meshes: list[MeshDesc]
  weights: list[float]
  norm_factor: float = -1

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.norm_factor = sum(self.weights)

  @property
  def bounds(self) -> AABB:
    return AABB.FromPoints(np.array(list(itertools.chain(*[x.bounds.points for x in self.meshes]))))

  @property
  def entries(self):
    return [A(w=w / self.norm_factor, m=m) for m, w in zip(self.meshes, self.weights)]

  def rejection_sampling0(self, n):
    if not self.meshes:
      return super().rejection_sampling0(n)
    return np.vstack([x.m.rejection_sampling0(int(n * x.w)) for x in self.entries])

  def importance_sampling0(self, n):
    res = [
        np.array(list(itertools.chain(*x)))
        for x in zip(*[y.m.importance_sampling(int(n * y.w), wtarget=y.w) for y in self.entries])
    ]
    if not res: return super().importance_sampling0(n)
    return res

  def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
    res = TriangleActorBase()
    for x in self.meshes:
      res.push_peer(x.surface_mesh(params))
    res.build()
    return res


class PointMesh(MeshDesc):

  @property
  def bounds(self) -> AABB:
    return AABB([-0, -0, -0], [0, 0, 0])

  def rejection_sampling0(self, n):
    return np.array([[0, 0, 0, 1]])

  def importance_sampling0(self, n):
    return np.array([[0, 0, 0, 1]]), np.array([1])

  def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
    return self.bounds.surface_mesh


class UnitConeDesc(MeshDesc):
  # tip up
  @property
  def bounds(self) -> AABB:
    return AABB([-1, -1, -1 / 4], [2, 2, 1])

  def rejection_sampling0(self, n):
    return rejection_sampling(
        n, (self.bounds.low, self.bounds.high), lambda tb:
        (np.linalg.norm(tb[:, :2], ord=2, axis=1) < 3 / 4 - tb[:, 2])
    )

  def importance_sampling0(self, n):
    return importance_sampling(
        n, (self.bounds.low, self.bounds.high), weight_func=lambda tb: np.full(tb.shape[0], 1)
    )

  def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
    res = TriangleActorBase()
    p0 = np.array([0, 0, -1 / 3])
    p1 = np.array([0, 0, 2 / 3])
    id0, id1 = res.add_points([p0, p1])
    npt = 10
    alphal = np.linspace(0, 2 * np.pi, npt)
    pl = res.add_points(np.array([np.cos(alphal), np.sin(alphal), alphal * 0 - 1 / 3]).T)
    for a, b in itertools.pairwise(cmisc.loop(range(npt))):
      res.push_triangle(np.array([id0, pl[a], pl[b]]))
      res.push_triangle(np.array([pl[b], pl[a], id1]))
    res.build()
    return res


class UnitCylinderDesc(MeshDesc):

  @property
  def bounds(self) -> AABB:
    return AABB([-1, -1, -0.5], [2, 2, 1])

  def rejection_sampling0(self, n):
    return rejection_sampling(
        n, (self.bounds.low, self.bounds.high), lambda tb: (tb[:, 0]**2 + tb[:, 1]**2 <= 1)
    )

  def importance_sampling0(self, n):
    return importance_sampling(
        n, (self.bounds.low, self.bounds.high), weight_func=lambda tb: np.full(tb.shape[0], 1)
    )

  @staticmethod
  def cylinder2xyz_unpacked(r, alpha, z):
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)
    z = 0 * alpha + z
    return np.array([x, y, z]).T

  def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
    res = TriangleActorBase()
    zl = np.linspace(-0.5, 0.5, 20)
    alphal = np.linspace(0, 2 * np.pi, 10)
    for za, zb in itertools.pairwise(zl):
      la = UnitCylinderDesc.cylinder2xyz_unpacked(1, alphal, za)
      lb = UnitCylinderDesc.cylinder2xyz_unpacked(1, alphal, zb)
      ids_a = res.add_points(la)
      ids_b = res.add_points(lb)
      assert len(ids_a) == len(alphal)
      for i in range(len(alphal) - 1):
        res.push_quad(np.array([ids_a[i], ids_a[i + 1], ids_b[i], ids_b[i + 1]]), pow2=1)
    res.build()
    return res


class CubeDesc(MeshDesc):

  @property
  def bounds(self) -> AABB:
    return AABB([-0.5, -0.5, -0.5], [1, 1, 1])

  def rejection_sampling0(self, n):
    return rejection_sampling(
        n, (self.bounds.low, self.bounds.high), lambda tb: np.full(tb.shape[0], True)
    )

  def importance_sampling0(self, n):
    return importance_sampling(
        n, (self.bounds.low, self.bounds.high), weight_func=lambda tb: np.full(tb.shape[0], 1)
    )

  def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
    return self.bounds.surface_mesh


class SphereDesc(MeshDesc):

  @property
  def bounds(self) -> AABB:
    return AABB([-1, -1, -1], [2, 2, 2])

  @staticmethod
  def spherical2xyz_unpacked(r, alpha, phi):
    x = r * np.cos(alpha) * np.cos(phi)
    y = r * np.sin(alpha) * np.cos(phi)
    z = r * np.sin(phi) * np.ones_like(alpha)
    return np.array([x, y, z]).T

  @staticmethod
  def spherical2xyz(sph):
    r = sph[:, 0]
    alpha = sph[:, 1]
    phi = sph[:, 2]
    return SphereDesc.spherical2xyz_unpacked(r, alpha, phi)

  @staticmethod
  def Weight(sph):
    r, alpha, phi = sph.T
    det_g_sqrt = r * r * np.abs(np.sin(alpha))
    det_g_sqrt = r * r * np.abs(np.cos(phi))
    return det_g_sqrt

  @property
  def bounds_spherical(self):
    return ((0, 0, -np.pi / 2), (1, 2 * np.pi, np.pi / 2))

  def rejection_sampling0(self, n):
    return rejection_sampling(
        n, (self.bounds.low, self.bounds.high),
        lambda pts: np.einsum('ij,ij->i', pts[:, :3], pts[:, :3]) <= 1
    )

  def importance_sampling0(self, n):
    return importance_sampling(
        n, self.bounds_spherical, weight_func=SphereDesc.Weight, map=SphereDesc.spherical2xyz
    )

  def surface_mesh(self, params: SurfaceMeshParams = None) -> TriangleActorBase:
    res = TriangleActorBase()
    alphal = np.array(list(loop(np.linspace(0, np.pi * 2, 20))))
    for phi_a, phi_b in itertools.pairwise(np.linspace(-np.pi / 2, np.pi / 2, 10)):
      la = SphereDesc.spherical2xyz_unpacked(1, alphal, phi_a)
      lb = SphereDesc.spherical2xyz_unpacked(1, alphal, phi_b)
      ids_a = res.add_points(la)
      ids_b = res.add_points(lb)
      for i in range(len(alphal) - 1):
        res.push_quad(np.array([ids_a[i], ids_a[i + 1], ids_b[i], ids_b[i + 1]]), pow2=1)
    res.build()
    return res


class InertialTensor(cmisc.PatchedModel):

  data: np.ndarray | jnp.ndarray = Field(default_factory=lambda: np.zeros((3, 3)))

  def shift_inertial_tensor(self, p: Vec3, mass) -> "InertialTensor":
    return self + InertialTensor.FromPointMass(p, mass)

  def get_world_tensor(self, wl: Transform) -> "InertialTensor":
    m = wl.rot @ self.data @ wl.rot.T
    return InertialTensor(data=m)

  def __add__(self, v) -> "InertialTensor":
    return InertialTensor(data=self.data + v.data)

  def around(self, v) -> float:
    return v @ self.data @ v

  def vel2mom(self, v: Vec3) -> Vec3:
    assert v.vec
    return Vec3(self.data @ v.vdata)

  def mom2vel(self, v: Vec3) -> Vec3:
    assert v.vec
    return Vec3(g_oph.solve(self.data, v.vdata))

  @staticmethod
  def DFromPointMass(p: Vec3, dp: Vec3, mass: float) -> "InertialTensor":
    grad = InertialTensor.GradFromPointMass(p, mass)
    return InertialTensor(data=grad @ dp.vdata)

  @staticmethod
  def GradFromPointMass(p: Vec3, mass: float) -> np.ndarray:
    px, py, pz = p.vdata
    mx = [
        [[0, 2 * py, 2 * pz], [-py, -px, 0], [-pz, 0, -px]],
        [[-py, -px, 0], [2 * px, 0, 2 * pz], [0, -pz, -py]],
        [[-pz, 0, -px], [0, -pz, -py], [2 * px, 2 * py, 0]],
    ]

    return np.array(mx) * mass

  @staticmethod
  def FromPointMass(p: Vec3, mass: float) -> "InertialTensor":
    assert p.vec
    px, py, pz = p.vdata
    mx = [
        [py * py + pz * pz, -px * py, -px * pz],
        [-px * py, px * px + pz * pz, -py * pz],
        [-px * pz, -py * pz, py * py + px * px],
    ]

    return InertialTensor(data=make_array(mx, type_hint=p.data) * mass)

  @staticmethod
  def FromPoints(pts, weights=None, com=False, mass=1):
    r = pts
    if weights is None: weights = np.full(len(pts), 1)
    sweight = sum(weights)
    weights = weights * (mass / sweight)
    if com:
      r = pts - np.average(pts, weights=weights, axis=0)
    r2 = np.sum(np.linalg.norm(r, axis=1)**2 * weights)
    e3r2 = r2 * np.identity(3)
    res = e3r2 - np.einsum('i,ij,ik->jk', weights, r, r)
    return res


class SolidSpec(cmisc.PatchedModel):
  mass: float = 0
  com: Vec3 = Field(default_factory=Vec3.ZeroPt)
  inertial_tensor: InertialTensor = Field(default_factory=InertialTensor)
  mesh: MeshDesc = Field(default_factory=MeshDesc)


  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert not self.com.vec

  @staticmethod
  def Point(mass):
    return SolidSpec(
        mass=mass, inertial_tensor=InertialTensor(data=np.zeros((3, 3))), desc=PointMesh()
    )

  @staticmethod
  def Cone(mass, r, h):
    ix = iy = 3 / 80 * h**2 + 3 / 20 * r**2
    iz = 3 / 10 * r**2
    mesh = TransformedMeshDesc(mesh=UnitConeDesc(), transform=Transform.From(scale=[r, r, h]))
    return SolidSpec(
        mass=mass,
        inertial_tensor=InertialTensor(data=np.identity(3) * mass * np.array([ix, iy, iz])),
        mesh=mesh
    )

  @staticmethod
  def Sphere(mass, r):
    v = mass * 2 / 5 * r**2
    mesh = TransformedMeshDesc(mesh=SphereDesc(), transform=Transform.From(scale=r))
    return SolidSpec(mass=mass, inertial_tensor=InertialTensor(data=np.identity(3) * v), mesh=mesh)

  @staticmethod
  def Cylinder(mass, r, h):
    iz = 1 / 2 * mass * r**2
    ix = iz / 2 + mass * h**2 / 12
    return SolidSpec(
        mass=mass,
        inertial_tensor=InertialTensor(data=np.diag(np.array([ix, ix, iz]))),
        mesh=TransformedMeshDesc(
            mesh=UnitCylinderDesc(), transform=Transform.From(scale=[r, r, h])
        ),
    )

  @staticmethod
  def Box(mass, x, y, z):
    c = mass / 12
    mesh = TransformedMeshDesc(mesh=CubeDesc(), transform=Transform.From(scale=[x, y, z]))
    return SolidSpec(
        mass=mass,
        inertial_tensor=InertialTensor(
            data=np.diag(np.array([y * y + z * z, x * x + z * z, x * x + y * y])) * c
        ),
        mesh=mesh
    )


class NumpyPackerEntry(cmisc.PatchedModel):
  shape: np.ndarray
  pos: int
  name: str

  @property
  def size(self) -> int:
    return np.multiply.reduce(self.shape)


class NumpyPacker(cmisc.PatchedModel):

  tb: list[NumpyPackerEntry] = Field(default_factory=list)
  pos: int = 0

  def add_dict(self, **kwargs):
    for k, v in kwargs.items():
      self.add(k, v)

  @property
  def default(self) -> np.ndarray:
    return np.zeros(self.pos)

  def add(self, name, shape):
    if isinstance(shape, int): shape = (shape,)
    shape = np.array(shape)
    x = NumpyPackerEntry(name=name, pos=self.pos, shape=shape)
    self.tb.append(x)
    self.pos += x.size

  def pack(self, data: dict | cmisc.PatchedModel | np.ndarray) -> np.ndarray:
    if not isinstance(data, (dict, cmisc.PatchedModel)):
      return data
    if isinstance(data, cmisc.PatchedModel):
      data = data.dict()
    res = []
    for x in self.tb:
      res.append(g_oph.array(data[x.name]).reshape(x.size))
    return g_oph.hstack(res)

  def unpack(self, v: dict | cmisc.PatchedModel | np.ndarray) -> dict:
    if isinstance(v, (dict, cmisc.PatchedModel)): return v
    res = A()
    for x in self.tb:
      res[x.name] = v[x.pos:x.pos + x.size].reshape(x.shape)
    return res
