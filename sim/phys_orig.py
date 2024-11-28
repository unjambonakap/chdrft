#!/usr/bin/env python

from typing import Tuple, Optional
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
import numpy as np
from chdrft.utils.opa_types import *
from pydantic.v1 import BaseModel, Field
from typing import Tuple
from scipy.spatial.transform import Rotation as R
from chdrft.display.base import TriangleActorBase
import itertools
from enum import Enum
import functools
import sympy as sp
import jax.numpy as jnp
import jax
import pygmo as pg
from chdrft.utils.fmt import Format
from chdrft.utils.path import FileFormatHelper

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class RigidBodyLinkType(Enum):
  RIGID = 1
  MOVE = 2


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


ops_desc = dict(norm=(lambda npx, x: npx.linalg.norm(x), lambda npx, x: sum(y**2 for y in x)**0.5),)


class OpHelper:

  def __init__(self):
    pass

  def get_npx(self, *args, _oph_type_hint=None):
    if len(args) == 1 and cmisc.is_list(args[0]):
      args = args[0]
    for y in [_oph_type_hint] + list(args):
      if y is None: continue
      if is_sympy(y): return sp
      if isinstance(y, jnp.ndarray): return jnp
    return np

  def __getattr__(self, name):
    fs = ops_desc.get(name, None)
    return lambda *args, **kwargs: self.call(name, fs, *args, **kwargs)

  def call(self, name, fs, *args, _oph_type_hint=None):
    if fs is not None: fs = fs[is_sympy(args[0])]
    else: fs = lambda npx, *args: getattr(npx, name)(*args)

    return fs(self.get_npx(*args, _oph_type_hint=_oph_type_hint), *args)

  def inv(self, x):
    if not is_sympy(x): return self.get_npx(x).linalg.inv(x)
    m = sp.Matrix(x)
    return np.array(m.inv())

  def solve(self, x, y):
    if not is_sympy(x): return self.get_npx(x).linalg.solve(x, y)
    m = sp.Matrix(x)
    return np.array(list(m.solve(sp.Matrix(y))))


g_oph = OpHelper()


class Vec3(cmisc.PatchedModel):
  data: np.ndarray = None
  vec: bool = True

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

  class Config:
    arbitrary_types_allowed = True

  @staticmethod
  def Pt(x) -> "Vec3":
    return Vec3(x, vec=0)

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
    return self.like(-a.data)

  def __sadd__(a, b):
    self.data += b.data
    return self

  def __add__(a, b):
    return a.like(a.data + b.data)

  def like(self, data):
    return Vec3(data, vec=self.vec)

  def __mul__(a, b):
    return a.like(a.data * Vec3.Natif(b))

  def __sub__(a, b):
    return Vec3(a.canon.data - b.canon.data, vec=True)

  def __truediv__(a, b):
    return a.like(a.data / Vec3.Natif(b))

  @property
  def canon(self) -> "Vec3":
    if self.vec: return self
    return Vec3(self.data / self.data[3])

  def dot(self, vec: "Vec3 | np.ndarray") -> "Vec3":
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
    nv = g_oph.where(bad, Vec3.X().data, (self / g_oph.where(bad, 1, n)).data)
    return n, Vec3(nv)

  def exp_rot_u(self, rot: float, around=None) -> "Transform":
    if around is None: around = Vec3.Zero(vec=0)
    w1 = self.skew_matrix
    w2 = w1 @ w1
    res = np.identity(3) + g_oph.sin(rot) * w1 + (1 - g_oph.cos(rot)) * w2
    tmp = around.vdata - res @ around.vdata
    ans = Transform.From(rot=res, pos=tmp)
    return ans

  def exp_rot(self, around=None) -> "Transform":
    if 0:
      if around is None: around = Vec3.Zero(vec=0)
      w1 = self.skew_matrix
      w2 = w1 @ w1
      rot = self.norm

      res = np.identity(3) + g_oph.sinc(rot) * w1 + w2 * g_oph.cosc(rot)
      tmp = around.vdata - res @ around.vdata
      ans = Transform.From(rot=res, pos=tmp)
      return ans

    rot, v = self.norm_and_uvec
    return v.exp_rot_u(rot, around)


class Transform(cmisc.PatchedModel):
  data: np.ndarray | jnp.ndarray

  def __init__(self, data=None):
    if data is None:
      data = np.identity(4)
    assert data.shape == (4, 4)
    super().__init__(data=data)

  @property
  def inv(self) -> "Transform":
    return Transform(data=g_oph.inv(self.data))

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
    return Vec3(self.pos)

  @property
  def pos(self) -> np.ndarray:
    return self.data[:3, 3]

  @pos.setter
  def pos(self, v):
    if isinstance(v, jnp.ndarray): self.data = jnp.array(self.data)
    if isinstance(self.data, jnp.ndarray): self.data = self.data.at[:3, 3].set(v[:3])
    else: self.data[:3, 3] = v[:3]

  @pos_v.setter
  def pos_v(self, v):
    self.pos = v.data

  @property
  def rot(self) -> np.ndarray:
    return self.data[:3, :3]

  @property
  def tsf_rot(self):
    return Transform.From(rot=self.rot)

  @property
  def rot_R(self) -> R:
    return R.from_matrix(self.rot)

  def scale(self, v):
    res = self.data[:3, :3] * np.array(v)
    if isinstance(v, jnp.ndarray): self.data = jnp.array(self.data)
    if isinstance(self.data, jnp.ndarray): self.data = self.data.at[:3, :3].set(res)
    else: self.data[:3, :3] = res

  @rot.setter
  def rot(self, v):
    if isinstance(v, jnp.ndarray): self.data = jnp.array(self.data)
    if isinstance(self.data, jnp.ndarray): self.data = self.data.at[:3, :3].set(v)
    else: self.data[:3, :3] = v

  def apply(self, v: Vec3) -> Vec3:
    assert isinstance(v, Vec3)
    return Vec3(self.map(v.data)).canon

  def apply_vec(self, v: Vec3) -> Vec3:
    return Vec3(self.rot @ v.data)

  def map(self, v):
    if len(v) == 0: return np.zeros((0, 4))
    return (self.data @ v.T).T

  def imap(self, v):
    return self.inv.map(v)

  def __mul__(self, peer):
    return Transform(data=self.data @ peer.data)

  def __matmul__(self, peer):
    if isinstance(peer, Vec3):
      return self.apply(peer)
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

  class Config:
    arbitrary_types_allowed = True


def make_rot(all_good=False, **kwargs) -> np.ndarray:
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
    d0 = np.zeros((0, 3))
    return Particles(weights=d0, p=d0, v=d0, id_obj=d0, n=0)

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
    from chdrft.display.service import g_plot_service
    data = A(points=pts, lines=lines, conf=A(mode='3D'))
    if by_col:
      from chdrft.utils.colors import ColorMapper
      vals = set(self.id_obj)
      cm = ColorMapper(vals)
      data.points_color = [cm.get(x) for x in self.id_obj]
      data.lines = [A(polyline=x, color=cm.get(id)) for id, x in zip(self.id_obj, data.lines)]
    g_plot_service.plot(data)

  class Config:
    arbitrary_types_allowed = True


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
    return AABB.FromPoints(list(itertools.chain(*[x.bounds.points for x in self.meshes])))

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


class MoveDesc(cmisc.PatchedModel):
  v_l: Vec3 = cmisc.pyd_f(Vec3.Zero)
  rotvec_ang_l: Vec3 = cmisc.pyd_f(Vec3.X)
  rotspeed: float = 0

  def add_rotspeed(self, delta: float):
    self.rotspeed += delta

  def load_rotvec(self, rotvec: Vec3):
    self.rotspeed, self.rotvec_ang_l = rotvec.norm_and_uvec
    #assert self.rotvec_ang_l.norm > 0.5

  @classmethod
  def From(cls, rotvec: Vec3) -> "MoveDesc":
    res = MoveDesc()
    res.load_rotvec(rotvec)
    return res

  @property
  def rotvec_l(self) -> Vec3:
    return self.rotvec_ang_l * self.rotspeed

  @property
  def skew_matrix(self) -> np.ndarray:
    return self.rotvec_l.skew_matrix

  @property
  def skew_transform(self) -> Transform:
    return self.rotvec_l.skew_transform

  class Config:
    arbitrary_types_allowed = True


class InertialTensor(BaseModel):

  data: np.ndarray | jnp.ndarray = cmisc.pyd_f(lambda: np.zeros((3, 3)))

  class Config:
    arbitrary_types_allowed = True

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
    return Vec3(self.data @ v.vdata)

  def mom2vel(self, v: Vec3) -> Vec3:
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
  com: Vec3 = cmisc.pyd_f(Vec3.ZeroPt)
  inertial_tensor: InertialTensor = cmisc.pyd_f(InertialTensor)
  mesh: MeshDesc = cmisc.pyd_f(MeshDesc)

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

  class Config:
    arbitrary_types_allowed = True


class SceneContext(cmisc.PatchedModel):
  cur_id: int = 0
  mp: dict = cmisc.pyd_f(dict)
  obj2name: "dict[RigidBody,str]" = cmisc.pyd_f(dict)
  name2obj: "dict[str,RigidBody]" = cmisc.pyd_f(dict)
  basename2lst: dict[str, list] = cmisc.pyd_f(lambda: cmisc.defaultdict(list))

  def single(self, name: str) -> "RigidBody":
    tb = self.basename2lst[name]
    assert len(tb) == 1
    return tb[0]

  def get_name(self, obj: "RigidBody") -> str:
    if not obj in self.obj2name:
      bname = obj.data.base_name
      lst = self.basename2lst[bname]
      name = f'{bname}_{len(lst):03d}'
      self.obj2name[obj] = name
      self.name2obj[name] = obj
      lst.append(obj)
    return self.obj2name[obj]

  def get_id(self, obj) -> int:
    if obj not in self.mp:
      self.mp[obj] = self.next_id()
    return self.mp[obj]

  def next_id(self) -> int:
    res = self.cur_id
    self.cur_id += 1
    return res

  def register_link(self, link: "RigidBodyLink"):
    self.get_name(link.rb)

  def create_rb(self, **kwargs) -> "RigidBody":
    return RigidBody(ctx=self, **kwargs)


class LinkData(cmisc.PatchedModel):
  static: bool = True
  pivot_rotaxis: Vec3 = None
  free: bool = False
  pivot_rotang: float | jnp.ndarray = None
  wl_free: Transform = cmisc.pyd_f(Transform.From)

  @property
  def wl(self) -> Transform:
    if self.free: return self.wl_free
    return self.pivot_rotaxis.exp_rot_u(self.pivot_rotang)

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert self.static or (self.pivot_rotaxis is not None or self.free)

  class Config:
    arbitrary_types_allowed = True


class RigidBodyLink(cmisc.PatchedModel):
  wl0: Transform = cmisc.pyd_f(Transform.From)
  move_desc: MoveDesc = cmisc.pyd_f(MoveDesc)
  link_data: LinkData = cmisc.pyd_f(LinkData)
  rb: "RigidBody"

  parent: "Optional[RigidBodyLink]" = Field(repr=False, default=None)

  def __init__(self, update_link=True, **kwargs):
    super().__init__(**kwargs)
    if update_link:
      self.rb.self_link = self
      for x in self.rb.move_links:
        x.parent = self

  @property
  def children(self) -> "list[RigidBodyLink]":
    return self.rb.move_links

  @property
  def wl(self) -> Transform:
    return self.wl0 @ self.link_data.wl

  @staticmethod
  def FixedLink(rb: "RigidBody") -> "RigidBodyLink":
    return RigidBodyLink(rb=rb)

  def remap(self, lw: Transform) -> "RigidBodyLink":
    res = self.copy()
    res.wl = (lw @ self.lw).inv
    return res

  @property
  def mass(self) -> float:
    return self.rb.mass

  @property
  def cur_mesh(self) -> MeshDesc:
    return TransformedMeshDesc(transform=self.wl, mesh=self.rb.cur_mesh)

  @property
  def com_w(self) -> Vec3:
    return self.wl @ self.rb.com

  @property
  def com_l(self) -> Vec3:
    return self.rb.com

  @property
  def lw(self) -> Transform:
    return self.wl.inv

  @property
  def lw_rot(self) -> Transform:
    return self.wl_rot.inv

  @property
  def wl_rot(self) -> Transform:
    return self.wl.tsf_rot

  @property
  def world_vel(self) -> Vec3:
    return Vec3(self.wl.rot_R.apply(self.move_desc.v_l.data))

  @property
  def v_l(self) -> Vec3:
    return self.move_desc.v_l

  def aabb(self, only_self=False) -> AABB:
    pts = []
    pts.extend(self.rb.spec.mesh.bounds.points)
    if not only_self:
      for x in self.rb.move_links:
        pts.extend(x.aabb().points)

    return AABB.FromPoints(self.wl.map(np.array(pts)))

  def get_particles(self, n, **kwargs) -> Particles:
    return self.to_world_particles(self.rb.get_particles(n, **kwargs))

  @property
  def root_link(self) -> "RigidBodyLink":
    assert 0  # not working anymore
    if not self.parent: return self
    links = []
    x = self
    while x is not None:
      links.append(x)
      x = x.parent
    links = links[::-1]
    res = RigidBodyLink(rb=self.rb, update_link=False)
    rotvec_w = Vec3.Zero()

    for l in links:
      res.wl0 = res.wl @ l.wl
      rotvec_w += res.wl_rot @ l.rotvec_w
    res.move_desc.load_rotvec(res.lw_rot @ rotvec_w)
    return res

  @property
  def rotvec_l(self) -> Vec3:
    return self.move_desc.rotvec_l

  @rotvec_l.setter
  def rotvec_l(self, v):
    self.move_desc.load_rotvec(v)

  @property
  def rotvec_w(self) -> Vec3:
    return self.wl @ self.rotvec_l

  @rotvec_w.setter
  def rotvec_w(self, v):
    self.move_desc.load_rotvec(self.lw_rot @ v)

  def world_angular_momentum2(self) -> Vec3:
    res = Vec3.Zero()
    for x in self.rb.descendants():
      res += x.rb.root_angular_momentum()
    return res

  def world_angular_momentum(self, com: Vec3 = None) -> Vec3:
    if com is None: com = self.com_w
    a = self.world_inertial_tensor(com).vel2mom(self.rotvec_w)
    b = self.wl_rot @ self.rb.local_angular_momentum()
    return a + b

  def world_linear_momentum(self) -> Vec3:
    #res = self.child.local_linear_momentum + self.move_desc.skew_matrix @ self.com.data * self.child.mass
    res = self.rb.local_linear_momentum() + self.v_l * self.mass
    return Vec3(self.wl.rot_R.apply(res.data))

  def local_inertial_tensor(self) -> InertialTensor:
    return self.rb.local_inertial_tensor()

  def world_inertial_tensor(self, com: Vec3) -> InertialTensor:
    return self.local_inertial_tensor().get_world_tensor(
        self.wl
    ).shift_inertial_tensor(com - self.com_w, self.mass)

  def to_world_particles(self, x: Particles):
    if len(x.p) == 0:
      return x
    x.v = self.wl.map(
        x.v + self.move_desc.skew_transform.map(norm_pts(x.p) - norm_pts(self.com_l.data))
    )
    x.p = self.wl.map(x.p)
    return x

  @property
  def depth(self):
    return self.root_and_depth.depth

  @property
  def root(self):
    return self.root_and_depth.root

  @property
  def tree_expl(self):
    return TreeExpl(node=self)

  @property
  def root_and_depth(self):
    u = self
    cnt = 0
    while u.parent is not None:
      u = u.parent
      cnt += 1
    return A(root=u, depth=cnt)

  def inertial_tensor_for_par(self, mass=None) -> InertialTensor:
    if mass is None: mass = self.rb.mass
    return InertialTensor.FromPointMass(self.parent.com_l - self.com_w, mass)

  def inertial_tensor_for_par_l(self, mass=None) -> InertialTensor:
    if mass is None: mass = self.rb.mass
    return InertialTensor.FromPointMass(self.lw @ self.parent.com_l - self.com_l, mass)

  def d_angular_mom(self) -> Vec3:
    return self._d_angular_mom(Transform.From(), Vec3.Zero(), Vec3.Zero())

  def _d_angular_mom(self, wl: Transform, w_l_wl: Vec3, d_w_l_wl) -> Vec3:
    w_l_wl = self.lw_rot @ w_l_wl + self.rotvec_l
    d_w_l_wl = self.lw_rot @ d_w_l_wl + (-self.rotvec_l).skew_transform @ w_l_wl
    wl = wl @ self.wl_rot

    res = wl @ w_l_wl.skew_transform @ self.rb.spec.inertial_tensor.vel2mom(w_l_wl)
    res += wl @ self.rb.spec.inertial_tensor.vel2mom(d_w_l_wl)
    for child in self.rb.move_links:
      ix = child.inertial_tensor_for_par()
      p = child.com_w - self.com_l
      dix = InertialTensor.DFromPointMass(p, self.rotvec_l.skew_transform @ p, child.mass)
      res += wl @ dix.vel2mom(w_l_wl)
      res += wl @ ix.vel2mom(d_w_l_wl)
      res += wl @ w_l_wl.skew_transform @ ix.vel2mom(w_l_wl)
      res += child._d_angular_mom(wl, w_l_wl, d_w_l_wl)
    return res


kDefaultName = "noname"


class RBData(cmisc.PatchedModel):
  base_name: str = kDefaultName
  internal: bool = False


class RigidBody(cmisc.PatchedModel):
  spec: SolidSpec = cmisc.pyd_f(SolidSpec)
  data: RBData = cmisc.pyd_f(RBData)
  self_link: Optional[RigidBodyLink] = Field(repr=False, default=None)
  move_links: list[RigidBodyLink] = cmisc.pyd_f(list)

  ctx: SceneContext = Field(repr=False)

  @property
  def parent(self) -> "RigidBody | None":
    if not self.self_link.parent: return None
    return self.self_link.parent.rb

  @property
  def name(self) -> str:
    if self.data.internal: return f'{self.base_name}_internal'
    return self.ctx.get_name(self)

  def __str__(self):
    return f'RigidBody({self.idx=}, {self.name=})'

  def __repr__(self):
    return str(self)

  @property
  def idx(self) -> int:
    return self.ctx.get_id(id(self))

  def __eq__(self, peer):
    return id(self) == id(peer)

  @property
  def mass(self) -> float:
    return self.spec.mass + sum([x.mass for x in self.move_links])

  @property
  def com(self) -> Vec3:
    return sum(
        [x.com_w * x.mass for x in self.move_links], self.spec.com * self.spec.mass
    ) / self.mass

  @property
  def static_local_inertial_tensor(self) -> InertialTensor:
    return self.spec.inertial_tensor.shift_inertial_tensor(self.spec.com - self.com, self.spec.mass)

  def local_inertial_tensor(self) -> InertialTensor:
    return sum(
        [x.world_inertial_tensor(self.com) for x in self.move_links],
        self.static_local_inertial_tensor
    )

  def local_angular_momentum(self) -> Vec3:
    return sum([x.world_angular_momentum(self.com) for x in self.move_links], Vec3.Zero())

  def get_self_particles(
      self, n, use_importance=False, particles_filter=lambda _: True
  ) -> Particles:
    if not particles_filter(self): return Particles.Default()
    if use_importance:
      pts, w = self.spec.mesh.importance_sampling(n, self.spec.mass)
      n = len(pts)
    else:
      pts = self.spec.mesh.rejection_sampling(n)
      n = len(pts)
      w = np.ones((n,)) * (self.spec.mass / max(n, 1))

    return Particles(
        weights=w, p=pts, n=n, v=np.zeros((n, 4)), id_obj=np.ones((n,), dtype=int) * self.idx
    )

  def get_particles(self, n, **kwargs) -> Particles:
    fmass = self.mass
    tb = [self.get_self_particles(int(n * self.spec.mass / fmass), **kwargs)]
    tb.extend([x.get_particles(int(n * x.mass / fmass), **kwargs) for x in self.move_links])
    res = functools.reduce(Particles.reduce, tb)
    return res

  @property
  def root_wl(self) -> Transform:
    wl = self.self_link.wl
    if self.parent is not None:
      wl = self.parent.root_wl @ wl
    return wl

  @property
  def cur_mesh(self) -> MeshDesc:
    return ComposedMeshDesc(
        meshes=[self.spec.mesh] + [x.cur_mesh for x in self.move_links],
        weights=[self.spec.mass] + [x.mass for x in self.move_links]
    )

  def plot(self, npoints=10000):
    from chdrft.display.service import oplt
    from chdrft.display.vispy_utils import get_colormap
    pts, weights = self.spec.mesh.importance_sampling(npoints)
    viridis = get_colormap('viridis')
    cols = viridis.map(weights / max(weights))
    oplt.plot(A(points=pts, conf=A(mode='3D'), points_color=cols))

  def root_angular_momentum(self) -> Vec3:
    mass = self.spec.mass
    cur = self.spec.inertial_tensor.shift_inertial_tensor(self.com - self.spec.com, mass)
    lnk = self.self_link
    am = Vec3.Zero()
    com = self.spec.com
    while True:
      am += cur.vel2mom(lnk.rotvec_l)
      am = lnk.wl_rot @ am
      if lnk.parent is None:
        break
      cur = cur.get_world_tensor(lnk.wl) + lnk.inertial_tensor_for_par(mass)
      lnk = lnk.parent
    return am

  def descendants(self, wl: Transform = None) -> "list[DescendantEntry]":
    if wl is None: wl = Transform.From()

    res = []
    res.append(DescendantEntry(wl=wl, rb=self))
    for x in self.move_links:
      res.extend(x.rb.descendants(wl @ x.wl))
    return res

  def self_rot_angular_inertial_tensor(self) -> InertialTensor:
    res = InertialTensor()
    res += self.spec.inertial_tensor.shift_inertial_tensor(self.com - self.spec.com, self.spec.mass)
    for x in self.descendants()[1:]:  # skip self
      ix = x.rb.spec.inertial_tensor + x.rb.self_link.inertial_tensor_for_par_l()
      res += ix.get_world_tensor(x.wl)
    return res


class DescendantEntry(cmisc.PatchedModel):
  wl: Transform
  rb: RigidBody


RigidBody.update_forward_refs()
RigidBodyLink.update_forward_refs()


class RBBuilderEntry(cmisc.PatchedModel):
  spec: SolidSpec = None
  wl: Transform


class RBDescEntry(cmisc.PatchedModel):
  data: RBData = cmisc.pyd_f(RBData)
  spec: SolidSpec = None
  parent: "RBDescEntry" = None
  wl: Transform = cmisc.pyd_f(Transform.From)
  link_data: LinkData = cmisc.pyd_f(LinkData)
  move_desc: MoveDesc = None


class RBBuilderAcc(cmisc.PatchedModel):
  entries: list[RBBuilderEntry] = cmisc.pyd_f(list)
  src_entries: list[RBDescEntry] = cmisc.pyd_f(list)
  dyn_links: list[RigidBodyLink] = cmisc.pyd_f(list)
  sctx: SceneContext

  def build(self) -> RigidBody:
    l = self.entries
    weights = [x.spec.mass for x in l]
    mass = sum(weights)
    com = sum([(x.wl @ x.spec.com) * x.spec.mass for x in l], Vec3.Zero()) / mass
    mesh = ComposedMeshDesc(
        weights=weights,
        meshes=[TransformedMeshDesc(mesh=x.spec.mesh, transform=x.wl) for x in l],
    )
    it = sum(
        [
            x.spec.inertial_tensor.shift_inertial_tensor(com - x.wl @ x.spec.com,
                                                         x.spec.mass).get_world_tensor(x.wl)
            for x in l
        ], InertialTensor()
    )
    spec = SolidSpec(mass=mass, com=com, mesh=mesh, inertial_tensor=it)
    names = [x.data.base_name for x in self.src_entries if x.data.base_name != kDefaultName]
    agg_name = kDefaultName
    if len(names) == 1:
      agg_name = names[0]
    else:
      agg_name = f'Agg({",".join(names)})'

    return RigidBody(
        spec=spec,
        move_links=self.dyn_links,
        ctx=self.sctx,
        data=RBData(base_name=agg_name),
    )


class RBTree(cmisc.PatchedModel):
  entries: list[RBDescEntry] = cmisc.pyd_f(list)
  entry2child: dict[RBDescEntry,
                    list[RBDescEntry]] = cmisc.pyd_f(lambda: cmisc.defaultdict(list))
  child_links: dict[RBDescEntry,
                    list[RigidBodyLink]] = cmisc.pyd_f(lambda: cmisc.defaultdict(list))
  sctx: SceneContext

  def add(self, entry: RBDescEntry) -> RBDescEntry:
    self.entries.append(entry)
    self.entry2child[entry.parent].append(entry)
    return entry

  def add_link(self, cur: RBDescEntry, child: RigidBodyLink):
    self.child_links[cur].append(child)

  def create(self, cur: RBDescEntry, wl=Transform.From(), wl_from_com=False) -> RigidBodyLink:
    res = RBBuilderAcc(sctx=self.sctx)
    self.dfs(cur, Transform.From(), res)
    rb = res.build()
    assert cur.move_desc is not None
    rbl = RigidBodyLink(rb=rb, wl0=wl, move_desc=cur.move_desc, link_data=cur.link_data)
    self.sctx.register_link(rbl)
    if wl_from_com:
      wl.pos_v = wl.pos_v - (wl.tsf_rot @ rbl.rb.com)
    return rbl

  def dfs(self, cur: RBDescEntry, wl: Transform, res: RBBuilderAcc):
    children = self.entry2child[cur]
    static_children = [x for x in children if x.move_desc is None]
    dyn_children = [x for x in children if x.move_desc is not None]
    assert cur.link_data.pivot_rotang is None or cur.link_data.pivot_rotang is not None

    for lnk in self.child_links[cur]:
      lnk.wl = wl @ lnk.wl
    links = [self.create(x, wl @ x.wl) for x in dyn_children] + self.child_links[cur]
    res.dyn_links.extend(links)
    if cur.spec: res.entries.append(RBBuilderEntry(wl=wl, spec=cur.spec))
    res.src_entries.append(cur)

    for x in static_children:
      self.dfs(x, wl @ x.wl, res)

  def build(self):
    links = []
    for x in self.entry2child[None]:
      links.append(self.create(x))
    return links


class TreeExpl(cmisc.PatchedModel):
  node: RigidBody

  def get_path(self, a, b) -> list[Tuple[RigidBody, bool]]:
    ia = a
    ib = b
    left = []
    right = []
    while ia != ib:
      if ia.depth >= ib.depth:
        left.append(ia)
        ia = ia.parent
      if ib.depth > ia.depth:
        right.append(ib)
        ib = ib.parent
    right.append(ia)
    return [(x, 0) for x in left] + [(x, 1) for x in reversed(right)]

    return path

  def eval(self, path: list[Tuple[RigidBody, bool]]) -> Transform:
    res = Transform.From()
    for i in range(len(path) - 1):
      nx, down = path[i]
      if down:
        res = path[i + 1][0].wl.inv * res
      else:
        res = nx.wl * res
    return res

  def to_node(self, peer):
    return self.eval(self.get_path(self.node, peer))

  def from_node(self, peer):
    return self.eval(self.get_path(peer, self.node))


class ReactionWheelHelper(cmisc.PatchedModel):
  """
  Z: frame rot axis
  """
  i_frame: InertialTensor
  i_wheel: InertialTensor
  body2frame: Transform
  body_speed: AngularSpeed

  ang_frame: float
  w_frame: float
  p_wheel: Vec3

  @property
  def frame2wheel(self) -> R:
    return R.from_euler('z', self.ang_frame)

  def compute_angular_momentum(self):
    pass


class LinkChainEntry(cmisc.PatchedModel):
  lw_rot: Transform
  rot_w_wl: Vec3


class TorqueComputer:

  def __init__(self):
    self.mp = dict()
    self.updates: dict[RigidBody, A] = cmisc.defaultdict(A)
    self.root: RigidBodyLink = None
    self.dt = None
    self.req_link_torque: dict[RigidBody, float] = cmisc.defaultdict(float)

  def setup_(self, root: RigidBodyLink):
    self.updates[root.rb]
    for x in root.rb.move_links:
      self.setup_(x)

  def setup(self, root: RigidBodyLink, dt: float):
    self.root = root
    self.dt = dt
    self.orig_angular_mom = root.world_angular_momentum()
    self.setup_(root)

  def process(self):
    self.compute_wl()
    self.update_wl()
    self.compute_rotvec()
    self.update_rotvec()
    cur = self.root.world_angular_momentum()
    target_mom = self.orig_angular_mom + self.updates[self.root.rb].torque * self.dt
    ix = self.root.rb.self_rot_angular_inertial_tensor().get_world_tensor(self.root.wl)
    err_mom = target_mom - cur
    err_w = ix.mom2vel(err_mom)
    self.root.rotvec_w += err_w
    final = self.root.world_angular_momentum()

  def compute_torques(self):
    return self._compute_torque(self.root)

  def _compute_torque(self, link: RigidBodyLink) -> Vec3:
    torque = Vec3.Zero()
    for x in link.children:
      torque += self._compute_torque(x)
    data = self.updates[link.rb]

    if link.link_data.free:
      data.torque = torque
      data.torque = Vec3.Zero()
    else:
      req_torque = self.req_link_torque[link]
      iv = link.local_inertial_tensor().vel2mom(link.link_data.pivot_rotaxis)
      if 0:
        proj_on = iv.uvec
        torque += link.link_data.pivot_rotaxis * req_torque
        torque_on_axis = torque.proj(proj_on)
        torque_off_axis = torque - torque_on_axis
        tnorm_axis = torque_on_axis.norm + req_torque
      else:
        #sketchy if iv <> pivot_rotaxis
        proj_on = link.link_data.pivot_rotaxis
        torque_on_axis = torque.proj(proj_on)
        torque_off_axis = torque - torque_on_axis
        tnorm_axis = torque_on_axis.norm + req_torque

      data.dw = tnorm_axis / iv.norm

      torque = torque_off_axis - link.link_data.pivot_rotaxis * req_torque
    return link.wl_rot @ torque

  def compute_rotvec_(self, link: RigidBodyLink):
    base_link = link.rb.self_link
    data = self.updates[link.rb]
    if link.rb.mass == 0:
      data.delta_rotspeed = 0
      return

    if link.link_data.free:
      assert link.parent is None
      torque = data.torque
      torque_w = link.wl_rot @ torque
      mom = link.world_angular_momentum()
      nmom = mom + torque_w * self.dt

      tensor = link.local_inertial_tensor()
      drotvec_l = tensor.mom2vel(torque * self.dt)
      data.rotvec = base_link.rotvec_l + drotvec_l
    else:
      data.delta_rotspeed = data.dw * self.dt
    #print(
    #    f'OLD rotvec {base_link.rb.name=}, {base_link.move_desc.rotvec_l=}, {torque=}, {link.link_data.pivot_rotaxis=}'
    #)

  def compute_wl(self):
    for child, data in self.updates.items():
      base_link = child.self_link
      if base_link.link_data.free:
        start_pos_v = base_link.wl.pos_v
        data.wl = base_link.wl @ (base_link.move_desc.rotvec_l * self.dt).exp_rot(base_link.com_l)
        data.wl.pos_v = start_pos_v
      else:
        data.pivot_rotang = base_link.link_data.pivot_rotang + base_link.move_desc.rotspeed * self.dt

  def compute_rotvec(self):
    self.compute_torques()
    for child in self.updates.keys():
      self.compute_rotvec_(child.self_link)

  def set_rotvec(self, child: RigidBody, data):
    if 'rotvec' in data:
      child.self_link.move_desc.load_rotvec(data.rotvec)
    else:
      child.self_link.move_desc.add_rotspeed(data.delta_rotspeed)

  def set_wl(self, child: RigidBody, data):
    cl = child.self_link
    if child.self_link.link_data.free:
      cl.link_data.wl_free = cl.wl0.inv @ data.wl
    else:
      cl.link_data.pivot_rotang = data.pivot_rotang

  def update_wl(self):
    self._update(self.set_wl)

  def update_rotvec(self):
    self._update(self.set_rotvec)

  def _update(self, func):
    for child, data in self.updates.items():
      func(child, data)


class NumpyPackerEntry(cmisc.PatchedModel):
  shape: np.ndarray
  pos: int
  name: str

  @property
  def size(self) -> int:
    return np.multiply.reduce(self.shape)

  class Config:
    arbitrary_types_allowed = True


class NumpyPacker:

  def __init__(self):
    self.tb: list[NumpyPackerEntry] = []
    self.pos = 0

  def add_dict(self, **kwargs):
    for k, v in kwargs.items():
      self.add(k, v)

  @property
  def zeros_packed(self) -> np.ndarray:
    return np.zeros(self.pos)

  def add(self, name, shape):
    if isinstance(shape, int): shape = (shape,)
    shape = np.array(shape)
    x = NumpyPackerEntry(name=name, pos=self.pos, shape=shape)
    self.tb.append(x)
    self.pos += x.size

  def pack(self, data: dict) -> np.ndarray:
    res = []
    for x in self.tb:
      res.append(g_oph.array(data[x.name]).reshape(x.size))
    return g_oph.hstack(res)

  def unpack(self, v: np.ndarray) -> dict:
    res = A()
    for x in self.tb:
      res[x.name] = v[x.pos:x.pos + x.size].reshape(x.shape)
    return res


class Simulator(cmisc.PatchedModel):

  class Config:
    arbitrary_types_allowed = True

  root: RigidBody = None
  sctx: SceneContext
  name2pos: dict[str, int]
  tc: TorqueComputer = cmisc.pyd_f(TorqueComputer)
  objs: list[RigidBody] = None
  packer: NumpyPacker = None

  @property
  def nobj(self) -> int:
    return len(self.objs)

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.root = self.sctx.single('root')
    objs = []
    for name, lst in self.sctx.basename2lst.items():
      if name not in self.name2pos: continue
      a: RigidBody = lst[0]
      pos = self.name2pos[name]
      objs.append(a)
    self.objs = objs

    self.packer = NumpyPacker()
    self.packer.add_dict(root_wl=(4, 4), root_rotvec_l=3, wh_w=self.nobj, wh_ang=self.nobj)

  def load_state0(self, state):
    self.load_state(**self.packer.unpack(state))

  def dump_state0(self):
    return self.packer.pack(self.dump_state())

  def load_state(self, root_wl=None, root_rotvec_l=None, wh_w=None, wh_ang=None):

    for i, obj in enumerate(self.objs):
      obj.self_link.link_data.pivot_rotang = wh_ang[i]
      obj.self_link.move_desc.rotspeed = wh_w[i]

    self.tc.set_wl(self.root, A(wl=Transform(data=root_wl)))
    self.root.self_link.rotvec_l = Vec3(root_rotvec_l)

  def update(self, dt: float, u_torques: np.ndarray):
    for i, obj in enumerate(self.objs):
      self.tc.req_link_torque[obj.self_link] = u_torques[i]
    self.tc.setup(self.root.self_link, dt)
    self.tc.process()

  def dump_state(self):
    o_wh_ang = make_array([x.self_link.link_data.pivot_rotang for x in self.objs])
    o_wh_w = make_array([x.self_link.move_desc.rotspeed for x in self.objs])
    return dict(
        root_wl=self.root.self_link.wl.data,
        root_rotvec_l=self.root.self_link.rotvec_l.vdata,
        wh_w=o_wh_w,
        wh_ang=o_wh_ang
    )

  def do_step(self, dt, controls, state):
    self.load_state0(state)
    for wh_torque in controls:
      self.update(dt, wh_torque)
    return self.dump_state0()


class SimulParams(cmisc.PatchedModel):
  ndt_const: int
  dt: float
  integ_nsteps: int
  nt: int
  nctrl: int
  i_state: A
  f_conds: A

  packer: NumpyPacker = None

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.packer = NumpyPacker()
    self.packer.add_dict(ctrl=(self.nt, self.nctrl))

  @property
  def actual_dt(self) -> float:
    return self.dt / self.integ_nsteps

  class Config:
    arbitrary_types_allowed = True


class RB_PbAnalysis(cmisc.PatchedModel):
  dx: A = None
  ctrl_bounds: list = cmisc.pyd_f(lambda: [-10, 10])
  sim: Simulator = None

  class Config:
    arbitrary_types_allowed = True
    keep_untouched = (cmisc.cached_property,)

  def setup(self):
    self.sim = self.dx.create_simulator()
    #self.jit_ctrl
    #self.jac_func

  @property
  def state_packer(self) -> NumpyPacker:
    return self.sim.packer

  @property
  def args1(self):
    return A(
        u=np.zeros((1, self.dx.nctrl)),
        state=A(
            root_wl=np.identity(4),
            root_rotvec_l=np.identity(3)[1],
            wh_w=np.zeros(self.dx.nctrl),
            wh_ang=np.zeros(self.dx.nctrl),
        ),
    )

  @property
  def args0(self):
    return A(
        u=np.zeros((1, self.dx.nctrl)),
        state=A(
            root_wl=np.identity(4),
            root_rotvec_l=np.zeros(3),
            wh_w=np.zeros(self.dx.nctrl),
            wh_ang=np.zeros(self.dx.nctrl),
        ),
    )

  @cmisc.cached_property
  def jac_func(self):

    def build_jit(nsteps, *args):
      res = jax.jacrev(lambda *b: px.integrate(*b, nsteps), argnums=(1,))
      return res(*args)

    return jax.jit(build_jit, static_argnums=0)

  def do_control(self, dt, controls, state):
    sx: Simulator = self.dx.create_simulator()
    return sx.do_step(dt, controls, state)

  @cmisc.cached_property
  def jit_ctrl(self):
    jit_ctrl = jax.jit(lambda *args: self.do_control(*args))
    jit_ctrl(0, self.args0.u, self.state_packer.zeros_packed)  # trigger jit
    return jit_ctrl

  def integrate1(self, dt, u, state, rk4=False, use_jit=1):
    assert not rk4, "Dont know how to handle this with angular momentum conservation normalization"
    # Would need sympletic integration scheme?
    fx = self.jit_ctrl if use_jit else self.do_control
    return fx(dt, u, state)

  def integrate(self, dt, u, state, nsteps, **kwargs):
    for _ in range(nsteps):
      state = self.integrate1(dt, u, state, **kwargs)
    return state

  def proc(self, params: SimulParams, ctrl):

    ctrl = params.packer.unpack(ctrl).ctrl
    state = self.state_packer.pack(params.i_state)
    ntx = len(ctrl)
    obj = 0
    for i in range(ntx):
      state = self.integrate(
          params.actual_dt,
          (ctrl[i],),
          state,
          params.ndt_const * params.integ_nsteps,
      )
    obj += self.eval_obj(params, state)
    return (obj,)

  def eval_obj(self, params: SimulParams, state: np.ndarray) -> np.ndarray:
    state = self.state_packer.unpack(state)

    cur_f_root_rotvec_w = state.root_wl[:3, :3] @ state.root_rotvec_l
    #ans = (last_t, np.linalg.norm(cur_f_root_rotvec_w - f_root_rotvec_w))
    obj = 0

    f_root_rotvec_w = params.f_conds.get('f_root_rotvec_w')
    f_root_wl = params.f_conds.get('f_root_wl')
    if f_root_rotvec_w is not None:
      obj += g_oph.norm(cur_f_root_rotvec_w - f_root_rotvec_w)
    if f_root_wl is not None:
      #obj += g_oph.norm((state.root_wl - f_root_wl).reshape((-1,)))
      obj += np.linalg.norm(state.root_wl - f_root_wl)
    return obj

  def get_pb(this, params: SimulParams):

    def do_proc(ctrl):
      return this.proc(params, ctrl)

    def build_jit(ctrl):
      res = jax.jacrev(do_proc)
      return res(ctrl)

    #jac_func = jax.jit(build_jit)
    #jac_func(params.packer.zeros_packed)

    class Pb1:

      def __init__(self, params: SimulParams):
        self.dim = params.packer.pos
        self.params = params

      def get_nec(self):
        return 0

      def batch_fitness(self, data):
        print('GOOT ', data.shape)
        assert 0
      def fitness(self, ctrl):
        return do_proc(ctrl)

      def gradient(self, ctrl):
        #return jac_func(ctrl)[0]
        return pg.estimate_gradient_h(lambda x: self.fitness(x), ctrl)

      def get_bounds(self):
        return (
            np.full((self.dim,), this.ctrl_bounds[0]), np.full((self.dim,), this.ctrl_bounds[1])
        )

    return Pb1(params)

  def solve_pb(self, pb):
    prob = pg.problem(pb)
    nl = pg.nlopt('slsqp')
    algo = pg.algorithm(pg.nlopt("slsqp"))
    nl.ftol_rel = 1e-4
    nl.ftol_abs = 1e-2
    nl.maxeval = 1000

    ip = pg.ipopt()
    ip.set_numeric_option("tol",1E-3) # Change the relative convergence tolerance
    ip.get_numeric_options() 

    algo = pg.algorithm(nl)
    algo.set_verbosity(1)

    prob.c_tol = [1e-3] * prob.get_nc()
    pop = pg.population(prob, size=1000)
    print('start solve')
    pop = algo.evolve(pop)
    raw_ans = pop.champion_x
    ctrl = pb.params.packer.unpack(raw_ans).ctrl
    ans = []
    for grp in Format(pop.champion_x).bucket(pb.params.nctrl).v:
      ans.extend([grp] * pb.params.ndt_const)
    ans = np.array(ans)
    return A(algo=algo, pop=pop, prob=prob, ans=ans, raw_ans=raw_ans, ctrl=ctrl)


#class RBCompactor:
#
#  def __init__(self, wl: Transform, rotvec_w: Vec3):
#    self.wl = wl
#    self.statics: list[Tuple[Transform, RigidBodyLink]] = []
#    self.rotvec_w = rotvec_w
#    self.res_children: list[RigidBodyLink] = []
#
#  @classmethod
#  def Build(cls, link: RigidBodyLink) -> RigidBodyLink:
#    return link.child.ctx.compose(cls.Compact(link))
#
#  @classmethod
#  def Compact(cls,
#              link: RigidBodyLink,
#              wl: Transform = None,
#              rotvec_w: Vec3 = None) -> list[RigidBodyLink]:
#    if wl is None:
#      wl = link.wl
#      rotvec_w = link.rotvec_w
#    return cls(wl, rotvec_w).go(link)
#
#  def go(self, link: RigidBodyLink) -> list[RigidBodyLink]:
#    self.dfs(link, Transform.From())
#    mass = sum([x.child.spec.mass for wl, x in self.statics], 0)
#    com = Vec3.Zero()
#    tensor = InertialTensor()
#    tsf = self.wl.clone()
#    if mass > 0:
#      com = sum([wl.pos_v * x.child.spec.mass for wl, x in self.statics], Vec3.Zero()) / mass
#      tensor = sum(
#          [
#              x.child.spec.inertial_tensor.shift_inertial_tensor(com - wl.pos_v, x.child.spec.mass
#                                                                ).get_world_tensor(wl)
#              for wl, x in self.statics
#          ], InertialTensor()
#      )
#      tsf.pos_v -= com
#
#    spec = SolidSpec(
#        mass=mass,
#        inertial_tensor=tensor,
#    )
#    base_name = '|'.join(x.child.name for wl, x in self.statics)
#    cur = RigidBodyLink(
#        child=RigidBody(spec=spec, ctx=link.child.ctx, base_name=base_name),
#        wl=tsf,
#        move_desc=MoveDesc.From(self.rotvec_w)
#    )
#    self.res_children.append(cur)
#
#    return self.res_children
#
#  def dfs(self, link: RigidBodyLink, wl: Transform):
#    self.statics.append((wl, link))
#    for clink in link.child.links:
#      nwl = wl @ clink.wl
#      if clink.link_data.static:
#        self.dfs(clink, nwl)
#      else:
#        self.res_children.extend(
#            RBCompactor.Compact(clink, nwl, self.rotvec_w + wl.tsf_rot @ clink.rotvec_w)
#        )
#


def control_simple(nx, wh_w, wh_ang):
  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Box(1, 1, 1, 1),
          move_desc=MoveDesc.From(Vec3.Z()),
          link_data=LinkData(static=False, free=True),
      )
  )

  wheel_r = 0.5
  wheel_h = 0.3
  data = [
      A(name='wh_x', wl=Transform.From(rot=make_rot(z=Vec3.X()), pos=[2, 0, 0])),
      A(name='wh_y', wl=Transform.From(rot=make_rot(z=Vec3.Y()), pos=[0, 2, 0])),
      A(name='wh_z', wl=Transform.From(rot=make_rot(z=Vec3.Z()), pos=[0, 0, 2])),
  ][:nx]

  for i, dx in enumerate(data):
    tx.add(
        RBDescEntry(
            data=RBData(base_name=dx.name),
            move_desc=MoveDesc(rotvec_ang_l=Vec3.Z(), rotspeed= wh_w[i]),
            wl=dx.wl,
            link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z(), pivot_rotang=wh_ang[i]),
            spec=SolidSpec.Cylinder(2, wheel_r, wheel_h),
            parent=root,
        )
    )
  res = tx.create(root, wl_from_com=True)
  return A(target=res, sctx=sctx)


def test3(ctx):
  d = scene_small()
  vp = d.target.cur_mesh.surface_mesh(SurfaceMeshParams()).vispy_data

  import chdrft.utils.K as K
  K.g_plot_service.plot(vp)
  input()


def test2(ctx):

  #if 0:
  #  d = scene_small()
  #else:
  d = rocket_scene()
  #d = scene_small()
  tg: RigidBodyLink = d.target
  sctx = d.sctx
  vp = tg.cur_mesh.surface_mesh(SurfaceMeshParams()).vispy_data
  print(f'start >> ', tg.world_angular_momentum(), tg.world_angular_momentum2())
  for i in range(2):
    print('FUU ', tg.get_particles(1000000).angular_momentum())

  return
  import chdrft.utils.K as K
  if 0:
    K.g_plot_service.plot(vp)
    input()
    return

  if 1:
    px = tg.get_particles(30000)
    px.plot(by_col=True, fx=0.02)
    input()
    return

  if 0:
    for i in range(10):
      print('FUU ', tg.get_particles(10000000).angular_momentum())
    return

  dt = 1e-2
  nsz = 300000
  for i in range(200):
    print()
    print()
    print()
    print()
    tc = TorqueComputer()
    tc.setup(tg, dt)
    tc.process()
    print(tg.world_angular_momentum(), tg.world_angular_momentum2())
    #tc.compute_wl()
    #tc.update_wl()
    continue

    sx = Vec3.Zero()
    print(sctx.obj2name.values())
    for x in sctx.obj2name.keys():
      rl = x.self_link.root_link
      sx += x.root_angular_momentum()
      #print(f'>>>>>>>> {x.name=} {rl.rotvec_w=} {rl.world_angular_momentum()=}')
    print(f'{i:03d} {i*dt:.2f}', tg.world_angular_momentum(), sx)


def get_control_test_data(nwheel):
  nx = nwheel

  names = 'wh_x wh_y wh_z'.split()
  name2pos = {a: i for i, a in enumerate(names)}

  def create_simulator():
    wh_w_params = np.zeros(nx)
    wh_ang_params = np.zeros(nx)

    d = control_simple(nx, wh_w_params, wh_ang_params)
    sctx: SceneContext = d.sctx

    sx = Simulator(sctx=sctx, name2pos=name2pos)
    return sx

  root_rotvec_l=Vec3.X().vdata * 1
  root_rotvec_l=(Vec3.X() + Vec3.Y() * 2 - Vec3.Z() * 1 / 3).vdata * 1
  #root_rotvec_l=(Vec3.X() + Vec3.Y() * 2 / 3).vdata * 1
  #root_rotvec_l=Vec3.X().vdata + Vec3.Y().vdata
  consts = A(
      state=A(
          root_wl=Transform.From().data,
          root_rotvec_l=root_rotvec_l,
          wh_w=np.identity(nx)[0] * 0.1,
          wh_ang=np.zeros(nx),
      ),
      conds=A(
          f_root_rotvec_w=(Vec3.X() * 0.0).vdata,
          #f_root_rotvec_w=None,
          f_root_wl=Transform.From().data,
      ),
  )
  return A(create_simulator=create_simulator, consts=consts, nctrl=nx)


def test_control(ctx):
  nxx = 3
  px = RB_PbAnalysis(dx=get_control_test_data(nxx))
  px.setup()

  sx: Simulator = px.dx.create_simulator()
  consts = px.dx.consts
  params = SimulParams(
      dt=1e-2,
      ndt_const=3,
      integ_nsteps=1,
      nt=10,
      nctrl=nxx,
      i_state=consts.state,
      f_conds=consts.conds
  )
  #state = px.state_packer.pack(params.i_state)
  #for jk in range(5): state = sx.do_step(params.actual_dt, ([1,1,1],), state)
  #print(sx.dump_state())

  #state = px.state_packer.pack(params.i_state)
  #for jk in range(5): state = sx.do_step(params.actual_dt, ([0,0,0],), state)
  #print(sx.dump_state())
  #return

  #px.integrate(px.args0[0], px.args0[1:], 1)
  print(px.proc(params, params.packer.zeros_packed))
  pb = px.get_pb(params)
  res = px.solve_pb(pb)
  FileFormatHelper.Write('./res2.pickle', A(ans=res.ans, params=params, ctrl=res.ctrl))

  print(pb.fitness(res.raw_ans))
  print(px.proc(params, res.raw_ans))
  print('FUUU', res.ctrl)


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
