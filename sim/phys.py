#!/usr/bin/env python

from typing import Tuple, Optional
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

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class RigidBodyLinkType(Enum):
  RIGID = 1
  MOVE = 2


class Vec3(cmisc.PatchedModel):
  data: np.ndarray = None

  def __init__(self, *args, data=None):
    super().__init__()
    if data is None:
      if len(args) == 1:
        data = args[0]
        if isinstance(data, Vec3):
          data = data.data
        if not cmisc.is_list(data): data = [data, data, data]
      elif len(args) == 0:
        data = [0, 0, 0]
      else:
        data = args

    self.data = np.array(data)

  class Config:
    arbitrary_types_allowed = True

  @staticmethod
  def Natif(v):
    if isinstance(v, Vec3): return v.data
    return v

  @staticmethod
  def Make(v):
    if v is None: return v
    if isinstance(v, Vec3): return v
    return Vec3(v)

  @staticmethod
  def Zero():
    return Vec3(0, 0, 0)

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
  def Rand():
    return Vec3(np.random.rand(3))

  @property
  def skew_matrix(self) -> np.ndarray:
    return np.array(
        [
            [0, -self.data[2], self.data[1]],
            [self.data[2], 0, -self.data[0]],
            [-self.data[1], self.data[0], 0],
        ]
    )

  @property
  def skew_transform(self) -> "Transform":
    return Transform.From(rot=self.skew_matrix)

  def __neg__(a):
    return Vec3(-a.data)

  def __sadd__(a, b):
    self.data += Vec3.Natif(b).data
    return self

  def __add__(a, b):
    return Vec3(a.data + Vec3.Natif(b))

  def __mul__(a, b):
    return Vec3(a.data * Vec3.Natif(b))

  def __sub__(a, b):
    return Vec3(a.data - Vec3.Natif(b))

  def __truediv__(a, b):
    return Vec3(a.data / Vec3.Natif(b))

  def proj(self, vec: "Vec3 | np.ndarray") -> "Vec3":
    vec = Vec3.Make(vec)
    return Vec3.Make(vec * np.dot(self.data, vec.data))

  def orth_proj(self, vec: "Vec3 | np.ndarray") -> "Vec3":
    return self - self.proj(vec)

  @property
  def norm(self) -> float:
    return np.linalg.norm(self.data)

  def make_norm(self) -> "Vec3":
    return self / self.norm

  @property
  def norm_and_uvec(self) -> "Tuple[float, Vec3]":
    n = self.norm
    if n == 0: return n, Vec3.X()
    return n, self / n

  def exp_rot(self) -> "Transform":
    rot, v = self.norm_and_uvec
    w1 = v.skew_matrix
    w2 = w1 @ w1
    res = np.identity(3) + np.sin(rot) * w1 + (1 - np.cos(rot)) * w2
    return Transform.From(rot=res)


class Transform(cmisc.PatchedModel):
  data: np.ndarray

  @property
  def inv(self) -> "Transform":
    return Transform(data=np.linalg.inv(self.data))

  @property
  def pos_v(self) -> Vec3:
    return Vec3(self.pos)

  @property
  def pos(self) -> np.ndarray:
    return self.data[:3, 3]

  @pos.setter
  def pos(self, v):
    self.data[:3, 3] = v

  @pos_v.setter
  def pos_v(self, v):
    self.data[:3, 3] = v.data

  @property
  def rot(self) -> np.ndarray:
    return self.data[:3, :3]

  @property
  def tsf_rot(self):
    return Transform.From(rot=self.rot)

  @property
  def rot_R(self) -> R:
    return R.from_matrix(self.rot)

  @rot.setter
  def rot(self, v):
    self.data[:3, :3] = v

  def apply(self, v: Vec3) -> Vec3:
    return Vec3(self.map(v.data))

  def apply_vec(self, v: Vec3) -> Vec3:
    return Vec3(self.rot @ v.data)

  def map(self, v):
    if len(v) == 0: return np.zeros((0, 3))
    if len(v.shape) == 2:
      return MatHelper.mat_apply_nd(self.data, v.T, point=1).T
    return MatHelper.mat_apply_nd(self.data, v, point=1)

  def imap(self, v):
    return self.inv.map(v)

  def __mul__(self, peer):
    return Transform(data=self.data @ peer.data)

  def __matmul__(self, peer):
    if isinstance(peer, Vec3):
      return self.apply(peer)
    return Transform(data=self.data @ peer.data)

  @staticmethod
  def From(pos=None, rot=None, scale=None, data=None):
    if data is not None:
      return Transform(data=data)
    res = np.identity(4)
    tsf = Transform(data=res)
    pos = Vec3.Make(pos)
    if scale is not None:
      res[:3, :3] *= scale
      scale = None
    elif pos is not None:
      res[:3, 3] = pos.data
      pos = None

    elif rot is not None:
      if isinstance(rot, R):
        rot = rot.as_matrix()
      res[:3, :3] = rot
      rot = None
    else:
      return tsf

    return tsf * Transform.From(pos=pos, rot=rot, scale=scale)

  def clone(self) -> "Transform":
    return Transform.From(data=np.array(self.data))

  class Config:
    arbitrary_types_allowed = True


def make_rot(**kwargs) -> np.ndarray:
  mp = dict(x=0, y=1, z=2)
  vx = None
  tb: list[Tuple[int, Vec3]] = []

  def make_orth(v):
    v = Vec3(v)
    for _, e in tb:
      v = v.orth_proj(e)
    if v.norm < 1e-5: return None
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
      a = make_orth(Vec3.Rand())
      if a is not None:
        tb.append((i, a))
        break

  mat = np.zeros((3, 3))
  for p, v in tb:
    mat[:, p] = v.data
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

  def corner(self, m):
    v = np.array(self.v)
    for i in range(self.n):
      v[i] *= (m >> i & 1)
    return self.p + v

  @property
  def points(self) -> np.ndarray:
    res = []
    if self.empty: return res
    for i in range(2**self.n):
      res.append(self.corner(i))
    return np.array(res)

  @property
  def surface_mesh(self):
    coords = [self.corner(i) for i in range(8)]
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


def rejection_sampling(n, bounds, checker, grp_size=1000):
  res = []
  while len(res) < n:
    cnds = np.random.uniform(low=bounds[0], high=bounds[1], size=(n, len(bounds[0])))
    cnds = cnds[checker(cnds)]
    res.extend(cnds)

  return np.array(res[:n])


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
    return np.sum(np.cross(self.p, self.v) * self.weights.reshape((-1, 1)), axis=0)

  def plot(self, fx=1, by_col=False):
    lines = np.hstack((self.p, self.p + self.v * fx))
    lines.resize((self.n, 2, 3))
    points_color = []
    from chdrft.display.service import g_plot_service, grid
    data = A(points=self.p, lines=lines, conf=A(mode='3D'))
    if by_col:
      from chdrft.utils.colors import ColorMapper
      vals = set(self.id_obj)
      cm = ColorMapper(vals)
      data.points_color = [cm.get(x) for x in self.id_obj]
      data.lines = [A(polyline=x, color=cm.get(id)) for id, x in zip(self.id_obj, data.lines)]
    g_plot_service.plot(data)

  class Config:
    arbitrary_types_allowed = True


def importance_sampling(n, bounds, weight_func, map=None):
  cnds = np.random.uniform(low=bounds[0], high=bounds[1], size=(n, len(bounds[0])))
  weights = weight_func(cnds)
  weights = weights / sum(weights)
  if map is not None: cnds = map(cnds)
  return cnds, weights


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
    return np.zeros((0, 3))

  def importance_sampling0(self, n):
    return np.array([]), np.zeros((0, 3))

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
    res.points = [self.transform.apply(x).data for x in res.points]
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
    return np.array([[0, 0, 0]])

  def importance_sampling0(self, n):
    return np.array([[0, 0, 0]]), np.array([1])

  def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
    return self.bounds.surface_mesh


class UnitConeDesc(MeshDesc):
  # tip up
  @property
  def bounds(self) -> AABB:
    return AABB([-1, -1, -1 / 3], [2, 2, 2 / 3])

  def rejection_sampling0(self, n):
    return rejection_sampling(
        n, (self.bounds.low, self.bounds.high), lambda tb: np.full(tb.shape[0], True)
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
        n, (self.bounds.low, self.bounds.high), lambda tb: (tb[:, 0]**2 + tb[:, 1]**2 <= 1) &
        (np.abs(tb[:, 2]) < 0.5)
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
    zl = np.linspace(-1, 1, 20)
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
        n, (self.bounds.low, self.bounds.high), lambda pts: np.einsum('ij,ij->i', pts, pts) <= 1
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


class MoveDesc(BaseModel):
  v_l: Vec3 = Vec3.Zero()
  rotang_l: Vec3 = Vec3.Zero()
  rotspeed: float = 0

  def load_rotvec(self, rotvec: Vec3):
    self.rotspeed, self.rotang_l = rotvec.norm_and_uvec

  @classmethod
  def From(cls, rotvec: Vec3) -> "MoveDesc":
    res = MoveDesc()
    res.load_rotvec(rotvec)
    return res

  @property
  def rotvec_l(self) -> Vec3:
    return self.rotang_l * self.rotspeed

  @property
  def skew_matrix(self) -> np.ndarray:
    return self.rotvec_l.skew_matrix

  @property
  def skew_transform(self) -> Transform:
    return self.rotvec_l.skew_transform

  class Config:
    arbitrary_types_allowed = True


class InertialTensor(BaseModel):

  data: np.ndarray = Field(default_factory=lambda: np.zeros((3, 3)))

  class Config:
    arbitrary_types_allowed = True

  def shift_inertial_tensor(self, p: Vec3, mass) -> "InertialTensor":
    res = np.array(self.data)
    px, py, pz = p.data
    mx = [
        [py * py + pz * pz, -px * py, -px * pz],
        [-px * py, px * px + pz * pz, -py * pz],
        [-px * pz, -py * pz, py * py + px * px],
    ]

    res += np.array(mx) * mass
    return InertialTensor(data=res)

  def get_world_tensor(self, wl: Transform) -> "InertialTensor":
    m = wl.rot @ self.data @ wl.rot.T
    return InertialTensor(data=m)

  def __add__(self, v) -> "InertialTensor":
    return InertialTensor(data=self.data + v.data)

  def around(self, v) -> float:
    return v @ self.data @ v

  def vel2mom(self, v: Vec3) -> Vec3:
    return Vec3(self.data @ v.data)

  def mom2vel(self, v: Vec3) -> Vec3:
    return Vec3(np.linalg.solve(self.data, v.data))

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


class SolidSpec(cmisc.PatchedModel):
  mass: float = 0
  com: Vec3 = Field(default_factory=Vec3.Zero)
  inertial_tensor: InertialTensor = Field(default_factory=InertialTensor)
  mesh: MeshDesc = Field(default_factory=MeshDesc)

  @staticmethod
  def Point(mass):
    return SolidSpec(
        mass=mass, inertial_tensor=InertialTensor(data=np.zeros((3, 3))), desc=PointMesh()
    )

  @staticmethod
  def Cone(mass, r, h):
    ix = iy = 3 / 5 * h**2 + 3 / 20 * r**2
    iz = 3 / 10 * r**2
    mesh = TransformedMeshDesc(mesh=UnitConeDesc(), transform=Transform.From(scale=[r, r, h]))
    return SolidSpec(
        mass=mass,
        inertial_tensor=InertialTensor(data=np.identity(3) * mass * [ix, iy, iz]),
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
        inertial_tensor=InertialTensor(data=np.diag([ix, ix, iz])),
        mesh=TransformedMeshDesc(
            mesh=UnitCylinderDesc(), transform=Transform.From(scale=[r, r, h / 2])
        ),
    )

  @staticmethod
  def Box(mass, x, y, z):
    c = mass / 12
    mesh = TransformedMeshDesc(mesh=CubeDesc(), transform=Transform.From(scale=[x, y, z]))
    return SolidSpec(
        mass=mass,
        inertial_tensor=InertialTensor(
            data=np.diag([y * y + z * z, x * x + z * z, x * x + y * y]) * c
        ),
        mesh=mesh
    )

  class Config:
    arbitrary_types_allowed = True


class SceneContext(cmisc.PatchedModel):
  cur_id: int = 0
  mp: dict = Field(default_factory=dict)
  obj2name: "dict[RigidBody,str]" = Field(default_factory=dict)
  name2obj: "dict[str,RigidBody]" = Field(default_factory=dict)
  basename2lst: dict[str, list] = Field(default_factory=lambda: cmisc.defaultdict(list))

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
    self.get_name(link.child)

  def create_rb(self, **kwargs) -> "RigidBody":
    return RigidBody(ctx=self, **kwargs)

class LinkData(cmisc.PatchedModel):
  static: bool = True
  pivot_rotaxis: Vec3 = None
  free: bool = False

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert self.static or (self.pivot_rotaxis is not None or self.free)


class RigidBodyLink(cmisc.PatchedModel):
  wl: Transform = Field(default_factory=Transform.From)
  move_desc: MoveDesc = Field(default_factory=MoveDesc)
  link_data: LinkData = Field(default_factory=LinkData)
  rb: "RigidBody"

  parent: "Optional[RigidBodyLink]" = Field(repr=False, default=None)

  def __init__(self, update_link=True, **kwargs):
    super().__init__(**kwargs)
    if update_link: 
      self.rb.self_link = self
      for x in self.rb.move_links:
        x.parent = self

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
  def com(self) -> Vec3:
    return self.wl @ self.rb.com

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
      res.wl = res.wl @ l.wl
      rotvec_w += res.wl_rot @ l.rotvec_w
    res.move_desc.load_rotvec(res.lw_rot @ rotvec_w)
    return res

  @property
  def rotvec_l(self) -> Vec3:
    return self.move_desc.rotvec_l

  @property
  def rotvec_w(self) -> Vec3:
    return self.wl_rot @ self.move_desc.rotvec_l

  @rotvec_w.setter
  def rotvec_w(self, v):
    self.move_desc.load_rotvec(self.lw_rot @ v)

  def world_angular_momentum(self, com: Vec3 = None) -> Vec3:
    if com is None: com = self.com
    return self.world_inertial_tensor(com).vel2mom(self.rotvec_w
                                                  ) + self.wl_rot @ self.rb.local_angular_momentum()

  def world_linear_momentum(self) -> Vec3:
    #res = self.child.local_linear_momentum + self.move_desc.skew_matrix @ self.com.data * self.child.mass
    res = self.rb.local_linear_momentum() + self.v_l * self.mass
    return Vec3(self.wl.rot_R.apply(res.data))

  def local_inertial_tensor(self) -> InertialTensor:
    return self.rb.local_inertial_tensor()

  def world_inertial_tensor(self, com: Vec3) -> InertialTensor:
    return self.local_inertial_tensor().get_world_tensor(
        self.wl
    ).shift_inertial_tensor(com - self.com, self.mass)

  def to_world_particles(self, x: Particles):
    if len(x.p) == 0:
      return x
    x.v = x.v - x.p @ self.move_desc.skew_matrix
    x.v = self.wl.rot_R.apply(x.v) + self.world_vel.data
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


kDefaultName = "noname"


class RBData(cmisc.PatchedModel):
  base_name: str = kDefaultName
  internal: bool = False


class RigidBody(cmisc.PatchedModel):
  spec: SolidSpec = Field(default_factory=SolidSpec)
  data: RBData = Field(default_factory=RBData)
  self_link: Optional[RigidBodyLink] = Field(repr=False, default=None)
  move_links: list[RigidBodyLink] = Field(default_factory=list)

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
        [x.com * x.mass for x in self.move_links], self.spec.com * self.spec.mass
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
        weights=w, p=pts, n=n, v=np.zeros((n, 3)), id_obj=np.ones((n,), dtype=int) * self.idx
    )

  def get_particles(self, n, **kwargs) -> Particles:
    fmass = self.mass
    tb = [self.get_self_particles(int(n * self.spec.mass / fmass), **kwargs)]
    tb.extend([x.get_particles(int(n * x.mass / fmass), **kwargs) for x in self.move_links])
    res = functools.reduce(Particles.reduce, tb)
    return res

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


RigidBodyLink.update_forward_refs()


class RBBuilderEntry(cmisc.PatchedModel):
  spec: SolidSpec = None
  wl: Transform


class RBDescEntry(cmisc.PatchedModel):
  data: RBData = Field(default_factory=RBData)
  spec: SolidSpec = None
  parent: "RBDescEntry" = None
  wl: Transform = Field(default_factory=Transform.From)
  link_data: LinkData = Field(default_factory=LinkData)
  move_desc: MoveDesc = None


class RBBuilderAcc(cmisc.PatchedModel):
  entries: list[RBBuilderEntry] = Field(default_factory=list)
  src_entries: list[RBDescEntry] = Field(default_factory=list)
  dyn_links: list[RigidBodyLink] = Field(default_factory=list)
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
            x.spec.inertial_tensor.shift_inertial_tensor(com - x.spec.com,
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
  entries: list[RBDescEntry] = Field(default_factory=list)
  entry2child: dict[RBDescEntry,
                    list[RBDescEntry]] = Field(default_factory=lambda: cmisc.defaultdict(list))
  child_links: dict[RBDescEntry,
                    list[RigidBodyLink]] = Field(default_factory=lambda: cmisc.defaultdict(list))
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
    rbl = RigidBodyLink(rb=rb, wl=wl, move_desc=cur.move_desc, link_data=cur.link_data)
    if wl_from_com:
      wl.pos_v = wl.pos_v - (wl.tsf_rot @ rbl.rb.com)
    return rbl

  def dfs(self, cur: RBDescEntry, wl: Transform, res: RBBuilderAcc):
    children = self.entry2child[cur]
    static_children = [x for x in children if x.move_desc is None]
    dyn_children = [x for x in children if x.move_desc is not None]

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


def test(ctx):
  m0 = MoveDesc(worldvel=np.array([0, 0, 0]))
  r0 = RigidBody(spec=SolidSpec.Box(1, 1, 1, 1), wl=Transform.From(), move_desc=m0)
  r0.wl.pos += [0, 0, 3]

  r1 = RigidBody(spec=SolidSpec.Sphere(1, 1), wl=Transform.From(pos=[0, 0, -3]), move_desc=m0)
  r2 = RigidBody.Compose([r1, r0])

  r00 = RigidBody(spec=SolidSpec.Box(1, 1, 1, 1), wl=Transform.From(), move_desc=m0)
  r00.wl.pos += [0, 0, 3]

  r11 = RigidBody(spec=SolidSpec.Sphere(1, 1), wl=Transform.From(pos=[0, 0, -3]), move_desc=m0)
  r22 = RigidBody.Compose([r11, r00])
  r22.wl.pos += [100, 0, 0]

  r3 = RigidBody.Compose([r2, r22])

  print([x[1] for x in r0.tree_expl.get_path(r00, r3)])
  print(r0.tree_expl.to_node(r22))


class LinkChainEntry(cmisc.PatchedModel):
  lw_rot: Transform
  rot_w_wl: Vec3


class TorqueComputer:

  def __init__(self):
    self.mp = dict()
    self.pivot_torque = dict()
    self.updates: dict[RigidBody, A] = cmisc.defaultdict(A)
    self.root = None
    self.dt = None

  def setup_(self, root: RigidBodyLink):
    self.updates[root.rb]
    for x in root.rb.move_links:
      self.setup_(x)

  def setup(self, root: RigidBodyLink, dt: float):
    self.root = root
    self.dt = dt
    self.orig_angular_mom = root.world_angular_momentum()
    self.setup_(root)

  def compute_torque(self, link: RigidBodyLink, link_chain: list[LinkChainEntry] = []) -> Vec3:
    link_chain = list(link_chain)
    link_chain.append(LinkChainEntry(lw_rot=link.lw_rot, rot_w_wl=link.rotvec_w))

    lw = Transform.From()
    rot_l_wl = Vec3.Zero()
    for x in link_chain:
      lw = x.lw_rot @ lw
      rot_l_wl = x.lw_rot @ (rot_l_wl + x.rot_w_wl)

    i_l = link.rb.local_inertial_tensor()
    h = lw.inv @ i_l.vel2mom(rot_l_wl)
    d_rot_l_wl = Vec3.Zero()

    cur_lw = Transform.From()
    cur_rot_w_lw = Vec3.Zero()
    for x in reversed(link_chain):
      cur_lw = cur_lw @ x.lw_rot
      cur_rot_w_lw = x.lw_rot.inv @ cur_rot_w_lw
      d_rot_l_wl += cur_lw @ (-cur_rot_w_lw).skew_transform @ x.rot_w_wl
      cur_rot_w_lw += x.rot_w_wl

    d_h = (lw.inv @ rot_l_wl).skew_transform @ h + lw.inv @ i_l.vel2mom(d_rot_l_wl)

    outside_torque = lw @ d_h
    for cl in link.rb.move_links:
      cur = self.compute_torque(cl, link_chain)
      outside_torque += cur
    #print(
    #    'FUU ', link.rb.name, lw.inv @ d_rot_l_wl, cur_rot_w_lw, lw.inv @ i_l.vel2mom(d_rot_l_wl),
    #    outside_torque
    #)

    if not link.link_data.static and link.parent:
      on_pivot = outside_torque.proj(link.link_data.pivot_rotaxis)
      motor_torque = link.link_data.pivot_rotaxis * 0  #random
      self.pivot_torque[link.rb] = on_pivot + motor_torque
      outside_torque -= self.pivot_torque[link.rb]

    self.updates[link.rb].torque = outside_torque
    return link.wl_rot @ -outside_torque

  def compute_rotvec_(self, link: RigidBodyLink):
    base_link = link.rb.self_link
    if link.rb.mass == 0:
      self.updates[link.rb].rotvec = link.rb.self_link.move_desc.rotvec_l
      return

    torque = self.updates[link.rb].torque if link.link_data.free else self.pivot_torque[link.rb]
    rl = link.root_link
    torque_w = rl.wl_rot @ torque
    mom = rl.world_angular_momentum()
    nmom = mom + torque_w * self.dt

    tensor = link.local_inertial_tensor()
    drotvec_l = tensor.mom2vel(torque * self.dt)
    self.updates[link.rb].rotvec = base_link.rotvec_l + drotvec_l
    #print(
    #    f'OLD rotvec {base_link.rb.name=}, {base_link.move_desc.rotvec_l=}, {torque=}, {link.link_data.pivot_rotaxis=}'
    #)

  def compute_wl(self):
    for child, data in self.updates.items():
      base_link = child.self_link
      data.wl = base_link.wl @ (base_link.move_desc.rotvec_l * self.dt).exp_rot()

  def compute_rotvec(self):
    self.compute_torque(self.root)
    for child in self.updates.keys():
      self.compute_rotvec_(child.self_link)

  def update_wl(self):

    def set_wl(e, data):
      e.self_link.wl = data.wl

    self._update(set_wl)

  def update_rotvec(self):
    self._update(lambda e, data: e.self_link.move_desc.load_rotvec(data.rotvec))

  def _update(self, func):
    for child, data in self.updates.items():
      func(child, data)

  def update(self):
    self.compute_rotvec()
    self.compute_wl()
    for child, data in self.updates.items():
      child.self_link.wl = data.wl
      child.self_link.move_desc.load_rotvec(data.rotvec)

    return
    rx = RBCompactor.Compact(self.root)
    actual = self.root.world_angular_momentum()
    err = self.orig_angular_mom - actual
    it = sum([x.world_inertial_tensor() for x in rx], InertialTensor())
    dv = it.mom2vel(err)
    self.root.rotvec_w += dv
    final_am = self.root.world_angular_momentum()
    print(
        '#############', actual, self.orig_angular_mom, err.norm,
        (final_am - self.orig_angular_mom).norm
    )
    print('kkkkk', it.vel2mom(self.root.rotvec_w) - final_am)


class RBCompactor:

  def __init__(self, wl: Transform, rotvec_w: Vec3):
    self.wl = wl
    self.statics: list[Tuple[Transform, RigidBodyLink]] = []
    self.rotvec_w = rotvec_w
    self.res_children: list[RigidBodyLink] = []

  @classmethod
  def Build(cls, link: RigidBodyLink) -> RigidBodyLink:
    return link.child.ctx.compose(cls.Compact(link))

  @classmethod
  def Compact(cls,
              link: RigidBodyLink,
              wl: Transform = None,
              rotvec_w: Vec3 = None) -> list[RigidBodyLink]:
    if wl is None:
      wl = link.wl
      rotvec_w = link.rotvec_w
    return cls(wl, rotvec_w).go(link)

  def go(self, link: RigidBodyLink) -> list[RigidBodyLink]:
    self.dfs(link, Transform.From())
    mass = sum([x.child.spec.mass for wl, x in self.statics], 0)
    com = Vec3.Zero()
    tensor = InertialTensor()
    tsf = self.wl.clone()
    if mass > 0:
      com = sum([wl.pos_v * x.child.spec.mass for wl, x in self.statics], Vec3.Zero()) / mass
      tensor = sum(
          [
              x.child.spec.inertial_tensor.shift_inertial_tensor(com - wl.pos_v, x.child.spec.mass
                                                                ).get_world_tensor(wl)
              for wl, x in self.statics
          ], InertialTensor()
      )
      tsf.pos_v -= com

    spec = SolidSpec(
        mass=mass,
        inertial_tensor=tensor,
    )
    base_name = '|'.join(x.child.name for wl, x in self.statics)
    cur = RigidBodyLink(
        child=RigidBody(spec=spec, ctx=link.child.ctx, base_name=base_name),
        wl=tsf,
        move_desc=MoveDesc.From(self.rotvec_w)
    )
    self.res_children.append(cur)

    return self.res_children

  def dfs(self, link: RigidBodyLink, wl: Transform):
    self.statics.append((wl, link))
    for clink in link.child.links:
      nwl = wl @ clink.wl
      if clink.link_data.static:
        self.dfs(clink, nwl)
      else:
        self.res_children.extend(
            RBCompactor.Compact(clink, nwl, self.rotvec_w + wl.tsf_rot @ clink.rotvec_w)
        )


def define_wheel(sctx, wheel_r, wheel_h):
  ma = MoveDesc(v_l=Vec3([0, 0, 0]), rotang_l=Vec3.Z(), rotspeed=50)
  ral = sctx.compose(
      [],
      rb_data=dict(base_name='Wheel', spec=SolidSpec.Cylinder(2, wheel_r, wheel_h)),
      wl=Transform.From(pos=[0, 0, 0], rot=make_rot(z=Vec3.X())),
      move_desc=ma,
      link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z()),
  )
  rbl = sctx.compose(
      [],
      rb_data=dict(
          base_name='WheelCase',
          spec=SolidSpec.Box(10, 2 * wheel_r + wheel_h, 2 * wheel_r + wheel_h, 2 * wheel_r)
      ),
  )
  wlink = sctx.compose(
      [ral, rbl],
      rb_data=dict(base_name='WheelSys'),
      move_desc=MoveDesc(v_l=Vec3([0, 0, 0]), rotang_l=Vec3.Z(), rotspeed=0),
      wl=Transform.From(pos=[0, 0, 0], rot=R.from_rotvec(Vec3.Y().data * 0)),
      link_data=LinkData(static=True, pivot_rotaxis=Vec3.Z()),
  )
  return wlink


def rocket_scene():
  sctx = SceneContext()

  cylinder_h = 10
  cylinder_r = 1
  cone_h = 2
  wheel_r = 0.5
  wheel_h = 0.3

  cylinder = sctx.compose(
      [],
      rb_data=dict(base_name='cylinder', spec=SolidSpec.Cylinder(10, cylinder_r, cylinder_h)),
  )
  cone = sctx.compose(
      [],
      rb_data=dict(base_name='cone', spec=SolidSpec.Cone(10, cylinder_r, cone_h)),
      wl=Transform.From(pos=[0, 0, cylinder_h / 2 + cone_h * 1 / 3]),
  )

  nw = 4
  wheels = []

  for i in range(nw):
    w = define_wheel(sctx, wheel_r, wheel_h)
    ang = i * 2 * np.pi / nw
    dir = np.array([np.cos(ang), np.sin(ang), 0])
    rot = make_rot(z=dir, x=Vec3.Z().data)
    cur = sctx.compose([w], wl=Transform.From(pos=dir * (cylinder_r + wheel_r), rot=rot))
    wheels.append(cur)

  body = sctx.compose(
      [cylinder, cone],
      rb_data=dict(base_name='body'),
  )
  if 1:
    rocket = sctx.compose(
        [body] + wheels,
        rb_data=dict(base_name='rocket'),
        link_data=LinkData(static=False, free=True),
        move_desc=MoveDesc(v_l=Vec3([0, 0, 0]), rotang_l=Vec3.X(), rotspeed=1),
    )

  else:
    rocket = sctx.compose(
        [wheels[0]],
        rb_data=dict(base_name='rocket'),
        link_data=LinkData(static=False, free=True),
    )
  return A(target=rocket, body=body, sctx=sctx)


def scene_small():
  sctx = SceneContext()

  cylinder_h = 10
  cylinder_r = 1
  cone_h = 2
  wheel_r = 0.5
  wheel_h = 0.3

  cylinder = sctx.compose(
      [],
      rb_data=dict(base_name='cylinder', spec=SolidSpec.Box(10, 1, 1, 1)),
  )
  ma = MoveDesc(v_l=Vec3([0, 0, 0]), rotang_l=Vec3.Z(), rotspeed=1)
  wheel = sctx.compose(
      [],
      rb_data=dict(base_name='Wheel', spec=SolidSpec.Sphere(2, 1)),
      wl=Transform.From(pos=[0, 20, 0]),
      move_desc=ma,
      link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z()),
  )
  target = sctx.compose(
      [cylinder, wheel],
      rb_data=dict(base_name='rocket'),
      link_data=LinkData(static=False, free=True),
      move_desc=MoveDesc(v_l=Vec3([0, 0, 0]), rotang_l=Vec3.X(), rotspeed=1),
  )
  return A(target=target, sctx=sctx)


def scene_test():
  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          move_desc=MoveDesc(v_l=Vec3([0, 0, 0]), rotang_l=Vec3.X(), rotspeed=1),
          link_data=LinkData(static=False, free=True),
      )
  )
  b1 = tx.add(
      RBDescEntry(
          data=RBData(base_name='tb'),
          spec=SolidSpec.Box(1, 1, 1, 1),
          wl=Transform.From(pos=[10, 0, 0]),
          parent=root,
      )
  )
  b2 = tx.add(
      RBDescEntry(
          data=RBData(base_name='tb2'),
          spec=SolidSpec.Box(1, 1, 1, 1),
          wl=Transform.From(pos=[-10, 0, 0]),
          parent=root,
      )
  )
  a = tx.add(
      RBDescEntry(
          data=RBData(base_name='ta'),
          spec=SolidSpec.Box(10, 4, 4, 4),
          wl=Transform.From(pos=[-10, 0, 0]),
          parent=b1,
      )
  )

  wheel_r = 0.5
  wheel_h = 0.3
  wheel = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel'),
          spec=SolidSpec.Cylinder(2, wheel_r, wheel_h),
          wl=Transform.From(rot=make_rot(z=Vec3.X())),
          move_desc=MoveDesc.From(Vec3.Z() * 50),
          link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z()),
      )
  )
  wheel_sys = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel_sys'),
          move_desc=MoveDesc(rotang_l=Vec3.Z(), rotspeed=0),
          wl=Transform.From(pos=[0, 0, 0], rot=R.from_rotvec(Vec3.Y().data * 0)),
          link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z()),
      )
  )
  wheel_case = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel_case'),
          wl=Transform.From(pos=[1, 0, 0], rot=R.from_rotvec(Vec3.Y().data * 0)),
          spec=SolidSpec.Box(10, 2 * wheel_r + wheel_h, 2 * wheel_r + wheel_h, 2 * wheel_r),
          parent=wheel_sys
      )
  )
  wobj = tx.create(wheel, wl=Transform.From(pos=[1, 0, 0]), wl_from_com=True)
  tx.add_link(wheel_sys, wobj)
  wsys = tx.create(
      wheel_sys,
      wl_from_com=1,
  )
  tx.add_link(a, wsys)

  res = tx.create(root, wl_from_com=True)
  return A(target=res, sctx=sctx)


def test3(ctx):
  d = scene_small()
  vp = d.target.cur_mesh.surface_mesh(SurfaceMeshParams()).vispy_data

  import chdrft.utils.K as K
  K.g_plot_service.plot(vp)
  input()


def test2(ctx):

  #d = rocket_scene()
  #d = scene_small()
  d = scene_test()
  tg: RigidBodyLink = d.target
  sctx = d.sctx
  vp = tg.cur_mesh.surface_mesh(SurfaceMeshParams()).vispy_data

  import chdrft.utils.K as K
  if 1:
    K.g_plot_service.plot(vp)
    input()
    return

  if 0:
    px = tg.get_particles(3000)
    px.plot(by_col=True, fx=0.2)
    input()

  print(f'start >> ', tg.world_angular_momentum())
  print('FUU ', tg.get_particles(1000000).angular_momentum())
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
    tc.update()
    for x in sctx.obj2name.keys():
      rl = x.self_link.root_link
      #print(f'>>>>>>>> {x.name=} {rl.rotvec_w=} {rl.world_angular_momentum()=}')
    print(f'{i:03d} {i*dt:.2f}', tg.world_angular_momentum())


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
