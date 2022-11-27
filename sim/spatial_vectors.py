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
from chdrft.sim.rb.base import Vec3, Transform, InertialTensor, g_oph, np_array_like
from enum import Enum

global flags, cache
flags = None
cache = None


class SpatialVector(cmisc.PatchedModel):
  v: Vec3 = Field(default_factory=Vec3.Zero, alias='f')
  w: Vec3 = Field(default_factory=Vec3.Zero, alias='m')
  dual: bool

  def __repr__(self) -> str:
    return str(self)

  def __str__(self) -> str:
    return 'SV' + ('*' if self.dual else '') + f'({self.v.vdata}, {self.w.vdata})'

  @property
  def f(self) -> Vec3:
    return self.v

  @property
  def m(self) -> Vec3:
    return self.w

  @property
  def norm(self) -> float:
    return np.linalg.norm(self.data)

  def like_data(self, data: np.ndarray) -> "SpatialVector":
    return SpatialVector.From(data, dual=self.dual)

  def like(self, **kwargs) -> "SpatialVector":
    return SpatialVector(dual=self.dual, **kwargs)

  def __neg__(a):
    return a.like_data(-a.as_array)

  def __add__(self, peer: "SpatialVector") -> "SpatialVector":
    return self.like(v=self.v + peer.v, w=self.w + peer.w)

  def __sub__(self, peer: "SpatialVector") -> "SpatialVector":
    return self.like(v=self.v - peer.v, w=self.w - peer.w)

  def __mul__(self, k: float) -> "SpatialVector":
    return self.like(v=self.v * k, w=self.w * k)

  def inv(self) -> "SpatialVector":
    return self.like(v=-self.v, w=-self.w)

  @property
  def m4_tsf(self) -> Transform:
    res = self.w.exp_rot()
    res.pos_v = self.v
    return res

  def around(self, dp: Vec3) -> "SpatialVector":
    if self.dual:
      return self.like(f=self.f, m=self.m_at(dp))
    else:
      return self.like(v=self.v_at(dp), w=self.w)

  def v_at(self, dp: Vec3) -> Vec3:
    return self.v + self.w.cross(dp)

  def m_at(self, dp: Vec3) -> Vec3:
    return self.m + self.f.cross(dp)

  def mat_tsf(self, tsf: Transform) -> "SpatialVector":
    return self.like(v=tsf @ self.v, w=tsf @ self.w).around(-tsf.pos_v)

  def apply(self, target: "SpatialVector") -> "SpatialVector":
    if target.dual:
      return target.like(
          f=self.w.skew_transform @ target.f,
          m=self.w.skew_transform @ target.m + self.v.skew_transform @ target.f,
      )
    else:
      return target.like(
          v=self.w.skew_transform @ target.v + self.v.skew_transform @ target.w,
          w=self.w.skew_transform @ target.w
      )

  @property
  def make_tsf(self, dual=False) -> np.ndarray:
    res = g_oph.get_npx(self.data).zeros((6, 6))
    res = g_oph.set(res, self.w.skew_matrix)[:3, :3]
    res = g_oph.set(res, self.w.skew_matrix)[3:, 3:]
    if dual:
      res = g_oph.set(res, self.v.skew_matrix)[3:, :3]
    else:
      res = g_oph.set(res, self.v.skew_matrix)[:3, 3:]

    return res

  @property
  def tsf(self) -> "Transform_SV":
    return Transform_SV(data=self.make_tsf(dual=self.dual), kind=TransformKind.Map(self.dual))

  def to_spatial_momentum(self, m: float, i: InertialTensor) -> "SpatialVector":
    return SpatialVector(dual=1, f=self.v * m, m=i.data @ self.m)

  @property
  def as_array(self) -> np.ndarray:
    return g_oph.concatenate((self.v.vdata, self.w.vdata), axis=0)

  @property
  def data(self) -> np.ndarray:
    return self.as_array

  @classmethod
  def From(cls, data: np.ndarray = None, **kwargs) -> "SpatialVector":
    return SpatialVector(v=Vec3(data[:3]), w=Vec3(data[3:]), **kwargs)

  @classmethod
  def Force(cls, data=None) -> "SpatialVector":
    if data is None: data = np.zeros(6)
    return cls.From(data, dual=True)

  @classmethod
  def Vector(cls, data=None) -> "SpatialVector":
    if data is None: data = np.zeros(6)
    return cls.From(data, dual=0)


class TransformKind(int, Enum):
  FF = 0
  FM = 1
  MF = 2
  MM = 3
  # in the dyadic sense XY. dyadic XY
  # M=motion, F=force
  # XY=12
  # mi => M as right parameter
  # mo => M as left parameter
  @property
  def T(self) -> "TransformKind":
    return TransformKind.From(mo=self.mi, mi=self.mo)

  @property
  def mo(self):
    return not self.m0

  @property
  def fo(self):
    return not self.mo

  @property
  def mi(self):
    return not self.m1

  @property
  def fi(self):
    return not self.mi

  @property
  def m0(self):
    return bool(int(self) & 2)

  @property
  def m1(self):
    return bool(int(self) & 1)

  @property
  def f0(self):
    return not self.m0

  @property
  def f1(self):
    return not self.m1

  @classmethod
  def Combine(cls, a: "TransformKind", b: "TransformKind") -> "TransformKind":
    assert a.mi == b.mo
    return cls.From(mo=a.mo, mi=b.mi)

  @classmethod
  def From(cls, mo: bool, mi: bool) -> "TransformKind":
    return cls(int(mo) * 2 + int(not mi))

  @classmethod
  def Map(cls, mi: bool) -> "TransformKind":
    return cls(int(mo) * 2 + int(not mi))


class Transform_SV(cmisc.PatchedModel):
  data: np_array_like
  kind: TransformKind

  @property
  def T(self) -> "Transform_SV":
    return Transform_SV(data=self.data.T, kind=self.kind.T)

  def apply(self, v: SpatialVector) -> SpatialVector:
    return SpatialVector.From(data=self.data @ v.as_array, dual=self.kind.f0)

  def change_space(self, wl: Transform) -> "Transform_SV":
    winv = wl.inv
    wo = Transform_SV.FromTransform(wl, dual=self.kind.f0).data
    wi = Transform_SV.FromTransform(winv, dual=self.kind.fi).data
    data = wo @ self.data @ wi
    return self.like(data)

  @classmethod
  def FromTransform(cls, wl: Transform, dual: bool) -> "Transform_SV":
    res = g_oph.get_npx(wl.data).zeros((6, 6))
    res = g_oph.set(res, wl.rot)[:3, :3]
    res = g_oph.set(res, wl.rot)[3:, 3:]
    if dual:
      res = g_oph.set(res, wl.pos_v.skew_matrix)[3:, :3]
    else:
      res = g_oph.set(res, wl.pos_v.skew_matrix)[:3, 3:]

    return Transform_SV(data=res, kind=TransformKind.FM if dual else TransformKind.MF)

  @classmethod
  def MakeSpatialTensor(cls, m: float, i: InertialTensor) -> "Transform_SV":
    res = np.zeros((6, 6))
    res[:3, :3] = np.identity(3) * m
    res[3:, 3:] = i.data
    return cls(data=res, kind=TransformKind.FF)

  def derivate(self, v: SpatialVector) -> "Transform_SV":
    # assuming the transform is constant in the body-space with velocity v
    res = self.zero_like()
    res += v.make_tsf(dual=self.kind.f0) @ self
    res -= self @ v.make_tsf(dual=self.f1)
    return res

  def tsf_ab(self, x_ab: SpatialVector) -> "Transform_SV":
    data = self.data
    data = x_ab.inv.make_tsf(dual=self.kind.f0) @ data @ x_ab.make_tsf(dual=self.kind.f1)
    return self.like(data=data)

  def like(self, data) -> "Transform_SV":
    return Transform_SV(data=data, kind=self.kind)

  def zero_like(self, data) -> "Transform_SV":
    return self.like(np.zeros((6, 6)))

  def __add__(self, peer: "Transform_SV") -> "Transform_SV":
    assert self.kind == peer.kind
    return self.like(data=self.data + peer.data)

  def __matmul__(self, peer):
    if isinstance(peer, SpatialVector):
      return self.apply(peer)

    return self.like(data=self.data @ peer.data, kind=TransformKind.Combine(self.kind, peer.kind))

  def d_tsf_ab(x_ab: "Transform_SV", a: SpatialVector, b: SpatialVector) -> "Transform_SV":
    return -a.tsf @ x_ab + x_ab @ b.tsf


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
