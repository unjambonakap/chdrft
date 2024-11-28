#!/usr/bin/env python

import typing
from typing import Optional
from chdrft.cmds import CmdsList
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
from chdrft.utils.cache import Cachable
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.opa_types import *
from pydantic.v1 import Field
from typing import Callable
from enum import Enum
import functools
import jax.numpy as jnp
import jax
from chdrft.sim.rb.base import *
from chdrft.sim.spatial_vectors import SpatialVector, Transform_SV

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


float_or = float | jnp.ndarray


class RigidBodyLinkType(Enum):
  FREE = -1
  RIGID = 0
  PIVOT_Z = 1
  XLT_Z = 2
  FREE_XY = 3
  PIVOT_XYZ = 4


class LinkTypeData(cmisc.PatchedModel):
  dof: int
  util: np.ndarray
  constraint: np.ndarray
  q_size: int = None

  @property
  def qd_size(self) -> int:
    return self.dof

  def size(self, v) -> int:
    return self.qd_size if v else self.q_size

  def mutate(self, **kwargs) -> "LinkTypeData":
    for k, v in kwargs.items():
      setattr(self, k, v)
    return self

  @classmethod
  def Make(cls, dof, util):
    util = tuple(util)
    constraint = tuple(sorted(set(range(6)) - set(util)))
    return LinkTypeData(
        dof=dof,
        util=np.array(util, dtype=int),
        constraint=np.array(constraint),
        q_size=dof,
    )


linktype2data = {
    RigidBodyLinkType.FREE: LinkTypeData.Make(dof=6, util=range(6)).mutate(q_size=7),
    RigidBodyLinkType.RIGID: LinkTypeData.Make(dof=0, util=()),
    RigidBodyLinkType.PIVOT_Z: LinkTypeData.Make(dof=1, util=(5,)),
    RigidBodyLinkType.PIVOT_XYZ: LinkTypeData.Make(dof=3, util=(3, 4, 5)).mutate(q_size=4),
    RigidBodyLinkType.XLT_Z: LinkTypeData.Make(dof=1, util=(2,)),
    RigidBodyLinkType.FREE_XY: LinkTypeData.Make(dof=3, util=(0, 1, 5)),
}


class LinkSpec(cmisc.PatchedModel):
  type: RigidBodyLinkType
  wr: Transform = cmisc.pyd_f(Transform.From)
  rl: Transform = cmisc.pyd_f(Transform.From)

  def is_rigid(self) -> bool:
    return self.type is RigidBodyLinkType.RIGID

  @property
  def data(self) -> LinkTypeData:
    return linktype2data[self.type]

  @property
  def f_util(self) -> np.array:
    res = None
    mp = np.identity(6)
    res = mp[self.data.util]
    return res.T

  @property
  def f_constraint(self) -> np.array:
    res = None
    mp = np.identity(6)
    res = mp[self.data.constraint]
    return res.T


#TODO: quat
class FreeJoint(cmisc.PatchedModel):
  p_w: Vec3 = None
  aa: Vec3 = None

  @property
  def data(self) -> np.ndarray:
    return g_oph.concatenate((self.p_w.vdata, self.aa.vdata, np.ones(1)))

  @property
  def data_aa(self) -> np.ndarray:
    return g_oph.concatenate((self.aa.vdata, np.ones(1)))

  @classmethod
  def From(
      cls,
      data: np.ndarray = None,
      tsf: Transform = None,
      data_pivot: np.ndarray = None
  ) -> "FreeJoint":
    if data_pivot is not None:
      return FreeJoint(p_w=Vec3.Zero(), aa=Vec3(data_pivot[:3]))

    if data is not None:
      return FreeJoint(p_w=Vec3(data[:3]), aa=Vec3(data[3:6]))
    fj = FreeJoint()
    fj.tsf = tsf
    return fj

  @property
  def tsf(self) -> Transform:
    return Transform.From(pos=self.p_w, rot=self.aa.exp_rot_3())

  @tsf.setter
  def tsf(self, tsf: Transform):
    self.p_w = tsf.pos_v
    self.aa = tsf.get_axis_angle()

  def integrate(self, dp_w: Vec3, dw_w: Vec3):
    self.p_w = self.p_w + dp_w
    self.aa = get_axis_angle(dw_w.exp_rot_3() @ self.aa.exp_rot_3())

  def integrate_aa(self, dw_w: Vec3):
    self.integrate(Vec3.Zero(), dw_w)


class JointSV(cmisc.PatchedModel):
  spec: LinkSpec = None
  q: np_array_like
  is_v: bool = False

  @property
  def type(self) -> RigidBodyLinkType:
    return self.spec.type

  def load(self, q: np.ndarray, scale: float=None) -> None:
    self.q = q
    if scale is not None:
      if self.type is RigidBodyLinkType.FREE:
        self.q[:3] *= scale
      elif self.type is RigidBodyLinkType.RIGID:
        pass
      else:
        raise NotImplemented('nope')


  def dump(self) -> np.ndarray:
    return self.q

  @property
  def v_free(self) -> SpatialVector:
    return SpatialVector.Vector(self.q)

  @v_free.setter
  def v_free(self, sv: SpatialVector):
    self.q[:6] = sv.data

  @property
  def v_free_pivot(self) -> SpatialVector:
    return SpatialVector.Vector(w=Vec3(self.q))

  @property
  def v_xy(self) -> SpatialVector:
    return SpatialVector(
        dual=0, v=Vec3.X() * self.q[0] + Vec3.Y() * self.q[1], w=Vec3.Z() * self.q[2]
    )

  @v_xy.setter
  def v_xy(self, v: SpatialVector):
    data = v.data
    self.q = g_oph.array([data[0], data[1], data[5]])

  @property
  def v_pivot_rotspeed(self) -> float:
    return self.q[0]

  @property
  def v_xlt(self) -> float:
    return self.q[0]

  def default_data(self) -> np.ndarray:
    return np.zeros(self.spec.data.size(self.is_v))

  def v_rl(self, link_data: "LinkData") -> SpatialVector:
    res = None
    if self.type is RigidBodyLinkType.FREE:
      tsf = link_data.q_joint.r_lw.tsf_rot
      # acc/vel is in world coords
      vf = self.v_free
      return SpatialVector(dual=0, v=tsf @ vf.v, w=tsf @ vf.w)
    elif self.type is RigidBodyLinkType.FREE_XY:
      tsf = link_data.q_joint.r_lw.tsf_rot
      vf = self.v_xy
      return SpatialVector(dual=0, v=tsf @ vf.v, w=tsf @ vf.w)
    elif self.type is RigidBodyLinkType.RIGID:
      res = SpatialVector.Vector()
    elif self.type is RigidBodyLinkType.PIVOT_Z:
      res = SpatialVector(w=Vec3.Z() * self.v_pivot_rotspeed, dual=False)
    elif self.type is RigidBodyLinkType.PIVOT_XYZ:
      tsf = link_data.q_joint.r_lw.rot
      # acc/vel is in world coords
      res = SpatialVector(dual=0, w=Vec3(tsf @ self.q))
    elif self.type is RigidBodyLinkType.XLT_Z:
      res = SpatialVector(v=Vec3.Z() * self.v_xlt, dual=False)
    else:
      assert 0

    return res

  def integrate_q(self, link_data: "LinkData", dq: np.ndarray) -> Vec3:
    if self.type is RigidBodyLinkType.FREE:
      fj = self.free_joint
      prev = fj.p_w
      qdj = link_data.make_qd_j(dq)
      vf = qdj.v_free
      fj.integrate(vf.v, vf.w)
      self.q = fj.data
      return fj.p_w - prev

    elif self.type is RigidBodyLinkType.FREE_XY:
      self.q += dq
      return Vec3.Vec([dq[0], dq[1], 0])
    elif self.type is RigidBodyLinkType.PIVOT_XYZ:
      fj = self.free_joint_pivot
      fj.integrate_aa(Vec3(dq))
      self.q = fj.data_aa
    else:
      self.q = self.q + dq
    return Vec3.Zero()

  def integrate_qd(self, link_data: "LinkData", dq: np.ndarray, dp: Vec3):
    self.q = self.q + dq
    if self.type is RigidBodyLinkType.FREE:
      self.q = self.v_free.around(dp).data
    elif self.type is RigidBodyLinkType.FREE_XY:
      self.v_xy = self.v_xy.around(dp)

  @property
  def free_joint(self) -> FreeJoint:
    return FreeJoint.From(self.q)

  @free_joint.setter
  def free_joint(self, v: FreeJoint) -> FreeJoint:
    self.q = v.data

  @property
  def free_joint_pivot(self) -> FreeJoint:
    return FreeJoint.From(data_pivot=self.q)

  @property
  def free2tsf(self) -> Transform:
    return self.free_joint.tsf

  @property
  def free_pivot2tsf(self) -> Transform:
    return self.free_joint_pivot.tsf

  @property
  def r_lw(self) -> Transform:
    return self.r_wl.inv

  @property
  def r_wl(self) -> Transform:
    if self.type is RigidBodyLinkType.FREE:
      res = self.free2tsf
    elif self.type is RigidBodyLinkType.FREE_XY:
      v_xy = self.v_xy
      res = Transform.From(pos=v_xy.v, rot=v_xy.w.exp_rot_3())
    elif self.type is RigidBodyLinkType.RIGID:
      res = Transform.From()
    elif self.type is RigidBodyLinkType.PIVOT_Z:
      res = Vec3.Z().exp_rot_u(self.q[0])
    elif self.type is RigidBodyLinkType.PIVOT_XYZ:
      res = self.free_pivot2tsf
    elif self.type is RigidBodyLinkType.XLT_Z:
      res = Transform.From(pos=Vec3.Z() * self.q[0])
    else:
      assert 0, self.type
    return res


class LinkData(cmisc.PatchedModel):
  spec: LinkSpec

  qd_joint: JointSV = None
  q_joint: JointSV = None

  def v_rl(self, qd_joint: JointSV = None) -> SpatialVector:
    qd_joint = qd_joint or self.qd_joint
    return qd_joint.v_rl(self)

  def integrate(self, dt: float, qd: np.ndarray, qdd: np.ndarray):
    dp = self.q_joint.integrate_q(self, qd * dt)
    self.qd_joint.integrate_qd(self, qdd * dt, dp)

  def integrate_dd(self, dt: float, qdd: np.ndarray):
    self.integrate(dt, self.qd_joint.q.copy(), qdd)

  def make_qd_j(self, data=None) -> JointSV:
    if data is None:
      data = np.zeros(self.spec.data.qd_size)
    return JointSV(spec=self.spec, q=data)

  def get_joint(self, v=0) -> JointSV:
    return self.qd_joint if v else self.q_joint

  def load_joint(self, data: np.ndarray, v=0, scale: float=None) -> None:
    self.get_joint(v).load(data, scale=scale)

  def dump_joint(self, v=0) -> np.ndarray:
    return self.get_joint(v).dump()

  @property
  def v_com_l(self) -> SpatialVector:
    return self.lr @ self.v_rl()

  @property
  def wl(self) -> Transform:
    return self.wr @ self.spec.rl

  @property
  def rw(self) -> Transform:
    return self.wr.inv

  @property
  def lr(self) -> Transform:
    return self.rl.inv

  @property
  def wr(self) -> Transform:
    return self.spec.wr @ self.q_joint.r_wl

  @property
  def rl(self) -> Transform:
    return self.spec.rl

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    if self.qd_joint is None:
      self.qd_joint = JointSV(q=np.zeros(self.spec.data.qd_size))
    if self.q_joint is None:
      self.q_joint = JointSV(q=np.zeros(self.spec.data.q_size))
    self.q_joint.spec = self.spec
    self.qd_joint.spec = self.spec
    self.qd_joint.is_v = True


class RigidBodyLink(cmisc.PatchedModel):
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
  def descendants(self) -> "list[RigidBodyLink]":
    "with self"
    res = [self]
    for x in self.children:
      res.extend(x.descendants)
    return res

  @property
  def wl(self) -> Transform:
    return self.link_data.wl

  @property
  def wr(self) -> Transform:
    return self.link_data.wr

  @property
  def rw(self) -> Transform:
    return self.wr.inv

  @property
  def lr(self) -> Transform:
    return self.rl.inv

  @property
  def rl(self) -> Transform:
    return self.link_data.rl

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
    return self.wl @ self.com_l

  @property
  def wl(self) -> Transform:
    return self.link_data.wl

  @property
  def lw(self) -> Transform:
    return self.wl.inv

  @property
  def com_l(self) -> Vec3:
    return self.rb.com

  def aabb(self, only_self=False) -> AABB:
    return self.wl @ self.rb.aabb(only_self=only_self)

  def get_particles(self, n, **kwargs) -> Particles:
    return self.to_world_particles(self.rb.get_particles(n, **kwargs))

  def to_world_particles(self, x: Particles):
    if len(x.p) == 0:
      return x

    v_com_l = self.link_data.v_com_l
    cp = np.cross(v_com_l.w.vdata, (norm_pts(x.p) - norm_pts(self.com_l.data))[:, :3])
    x.v = self.wl.map(x.v + v_com_l.v.data + as_pts4(cp))
    x.p = self.wl.map(x.p)
    return x

  def compute_momentum(self, wl: Transform, v_par: SpatialVector) -> SpatialVector:
    v_l = self.lr @ (self.rw @ v_par + self.link_data.v_rl())
    m_l = self.rb.local_inertial_tensor @ v_l
    wl = wl @ self.wl
    m_h = wl @ m_l
    for child in self.children:
      m_h += child.compute_momentum(wl, v_l)
    return m_h

  @property
  def agg_mass(self) -> float:
    return sum([x.agg_mass for x in self.children], self.mass)

  @property
  def agg_com(self) -> Vec3:
    # not optimized
    tot = self.agg_mass
    res = Vec3.LinearComb(
        [(x.agg_com, x.agg_mass / tot) for x in self.children] + [(self.rb.com, self.mass / tot)]
    )
    res = self.wl @ res
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

  ctx: "SceneContext" = Field(repr=False)

  @property
  def parent(self) -> "RigidBody | None":
    if not self.self_link.parent: return None
    return self.self_link.parent.rb

  def aabb(self, only_self=False) -> AABB:
    pts = []
    pts.extend(self.spec.mesh.bounds.points)
    if not only_self:
      for x in self.move_links:
        pts.extend(x.aabb().points)

    return AABB.FromPoints(np.array(pts))

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
    return self.spec.inertial.mass

  @property
  def com(self) -> Vec3:
    return self.spec.com

  @property
  def local_inertial_tensor(self) -> Transform_SV:
    return Transform_SV.MakeSpatialTensor(self.spec.inertial.shift(-self.com.as_vec))

  def get_self_particles(
      self, n, use_importance=False, particles_filter=lambda _: True
  ) -> Particles:
    if not particles_filter(self): return Particles.Default()
    if use_importance:
      pts, w = self.spec.mesh.importance_sampling(n, self.mass)
      n = len(pts)
    else:
      pts = self.spec.mesh.rejection_sampling(n)
      n = len(pts)
      w = np.ones((n,)) * (self.mass / max(n, 1))

    return Particles(
        weights=w, p=pts, n=n, v=np.zeros((n, 4)), id_obj=np.ones((n,), dtype=int) * self.idx
    )

  def get_particles(self, n, **kwargs) -> Particles:
    fmass = self.self_link.agg_mass
    tb = [self.get_self_particles(int(n * self.mass / fmass), **kwargs)]
    tb.extend([x.get_particles(int(n * x.agg_mass / fmass), **kwargs) for x in self.move_links])
    res = functools.reduce(Particles.reduce, tb)
    return res

  @property
  def cur_mesh(self) -> MeshDesc:
    return ComposedMeshDesc(
        meshes=[self.spec.mesh] + [x.cur_mesh for x in self.move_links],
        weights=[self.mass] + [x.mass for x in self.move_links]
    )

  def plot(self, npoints=10000):
    from chdrft.display.service import oplt
    from chdrft.display.vispy_utils import get_colormap
    pts, weights = self.spec.mesh.importance_sampling(npoints)
    viridis = get_colormap('viridis')
    cols = viridis.map(weights / max(weights))
    oplt.plot(A(points=pts[:, :3], conf=A(mode='3D'), points_color=cols))


class Slice(cmisc.PatchedModel):
  pos: int
  size: int

  @property
  def endpos(self) -> int:
    return self.pos + self.size

  @property
  def slice(self) -> slice:
    return slice(self.pos, self.endpos)


class SlicedNumpyArrayEntry(cmisc.PatchedModel):
  slice: Slice
  item: object
  default: np.ndarray


class SliceNumpyArrayDesc(cmisc.PatchedModel):

  tb: list[SlicedNumpyArrayEntry] = cmisc.pyd_f(list)
  mp: dict[object, SlicedNumpyArrayEntry] = cmisc.pyd_f(dict)
  pos: int = 0
  dim: int

  @property
  def default(self) -> np.ndarray:
    if self.dim == 0:
      res = np.zeros((self.pos,))
      for x in self.tb:
        res[x.slice.slice] = x.default
      return res
    assert 0
    return np.zeros((self.pos, self.dim))

  def add(self, item, size: int, default=None):
    if default is None:
      default = np.zeros(size)
    x = SlicedNumpyArrayEntry(item=item, slice=Slice(pos=self.pos, size=size), default=default)
    self.tb.append(x)
    self.mp[item] = x
    self.pos += x.slice.size

  @property
  def size(self) -> int:
    return self.pos

  def unpack(self, data: np.ndarray) -> dict[object, np.ndarray]:
    return {x.item: data[x.slice.slice] for x in self.tb}

  def pack(self, a: dict) -> np.ndarray:
    lst = []
    for x in self.tb:
      lst.append(a.get(x.item, x.default))
    for k in a.keys():
      assert k in self.mp, (k, self.mp.keys())
    return g_oph.concatenate(lst, axis=0)


class ControlInputState(cmisc.PatchedModel):
  q: np_array_like
  qd: np_array_like


class ControlInputEndConds(cmisc.PatchedModel):
  q: np_array_like = None
  qd: np_array_like = None
  q_weights: np_array_like | float = 1
  qd_weights: np_array_like | float = 1
  only_end: bool = False


class ControlInput(cmisc.PatchedModel):
  state: ControlInputState
  end_conds: ControlInputEndConds = None


class RBSystemSpec(cmisc.PatchedModel):
  root: RigidBodyLink
  q_desc: SliceNumpyArrayDesc
  qd_desc: SliceNumpyArrayDesc
  f_desc: SliceNumpyArrayDesc
  f_full_desc: SliceNumpyArrayDesc

  state_packer: NumpyPacker = cmisc.pyd_f(NumpyPacker)
  d_packer: NumpyPacker = cmisc.pyd_f(NumpyPacker)
  ctrl_packer: NumpyPacker = cmisc.pyd_f(NumpyPacker)
  name2rbl: dict[str, RigidBodyLink]

  def load_state(self, root: RigidBodyLink, dx: ControlInputState | np.ndarray = None, scale: float = None) -> None:
    if not isinstance(dx, (A, ControlInputState)): dx = self.state_packer.unpack(dx)
    self.load_vp(root, dx.q, v=0, scale=scale)
    self.load_vp(root, dx.qd, v=1, scale=scale)

  def dump_state(self, root: RigidBodyLink, to_struct=False) -> np.ndarray | ControlInputState:
    dx = A(q=self.dump_vp(root, v=0), qd=self.dump_vp(root, v=1))
    if to_struct: return dx
    return self.state_packer.pack(dx)

  def get_qx_desc(self, v) -> SliceNumpyArrayDesc:
    return self.qd_desc if v else self.q_desc

  def load_vp(self, root: RigidBodyLink, q: np.ndarray, v=0, scale: float=None) -> None:

    mp = self.get_qx_desc(v).unpack(q)
    for rbl in root.descendants:
      rbl.link_data.load_joint(mp[rbl.rb.name], v=v, scale=scale)

  def dump_vp(self, root: RigidBodyLink, v=0) -> None:
    d = self.get_qx_desc(v)
    data = {rbl.rb.name: rbl.link_data.dump_joint(v=v) for rbl in root.descendants}
    return d.pack(data)
    return g_oph.concatenate(lst, axis=0)

  @property
  def f_util_base(self) -> np.ndarray:
    res = np.zeros((self.qd_desc.size, self.f_full_desc.size))
    for x in self.root.descendants:
      name = x.rb.name
      res[self.qd_desc.mp[name].slice.slice,
          self.f_full_desc.mp[name].slice.slice] = x.link_data.spec.f_util.T
    return res

  @classmethod
  def Build(cls, root: RigidBodyLink):
    q_desc = SliceNumpyArrayDesc(dim=0)
    qd_desc = SliceNumpyArrayDesc(dim=0)
    f_desc = SliceNumpyArrayDesc(dim=0)
    f_full_desc = SliceNumpyArrayDesc(dim=0)
    items = root.descendants
    for x in root.descendants:
      q_desc.add(
          x.rb.name, x.link_data.spec.data.q_size, default=x.link_data.q_joint.default_data()
      )
      qd_desc.add(x.rb.name, x.link_data.spec.data.qd_size)
      f_desc.add(x.rb.name, x.link_data.spec.data.qd_size)
      f_full_desc.add(x.rb.name, 6)

    res = cls(
        root=root,
        q_desc=q_desc,
        qd_desc=qd_desc,
        f_desc=f_desc,
        f_full_desc=f_full_desc,
        name2rbl={rbl.rb.name: rbl for rbl in root.descendants},
    )

    res.d_packer.add_dict(qd=qd_desc.pos)
    res.d_packer.add_dict(qdd=qd_desc.pos)

    res.state_packer.add_dict(q=q_desc.pos)
    res.state_packer.add_dict(qd=qd_desc.pos)
    res.ctrl_packer.add_dict(q=f_desc.pos)

    return res


class SceneContext(cmisc.PatchedModel):
  cur_id: int = 0
  mp: dict = cmisc.pyd_f(dict)
  obj2name: "dict[RigidBody,str]" = cmisc.pyd_f(dict)
  name2obj: "dict[str,RigidBody]" = cmisc.pyd_f(dict)
  basename2lst: dict[str, list] = cmisc.pyd_f(lambda: cmisc.defaultdict(list))

  @property
  def roots(self) -> "list[RigidBody]":
    return [x for x in self.obj2name.keys() if x.self_link.parent is None]

  def single(self, name: str) -> "RigidBody":
    tb = self.basename2lst[name]
    assert len(tb) == 1, self.basename2lst.keys()
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

  @cmisc.cached_property
  def sys_spec(self) -> RBSystemSpec:
    assert len(self.roots) == 1
    return RBSystemSpec.Build(self.roots[0].self_link)


class RBState(cmisc.PatchedModel):
  x: RigidBodyLink
  wl: Transform
  a: SpatialVector
  v: SpatialVector


def bias_force_0(s: RBState) -> SpatialVector:
  return SpatialVector.Force()


class ForceModel(cmisc.PatchedModel):
  nctrl_f: Callable[[int], int]
  bias_force_f: Callable[[RBState], SpatialVector] = cmisc.pyd_f(lambda: bias_force_0)
  ctrl2model: Callable[["Simulator", np_array_like], np_array_like]
  model2ctrl: Callable[["Simulator", np_array_like], np_array_like]

  @classmethod
  def Id(cls) -> "ForceModel":
    id = lambda x: x
    id1 = lambda sim, x: x
    return ForceModel(nctrl_f=id, ctrl2model=id1, model2ctrl=id1)


class RBBuilderEntry(cmisc.PatchedModel):
  spec: SolidSpec = None
  wl: Transform


class RBDescEntry(cmisc.PatchedModel):
  data: RBData = cmisc.pyd_f(RBData)
  spec: SolidSpec = None
  parent: "RBDescEntry" = None
  link_data: LinkData = cmisc.pyd_f(LinkData)


class NameRegister(cmisc.PatchedModel):
  obj2name: dict[typing.Any, str] = cmisc.pyd_f(dict)
  name2cnt: dict[str, int] = cmisc.pyd_f(lambda: cmisc.defaultdict(int))

  def register(self, obj, proposal: str) -> str:
    num = self.name2cnt[proposal]
    self.name2cnt[proposal] += 1
    if num:
      proposal = f'{proposal}_{num:03d}'
      assert proposal not in self.name2cnt
    self.obj2name[obj] = proposal
    return proposal


class RBBuilderAcc(cmisc.PatchedModel):
  entries: list[RBBuilderEntry] = cmisc.pyd_f(list)
  src_entries: list[RBDescEntry] = cmisc.pyd_f(list)
  dyn_links: list[RigidBodyLink] = cmisc.pyd_f(list)
  sctx: SceneContext

  def build(self, link_data: LinkData, wl: Transform = None, center_com=True):
    l = self.entries
    weights = [x.spec.inertial.mass for x in l]
    mass = sum(weights)

    com = Vec3.LinearComb([(x.wl @ x.spec.com, x.spec.inertial.mass / mass) for x in l])
    ocom = com
    if center_com:
      for x in l:
        x.wl = Transform.From(pos=-com) @ x.wl
      com = Vec3.LinearComb([(x.wl @ x.spec.com, x.spec.inertial.mass / mass) for x in l])
      #com = Vec3.ZeroPt()

    if len(l) == 1 and l[0].wl.is_id:
      mesh = l[0].spec.mesh
    else:
      mesh = ComposedMeshDesc(
          weights=weights,
          meshes=[TransformedMeshDesc(mesh=x.spec.mesh, transform=x.wl) for x in l],
      )

    it = sum(
        [x.spec.inertial.shift(com - x.wl @ x.spec.com).get_world_tensor(x.wl) for x in l],
        Inertial()
    )
    assert it.mass == mass
    spec = SolidSpec(com=com, mesh=mesh, inertial=it)
    if len(l) == 1:
      spec.type = l[0].spec.type

    assert not spec.com.vec
    names = [x.data.base_name for x in self.src_entries if x.data.base_name != kDefaultName]
    agg_name = kDefaultName
    if len(names) == 1:
      agg_name = names[0]
    else:
      agg_name = f'Agg({",".join(names)})'

    rb = RigidBody(
        spec=spec,
        move_links=self.dyn_links,
        ctx=self.sctx,
        data=RBData(base_name=agg_name),
    )
    rbl = RigidBodyLink(rb=rb, link_data=link_data)
    if wl is not None: rbl.link_data.spec.wr = wl
    if link_data.spec.type == RigidBodyLinkType.FREE:
      rbl.link_data.spec.wr = rbl.link_data.spec.wr @ Transform.From(pos=ocom - com)
    else:
      rbl.link_data.spec.rl = rbl.link_data.spec.rl @ Transform.From(pos=ocom - com)
    return rbl


class RBTree(cmisc.PatchedModel):
  entries: list[RBDescEntry] = cmisc.pyd_f(list)
  entry2child: dict[RBDescEntry,
                    list[RBDescEntry]] = cmisc.pyd_f(lambda: cmisc.defaultdict(list))
  child_links: dict[RBDescEntry,
                    list[RigidBodyLink]] = cmisc.pyd_f(lambda: cmisc.defaultdict(list))
  sctx: SceneContext
  split_rigid: bool = False

  def add(self, entry: RBDescEntry) -> RBDescEntry:
    self.entries.append(entry)
    self.entry2child[entry.parent].append(entry)
    return entry

  def add_link(self, cur: RBDescEntry, child: RigidBodyLink):
    # TODO: clean this up - child_links / entry2child one is redundant?
    self.child_links[cur].append(child)

  def create(self, cur: RBDescEntry, wl=None, **kwargs) -> RigidBodyLink:
    res = RBBuilderAcc(sctx=self.sctx)
    self.dfs(cur, Transform.From(), res)
    rbl = res.build(link_data=cur.link_data, wl=wl, **kwargs)
    self.sctx.register_link(rbl)
    return rbl

  def dfs(self, cur: RBDescEntry, wl: Transform, res: RBBuilderAcc):
    children = self.entry2child[cur]
    static_children = [x for x in children if x.link_data.spec.is_rigid() and not self.split_rigid]
    dyn_children = [x for x in children if not x.link_data.spec.is_rigid() or self.split_rigid]

    for lnk in self.child_links[cur]:
      lnk.link_data.spec.wr = wl @ lnk.link_data.spec.wr
    links = [self.create(x, wl @ x.link_data.spec.wr) for x in dyn_children] + self.child_links[cur]
    res.dyn_links.extend(links)
    if cur.spec: res.entries.append(RBBuilderEntry(wl=wl, spec=cur.spec))
    res.src_entries.append(cur)

    for x in static_children:
      self.dfs(x, wl @ x.link_data.wl, res)

  def local_name(self, cur: RBDescEntry) -> str:
    #TODO: cache this?
    if (par := cur.parent) is None: return 'root'
    children = self.entry2child[par]

    names = NameRegister()
    for entry in children:
      cnd = entry.data.base_name
      if cnd == kDefaultName:
        cnd = f'{entry.link_data.spec.type}_{entry.spec.type}'
      names.register(entry, cnd)

    return names.obj2name[cur]

  def path_name(self, cur: RBDescEntry) -> str:
    name = self.local_name(cur)
    if cur.parent is None: return name
    return f'{self.path_name(cur.parent)}.{name}'

  def build(self):
    links = []
    for x in self.entry2child[None]:
      links.append(self.create(x))
    return links


class SceneState(cmisc.PatchedModel):
  t: float
  root2data: dict[str, np.ndarray] = cmisc.pyd_f(dict)


class SceneData(cmisc.PatchedModel):
  sctx: SceneContext
  fm: ForceModel
  tree: RBTree | None
  idx: str = None


class SceneSaver(cmisc.PatchedModel):
  sd: SceneData
  states: list[SceneState] = cmisc.pyd_f(list)

  def push_state(self, t):
    res = SceneState(t=t)
    for x in self.sd.sctx.roots:
      res.root2data[x.name] = self.sd.sctx.sys_spec.dump_state(x.self_link, to_struct=False)
    self.states.append(res)

  def dump_obj(self):
    assert self.sd.idx is not None
    return A(idx=self.sd.idx, states=self.states)

  @classmethod
  def Load(cls, load_obj):
    sd = SceneRegistar.Registar.get(load_obj.idx)()
    return SceneSaver(sd=sd, states=load_obj.states)


RigidBody.update_forward_refs()
RigidBodyLink.update_forward_refs()


class InverseDynamicsData(cmisc.PatchedModel):
  delta_a_map: dict[str, JointSV] = cmisc.pyd_f(dict)
  q0: np_array_like
  sd: SceneData

  @property
  def sys_spec(self) -> RBSystemSpec:
    return self.sd.sctx.sys_spec

  @classmethod
  def Make(cls, sd: SceneData, q: np.ndarray) -> "InverseDynamicsData":
    res = InverseDynamicsData(sd=sd, q0=q)
    for x in res.sys_spec.qd_desc.tb:
      rbl: RigidBodyLink = res.sys_spec.name2rbl[x.item]
      res.delta_a_map[x.item] = rbl.link_data.make_qd_j(q[x.slice.slice])
    return res


class InverseDynamicsOutput(cmisc.PatchedModel):
  f_map: dict[str, SpatialVector] = cmisc.pyd_f(dict)
  sys_spec: RBSystemSpec

  def pack(self) -> np.ndarray:
    res = self.sys_spec.f_full_desc.default
    for name, v in self.f_map.items():
      x = self.sys_spec.f_full_desc.mp[name]
      res = g_oph.set(res, v.as_array)[x.slice.slice]
    return res


class InverseDynamicRecurrence(cmisc.PatchedModel):
  input: InverseDynamicsData
  output: InverseDynamicsOutput

  def process(self, state: RBState) -> SpatialVector:
    x = state.x
    lw = x.lw

    j_lw = x.link_data.q_joint.r_lw
    ov = x.link_data.v_rl()
    tv = x.rw @ state.v + ov
    ta = x.rw @ state.a + tv.apply(ov) + x.link_data.v_rl(self.input.delta_a_map[x.rb.name])
    i = x.rb.local_inertial_tensor.change_space(wl=x.rl)

    ns = RBState(x=x, wl=state.wl @ x.wl, v=x.lr @ tv, a=x.lr @ ta)
    la = x.link_data.v_rl(self.input.delta_a_map[x.rb.name])
    wtsf = (state.wl @ x.wr).tsf_rot

    tot_force = SpatialVector.Force()
    bias = self.input.sd.fm.bias_force_f(ns)
    tot_force += x.rl @ bias
    child_forces = SpatialVector.Force()

    for c in x.children:
      ns.x = c
      child_forces += self.process(ns)
    child_forces = x.rl @ child_forces

    wr = state.wl @ x.wr

    tot_force += child_forces
    dp = i @ ta + tv.apply(i @ tv)
    frem = dp - tot_force
    #print('tv', tv)
    #print(tv.apply(i@tv))
    #print('frem', frem)
    #print(x.link_data.q_joint.r_wl )
    #print(x.link_data.v_rl(self.input.delta_a_map[x.rb.name]))
    #print(i @ ta)
    #print(x.wl)
    #print()

    self.output.f_map[x.rb.name] = frem
    frem = x.link_data.wr @ frem
    #print('>>> ', frem)

    return -frem


def inverse_dynamics(data: InverseDynamicsData, root: RigidBodyLink) -> InverseDynamicsOutput:
  rec = InverseDynamicRecurrence(
      input=data,
      output=InverseDynamicsOutput(sys_spec=data.sys_spec),
  )
  rec.process(
      RBState(
          x=root,
          a=SpatialVector.Vector(),
          v=SpatialVector.Vector(),
          wl=Transform.From(),
      )
  )
  return rec.output


class ForwardDynamicsData(cmisc.PatchedModel):
  fq: np_array_like


class ForwardDynamicsOutput(cmisc.PatchedModel):
  sd: np_array_like


def forward_dynamics(data: ForwardDynamicsData, sd: SceneData) -> ForwardDynamicsOutput:
  sctx = sd.sctx
  root = sctx.roots[0].self_link
  ss = sctx.sys_spec

  na = ss.qd_desc.size
  fmap = ss.f_desc.unpack(data.fq)
  idd = inverse_dynamics(InverseDynamicsData.Make(sd, ss.qd_desc.default), root).pack()
  c = ss.f_util_base @ idd

  hl = []
  for col in np.identity(na):
    e = ss.f_util_base @ inverse_dynamics(InverseDynamicsData.Make(sd, col), root).pack()
    hx = e - c
    hl.append(hx)
  h = g_oph.array(hl).T

  res = g_oph.solve(h, data.fq - c)

  if 0:
    print('CHEEECK')
    check = inverse_dynamics(InverseDynamicsData.Make(sd, res), root).pack()
    print(data.fq)
    print(check)
  #print(data.fq, c)
  #print(h)
  #print('DET >> ', g_oph.det(h))
  #print('FORWARD >> ', sd)
  return ForwardDynamicsOutput(sd=res)


def jax_forctrl(f_u_s, u, s):

  def f(i, us):
    u, s = us
    return u, f_u_s(u[i], s)

  return jax.lax.fori_loop(0, len(u), f, (u, s))[1]


class Simulator(cmisc.PatchedModel):

  root: RigidBody = None
  sd: SceneData

  @property
  def sctx(self) -> SceneContext:
    return self.sd.sctx

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.root = self.sctx.roots[0]

  @property
  def rootl(self) -> RigidBodyLink:
    return self.root.self_link

  @property
  def ss(self) -> RBSystemSpec:
    return self.sctx.sys_spec

  def dump_state(self) -> np.ndarray:
    return self.ss.dump_state(self.rootl)

  def load_state(self, state: ControlInputState):
    self.ss.load_state(self.rootl, state)

  def derivative(self, controls: np.ndarray, state: np.ndarray = None) -> np.ndarray:
    if state is None:
      state = self.dump_state()
    dx = self.ss.state_packer.unpack(state)
    self.load_state(state.copy())
    controls = self.sd.fm.ctrl2model(self, controls)
    fd = ForwardDynamicsData(fq=controls)
    output = forward_dynamics(fd, self.sd)
    dx.qdd = output.sd
    return self.ss.d_packer.pack(dx)

  def integrate(self, dt: float, d: np.ndarray, dump=True) -> np.ndarray | None:
    dx = self.ss.d_packer.unpack(d)
    #ci = ss.state_packer.unpack(sd)
    #qd = ss.q_desc.unpack(ci.q)
    #qdd = ss.q_desc.unpack(ci.qd)
    qdd = self.ss.qd_desc.unpack(dx.qdd)
    qd = self.ss.qd_desc.unpack(dx.qd)
    for k, qddi in qdd.items():
      rbl: RigidBodyLink = self.sctx.name2obj[k].self_link
      rbl.link_data.integrate(dt, qd[k], qddi)
    if dump:
      return self.dump_state()

  def do_step(self, dt, controls: np.ndarray, state: np.ndarray, rk4: bool):
    state = g_oph.array(self.ss.state_packer.pack(state))
    controls = g_oph.array(self.ss.ctrl_packer.pack(controls))

    if rk4:
      k1 = self.derivative(controls, state)
      k2 = self.derivative(controls, self.integrate(dt / 2, k1))
      k3 = self.derivative(controls, self.integrate(dt / 2, k2))
      k4 = self.derivative(controls, self.integrate(dt, k3))
      k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
      self.load_state(state)
      return self.integrate(dt, k)
    else:
      d = self.derivative(controls, state)
      return self.integrate(dt, d)

  def mom_fixer(self, target_mom: SpatialVector, state: np.ndarray = None):
    assert self.rootl.link_data.spec.type in (RigidBodyLinkType.FREE, RigidBodyLinkType.FREE_XY)

    assert self.rootl.link_data.spec.type == RigidBodyLinkType.FREE, 'not impl'
    if state is None: state = self.dump_state()
    dx = self.ss.state_packer.unpack(state)
    self.load_state(dx)
    cur_mom = self.mom

    id = np.identity(6)
    tb = []
    for i in range(6):
      self.rootl.link_data.qd_joint.load(id[i])
      tb.append(self.mom.data)
    sol = g_oph.solve(g_oph.array(tb).T, target_mom.data - cur_mom.data)
    dx.qd = g_oph.set(dx.qd, dx.qd[:6] + sol)[:6]
    self.load_state(dx)

  @property
  def mom(self) -> SpatialVector:
    com = self.rootl.agg_com
    return self.rootl.compute_momentum(Transform.From(pos=com), SpatialVector.Vector())

  @property
  def est_v_com(self) -> Vec3:
    return self.mom.v / self.rootl.agg_mass


class SceneRegistar(cmisc.PatchedModel):
  mp: dict[str, Callable[[], SceneData]] = cmisc.pyd_f(dict)

  def register(self, s: str, func: Callable[[], SceneData], force: bool = False) -> str:
    assert force or s not in self.mp
    self.mp[s] = func
    return s

  @classmethod
  def reg_attr(cls, name: str = None):

    dx = A(name=name)

    def wrapper(f):

      @cmisc.functools.wraps(f)
      def wrap_f(*args, **kwargs):
        res = f(*args, **kwargs)
        res.idx = idx
        return res

      if dx.name is None:
        dx.name = f.__name__
      idx = cls.Registar.register(dx.name, wrap_f)
      wrap_f.idx = idx
      return wrap_f

    return wrapper

  def get(self, s: str) -> Callable[[], SceneData]:
    return self.mp[s]

  @cmisc.cached_classproperty
  def Registar(cls):
    return SceneRegistar()


class ModelData(cmisc.PatchedModel):
  func_id: str

  @cmisc.cached_property
  def sctx(self) -> SceneContext:
    return self.create_sctx()

  @cmisc.cached_property
  def ss(self) -> RBSystemSpec:
    return self.sctx.sys_spec

  @cmisc.cached_property
  def force_model(self) -> ForceModel:
    return self.scene_data.fm

  @cmisc.cached_property
  def scene_data(self) -> SceneData:
    return self.func()

  @cmisc.cached_property
  def func(self) -> Callable[[], SceneData]:
    return SceneRegistar.Registar.get(self.func_id)

  def create_sctx(self) -> SceneContext:
    return self.func().sctx

  def create_simulator(self) -> Simulator:
    return Simulator(sd=self.func())

  @property
  def nctrl(self) -> int:
    return self.force_model.nctrl_f(self.ss.ctrl_packer.pos)


class ControlParameters(cmisc.PatchedModel):
  ndt_const: int
  dt: float
  integ_nsteps: int
  nt: int
  use_jit: bool = True
  use_rk4: bool = True
  enable_batch: bool = True
  full_jit: bool = True
  ctrl_bounds: np_array_like | list = cmisc.pyd_f(lambda: [-10, 10])

  @property
  def actual_dt(self) -> float:
    return self.dt / self.integ_nsteps

  @property
  def dt_ctrl(self) -> float:
    return self.dt * self.ndt_const

  @property
  def end_t(self) -> float:
    return self.dt * self.ndt_const * self.nt

  @end_t.setter
  def end_t(self, v):
    self.dt = v / (self.ndt_const * self.nt)

  @property
  def tot_steps(self) -> int:
    return self.ndt_const * self.integ_nsteps * self.nt

  def to_time_annotated(sef, ctrls: np.ndarray) -> np.ndarray:
    tl = (np.arange(len(ctrls)) * sef.dt_ctrl).reshape((-1, 1))
    return g_oph.concatenate((tl, ctrls), axis=1)

  def to_delta_time_annotated(sef, ctrls: np.ndarray) -> np.ndarray:
    tl = (np.ones(len(ctrls)) * sef.dt_ctrl).reshape((-1, 1))
    return g_oph.concatenate((tl, ctrls), axis=1)

  def from_delta_time_annotated(self, t_ctrls: np.ndarray) -> np.ndarray:
    t_ctrls = t_ctrls.copy()
    t_ctrls[:, 0] = np.add.accumulate(t_ctrls[:, 0])
    return self.from_time_annotated(t_ctrls)

  def from_time_annotated(self, t_ctrls: np.ndarray) -> np.ndarray:
    res = []
    pos = 0
    for t in np.arange(0, self.end_t, self.dt_ctrl):
      while pos + 1 < len(t_ctrls) and t_ctrls[pos, 0] <= t:
        pos += 1
      res.append(t_ctrls[pos, 1:])
    return g_oph.array(res)


class ControlSpec(cmisc.PatchedModel):
  cparams: ControlParameters = None
  mdata: ModelData
  end_conds: ControlInputEndConds = None

  @cmisc.cached_property
  def raw_ctrl_packer(self) -> NumpyPacker:
    return self.mdata.ss.ctrl_packer

  @cmisc.cached_property
  def ctrl_packer(self) -> NumpyPacker:
    # takes into account the forcemodel
    packer = NumpyPacker()
    packer.add_dict(ctrl=self.mdata.nctrl)
    return packer

  @cmisc.cached_property
  def ctrl_sim_packer(self) -> NumpyPacker:
    packer = NumpyPacker()
    packer.add_dict(ctrl=(self.cparams.nt, self.mdata.nctrl))
    return packer

  @cmisc.cached_property
  def state_packer(self) -> NumpyPacker:
    return self.mdata.ss.state_packer

  def make_packed(self, state: np.ndarray | ControlInputState) -> np.ndarray:
    return self.state_packer.pack(state)


class ControlFullSpec(cmisc.PatchedModel):
  consts: ControlInput
  spec: ControlSpec


class RB_PbAnalysis(cmisc.PatchedModel):
  spec: ControlSpec

  @property
  def mdata(self) -> ModelData:
    return self.spec.mdata

  def setup(self, setup_jit=1):
    if setup_jit:
      self.jit_ctrl
      self.fix_mom_jit
    #self.jac_func

  @property
  def fix_mom_jit(self):
    jitf = jax.jit(self.fix_mom)
    jitf(self.spec.state_packer.default, SpatialVector.Vector().data)  # trigger jit
    return jitf

  def fix_mom(self, state: np.ndarray, mom: np.ndarray) -> np.ndarray:
    sx: Simulator = self.mdata.create_simulator()
    sx.mom_fixer(SpatialVector.Vector(mom), state)
    return sx.dump_state()

  def advance(self, sim: Simulator, dt: float, u: np.ndarray, **kwargs):
    state = sim.dump_state()
    #print()
    #print()
    #print('INPUT ', dt, sim.dump_state(), u, kwargs)
    res = np.array(self.integrate(dt, u, state, 1, **kwargs))
    sim.load_state(res)
    #print('FUU ', sim.dump_state())

  @cmisc.cached_property
  def jac_func(self):

    def build_jit(nsteps, *args):
      res = jax.jacrev(lambda *b: px.integrate(*b, nsteps), argnums=(1,))
      return res(*args)

    return jax.jit(build_jit, static_argnums=0)

  def do_control(self, dt, controls, state, rk4=False):
    sx: Simulator = self.mdata.create_simulator()
    return sx.do_step(dt, controls, state, rk4)

  def get_integ_func(self, rk4):
    return lambda *args: self.do_control(*args, rk4=rk4)

  def reset_jit(self):
    Cachable.ResetCache(fileless=True)

  @Cachable.cachedf(fileless=True, method=True)
  def jit_ctrl(self, rk4):
    jit_ctrl = jax.jit(self.get_integ_func(rk4))
    jit_ctrl(0., self.spec.ctrl_packer.default, self.spec.state_packer.default)  # trigger jit
    return jit_ctrl

  def integrate1(self, dt: float, u: np.ndarray, state: np.ndarray, rk4=False, use_jit=None):
    # Would need sympletic integration scheme?
    use_jit = self.spec.cparams.use_jit if use_jit is None else use_jit
    rk4 = self.spec.cparams.use_rk4 if rk4 is None else rk4
    fx = self.jit_ctrl(rk4) if use_jit else self.get_integ_func(rk4)
    return make_array(fx(dt, u, state))

  def integrate(self, dt: float, u: np.ndarray, state: np.ndarray, nsteps, **kwargs):
    for _ in range(nsteps):
      state = self.integrate1(dt, u, state, **kwargs)
    return state

  def eval_obj(self, f_conds: ControlInputEndConds, state: np.ndarray, t=None, end=0) -> np.ndarray:
    if not f_conds: return 0
    if f_conds.only_end and not end: return 0
    state = self.spec.state_packer.unpack(state)
    obj = 0
    if f_conds.q is not None: obj += g_oph.norm((state.q - f_conds.q) * f_conds.q_weights)
    if f_conds.qd is not None: obj += g_oph.norm((state.qd - f_conds.qd) * f_conds.qd_weights)

    return obj


class SimulHelper(cmisc.PatchedModel):
  spec: ControlSpec
  tot_batch_size: int = 16 * 20
  t: float = 0

  @property
  def end_conds(self) -> ControlInputEndConds:
    return self.spec.end_conds

  def setup(self):
    if self.ctrl_params.full_jit:
      self.jit_fx
      self._dispatcher_jit

  @cmisc.cached_property
  def sim(self) -> Simulator:
    sim = self.spec.mdata.create_simulator()
    return sim

  def fix_mom(self, mom: SpatialVector):
    if self.ctrl_params.use_jit:
      ostate = self.runner.fix_mom_jit(self.sim.dump_state(), mom.data)
    else:
      ostate = self.runner.fix_mom(self.sim.dump_state(), mom.data)
    self.sim.load_state(np.array(ostate))

  def integrate1(self, dt: float, u: np.ndarray, **kwargs):
    self.t += dt
    ostate = np.array(self.runner.integrate1(dt, u, self.sim.dump_state(), **kwargs))
    self.sim.load_state(ostate)

  @cmisc.cached_property
  def runner(self) -> RB_PbAnalysis:
    res = RB_PbAnalysis(spec=self.spec)
    return res

  @property
  def ctrl_params(self) -> ControlParameters:
    return self.spec.cparams

  @Cachable.cachedf()
  @staticmethod
  def Get(spec: ControlSpec) -> "SimulHelper":
    return SimulHelper(spec=spec)

  def get_end_state(
      self,
      ctrl: np.ndarray,
      i_state: ControlInputState,
      obj_cb=lambda state, **kwargs: 0
  ) -> tuple:
    ctrl = self.spec.ctrl_sim_packer.unpack(ctrl).ctrl
    state = self.spec.make_packed(i_state)
    ntx = len(ctrl)
    obj = 0
    for i in range(ntx):
      state = self.runner.integrate(
          self.ctrl_params.actual_dt,
          ctrl[i],
          state,
          self.ctrl_params.ndt_const * self.ctrl_params.integ_nsteps,
      )
      obj += obj_cb(state, end=0, t=(i + 1) / ntx)
    return state, obj

  def proc(self, ctrl, i_state: ControlInputState):
    state, obj = self.get_end_state(
        ctrl, i_state,
        lambda *args, **kwargs: self.runner.eval_obj(self.end_conds, *args, **kwargs)
    )
    obj += self.runner.eval_obj(self.end_conds, state, end=1)
    return np.array(state), obj

  @cmisc.cached_property
  def jit_fx(self):
    func = jax.jit(self.fx)
    if self.ctrl_params.full_jit:
      glog.warn("Start jit_fx")
      func(self.spec.ctrl_sim_packer.default, self.spec.state_packer.default)
      glog.warn("End jit_fx")
    return func

  def eval(self, ctrl: np.ndarray, state: np.ndarray) -> np.ndarray:
    if self.ctrl_params.full_jit:
      return self.jit_fx(ctrl, state)

    ostate, val = self.proc(ctrl, state)
    assert not np.isnan(val), (ctrl, state)
    return val

  def fx(self, ctrl, state):
    ctrl = self.spec.ctrl_sim_packer.unpack(ctrl).ctrl
    ntx = len(ctrl)
    assert ntx > 0
    obj = 0

    def fus(u, _s):
      s, obj, sid = _s
      s = jax.lax.fori_loop(
          0, self.ctrl_params.ndt_const * self.ctrl_params.integ_nsteps,
          lambda i, state: self.runner.jit_ctrl(self.ctrl_params.use_rk4)
          (self.ctrl_params.actual_dt, u, state), s
      )
      obj += self.runner.eval_obj(self.end_conds, s, t=(sid + 1) / ntx)
      return s, obj, sid + 1

    state, obj, _ = jax_forctrl(fus, ctrl, (state, 0, 0))
    obj += self.runner.eval_obj(self.end_conds, state, end=1)

    return obj

  def _dispatcher(self, u_lst, s_lst):
    nv = len(s_lst)
    res = jnp.zeros(nv)

    def f_usr(i, usr):
      u, s, r = usr
      ri = self.fx(u[i, :], s[i, :])
      r = r.at[i].set(ri)
      return (u, s, r)

    u, s, r = jax.lax.fori_loop(0, nv, f_usr, (u_lst, s_lst, res))
    return r

  @property
  def ndevices(self) -> int:
    return jax.device_count()

  @property
  def batch_size(self) -> int:
    return self.tot_batch_size // self.ndevices

  @property
  def ntake(self) -> int:
    return self.ndevices * self.batch_size

  @Cachable.cachedf(fileless=True, method=True)
  def _dispatcher_jit_pmap(self, batch_size):
    func = jax.jit(self._dispatcher)
    return jax.pmap(func)

  @Cachable.cachedf(fileless=True, method=True)
  def make_dispatch_jit(self, batch_size):
    func = self._dispatcher_jit_pmap(batch_size)
    ndevices = jax.device_count()
    u0 = np.random.rand(*(batch_size, self.spec.ctrl_sim_packer.pos))
    s0_0 = self.runner.spec.state_packer.default
    s0 = np.broadcast_to(s0_0, (batch_size,) + s0_0.shape)

    if self.ctrl_params.full_jit:
      _ = np.array(func(u0, s0))
    return func

  @cmisc.cached_property
  def _dispatcher_jit(self):
    return self.make_dispatch_jit(self.batch_size)

  def dispatcher(self, u_lst, s_lst):
    res = []
    for i in range(0, len(u_lst), self.ntake):
      u = u_lst[i:i + self.ntake]
      s = s_lst[i:i + self.ntake]

      ndevices = (len(u) + self.batch_size - 1) // self.batch_size
      target = ndevices * self.batch_size
      add = target - len(u)
      if add:
        u = np.concatenate((u, np.zeros((add, self.spec.ctrl_sim_packer.pos))), axis=0)
        s = np.concatenate((s, np.zeros((add, self.spec.state_packer.pos))), axis=0)

      u = u.reshape((ndevices, self.batch_size, -1))
      s = s.reshape((ndevices, self.batch_size, -1))
      cur = np.array(self._dispatcher_jit_pmap(self.batch_size)(u, s)).reshape((-1,))
      if add:
        cur = cur[:-add]
      res.extend(cur)
    return np.array(res)

  def get_response(self, s0: np.ndarray | ControlInputState, ctrls: np.ndarray) -> list[np.ndarray]:

    state = self.spec.state_packer.pack(s0)
    res = []
    for ctrl in ctrls:
      state = np.array(
          self.runner.integrate(
              self.ctrl_params.actual_dt,
              ctrl,
              state,
              self.ctrl_params.ndt_const * self.ctrl_params.integ_nsteps,
          )
      )
      res.append(state)
    return res
