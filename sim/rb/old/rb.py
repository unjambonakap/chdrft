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
import sympy as sp
import jax.numpy as jnp
import jaxlib
import jax
import pygmo as pg
from chdrft.utils.fmt import Format
from chdrft.utils.path import FileFormatHelper
from chdrft.sim.rb.base import *

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class RigidBodyLinkType(Enum):
  RIGID = 1
  MOVE = 2




class MoveDesc(cmisc.PatchedModel):
  link: "RigidBodyLink" = None
  v_l: Vec3 = Field(default_factory=Vec3.Zero)
  rotvec_l: Vec3 = Field(default_factory=Vec3.Zero)

  def load_rotvec(self, rotvec: Vec3):
    self.rotvec_l = rotvec

  @classmethod
  def From(cls, rotvec: Vec3) -> "MoveDesc":
    res = MoveDesc()
    res.load_rotvec(rotvec)
    return res

  @property
  def rotspeed(self) -> float:
    assert not self.link.link_data.free
    return self.link.link_data.pivot_rotaxis.dot(self.rotvec_l)

  @rotspeed.setter
  def rotspeed(self, v):
    assert not self.link.link_data.free
    self.rotvec_l = self.link.link_data.pivot_rotaxis * v

  @property
  def skew_matrix(self) -> np.ndarray:
    return self.rotvec_l.skew_matrix

  @property
  def skew_transform(self) -> Transform:
    return self.rotvec_l.skew_transform




class SceneContext(cmisc.PatchedModel):
  cur_id: int = 0
  mp: dict = Field(default_factory=dict)
  obj2name: "dict[RigidBody,str]" = Field(default_factory=dict)
  name2obj: "dict[str,RigidBody]" = Field(default_factory=dict)
  basename2lst: dict[str, list] = Field(default_factory=lambda: cmisc.defaultdict(list))


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


class LinkData(cmisc.PatchedModel):
  static: bool = True
  pivot_rotaxis: Vec3 = None
  free: bool = False
  pivot_rotang: float | jnp.ndarray = None
  wl_free: Transform = Field(default_factory=Transform.From)

  @property
  def wl(self) -> Transform:
    if self.free: return self.wl_free
    return self.pivot_rotaxis.exp_rot_u(self.pivot_rotang)

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert self.static or (self.pivot_rotaxis is not None or self.free)


class RigidBodyLink(cmisc.PatchedModel):
  wl0: Transform = Field(default_factory=Transform.From)
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
    self.move_desc.link = self

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
    return self.wl @ self.com_l

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
    self.rotvec_l = self.lw @ v

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
    assert not com.vec
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
    tot_mass = self.mass
    return Vec3.LinearComb(
        [(x.com_w, x.mass / tot_mass) for x in self.move_links] +
        [(self.spec.com, self.spec.mass / tot_mass)]
    )

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
MoveDesc.update_forward_refs()


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
    com = Vec3.LinearComb([(x.wl @ x.spec.com, x.spec.mass / mass) for x in l])
    assert not com.vec
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


#class ReactionWheelHelper(cmisc.PatchedModel):
#  """
#  Z: frame rot axis
#  """
#  i_frame: InertialTensor
#  i_wheel: InertialTensor
#  body2frame: Transform
#  body_speed: AngularSpeed
#
#  ang_frame: float
#  w_frame: float
#  p_wheel: Vec3
#
#  @property
#  def frame2wheel(self) -> R:
#    return R.from_euler('z', self.ang_frame)
#
#  def compute_angular_momentum(self):
#    pass


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
      else:
        #sketchy if iv <> pivot_rotaxis
        proj_on = link.link_data.pivot_rotaxis
        torque_on_axis = torque.proj(proj_on)
        torque_off_axis = torque - torque_on_axis

      tnorm_axis = torque_on_axis.dot(proj_on) + req_torque

      data.dw = tnorm_axis / iv.norm

      torque = torque_off_axis - link.link_data.pivot_rotaxis * req_torque
    return link.wl_rot @ torque

  def compute_rotvec_(self, link: RigidBodyLink):
    base_link = link.rb.self_link
    data = self.updates[link.rb]
    if link.rb.mass == 0:
      data.rotvec = base_link.rotvec_l
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
      data.rotvec = base_link.rotvec_l + link.link_data.pivot_rotaxis * (data.dw * self.dt)
    #print(
    #    f'OLD rotvec {base_link.rb.name=}, {base_link.move_desc.rotvec_l=}, {torque=}, {link.link_data.pivot_rotaxis=}'
    #)

  def compute_wl(self):
    for child, data in self.updates.items():
      base_link = child.self_link
      if base_link.link_data.free:
        start_pos_v = base_link.wl.pos_v
        tx = (base_link.move_desc.rotvec_l * self.dt).exp_rot(base_link.com_l)

        data.wl = base_link.wl @ tx
        #data.wl.pos_v = start_pos_v
      else:
        data.pivot_rotang = base_link.link_data.pivot_rotang + base_link.move_desc.rotspeed * self.dt

  def compute_rotvec(self):
    self.compute_torques()
    for child in self.updates.keys():
      self.compute_rotvec_(child.self_link)

  def set_rotvec(self, child: RigidBody, data):
    child.self_link.rotvec_l = data.rotvec

  def set_wl(self, child: RigidBody, data):
    cl = child.self_link
    if child.self_link.link_data.free:
      cl.link_data.wl_free = cl.wl0.inv @ data.wl
    else:
      cl.link_data.pivot_rotang = data.pivot_rotang % (2 * np.pi)

  def update_wl(self):
    self._update(self.set_wl)

  def update_rotvec(self):
    self._update(self.set_rotvec)

  def _update(self, func):
    for child, data in self.updates.items():
      func(child, data)


