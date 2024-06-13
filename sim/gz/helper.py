#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.types import *
from pydantic.v1 import Field
from chdrft.utils.path import FileFormatHelper
import chdrft.sim.rb.rb_gen as rb_gen
import chdrft.sim.rb.base as rb_base
from chdrft.sim.rb.base import Vec3, Transform, R
from chdrft.sim.rb import scenes
import sys
import enum
import os
import pandas as pd

os.environ['GZ_DEBUG_COMPONENT_FACTORY'] = 'true'
sys.path.append('/usr/lib/python/')
sys.path.append('/usr/local/lib')
sys.path.append('/usr/local/lib/python')

from gz.common import set_verbosity
from gz.sim7 import TestFixture, World, world_entity
from gz.math7 import Vector3d
from gz import sim7, math7
import sdformat13 as sdf
import chdrft.sim.rb.base as rb_base
import pydantic.v1 as pydantic
from chdrft.utils.rx_helpers import ImageIO

from gz import msgs
from gz.msgs.wrench_pb2 import Wrench
from gz.msgs.vector3d_pb2 import Vector3d
from gz.msgs.world_control_pb2 import WorldControl
from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.physics_pb2 import Physics
from gz.transport12 import Node

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  pass


class Conv:

  @staticmethod
  def tsf2gz(tsf: rb_base.Transform) -> math7.Pose3d:
    quat = math7.Quaterniond(*tsf.rot_wxyz)
    return math7.Pose3d(math7.Vector3d(*tsf.pos), quat)

  @staticmethod
  def vec2gz(vec: rb_base.Vec) -> math7.Vector3d:
    return math7.Vector3d(*vec.vdata)

  @staticmethod
  def vec2gzproto(vec: rb_base.Vec) -> msgs.vector3d_pb2.Vector3d:
    return msgs.vector3d_pb2.Vector3d(x=vec[0], y=vec[1], z=vec[2])

  @staticmethod
  def to_proto(e):
    if isinstance(e, math7.Vector3d):
      return msgs.vector3d_pb2.Vector3d(x=e[0], y=e[1], z=e[2])
    else:
      raise NotImplemented(f'dunno {type(e)}')

  @staticmethod
  def inertial2gz(inertial: rb_base.Inertial) -> math7.Inertiald:

    b = math7.MassMatrix3d()
    b.set_mass(inertial.mass)
    b.set_diagonal_moments(math7.Vector3d(inertial.diag_force.to_gz()))
    return math7.Inertiald(b, math7.Pose3d())


class GZCallbackMode(enum.Enum):
  PRE0 = enum.auto()
  PRE = enum.auto()
  POST = enum.auto()
  UPDATE = enum.auto()
  RESET = enum.auto()


class GZRunner(cmisc.PatchedModel):
  fixture: TestFixture
  fname: str
  world_name: str
  info: sim7.UpdateInfo = None
  ecm: sim7.EntityComponentManager = None
  server: sim7.Server = None
  node: Node = None
  callbacks: dict[GZCallbackMode,
                  list] = Field(default_factory=lambda: cmisc.defaultdict(list))

  def set_debug(self):
    set_verbosity(4)

  @classmethod
  def Build(cls, fname: str) -> GZRunner:
    fixture = TestFixture(fname)
    world_name = SDFConverter.LoadFile(fname).world_by_index(0).name()
    res = cls(fixture=fixture, fname=fname, node=Node(), world_name=world_name)
    res.build()
    return res

  def init_cbs(self):

    def do_pre(info, ecm):
      self.cb(GZCallbackMode.PRE0, info, ecm)
      self.cb(GZCallbackMode.PRE, info, ecm)

    self.fixture.on_pre_update(do_pre)
    self.fixture.on_update(lambda info, ecm: self.cb(GZCallbackMode.UPDATE, info, ecm))
    self.fixture.on_post_update(lambda info, ecm: self.cb(GZCallbackMode.POST, info, ecm))
    self.fixture.on_reset(lambda info, ecm: self.cb(GZCallbackMode.RESET, info, ecm))

  def build(self):
    self.init_cbs()
    self.fixture.finalize()
    self.server = self.fixture.server()
    self.ecm = self.server.entity_comp_mgr(0)

  @cmisc.logged_failsafe
  def cb(self, mode: GZCallbackMode, info: sim7.UpdateInfo, ecm: sim7.EntityComponentManager):
    self.info = info
    self.ecm = ecm
    for cb in self.callbacks[mode]:
      cb(self)

  @property
  def world_prefix(self) -> str:
    return f'/world/{self.world_name}'

  def service(self, name):
    return f'{self.world_prefix}/{name}'

  def set_physics(self, max_step_size, **kwargs):
    self.node.request(
        self.service('set_physics'), Physics(max_step_size=max_step_size, **kwargs), 1, Boolean()
    )

  def set_gravity(self, g: rb_base.Vec3):
    sim7.GravityCmd.GetOrCreate(self.ecm, self.world.entity()).set_data(g.to_gz())

  def set_wl(self, model, wl: rb_base.Transform):
    sim7.WorldPoseCmd.GetOrCreate(self.ecm, model.entity()).set_data(wl.to_gz())

  def set_inertial(self, link, inertial: rb_base.Inertial):
    sim7.InertialCmd.GetOrCreate(self.ecm, link.entity()).set_data(inertial.to_gz())

  def set_vel_l(self, model, vel: rb_base.Vec3):
    sim7.LinearVelocityCmd.GetOrCreate(self.ecm, model.entity()).set_data(vel.to_gz())

  def set_vel_w(self, model, vel: rb_base.Vec3):
    sim7.WorldLinearVelocityCmd.GetOrCreate(self.ecm, model.entity()).set_data(vel.to_gz())

  def set_force(self, link, force: rb_base.Vec3):
    w = Wrench(force=Conv.to_proto(force.to_gz()))
    a = sim7.ExternalWorldWrenchCmd.GetOrCreate(self.ecm, link.entity()).set_data(w)

  @property
  def world(self) -> sim7.World:
    wid = world_entity(self.ecm)
    return World(wid)

  def model(self, name: str) -> sim7.Model:
    mid = self.world.model_by_name(self.ecm, name)
    return sim7.Model(mid)

  def model_link(self, model: sim7.Model):
    return sim7.Link(model.link(self.ecm))
    #link.world_linear_velocity(mg)

  def reset(self):
    wc = WorldControl(reset=dict(all=True), pause=False)
    sl = self.service('control')
    self.node.request(sl, wc, 1, Boolean())
    self.server.run(True, 1, False)
    self.init_cbs()


class GZDataMapping(cmisc.PatchedModel):
  cl: object
  converter: object
  name: str

  def make_request(self, target: str, entity: int) -> StatRequest:
    return StatRequest(mapping=self, target=target, entity=entity)

  def query(self, runner, entity):
    a = self.cl.GetOrCreate(runner.ecm, entity)
    return self.converter(a.data())


def gzquat2local(quat: math7.Quaterniond) -> rb_base.R:
  return rb_base.R.from_quat(quat.xyzw())


class GZDatas:
  l_velocity_l = GZDataMapping(
      name='vl_l', cl=sim7.LinearVelocity, converter=lambda x: rb_base.Vec3(np.array(x))
  )
  l_velocity_w = GZDataMapping(
      name='vl_w', cl=sim7.WorldLinearVelocity, converter=lambda x: rb_base.Vec3(np.array(x))
  )
  a_velocity = GZDataMapping(
      name='va', cl=sim7.WorldAngularVelocity, converter=lambda x: rb_base.Vec3(np.array(x))
  )
  tsf = GZDataMapping(
      name='tsf',
      cl=sim7.Pose,
      converter=lambda x: rb_base.Transform.From(pos=np.array(x.pos()), rot=gzquat2local(x.rot()))
  )


class StatRequest(cmisc.PatchedModel):
  mapping: GZDataMapping
  target: str
  entity: int

  @property
  def col_name(self) -> str:
    return f'{self.target}.{self.mapping.name}'

  def fill(self, record: dict, runner: GZRunner):
    record[self.col_name] = self.mapping.query(runner, self.entity)


class StatEntry(cmisc.PatchedModel):
  record: dict[GZDatas, object]


class StatsGatherer(cmisc.PatchedModel):
  records: list[dict[str, object]] = Field(default_factory=list)
  requests: list[StatRequest]
  iter_downsample: int = 1
  active_record: dict = None
  connector: ImageIO = Field(default_factory=ImageIO)

  def register(self, runner: GZRunner):
    runner.callbacks[GZCallbackMode.PRE0].append(self.record)
    runner.callbacks[GZCallbackMode.POST].append(self.push_record)

  def record(self, runner: GZRunner):
    if runner.info.iterations % self.iter_downsample != 0: return
    self.active_record = dict()

    self.active_record['sim_time'] = runner.info.sim_time
    self.active_record['real_time'] = runner.info.real_time
    self.active_record['iterations'] = runner.info.iterations
    for req in self.requests:
      req.fill(self.active_record, runner)

  def push_record(self, runner: GZRunner):
    if self.active_record is not None:
      self.records.append(self.active_record)
      self.connector.push(self.active_record)
      self.active_record = None


def cb(a: str) -> str:
  print('qq', a)
  return "/home/benoit/.gz/fuel/fuel.gazebosim.org/openrobotics/models/moon dem/1/model.sdf"
  assert 0


g_sdf_conf = sdf.ParserConfig()
g_sdf_conf.set_find_callback(cb)


class SDFConverter:
  MODEL_NAME = 'model'

  def __init__(self, fname, world_name):
    self.root = SDFConverter.LoadFile(fname)
    self.world_name = world_name
    self.model = self.create_model()

  @staticmethod
  def LoadFile(fname: str) -> sdf.Root:
    root = sdf.Root()
    root.load(fname, g_sdf_conf)
    return root

  def get_str(self):
    w = self.root.world_by_index(0)
    w.set_name(self.world_name)
    w.clear_models()
    w.add_model(self.model)
    return self.root.to_string()

  def write(self, fname):

    with open(fname, "w") as f:
      f.write(self.get_str())

  def create_model(self) -> sdf.Model:
    model = sdf.Model()
    model.set_name(SDFConverter.MODEL_NAME)
    model.set_self_collide(False)
    return model

  def spec2geo(self, spec: rb_gen.SolidSpec) -> sdf.Geometry:
    geometry = sdf.Geometry()
    #TODO: this is bad - spec.type is not set correctly most of the time (composition of several RBDescEntry)
    if spec.type == rb_base.SolidSpecType.BOX:
      x = sdf.Box()
      x.set_size(spec.mesh.transform.get_scale().to_gz())
      geometry.set_box_shape(x)
      geometry.set_type(sdf.GeometryType.BOX)
    elif spec.type == rb_base.SolidSpecType.SPHERE:
      x = sdf.Sphere()
      x.set_radius(spec.mesh.transform.get_scale()[0])
      geometry.set_sphere_shape(x)
      geometry.set_type(sdf.GeometryType.SPHERE)
    elif spec.type == rb_base.SolidSpecType.CYLINDER:
      x = sdf.Cylinder()
      x.set_radius(spec.mesh.transform.get_scale()[0])
      x.set_length(spec.mesh.transform.get_scale()[2])
      geometry.set_cylinder_shape(x)
      geometry.set_type(sdf.GeometryType.CYLINDER)
    else:
      assert 0
    return geometry

  def create_link(self, spec: rb_gen.SolidSpec, link_name='link') -> sdf.Link:

    geo = self.spec2geo(spec)
    collision = sdf.Collision()
    collision.set_name('collision_box')
    collision.set_geometry(geo)

    visual = sdf.Visual()
    visual.set_name('visual_box')
    visual.set_geometry(geo)

    link = sdf.Link()
    link.set_name(link_name)
    link.add_visual(visual)
    link.set_inertial(spec.inertial.to_gz())
    #link.add_collision(collision)
    return link

  def fill_with_rbtree(self, tree: RBTree):

    roots = tree.entry2child[None]
    assert len(roots) == 1
    root = roots[0]

    entry2link = {
        x: self.create_link(x.spec, tree.path_name(x)) for x in tree.entries
    }

    joint_names = rb_gen.NameRegister()

    for k, v in entry2link.items():
      v.set_raw_pose(k.link_data.wl.to_gz())
      self.model.add_link(v)
      if k.parent is not None:
        j = sdf.Joint()
        j.set_name(joint_names.register(None, 'joint'))
        j.set_parent_name(entry2link[k.parent].name())
        j.set_child_name(v.name())
        assert k.link_data.spec.type is rb_gen.RigidBodyLinkType.RIGID
        j.set_type(sdf.JointType.FIXED)
        self.model.add_joint(j)

  def fill_with_rbl(self, rbl: rb_gen.RigidBodyLink):
    link = self.create_link(rbl.rb.spec)
    self.model.set_raw_pose(rbl.wl.to_gz())
    self.model.add_link(link)

    # mx = sdf.Mesh()
    # mx.set_uri('file:///home/benoit/programmation/blender/resources/untitled.dae')
    # gx = sdf.Geometry()
    # gx.set_mesh_shape(mx)
    # gx.set_type(sdf.GeometryType.MESH)
    # v2 = sdf.Visual()
    # v2.set_name('test')
    # v2.set_geometry(gx)
    # link.add_visual(v2)


class SimInstance(cmisc.PatchedModel):
  runner: GZRunner = None
  model: sim7.Model = None
  sh: SimHelper = None
  id: int = 0

  def setup(self, sh: SimHelper):
    self.runner = sh.runner
    self.model = sh.model
    self.sh = sh
    self._setup(sh)
    self.id = 0

  def _setup(self, sh: SimHelper):
    ...

  @property
  def stats(self):
    return self.sh.stats

  def __call__(self):
    self.id += 1
    self._proc()

  def _proc(self):
    ...


class SimHelper(cmisc.PatchedModel):
  _runner: typing.ClassVar[GZRunner] = None

  sd: rb_gen.SceneData
  step: float = 0.01
  stats_downsample: int = 10
  si: SimInstance

  runner: GZRunner = None
  model: sim7.Model = None
  model_link: sim7.Link = None
  stats: StatsGatherer = None
  tf: object = None

  @classmethod
  def SetRunner(cls, runner: GZRunner):
    if cls._runner is not None:
      cls._runner.server.stop()
      cls._runner.fixture.release()
    cls._runner = runner

  def init(self):
    tf = cmisc.tempfile.NamedTemporaryFile()
    self.tf = tf

    base = cmisc.path_here('./traj.sdf')
    conv = SDFConverter(base, world_name='scene1')
    conv.fill_with_rbtree(self.sd.tree)
    conv.write(tf.name)
    print('model at ', tf.name, conv.get_str())

    runner = GZRunner.Build(tf.name)
    SimHelper.SetRunner(runner)
    self.runner = runner

    self.model = runner.model(SDFConverter.MODEL_NAME)
    self.model_link = runner.model_link(self.model)
    stats = StatsGatherer(
        requests=[
            GZDatas.tsf.make_request('model', self.model.entity()),
        ],
        iter_downsample=self.stats_downsample
    )
    stats.register(runner)
    self.stats = stats

    self.si.setup(self)
    runner.set_physics(self.step)

    runner.callbacks[GZCallbackMode.PRE].append(lambda *args: self.si())


  def run(self, sim_time: float):
    #runner.reset()
    self.runner.server.run(True, int(sim_time / self.step), False)
    dfx = pd.DataFrame.from_records(self.stats.records)
    return dfx


SimInstance.update_forward_refs()


def sdf_writer_test(ctx):
  s = scenes.box_scene()
  rbl = s.sctx.roots[0].self_link
  cx = SDFConverter('./traj.sdf', 'world0')
  cx.fill_with_rbl(rbl)
  #cx.write('./traj_new.sdf')

  cx = SDFConverter('./traj.sdf', 'world0')
  cx.fill_with_rbtree(s.tree)
  #cx.write('./traj_new2.sdf')


def sdf_balance_scene(ctx):
  s = scenes.balance_scene()
  cx = SDFConverter('./traj.sdf', 'world0')
  cx.fill_with_rbtree(s.tree)
  print(cx.get_str())
  cx.write('./traj_new.sdf')


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
