#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import chdrft.utils.Z as Z
import numpy as np
from pydantic.v1 import Field
import krpc
from scipy.spatial.transform import Rotation as R
import time
from chdrft.sim.rb.base import *
from chdrft.sim.traj.lander import *
from chdrft.sim.traj import tgo

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def unity_ylookat(dy, dz=None):
  dy = np.array(dy)
  if dz is None: dz = Z.opa_math.make_norm(np.random.uniform(-1, 1, 3))
  dz = Z.opa_math.make_orth_norm(dz, dy)
  r1 = Z.opa_math.rot_look_at(dz, dy)
  return r1


def stereo_XY2xyz(X, Y, r):
  X = X / 2 / r
  Y = Y / 2 / r

  a = X**2 + Y**2
  return np.array([2 * X, 2 * Y, -a + 1]) / (a + 1) * r


class KRPCUtils(cmisc.PatchedModel):

  conn: krpc.Client = None

  def connect(self):
    conn = krpc.connect(
        name='My Example Program', address='localhost', rpc_port=50000, stream_port=50001
    )
    self.conn = conn
    cheats = conn.space_center.cheats
    cheats.unbreakable_joints = True
    cheats.no_crash_damage = True
    cheats.ignore_max_temperature = True

  def launch_vessel(self, path: str):
    self.conn.space_center.launch_vessel2(
        'VAB', "./Ships/VAB/SpaceX Falcon 9 Block 5.craft", 'LaunchPad', True
    )

  def get_vessel(self) -> Vessel:
    return Vessel(internal_=self.conn.space_center.active_vessel)

  def get_body(self, body_name: str) -> "krpc.spacecenter.SpaceCenter.CelestialBody":
    return self.conn.space_center.bodies[body_name]


  def set_cam(
      self, tsf: R | None, target_part: object | None = None, target_vessel: object | None = None
  ):
    cam = self.conn.space_center.camera
    if tsf is not None:
      cam.transform.rotation = tuple(tsf.as_quat())
    if target_part is not None:
      cam.set_target_part(target_part)
    if target_vessel is not None:
      cam.set_target_vessel(target_vessel)


g_utils = KRPCUtils()


class Vessel(cmisc.PatchedModel):
  internal_: object

  @property
  def internal(self) -> "krpc.spacecenter.SpaceCenter.Vessel":
    return self.internal_

  def get_position(self, rf) -> Vec3:
    return Vec3.Pt(self.internal.position(rf))

  @property
  def engines(self) -> "list[krpc.spacecenter.SpaceCenter.Engine]":
    return self.internal.active_engines

  def get_resources(self, name):
    return self.internal.resources.with_resource(name)

  @property
  def resources(self):
    rscs = set()
    for engine in self.internal.active_engines:
      for prop in engine.propellants:
        for rsc in self.internal.resources.with_resource_by_id(prop.id):
          rscs.add(rsc)
    return rscs

  def set_prop(self, v: float, target_name: str = None, resources=None):
    if resources is None: resources = self.resources
    for rsc in resources:
      if target_name is None or target_name == rsc.part.name:
        rsc.amount = rsc.max * v

  def get_flight(self, body):
    return self.internal.flight(body.reference_frame)

  def set_angular(self, v: Vec3, ref_frame):
    self.internal.set_angular_velocity(v.vdata, ref_frame)
    g_utils.conn.nop(req_phys_loop=1)
    self.internal.angular_velocity(ref_frame)

  def set_velocity(self, v: Vec3, ref_frame):
    g_utils.conn.nop(req_phys_loop=1)
    actual = np.array(self.internal.velocity(ref_frame))
    rb_v = self.internal.rb_velocity(ref_frame)
    self.internal.set_velocity(v.vdata + -np.array(actual) + rb_v, ref_frame)

  def wl(self, ref_frame=None) -> Transform:
    world = False
    if ref_frame is None:
      world = True
      ref_frame = self.internal.orbit.body.reference_frame

    pos = self.internal.position(ref_frame)
    rot = self.internal.rotation(ref_frame)
    if world:
      pos = ref_frame.position_to_world_space(pos)
      rot = ref_frame.rotation_to_world_space(rot)
    return xyzw2rot(rot).update(pos_v=Vec3.Pt(pos))

  def disable_auto_gimbal(self):
    for e0 in self.engines:
      e0.gimbal.enable_gimbal = False
      r = e0.gimbal.actuation2_rot([0, 0])
      e0.gimbal.set_gimbal_rot(tuple(r))

  def compute_bg(self, e0: "krpc.spacecenter.SpaceCenter.Engine") -> Transform:
    wb = xyzw2rot(self.internal.reference_frame.rotation)
    wg = xyzw2rot(e0.gimbal.rotation(0))
    return wb.inv @ wg

  @Z.Cachable.cachedf(args_serializer=lambda x: (x.part.part_id,), method=True)
  def base_bg_gt(self, e0: "krpc.spacecenter.SpaceCenter.Engine") -> Transform:
    e0.gimbal.set_gimbal_rot(Transform.From().rot_xyzw, True)
    bg = self.compute_bg(e0)
    # thrust dir: wt @ -Z
    wt = xyzw2rot(e0.thrusters[0].world_transform_quat)
    gt = bg.inv @ self.wl().inv @ wt
    return bg, gt

  @Z.Cachable.cachedf(args_serializer=lambda x: (x.part.part_id,), method=True)
  def impulse_dir(self, e0: "krpc.spacecenter.SpaceCenter.Engine") -> Vec3:
    e0.gimbal.set_gimbal_rot(Transform.From().rot_xyzw, False)
    return self.wl().inv @ xyzw2rot(e0.thrusters[0].world_transform_quat) @ -Vec3.Z()

  def base_impulse_dir(self) -> Vec3:
    return self.impulse_dir(self.engines[0])

  def set_gimbal(self, l_dir: Vec3):
    # for gimbals, their Z is mapped to the rocket body
    target_bt = make_rot_tsf(z=-l_dir)
    for e0 in self.engines:
      assert not e0.gimbal.enable_gimbal
      bg0, gt = self.base_bg_gt(e0)
      bg = bg0.inv @ target_bt @ gt.inv
      e0.gimbal.set_gimbal_rot(bg.rot_xyzw, True)



def wxyz2rot(wxyz) -> Transform:
  xyzw = np.array(wxyz)[[1, 2, 3, 0]]
  return xyzw2rot(xyzw)


def xyzw2rot(xyzw) -> Transform:
  return Transform.From(rot=R.from_quat(xyzw))


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
