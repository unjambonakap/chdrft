#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.opa_types import *
from pydantic.v1 import Field
from chdrft.utils.path import FileFormatHelper

from chdrft.sim.rb import rb_gen
from chdrft.sim.rb import base as rb_base
from chdrft.sim.rb.base import Vec3, Transform
from chdrft.sim.rb import scenes
from chdrft.gnc import ctrl
import chdrft.dsp.utils as dsp_utils
from chdrft.geo.satsim import gl
from chdrft.daq.stream import influx_connector
from astropy import constants as const
from astropy import coordinates
from astropy import units as u
from chdrft.sim.traj import tgo

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class PIDZParameters(cmisc.PatchedModel):
  max_ang: float = np.pi / 8
  kp: float
  kd: float


class PIDZController(cmisc.PatchedModel):
  target_z: Vec3
  err: object = None
  params: PIDZParameters
  step_sec: float

  @cmisc.cached_property
  def pid(self) -> ctrl.PIDController:
    return ctrl.PIDController(
        kp=self.params.kp,
        kd=self.params.kd,
        control_range=ctrl.Range1D(-self.params.max_ang, self.params.max_ang),
        step_sec=self.step_sec
    )

  def proc(self, wl: rb_base.Transform) -> rb_base.Vec3:
    target_local = wl.inv @ self.target_z
    proj = -target_local[:2]
    self.err = proj
    action = self.pid.push(proj)
    rotx = rb_base.Transform.From(rot=rb_base.R.from_euler('yx', action * [1, -1]))
    return wl @ rotx @ Vec3.Z()


class LanderControllerTGOState(cmisc.PatchedModel):
  snapshot: tgo.TGOSnapshot
  ctrl: PIDZController


class LanderController(cmisc.PatchedModel):
  tgo: tgo.TGOSolver
  t_tgo: float
  mass: float
  state: LanderControllerTGOState = None
  refresh_t_seconds: int = 1
  step_sec: float
  pidz_params: PIDZParameters
  local_thrust_dir: Vec3
  target_normal: Vec3 = None
  threshold_final_mode: float = None

  @cmisc.cached_property
  def z2thrust(self) -> Transform:
    return rb_base.make_rot_tsf(z=self.local_thrust_dir.vdata)

  def reset(self):
    self.state = None

  def process(self, t: float, p: Transform, dp: Vec3, gravity: Vec3) -> Vec3:
    final_mode = False
    if self.threshold_final_mode is not None:
      pe = Vec3.Pt(self.tgo.dkp_end[0])
      dist = (p.pos_v - pe).proj(self.target_normal).norm
      print('DIST ', dist)
      if dist < self.threshold_final_mode:
        print('activate proj')
        final_mode = True
        p = Transform.From(rot=p.rot, pos=pe + (p.pos_v - pe).proj(self.target_normal))
        dp = dp.proj(self.target_normal)

    if final_mode or self.state is None or t - self.state.snapshot.t0 > self.refresh_t_seconds:
      print('Reset state')
      self.state = LanderControllerTGOState(
          snapshot=tgo.TGOSnapshot(tgo=self.tgo, p=p.pos_v, dp=dp, tgo_t0=self.t_tgo - t, t0=t),
          ctrl=PIDZController(
              target_z=p @ self.local_thrust_dir, step_sec=self.step_sec, params=self.pidz_params
          ),
      )
    want = self.state.snapshot.get_p(t)
    target_acc = want - gravity
    target_norm = target_acc.norm
    world_thrust = p @ self.z2thrust @ Vec3.Z()
    print('TARGET >> ', target_acc, target_norm, world_thrust)
    if target_acc.dot(world_thrust) < 0: 
      target_norm = 0
    self.state.ctrl.target_z = target_acc.uvec
    action = self.state.ctrl.proc(p @ self.z2thrust)
    res = action * target_norm * self.mass
    return res


def test(ctx):
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
