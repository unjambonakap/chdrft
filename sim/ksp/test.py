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
from pydantic import Field
import krpc
from scipy.spatial.transform import Rotation as R
import time
from chdrft.sim.rb.base import *
from chdrft.sim.traj.lander import *
from chdrft.sim.traj import tgo
from chdrft.sim.ksp.base import *

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



def test(ctx):

  g_utils.connect()
  g_utils.launch_vessel("./Ships/VAB/SpaceX Falcon 9 Block 5.craft")

  v = g_utils.get_vessel()
  earth = g_utils.get_body('Kerbin')
  moon = g_utils.get_body('Mun')
  vmoon = v.get_flight(moon)
  flight = v.get_flight(earth)
  vframe = v.internal.reference_frame

  v.internal.control.activate_next_stage()
  v.internal.control.activate_next_stage()


  #v.set_angular(Vec3.Zero(), earth.reference_frame)
  #v.set_velocity(-100*Vec3.Z(), earth.reference_frame)

  v.internal.control.throttle = 1
  vpos = v.internal.position(earth.reference_frame)
  flight.set_position(tuple(np.array(vpos) * 1.5))
  g_utils.conn.nop(req_phys_loop=1)
  flight.set_position(tuple(np.array(vpos) * 1.9))
  g_utils.conn.nop(req_phys_loop=1)
  flight.set_position(tuple(np.array(vpos) * 3.9))
  g_utils.conn.nop(req_phys_loop=1)
  flight.set_position(tuple(np.array(vpos) * 10.9))
  g_utils.conn.nop(req_phys_loop=1)
  v.internal.control.throttle = 0

  cam = g_utils.conn.space_center.camera
  cam.mode = g_utils.conn.space_center.CameraMode.locked
  g_utils.conn.nop(req_phys_loop=3)
  cam.pitch = 0
  cam.yaw = 0
  cam.pitch = 0
  cam.yaw = 180
  v.disable_auto_gimbal()

  v.internal.control.activate_next_stage()
  v.internal.control.activate_next_stage()
  v.internal.control.activate_next_stage()
  v.disable_auto_gimbal()

  tmpx = make_rot_tsf(y=(Vec3.Y() + Vec3.X() * 0.01), z=Vec3.Z())  
  w0 = v.wl(moon.reference_frame) @ tmpx
  g_utils.conn.nop(req_phys_loop=1)
  v.internal.set_rotation(w0.rot_xyzw, moon.reference_frame, False)
  g_utils.conn.nop(req_phys_loop=10)
  g_utils.conn.space_center.time_warp_helper.set_warp_rate(0, 1)
  g_utils.conn.space_center.time_warp_helper.set_physical_warp_rate(0, 1)
  g_utils.conn.space_center.physics_warp_factor = 0
  g_utils.conn.nop(req_phys_loop=10)
  v.internal.control.yaw = 0

  lat, lon = 1, 30
  alt = moon.surface_height(lat, lon)
  print(alt)
  g_utils.conn.nop(req_phys_loop=1)
  tpos = Vec3.Pt(moon.position_at_altitude(lat, lon, alt + 500000, moon.reference_frame))
  p0 = tpos
  vmoon.set_position(tpos.vdata)
  w0 = make_rot_tsf(y=p0.vdata, x=Vec3.X())
  g_utils.conn.nop(req_phys_loop=1)
  v.internal.set_rotation(w0.rot_xyzw, moon.reference_frame)


  v0 = Vec3.Zero()
  gspec = tgo.GravitySpec(center=Vec3.ZeroPt(), mass=moon.mass)
  t_tgo = 30
  pe = Vec3.Pt(moon.position_at_altitude(lat, lon, alt + 30, moon.reference_frame))
  p0 = Vec3.Pt(moon.position_at_altitude(lat, lon, alt + 200, moon.reference_frame))
  sx = tgo.TGOSolver(np.ones(3, dtype=int) * 3, [pe.vdata, np.zeros(3)])

  g_utils.conn.space_center.get_gravity(p0.vdata, moon.reference_frame)
  gspec(p0)

  g_utils.conn.space_center.time_warp_helper.set_physical_warp_rate(0, 1)
  g_utils.conn.space_center.physics_warp_factor = 0
  g_utils.conn.nop(req_phys_loop=10)

  v.set_angular(Vec3.Zero(), moon.reference_frame)
  g_utils.conn.nop(req_phys_loop=10)
  v.set_velocity(Vec3.Zero(), moon.reference_frame)
  g_utils.conn.nop(req_phys_loop=10)
  v.set_angular(Vec3.Zero(), moon.reference_frame)
  g_utils.conn.nop(req_phys_loop=10)
  v.set_velocity(Vec3.Zero(), moon.reference_frame)
  g_utils.conn.nop(req_phys_loop=10)

  w0 = make_rot_tsf(y=p0.vdata, x=Vec3.X())
  g_utils.conn.nop(req_phys_loop=1)
  v.internal.set_rotation(w0.rot_xyzw, moon.reference_frame)
  v.internal.set_position(p0.vdata, moon.reference_frame)
  g_utils.conn.nop(req_phys_loop=1)


  lc = LanderController(
      tgo=sx,
      t_tgo=t_tgo,
      mass=v.internal.mass,
      refresh_t_seconds=5,
      pidz_params=dict(kp=10, kd=3, max_ang=np.pi/20),
      step_sec=g_utils.conn.space_center.time_warp_helper.fixed_delta_time,
      local_thrust_dir=v.base_impulse_dir(),
  )

  start_time = g_utils.conn.space_center.ut
  for i in range(100):
    t = g_utils.conn.space_center.ut - start_time
    lc.mass = v.internal.mass
    wl = v.wl(moon.reference_frame)
    g = gspec(wl.pos_v)
    vl = Vec3(v.internal.velocity(moon.reference_frame))
    want = lc.process(t, wl, vl, gravity=g)
    local_w = wl.inv @ want
    th = v.internal.available_thrust

    v.set_gimbal(local_w.uvec)
    throttle = want.norm / th
    print(i ,t, throttle)
    v.internal.control.throttle = min(1, throttle)
    g_utils.conn.nop(req_phys_loop=1)


#conn.space_center.physics_warp_factor = 1

  g_utils.set_cam(Z.opa_math.rot_look_at([0, 0, 1], [0, 1, 0]))
  flight.set_rotation(tuple(unity_ylookat([1, 0, 0]).as_quat()))
  flight.set_rotation(tuple(unity_ylookat([0, 1, 0]).as_quat()))

  cam.transform.rotation = [0, 0, 0, 1]

  for i in np.linspace(-1, 1, 2):
    for j in np.linspace(-1, 1, 5):
      time.sleep(1)
      for e0 in v.engines:
        e0.gimbal.enable_gimbal = False
        r = e0.gimbal.actuation2_rot([i, j])
        e0.gimbal.set_gimbal_rot(tuple(r))

  #print(t.thrust_direction(vframe))
  #v.set_gimbal(R.from_euler('xyz', [-20,0,0], degrees=True))


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
