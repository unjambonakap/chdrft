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
from chdrft.sim.ksp.base import *
import datetime

import seaborn as sns
import pandas as pd

init_jupyter()
try:
  get_ipython().run_line_magic('matplotlib', 'qt')
except:
  ...

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def orbit(target, dalt=5000, radius_mult=1, rotating=False, **kwargs):
  pos = v.get_position(target.reference_frame)
  lat, lon = get_latlon(target)
  alt = target.equatorial_radius * (radius_mult - 1) + real2body_alt((lat, lon), dalt, target)
  tpos = sph2body((lat, lon), alt, target)
  print(pos, tpos)
  print(lat, lon, body2sph(tpos, target), body2sph(pos, target))
  travel(
      tpos, target.reference_frame if rotating else target.non_rotating_reference_frame, **kwargs
  )


def sph2body(latlon, alt, body):
  lat, lon = Z.rad2deg(np.array(latlon))
  return Vec3.Pt(body.position_at_altitude(lat, lon, alt, body.reference_frame))


def body2sph(pos, body):
  rf = body.reference_frame
  lat = body.latitude_at_position(pos.vdata, rf)
  lon = body.longitude_at_position(pos.vdata, rf)
  alt = body.altitude_at_position(pos.vdata, rf)
  return list(Z.deg2rad(np.array([lat, lon]))) + [alt]


def get_latlon(target):
  return body2sph(v.get_position(target.reference_frame), target)[:2]


def travel(target: Vec3, ref_frame, linear=False, nphys=10, nsteps=10):
  curp = v.get_position(ref_frame)
  diff = target - curp
  if diff.norm < 1: return

  sgen = np.linspace if linear else np.geomspace
  steps = (
      list(sgen(1, diff.norm / 2, num=nsteps, endpoint=False)) +
      list(sgen(diff.norm, diff.norm / 2, num=nsteps))[::-1]
  )

  for plan in steps:
    p = curp + diff.uvec * plan
    if abs(plan) > 1000:
      print(plan)
      v.internal.set_position(p.vdata, ref_frame)
      v.set_angular(Vec3.Zero(), ref_frame)
      v.set_velocity(Vec3.Zero(), ref_frame)
      g_utils.conn.nop(req_phys_loop=nphys)


def transfer_orbit(target, dt, rf):
  pos = v.get_position(rf)
  t = g_utils.conn.space_center.ut
  te = t + dt
  tpos = Vec3.Pt(target.orbit.position_at(te, rf))
  dp = tpos - pos

  v0 = dp / dt
  v.set_velocity(v0, rf)
  g_utils.conn.nop(req_phys_loop=100)
  g_utils.conn.space_center.rails_warp_factor = 7
  wait_for(lambda: g_utils.conn.space_center.ut - t > 0.6 * dt, sleep_s=0.1)
  g_utils.conn.space_center.rails_warp_factor = 0
  g_utils.conn.space_center.physics_warp_factor = 0
  g_utils.conn.nop(req_phys_loop=10)


def wait_for(pred, sleep_s=1):
  while not pred():
    time.sleep(sleep_s)


def real2body_alt(latlon, real_alt, body):
  latlon = Z.rad2deg(np.array(latlon))
  surface = body.surface_height(latlon[0], latlon[1])
  return real_alt + surface


def sph2body(latlon, alt, body):
  lat, lon = Z.rad2deg(np.array(latlon))
  return Vec3.Pt(body.position_at_altitude(lat, lon, alt, body.reference_frame))


def make_drawing(target, lle, dalt=100, thickness=100, r=0.01):

  pts = Z.geo_utils.circle_polyline(A(r=r, pos=lle), npts=10)

  alt = real2body_alt(lle, dalt, target)
  res_pts = []
  for llx in pts:
    res_pts.append(sph2body(llx, alt, target).vdata)

  pl = g_utils.conn.drawing.add_polygon(res_pts, target.reference_frame)
  pl.thickness = thickness
  pl.color = [1, 0, 0]


def test(ctx):

  g_utils.connect()
  g_utils.launch_vessel("./Ships/VAB/SpaceX Falcon 9 Block 5.craft")

  v = g_utils.get_vessel()
  earth = g_utils.get_body('Kerbin')
  moon = g_utils.get_body('Duna')
  sun = g_utils.get_body('Sun')
  vmoon = v.get_flight(moon)
  flight = v.get_flight(earth)
  vframe = v.internal.reference_frame

  v.internal.control.activate_next_stage()
  v.internal.control.activate_next_stage()

  #v.set_angular(Vec3.Zero(), earth.reference_frame)
  #v.set_velocity(-100*Vec3.Z(), earth.reference_frame)
  vpos = v.get_position(earth.reference_frame)

  g_utils.conn.nop(req_phys_loop=1)
  v.internal.control.throttle = 1
  orbit(earth, dalt=1e6, rotating=True)
  v.internal.control.throttle = 0

  cam = g_utils.conn.space_center.camera
  cam.mode = g_utils.conn.space_center.CameraMode.locked
  g_utils.conn.nop(req_phys_loop=3)
  cam.pitch = 0
  cam.yaw = 0
  cam.pitch = 0
  cam.yaw = 180
  v.disable_auto_gimbal()

  orbit(earth, dalt=3e8)

  dt = datetime.timedelta(days=1).total_seconds()
  transfer_orbit(moon, dt, sun.non_rotating_reference_frame)
  transfer_orbit(moon, dt, sun.non_rotating_reference_frame)

  transfer_orbit(moon, dt, sun.non_rotating_reference_frame)

  dt = datetime.timedelta(days=5).total_seconds()
  transfer_orbit(moon, dt, sun.non_rotating_reference_frame)
  transfer_orbit(moon, dt, sun.non_rotating_reference_frame)

  dt = datetime.timedelta(days=1).total_seconds()
  transfer_orbit(moon, dt, sun.non_rotating_reference_frame)
  transfer_orbit(moon, dt, sun.non_rotating_reference_frame)
  v.set_prop(v=1, resources=v.get_resources('ElectricCharge'))

  v.set_angular(Vec3.Zero(), moon.non_rotating_reference_frame)
  v.set_velocity(Vec3.Zero(), moon.non_rotating_reference_frame)
  orbit(moon, dalt=3e5)

  g_utils.conn.space_center.physics_warp_factor = 0
  g_utils.conn.space_center.rails_warp_factor = 0
  orbit(moon, nsteps=10, radius_mult=100)
  v.internal.set_position([0, 0, 0], earth.reference_frame)

  lat, lon = 1, 30
  alt = moon.surface_height(lat, lon)
  g_utils.conn.nop(req_phys_loop=1)
  tpos = sph2body(lat, lon, alt + 500000, moon)
  travel(tpos, moon.reference_frame)

  v.internal.set_position(tuple(np.array(vpos) * 1.5), earth.reference_frame)

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

  p0 = tpos
  vmoon.set_position(tpos.vdata)

  #init section

  #v.internal.set_position(moon.position_at_altitude(90, 0, 5e5, moon.reference_frame), moon.reference_frame)
  orbit(moon, dalt=3e5)

  ll0 = get_latlon(moon)
  alt0 = real2body_alt(ll0, 15000, moon)

  lle = ll0 + Z.deg2rad(np.array([2, -1])) * 0.5
  alte = real2body_alt(lle, 30, moon)

  p0 = sph2body(ll0, alt0, moon)
  pe = sph2body(lle, alte, moon)

  g_utils.conn.drawing.clear()
  make_drawing(moon, lle, dalt=300, thickness=10, r=1e-3)

  gspec = tgo.GravitySpec(center=Vec3.ZeroPt(), mass=moon.mass)
  t_tgo = 100

  vx = Vec3(moon.position_at_altitude(0, 0, 0, moon.reference_frame)).uvec.vdata
  vy = Vec3(moon.position_at_altitude(0, 90, 0, moon.reference_frame)).uvec.vdata
  vz = Vec3(moon.position_at_altitude(90, 0, 0, moon.reference_frame)).uvec.vdata
  c0 = coordinates.SphericalRepresentation(
      lon=ll0[1] * u.rad,
      lat=ll0[0] * u.rad,
      distance=(moon.equatorial_radius + alt0) * u.m,
  )

  cx = coordinates.SphericalRepresentation(
      lon=00 * u.deg,
      lat=90 * u.deg,
      distance=1 * u.m,
  )
  cx.to_cartesian()

  sph2rf = Transform.From(rot=np.array([vx, vy, vz]).T)
  sph2rf @ Vec3(c0.to_cartesian().xyz.value)

  v0 = sph2rf @ Vec3(
      coordinates.SphericalDifferential(
          d_lon=(ll0[1] - lle[1]) / (t_tgo * 4) * u.rad,
          d_lat=(lle[0] - ll0[0]) / (t_tgo / 3) * u.rad,
          d_distance=1 * (alte - alt0) / (t_tgo * 0.5) * u.m
      ).to_cartesian(base=c0).xyz.value
  )

  sx = tgo.TGOSolver(np.ones(3, dtype=int) * 3, [pe.vdata, np.zeros(3)])
  an1 = tgo.analyze_tgo(sx, p0, v0, t_tgo, gspec)
  ndf = pd.DataFrame.from_dict(
      dict(
          norm=np.linalg.norm(an1.p, axis=1),
          na=np.linalg.norm(an1.a, axis=1),
      )
  )

  sns.lineplot(data=an1.df['ap0'])
  sns.lineplot(data=ndf['na'])
  sns.lineplot(data=ndf['norm'])
  ndf['na'].max(), v.internal.max_thrust / v.internal.mass

  d0 = Vec3(sx.get(xp=p0.vdata, vp=v0.vdata, tgo=t_tgo)) - gspec(p0)
  wl0 = rb_base.make_rot_tsf(y=d0.uvec, x=Vec3.X())

  v.internal.set_position(p0.vdata, moon.reference_frame)
  v.internal.set_rotation(wl0.rot_xyzw, moon.reference_frame)
  v.set_angular(Vec3.Zero(), moon.non_rotating_reference_frame)
  v.set_velocity(Vec3.Zero(), moon.reference_frame)
  v.set_velocity(v0, moon.reference_frame)
  v.set_prop(1)
  v.set_prop(v=1, resources=v.get_resources('ElectricCharge'))

  lc = LanderController(
      tgo=sx,
      t_tgo=t_tgo,
      mass=v.internal.mass,
      refresh_t_seconds=5,
      pidz_params=dict(kp=10, kd=3, max_ang=np.pi / 20),
      step_sec=g_utils.conn.space_center.time_warp_helper.fixed_delta_time,
      local_thrust_dir=v.base_impulse_dir(),
      target_normal=pe.uvec,
      threshold_final_mode=70,
  )

  records = []
  start_time = g_utils.conn.space_center.ut
  for i in range(400):
    t = g_utils.conn.space_center.ut - start_time

    lc.mass = v.internal.mass
    wl = v.wl(moon.reference_frame)
    g = gspec(wl.pos_v)
    vl = Vec3(v.internal.velocity(moon.reference_frame))
    want = lc.process(t, wl, vl, gravity=g)
    local_w = wl.inv @ want
    th = v.internal.available_thrust

    throttle = want.norm / th
    if throttle > 1e-5:
      v.set_gimbal(local_w.uvec)
    print(i, t, throttle)
    if False and throttle < 0.25:
      throttle = 0
      break
    v.internal.control.throttle = min(1, throttle)
    record = dict(t=t, mass=lc.mass, wl=wl, g=g, vl=vl, local_thrust=local_w, throttle=throttle)
    records.append(record)
    g_utils.conn.nop(req_phys_loop=1)
    print()
  v.internal.control.throttle = 0


def section1():
  lat, lon = 1, 30
  alt = moon.surface_height(lat, lon)
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
  pe = Vec3.Pt(moon.position_at_altitude(lat, lon, alt + 3000, moon.reference_frame))
  p0 = Vec3.Pt(moon.position_at_altitude(lat, lon, alt + 20000, moon.reference_frame))
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
      pidz_params=dict(kp=10, kd=3, max_ang=np.pi / 20),
      step_sec=g_utils.conn.space_center.time_warp_helper.fixed_delta_time,
      local_thrust_dir=v.base_impulse_dir(),
  )

  start_time = g_utils.conn.space_center.ut
  for i in range(300):
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
    print(i, t, throttle)
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
