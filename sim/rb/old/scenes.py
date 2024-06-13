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
from pydantic.v1 import BaseModel, Field
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

from chdrft.sim.rb.old.rb import *
from chdrft.sim.rb.old.rb_analysis import *

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class SceneRegistar(cmisc.PatchedModel):
  mp: dict[str, Callable[[], SceneContext]] = Field(default_factory=dict)
  def register(self, s: str, func: Callable[[], SceneContext]) -> str:
    assert not s in self.mp
    self.mp[s] = func
    return s

  def get(self, s:str) -> Callable:
    return self.mp[s]

g_registar = SceneRegistar()



def define_wheel(tx, wheel_r, wheel_h, **kwargs):

  ma = MoveDesc(v_l=Vec3([0, 0, 0]), rotvec_l=Vec3.Z() * 30)
  wheel_r = 0.5
  wheel_h = 0.3

  wheel_sys = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel_sys'),
          move_desc=MoveDesc(rotvec_l=Vec3.Z() * 0),
          link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z(), pivot_rotang=0),
          **kwargs
      )
  )
  wheel = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel'),
          spec=SolidSpec.Cylinder(1, wheel_r, wheel_h),
          wl=Transform.From(rot=make_rot(z=Vec3.X())),
          move_desc=ma,
          link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z(), pivot_rotang=0),
          parent=wheel_sys,
      )
  )
  wheel_case = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel_case'),
          wl=Transform.From(pos=[0, 0, 0], rot=R.from_rotvec(Vec3.Y().vdata * 0)),
          spec=SolidSpec.Box(10, 2 * wheel_r + wheel_h, 2 * wheel_r + wheel_h, 2 * wheel_r),
          parent=wheel_sys
      )
  )


def rocket_scene():
  sctx = SceneContext()
  tx = RBTree(sctx=sctx)

  cylinder_h = 1e1
  cylinder_r = 1e0
  cone_h = 2
  wheel_r = 0.5
  wheel_h = 0.3

  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          move_desc=MoveDesc(v_l=Vec3([0, 0, 0]), rotvec_l=Vec3.X() * 1),
          link_data=LinkData(static=False, free=True),
      )
  )
  if 1:
    cylinder = tx.add(
        RBDescEntry(
            data=RBData(base_name='cylinder'),
            spec=SolidSpec.Cylinder(3, cylinder_r, cylinder_h),
            parent=root,
        )
    )
  if 1:
    cone = tx.add(
        RBDescEntry(
            data=RBData(base_name='cone'),
            spec=SolidSpec.Cone(1e1, cylinder_r, cone_h),
            wl=Transform.From(pos=[0, 0, cylinder_h / 2 + cone_h * 1 / 4]),
            parent=root,
        )
    )

  nw = 4
  wheels = []

  for i in range(nw):
    ang = i * 2 * np.pi / nw
    dir = np.array([np.cos(ang), np.sin(ang), 0])
    pdir = np.array([-np.sin(ang), np.cos(ang), 0])
    rot = make_rot(z=dir, x=Vec3.Z().vdata, y=-pdir, all_good=True)
    wl = Transform.From(pos=dir * (cylinder_r + wheel_r) + [0, 0, cylinder_h / 3], rot=rot)
    define_wheel(tx, wheel_r, wheel_h, wl=wl, parent=root)

  res = tx.create(root, wl_from_com=True)
  return A(target=res, sctx=sctx)


def scene_small():
  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  wheel_r = 0.5
  wheel_h = 0.3
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Box(1e1, 1, 1, 1),
          move_desc=MoveDesc(v_l=Vec3([0, 0, 0]), rotvec_l=Vec3.Z()),
          link_data=LinkData(static=False, free=True),
      )
  )
  wheel = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel'),
          spec=SolidSpec.Cylinder(2, wheel_r, wheel_h),
          wl=Transform.From(rot=make_rot(z=Vec3.X(), y=Vec3.Z())),
          move_desc=MoveDesc.From(Vec3.Z() * 50),
          link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z(), pivot_rotang=0),
          parent=root
      )
  )
  res = tx.create(root, wl_from_com=True)
  return A(target=res, sctx=sctx)


def scene_test():
  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          move_desc=MoveDesc(v_l=Vec3([0, 0, 0]), rotvec_l=Vec3.X() * 1),
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
          move_desc=MoveDesc(rotvec_l=Vec3.Z() * 0),
          wl=Transform.From(pos=[0, 0, 0], rot=R.from_rotvec(Vec3.Y().vdata * 0)),
          link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z()),
      )
  )
  wheel_case = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel_case'),
          wl=Transform.From(pos=[1, 0, 0], rot=R.from_rotvec(Vec3.Y().vdata * 0)),
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

def scene_T_def(*args) -> A:
  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          move_desc=MoveDesc(v_l=Vec3([0, 0, 0]), rotvec_l=Vec3.X() * 1),
          link_data=LinkData(static=False, free=True),
      )
  )
  b1 = tx.add(
      RBDescEntry(
          data=RBData(base_name='|'),
          spec=SolidSpec.Box(9, 1, 1, 8),
          wl=Transform.From(pos=[0, 0, 0]),
          parent=root,
      )
  )
  b2 = tx.add(
      RBDescEntry(
          data=RBData(base_name='----'),
          spec=SolidSpec.Box(1, 5, 1, 1),
          wl=Transform.From(pos=[5.5, 0, 0]),
          parent=root,
      )
  )
  res = tx.create(root, wl_from_com=True)
  return A(sctx=sctx)

def scene_T() -> ControlFullSpec:

  nx = 0
  name2pos = {}


  consts = ControlInput(
      state=ControlInputState(
          root_wl=Transform.From().data,
          root_rotvec_l=(Vec3.X() * 30 + Vec3.Z()*1).data,
          wh_w=np.identity(nx) * 0,
          wh_ang=np.zeros(nx),
      ),
  )
  creator = SimulatorCreator(name2pos=name2pos, func_id=scene_T_idx)
  mdata =  ModelData(creator=creator, nctrl=nx)
  spec=ControlSpec(mdata=mdata, end_conds=consts.end_conds,
        cparams = ControlParameters(
            dt=1e-2,
            ndt_const=10,
            integ_nsteps=1,
            nt=10,
        )
                   )
  return ControlFullSpec(spec=spec, consts=consts)



def test3(ctx):
  d = scene_T()
  sim:Simulator = d.spec.mdata.create_simulator()
  #sim:Simulator = d.create_simulator()
  obj = sim.sctx.roots[0]
  print(obj.spec.mesh.bounds)
  vp = obj.cur_mesh.surface_mesh(SurfaceMeshParams()).vispy_data

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

  def an1(wh0_w, wh0_ang, wh0_torque):
    rs = rocket_scene()
    tg = rs.target
    sctx = rs.sctx

    a = sctx.name2obj['Wheel_000']
    a.self_link.move_desc.rotspeed = wh0_w
    a.self_link.link_data.pivot_rotang = wh0_ang

    tc = TorqueComputer()
    tc.setup(tg, 1e-2)
    tc.req_link_torque[a.self_link] = wh0_torque
    tc.process()
    return tg.rotvec_l.data

  ex = jax.make_jaxpr(an1)
  ex(1, 2, 3)
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
            move_desc=MoveDesc(rotvec_l=Vec3.Z() * wh_w[i]),
            wl=dx.wl,
            link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z(), pivot_rotang=wh_ang[i]),
            spec=SolidSpec.Cylinder(0.2, wheel_r, wheel_h),
            parent=root,
        )
    )
  res = tx.create(root, wl_from_com=0)
  return A(target=res, sctx=sctx)


def get_control_test_data(nwheel) -> ControlFullSpec:
  nx = nwheel

  names = 'wh_x wh_y wh_z'.split()
  name2pos = {a: i for i, a in enumerate(names)}

  root_rotvec_l = Vec3.X().vdata * 1
  root_rotvec_l = (Vec3.X() + Vec3.Y() * 2 - Vec3.Z() * 1 / 3).vdata * 2
  #root_rotvec_l=(Vec3.X() + Vec3.Y() * 2 / 3).vdata * 1
  #root_rotvec_l=Vec3.X().vdata + Vec3.Y().vdata
  consts = ControlInput(
      state=ControlInputState(
          root_wl=Transform.From().data,
          root_rotvec_l=root_rotvec_l,
          wh_w=np.identity(nx)[0] * 0,
          wh_ang=np.zeros(nx),
      ),
      end_conds=ControlInputEndConds(
          f_root_rotvec_w=(Vec3.X()).vdata * 0,
          #f_root_rotvec_l=(Vec3.Y()).vdata*0,
          #f_root_rotvec_l=root_rotvec_l,
          #f_root_rotvec_w=None,
          #f_root_wl=Transform.From().data,
          f_root_wl_rot=Transform.From().rot,
          #f_root_wl_rot=make_rot(z=Vec3.X()),
      ),
  )
  creator = SimulatorCreator(name2pos=name2pos, func_id=simple_gen)
  mdata =  ModelData(creator=creator, nctrl=nx)
  spec=ControlSpec(mdata=mdata, end_conds=consts.end_conds)
  return ControlFullSpec(spec=spec, consts=consts)


simple_gen = g_registar.register('simple_gen', control_simple)
scene_T_idx = g_registar.register('scene_T', scene_T_def)

def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
