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

from chdrft.sim.rb.rb_gen import *

global flags, cache
flags = None
cache = None


def bias_force_g(s: RBState) -> SpatialVector:
  dir = s.wl.inv @ (Vec3.Z() * -9.81)
  return SpatialVector(f=dir * s.x.mass, dual=True)


def bias_force_gy(s: RBState) -> SpatialVector:
  dir = s.wl.inv @ (Vec3.Y() * -9.81) * 0
  return SpatialVector(f=dir * s.x.mass, dual=True)


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def box_scene() -> SceneContext:
  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Box(1, 1, 1, 1),
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.XLT_Z)),
      )
  )

  res = tx.create(root)
  fm = ForceModel.Id()
  return SceneData(sctx=sctx, fm=fm)


def define_wheel(tx, wheel_r, wheel_h, parent=None, **kwargs):

  wheel_r = 0.5
  wheel_h = 0.3

  wheel_sys = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel_sys'),
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.PIVOT_Z), **kwargs),
          parent=parent,
      )
  )

  wheel = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel'),
          spec=SolidSpec.Cylinder(1, wheel_r, wheel_h),
          link_data=LinkData(
              spec=LinkSpec(type=RigidBodyLinkType.PIVOT_Z),
              wr=Transform.From(rot=make_rot(z=Vec3.X())),
          ),
          parent=wheel_sys,
      )
  )
  wheel_case = tx.add(
      RBDescEntry(
          data=RBData(base_name='Wheel_case'),
          link_data=LinkData(
              spec=LinkSpec(type=RigidBodyLinkType.RIGID),
              wr=Transform.From(pos=[0, 0, 0], rot=R.from_rotvec(Vec3.Y().vdata * 0)),
          ),
          spec=SolidSpec.Box(10, 2 * wheel_r + wheel_h, 2 * wheel_r + wheel_h, 2 * wheel_r),
          parent=wheel_sys
      )
  )


def rocket_scene(nw):
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
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.FREE)),
      )
  )
  if 1:
    cylinder = tx.add(
        RBDescEntry(
            data=RBData(base_name='cylinder'),
            spec=SolidSpec.Cylinder(3, cylinder_r, cylinder_h),
            link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.RIGID)),
            parent=root,
        )
    )
  if 1:
    cone = tx.add(
        RBDescEntry(
            data=RBData(base_name='cone'),
            spec=SolidSpec.Cone(1e1, cylinder_r, cone_h),
            link_data=LinkData(
                spec=LinkSpec(
                    type=RigidBodyLinkType.RIGID,
                    wr=Transform.From(pos=[0, 0, cylinder_h / 2 + cone_h * 1 / 4]),
                )
            ),
            parent=root,
        )
    )

  wheels = []

  for i in range(nw):
    ang = i * 2 * np.pi / nw
    dir = np.array([np.cos(ang), np.sin(ang), 0])
    pdir = np.array([-np.sin(ang), np.cos(ang), 0])
    rot = make_rot(z=dir, x=Vec3.Z().vdata, y=-pdir, all_good=True)
    wl = Transform.From(pos=dir * (cylinder_r + wheel_r) + [0, 0, cylinder_h / 3], rot=rot)
    define_wheel(tx, wheel_r, wheel_h, wl=wr, parent=root)

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


def scene_gyro_wheel():
  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  wheel_r = 1
  wheel_h = 0.1
  attach_r = 0.03
  attach_l = 0.4

  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Cylinder(0.1, attach_r, attach_l),
          link_data=LinkData(
              spec=LinkSpec(
                  type=RigidBodyLinkType.PIVOT_XYZ,
                  rl=Transform.From(pos=[0, 0, -attach_l / 2]),
                  wr=Transform.From(rot=make_rot_ab(x=Vec3.Z(), y=Vec3.Y()))
              )
          ),
      )
  )
  tx.add(
      RBDescEntry(
          data=RBData(base_name='attach'),
          spec=SolidSpec.Cylinder(5, wheel_r, wheel_h),
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.PIVOT_Z,)),
          parent=root,
      )
  )
  res = tx.create(root, center_com=1)
  fm = ForceModel.Id()
  fm.bias_force_f = bias_force_g
  return SceneData(sctx=sctx, fm=fm)


def scene_T():

  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.FREE,)),
      )
  )
  b1 = tx.add(
      RBDescEntry(
          data=RBData(base_name='|'),
          spec=SolidSpec.Box(9, 1, 1, 5),
          link_data=LinkData(
              spec=LinkSpec(
                  type=RigidBodyLinkType.RIGID,
                  wr=Transform.From(pos=[0, 0, 0]),
              )
          ),
          parent=root,
      )
  )
  b2 = tx.add(
      RBDescEntry(
          data=RBData(base_name='----'),
          spec=SolidSpec.Box(5, 3, 1, 1),
          link_data=LinkData(
              spec=LinkSpec(
                  type=RigidBodyLinkType.RIGID,
                  wr=Transform.From(pos=[2, 0, 0]),
              )
          ),
          parent=root,
      )
  )

  res = tx.create(root, center_com=1)
  fm = ForceModel.Id()
  return SceneData(sctx=sctx, fm=fm)


def scene_T2D():

  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.FREE_XY,)),
      )
  )
  b1 = tx.add(
      RBDescEntry(
          data=RBData(base_name='|'),
          spec=SolidSpec.Box(9, 1, 1, 5),
          link_data=LinkData(
              spec=LinkSpec(
                  type=RigidBodyLinkType.RIGID,
                  wr=Transform.From(pos=[0, 0, 0]),
              )
          ),
          parent=root,
      )
  )
  b2 = tx.add(
      RBDescEntry(
          data=RBData(base_name='----'),
          spec=SolidSpec.Box(5, 3, 1, 1),
          link_data=LinkData(
              spec=LinkSpec(
                  type=RigidBodyLinkType.RIGID,
                  wr=Transform.From(pos=[2, 0, 0]),
              )
          ),
          parent=root,
      )
  )

  res = tx.create(root, center_com=1)
  fm = ForceModel.Id()
  return SceneData(sctx=sctx, fm=fm)


def two_mass():
  sctx = SceneContext()
  tx = RBTree(sctx=sctx, split_rigid=True)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Box(1, 1, 1, 1),
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.FREE,)),
      )
  )

  tx.add(
      RBDescEntry(
          data=RBData(base_name='m2'),
          link_data=LinkData(
              spec=LinkSpec(type=RigidBodyLinkType.RIGID, wr=Transform.From(pos=Vec3.X() * 3))
          ),
          spec=SolidSpec.Sphere(1, 1),
          parent=root,
      )
  )
  res = tx.create(root)

  def ctrl2model(sim: Simulator, ctrl):
    return (sim.rootl.rw @ SpatialVector.Force(ctrl).around(sim.rootl.wl @ -sim.rootl.agg_com)).data

    # would only be valid if split_rigid=False
    res = sim.rootl.rl @ sim.rootl.lw.tsf_rot @ SpatialVector.Force(ctrl)
    return res.data

  fm = ForceModel.Id()
  fm.ctrl2model = ctrl2model
  return SceneData(sctx=sctx, fm=fm)


def pivot():
  sctx = SceneContext()
  tx = RBTree(sctx=sctx, split_rigid=0)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Box(3, 1, 8, 1),
          link_data=LinkData(
              spec=LinkSpec(type=RigidBodyLinkType.PIVOT_Z, rl=Transform.From(pos=[0, -4, 0]))
          ),
      )
  )

  res = tx.create(root)

  fm = ForceModel.Id()
  fm.bias_force_f = bias_force_gy
  return SceneData(sctx=sctx, fm=fm)


def inv_pendulum():
  sctx = SceneContext()
  tx = RBTree(sctx=sctx, split_rigid=0)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Box(3, 1, 1, 3),
          link_data=LinkData(
              spec=LinkSpec(
                  type=RigidBodyLinkType.XLT_Z,
                  wr=Transform.From(rot=make_rot_ab(x=Vec3.Z(), y=Vec3.Y()))
              )
          ),
      )
  )

  axis_l = 10
  axis = tx.add(
      RBDescEntry(
          data=RBData(base_name='axis'),
          link_data=LinkData(
              spec=LinkSpec(
                  type=RigidBodyLinkType.PIVOT_Z,
                  wr=Transform.From(rot=make_rot_ab(y=Vec3.Z(), x=Vec3.Y()), pos=[0, 0, 0]),
                  rl=Transform.From(pos=[0, -axis_l / 2, 0])
              )
          ),
          spec=SolidSpec.Box(0.2, 0.5, axis_l, 0.2),
          parent=root,
      )
  )

  tx.add(
      RBDescEntry(
          data=RBData(),
          link_data=LinkData(
              spec=LinkSpec(
                  type=RigidBodyLinkType.RIGID, wr=Transform.From(pos=[0, -axis_l / 2, 0])
              )
          ),
          spec=SolidSpec.Sphere(2, 1),
          parent=axis,
      )
  )
  res = tx.create(root)

  def ctrl2model(sim, ctrl):
    res = np.zeros(sctx.sys_spec.ctrl_packer.pos)
    res = g_oph.set(res, ctrl[0:1])[0:1]  # z xlt
    return res

  fm = ForceModel(
      nctrl_f=lambda n_: 1,
      model2ctrl=lambda sim, model: model[0:1],
      ctrl2model=ctrl2model,
      bias_force_f=bias_force_g,
  )
  return SceneData(sctx=sctx, fm=fm)


def control_simple(nx):
  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Box(3, 1, 1, 1),
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.FREE,)),
      )
  )

  wheel_r = 0.5
  wheel_h = 0.3
  data = [
      A(
          name='wh_x',
          wr=Transform.From(rot=make_rot(z=Vec3.X(), x=(Vec3.Y() + Vec3.Z()).uvec), pos=[2, 0, 0])
      ),
      A(name='wh_y', wr=Transform.From(rot=make_rot(z=Vec3.Y(), y=Vec3.Z()), pos=[0, 2, 0])),
      A(name='wh_z', wr=Transform.From(rot=make_rot(z=Vec3.Z(), y=Vec3.Y()), pos=[0, 0, 2])),
  ][:nx]

  for i, dx in enumerate(data):
    tx.add(
        RBDescEntry(
            data=RBData(base_name=dx.name),
            link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.PIVOT_Z, wr=dx.wr)),
            spec=SolidSpec.Cylinder(0.2, wheel_r, wheel_h),
            parent=root,
        )
    )
  res = tx.create(root)

  def ctrl2model(sim, ctrl):
    res = np.zeros(sctx.sys_spec.ctrl_packer.pos)
    return g_oph.set(res, ctrl)[6:]

  fm = ForceModel(
      nctrl_f=lambda n_: nx,
      model2ctrl=lambda sim, model: model[6:],
      ctrl2model=ctrl2model,
  )
  return SceneData(sctx=sctx, fm=fm)


scene_gyro_wheel_idx = SceneRegistar.Registar.register('scene_gyro_wheel', scene_gyro_wheel)
box_scene_idx = SceneRegistar.Registar.register('box_scene', box_scene)
control_simple_idx1 = SceneRegistar.Registar.register('control_simple1', lambda: control_simple(1))
control_simple_idx2 = SceneRegistar.Registar.register('control_simple2', lambda: control_simple(2))
control_simple_idx3 = SceneRegistar.Registar.register('control_simple3', lambda: control_simple(3))
simple_gen = SceneRegistar.Registar.register('simple_gen', control_simple)

scene_T_idx = SceneRegistar.Registar.register('scene_T', scene_T)
scene_T2D_idx = SceneRegistar.Registar.register('scene_T2D', scene_T2D)
two_mass_idx = SceneRegistar.Registar.register('two_mass', two_mass)
inv_pendulum_idx = SceneRegistar.Registar.register('inv_pendulum', inv_pendulum)
pivot_idx = SceneRegistar.Registar.register('pivot', pivot)


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
