#!/usr/bin/env python

import numpy as np
from chdrft.utils.opa_types import *
import chdrft.utils.Z as Z

from chdrft.sim.blender import BlenderPhysHelper
from chdrft.display.blender import clear_scene
from chdrft.sim.rb.rb_gen import *
from chdrft.sim.rb.ctrl import *
from chdrft.sim.rb import blender_helper

from chdrft.sim.rb.scenes import *
from chdrft.geo.satsim import gl
from chdrft.sim.base import *
from chdrft.sim.rb.blender_helper import *
from chdrft.sim.blender import *
from chdrft.config.env import init_jupyter

init_jupyter(run_app=True) # required to make qt work

global flags, cache
flags = None
cache = None


def go():
  clear_scene()
  ss = SceneSaver.Load(FileFormatHelper.Read('/tmp/state.pickle'))

  ll = np.zeros(2)
  if 1:
    p = gl.geocode('Paris')
    ll = Z.deg2rad(np.array([p.longitude, p.latitude]))

  md = 17
  u = create_earth_actors(
    BlenderTriangleActor,
    max_depth=md,
    tile_depth=md,
    params = TMSQuadParams(m2u=1, ell=Consts.EARTH_ROUND_ASTROPY),
    ll_box=Box(center=ll, size=(np.pi / 50000, np.pi / 50000))
  )

  pts = np.array(u.points)
  center = Vec3(np.mean(pts, axis=0))
  pts -= center.vdata

  helper = BlenderPhysHelper(shift_p=-center)
  a = BlenderObjWrapper(actors_to_obj('x0', helper.main_col, u.actors))
  a.internal.location = np.array(a.internal.location) - center.vdata
  helper.update_context()
  print(a.wl, a.internal.location)


  helper.set_cam_focus(a.aabb_w.points, dv_abs = Vec3.X() * 300, expand=1.2)
  def set_cam():
    bb = rh.bobj.aabb_w
    helper.set_cam_focus(bb.points, dv_abs = Vec3(np.array([50, 50, 50])), expand=1.2)

  rh = blender_helper.ReplayHelper(ss=ss, helper=helper, cb=set_cam)

  rh.make()



helper = go()
