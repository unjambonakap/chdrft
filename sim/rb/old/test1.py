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
from chdrft.sim.rb.old.rb import *
from chdrft.sim.rb.old.rb_analysis import *
import chdrft.utils.Z as Z

from chdrft.sim.blender import BlenderPhysHelper, ObjectSync
from chdrft.display.blender import clear_scene, AnimationSceneHelper, KeyframeObjData
from chdrft.sim.base import compute_cam_parameters
from chdrft.sim.rb.old.rb_player import SceneController
import chdrft.sim.rb.old.scenes as scenes

from chdrft.utils.rx_helpers import ImageIO
from chdrft.display.ui import TimerHelper
from chdrft.config.blender import init_blender
import bpy

#init_blender()  # required to make qt work

global flags, cache
flags = None
cache = None


def go():

  clear_scene()
  #sol = Z.FileFormatHelper.Read('/home/benoit/programmation/res.pickle')
  #params = sol.ctrl_params

  use_jit = 0
  nnx = 3
  mode = 'scene'
  data = []

  dt = 1e-3
  if mode == 'ctrl':
    fspec = scenes.get_control_test_data(nnx)
    data += list(sol.ctrl)
    data += [[0] * fspec.spec.mdata.nctrl] * 0
  else:
    fspec =  scenes.scene_T()
    data += [[0] * fspec.spec.mdata.nctrl] * 300
    params = fspec.spec.cparams
    params.dt = dt

  ctrl = SceneController(
      fspec=fspec,
      dt_base=dt,
      use_jit=use_jit,
      use_gamepad=0,
  )
  ctrl.setup()

  sx = ctrl.sim

  tg = sx.root
  sctx = sx.sctx

  #sx.load_state(**params.i_state)

  helper = BlenderPhysHelper()
  helper.load(tg)
  cam_loc = np.array([5, 0, 0])
  aabb = tg.self_link.aabb()
  cam_params = compute_cam_parameters(
      cam_loc, aabb.center, Vec3.Z().vdata, aabb.points[:, :3], blender=True
  )
  helper.cam.data.angle_y = cam_params.angle_box.yn
  helper.cam.mat_world = cam_params.toworld

  #r0l = sctx.compose([wlink])
  #tg = r0l
  #px.plot(by_col=True, fx=0.2)

  print(f'start >> ', tg.self_link.world_angular_momentum())
  #for i in range(10):
  #  print('FUU ', tg.self_link.get_particles(10000000).angular_momentum())
  #return

  #osync = ObjectSync(helper.obj2blender[rbl.child], helper.cam)

  animation = AnimationSceneHelper(frame_step=1)
  animation.start()
  helper.update(animation)

  sx.load_state_input(fspec.consts.state)
  #data += [np.identity(params.nctrl)[0] * 10] * (100 // (params.integ_nsteps * params.ndt_const))
  #data += [np.identity(params.nctrl)[0] * 0] * (200 // (params.integ_nsteps * params.ndt_const))

  if 0:
    #ctrl.debug(True, True)
    #ctrl.obs_scene.connect_to(ImageIO(f=lambda *args: helper.update()))

    def do_update():
      #ctrl.update()
      ctrl._update_scene(ctrl.ctrl_obs.last)
      helper.update()
      return 1e-1

    bpy.app.timers.register(do_update)
    #th = TimerHelper()
    #th.add(ctrl.update, 1e-1)
    #th.add(lambda: print('123'), 1)
    #input()
    return helper

  t = 0
  for i, ex in enumerate(data):
    for ix in range(params.ndt_const):
      print()
      print()
      print()
      print()
      print(t, i, len(data))
      t += params.dt

      for jk in range(params.integ_nsteps):
        ctrl.anx.advance(sx, params.actual_dt, ex, use_jit=use_jit)
        #sx.update(params.actual_dt, ex)

      for x in sctx.obj2name.keys():
        rl = x.self_link
        print(
            f'>>>>>>>> {x.name=} {rl.rb.root_angular_momentum()=} {rl.rotvec_w=} {rl.rotvec_l=} {rl.world_angular_momentum()=} {rl.com_w=}'
        )
      print(f'{i:03d} {i*params.dt:.2f}', tg.self_link.world_angular_momentum())
      print(sx.dump_state())

      helper.update(animation)

  animation.finish()
  cam = helper.cam

  return helper


helper = go()
