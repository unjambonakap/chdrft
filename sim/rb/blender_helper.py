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
import chdrft.utils.Z as Z

from chdrft.sim.blender import BlenderPhysHelper, ObjectSync
from chdrft.display.blender import clear_scene, AnimationSceneHelper, KeyframeObjData
from chdrft.sim.base import compute_cam_parameters
from chdrft.sim.rb.rb_gen import *
from chdrft.sim.rb.rb_player import SceneController, InputControllerParameters
from chdrft.sim.rb.ctrl import *

from chdrft.utils.rx_helpers import ImageIO
from chdrft.display.ui import TimerHelper
from chdrft.config.blender import init_blender
from chdrft.sim.rb.scenes import *
import bpy

global flags, cache
flags = None
cache = None


class BlenderRBHelper:
  Singleton = 0

  def __init__(self):
    assert not BlenderRBHelper.Singleton
    BlenderRBHelper.Singleton = 1

    self.running_sim = False
    self.timer_freq = 60 * 2
    self.setup_timer()

  @cmisc.logged_failsafe
  def do_update(self):
    if self.running_sim:
      self.ctrl.update()
      self.helper.update()
      mom = self.root.compute_momentum(Transform.From(), SpatialVector.Vector())
      print(self.ctrl.sh.t, 'mom >>> ', mom.around(self.root.agg_com), self.root.agg_com)

  def setup_timer(self):

    def timer():
      self.do_update()
      return 1 / self.timer_freq

    self.timer = bpy.app.timers.register(timer)

  def load_scene(
      self,
      scene_idx: str,
      q0: np.ndarray = None,
      qd0: np.ndarray = None,
      use_gamepad=0,
      use_jit=0,
      cparams: ControlParameters = None,
  ):
    self.running_sim = False
    clear_scene()

    mdata = ModelData(func_id=scene_idx)
    if q0 is None:
      q0 = mdata.ss.q_desc.default
    if qd0 is None:
      qd0 = mdata.ss.qd_desc.default
    if cparams is None:
      cparams = ControlParameters(ndt_const=1, dt=1e-2, integ_nsteps=1, nt=5, use_jit=use_jit)

    i0 = ControlInputState(q=q0, qd=qd0)

    cspec = ControlSpec(mdata=mdata, cparams=cparams, end_conds=None)

    fspec = ControlFullSpec(consts=ControlInput(state=i0), spec=cspec)

    ctrl = SceneController(
        fspec=fspec,
        dt_base=cparams.dt,
        use_jit=cparams.use_jit,
        use_gamepad=use_gamepad,
        parameters=InputControllerParameters(map=lambda x: x),
    )

    ctrl.setup()
    ctrl.s0.stop = 0
    ctrl.s0.precision = 1
    ctrl.s0.fix_mom = False

    root = ctrl.sim.rootl

    helper = BlenderPhysHelper()
    helper.load(root.rb)
    cam_loc = np.array([5, 0, 0])
    aabb = root.aabb()
    cam_params = compute_cam_parameters(
        cam_loc, aabb.center, Vec3.Z().vdata, aabb.points[:, :3], blender=True
    )
    helper.cam.data.angle_y = cam_params.angle_box.yn
    helper.cam.mat_world = cam_params.toworld

    #osync = ObjectSync(helper.obj2blender[rbl.child], helper.cam)
    ctrl.debug(True, True)

    self.sim = ctrl.sim
    self.helper = helper
    self.ctrl = ctrl
    self.fspec = fspec
    self.root = root
    self.i0 = i0
    self.sim.load_state(i0)
    self.sh = ctrl.sh



  def run_sim(self, override_ctrl: np.ndarray|Callable = None):
    self.running_sim = True
    self.ctrl.set_override(override_ctrl)
    self.do_update()

  def stop_sim(self):
    self.running_sim = False

  def make_animation(self, states, frame_step=1):
    self.running_sim = False
    animation = AnimationSceneHelper(frame_step=frame_step)
    animation.start()

    for i, state in enumerate(states):
      self.sim.load_state(state)
      self.helper.update(animation)

    animation.finish()


g_bh = BlenderRBHelper()
