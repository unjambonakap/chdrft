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
from typing import Callable
from chdrft.sim.rb.old.rb import *
from chdrft.sim.rb.old.rb_analysis import *
from chdrft.inputs.controller import OGamepad, SceneControllerInput
from chdrft.utils.rx_helpers import ImageIO, rx, pipe_connect
from pydantic.v1 import Field
from chdrft.display.ui import TimerHelper

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--use-jit', action='store_true')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def gamepad2scenectrl_io(state: SceneControllerInput = None) -> ImageIO:
  state = state or SceneControllerInput()

  def mixaxis(v1, v2):
    v = (max(v1, v2) + 1) / 2
    if v1 > v2: return -v
    return v

  mp = {
      ('LT', 'RT'): lambda s, v: s.inputs.__setitem__(0, mixaxis(*v)),
      'LEFT-X': lambda s, v: s.inputs.__setitem__(1, v),
      'RIGHT-X': lambda s, v: s.inputs.__setitem__(2, v),
      ('XBOX', True): lambda s: s.__setattr__('use_jit', not s.use_jit),
      ('LB', True): lambda s: s.__setattr__('scale', max(s.scale - 1, 0)),
      ('RB', True): lambda s: s.__setattr__('scale', s.scale + 1),
      ('START', True): lambda s: s.__setattr__('stop', not s.stop),
      ('DPAD-LEFT', True): lambda s: s.update_speed(s.speed - 1),
      ('DPAD-RIGHT', True): lambda s: s.update_speed(s.speed + 1),
      ('DPAD-UP', True): lambda s: s.update_speed(s.speed * 1.2),
      ('DPAD-DOWN', True): lambda s: s.update_speed(s.speed / 1.2),
  }

  def proc(tb):
    for k, cb in mp.items():
      if isinstance(k, tuple):
        if isinstance(k[-1], bool):
          if tb.get(k):
            cb(state)
        else:
          cb(state, (tb[x] for x in k))
      else:
        val = tb.get(k)
        cb(state, val)
    return state

  return ImageIO(proc)


class SceneController(cmisc.PatchedModel):
  gp: OGamepad = None
  fspec: ControlFullSpec = None
  anx: RB_PbAnalysis = None
  sim: Simulator = None
  dt_base: float = 1e-2
  ctrl_obs: ImageIO = None
  obs_scene: ImageIO = None
  use_jit: bool = False
  use_gamepad: bool = True

  def dispose(self):
    if self.gp is None: return
    self.gp.dispose()
    self.gp = None

  def setup(self):
    self.dispose()
    s0 = SceneControllerInput()
    s0.stop = True
    self.ctrl_obs = gamepad2scenectrl_io(s0)
    self.sim = self.fspec.spec.mdata.create_simulator()
    self.sim.load_state_input(self.fspec.consts.state)

    self.anx = RB_PbAnalysis(spec=self.fspec.spec)
    self.anx.setup(setup_jit=self.use_jit)
    self.obs_scene = ImageIO(f=self._update_scene)
    if self.use_gamepad:
      self.gp = OGamepad()
      pipe_connect(
          self.gp.state_src,
          self.ctrl_obs,
      )
      self.gp.start()


  def update(self):
    self.obs_scene.push(self.ctrl_obs.last)

  def _update_scene(self, input: SceneControllerInput):
    if input.stop: return
    if input.speed <= 0: return
    ctrl = np.zeros(self.fspec.spec.cparams.nctrl)
    for k, v in input.inputs.items():
      if isinstance(k, int):
        ctrl[k] = v

    ctrl = ctrl * input.scale

    self.anx.advance(self.sim, self.dt_base * input.speed, ctrl, use_jit=self.use_jit and input.use_jit)
    return ImageIO.kNop

  def debug(self, gamepad_state=False, control_state=False):
    from chdrft.display.service import oplt
    if gamepad_state:
      oplt.plot(dict(obs=self.gp.state_src), 'metric')

    if control_state:
      oplt.plot(dict(obs=self.ctrl_obs), 'metric')


g_controller = SceneController()


def test1(ctx):
  nnx = 3
  cdata = get_control_test_data(nnx)

  ctrl = SceneController(cdata=cdata, use_jit=ctx.use_jit)
  ctrl.setup()
  ctrl.debug(True, True)

  with TimerHelper.Create() as th:
    th.add(ctrl.update, 1e-1)
    input()


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
