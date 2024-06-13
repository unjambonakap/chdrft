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
from typing import Tuple, Callable
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
from chdrft.sim.rb.rb_gen import *
from chdrft.inputs.controller import OGamepad, SceneControllerInput, InputControllerParameters, GamepadButtons
from chdrft.utils.rx_helpers import ImageIO, rx, pipe_connect
from pydantic.v1 import Field
from chdrft.display.ui import TimerHelper

global flags, cache
flags = None
cache = None

class ButtonKind(int, Enum):
  LTRT = 0
  LEFT_X = 1
  LEFT_Y = 2
  RIGHT_X = 3
  RIGHT_Y = 4

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
      ('LT', 'RT'): lambda s, v: s.set(ButtonKind.LTRT, mixaxis(*v)),
      'LEFT-X': lambda s, v: s.set(ButtonKind.LEFT_X, v),
      'LEFT-Y': lambda s, v: s.set(ButtonKind.LEFT_Y, v),
      'RIGHT-X': lambda s, v: s.set(ButtonKind.RIGHT_X, v),
      'RIGHT-Y': lambda s, v: s.set(ButtonKind.RIGHT_Y, v),
      ('XBOX', True): lambda s: s.__setattr__('use_jit', not s.use_jit),
      'BACK': lambda s, v: s.__setattr__('reset', v),
      ('LB', True): lambda s: s.__setattr__('scale', max(s.scale - 1, 0)),
      ('RB', True): lambda s: s.__setattr__('scale', s.scale + 1),
      ('START', True): lambda s: s.__setattr__('stop', not s.stop),
      (GamepadButtons.X, True): lambda s: s.__setattr__('fix_mom', not s.fix_mom),
      GamepadButtons.Y: lambda s, v: s.__setattr__('mod1', v),
      GamepadButtons.A: lambda s, v: s.__setattr__('mod2', v),
      GamepadButtons.B: lambda s, v: s.__setattr__('mod3', v),
      ('DPAD-LEFT', True): lambda s: s.update_speed(s.speed - 1),
      ('DPAD-RIGHT', True): lambda s: s.update_speed(s.speed + 1),
      ('DPAD-UP', True): lambda s: s.update_speed(s.speed * 1.2),
      ('DPAD-DOWN', True): lambda s: s.update_speed(s.speed / 1.2),
  }
  for x in list(ButtonKind):
    state.set(x, 0)

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

  return ImageIO(proc, last=state)


class SceneController(cmisc.PatchedModel):
  gp: OGamepad = None
  fspec: ControlFullSpec = None
  sh: SimulHelper = None
  sim: Simulator = None
  dt_base: float = 1e-2
  ctrl_obs: ImageIO = None
  obs_scene: ImageIO = None
  use_jit: bool = False
  use_gamepad: bool = True
  parameters: InputControllerParameters = None
  s0: SceneControllerInput = None
  mom: SpatialVector = None
  override_ctrl: np.ndarray | Callable = None

  def dispose(self):
    if self.gp is None: return
    self.gp.dispose()
    self.gp = None

  def setup(self):
    self.dispose()
    s0 = SceneControllerInput(parameters=self.parameters)
    s0.stop = True
    self.s0 = s0
    self.ctrl_obs = gamepad2scenectrl_io(s0)

    self.sh = SimulHelper.Get(self.fspec.spec)
    self.sim = self.sh.sim
    self.sim.load_state(self.fspec.consts.state)
    self.obs_scene = ImageIO(f=self._update_scene)
    self.mom = self.sim.mom
    if self.use_gamepad:
      self.gp = OGamepad()
      pipe_connect(
          self.gp.state_src,
          self.ctrl_obs,
      )
      self.gp.start()

  def set_override(self, override: np.ndarray | Callable):
    if not callable(override) and override is not None:
      res = lambda *args: override
    else:
      res = override
    self.override_ctrl = res

  def update(self):
    if self.ctrl_obs.last:
      self.obs_scene.push(self.ctrl_obs.last)

  @cmisc.logged_failsafe
  def _update_scene(self, input: SceneControllerInput):
    if input.reset:
      self.sim.load_state(self.fspec.consts.state)
      return
    if input.stop: return
    if input.speed <= 0: return
    ctrl = input.get_ctrl()

    for i in range(input.precision):
      if self.override_ctrl is not None:
        ctrl = self.override_ctrl(self.sim)
      self.sh.integrate1(
          self.dt_base * input.speed / input.precision,
          ctrl,
          use_jit=self.use_jit and input.use_jit,
      )
    if not input.fix_mom or self.mom is None:
      self.mom = self.sh.sim.mom
    else:
      self.sh.fix_mom(self.mom)
    return ImageIO.kNop

  def debug(self, gamepad_state=False, control_state=False):
    if not self.gp: return
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
