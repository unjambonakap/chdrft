#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.utils.misc import Attributize as A
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.types import *
from chdrft.external.gamepad.controllers import Xbox360
import chdrft.external.gamepad.gamepad as gamepad
from chdrft.utils.rx_helpers import ImageIO
from pydantic import Field
from chdrft.utils.path import FileFormatHelper
from chdrft.dsp.utils import linearize_clamp
from typing import Callable
import enum

global flags, cache
flags = None
cache = None

class GamepadButtons(str, enum.Enum):
  X = 'X'
  Y = 'Y'
  A = 'A'
  B = 'B'

def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class OGamepad(cmisc.PatchedModel):
  gp: gamepad.Gamepad = None
  src: ImageIO = Field(default_factory=ImageIO)
  state_src: ImageIO = Field(default_factory=ImageIO)
  state: dict = Field(default_factory=dict)
  axis2button: dict = None

  def register_button(self, name):
    self.state[name] = 0
    self.state[(name, False)] = 0
    self.state[(name, True)] = 0

  def set_led(self, value):
    FileFormatHelper.Write('/sys/class/leds/xpad0/brightness', str(value))

  def __init__(self, num=0):
    super().__init__()
    self.gp: Gamepad = Xbox360(num)
    self.axis2button = {
        'DPAD-X': {
            -1: 'DPAD-LEFT',
            1: 'DPAD-RIGHT'
        },
        'DPAD-Y': {
            -1: 'DPAD-UP',
            1: 'DPAD-DOWN'
        },
    }

    for x in self.axis2button.values():
      for k in x.values():
        self.register_button(k)

    def add_cb(name, axis=False):
      self.register_button(name)
      if axis:
        self.gp.addAxisMovedHandler(name, lambda pos: self.cb(name, val=pos))
      else:
        self.gp.addButtonChangedHandler(name, lambda pressed: self.cb(name, pressed=pressed))

    for x in self.gp.axisNames.values():
      add_cb(x, axis=True)
    for x in self.gp.buttonNames.values():
      add_cb(x, axis=False)

  def cleanup(self):
    self.gp.removeAllEventHandlers()

  def cb(self, name, val=None, pressed=None):
    if val is not None and name in self.axis2button:
      for tval, mp in self.axis2button[name].items():
        self.cb(mp, pressed=(abs(val - tval) < 1e-5))

    self.state[name] = pressed if val is None else val

    if val is None:
      self.state[(name, pressed)] = True

    self.src.push(A(name=name, val=val, pressed=pressed))
    self.push_state()
    if val is None:
      self.state[(name, pressed)] = False

  def push_state(self):
    self.state_src.push(dict(self.state))

  def start(self):
    self.gp.startBackgroundUpdates()
    self.push_state()

  def dispose(self):
    self.gp.disconnect()
    self.src.dispose()
    self.state_src.dispose()


class InputControllerParameters(cmisc.PatchedModel):
  map: Callable[[int], int] = Field(default_factory=lambda: cmisc.identity)
  controller2ctrl: Callable[[SceneControllerInput], np.ndarray]


class SceneControllerInput(cmisc.PatchedModel):
  inputs: dict[int, float] = Field(default_factory=lambda: cmisc.defaultdict(float))
  parameters: InputControllerParameters = Field(default_factory=InputControllerParameters)
  speed: float = 1
  scale: float = 1
  stop: bool = False
  use_jit: bool = True
  reset: bool = False
  precision: int = 1
  fix_mom: bool = True
  mod1: bool = False
  mod2: bool = False
  mod3: bool = False

  @property
  def scaled_ctrl(self) -> dict[int, float]:
    return {k: v * self.scale for k, v in self.inputs.items()}

  @property
  def scaled_ctrl_packed(self) -> np.ndarray:
    ctrl = np.zeros(max(self.inputs.keys()))
    for k, v in self.scaled_ctrl.items():
      ctrl[k] = v
    return ctrl

  def get_ctrl(self) -> np.ndarray:
    if self.parameters.controller2ctrl: return self.parameters.controller2ctrl(self)
    return self.scaled_ctrl_packed

  def set(self, x, v):
    v = cmisc.sign(v) * linearize_clamp(abs(v), 0.1, 1, 0, 1)
    self.inputs[self.parameters.map(x)] = v

  def update_speed(self, ns: float):
    self.speed = max(ns, 0)


InputControllerParameters.update_forward_refs()

def jupyter_print(x):
  from IPython.display import clear_output
  clear_output()
  print(cmisc.json_dumps(x))


def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
