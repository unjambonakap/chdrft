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
from chdrft.utils.opa_types import *
from chdrft.external.gamepad.controllers import Xbox360
import chdrft.external.gamepad.gamepad as gamepad
from chdrft.utils.rx_helpers import ImageIO
from chdrft.utils.path import FileFormatHelper
from chdrft.dsp.utils import linearize_clamp
from typing import Callable
import enum
import time
import contextlib

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


class JoystickHelper:
  @classmethod
  def find_number(cls, typ: str) -> int:
    assert typ == 'xbox360'
    for group in FileFormatHelper.Read('/proc/bus/input/devices', mode='txt').split('\n\n'):
      if 'Name="Microsoft X-Box 360 pad"' in group:
        return int(cmisc.re.search(r'Handlers=[^ ]+ js(?P<num>\d)', group).group('num'))

    return None 



class OGamepad(cmisc.PatchedModel):
  gp: gamepad.Gamepad = None
  src: ImageIO = cmisc.pyd_f(ImageIO)
  state_src: ImageIO = cmisc.pyd_f(ImageIO)
  state: dict = cmisc.pyd_f(dict)
  prev_state: dict = None
  axis2button: dict = None

  def register_button(self, name):
    self.state[name] = 0
    self.state[(name, False)] = 0
    self.state[(name, True)] = 0

  def set_led(self, value):
    FileFormatHelper.Write('/sys/class/leds/xpad0/brightness', str(value))

  def __init__(self, num=-1):
    super().__init__()
    if num == -1:
      num = JoystickHelper.find_number('xbox360')
      assert num is not None

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
    if self.prev_state is None:
      self.prev_state = self.state

    self.push_state()
    if val is None:
      self.state[(name, pressed)] = False

  def push_state(self):
    self.state_src.push(dict(self.state))

  def start(self):
    self.gp.startBackgroundUpdates(waitForReady=True)
    for k, v in self.gp.axisMap.items():
      self.cb(self.gp.axisNames[k], val=v)
    for k, v in self.gp.pressedMap.items():
      self.cb(self.gp.buttonNames[k], pressed=v)

    self.push_state()

  def dispose(self):
    self.gp.disconnect()
    self.src.dispose()
    self.state_src.dispose()


class InputControllerParameters(cmisc.PatchedModel):
  map: Callable[[int], int] = cmisc.pyd_f(lambda: cmisc.identity)
  controller2ctrl: Callable[[SceneControllerInput], np.ndarray] = None


class SceneControllerInput(cmisc.PatchedModel):
  inputs: dict[int, float] = cmisc.pyd_f(lambda: cmisc.defaultdict(float))
  parameters: InputControllerParameters = cmisc.pyd_f(InputControllerParameters)
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
  cur_buttons: dict = None

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
    if isinstance(v, float):
      v = cmisc.sign(v) * linearize_clamp(abs(v), 0.1, 1, 0, 1)
    self.inputs[self.parameters.map(x)] = v

  def inc(self, x):
    self.inputs[self.parameters.map(x)] += 1

  def get(self, x):
    return self.inputs[self.parameters.map(x)]

  def update_speed(self, ns: float):
    self.speed = max(ns, 0)


InputControllerParameters.update_forward_refs()

def jupyter_print(x):
  from IPython.display import clear_output
  clear_output()
  print(cmisc.json_dumps(x))


@contextlib.contextmanager
def configure_joy(gp: OGamepad, mp, init_dict={}):
  cur = SceneControllerInput()

  def proc(tb):
    cur.cur_buttons =tb
    for k, cb in mp.items():
      if isinstance(k, tuple):
        if isinstance(k[-1], bool):
          if tb.get(k):
            cb(cur)
        else:
          cb(cur, tuple(tb[x] for x in k))
      else:
        val = tb.get(k)
        cb(cur, val)


    return cur
  cur.inputs.update(init_dict)

  state= ImageIO(proc, last=cur)
  gp.state_src.connect_to(state)
  gp.start()
  try:
    yield state
  finally:
    gp.dispose()

def debug_controller_state(state: ImageIO, cb_ser=lambda x: cmisc.json_dumps(x.inputs)):
  import rich.live, rich.json
  with rich.live.Live(refresh_per_second=10) as live:
    while True:
      live.update(rich.json.JSON(cb_ser(state.last)))
      time.sleep(0.05)

def test_controller(ctx):
  mp = {
      'LEFT-X': lambda s, v: s.set('yaw', v),
      'RIGHT-Y': lambda s, v: s.set('pitch', v),
      'LB': lambda s, v: s.set('LB', v),
      'RB': lambda s, v: s.set('RB', v),
      'DPAD-X': lambda s, v: s.set('dpad-x', v),
      ('X', True): lambda s: s.inc('x_press_count'),
      ('LB', 'RB'): lambda s, v: s.set('combo', v),
  }

  gp = OGamepad()
  with configure_joy(gp, mp) as state:
    state.last.inputs['x_press_count'] = 0
    debug_controller_state(state)



def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
