#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.types import *
from pydantic import Field
from chdrft.utils.path import FileFormatHelper
from chdrft.struct.base import Range1D
import math
import chdrft.dsp.base as dsp_base

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  pass


class Integrator:

  def __init__(self, a=0.95, clamp=None):
    self.a = a
    self.v = 0
    self.n = 0
    self.clamp = clamp

  def add(self, x):
    self.v = self.a * self.v + (1 - self.a) * x
    if self.clamp: self.v = self.clamp(self.v)
    return self.v


class Derivator:

  def __init__(self, ts=1, nv=2):
    self.ts = ts
    self.nv = nv
    self.tb = []
    self.v = 0
    self.iir = dsp_base.SimpleIIR(alpha=0.5)

  def add(self, x):
    self.tb.append(x)
    if len(self.tb) < self.nv: return 0
    if len(self.tb) > self.nv: self.tb.pop(0)
    self.v = self.iir.push((self.tb[-1] - self.tb[0]) / ((self.nv - 1) * self.ts))
    return self.v


class PIDController:

  def __init__(
      self,
      kp=0,
      ki=0,
      kd=0,
      control_range=Range1D(-math.inf, math.inf),
      step_sec=1,
      derivator_args={},
  ):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.step_sec = step_sec
    self.control_range = control_range
    self.i_v = 0
    self.d_v = Derivator(self.step_sec, **derivator_args)

  def push(self, err):
    self.i_v += err
    self.d_v.add(err)
    vkp = self.kp * err
    vki = self.ki * self.i_v
    vkd = self.kd * self.d_v.v
    ctrl_val = vkp + vki + vkd
    final_val = self.control_range.clampv(ctrl_val)
    return final_val


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
