#!/usr/bin/env python


import itertools

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import numpy as np




def norm_angle(x):
  while abs(x) > math.pi:
    if x > 0:
      x -= 2 * math.pi
    else:
      x += 2 * math.pi
  return x


def norm_angle_from(src, dest):
  return src + norm_angle(dest - src)

def linearize(x, a, b, na, nb):
  ratio = (nb - na) / (b - a)
  return (x - a) * ratio + na


def linearize_clamp(x, a, b, na, nb):
  tmp = linearize(x, a, b, na, nb)
  return np.clip(tmp, na, nb)



def to_db(p):
  return 10 * np.log(np.abs(p) + 1e-9)



def argmax2d(a):
  return np.unravel_index(np.argmax(a, axis=None), a.shape)


def argmin2d(a):
  return np.unravel_index(np.argmin(a, axis=None), a.shape)


def iter_shape(shape):
  xr = map(range, shape)
  return itertools.product(*list(xr))

