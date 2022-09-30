#!/usr/bin/env python


from IPython.utils.frame import extract_module_locals
from asq.initiators import query as asq_query
from enum import Enum
from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE, USE_PYQT5
from rx import operators as ops
from scipy import signal
from scipy.stats.mstats import mquantiles
import cv2
import glog
import glog
import itertools
import math
import math, sys, os
import numpy as np
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import scipy.ndimage as ndimage
import sys
import tempfile

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.struct.base import Box, Range2D, g_unit_box
from chdrft.struct.base import Range1D, Range2D, Intervals, ListTools, Box
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.colors import ColorPool
from chdrft.utils.fmt import Format
from chdrft.utils.fmt import Format
from chdrft.utils.misc import Attributize
from chdrft.utils.misc import Attributize as A
from chdrft.utils.misc import to_list, Attributize, proc_path
from chdrft.utils.path import FileFormatHelper
from chdrft.utils.swig import swig
from chdrft.utils.types import *
import chdrft.dsp.datafile as Dataset2d

import chdrft.display.base
import chdrft.utils.misc as cmisc
import chdrft.utils.misc as cmisc

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import math, sys, os
import numpy as np
from chdrft.utils.types import *
import cv2
import chdrft.utils.math as opa_math
import chdrft.utils.geo as opa_geo

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def warp_affine_xy(img, mat, dim):
  nmat = np.array(mat[:2,:3])
  #nmat[0,2] = mat[1,2]
  #nmat[1,2] = mat[0,2]
  return cv2.warpAffine(np.array(img), nmat, tuple(dim), flags=cv2.WARP_INVERSE_MAP)

def genbox_subimg(img, gb):
  mat = gb.normed.mat2world
  dim = np.ceil(gb.norms).astype(int)
  return warp_affine_xy(img, mat, dim)


def shift_image(im, offset):
  dx, dy = offset
  if dx <= 0: im = im[:, -dx:]
  else: im = im[:, :-dx]

  if dy <= 0: im = im[-dy:,]
  else: im = im[:-dy, :]
  return im




def downscale_img(img, downscale):
  target_scale = np.rint(np.array(img.shape[:2]) / downscale).astype(int)
  return cv2.resize(img, tuple(target_scale[::-1]))



def plot_img(img):
  from chdrft.display.ui import GraphHelper, ImageEntry
  G = GraphHelper()
  p1 = Dataset2d(img)
  G.create_plot(images=[p1])
  G.run()


def norm_img(img):
  if np.max(img) > 1.1:
    img = img / 255.
  return img

def to_uint8_image(res):
  if res.dtype == np.float64:
    res = (res * 255).astype(np.uint8)
  return res


def save_image(fname, res):
  cv2.imwrite(fname, to_uint8_image(res))

def read_tiff(fname):
  res = cv2.imread(fname, 0)
  assert res is not None, fname
  return norm_img(res)


class ImageData(cmisc.Attr):

  @staticmethod
  def Make(data):
    if isinstance(data, ImageData):
      return data.clone()
    if isinstance(data, dict):
      return ImageData(**data)
    if isinstance(data, np.ndarray):
      return ImageData(img=data)
    return ImageData(**data)

  @staticmethod
  def Zero(box, u2px, yx=1, **kwargs):
    b = (box * u2px).to_int_round().make_zero_image(dtype=np.float, yx=yx)
    return ImageData(b, box=box, yx=yx, **kwargs)


  def __init__(
      self,
      img=None,
      pos=None,
      box=None,
      gridpos=None,
      _obj=None,
      inv=0,
      nobuild=0,
      zpos=None,
      store_on_disk=0,
    yx=1,
    stuff=None,
      **kwargs,
  ):
    super().__init__()
    if nobuild:
      return
    self.data = A(kwargs)
    self.stuff = stuff

    self.yx = yx
    self.zpos = zpos
    self.gridpos = cmisc.to_numpy(gridpos)
    self._store_on_disk = store_on_disk
    self._img = None

    if box is None:

      if pos is not None:
        box = Box.FromImage(img, self.yx) + pos
      elif gridpos is not None:
        box = (g_unit_box + gridpos) * Box.FromImage(img, self.yx).dim
      elif img is not None:
        box = Box.FromImage(img, self.yx)
        if inv:
          box = Box(xr=box.xr, yr=(box.yh, box.yl))

    self.configure_box(box)
    self.set_image(img)
    self._obj = _obj

  def configure_box(self, box):
    self.pos = box.low
    self.box = box
    self.dim = box.dim
    self.geo = box.shapely

  @property
  def img(self):
    if self._store_on_disk:
      return read_tiff(self._img)
    return self._img

  def set_image(self, v):
    if self._store_on_disk:
      dir = os.environ.get('IMAGE_CACHE_DIR', '/tmp/cache')
      cmisc.makedirs(dir)
      tx = cmisc.tempfile.mktemp(suffix='.tif', dir=dir)
      save_image(tx, v)
      self._img = tx

    else:
      self._img = v

    if v is not None:
      self.img_box = Box.FromImage(v, self.yx)
      self.u2px = self.img_box.xn / self.dim[0]
    else:
      self.img_box = None
      self.u2px = None

  def get_at(self, p):
    pos = self.img_box.clampv(self.img_box.from_box_space(p))
    return self.img[tuple(pos[::-1])]

  def clone(self):
    res = ImageData(nobuild=1)
    for k, v in self.items():
      if isinstance(v, np.ndarray): v = np.array(v)
      if isinstance(v, Box): v = Box(v)
      res[k] = v
    return res

  def nobox(self):
    return ImageData(img=self.img)

  def upscale(self, factor):
    self.u2px *= factor
    self.set_image(downscale_img(self.img, 1 / factor))
    self.img_box = Box.FromImage(self.img, self.yx)
    return self

  @property
  def u8(self):
    r = self.clone()
    r.set_image((r.img * 255).astype(np.uint8))
    return r

  @property
  def float(self):
    r = self.clone()
    if r.img.dtype == np.uint8:
      r.set_image((r.img / 255).astype(np.float32))
    else:
      r.img.set_image(r.img.astype(np.float32))

    return r

  @property
  def obj(self):
    if self._obj is None:
      return cmisc.Attr(typ='rect', gridpos=self.gridpos, box=self.box.fix(), geo=self.box.shapely)
    return self._obj

  def __hash__(self):
    return hash(tuple(self.gridpos))

  def subimg(self, region, v=None):
    region_pix = self.box.change_rect_space(self.img_box,
                                            region).to_int_round().intersection(self.img_box)
    return region_pix.subimg(self.img, v=v, yx=self.yx)

  def plot(self, **kwargs):
    from chdrft.display.service import g_plot_service
    return g_plot_service.plot(cmisc.Attr(images=[self], **kwargs))

def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
