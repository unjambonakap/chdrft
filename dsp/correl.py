#!/usr/bin/env python


from IPython.utils.frame import extract_module_locals
from asq.initiators import query as asq_query
from enum import Enum
from pyqtgraph.Qt import QtGui, QtCore
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
from chdrft.dsp.utils import *

class FFTHelper:

  def __init__(self):
    from vispy.color import get_colormap
    self.cmap = get_colormap('viridis')

  def map0(self, f, maxv=0, minv=-60, quant=None):
    f = np.ravel(f)
    if quant is not None:
      minv, maxv = mquantiles(f, [quant, 1 - quant])

    return linearize_clamp(f, minv, maxv, 0, 1)

  def map(self, f, **kwargs):
    shape = np.shape(f)
    cf = self.cmap.map(self.map0(f, **kwargs))
    res = np.reshape(cf, shape + (4,))
    return res

class CorrelHelper:

  def __init__(
      self,
      a,
      b,
      norm=0,
      mode='full',
      auto_remap=0,
      norm_with_needle=0,
      min_overlap=(0.5, 0.5),
      **kwargs
  ):
    # a: haystack
    # b: needle
    # findind offset a+dv ~= b
    a = np.array(a)
    b = np.array(b)
    if norm or norm_with_needle: a, b = norm_for_correl(a, b, norm_with_needle=0)
    self.a = a.astype(float)
    self.b = b.astype(float)
    self.mode = mode
    self.a_box = Box.FromImage(a)
    self.b_box = Box.FromImage(b)

    self.a_shape = np.array(a.shape)[:2]
    self.b_shape = np.array(b.shape)[:2]
    if mode == 'valid': assert np.all(self.a_shape >= self.b_shape)

    self.minv = 0
    self.maxv = self.minv + 50
    self.auto_remap = auto_remap
    self.min_overlap = min_overlap

  def normalize_incomplete_correls(self):
    if self.mode == 'valid': return self
    a = self.a
    b = self.b

    for dp in iter_shape(self.res.shape):
      nf = self.get_norm_factor(self.array2offset(dp))
      self.res[dp] /= nf
    self.to_db()
    return self

  def get_norm_factor(self, offset):
    return np.product(self.get_pos_ranges(offset))

  def get_pos_ranges(self, offset):
    tb = []
    a = self.a
    b = self.b

    ra = self.a_box + offset
    inter = ra.intersection(self.b_box)
    if inter.empty:
      return [1e-9, 1e-9]
    return inter.dim

  def compute_best(self, ROI=None):
    if self.mode == 'valid' or self.min_overlap is None:
      abx = np.abs(self.res)
      bpos = np.array(argmax2d(abx))
      val = abx[tuple(bpos)]
      offset = self.array2offset(bpos)
      return cmisc.Attr(val=val, offset=offset)

    assert self.mode == 'full'
    mx = np.min([self.a_box.dim, self.b_box.dim], axis=0)

    cnds = []
    for dp in iter_shape(self.res.shape):
      offset = self.array2offset(dp)
      if ROI is not None and offset not in ROI: continue
      rx = self.get_pos_ranges(offset) / mx
      if min(rx) < min(self.min_overlap) or max(rx) < max(self.min_overlap): continue

      cnds.append(cmisc.Attr(offset=offset, val=abs(self.res[dp]), olap=rx))
    return max(cnds, key=lambda x: x.val)

  def array2offset_yx(self, v):
    if self.mode == 'valid': return np.array(v) + self.b_shape - self.a_shape
    return np.array(v) - self.a_shape + 1

  def array2offset(self, v):
    return self.array2offset_yx(v)[::-1]

  def offset2array(self, v):
    return self.offset_yx2array(v[::-1])

  def offset_yx2array(self, v):
    return -np.array(v) + self.a_shape - 1

  def compute_full_correl(self):
    self.res = compute_normed_correl2(self.a, self.b, normalized=1, mode=self.mode)
    return self

  def compute_score_at_offset(self, dp, use_mse=0, debug=0):
    dp = np.array(dp)
    ra = Range2D.FromArray_yx(self.a)
    rb = Range2D.FromArray_yx(self.b)

    ra_shift = ra + dp
    interb = ra_shift.intersection(rb)
    intera = interb - ra_shift.low

    if intera.empty: return -math.inf
    assert not intera.empty
    assert not interb.empty
    ia = intera.subimg(self.a)
    ib = interb.subimg(self.b)
    cnt = np.product(ia.shape)

    if use_mse:
      alpha = 2
      tb = np.abs(ia - ib)**alpha
      if debug:
        import chdrft.utils.K as K
        K.vispy_utils.render_image_as_grid((tb, ia, ib))
      score = -np.sum(tb / cnt)
    else:
      score = np.abs(np.sum(ia * ib)) / cnt
    return score

  def compute_at_offsets(self, offsets=None, use_mse=0, **kwargs):
    if offsets is None:
      if self.mode == 'valid':
        ry = np.arange(0, self.a_shape[0] - self.b_shape[0] + 1)
        rx = np.arange(0, self.a_shape[1] - self.b_shape[1] + 1)

      else:
        ry = np.arange(-self.b_shape[0] + 1, self.a_shape[0])
        rx = np.arange(-self.b_shape[1] + 1, self.a_shape[1])
      offsets = itertools.product(rx, ry)

    acc = cmisc.defaultdict(int)
    for dx, dy in offsets:
      score = self.compute_score_at_offset((dx, dy), use_mse=use_mse)
      acc[(dx, dy)] = score
    return acc

  def compute_best_with_offsets(self, offsets=None, use_mse=0, **kwargs):
    acc = self.compute_at_offsets(offsets, use_mse, **kwargs)
    dxlist = set()
    dylist = set()
    for dx, dy in offsets:
      dxlist.add(dx)
      dylist.add(dy)

    arr = np.zeros((len(dylist), len(dxlist)))
    for iy, y in enumerate(sorted(dylist)):
      for ix, x in enumerate(sorted(dxlist)):
        arr[iy, ix] = acc[(x, y)]
    self.res = arr
    self.to_db(use_mse=use_mse)

    best = max(acc.items(), key=lambda x: x[1])[0]
    return cmisc.Attr(offset=best, best=acc[best])

  def to_db(self, use_mse=False):
    if use_mse:
      self.minv = np.min(self.res)
      self.maxv = np.max(self.res)
      power = self.res
    else:
      power = to_db(self.res)
    if self.auto_remap:
      epx = 0.5
      self.minv = np.quantile(power, epx)
      self.maxv = np.max(power)
    else:
      power = power - np.quantile(power, 0.1) + self.minv
    self.power = power

  def get_colored_powerlog(self):
    power_colored = g_fft_helper.map(self.power, minv=self.minv, maxv=self.maxv)
    return power_colored


def compute_normed_correl2(haystack, needle, normalized=0, mode='full'):
  if not normalized: haystack, needle = norm_for_correl(haystack, needle)

  if len(haystack.shape) == 2:
    res = signal.correlate2d(needle, haystack, mode=mode)
    return res

  else:
    return signal.correlate(haystack, needle, mode=mode)


def compute_normed_correl(haystack, needle, mode='full'):
  needle = needle - np.mean(needle)
  needle = needle / np.linalg.norm(needle)
  haystack = haystack - np.mean(haystack)
  haystack = haystack / np.linalg.norm(haystack) * len(haystack) / len(needle)

  if len(haystack.shape) == 2:
    return signal.correlate2d(haystack, needle, mode=mode)
  else:
    return signal.correlate(haystack, needle, mode=mode)

def norm_for_correl(haystack, needle, norm_with_needle=0):
  needle_mean = np.mean(needle)
  needle = needle - needle_mean
  needle_norm = np.linalg.norm(needle)
  needle = needle / needle_norm

  if norm_with_needle:
    haystack = haystack - needle_mean
    haystack = haystack / (needle_norm / len(needle) * len(haystack))
  else:
    haystack = haystack - np.mean(haystack)
    haystack_norm = np.linalg.norm(haystack)
    haystack = haystack / haystack_norm * len(haystack) / len(needle)

  return haystack, needle


def correl_processing(a, b, want_color=0, ROI=None, **kwargs):
  ch = CorrelHelper(a, b, **kwargs)
  ch.compute_full_correl()
  ch.normalize_incomplete_correls()
  best = ch.compute_best(ROI=ROI)
  power_colored = ch.get_colored_powerlog() if want_color else None
  return cmisc.Attr(best=best, power_colored=power_colored, offset=best.offset)


def compute_flow(
    im0,
    im1,
    downscale=None,
    max_fft_size=200,
    max_rework_size=40,
    ROI=None,
    offset=None,
    debug=0,
    **kwargs
):
  # im1.offset = res.offset + im0.offset

  kReworkFactor = 6
  im0 = im0.astype(float)
  im1 = im1.astype(float)
  rework_range = max_rework_size

  mshape = max(im0.shape)

  downscale_max = max_rework_size // kReworkFactor
  if offset is not None:
    b0 = Box.FromImage(im0) + offset
    b1 = Box.FromImage(im1)
    b01 = b0.intersection(b1).expand_l(max_rework_size)
    downscale = 1

  if downscale is None:
    downscale = mshape / max_fft_size

  downscale = int(math.ceil(downscale))
  downscale = min(downscale, downscale_max)

  data = cmisc.Attr()
  if downscale > 1:
    rework_range = kReworkFactor * downscale
    noffset = None
    if offset is not None: noffset = offset / downscale

    v0 = downscale_img(im0, downscale)
    v1 = downscale_img(im1, downscale)
    downscale_roi = None
    if ROI is not None:
      downscale_roi = ROI / downscale
    res = compute_flow(v0, v1, ROI=downscale_roi, offset=noffset, debug=debug, **kwargs)
    offset = res.offset * downscale

  elif offset is None:
    return correl_processing(im0, im1, ROI=ROI, **kwargs)

  rx = np.arange(-rework_range, rework_range) + offset[0]
  ry = np.arange(-rework_range, rework_range) + offset[1]

  #im0, im1 = Z.dsp_utils.norm_for_correl(im0, im1)

  print('GUESS >> ', im0.shape, im1.shape, offset)
  ch = CorrelHelper(im0, im1, **kwargs)
  best = ch.compute_best_with_offsets(itertools.product(rx, ry), **kwargs)
  print('BEST >> ', best.offset, 'remapped > >', best.offset - offset, rework_range)
  if debug:
    import chdrft.utils.K as K
    import chdrft.utils.Z as Z
    G = K.GraphHelper()
    G.create_plot(images=[Dataset2d(ch.get_colored_powerlog()).shift((rx[0], ry[0]))])
    G.run()
    Z.plt.hist(np.ravel(ch.power), bins=256)
    Z.plt.show()
  data.best = best
  data.offset = best.offset
  return data

g_fft_helper = FFTHelper()


