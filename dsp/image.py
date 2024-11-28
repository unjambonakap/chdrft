#!/usr/bin/env python

from __future__ import annotations

import cv2
import os
import numpy as np

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.struct.base import Box, g_unit_box
from chdrft.struct.base import Box
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
import chdrft.dsp.datafile as Dataset2d

from chdrft.utils.cmdify import ActionHandler
import os
import numpy as np
import cv2

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def warp_affine_xy(img, mat, dim):
  nmat = np.array(mat[:2, :3])
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

  if dy <= 0: im = im[
      -dy:,
  ]
  else: im = im[:-dy, :]
  return im


def downscale_img(img, downscale):
  target_scale = np.rint(np.array(img.shape[:2]) / downscale).astype(int)
  return cv2.resize(img, tuple(target_scale[::-1]))


def plot_img(img):
  from chdrft.display.ui import GraphHelper
  G = GraphHelper()
  p1 = Dataset2d(img)
  G.create_plot(images=[p1])
  G.run()


def from_cv_norm_img(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  if np.max(img) > 1.1:
    img = img / 255.
  return img.astype(np.float32)


def to_cv_norm_img(img):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  if img.dtype in (np.float64, np.float32):
    img = (img * 255).astype(np.uint8)
  return img


def save_image(fname, res):
  cv2.imwrite(fname, to_cv_norm_img(res))


def read_tiff(fname):
  res = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
  assert res is not None, fname
  return from_cv_norm_img(res)


class ImageData(A):

  @staticmethod
  def Make(data, **kwargs):
    if isinstance(data, str):
      return ImageData(img=read_tiff(data), **kwargs)
    if isinstance(data, bytes) or (isinstance(data, np.ndarray) and len(data.shape) == 1):
      return ImageData(
          img=from_cv_norm_img(cv2.imdecode(np.frombuffer(data), cv2.IMREAD_UNCHANGED)), **kwargs
      )
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

  @property
  def channels(self) -> int:
    if len(self.img.shape) == 2: return 1
    return self.img.shape[2]

  def set_pix(self, pos_xy, v: float, channel: int = None):
    if self.yx: pos_xy = pos_xy[::-1]
    if len(self.img.shape) == 2 or channel is None:
      assert channel == 0 or channel is None
      self.img[tuple(pos_xy)] = v
    else:
      self.img[(
          pos_xy[0],
          pos_xy[1],
          channel,
      )] = v

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

  def save(self, fname):
    save_image(fname, self.img)

  def encode(self):
    return cv2.imencode(
        '.png', cv2.cvtColor(to_cv_norm_img(self.img), cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 0]
    )[1]

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
  def norm_cv(self):
    r = self.clone()
    r.set_image(to_cv_norm_img(r.img))
    return r

  @property
  def flip_y(self):
    r = self.clone()
    r.set_image(self.img[::-1])
    return r

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
      return A(typ='rect', gridpos=self.gridpos, box=self.box.fix(), geo=self.box.shapely)
    return self._obj

  def __hash__(self):
    return hash(tuple(self.gridpos))

  def subimg_id(self, region) -> ImageData:
    return ImageData(img=self.subimg(region))

  def subimg(self, region, v=None):
    region_pix = self.box.change_rect_space(self.img_box,
                                            region).to_int_round().intersection(self.img_box)
    return region_pix.subimg(self.img, v=v, yx=self.yx)

  def plot(self, **kwargs):
    from chdrft.display.service import g_plot_service
    return g_plot_service.plot(A(images=[self], **kwargs))


def test(ctx):
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
