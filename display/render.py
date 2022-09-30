#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.display.vispy_utils as vispy_utils
from chdrft.display.vispy_utils import ImageData
from chdrft.display.service import g_plot_service as oplt

import chdrft.display.utils as dsp_utils
import random
from chdrft.struct.base import Box
import numpy as np
import cv2
from chdrft.struct.geo import QuadTree
import skimage.transform

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def composeit(outfile, data):
  cx = ImageComposer()
  for entry in data:
    cx.add_img(cmisc.Attr(data=get_img(entry), box=entry.box))

  res = cx.render()
  if outfile:
    dsp_utils.save_image(outfile, res)
  else:
    dsp_utils.plot_img(res)


def get_boundaries(imgs):
  colorset = ('red', 'blue', 'green', 'yellow')

  def get_color(gridpos):
    if gridpos is None: return 'orange'
    return colorset[(gridpos[0] & 1) << 1 | (gridpos[1] & 1)]

  boundaries = []

  for img_data in imgs:
    boundaries.append(
        cmisc.Attr(
            polyline=img_data.box.to_double().expand(random.uniform(0, 1) * 0.001 +
                                                     1).poly_closed(),
            color=get_color(img_data.gridpos)
        )
    )
  return boundaries


def render_compose(in_imgs, extra_meshes=[], render_boundaries=0, as_grid=0, **kwargs):

  images = []
  for img_data in in_imgs:
    img_data = ImageData.Make(img_data)
    images.append(ImageData(
        img=img_data.img,
        gridpos=img_data.gridpos,
        box=img_data.box,
    ))

  if as_grid: images = ImageGrid(images=images, **kwargs).get_images()

  meshes = []
  meshes.extend(extra_meshes)
  if render_boundaries: meshes.append(cmisc.Attr(lines=get_boundaries(images)))

  rects = []
  for img_data in images:
    rects.append(img_data.box)
  vb = Box.Union(rects)
  meshes.append(cmisc.Attr(images=images, cmap='grays', clim=(0, 1)))

  vctx= oplt.plot(meshes, camera_viewbox=vb, typ='vispy', o=1).w.vctx
  enable_cycling(vctx)
  return vctx




def render_with_gridpos(imgs, outfile=None, spacing=1.1):

  cx = ImageComposer()
  for obj in imgs:
    h, w = obj.img.shape[:2]
    dim_xy = np.array((w, h))
    cx.add_img(
        cmisc.Attr(data=obj.img, box=Box(low=np.floor(obj.gridpos * dim_xy * spacing), dim=dim_xy))
    )

  if outfile:
    dsp_utils.save_image(outfile, cx.render(upscale=1))
  else:
    dsp_utils.plot_img(cx.render(upscale=1))


def overlay_images(a, b, dab, **kwargs):
  imx = [
      ImageData(img=a, pos=dab),
      ImageData(img=b, pos=np.zeros(2)),
  ]
  render_compose(imx, **kwargs)


class ImageGrid:

  def __init__(self, n=None, base_dim=None, images=None, spacing=0.1, nr=None, nc=None):
    if n is None: n = len(images)
    self.n = n

    def get_other(nx):
      return int(np.ceil(self.n / nx))

    if nc is not None: nr = get_other(nc)
    elif nr is not None: nc = get_other(nr)
    else:
      nr = int(self.n**0.5 + 1)
      nc = get_other(nr)

    self.nr = nr
    self.nc = nc
    self.data = []
    if images is not None:
      images = list(map(ImageData.Make, images))

    if base_dim is None:
      dims = [x.box.dim for x in images]
      base_dim = np.max(dims, axis=0)
    base_dim = np.array(base_dim)

    dim_spacing = base_dim * (1 + spacing)
    box = Box(low=(0, 0), dim=base_dim)

    for i in range(self.n):
      y = i // self.nc
      x = i % self.nc
      offset = dim_spacing * (x, y)
      self.data.append(ImageData(gridpos=(x, y), box=box+offset))
    self.dim_spacing = dim_spacing
    self.setup(images)

  def setup(self, images):
    if not images: return
    for i in range(len(images)):
      self.data[i].set_image(images[i].img)

      pos = self.data[i].pos
      if images[i].gridpos is not None:
        pos = self.dim_spacing * images[i].gridpos
        self.data[i].gridpos = self.images[i].gridpos
      self.data[i].stuff = images[i].stuff
      self.data[i].configure_box(images[i].box.zero_corner() + pos)

  def get_images(self):
    return list(self.data)


class ImageComposer:

  def __init__(self, imgs=[]):
    self.imgs = list(imgs)
    self.qtree = QuadTree(imgs)

  def add_img(self, img_data):
    self.imgs.append(img_data)
    self.qtree.add(img_data)

  def render(self, upscale=2):
    assert len(self.imgs) > 0
    boxes = list(img.box for img in self.imgs)
    pix_per_units = np.max([img.data.shape / img.box.shape for img in self.imgs], axis=0) * upscale

    self.rx = Range2D.Union(boxes)
    img_box = (self.rx * pix_per_units).to_int()
    self.pix_per_units = pix_per_units
    base_img = np.zeros(img_box.shape)
    for img in self.imgs:
      self.render_img(base_img, img)
    return base_img

  def render_img(self, base_img, img):
    new_box = ((img.box - self.rx.low) * self.pix_per_units).to_int()
    new_img = skimage.transform.resize(img.data, new_box.shape)
    base_img[new_box.window_yx] = new_img

  def render_box_fixed_dim(self, box, target_dim=None, upscale=1, **kwargs):
    u2px = self.imgs[0].u2px

    if target_dim is None:
      res = ImageData.Zero(box, u2px * upscale, **kwargs)
    else:
      target_box = Box.FromSize(target_dim)
      res = ImageData(img=target_box.make_zero_image(), box=target_box, **kwargs)

    elems = self.qtree.query_box(box, key=lambda x: x.box)
    for img in elems:
      ix = box.intersection(img.box)

      dest_box = box.change_rect_space(res.img_box, ix).to_int_round()
      src_box = img.box.change_rect_space(img.img_box, ix).to_int_round()
      if src_box.empty or dest_box.empty: continue
      simg = img.subimg(ix)
      print(ix, dest_box, simg.shape, dest_box.get_dim(res.yx), res.yx, res.box, res.img.shape)
      rescaled_img = skimage.transform.resize(simg, dest_box.get_dim(res.yx), order=0)
      res.subimg(ix, v=rescaled_img)
    return res


def render_grid(grid):
  vctx = grid.render(render_boundaries=1)
  enable_cycling(vctx)
  return vctx


def cycle_ev(d):

  imgs = [x.obj for x in d.cnds if isinstance(x.obj.obj, ImageData) and x.score == 0]
  imgs.sort(key=lambda x: x.ctxobj.vispy.transform.translate[2])

  n = len(imgs)
  for i in range(n):
    tsf = imgs[i].ctxobj.vispy.transform
    translate = tsf.translate
    translate[2] = 0.1 + (i + 1) % n  #zindex
    imgs[i].ctxobj.vispy.transform = vispy_utils.transforms.STTransform(
        scale=tsf.scale, translate=translate
    )
    imgs[i].ctxobj.vispy.update()


def enable_cycling(vctx):
  vctx.click_sub.observers.clear()
  vctx.click_sub.subscribe(cycle_ev)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
