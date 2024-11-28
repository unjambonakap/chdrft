#!/usr/bin/env python

import svgpathtools
from svgpathtools import *
from chdrft.struct.base import Box
import numpy as np
import math
from vispy.color import Color


import glog

global flags, cache
flags = None
cache = None


def read_size_mm(x):
  assert x.endswith('mm')
  return float(x[:-2])

def read_box(x):
  tmp =  list([float(y) for y in x.split(' ')])
  return Box(low=tmp[:2], high=tmp[2:])


class Remapper:
  def __init__(self, mat, to_c=True):
    self.mat = mat
    self.to_c = to_c

  def __call__(self, v):
    res = np.matmul(self.mat, [v[0], v[1], 1.]).A1
    res[0] /= res[2]
    res[1] /= res[2]
    if self.to_c: return res[0] +1j*res[1]
    return np.array((res[0], res[1]))


def box_remapper(src, dest, **kwargs):
  return Remapper(dest.mat_to() * src.mat_from(), **kwargs)

def get_arc(low, high, steps=100):
  if high < low: high + 2 * math.pi
  angles = np.linspace(low, high, steps+1)
  px, py = np.cos(angles), np.sin(angles)
  return np.array([px, py])



class SVGBuilder:
  def __init__(self, rmp, units='mm'):
    self.paths = []
    self.rmp = rmp
    self.cur_path = []
    self.units = units
    self.attrs = {}
    self.svg_attrs = {}

  def push_segs(self, segs, rmp=None, stroke_width=1, **kwargs):
    if rmp is None: rmp = self.rmp
    lines = []
    for seg in segs:
      lines.append(Line(rmp(seg[0]), rmp(seg[1])))

    attrs = dict(stroke_width=stroke_width, **kwargs)
    for x in ('col', 'fill'):
      v = kwargs.get(x, None)
      if v is None: continue
      if not isinstance(v, str) or v!='none': v = Color(v).hex
      attrs[x] = v

    if 'col' in attrs:
      attrs['stroke'] = attrs['col']
      del attrs['col']


    self.paths.append((Path(*lines), attrs))

  def push_path(self, path, rmp=None, **kwargs):
    segs = []
    for i in range(len(path)-1):
      segs.append((path[i], path[i+1]))
    self.push_segs(segs, rmp=rmp, **kwargs)
    #if rmp is None: rmp = self.rmp
    #npath = [rmp(x) for x in path]
    #self.segs.append(Attributize(path=Path(npath), col=col, stroke_width=stroke_width))

  def add_seg(self, a, b):
    self.cur_path.append((a,b))


  def add_box(self, box):
    pts = []
    for i in range(5): pts.append(box.get(i))
    self.push_polyline(np.array(pts).T)

  def push_polyline(self, points, close=False, **kwargs):
    assert np.shape(points)[0] == 2
    mod = np.shape(points)[1]
    n = mod
    lst = points.tolist()
    if close:  lst.append(points[0,:])
    self.push_path(lst, **kwargs)

  def set_viewbox(self, vb, dim):
    u = self.units
    #self.svg_attrs['viewBox'] = f'{vb.low[0]} {vb.low[1]} {vb.high[0]} {vb.high[1]}'
    self.svg_attrs['viewBox'] = f'{0} {0} {vb.high[0]} {vb.high[1]}'
    #print(self.svg_attrs['viewBox'])
    self.svg_attrs['width'] = f'{dim[0]}{u}'
    self.svg_attrs['height'] = f'{dim[1]}{u}'
    self.svg_attrs['size'] = (self.svg_attrs['width'], self.svg_attrs['height'])
    glog.info('setting viewbox >> %s', self.svg_attrs)


  def write(self, filename):
    colors = []
    paths = []
    paths, attrs= zip(*self.paths)
    assert len(self.paths) != 0
    svgpathtools.wsvg(paths, svg_attributes=self.svg_attrs, attributes=attrs, filename=filename)

