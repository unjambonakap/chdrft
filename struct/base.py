import sys
import glog
import bisect
import numpy as np
from collections import defaultdict
from copy import copy

from chdrft.utils.misc import to_list, Attributize, proc_path
import chdrft.utils.misc as cmisc
from sortedcontainers import SortedSet
import math
import io
import itertools
import functools
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.base import BaseGeometry
from chdrft.utils.math import MatHelper
from chdrft.utils.types import *
from chdrft.utils.geo import Line
import chdrft.utils.math as opa_math

g_eps = 1e-7
kInfInt = 2**100


class Range1D:
  @staticmethod
  def Union(rlist):
    return functools.reduce(Range1D.union, rlist)

  @staticmethod
  def Range01():
    return Range1D(0, 1)

  @staticmethod
  def Empty(is_int=0):
    return Range1D(1, 0, is_int=is_int)

  @staticmethod
  def All(is_int=0):
    if is_int:
      return Range1D(-kInfInt, kInfInt)
    return Range1D(-1e100, 1e100)

  @staticmethod
  def FromSet(s):
    res = Range1D(min(s), max(s) + 1, is_int=1)
    assert len(s) == len(res)
    return res

  @staticmethod
  def FromArray(a):
    return Range1D(0, len(a), is_int=1)

  @staticmethod
  def IntFromString(a):
    a, b = a.split(' ')
    res = Range1D(int(a), int(b), is_int=1)
    return res

  @staticmethod
  def FromString(a):
    a, b = a.split(' ')
    res = Range1D(float(a), float(b))
    return res

  def __init__(self, *args, n=None, is_int=0, center=None, dim=None, low=None, high=None):
    if dim is not None: n = dim

    if len(args) == 1:
      range1d = args[0]
      if isinstance(range1d, Range1D):
        low = range1d.low
        high = range1d.high
        is_int = range1d.is_int
      else:
        if not is_list(range1d):
          low = range1d
          high = low + n
        else:
          low = range1d[0]
          high = range1d[1]
    elif len(args) == 2:
      low = args[0]
      high = args[1]
    self.is_int = is_int

    if center is not None and n is not None:
      low = center - n / 2
      high = center + n / 2
    elif low is not None and n is not None:
      high = low + n
    elif high is not None and n is not None:
      low = high + n
    elif low is not None and center is not None:
      high = 2*center-low
    elif high is not None and center is not None:
      low = 2*center-high


    if low is None:
      if is_int: low = -kInfInt
      else: low = -math.inf
    if high is None:
      if is_int: high = kInfInt
      else: high = math.inf

    if is_int:
      low = int(round(low))
      high = int(round(high))

    self.low = low
    self.high = high
    if self.is_int:
      self._empty = self.low >= self.high
    else:
      self._empty = self.low > self.high

  def uniform(self):
    if self.is_int:
      return np.random.random_integers(self.low, self.high - 1)
    return np.random.uniform(self.low, self.high)

  def to_double(self):
    if self.inf:
      return Range1D.All(is_int=0)
    if self.empty:
      return Range1D(is_int=0)
    return Range1D(self.low, self.high, is_int=0)

  def to_int(self):
    if self.inf:
      return Range1D.All(is_int=1)
    if self.empty:
      return Range1D(is_int=1)
    return Range1D(self.low, self.high, is_int=1)

  def to_int_contained(self):
    if self.inf:
      return Range1D.All(is_int=1)
    if self.empty:
      return Range1D(is_int=1)
    return Range1D(math.ceil(self.low), math.floor(self.high), is_int=1)

  @property
  def int_round(self):
    if self.inf:
      return Range1D.All(is_int=1)
    return Range1D(round(self.low), round(self.high), is_int=1)

  @property
  def numpy(self):
    return np.array((self.low, self.high))

  @property
  def window(self):
    if self.empty:
      return slice(0, 0)
    return slice(self.low, self.high)

  def extract(self, x):
    return x[self.window]

  @property
  def range(self):
    return range(self.low, self.high)

  def clone(self):
    return self.make_new(self.low, self.high)

  @property
  def empty(self):
    return self._empty

  @property
  def inf(self):
    return self.inf_l and self.inf_r

  @property
  def inf_l(self):
    if self.is_int:
      return self.low == -kInfInt
    return np.isneginf(self.low)

  @property
  def inf_r(self):
    if self.is_int:
      return self.high == kInfInt
    return np.isposinf(self.high)

  def __getitem__(self, v):
    assert v == 0 or v == 1
    if v == 0:
      return self.low
    return self.high

  def clamp(self, *args):
    other = self.make_new(*args)
    self.low = max(self.low, other.low)
    self.high = min(self.high, other.high)
    return self

  def importv(self, v):
    if self.is_int:
      if isinstance(v, int):
        return v
      return int(round(v))
    else:
      # ?????
      if not self.is_int:
        return v
      return float(v)

  def clampv(self, v):
    v = self.importv(v)
    if self.low is not None:
      v = np.maximum(v, self.low)
    if self.high is not None:
      if self.is_int:
        v = np.minimum(v, self.high - 1)
      else:
        v = np.minimum(v, self.high)
    return v

  def linspace(self, stride=None, npoints=None):
    if npoints is not None:
      return np.linspace(self.low, self.high, npoints)
    return np.arange(self.low, self.high, stride)

  def make_new(self, *args, **kwargs):
    return Range1D(*args, is_int=self.is_int, **kwargs)

  def make(self, **kwargs):
    kwargs = dict(kwargs)
    for k, v in list(kwargs.items()):
      if v is None:
        kwargs[k] = getattr(self, k)
    return self.make_new(**kwargs)

  def contains(self, other):
    if isinstance(other, Range1D):
      return self.low <= other.low and other.high <= self.high
    return self.cond(other)

  @cmisc.yield_wrapper
  def filter(self, lst):
    for x in lst:
      if x in self:
        yield x

  def __contains__(self, other):
    return self.contains(other)

  def expand_l(self, l):
    return self.make_new(center=self.center, dim=self.length + l)

  @property
  def center(self):
    return self.mid

  def expand(self, ratio):
    mid = self.mid
    l = self.length / 2 * ratio
    return self.make_new(mid - l, mid + l)

  def cond(self, x):
    return np.logical_and(*self.cond_tuple(x))

  def cond_tuple(self, x):
    if self.is_int:
      return (self.low <= x, x < self.high)
    else:
      return (self.low <= x, x <= self.high)

  def shift(self, v):
    return self.make_new(self.low + v, self.high + v)

  @property
  def length(self):
    return max(self.high - self.low, 0)

  @property
  def n(self):
    return self.high - self.low

  @property
  def dim(self):
    return self.n
  @property
  def size(self):
    return self.n

  def __len__(self):
    return self.length

  def intersects(self, other, adj=0):
    return not self.intersection(other, adj=adj).empty

  def normalize_other(self, other):
    if is_list(other):
      return Range1D(other)
    assert isinstance(other, Range1D)
    return other

  def intersection(self, other, adj=0):
    other = self.normalize_other(other)
    return self.make_new(max(self.low, other.low), min(self.high, other.high) + adj)

  def union(self, other):
    other = self.normalize_other(other)
    if self.empty:
      return other
    if other.empty:
      return self
    return self.make_new(min(self.low, other.low), max(self.high, other.high))

  def __lt__(self, other):
    return self.as_tuple < other.as_tuple

  def __eq__(self, other):
    return self.as_tuple == other.as_tuple

  def __xor__(self, v):
    return self.intersection(v)

  def __mul__(self, v):
    return self.make_new(self.numpy * v)

  def __add__(self, v):
    if isinstance(v, Range1D):
      return self.make_new(self.low + v.low, self.high + v.high)
    return self.make_new(self.numpy + v)

  def __sub__(self, v):
    if isinstance(v, Range1D):
      return self.make_new(self.low - v.high, self.high - v.low)
    return self.make_new(self.numpy - v)

  def __neg__(self):
    return self.make_new(-self.high, -self.low)

  def __floordiv__(self, v):
    return self.make_new(self.numpy // v)

  def __truediv__(self, v):
    return self.make_new(self.numpy / v)

  def mod(self, v):
    return (v - self.low) % self.n + self.low

  @property
  def mid(self):
    return (self.low + self.high) / 2

  @property
  def as_tuple(self):
    return (self.low, self.high)

  @property
  def numpy(self):
    return np.array(self.as_tuple)

  def __hash__(self):
    return self.as_tuple.__hash__()

  def from_local(self, p):
    return p * self.dim + self.low

  def to_local(self, p):
    return (p - self.low) / self.dim

  def __str__(self):
    if self.is_int:
      #return 'Range1D(%x, %x)' % (self.low, self.high)
      return 'Range1D(%d, %d)' % (self.low, self.high)
    return f'Range1D({self.low}, {self.high})'

  def __repr__(self):
    #if isinstance(self.low, int): return repr(f'{self.low:x}, {self.high:x}')
    if isinstance(self.low, int):
      return repr(f'{self.low:d}, {self.high:d}')
    return repr(f'{self.low}, {self.high}')

  def __iter__(self):
    return self.range


def Range1D_int(*args, **kwargs):
  return Range1D(*args, **kwargs, is_int=1)


class Range1DWithData(Range1D):

  def __init__(self, *args, data=None, merge_data=0, **kwargs):
    super().__init__(*args, **kwargs)
    self.data = data
    self.merge_data = merge_data

  def union(self, other):
    if other.high == self.low:
      return other.union(self)
    assert self.high == other.low, f'{self} {other}'
    if self.merge_data:
      return Range1DWithData(self.low, other.high, data=self.data + other.data)
    return Range1DWithData(self.low, other.high, data=self.data)

  def get_data(self, qrange):
    assert self.contains(qrange)
    i1 = qrange.low - self.low
    i2 = qrange.high - self.low
    return self.data[i1:i2 + 1]


class Range2D:

  @staticmethod
  def All(is_int=0):
    return Range2D(Range1D.All(is_int=is_int), Range1D.All(is_int=is_int))
  @staticmethod
  def Empty(is_int=0):
    return Range2D(Range1D.Empty(is_int=is_int), Range1D.Empty(is_int=is_int))

  @staticmethod
  def FromShapely(geo):
    mx, my, Mx, My = geo.bounds
    return Range2D(xr=(mx, Mx), yr=(my, My))

  def __init__(self, *args, **kwargs):
    xr, yr = self.setup(*args, **kwargs)
    self.xr = xr
    self.yr = yr
    self.low = np.array((self.xr.low, self.yr.low))
    self.high = np.array((self.xr.high, self.yr.high))
    self.dim = np.array((self.xr.dim, self.yr.dim))

  def range1d(self, coord):
    return (self.xr, self.yr)[coord]

  def setup(self, *args, is_int=0, **kwargs):
    xr, yr = None, None
    if 'size' in kwargs:
      kwargs['dim'] = kwargs['size']

    if len(args) == 1:
      range2d = args[0]
      if isinstance(range2d, Range2D):
        return Range1D(range2d.xr), Range1D(range2d.yr)
      else:
        xr = Range1D(range2d[0], is_int=is_int)
        yr = Range1D(range2d[1], is_int=is_int)
        return xr, yr
    elif len(args) == 2:
      xr = Range1D(args[0], is_int=is_int)
      yr = Range1D(args[1], is_int=is_int)
      return xr, yr
    elif len(args) == 4:
      xl, yl, xh, yh = args
      xr = Range1D((xl, xh), is_int=is_int)
      yr = Range1D((yl, yh), is_int=is_int)
      return xr, yr
    elif len(args) == 0:
      if len(kwargs) == 0:
        return Range1D(is_int=is_int), Range1D(is_int=is_int)

      if 'xr' in kwargs:
        xr = Range1D(kwargs['xr'])
      if 'yr' in kwargs:
        yr = Range1D(kwargs['yr'])
      if xr is not None or yr is not None:
        if xr is None:
          xr = Range1D.All(is_int=is_int)
        if yr is None:
          yr = Range1D.All(is_int=is_int)
      else:

        d = defaultdict(lambda: None, kwargs)
        for k in d.keys():
          if not is_list(d[k]):
            d[k] = [d[k], d[k]]
          d[k] = np.array(d[k])
        dim, dim_yx, low, high, center = d['dim'], d['dim_yx'], d['low'], d['high'], d['center']

        if len(args) == 4 and len(kwargs) == 0:
          low = np.array((args[:2]))
          high = np.array((args[2:]))
        elif len(args) == 2 and len(kwargs) == 0:
          low, high = np.array(args)
        elif len(args) == 1 and len(kwargs) == 0:
          low = high = args[0]

        if dim_yx is not None:
          dim = dim_yx[::-1]

        if center is not None:
          center = np.array(center)
        if dim is not None:
          dim = np.array(dim)
        if low is not None:
          low = np.array(low)
        if high is not None:
          high = np.array(high)

        if dim is None:
          flow = low is not None
          fhigh = high is not None
          fcenter = center is not None
          if flow and fhigh:
            dim = high - low
          elif flow and fcenter:
            dim = (center - low) * 2
          elif fhigh and fcenter:
            dim = (high - center) * 2

        if dim is not None:
          if high is not None:
            low = high - dim
          elif center is not None:
            low = center - dim / 2
            high = center + dim / 2
          elif low is not None:
            high = low + dim

        xr = Range1D(low[0], n=dim[0], is_int=is_int)
        yr = Range1D(low[1], n=dim[1], is_int=is_int)
    else:
      assert 0

    return xr, yr

  def uniform(self):
    return np.array((self.xr.uniform(), self.yr.uniform()))

  @property
  def inf(self):
    return self.xr.inf and self.yr.inf

  @property
  def is_int(self):
    return self.xr.is_int

  def make_line(self, pos, is_h=None):
    assert is_h is not None
    if is_h:
      return self.make_new(low=(self.xl, pos), high=(self.xh, pos), is_int=0)
    else:
      return self.make_new(low=(pos, self.yl), high=(pos, self.yh), is_int=0)

  def make_line_h(self, pos):
    return self.make_line(pos, is_h=1)

  def make_line_v(self, pos):
    return self.make_line(pos, is_h=0)

  def as_numpy_line(self):
    return np.array((self.low, self.high))

  def __str__(self):
    return f'Range2D(low={tuple(self.low)}, high={tuple(self.high)})'

  def __contains__(self, other):
    return self.contains(other)

  def contains(self, other):
    if is_list(other):
      return other[0] in self.xr and other[1] in self.yr
    return self.xr.contains(other.xr) and self.yr.contains(other.yr)

  def iter(self):
    return list(iter(self))

  def __iter__(self):
    for y, x in itertools.product(self.yr.range, self.xr.range):
      yield x, y

  def __floordiv__(self, v):
    return self.make_new(self.xr / v, self.yr / v)

  def __truediv__(self, v):
    return self.make_new(self.xr / v, self.yr / v)

  def shift(self, vec):
    return self.make_new(self.xr.shift(vec[0]), self.yr.shift(vec[1]))

  def intersection_x(self, other):
    return self.make_new(self.xr.intersection(other), self.yr)

  def intersection_y(self, other):
    return self.make_new(self.xr, self.yr.intersection(other))

  def intersection(self, other):
    return self.make_new(self.xr.intersection(other.xr), self.yr.intersection(other.yr))

  def intersects(self, other):
    return self.xr.intersects(other.xr) and self.yr.intersects(other.yr)

  def make_new(self, *args, **kwargs):
    if 'is_int' not in kwargs:
      kwargs['is_int'] = self.is_int
    return Range2D(*args, **kwargs)

  def __mul__(self, v):
    if is_list(v):
      return self.make_new(self.xr * v[0], self.yr * v[1])
    return self.__mul__((v, v))

  def __xor__(self, v):
    return self.intersection(v)

  def __add__(self, v):
    if isinstance(v, Range2D):
      return self.make_new(self.xr + v.xr, self.yr + v.yr)
    if is_list(v):
      return self.make_new(self.xr + v[0], self.yr + v[1])
    return self.__add__((v, v))

  def __neg__(self):
    return self.make_new(-self.xr, -self.yr)

  def __sub__(self, v):
    if isinstance(v, Box):
      return self.make_new(self.xr - v.xr, self.yr - v.yr)
    if is_list(v):
      return self.make_new(self.xr - v[0], self.yr - v[1])
    return self.__sub__((v, v))

  def __floordiv__(self, v):
    if is_list(v):
      return self.make_new(self.xr // v[0], self.yr // v[1])
    return self.__floordiv__((v, v))

  def __truediv__(self, v):
    if is_list(v):
      return self.make_new(self.xr / v[0], self.yr / v[1])
    return self.__truediv__((v, v))

  def length(self):
    return pg.Point(self.xr.length, self.yr.length)

  @property
  def empty(self):
    return self.xr.empty or self.yr.empty

  @property
  def center(self):
    return np.array((self.xr.center, self.yr.center))

  @property
  def mid(self):
    return self.center

  @property
  def shape(self):
    return np.array((self.height, self.width))

  @property
  def shape_int(self):
    return np.ceil(self.shape).astype(int)

  def to_double(self):
    if self.inf:
      return Range2D.All(is_int=0)
    return Range2D(self.xr.to_double(), self.yr.to_double(), is_int=0)

  def to_int(self):
    if self.inf:
      return Range2D.All(is_int=1)
    return Range2D(self.xr.to_int(), self.yr.to_int(), is_int=1)

  def to_int_contained(self):
    if self.inf:
      return Range2D.All(is_int=1)
    return Range2D(self.xr.to_int_contained(), self.yr.to_int_contained(), is_int=1)

  def to_int_round(self):
    if self.inf:
      return Range2D.All(is_int=1)
    return Range2D(self.xr.int_round, self.yr.int_round, is_int=1)

  @property
  def yx(self):
    return self.make_new(self.yr, self.xr)

  def union(self, other):
    if self.empty:
      return other
    if other.empty:
      return self
    return self.make_new(self.xr.union(other.xr), self.yr.union(other.yr))

  def make_zero_image(self, yx=1, **kwargs):
    return np.zeros(self.get_dim(yx), **kwargs)

  @property
  def window_yx(self):
    return self.window[::-1]

  def get_window(self, yx):
    return self.window_yx if yx else self.window

  def get_dim(self, yx):
    return self.dim_yx if yx else self.dim

  @property
  def window(self):
    assert self.is_int
    return (self.xr.window, self.yr.window)

  def window_with_base(self, base, yx=0):
    cur = self
    if base is not None:
      cur = cur - base
    return cur.get_window(yx)

  def subimg(self, img, base=None, v=None, yx=1):
    if base is not None:
      if v is None:
        return img[self.window_with_base(base=base, yx=yx)]
      else:
        img[self.window_with_base(base=base, yx=yx)] = v
    window = self.get_window(yx)

    if v is None:
      return img[window]
    img[window] = v

  def img_subregion(self, img, subregion, yx=1):
    assert subregion in self
    image_region = Box.FromImage(img, yx=yx)
    sel = self.change_rect_space(image_region, subregion)
    return sel.subimg(img, yx=yx)

  def img_subregion_samescale(self, img, subregion, v=None, yx=1):
    assert subregion in self
    assert np.all(self.get_dim(yx) == img.shape), (img.shape, self.get_dim(yx))
    sel = subregion - self.low
    return sel.subimg(img, v=v, yx=yx)

  def to_qrectf(self):
    from pyqtgraph.Qt import QtCore
    return QtCore.QRectF(self.xr.low, self.yr.low, self.xr.length, self.yr.length)

  def union_points(self, pts):
    if isinstance(pts, np.ndarray):
      tb = pts
    else:
      pts = list(pts)
      tb = np.array(pts)
    return self.union(Range2D.FromPoints(tb))

  def get_yx(self, v):
    return v[::-1]

  @property
  def size(self):
    return self.dim

  @property
  def dim_yx(self):
    return self.get_yx(self.dim)

  @property
  def low_yx(self):
    return self.get_yx(self.low)

  @property
  def xl(self):
    return self.low[0]

  @property
  def yl(self):
    return self.low[1]

  @property
  def xh(self):
    return self.high[0]

  @property
  def yh(self):
    return self.high[1]

  @property
  def xn(self):
    return self.dim[0]

  @property
  def yn(self):
    return self.dim[1]

  @property
  def width(self):
    return self.dim[0]

  @property
  def height(self):
    return self.dim[1]

  @property
  def aspect(self):
    return self.width / self.height

  def force_aspect(self, target_aspect):
    change_ratio = target_aspect / self.aspect
    return self.expand((max(1, change_ratio), min(1, change_ratio)))

  def mat_translate_from(self):
    return MatHelper.mat_translate(self.low)

  def mat_translate_to(self):
    return MatHelper.mat_translate(-self.low)

  def mat_scale_from(self):
    return MatHelper.mat_scale(self.dim)

  def mat_scale_to(self):
    return MatHelper.mat_scale(1. / self.dim)

  def mat_to(self):
    return np.matmul(self.mat_scale_to(), self.mat_translate_to())

  def mat_from(self):
    return np.matmul(self.mat_translate_from(), self.mat_scale_from())

  def poly_closed(self, z_coord=None):
    return self.poly(closed=True, z_coord=z_coord)

  def poly(self, closed=False, z_coord=None):
    res = []
    for i in range(4):
      res.append(self.get(i, z_coord=z_coord))
    if closed:
      res.append(self.get(0, z_coord=z_coord))
    return np.array(res)

  @property
  def quad(self):
    return Quad(self.poly())

  def get(self, i, z_coord=None):
    p = [self.low[0], self.low[1]]
    if z_coord is not None:
      p.append(z_coord)
    if (i & 1) ^ (i >> 1 & 1):
      p[0] = self.high[0]
    if (i & 2):
      p[1] = self.high[1]
    return np.array(p)

  @property
  @cmisc.yield_wrapper
  def corners(self):
    for i in range(4):
      yield self.corner(i)

  def corner(self, i):
    return self.get_grid(i)

  def get_grid(self, i):
    p = [self.low[0], self.low[1]]
    if (i & 1):
      p[0] = self.high[0]
    if (i & 2):
      p[1] = self.high[1]
    return np.array(p)

  def __getitem__(self, i):
    return self.get(i)

  def zero_corner(self):
    return self.make_new(low=(0, 0), dim=self.dim)

  def closest_corner(self, p):
    return min(range(4), key=lambda x: np.linalg.norm(p - self.get_grid(x)))

  def with_corner(self, corner):
    return self.make_new(low=corner, dim=self.dim)

  def make_with_pos(self, low=None, high=None, center=None):
    return self.make_new(low=low, high=high, center=center, dim=self.dim)

  def make(self, **kwargs):
    kwargs = dict(kwargs)
    for k, v in list(kwargs.items()):
      if v is None:
        kwargs[k] = getattr(self, k)
    return self.make_new(**kwargs)

  def __str__(self):
    return f'Box(low={tuple(self.low)}, high={tuple(self.high)}, is_int={self.is_int})'

  def __repr__(self):
    return str(self)

  @property
  def area(self):
    return self.dim[0] * self.dim[1]

  def overlaps(self, other):
    low2 = np.maximum(self.low, other.low)
    high2 = np.minimum(self.high, other.high)
    return np.all(low2 <= high2)

  def fix(self):
    return Box.FromPoints([self.low, self.high], is_int=self.is_int)

  def flip_x(self, my):
    return self.make_new(
        low=(my - self.high[0], self.low[1]), high=(my - self.low[0], self.high[1])
    )

  def flip_y(self, my):
    return self.make_new(
        low=(self.low[0], my - self.high[1]), high=(self.high[0], my - self.low[1])
    )

  def scale(self, fx, fy):
    return self.make_new(low=self.low * (fx, fy), dim=self.dim * (fx, fy))

  def set_aspect(self, ratio):
    nwidth = max(self.width, ratio * self.height)
    nheight = max(self.width / ratio, self.height)
    return Box(center=self.center, dim=(nwidth, nheight))

  def expand(self, f):
    return self.make_new(center=self.center, dim=self.dim * f)

  def expand_l(self, l):
    return self.make_new(center=self.center, dim=self.dim + l)

  def translate(self, v):
    return self.make_new(center=self.center + np.array(v), dim=self.dim)

  def interpolate(self, other, alpha):
    # (1-alpha) * this + alpha  * other
    return Box(
        center=(1 - alpha) * self.center + alpha * other.center,
        dim=(1 - alpha) * self.dim + alpha * other.dim
    )

  def transform_by_mat(self, mat):
    plist = []
    for i in range(3):
      plist.append(mat_apply2d(mat, self.get_grid(i)))

    return RealBox(plist=plist)

  @property
  def shapely(self):
    return Polygon(self.poly(closed=True))

  def get_grid_pos_with_id(self, stride=None, npoints=None):
    if stride is not None:
      xr = np.arange(self.xl, self.xh, stride[0])
      yr = np.arange(self.yl, self.yh, stride[1])
    else:
      xr = np.linspace(self.xl, self.xh, npoints[0])
      yr = np.linspace(self.yl, self.yh, npoints[1])
    X, Y = np.meshgrid(xr, yr)
    res = np.stack((X.ravel(), Y.ravel()), axis=-1)

    X_id, Y_id = np.meshgrid(range(len(xr)), range(len(yr)))
    res_id = np.stack((X_id.ravel(), Y_id.ravel()), axis=-1)
    return res_id, res

  def get_grid_pos(self, stride=None, npoints=None):
    if stride is not None:
      xr = np.arange(self.xl, self.xh, stride[0])
      yr = np.arange(self.yl, self.yh, stride[1])
    else:
      xr = np.linspace(self.xl, self.xh, npoints[0])
      yr = np.linspace(self.yl, self.yh, npoints[1])
    X, Y = np.meshgrid(xr, yr)
    res = np.stack((X.ravel(), Y.ravel()), axis=-1)
    return res

  def importv(self, v):
    return np.array([self.xr.importv(v[0]), self.yr.importv(v[1])])

  def clampv(self, v):
    return np.array([self.xr.clampv(v[0]), self.yr.clampv(v[1])])

  @property
  def quadrants(self):
    mid = self.center
    xlist = [self.low[0], mid[0], self.high[0]]
    ylist = [self.low[1], mid[1], self.high[1]]
    for i in range(4):
      xp = i & 1
      yp = i >> 1
      yield Box(xlist[xp], ylist[yp], xlist[xp + 1], ylist[yp + 1])

  def line(self, i):
    from chdrft.geo.utils import Line
    return Line(self.get(i), self.get(i + 1))

  def point_distance(self, p):
    if isinstance(p, Point):
      x, y = p.xy
    else:
      x, y = p
    dx = 0
    if x < self.xl:
      dx = self.xl - x
    elif x > self.xh:
      dx = x - self.xh

    dy = 0
    if y < self.yl:
      dy = self.yl - y
    elif y > self.yh:
      dy = y - self.yh
    return (dx * dx + dy * dy)**0.5

  def distance(self, p):
    from chdrft.utils.geo import Line
    if isinstance(p, BaseGeometry):
      return p.distance(self.shapely)
    if isinstance(p, Line):
      return p.box_distance(self)
    return self.point_distance(p)

  def is_intersecting_obj(self, obj):
    return self.distance(obj) < g_eps

  def center_on(self, p):
    pt0 = np.abs(p - self.center) * 2
    return Box(center=p, dim=pt0 + self.dim)

  def from_box_space(self, p):
    if isinstance(p, np.ndarray) and len(np.shape(p)) > 1:
      return p * self.dim.reshape(-1, 2) + self.low.reshape(-1, 2)
    return p * self.dim + self.low

  def to_box_space(self, p):
    return (p - self.low) / self.dim

  def change_rect_space(self, target, p):
    if isinstance(p, Box) and p.inf:
      return p
    return target.from_box_space(self.to_box_space(p))

  def change_rect_boxspace(self, target, p):
    if isinstance(p, Box) and p.inf:
      return p
    return target.to_box_space(self.from_box_space(p))

  def as_gen_box(self):
    return GenBox(self.low, ((self.width, 0), (0, self.height)))

  def to_vispy_transform(self, zpos=0):
    from vispy.visuals import transforms
    return transforms.STTransform(
        scale=self.dim, translate=(
            self.low[0],
            self.low[1],
            zpos,
        ))

  @staticmethod
  def FromArray(a):
    shape = a.shape
    rx = Range1D(0, shape[0], is_int=1)
    ry = Range1D(0, shape[1], is_int=1)
    return Range2D(rx, ry, is_int=1)

  @staticmethod
  def FromArray_yx(a):
    return Range2D.FromArray(a).yx

  @staticmethod
  def FromSize(dim, **kwargs):
    return Range2D(low=(0, 0), dim=dim, **kwargs)

  @staticmethod
  def FromImage(a, yx=1):
    if yx:
      return Range2D.FromArray_yx(a)
    else:
      return Range2D.FromArray(a)

  @staticmethod
  def FromCorners(low, high):
    return Range2D(Range1D(low[0], high[0]), Range1D(low[1], high[1]))

  @staticmethod
  def FromPoints(pts, **kwargs):
    nlow = np.min(pts, axis=0)
    nhigh = np.max(pts, axis=0)
    return Box(low=nlow, high=nhigh, **kwargs)

  @staticmethod
  def Union(rlist):
    rlist = list(rlist)
    if not rlist: return Box.Empty()
    return functools.reduce(Range2D.union, rlist)

  @staticmethod
  def FromCJ(cjrect):
    if isinstance(cjrect, Box):
      return cjrect
    return Box(low=(cjrect.Left, cjrect.Top), high=(cjrect.Right, cjrect.Bottom))

  def to_cj_rect(self):
    return cmisc.Attr(Left=self.xl, Right=self.xh, Top=self.yl, Bottom=self.yh)

  @property
  def as_tuple(self):
    return ((self.xl, self.yl), (self.xh, self.yh))

  def __eq__(self, other):
    return self.as_tuple == other.as_tuple

  def __hash__(self):
    return self.as_tuple.__hash__()


Box = Range2D


############ END OF RANGE2D
class GenBox:

  @staticmethod
  def FromPoints(*pts):
    pts = np.array(pts)
    p = pts[0]
    v = []
    for i in range(1, len(pts)):
      v.append(pts[i] - p)
    return GenBox(p, v)

  def __init__(self, p, v):
    p = np.array(p).flatten()
    self.n = len(p)
    self.p = p
    self.pv = p.reshape(-1, self.n)
    self.v = np.array(v)
    self.m = np.shape(self.v)[1]

  def norm_shape(self, v, inshape):
    if inshape is None or len(inshape) == 1:
      return v[0]
    return v

  @cmisc.to_numpy_decorator_cl
  def vec(self, d):
    nd = d.reshape((-1, self.m))
    return self.norm_shape(np.matmul(nd, self.v), d.shape)

  @cmisc.to_numpy_decorator_cl
  def get(self, d):
    return self.norm_shape(self.p + self.vec(d), d.shape)

  def corner(self, d):
    p = np.array(self.pv)
    for i in range(self.m):
      if d >> i & 1:
        p += self.vec(opa_math.unit_vec(i, self.n))
    return self.norm_shape(p, None)
  @property
  def low(self): return self.p
  @property
  def high(self): return self.p + self.v

  @property
  def shapely(self):
    assert self.n == 2
    u = []
    for i in (0, 1, 3, 2):
      u.append(self.corner(i))
    return Polygon(u)

  @property
  def mat2world(self):
    res = np.zeros((self.n + 1, self.m + 1))
    res[:-1, :-1] = self.v.T
    res[-1, -1] = 1
    return np.matmul(MatHelper.mat_translate(self.p), res)

  @property
  def norms(self):
    return np.linalg.norm(self.v, axis=1)

  @property
  def area(self):
    return np.abs(np.linalg.det(self.v))

  @property
  def normed(self):
    return GenBox(self.p, self.v / self.norms.reshape((-1, 1)))

  @property
  @cmisc.yield_wrapper
  def corners(self):
    for i in range(4):
      yield self.corner(i)

  @property
  def box(self):
    return Box.FromPoints(self.corners)

  @property
  def mat2local(self):
    return np.linalg.inv(self.mat2world)


class Intervals:

  def __init__(
      self,
      xl=[],
      regions=None,
      use_range_data=False,
      merge=1,
      merge_dist=1,
      is_int=1,
  ):
    self.xl = SortedSet()
    self.use_range_data = use_range_data
    self.merge = merge
    self.merge_dist = merge_dist
    self.is_int = is_int
    if regions is not None:
      xl = [r.getRegion() for r in regions]

    for x in xl:
      self.add(x)
    glog.debug('Intervals >> %s', self.xl)

  def add(self, *args, **kwargs):
    kwargs = dict(kwargs)
    if self.is_int is not None:
      kwargs['is_int'] = self.is_int

    if self.use_range_data:
      elem = Range1DWithData(*args, **kwargs)
    else:
      elem = Range1D(*args, **kwargs)

    if elem.empty:
      return
    pos = self.xl.bisect_left(elem)
    cur = None
    next_pos = pos + 1
    if pos > 0:
      prev = self.xl[pos - 1]
      if self.merge and (prev.contains(elem.low) or prev.contains(elem.low - self.merge_dist)):
        self.xl.pop(pos - 1)

        ne = prev.union(elem)
        self.xl.add(ne)
        cur = ne
        next_pos = pos
      else:
        assert elem.low >= prev.high, (elem, prev)

    if cur is None:
      cur = elem
      self.xl.add(elem)

    next = None
    assert cur == self.xl.pop(next_pos - 1)
    next_pos -= 1
    while next_pos < len(self.xl):
      next = self.xl[next_pos]
      if cur.high < next.low:
        break
      self.xl.pop(next_pos)
      cur = cur.union(next)
      cur.high = max(cur.high, next.high)
    self.xl.add(cur)

    return self

  def filter_dataset(self, dataset):
    nx = []
    ny = []
    for i in range(dataset.n):
      if self.should_keep(dataset.x[i]):
        nx.append(dataset.x[i])
        ny.append(dataset.y[i])
    return dataset.make_new('filter', y=ny, x=nx)

  def should_keep(self, v):
    if self.xl is None:
      return True
    for xlow, xhigh in self.xl:
      if xlow <= v < xhigh:
        return True
    return False

  def split_data(self, dataset):
    res = []
    for v in self.xl:
      res.append(dataset[max(0, v.low):v.high])
    return res
  def make_new(self):
    return Intervals(is_int=self.is_int, merge_dist=self.merge_dist, merge=self.merge)

  @staticmethod
  def FromIndices(vals, merge_dist=1, is_int=1):
    last = None
    res = Intervals(merge_dist=merge_dist, is_int=is_int)
    for x in sorted(vals):
      if last is None or last.high + merge_dist < x:
        if last is not None:
          res.xl.add(last)
        last = Range1D(x, x, is_int=is_int)
      else:
        pass
      last.high = x
    if last is not None:
      res.xl.add(last)

    return res

  def complement(self, superset):
    superset = Range1D(superset, is_int=self.is_int)
    cur = Range1D(superset.low, 0, is_int=self.is_int)
    res = self.make_new()

    for e in list(self.xl) + [Range1D(superset.high, 0, is_int=self.is_int)]:
      if cur.low < e.low:
        cur.high = e.low - self.is_int
        res.xl.add(cur.clone())
      cur.low = e.high + self.is_int
    return res

  def shorten(self, val):
    res = self.make_new()
    for x in self.xl:
      res.add(Range1D(x.low + val, x.high - val, is_int=self.is_int))
    return res

  def expand(self, val):
    res = self.make_new()
    for x in self.xl:
      res.add(Range1D(x.low - val, x.high + val, is_int=self.is_int))
    return res

  def filter(self, func):
    res = self.make_new()
    for x in self.xl:
      if func(x):
        res.add(x)
    return res

  def shift(self, p):
    res = self.make_new()
    for x in self.xl:
      res.add(Range1D(x.low + p, x.high + p))
    return res

  def query(self, q, closest=0):
    pos = self.xl.bisect_left(Range1D(q, math.inf))
    if pos == 0:
      return None
    if closest or q <= self.xl[pos - 1].high:
      return self.xl[pos - 1]
    return None

  def query_data_do(self, q, action, fail_if_not=0):
    obj = self.query(q)

    assert obj is not None or not fail_if_not, hex(q)
    if obj is None:
      return None
    return action(obj.data, q - obj.low)

  def query_data_raw(self, q, **kwargs):
    q = self.query(q, **kwargs)
    if q is None:
      return None
    return q.data

    return found_range.get_data(qr)

  def query_data(self, qr):
    found_range = self.query(qr.low)
    if found_range is None:
      return bytes([0] * (qr.length + 1))
    return found_range.get_data(qr)

  def get_ordered_ranges(self):
    return list(self.xl)

  def intersection(self, other):

    prim_ranges = get_primitive_ranges(self.get_ordered_ranges(), other.get_ordered_ranges())
    res = self.make_new()
    for e in prim_ranges:
      if self.query(e.mid) is not None and other.query(e.mid) is not None:
        res.add(e)
    return res

  def __str__(self):
    s = io.StringIO()
    s.write('Interval:\n')
    for e in self.xl:
      s.write(str(e) + '\n')
    res = s.getvalue()
    s.close()
    return res

  def group_by(self, tb, res=defaultdict(list)):
    res = copy(res)
    for pos, data in tb:
      res[self.query(pos)].append(data)
    return res


def get_primitive_ranges(la, lb):
  res = []
  ia, ib = 0, 0
  pos = -math.inf
  while ia < len(la) and ib < len(lb):
    ra = la[ia]
    rb = lb[ib]
    pos = max(pos, ra.low, rb.low)
    npos = min(ra.high, rb.high)
    if pos <= npos:
      res.append(Range1D(pos, npos))

    if ra.high == npos:
      ia += 1
    if rb.high == npos:
      ib += 1

    pos = npos + 1e-9
  return res


class SparseList:

  def __init__(self):
    self.data = []

  def add(self, pos, tb):
    self.data.append([pos, tb])
    return self

  def clear(self):
    self.data = []

  def result(self):
    self.data.sort(key=lambda x: x[0])
    cur = None
    n = len(self.data)
    res = []
    for i in range(n):
      if cur is None or cur[0] + len(cur[1]) < self.data[i][0]:
        if cur is not None:
          res.append(cur)
        cur = [self.data[i][0], []]
      else:
        assert cur[0] + len(cur[1]) == self.data[i][0], 'must be disjoint'
      cur[1] += self.data[i][1]
    if cur is not None:
      res.append(cur)
    return res


class ListTools:

  @staticmethod
  def Where(lst, cond):
    return [i for i, v in enumerate(lst) if cond(v)]

  @staticmethod
  def SplitIntoBlocks(msg, ids, ignore_split=False):
    res = []
    for i in range(len(ids)):
      low = ids[i]
      if ignore_split:
        low += 1
      high = ids[i + 1] if i + 1 < len(ids) else len(msg)
      if low < high:
        res.append(msg[low:high])
    return res

  @staticmethod
  def BinDiff(lst):
    cur = np.zeros(len(lst[0]), dtype=int)

    lst = list([np.array(x, dtype=int) for x in lst])
    for i in range(len(lst) - 1):
      cur = cur | (lst[i] ^ lst[i + 1])

    return cur


class Quad:

  def __init__(self, pts):
    self.pts = pts

  def tr(self, i):
    return self.pts[Quad.Ids(i)]

  Ids = [
      np.array([0, 1, 2]),
      np.array([0, 2, 3]),
  ]
  IdsPow2 = [
      np.array([0, 1, 3]),
      np.array([0, 3, 2]),
  ]


class RealBox:

  def __init__(self, plist=None):
    if plist is not None:
      p0, p1, p2 = plist
      self.p0 = p0
      self.vx = p1 - p0
      self.vy = p2 - p0

  def __getitem__(self, i):
    p = np.array(self.p0)
    if i & 1:
      p += self.vx
    if i & 2:
      p += self.vy
    return p

  def __str__(self):
    return f'RealBox(p0={self.p0}, vx={self.vx}, vy={self.vy})'

  def get_aabb(self, inner=1):

    #not really correct for inner if too degenerate/skewed
    xlist = []
    ylist = []
    for i in range(4):
      p = self[i]
      xlist.append(p[0])
      ylist.append(p[1])
    xlist.sort()
    ylist.sort()

    if inner:
      return Range2D(Range1D(xlist[1], xlist[2]), Range1D(ylist[1], ylist[2]))
    else:
      return Range2D(Range1D(xlist[0], xlist[3]), Range1D(ylist[0], ylist[3]))


def get_grid_coord_for_n(n):
  nr = math.ceil(n**0.5)
  for i in range(n):
    yield np.array([i // nr, i % nr])


g_unit_box = Box(low=(0, 0), high=(1, 1))
g_one_box = Box(low=(-1, -1), high=(1, 1))
