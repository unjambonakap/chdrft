import sys
import glog
import bisect
import numpy as np
from collections import defaultdict

from chdrft.utils.misc import to_list, Attributize, proc_path
from sortedcontainers import SortedSet
import math
import io


class Range1D:

  def __init__(self, *args, n=None):
    if len(args) == 1:
      range1d = args[0]
      if isinstance(range1d, Range1D):
        self.low = range1d.low
        self.high = range1d.high
      else:
        if isinstance(range1d, int):
          self.low = range1d
          self.high = self.low + n
        else:
          self.low = range1d[0]
          self.high = range1d[1]
    elif len(args) == 2:
      self.low = args[0]
      self.high = args[1]
    else:
      assert 0

    self._empty = self.low > self.high

  def clone(self):
    return Range1D(self.low, self.high)

  @property
  def empty(self):
    return self._empty

  def clamp(self, *args):
    other = Range1D(*args)
    self.low = max(self.low, other.low)
    self.high = min(self.high, other.high)
    return self

  def clampv(self, v):
    if self.low is not None: v = max(v, self.low)
    if self.high is not None: v = min(v, self.high)
    return v

  def contains(self, other):
    if isinstance(other, Range1D):
      return self.low <= other.low and other.high <= self.high
    else:
      return self.low <= other <= self.high

  def double(self):
    mid = self.mid
    return Range1D([3 * self.low - 2 * mid, 3 * self.high - 2 * mid])

  def expand(self, ratio):
    mid = self.mid
    l = self.length() / 2 * ratio
    return Range1D(mid - l, mid + l)

  def cond(self, x):
    return self.low <= x and x <= self.high

  def cond_tuple(self, x):
    return (self.low <= x, x <= self.high)

  def shift(self, v):
    return Range1D(self.low + v, self.high + v)

  def length(self):
    return self.high - self.low

  def __len__(self):
    return self.length()

  def intersect(self, other, adj=0):
    return not self.intersection(other, adj=adj).empty

  def intersection(self, other, adj=0):
    return Range1D(max(self.low, other.low), min(self.high, other.high) + adj)

  def union(self, other):
    return Range1D(min(self.low, other.low), max(self.high, other.high))

  def __lt__(self, other):
    return self.as_tuple < other.as_tuple

  def __eq__(self, other):
    return self.as_tuple == other.as_tuple

  @property
  def mid(self):
    return (self.low + self.high) / 2

  @property
  def as_tuple(self):
    return (self.low, self.high)

  def __hash__(self):
    return self.as_tuple.__hash__()

  def __str__(self):
    if isinstance(self.low, int):
      return 'Range1D(%x, %x)' % (self.low, self.high)
    return f'Range1D({self.low}, {self.high})'

  def __repr__(self):
    if isinstance(self.low, int): return repr(f'{self.low:x}, {self.high:x}')
    return repr(f'{self.low}, {self.high}')


class Range1DWithData(Range1D):

  def __init__(self, *args, data=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.data = data

  def union(self, other):
    if other.high + 1 == self.low: return other.union(self)
    assert self.high + 1 == other.low
    return Range1DWithData(self.low, other.high, data=self.data + other.data)

  def get_data(self, qrange):
    assert self.contains(qrange)
    i1 = qrange.low - self.low
    i2 = qrange.high - self.low
    return self.data[i1:i2 + 1]


class Range2D:

  def __init__(self, *args):
    if len(args) == 1:
      range2d = args[0]
      if isinstance(range2d, Range2D):
        self.xr = range2d.xr
        self.yr = range2d.yr
      else:
        self.xr = Range1D(range2d[0])
        self.yr = Range1D(range2d[1])
    elif len(args) == 2:
      self.xr = Range1D(args[0])
      self.yr = Range1D(args[1])
    else:
      assert 0

  def contains(self, other):
    return self.xr.contains(other.xr) and self.yr.contains(other.yr)

  def double(self):
    return Range2D(self.xr.double(), self.yr.double())

  def shift(self, vec):
    return Range2D(self.xr.shift(vec[0]), self.yr.shift(vec[1]))

  def length(self):
    return pg.Point(self.xr.length(), self.yr.length())


class Intervals:

  def __init__(self, xl=[], regions=None, use_range_data=False):
    self.xl = SortedSet()
    self.use_range_data = use_range_data
    if regions is not None:
      xl = [r.getRegion() for r in regions]
    else:
      xl = to_list(xl)
      if len(xl) > 0 and not isinstance(xl[0], list) and not isinstance(xl[0], tuple):
        xl = [xl]

    for x in xl:
      self.add(x)
    glog.debug('Intervals >> %s', self.xl)

  def add(self, *args, **kwargs):

    if self.use_range_data:
      elem = Range1DWithData(*args, **kwargs)
    else:
      elem = Range1D(*args, **kwargs)

    if elem.empty: return
    pos = self.xl.bisect_left(elem)
    cur = None
    next_pos = pos + 1
    if pos > 0:
      prev = self.xl[pos - 1]
      if prev.contains(elem.low - 1):
        self.xl.pop(pos - 1)

        ne = prev.union(elem)
        self.xl.add(ne)
        cur = ne
        next_pos = pos

    if cur is None:
      cur = elem
      self.xl.add(elem)

    next = None
    assert cur == self.xl.pop(next_pos - 1)
    next_pos -= 1
    while next_pos < len(self.xl):
      next = self.xl[next_pos]
      if cur.high + 1 < next.low:
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

  @staticmethod
  def FromIndices(vals, merge_dist=1):
    last = None
    res = Intervals()
    for x in vals:
      if last is None or last.high + merge_dist < x:
        if last is not None:
          res.xl.add(last)
        last = Range1D(x, x)
      last.high = x
    if last is not None:
      res.xl.add(last)

    return res

  def complement(self, superset):
    superset = Range1D(superset)
    cur = Range1D(superset.low, 0)
    res = Intervals()

    for e in list(self.xl) + [Range1D(superset.high, 0)]:
      cur.high = e.low - 1
      res.xl.add(cur.clone())
      cur.low = e.high + 1
    return res

  def shorten(self, val):
    res = Intervals()
    for x in self.xl:
      res.add(Range1D(x.low + val, x.high - val))
    return res

  def expand(self, val):
    res = Intervals()
    for x in self.xl:
      res.add(Range1D(x.low - val, x.high + val))
    return res

  def filter(self, func):
    res = Intervals()
    for x in self.xl:
      if func(x):
        res.add(x)
    return res

  def shift(self, p):
    res = Intervals()
    for x in self.xl:
      res.add(Range1D(x.low + p, x.high + p))
    return res

  def query(self, q):
    pos = self.xl.bisect_left(Range1D(q, math.inf))
    if pos == 0: return None
    if q <= self.xl[pos - 1].high: return self.xl[pos - 1]
    return None

  def query_data(self, qr):
    found_range = self.query(qr.low)
    if found_range is None: return bytes([0] * (qr.length() + 1))
    return found_range.get_data(qr)

  def get_ordered_ranges(self):
    return list(self.xl)

  def intersection(self, other):

    prim_ranges = get_primitive_ranges(self.get_ordered_ranges(), other.get_ordered_ranges())
    res = Intervals()
    for e in prim_ranges:
      if self.query(e.low) and other.query(e.low):
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


def get_primitive_ranges(la, lb):
  res = []
  ia, ib = 0, 0
  pos = -math.inf
  while ia < len(la) and ib < len(lb):
    ra = la[ia]
    rb = lb[ib]
    pos = max(pos, ra.low, rb.low)
    npos = min(ra.high, rb.high)
    if pos <= npos: res.append(Range1D(pos, npos))

    if ra.high == npos: ia += 1
    if rb.high == npos: ib += 1

    pos = npos + 1
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
        if cur is not None: res.append(cur)
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
      if ignore_split: low += 1
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


def mat_translate(v):
  return np.matrix([[1, 0, v[0]], [0, 1, v[1]], [0, 0, 1]])


def mat_scale(v):
  return np.diag([v[0], v[1], 1])


def mat_rot(rot_z):
  return np.array(
      (
          (math.cos(rot_z), -math.sin(rot_z), 0),
          (math.sin(rot_z), math.cos(rot_z), 0),
          (0, 0, 1),
      )
  )


class Box:

  def __init__(self, *args, **kwargs):

    d = defaultdict(lambda: None, kwargs)
    for k in d.keys():
      d[k] = np.array(d[k])
    dim, low, high, center = d['dim'], d['low'], d['high'], d['center']
    if len(args) == 4 and len(kwargs) == 0:
      low = np.array((args[:2]))
      high = np.array((args[2:]))
    elif len(args) == 2 and len(kwargs) == 0:
      low = high = np.array(args)
    elif len(args) == 1 and len(kwargs) == 0:
      low = high = args[0]

    elif len(args) == 0 and len(kwargs) == 0:
      self.dim = np.array((0, 0))
      self.low = np.array((math.inf, math.inf))
      self.high = np.array((-math.inf, -math.inf))
      self.center = np.array((0, 0))
      return

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
    self.low = low
    self.dim = dim
    self.high = low + dim
    self.center = low + dim / 2

  def union_points(self, pts):
    if isinstance(pts, np.ndarray): tb = pts
    else:
      pts = list(pts)
      tb = np.array(pts)

    nlow = np.minimum(self.low, np.min(pts, axis=0)[:2])
    nhigh = np.maximum(self.high, np.max(pts, axis=0)[:2])
    res = Box(low=nlow, high=nhigh)
    return res

  def union(self, other):
    if not isinstance(other, Box): other = Box(other)
    return Box(low=np.minimum(self.low, other.low), high=np.maximum(self.high, other.high))

  def empty(self):
    return np.any(self.low >= self.high)

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
  def width(self):
    return self.dim[0]

  @property
  def height(self):
    return self.dim[1]

  @property
  def aspect(self):
    return self.width / self.height

  def mat_translate_from(self):
    return mat_translate(self.low)

  def mat_translate_to(self):
    return mat_translate(-self.low)

  def mat_scale_from(self):
    return mat_scale(self.dim)

  def mat_scale_to(self):
    return mat_scale(1. / self.dim)

  def mat_to(self):
    return np.matmul(self.mat_scale_to(), self.mat_translate_to())

  def mat_from(self):
    return np.matmul(self.mat_translate_from(), self.mat_scale_from())

  def poly(self, closed=False, z_coord=None):
    res = []
    for i in range(4):
      res.append(self.get(i, z_coord=z_coord))
    if closed: res.append(self.get(0, z_coord=z_coord))
    return np.array(res)

  def get(self, i, z_coord=None):
    p = [self.low[0], self.low[1]]
    if z_coord is not None: p.append(z_coord)
    if (i & 1) ^ (i >> 1 & 1): p[0] = self.high[0]
    if (i & 2): p[1] = self.high[1]
    return p

  def __getitem__(self, i):
    return self.get(i)

  def zero_corner(self):
    return Box(low=(0, 0), dim=self.dim)

  def __str__(self):
    return f'Box(low={self.low}, high={self.high}, center={self.center}, dim={self.dim})'

  def get_window(self):
    return (
        slice(self.low[1], self.low[1] + self.dim[1]),
        slice(self.low[0], self.low[0] + self.dim[0])
    )

  def contains(self, pt):
    return np.all((self.low <= pt) & (pt <= self.high))

  def area(self):
    return self.dim[0] * self.dim[1]

  def subimg(self, img):
    return get_subimg(img, *self.low, *self.high)

  def overlaps(self, other):
    low2 = np.maximum(self.low, other.low)
    high2 = np.minimum(self.high, other.high)
    return np.all(low2 <= high2)

  def flip_y(self, my):
    return Box(low=(self.low[0], my - self.high[1]), high=(self.high[0], my - self.low[1]))

  def scale(self, fx, fy):
    return Box(low=self.low * (fx, fy), dim=self.dim * (fx, fy))

  def set_aspect(self, ratio):
    nwidth = max(self.width, ratio * self.height)
    nheight = max(self.width / ratio, self.height)
    return Box(center=self.center, dim=(nwidth, nheight))

  def expand(self, f):
    return Box(center=self.center, dim=self.dim * f)

  def translate(self, v):
    return Box(center=self.center + np.array(v), dim=self.dim)

  def interpolate(self, other, alpha):
    # (1-alpha) * this + alpha  * other
    return Box(
        center=(1 - alpha) * self.center + alpha * other.center,
        dim=(1 - alpha) * self.dim + alpha * other.dim
    )
