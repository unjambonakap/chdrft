#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import A
import chdrft.utils.misc as cmisc
import glog
import numpy as np
import random
import shapely
from chdrft.utils.math import Normalizer
from chdrft.utils.math import *
from shapely.geometry import Polygon, LineString, Point, MultiPoint
from shapely.geometry.base import BaseGeometry
import chdrft.graph.base as graph_base
from scipy.spatial import distance
import math

global flags, cache
flags = None
cache = None
g_eps = 1e-5

def circle_polyline(c, npts=100):
  rx = np.linspace(0, 2 * np.pi, npts)
  return np.array(list(zip(*(np.cos(rx), np.sin(rx))))) * c.r + c.pos

def get_points(e):
  if isinstance(e, (shapely.geometry.Polygon, shapely.geometry.MultiPoint)):
    e=  e.exterior.coords[1:]
  elif isinstance(e, shapely.geometry.LinearRing):
    e= e.coords
  elif isinstance(e, shapely.geometry.Point):
    return np.array(e)
  elif isinstance(e, BaseGeometry):
    e= np.array(list(e))

  return np.array(e)

def to_shapely(obj, poly=1):
  from chdrft.struct.base import Box
  if isinstance(obj, Box): return obj.shapely
  if isinstance(obj, Line): return obj.shapely
  if isinstance(obj, BaseGeometry): return obj
  if isinstance(obj, np.ndarray):
    if len(np.shape(obj)) == 1: return Point(obj)
    if poly: return Polygon(list(obj))
    return MultiPoint(list(obj))
  if isinstance(obj, list):
    return shapely.geometry.GeometryCollection(list(map(to_shapely, obj)))





class Circle:
  def __init__(self, pos=None, r=None):
    self.pos = pos
    self.r = r

  def __repr__(self):
    return str(self)

  def __str__(self):
    return f'Circle(r={self.r}, pos={self.pos})'

  def __contains__(self, p):
    return np.linalg.norm(p - self.pos) < self.r + g_eps

  def polyline(self, npts=100):
    return circle_polyline(self, npts)

  def unwhiten(self, normalizer):
    return Circle(pos=normalizer.unwhiten(self.pos),
                      r=normalizer.unwhiten(self.r, vec=1),
                      )

  def __add__(self, other):
    if isinstance(other, Circle):
      return Circle(pos=self.pos+other.pos, r=self.r +other.r)
    else:
      return Circle(pos=self.pos+other, r=self.r)

  def _div__(self, other):
    return Circle(pos=self.pos / other, r=self.r / other)

class CircleState(Circle):

  def __init__(self, pfrom):
    self.pfrom = pfrom
    if len(pfrom) == 2:
      pos = (pfrom[0] + pfrom[1]) / 2
    else:
      assert len(pfrom) == 3
      l1 = Line.Bisect(pfrom[0], pfrom[1])
      l2 = Line.Bisect(pfrom[0], pfrom[2])
      pos = l1.intersection(l2)

    r=np.linalg.norm(pfrom[0] - pos)
    super().__init__(pos=pos, r=r)



def smallest_circle_p2(p1, p2, pts):
  curp = CircleState((p1, p2))
  l = Line(p1, p2)
  pl = []
  curside = cmisc.SingleValue()

  for i in range(len(pts)):
    if pts[i] in curp: continue
    side = l.side(pts[i])
    assert side != 0
    curside.set(side)
    nc = CircleState((p1, p2, pts[i]))
    pl.append(nc)
  res = max(pl, key=lambda x: x.r, default=curp)
  for pt in pts:
    assert pt in res
  return res


def smallest_circle_p1(p1, pts):
  curp = CircleState((p1, pts[0]))
  for i in range(1, len(pts)):
    if pts[i] not in curp:
      nc = smallest_circle_p2(p1, pts[i], pts[:i])
      assert nc.r >= curp.r
      curp = nc
  return curp


@cmisc.to_numpy_decorator
def smallest_circle(pts):
  normalizer = Normalizer(pts, same_factor=1)
  pts = normalizer.normalized
  np.random.shuffle(pts)

  curp = CircleState(pts[:2])
  for i in range(2, len(pts)):
    if pts[i] not in curp:
      nc = smallest_circle_p1(pts[i], pts[:i])
      assert nc.r >= curp.r - g_eps
      curp = nc
  for pt in pts:
    assert pt in curp
  return curp.unwhiten(normalizer)


class Line:

  @staticmethod
  @cmisc.to_numpy_decorator
  def Bisect(u, v):
    d = orth_v2(make_norm(u - v))
    p = (u + v) / 2
    return Line(p, p + d)

  def __init__(self, a, b, **kwargs):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    self.data = kwargs
    self.a = a
    self.b = b
    v = (b - a)
    self.v = v
    self.d = np.linalg.norm(v)
    self.vn = v / self.d

    if len(self.vn) == 2:
      self.k = orth_v2(self.vn)
      self.c = np.dot(self.k, self.a)

  def __call__(self, dist, normed_dist=0):
    if normed_dist: return self.a + self.v * dist
    return self.a + self.vn * dist


  def fix_coord(self, v, coord_pos):
    dist = (v - self.a[coord_pos]) / self.vn[coord_pos]
    return self(dist)

  def plane_intersection(self, plane):
    da = np.dot(plane.center - self.a, plane.normal) / np.dot(plane.normal, self.vn)
    return self(da)

  def intersection(self, other):
    assert isinstance(other, Line)
    m = np.zeros((2, 2))
    m[0, :] = self.k
    m[1, :] = other.k
    b = [self.c, other.c]
    x = np.linalg.solve(m, b)
    return x

  def side(self, pt):
    return np.sign(np.dot(self.k, pt) - self.c)

  @property
  def box(self):
    from  chdrft.struct.base import Box
    return Box(*self.bound)

  def bounds(self):
    m1 = np.minimum(self.a, self.b)
    m2 = np.maximum(self.a, self.b)
    return m1[0], m1[1], m2[0], m2[1]

  def proj(self, p):
    u = np.dot(self.k, p - self.a)
    pp = p - u * self.k
    x = np.dot(pp - self.a, self.vn)
    if x >= 0 and x <= self.d: return pp
    if x < 0: return self.a
    return self.b

  def point_distance(self, p):
    u = np.dot(self.k, p - self.a)
    pp = p - u * self.k
    x = np.dot(pp - self.a, self.vn)
    if x >= 0 and x <= self.d: return abs(u)
    if x < 0: return np.linalg.norm(self.a - p)
    return np.linalg.norm(self.b - p)

  def box_distance(self, box):
    #dist = math.inf
    #for i in range(4):
    #  dist=  min(dist, self.point_distance(box.get(i)))
    #  line = box.line(i)
    #  dist = min(dist, line.point_distance(self.a))
    #  dist = min(dist, line.point_distance(self.b))
    return box.shapely.distance(LineString([self.a, self.b]))

  def distance(self, p):
    if isinstance(p, Point): return self.point_distance((p.x, p.y))
    return self.point_distance(p)

  def __str__(self):
    return f'Line({self.a}, {self.b})'

  @property
  def shapely(self):
    return LineString([self.a, self.b])

def make_poly_ccw(pts):
  p = Polygon(pts)
  if not p.exterior.is_ccw: pts = pts[::-1]
  return pts


class PointCloud:

  def __init__(self, merge_dist):
    self.pts = []
    self.uj = graph_base.UnionJoinLax(key=id)
    self.merge_dist = merge_dist

  def should_merge(self, a, b):
    return distance.euclidean(a, b) < self.merge_dist

  def add(self, p):
    self.uj.consider(p)
    for pt in self.pts:
      if self.should_merge(pt, p):
        self.uj.join(pt, p)
    self.pts.append(p)

  def find_pt(self, p):
    for pt in self.pts:
      if self.should_merge(pt, p):
        return A(repr=self.uj.rmp.get(pt), n=len(self.uj.group(pt)))
    return A(repr=None, n=0)

  def repr(self, p):
    return self.uj.root(p)

  @property
  def result(self):
    res = list(self.uj.groups())
    from  chdrft.struct.base import Box
    for group in res:
      box = Box.FromPoints(group)
      assert max(box.width, box.height) < 2 * self.merge_dist
    return res







def mod_close(a, n):
  a %= n
  da = a - n
  return a if a < abs(da) else da


def check_contiguous(lst, n):
  pl = list(map(lambda x: mod_close(x, n), lst))
  return max(pl) - min(pl) == len(pl) - 1


def analyse_contiguous(lst):
  lst.sort()
  for i in range(1, len(lst)):
    if lst[i - 1] != lst[i] - 1:
      return cmisc.Attr(cont=0, skip=lst[i])
  return cmisc.Attr(cont=1)



def enumerate_lines(pts):
  n = len(pts)
  assert n > 1
  for i in range(n):
    yield pts[i], pts[(i + 1) % n]


def vector_angles(a, b):
  angle_norm = abs(np.dot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b)
  return math.acos(min(1, angle_norm))



@cmisc.to_numpy_decorator
def compute_plane(a, b, c):
  d1 = b-a
  d2 = c-a
  k  = make_norm(np.cross(d1, d2))
  return cmisc.Attr(center=a, normal=k, v=np.dot(a, k))


def colinear4(a, b, c, d):
  a = np.array(a)
  b = np.array(b)
  c = np.array(c)
  d = np.array(d)

  if not colinear(d - c, b - a):
    return False

  angle_norm = vector_angles(d - c, c - b)
  #print(a, b, c, d, angle_norm, d - c, c - b)
  return angle_norm < g_angle_colinear_max

def numpy_choice(samples, ratio):
  sel_items = np.random.choice(len(samples), size=int(len(samples) * ratio), replace=False)
  for sel_id in sel_items:
    yield samples[sel_id]

def make_uniq_circ(lst, circ=0):

  yield lst[0]
  for i in range(1, len(lst)):
    if lst[i] != lst[i - 1] and (circ == 0 or i + 1 != len(lst) or lst[i] != lst[0]):
      yield lst[i]



def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test_smallest(ctx):
  np.random.seed(0)
  npts = 100
  pts = np.random.uniform(size=(npts, 2)) * 100
  res = smallest_circle(pts)
  import chdrft.utils.K as K

  clines = circle_polyline(res)

  K.vispy_utils.render_for_meshes(
      cmisc.Attr(points=pts, color='r'),
      cmisc.Attr(points=[res.pos], lines=[clines], color='g'),
  )

def test_ops(ctx):
  print(rotate_vec((1,0,0), (0,0,1), 0.1))
  print(rotate_vec((1,0,0), (0,1,0), 0.1))
  print(rotate_vec((1,0,0), (0,1,0), np.pi))
  print(rotate_vec((1,0,0), (0,1,0), np.pi/2))

def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
