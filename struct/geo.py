#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
from scipy.spatial import cKDTree
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.base import BaseGeometry
import shapely.geometry
from chdrft.struct.base import Box
from chdrft.graph.base import UnionJoinLax
import math
import heapq
import bisect
import pprint
from sortedcontainers import SortedList

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)



class KBestContainer:
  def __init__(self, k):
    self.k = k
    self.data = SortedList()
    self.mapper = cmisc.IntMapper()

  def add(self, score, v):
    ix = self.mapper.add(v)
    entry = (score, ix)
    if len(self.data) < self.k:
      self.data.add(entry)
      return 1

    if max(self.data) <= entry: return 0
    del self.data[self.k-1]
    self.data.add(entry)
    return 1
  @cmisc.yield_wrapper
  def result(self):
    for x in self.data:
      yield cmisc.Attr(score=x[0], obj=self.mapper.tb[x[1]])



class DynamicKDTree:
  def __init__(self, leafsize=16):
    self.leafsize = leafsize
    self.data = []
    self.bucket0 = []
    self.subtrees = []

  def append(self, pt):
    self.data.append(pt)
    self.bucket0.append(pt)
    if len(self.bucket0) == self.leafsize:
      self.pushit()

  def pushit(self):
    items = self.bucket0
    self.bucket0 = []
    for i in range(len(self.subtrees)):
      if self.subtrees[i] is None:
        self.subtrees[i] = self.make_new(items)
        break
      items.extend(self.subtrees[i].data)
      self.subtrees[i] = None
    else:
      self.subtrees.append(self.make_new(items))

  def make_new(self, pts):
    return cKDTree(pts, leafsize=self.leafsize)

  def query(self, x, k, **kwargs):
    res = []
    for st in self.subtrees:
      if st is None: continue
      dists, ids =st.query(x, k=k, **kwargs)
      for dist, idx in zip(dists, ids):
        if idx != st.n:
          res.append((dist, st.data[idx]))
    for entry in self.bucket0:
      dist = np.linalg.norm(entry - x)
      res.append((dist, entry))

    res.sort(lambda x: x[0])
    return res[:k]

  def query_ball_point(self, x, r, **kwargs):
    res = []
    for st in self.subtrees:
      if st is None: continue
      ids =st.query_ball_point(x, r, **kwargs)
      res.extend(st.data[ids])

    for entry in self.bucket0:
      dist = np.linalg.norm(entry - x)
      if dist < r: res.append(entry)
    return res

class QuadTreeNode:

  def __init__(self, box, depth=0, objs=None, children=None):
    if objs is None: objs = []
    if children is None: children = []
    self.depth = depth
    self.box = box
    self.children = children
    self.objs = objs

  def __iter__(self):
    return iter(self.children)

class QEntry:

  def __init__(self, node, dist):
    self.node = node
    self.dist = dist

  def __lt__(self, other):
    return self.dist < other.dist

  def __cmp__(self, other):
    return cmp(self.dist, other.dist)


class QuadTree:

  def compute_obj_box(self, obj):
    box = self.maybe_get_box(obj)
    if box is not None: return box
    else:
      if isinstance(obj.geo, BaseGeometry):
        bounds = obj.geo.bounds
      else:
        bounds = obj.geo.bounds()
      minbounds = bounds[:2]
      maxbounds = bounds[2:]
      return Box(low=minbounds, high=maxbounds)

  def compute_objs_box(self, objs): return Box.Union([self.compute_obj_box(obj) for obj in objs])

  def __init__(self, objs=None, max_objs=10, max_depth=10, key=None):
    if objs is None: objs = []
    self.root = None
    self.key = key
    self.max_depth = max_depth
    self.max_objs = max_objs
    self.init(objs)

  def init(self, objs):
    self.objs = objs
    self.root = None

    if len(objs) == 0: return
    box = self.compute_objs_box(objs)
    self.root = self.build_node(objs, box.expand(2))


  @property
  def box(self):
    if self.root is None: return Box.Empty()
    return self.root.box

  def remove(self, obj):
    self.remove_internal(obj, self.root)
    self.objs.remove(obj)

  def remove_internal(self, obj, node):
    if not self.box_intersect(node.box, obj): return

    for child in node.children:
      self.remove_internal(obj, child)
    if obj in node.objs: node.objs.remove(obj)


  def add(self, obj):
    if self.root is None:
      self.init([obj])
      return

    bbox = self.compute_obj_box(obj)
    if self.root.box.contains(bbox):
      self.objs.append(obj)
      self.add_internal(obj, self.root)
    else:
      self.init(self.objs + [obj])


    return

    tbox = bbox.expand(1.5)
    expand_factor = max(tbox.dim / self.root.box)
    nsplit = int(math.ceil(math.log2(expand_factor)))
    ndim = self.root.box.dim * (2** nsplit)

    #if wanted to make smallest possible expansion, without rebuilding current root
    #we would have to do something like:
    #start = tbox.low - (tbox.low-self.root.low) % self.root.box

    fbox = tbox.make(center=None, dim=ndim)
    all_objs = self.objs
    all_objs.append(obj)
    self.root = self.build_node(all_objs, fbox)




  def build_node(self, objs, box, depth=0):
    n = QuadTreeNode(box, depth)
    for obj in objs: self.add_internal(obj, n)
    return n


  def maybe_split(self, node):
    if node.depth > self.max_depth or len(node.objs) <= self.max_objs:
      return

    for quadrant in node.box.quadrants:
      interobjs = []
      for obj in node.objs:
        if self.box_intersect(quadrant, obj):
          interobjs.append(obj)

      node.children.append(self.build_node(interobjs, quadrant, node.depth + 1))
    node.objs = []

  def add_internal(self, obj, node):
    if not self.box_intersect(node.box, obj): return

    if not node.children:
      node.objs.append(obj)
      self.maybe_split(node)
    else:
      for child in node.children:
        self.add_internal(obj, child)

  def maybe_get_box(self, obj):
    if self.key is not None: obj = self.key(obj)
    if isinstance(obj, Box):
      return obj
    return getattr(obj, 'box', None)

  def box_intersect(self, box, obj, for_build=1):
    objbox = self.maybe_get_box(obj)
    if objbox is not None:
      inter = box.intersects(objbox)
      if for_build or not inter: return inter
    return box.is_intersecting_obj(obj.geo)

  def get_dist(self, obj, pt, approx=0):
    if approx or not 'geo' in obj:
      objbox = self.maybe_get_box(obj)
      if objbox is not None: return objbox.distance(pt)

    return obj.geo.distance(Point(pt))



  def query(self, pt, filter=lambda x: True, k=1):
    if self.root is None: return None
    res = KBestContainer(k)

    q = [QEntry(self.root, 0)]
    best = None
    best_dist = math.inf
    while len(q) > 0:
      e = heapq.heappop(q)
      if e.dist >= best_dist: break
      n = e.node
      for children in n.children:
        heapq.heappush(q, QEntry(children, children.box.distance(pt)))
      for obj in n.objs:
        dist = self.get_dist(obj, pt, approx=1)
        if dist < best_dist:
          dist = self.get_dist(obj, pt, approx=0)
          if dist < best_dist and not filter(obj): continue
          best_dist = dist
        res.add(dist, obj)

    return cmisc.Attr(kbest=res.result(), dist=best_dist)


  def query_box(self, box, key=lambda x: x, make_uniq=1):
    if not self.root: return []
    vis = BoxQueryVisitor(self, box)
    do_visit(self.root, vis)
    if make_uniq: return cmisc.make_uniq(vis.res, key=key)
    return vis.res


def do_visit(obj, func):
  if func(obj):
    for x in obj:
      do_visit(x, func)


class BoxQueryVisitor:
  def __init__(self, qtree, box):
    self.res = list()
    self.box = box
    self.qtree = qtree

  def __call__(self, node):
    if self.box is not None and not node.box.intersects(self.box): return 1
    for obj in node.objs:
      if self.box is None or self.qtree.box_intersect(self.box, obj, for_build=0): self.res.append(obj)
    return 1


class KDUnionJoin:
  def __init__(self, key=None):
    if key is None: key = lambda x: x
    self.key = key
    self.mapper = cmisc.IntMapper()
    self.mp = cmisc.Attr(key_norm=cmisc.NormHelper.numpy_norm, default=list)

  def extend(self, entries):
    for entry in entries:
      self.add(entry)

  def add(self, entry):
    idx = self.mapper.add(entry)
    pt = self.key(entry)
    self.mp[pt].append(idx)

  def do_uj(self, r=None):
    uj = UnionJoinLax()
    kd = DynamicKDTree()
    rmp = cmisc.Remap.Numpy()
    for pt in self.mp.keys():
      rmp.get(pt)

    kd = cKDTree(rmp.inv)

    for ipt, pt in enumerate(rmp.inv):
      adjlist = kd.query_ball_point(pt, r)
      uj.consider(ipt)
      for adjpt in adjlist:
        uj.join(ipt, adjpt)


    groups = []
    for group in uj.groups():
      rgroup = []
      for entry in group:
        pt = rmp.inv[entry]
        rgroup.extend([self.mapper.tb[x] for x in self.mp[pt]])

      groups.append(rgroup)
    return groups


class OrderedRemapper:

  def __init__(self):
    self.tb = []
    self._t = None

  def add(self, v):
    self.tb.append(v)

  def map(self, pos):
    return self.t[pos]

  @property
  def t(self):
    if self._t is None:
      self._t = list(sorted(set(self.tb)))
    return self._t

  def remap(self, v):
    return bisect.bisect_left(self.t, v)


class PointRemapper:

  def __init__(self):
    self.rmp = cmisc.defaultdict(OrderedRemapper)

  def remap(self, pts):
    for pt in pts:
      for i in range(2):
        self.rmp[i].add(pt[i])
    res = []

    for pt in pts:
      c = []
      res.append(c)
      for i in range(2):
        c.append(self.rmp[i].remap(pt[i]))

    return res

  def remap_rect(self, rect):
    pos = []
    for i in range(2):
      pos.append((self.rmp[i].map(rect.low[i]), self.rmp[i].map(rect.high[i])))
    return Box(xr=pos[0], yr=pos[1])







def test(ctx):
  dkd = DynamicKDTree()
  import chdrft.utils.Z as Z
  import chdrft.utils.K as K
  dim = 3

  data = []
  for i in range(10):
    npts = Z.np.random.uniform(size=(100, dim))
    for j in range(len(npts)):
      dkd.append(npts[j])
    data.extend(npts)


  ptset = []
  for i in range(10):
    px = Z.np.random.uniform(size=(dim,))
    rx = 0.2
    pa = dkd.query_ball_point(px, r=rx)
    print('GOOT ', len(pa))
    ptset.append(cmisc.Attr(points=[px], color='r'))
    ptset.append(cmisc.Attr(points=pa, color='g'))

  ptset.append(cmisc.Attr(points=data, color='gray'))
  K.vispy_utils.render_for_meshes(ptset, cam=1)


def test_kduj(ctx):
  kduj = KDUnionJoin()
  kduj.add((1,1,1))
  kduj.add((0,0,0))
  kduj.add((0.1,0.1,0.1))
  res = kduj.do_uj(r=0.5)
  print(res)

  kduj = KDUnionJoin(key=lambda x: x.pt)
  kduj.add(cmisc.Attr(data=0, pt=(1,1,1)))
  kduj.add(cmisc.Attr(data=0, pt=(0,0,0)))
  kduj.add(cmisc.Attr(data=0, pt=(0.1,0.1,0.1)))
  res = kduj.do_uj(r=0.5)
  print(res)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
