from collections import defaultdict
import chdrft.utils.misc as cmisc


def lambda_minus1():
  return -1


def data_handler(u):
  return cmisc.Attr(handler=lambda x: (x, 1)), 1


class UnionJoinLax:

  def __init__(self, key=None, pairs=[], nodes=[]):
    self.rmp = cmisc.Remap.Numpy(key=key)
    self.par = defaultdict(lambda_minus1)
    self.data = cmisc.Attr(handler=data_handler)
    self._gid = cmisc.Remap()
    self._res = None
    for x in pairs: self.join(*x)
    for x in nodes: self.consider(x)

  def remap(self, a):
    return self.rmp.get(a)

  def consider(self, a):
    self.root(a)

  def root(self, a):
    return self.rmp.rget(self._root(self.remap(a)))

  def is_repr(self, a):
    return self.root(a) == a

  def _root(self, a):
    if self.par[a] == -1:
      return a
    self.par[a] = self._root(self.par[a])
    return self.par[a]

  def compute_res(self):
    if self._res is None:
      self._res = defaultdict(list)
      for k in self.par.keys():
        self._res[self._root(k)].append(self.rmp.rget(k))
      for k in self._res.keys(): self._gid.get(k)

    return self._res

  @property
  def res(self): return self.compute_res()
  @property
  def reprs(self): return [self.rmp.rget(x) for x in self.res.keys()]

  def gid(self, a):
    self.compute_res()
    return self._gid.get(self._root(self.rmp.get(a)), assert_in=1)

  def group(self, a):
    return self.res[self._root(self.remap(a))]

  def groups(self):
    return list(self.res.values())

  def join_multiple(self, lst, update_dict=None):
    objs = {}

    if update_dict is not None:
      for k, v in update_dict.items():
        objs[k] = self.data[k][self.root(v)]

    r = None
    for u in lst[1:]:
      r = self.join(lst[0], u)

    if update_dict is not None:
      for k, v in objs.items():
        self.data[k][r] = v
    return r

  def join(self, a, b, **kwargs):
    return self.make_par_child(a, b, lax=1)

  def make_par_child(self, a, b, update_dict=None, norm=1, lax=0):
    if norm:
      a = self._root(self.remap(a))
      b = self._root(self.remap(b))

    if not lax: assert a != b
    if a == b: return self.rmp.rget(a)
    # a is new root

    if update_dict is not None:
      for k, v in update_dict.items():
        self.data[k][a] = self.data[k][self._root(v)]
    assert a != b
    self.par[b] = a
    return self.rmp.rget(a)


class OpaGraph:

  class UnionJoin:

    def __init__(self, n):
      self.n = n
      self.par = [-1] * n

    def root(self, a):
      if self.par[a] == -1:
        return a
      self.par[a] = self.root(self.par[a])
      return self.par[a]

    def groups(self):
      res = defaultdict(list)
      for i in range(self.n):
        res[self.root(i)].append(i)
      return list(res.values())

    def join(self, a, b):
      b = self.root(b)
      a = self.root(a)
      if a == b: return
      assert a != b
      self.par[b] = a

  class Edge:

    def __init__(self, lca_req, dest):
      self.lca_req = lca_req
      self.dest = dest

  class LcaReq:

    def __init__(self, a, b):
      self.a = a
      self.b = b
      self.a.e_list.append(OpaGraph.Edge(self, b))
      self.b.e_list.append(OpaGraph.Edge(self, a))
      # a depends on b

    def set(self, r1, r2):
      if r1[0] != self.a:
        r1, r2 = r2, r1
      r2[1].sibling_dep.append(r1[1])

  class Node:

    def __init__(self, data=None):
      self.parent = None
      self.children = []
      self.e_list = []
      self.num = None

      self.vis = False
      self.open = None
      self.sibling_dep = []
      self.rmp = data
      self.order = []

    def dfs_num(self, tb):
      self.num = len(tb)
      tb.append(self)

      for x in self.children:
        x.dfs_num(tb)

    def disp(self):
      print('node ', self.num, 'children:', [x.num for x in self.children])
      for x in self.children:
        x.disp()

    def dfs_ordering(self):
      for x in self.children:
        x.count = 0

      for x in self.children:
        for y in x.sibling_dep:
          y.count += 1

      orig = [x for x in self.children if x.count == 0]
      q = deque(orig, len(self.children))

      self.order = []
      while len(q) > 0:
        cur = q.pop()
        self.order.append(cur)
        for dep in cur.sibling_dep:
          dep.count -= 1
          if dep.count == 0:
            q.appendleft(dep)
      for x in self.children:
        x.dfs_ordering()

      assert len(self.order) == len(self.children)

  def __init__(self, root):
    self.root = root

  def add_dep(self, a, b):
    OpaGraph.LcaReq(a, b)

  def solve_ordering(self):
    self.tb = []
    self.root.dfs_num(self.tb)
    self.n = len(self.tb)
    # self.root.disp()
    self.union = OpaGraph.UnionJoin(self.n)
    self.dfs_lca(self.root)
    self.root.dfs_ordering()

  def dfs_lca(self, node):
    for x in node.children:
      node.open = x
      self.dfs_lca(x)

    node.vis = True
    for e in node.e_list:
      if not e.dest.vis:
        continue

      b = self.union.root(e.dest.num)
      b_node = self.tb[b]
      if b_node.parent == node:
        # node ancestor of b
        continue

      e.lca_req.set((e.dest, self.tb[b]), (node, b_node.parent.open))
    node.open = None

    for e in node.children:
      self.union.join(node.num, e.num)
