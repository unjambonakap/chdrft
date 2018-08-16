from collections import defaultdict

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
      if a==b: return
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

    def __init__(self, data):
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
