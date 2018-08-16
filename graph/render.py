#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import glog
import networkx as nx

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst)


def get_node_data(tree, node):
  return tree.node[node]['data']

class DrawTree(object):

  def __init__(self, tree, node, parent=None, depth=0, number=1):
    self.x = -1.
    self.y = depth
    self.tree = tree
    children = list(tree.neighbors(node))
    children.sort(key=lambda x:get_node_data(tree, x).order)

    self.children = [DrawTree(tree, c, self, depth + 1, i + 1) for i, c in enumerate(children)]
    data = get_node_data(tree, node)
    data.renderdata = self

    self.parent = parent
    self.thread = None
    self.mod = 0
    self.ancestor = self
    self.change = self.shift = 0
    self._lmost_sibling = None
    #this is the number of the node in its group of siblings 1..n
    self.number = number

  def left(self):
    return self.thread or len(self.children) and self.children[0]

  def right(self):
    return self.thread or len(self.children) and self.children[-1]

  def lbrother(self):
    n = None
    if self.parent:
      for node in self.parent.children:
        if node == self: return n
        else: n = node
    return n

  def get_lmost_sibling(self):
    if not self._lmost_sibling and self.parent and self != \
    self.parent.children[0]:
      self._lmost_sibling = self.parent.children[0]
    return self._lmost_sibling

  lmost_sibling = property(get_lmost_sibling)

  def __str__(self):
    return "%s: x=%s mod=%s" % (self.tree, self.x, self.mod)

  def __repr__(self):
    return self.__str__()


def buchheim(tree):
  dt = firstwalk(tree)
  min = second_walk(dt)
  if min < 0:
    third_walk(dt, -min)
  return dt


def third_walk(tree, n):
  tree.x += n
  for c in tree.children:
    third_walk(c, n)


def firstwalk(v, distance=1.):
  if len(v.children) == 0:
    if v.lmost_sibling:
      v.x = v.lbrother().x + distance
    else:
      v.x = 0.
  else:
    default_ancestor = v.children[0]
    for w in v.children:
      firstwalk(w)
      default_ancestor = apportion(w, default_ancestor, distance)
    print("finished v =", v.tree, "children")
    execute_shifts(v)

    midpoint = (v.children[0].x + v.children[-1].x) / 2

    ell = v.children[0]
    arr = v.children[-1]
    w = v.lbrother()
    if w:
      v.x = w.x + distance
      v.mod = v.x - midpoint
    else:
      v.x = midpoint
  return v


def apportion(v, default_ancestor, distance):
  w = v.lbrother()
  if w is not None:
    #in buchheim notation:
    #i == inner; o == outer; r == right; l == left; r = +; l = -
    vir = vor = v
    vil = w
    vol = v.lmost_sibling
    sir = sor = v.mod
    sil = vil.mod
    sol = vol.mod
    while vil.right() and vir.left():
      vil = vil.right()
      vir = vir.left()
      vol = vol.left()
      vor = vor.right()
      vor.ancestor = v
      shift = (vil.x + sil) - (vir.x + sir) + distance
      if shift > 0:
        move_subtree(ancestor(vil, v, default_ancestor), v, shift)
        sir = sir + shift
        sor = sor + shift
      sil += vil.mod
      sir += vir.mod
      sol += vol.mod
      sor += vor.mod
    if vil.right() and not vor.right():
      vor.thread = vil.right()
      vor.mod += sil - sor
    else:
      if vir.left() and not vol.left():
        vol.thread = vir.left()
        vol.mod += sir - sol
      default_ancestor = v
  return default_ancestor


def move_subtree(wl, wr, shift):
  subtrees = wr.number - wl.number
  print(wl.tree, "is conflicted with", wr.tree, 'moving', subtrees, 'shift', shift)
  #print wl, wr, wr.number, wl.number, shift, subtrees, shift/subtrees
  wr.change -= shift / subtrees
  wr.shift += shift
  wl.change += shift / subtrees
  wr.x += shift
  wr.mod += shift


def execute_shifts(v):
  shift = change = 0
  for w in v.children[::-1]:
    print("shift:", w, shift, w.change)
    w.x += shift
    w.mod += shift
    change += w.change
    shift += w.shift + change


def ancestor(vil, v, default_ancestor):
  #the relevant text is at the bottom of page 7 of
  #"Improving Walker's Algorithm to Run in Linear Time" by Buchheim et al, (2002)
  #http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8757&rep=rep1&type=pdf
  if vil.ancestor in v.parent.children:
    return vil.ancestor
  else:
    return default_ancestor


def second_walk(v, m=0, depth=0, min=None):
  v.x += m
  v.y = depth

  if min is None or v.x < min:
    min = v.x

  for w in v.children:
    min = second_walk(w, m + v.mod, depth + 1, min)

  return min


r = 30
rh = r * 1.5
rw = r * 1.5


def drawt(root, depth):
  global r
  oval(root.x * rw, depth * rh, r, r)
  print(root.x)
  for child in root.children:
    drawt(child, depth + 1)


def drawconn(root, depth):
  for child in root.children:
    line(root.x * rw + (r / 2), depth * rh + (r / 2), child.x * rw + (r / 2),
         (depth + 1) * rh + (r / 2))
    drawconn(child, depth + 1)




def test(ctx):
  g = nx.DiGraph()
  g.add_node(0, order=len(g))
  g.add_node(1, order=len(g))
  g.add_node(2, order=len(g))
  g.add_node(3, order=len(g))
  g.add_node(4, order=len(g))
  g.add_node(5, order=len(g))
  g.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4), (4, 5)])
  tree = DrawTree(g, 0)
  buchheim(tree)
  print(nx.get_node_attributes(g, 'data'))
  for node in g:
    print(g[node])
    u = g.node[node]['data']
    print(u.x, u.y)



def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
