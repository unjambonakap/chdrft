#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class IntervalTree:

  def __init__(self, poslist):
    poslist = list(sorted(set(poslist)))
    poslist.append(poslist[-1] + 1)
    self.mp = {}

    for i, v in enumerate(poslist):
      self.mp[v] = i

    self.root = IntervalNode.BuildFrom(poslist)

  def to_id(self, x):
    return self.mp[x]

  def add_score(self, r1, r2, v):
    assert r1 in self.mp
    assert r2 in self.mp
    if r1 > r2: r1, r2 = r2, r1

    if r1 == r2: self.root.add_score_one(r1, v)
    else:
      self.root.add_score(r1, r2, v / (r2 - r1))  # repartition lineique du score

  def get_score(self, r1, r2):
    return self.root.get_score(r1, r2)


class IntervalNode:

  def __init__(self, T=None, H=None):
    self.L = None
    self.R = None
    self.T = T
    self.H = H
    self.score = 0
    self.score_impulse = 0

  @staticmethod
  def BuildFrom(lst):
    if len(lst) == 1: return None
    res = IntervalNode(T=lst[0], H=lst[-1])
    if len(lst) == 2: return res

    mid = len(lst) // 2
    res.L = IntervalNode.BuildFrom(lst[:mid + 1])
    res.R = IntervalNode.BuildFrom(lst[mid:])
    return res

  def add_score_one(self, r1, v):
    if r1 < self.T or r1 >= self.H: return
    self.score_impulse += v
    if self.L:
      self.L.add_score_one(r1, v)
      self.R.add_score_one(r1, v)

  def add_score(self, r1, r2, v):
    r1 = max(r1, self.T)
    r2 = min(r2, self.H)
    if r1 >= r2: return
    self.score_impulse += v * (r2 - r1)
    if r1 == self.T and r2 == self.H:
      self.score += v
      return

    if self.L:
      self.L.add_score(r1, r2, v)
      self.R.add_score(r1, r2, v)

  def get_score(self, r1, r2):
    r1 = max(r1, self.T)
    r2 = min(r2, self.H)
    if r1 <= self.T and self.H <= r2:
      return self.score_impulse
    if r1 >= r2: return 0

    res = self.score * (r2 - r1)
    if self.L:
      res += self.L.get_score(r1, r2)
      res += self.R.get_score(r1, r2)
    elif r1 == self.T:
      res += self.score_impulse
    return res


class SegmentTree:

  def __init__(self, poslist):
    poslist = list(sorted(set(poslist)))
    poslist.append(poslist[-1] + 1)
    self.mp = {}

    for i, v in enumerate(poslist):
      self.mp[v] = i

    self.root = SegmentTreeNode.BuildFrom(poslist)

  def to_id(self, x):
    return self.mp[x]


class SegmentTreeNode:

  def __init__(self, T=None, H=None):
    self.L = None
    self.R = None
    self.T = T
    self.H = H
    self.score = 0

  @staticmethod
  def BuildFrom(lst):
    if len(lst) == 1: return None
    res = SegmentTreeNode(T=lst[0], H=lst[-1])
    if len(lst) == 2: return res

    mid = len(lst) // 2
    res.L = SegmentTreeNode.BuildFrom(lst[:mid + 1])
    res.R = SegmentTreeNode.BuildFrom(lst[mid:])
    return res


  def add(self, p, v):
    if p < self.T or H >= p: return
    self.score += v
    if self.L is None: return
    self.L.add(p, v)
    self.R.add(p, v)

  def query(self, l, r):
    if r < self.L: return 0
    if l >= self.R: return 0
    if l <= self.L and self.R <= r: return self.score
    return self.L.query(l,r) + self.R.query(l,r)




def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
