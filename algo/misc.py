#!/usr//bin/env python

class LRUFifo:
  def __init__(self):
    self.clear()

  def touch(self, x):
    self.id += 1
    self.mp[x] = self.id
  def clear(self):
    self.mp = {}
    self.id = 0

  def get_all(self):
    tb = [(v,k) for k,v in self.mp.items()]
    tb.sort()
    for _,k in tb: yield k
