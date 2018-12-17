#!/usr/bin/env python

import math
import numpy as np

class SimpleIIR:

  def __init__(self, alpha=0.9, minv=None, maxv=None):
    self.minv = minv
    self.maxv = maxv
    self.alpha = alpha
    self.v = math.nan

  def get_or(self, default):
    if self.has_value(): return self.v
    return default

  def has_value(self):
    return not np.any(np.isnan(self.v))

  def push(self, x):
    if self.minv is not None: x= max(x, self.minv)
    if self.maxv is not None: x= min(x, self.maxv)

    if self.has_value():
      self.v = np.polyadd(np.polymul(self.alpha, self.v), np.polymul((1 - self.alpha), x))
      if not isinstance(x, np.ndarray): self.v = self.v[0]
    else:
      self.v = x
    return self.v
