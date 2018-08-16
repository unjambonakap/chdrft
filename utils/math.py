#!/usr/bin/env python

import math
import gmpy2


class MathUtils:
  def __init__(self, n, mod=None, pw=1):
    self.n = n
    self.mod = mod
    self.fact = [1] * (n+1)
    self.ifact = [1] * (n+1)
    self.isp = set()
    self.pw = pw

    for i in range(1, n+1):
      self.fact[i] = self.fact[i-1] * gmpy2.powmod(i, self.pw, self.mod)
      if mod is not None:
        self.fact[i] %= mod

    if mod is None:
      self.ifact = None
    else:
      for i in range(1, n+1):
        self.ifact[i] = gmpy2.invert(self.fact[i], mod)

  def cnk(self, n, k):
    if n<k or k<0: return 0
    if self.mod is not None:
      return self.fact[n] * self.ifact[k] % self.mod * self.ifact[n-k]%self.mod
    return self.fact[n] // self.fact[k] // self.fact[n-k]


class PrimeUtils:
  def __init__(self, n):
    self.primes = [2]
    self.n = n
    self.prev = [1] * n
    self.prev[2] = 1
    self.prev[1] = -1

    for i in range(3, n, 2):
      if self.prev[i] != -1: continue
      self.primes.append(i)
      for j in range(i*i,  n, i):
        self.prev[j] = j // i
