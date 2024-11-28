#!/usr/bin/env python

from chdrft.utils.cache import Cachable
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize

global flags, cache
flags = None
cache = None


def str_dist(a, b, case=True):
  if len(a) < len(b):
    a, b = b, a
  if not case:
    a = a.lower()
    b = b.lower()

  cost = list(range(len(b) + 1))
  cost[0] = 0
  for i in a:
    best = cost[0] + 2
    for j in range(len(b) + 1):
      best = min(best, cost[j] + 1)
      old = cost[j]
      cost[j] = best

      if j != len(b):
        c = 2
        if b[j] == i:
          c = 0

        best = min(best + 3, old + c)
  return cost[-1]


def find_closest(s, lst):
  tb = [(str_dist(x, s, case=False), x) for x in lst]
  tb.sort()
  best = tb[0][0]
  if best == tb[1][0] and best >= len(choice) // 2:
    print('could not decide between {} and {}'.format(tb[0][1], tb[1][1]))
    return None
  return tb[0][1]


class MatchingProba:

  class State:

    def __init__(self, npos, qpos, cont):
      self.npos = npos
      self.qpos = qpos
      self.cont = cont

    def __str__(self):
      return ','.join(map(str, [self.npos, self.qpos, self.cont]))

    def __hash__(self):
      return hash(str(self))

    def __eq__(self, a):
      return self.npos == a.npos and self.qpos == a.qpos and self.cont == a.cont

  def __init__(self, name, query):
    self.name = name
    self.query = query
    self.n = len(name)
    self.m = len(query)
    self.dp = {}

  def allow_match(self, a, b):
    return a.lower() == b.lower()

  def tsf(self, state, dn, dq):
    cost = self.go(self.State(state.npos - dn, state.qpos - dq, state.cont and dq == 1 and dn == 1))
    if dn == 1 and dq == 1:
      pass
    elif dn == 1:
      c = self.name[state.npos - 1]
      exponent = 0.7
      if c == '_' or c == '/': exponent = 0.01
      cost *= (1 - 1 / self.n)**exponent
    elif dq == 1:
      cost *= (1 - 1 / self.m)**1.5
    return cost

  def go(self, state):
    if state.qpos == 0:
      return 1.

    if state in self.dp:
      return self.dp[state]

    best = 0.
    if state.npos > 0 and state.qpos > 0:
      if self.allow_match(self.name[state.npos - 1], self.query[state.qpos - 1]):
        best = max(best, self.tsf(state, 1, 1))

    if state.npos > 0:
      best = max(best, self.tsf(state, 1, 0))
    if state.qpos > 0:
      best = max(best, self.tsf(state, 0, 1))

    self.dp[state] = best
    return best

  def solve(self):
    best = 0.
    for i in range(self.n + 1):
      best = max(best, self.go(self.State(i, self.m, 1)))
    return best


class FuzzyMatcher:

  def __init__(self):
    self.names = []
    self.debug = ''

  def add_name(self, name):
    self.names.append(name)
    self.names.sort()

  def get_score(self, name, x):
    return MatchingProba(name, x).solve()

  def reset(self):
    self.names = []

  @Cachable.cached2(cache_filename='/tmp/.chdrft.opa_string.cache')
  def find(self, x):
    return self.find_nocache(x)

  def find_nocache(self, x):
    if len(self.names) == 0:
      return None
    tb = []
    if x in self.names:
      return 1., x

    for a in self.names:
      tb.append((self.get_score(a, x), a))
    tb.sort()
    self.debug = tb
    #print('done')

    if tb[-1][0] > 1 - 1e-3 and tb[-1][0] - tb[-2][0] > 1e-3:
      return tb[-1]

    if tb[-1][0] < 0.5:
      return None
    if len(tb) >= 2:
      sc1 = tb[-1][0]
      sc2 = tb[-2][0]
      bad = (sc2 / sc1 > 0.95)
      if sc1 < 1 - 1e-3:
        diff = (sc1 - sc2) / (1 - sc1)
        if diff > 1:
          bad = False
      if bad:
        return None
    return tb[-1]


import jsonpickle


class FuzzyMatcherHandler(jsonpickle.handlers.BaseHandler):

  def flatten(self, obj, data):
    data['data'] = jsonpickle.encode(obj.names)
    return data

  def restore(self, obj):
    data['data'] = jsonpickle.encode(obj.names)
    res = FuzzyMatcher()
    res.names = jsonpickle.decode(obj['data'])
    return res


FuzzyMatcherHandler.handles(FuzzyMatcher)


def lcs(a, b):
  na = len(a)
  nb = len(b)
  if na > nb: return lcs(b, a)
  tb = [0] * (na + 1)
  for i in range(nb):
    for j in reversed(range(na)):
      if a[j] == b[i]:
        tb[j + 1] = max(tb[j] + 1, tb[j + 1])
  return max(tb)


def min_substr_diff(hay, needle, mod_req=1):
  best = (len(needle), None)
  for i in range(0, len(hay) - len(needle) + 1, mod_req):
    err = 0
    for j in range(len(needle)):
      err += needle[j] != hay[i + j]
    best = min(best, (err, i))
  return best


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)
  parser.add_argument('a', type=str)
  parser.add_argument('b', type=str)


def test(ctx):

  a = ctx.a
  b = ctx.b
  res = str_dist(a, b)
  print('Cost of transformation {} -> {}: {}'.format(a, b, res))

  mc = FuzzyMatcher()
  mc.add_name(a)
  print('FUZZY >> ', mc.find(b))


def test_lcs(ctx):
  print(lcs(ctx.a, ctx.b))


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
