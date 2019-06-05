#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
from chdrft.utils.cache import Cachable
import glog
import numpy as np
import yaml
from chdrft.utils.swig import swig
import itertools
from collections import defaultdict
import math
from chdrft.utils.fmt import Format
import networkx as nx
from collections import deque
import chdrft.utils.misc as cmisc
from scipy.stats.mstats import mquantiles

global flags, cache
flags = None
cache = None


def norm_array(x):
  return np.array(x, dtype=int)


def zero_if_neg(tb, pos):
  if pos < 0:
    return 0
  return tb[pos]


class ConvEnc:

  def __init__(self, mx, my=None):
    #mx size k x n x mem_x
    # my size n x mem_y
    self.mx = norm_array(mx)
    self.n = self.mx.shape[1]
    self.k = self.mx.shape[0]
    if my is None:
      my = [[]] * self.n
    self.my = norm_array(my)
    self.reset()

  def reset(self):
    self.il, self.ol = self.zero_state()

  def zero_state(self):
    ol = np.zeros((self.n, self.my.shape[1]), dtype=int)
    il = np.zeros((self.k, self.mx.shape[2]), dtype=int)
    return il, ol

  def state_nbits(self):
    return np.product(self.zero_state()[0].shape) + np.product(self.zero_state()[1].shape)

  def get_cost(self, target, obs):
    res = 1.
    for i, ob in enumerate(obs):
      if isinstance(ob, tuple):
        assert len(obs) == 2

        if ob[0] == target[i]: res *= ob[1]
        else: res *= 1 - ob[1]
      else:
        res *= 1 - abs(target[i] - ob)
    return res

  def next_raw_with_cost(self, state, tb, obs):
    cur, il, ol = self.next_raw(state, tb)
    return (il, ol), self.get_cost(cur, obs)

  def next_raw(self, state, tb):
    il, ol = state
    tb = np.reshape(tb, (self.k,))
    tb = norm_array(tb)
    il = np.roll(il, 1, axis=1)
    assert len(tb) == il.shape[0], f'{len(tb)} {il.shape} {ol.shape} {self.mx.shape} {self.my.shape} n={n}, k={k}'
    il[:, 0] = tb
    cur = np.zeros([self.n], dtype=int)
    for i in range(self.k):
      cur ^= self.mx[i] @ il[i] & 1

    if self.my.shape[1]:
      for i in range(self.n):
        cur[i] ^= np.dot(self.my[i], ol[i]) & 1

    if ol.shape[1] != 0:
      ol = np.roll(ol, 1, axis=1)
      ol[:, 0] = cur
    return cur, il, ol

  def next(self, tb):
    cur, self.il, self.ol = self.next_raw((self.il, self.ol), tb)
    return cur

  def get_response(self, inv):
    res = []
    for x in inv:
      res.append(self.next(x))
    return res

  def get_step_response(self, nx):
    inv = ConvEnc.StepInput(self.k, nx)
    return inv, self.get_response(inv)

  def get_response(self, inv):
    res = []
    self.reset()

    for x in np.reshape(inv, (-1, self.k)):
      res.append(self.next(x))
    return np.ravel(norm_array(res))

  @staticmethod
  def StepInput(k, npos):
    res = np.zeros((
        npos * k,
        k,
    ), dtype=int)
    for i in range(k):
      res[npos * i, i] = 1
    return res

  @staticmethod
  def Reverse(itrace, otrace, nx, ny, zero_prefix=True):
    n = otrace.shape[1]
    k = itrace.shape[1]

    start_pos = 0
    if not zero_prefix:
      start_pos = max(nx, ny + 1)

    npos = (itrace.shape[0] - start_pos)
    nr = npos * n
    nc = k * n * nx + n * ny

    m = np.zeros((nr, nc))
    v = np.zeros(nr, dtype=int)
    for pos in range(npos):
      # which round we consider
      for nn in range(n):
        # which output bit
        rp = pos * n + nn
        v[rp] = otrace[start_pos + pos, nn]

        for kp in range(k):
          for u in range(nx):
            m[rp, nx * (k * nn + kp) + u] = zero_if_neg(itrace[:, kp], start_pos + pos - u)
        for u in range(ny):
          m[rp, nx * k * n + nn * ny + u] = zero_if_neg(otrace[:, nn], start_pos + pos - 1 - u)
    opa_math = swig.opa_math_common_swig
    m2 = opa_math.Matrix_u32(opa_math.cvar.GF2, nr, nc)
    for i in range(nr):
      for j in range(nc):
        m2.set(i, j, int(m[i][j]))
    v = opa_math.v_u32(v.tolist())
    val = m2.solve(v)
    if len(val) == 0:
      return None

    mx = np.zeros((k, n, nx))
    my = np.zeros((n, ny))
    for ki, ni, nxi in itertools.product(range(k), range(n), range(nx)):
      mx[ki, ni, nxi] = val[nx * (k * ni + ki) + nxi]
    for ni, nyi in itertools.product(range(n), range(ny)):
      my[ni, nyi] = val[n * k * nx + ny * ni + nyi]
    return ConvEnc(mx, my)

  @staticmethod
  def FromTrace(itrace, otrace, concat_n=False, param_space=None, zero_prefix=True):
    for nx, ny in param_space:
      res = ConvEnc.Reverse(itrace, otrace, nx, ny, zero_prefix=zero_prefix)
      if res is not None:
        return res
    return None


class State:

  def __init__(self, state, score=0):
    self.state = state
    self.parent = None
    self.score = score
    self.bscore = 0
    self.action = None
    if score != 0:
      self.likelyhood = np.log2(score)
    else:
      self.likelyhood = None

  def add_pre(self, prev_state, trans_score, action):
    self.score = max(self.score, prev_state.score * trans_score)

    lh = prev_state.likelyhood + np.log2(max(1e-9, trans_score))
    if self.likelyhood is None or lh > self.likelyhood:
      self.likelyhood = lh

    if self.parent is None or self.parent.score < prev_state.score:
      self.action = action
      self.parent = prev_state

  @property
  def key(self):
    v = 0
    for st in self.state:
      if st.size == 0: continue
      for i in np.nditer(st):
        v = (v << 1) | i
    self.__dict__['key'] = v
    return v


class ConvTimeState:

  def __init__(self, encoder):
    self.encoder = encoder
    self.states = dict()

  def find_best(self):
    best = None
    for x in self.states.values():
      if best is None or best.bscore < x.bscore:
        best = x
    return best

  def get_state(self, state):
    if not isinstance(state, State): state = State(state)
    key = state.key
    if not key in self.states:
      self.states[key] = state
    return self.states[key]

  def one_transition(self, old_state, nstate, transition_score, action):
    self.get_state(nstate).add_pre(old_state, transition_score, action)

  def transition(self, nodes, next_store, obs):
    assert len(nodes) > 0
    for k in nodes:
      v = self.states[k]
      for kp in range(2**self.encoder.k):
        k_list = [(kp >> i & 1) for i in range(self.encoder.k)]
        nstate, transition_score = self.encoder.next_raw_with_cost(v.state, k_list, obs)
        #print(v.state, nstate, transition_score, k_list, obs)
        next_store.one_transition(v, nstate, transition_score, k_list)

  def finalize(self):
    tot = 0
    vals = []
    for x in self.states.values():
      vals.append(x.score)
    assert len(vals) > 0
    #assert tot >= 0.1
    limv = mquantiles(vals, 0.90)[0] / 3
    #limv = 1e-9
    tot = np.sum(vals)
    if tot < 1e-9: return False

    nstates = dict()
    for k, v in self.states.items():
      if v.score < limv: continue
      v.score /= tot
      nstates[k] = v
    assert len(nstates) > 0, vals
    self.states = nstates
    return True

  def backprogate(self, use_score):
    for x in self.states.values():
      score = x.bscore
      if use_score:
        score = x.score
      x.parent.bscore = max(x.parent.bscore, score)


def do_graph_reduce(g, reduce_func, vals, reverse=False):
  order = list(nx.algorithms.dag.topological_sort(g))
  if reverse: order = reversed(order)

  for n in order:
    assert n in vals, f'{n} {g.in_degree(n)} {g.out_degree(n)}'
    nxt = g.predecessors(n) if reverse else g.successors(n)
    for a in nxt:
      vals[a] = reduce_func(vals[a], vals[n])


class ConvViterbi:

  def __init__(self, encoder, collapse_depth=None, ans_states=None):
    self.encoder = encoder
    self.states = defaultdict(lambda: ConvTimeState(self.encoder))
    self.ans_states = ans_states
    self.pos = 0
    self.fail = 0

    self.collapse_depth = collapse_depth
    assert collapse_depth > 0

    self.output = {}
    self.reset()

  def reset(self):
    self.g = nx.DiGraph()
    self.nodes_at_depth = defaultdict(set)

  def backpropagate(self, target_pos):
    for pos in range(self.pos, target_pos, -1):
      cur_state = self.states[pos]
      cur_state.backprogate(pos == self.pos)
    res = self.states[target_pos].find_best()
    del self.states[target_pos]
    return res

  @staticmethod
  def GuessDataSource(data, encoder, fix_align=None, **kwargs):
    decs = []
    for align in range(encoder.n):
      if fix_align is None or fix_align == align:
        nd = data[align:]
        conf = cmisc.Attr(align=align, data=nd)
        conf.dec = ConvViterbi(encoder, **kwargs)
        decs.append(conf)

    if len(decs) == 1:
      return decs[0]

    for i in range(0, len(data), encoder.n):
      scores = []
      for conf in decs:
        grp = conf.data[i:i + encoder.n]
        if len(grp) != encoder.n: continue
        conf.dec.setup_next(grp)
        if conf.dec.fail: continue
        scores.append((conf.dec.get_max_likelyhood(),conf))
      scores.sort(reverse=True, key=lambda x: x[0])
      glog.info('Got scores %s'%scores)
      if len(scores) == 0: break
      if len(scores) == 1 or scores[0][0] > scores[1][0] + 30: return scores[0][1]
    return None

  @staticmethod
  def Solve(data, encoder, **kwargs):
    conf = ConvViterbi.GuessDataSource(data, encoder, **kwargs)
    glog.info(f'Selecting conf {conf}')

    dec = ConvViterbi(encoder, **kwargs)
    for i in range(0, len(conf.data) - encoder.n + 1, encoder.n):
      r = dec.setup_next(conf.data[i:i + encoder.n])
      if r is not None and r.action is not None:
        for a in r.action:
          yield a


  def feed(self, data):
    res = []
    for i in range(0, len(data)-self.encoder.n+1, self.encoder.n):
      r = self.setup_next(data[i:i+self.encoder.n])
      if r is not None and r.action is not None:
        res.extend(r.action)
    return res


  def get_max_likelyhood(self):
    return max(self.states[self.pos].states.values(), key=lambda x: x.likelyhood).likelyhood

  def set_equiprobable(self):
    # Generate all initial states of the conv code with equal prob
    for i in range(self.pos - self.collapse_depth, self.pos):
      self.do_collapse(i, self.pos-1)
    self.reset()
    nb = self.encoder.state_nbits()
    s0 = State(self.encoder.zero_state(), score=1)

    for cnd in itertools.product((0, 1), repeat=nb):

      n = np.product(s0.state[0].shape)
      ax = np.reshape(np.array(cnd[:n]), s0.state[0].shape)
      ay = np.reshape(np.array(cnd[n:]), s0.state[1].shape)
      s = State([ax, ay], 1 / (2**nb))
      self.states[self.pos].get_state(s)
    for k, v in self.states[self.pos].states.items():
      self.g.add_node((self.pos, k))
      self.nodes_at_depth[self.pos].add(k)

  def setup_next(self, obs):
    assert len(obs) == self.encoder.n
    if len(self.nodes_at_depth[self.pos]) == 0:
      glog.info('Fail, reset equi at %s', self.pos)
      self.set_equiprobable()

    self.pos += 1
    nstate = self.states[self.pos]
    pp = self.pos - 1
    p = self.pos
    cur = self.states[pp]
    cur.transition(self.nodes_at_depth[pp], nstate, obs)

    if self.ans_states:
      k0 = self.ans_states[p]
      print('ANSSS ', nstate.states[k0].score)

    if not nstate.finalize():
      self.fail = 1
      return None
    for k, v in nstate.states.items():
      self.nodes_at_depth[p].add(k)
      self.g.add_node((p, k))
      assert (p, k) in self.g
      assert (pp, v.parent.key) in self.g
      self.g.add_edge((pp, v.parent.key), (p, k))

    assert len(self.nodes_at_depth[p]) > 0
    for k, v in list(self.states[pp].states.items()):
      if self.g.out_degree((pp, k)) == 0:
        self.remove(pp, k)

    collapse_at = self.pos - self.collapse_depth
    if collapse_at < 0: return
    self.do_collapse(collapse_at, p)
    assert collapse_at in self.output
    return self.output[collapse_at]

  def do_collapse(self, collapse_at, p):
    if collapse_at < 0: return
    rem_at = self.nodes_at_depth[collapse_at]
    tmp = []
    if len(rem_at) > 1:
      scores = defaultdict(lambda: 0)
      for x in self.nodes_at_depth[p]:
        s = self.states[p].states[x]
        assert (p, x) in self.g
        scores[(p, x)] = s.score
      # backpropagating scores
      do_graph_reduce(self.g, max, scores, reverse=True)

    for k in list(rem_at):
      tmp.append((self.states[collapse_at].states[k].score, k))
    tmp.sort()
    for _, k in tmp:
      self.remove(collapse_at, k, collapse_remove=1)


  def remove(self, depth, k, collapse_remove=0):
    cd = self.nodes_at_depth[depth]
    if k not in cd:
      assert (depth, k) not in self.g
      return
    if len(cd) == 1:
      x = self.states[depth].states[k]
      self.output[depth] = x
    cur = (depth, k)

    preds = list(self.g.predecessors(cur))
    to_rem = []
    for cx in preds:
      self.g.remove_edge(cx, cur)
      if self.g.out_degree(cx) == 0:
        to_rem.append(cx)

    succs = list(self.g.successors(cur))
    for cx in succs:
      self.g.remove_edge(cur, cx)
      if self.g.in_degree(cx) == 0 and (not collapse_remove or self.g.out_degree(cx) == 0):
        to_rem.append(cx)

    self.g.remove_node(cur)
    cd.remove(k)
    for p, pk in to_rem:
      self.remove(p, pk)


def args(parser):
  clist = CmdsList().add(test).add(test2)
  ActionHandler.Prepare(parser, clist.lst)


def test(ctx):
  data = Attributize.FromYaml(open('./decode.data.out', 'r').read())
  dx = data.tb[4].raw
  print(Format(data.tb[4].ans).bitlist().bin2byte(lsb=True).v)
  dx = dx[:100]
  print(dx)
  dx = Format(dx).bitlist().bucket(2).v

  mx = [[[1, 0, 1, 1], [1, 1, 1, 1]]]
  x = ConvEnc(mx)
  print(x.get_response([[0], [0], [1], [0], [1], [0], [0]]))
  iobs, obs = x.get_step_response(14)

  obs[0][1] = 0.0
  obs[0][0] = 0.0
  obs[1][1] = 0.0
  print(obs)
  print(iobs)

  dec = ConvViterbi(x)
  for o in dx:
    dec.setup_next(o)

  tb = []
  for i in range(len(dx)):
    res = dec.backpropagate(i + 1)
    tb.append(res.action[0])

  print(Format(tb).bin2byte(lsb=0).v)

  #x = ConvEnc.FromSteps(data.step_response, n=2, k=1, concat_n=True, param_space=[(3, 0)])
  print(x)


def test2(ctx):

  mx = norm_array([[[1, 1], [1, 0]]])
  my = norm_array([[0, 0, 0], [1, 0, 1]])
  enc = ConvEnc(mx, my)
  print(enc.next([1]))
  print(enc.next([0]))
  print(enc.next([0]))
  print(enc.next([0]))
  print(enc.next([0]))
  print(enc.next([0]))
  print(enc.next([0]))
  print(enc.next([0]))
  print(enc.next([0]))
  iresponse, response = enc.get_step_response(8)

  print('STEPS >> ', iresponse, response)

  res = ConvEnc.FromTrace(iresponse, response, param_space=[(mx.shape[2], my.shape[1])])
  print(res)
  print(res.get_response(iresponse))
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
