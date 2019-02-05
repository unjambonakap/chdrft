from chdrft.utils.misc import DictWithDefault, Attributize
import chdrft.utils.misc as cmisc
from curses.ascii import isspace, isalpha, isprint, isdigit, isgraph
import re
from collections import defaultdict
import traceback as tb
import curses.ascii
from chdrft.utils.fmt import Format
import pandas as pd
import math
import numpy as np
import pprint


def normalize_probs(p):
  if not p:
    return []
  sp = sum(p)
  return np.array(p) / sp


def isgoodchar(x):
  return isspace(x) or isgraph(x)


def is_printable(c):
  return curses.ascii.isgraph(c) or c == ord(' ')

def str_is_printable(s):
  cnt = 0
  for c in s:
    if is_printable(c): cnt += 1
  return cnt >= len(s) * 3 / 4


def ngram_data():
  data = pd.read_csv(cmisc.path_here('./ngrams.csv'))
  res = {}
  for n in set(data['n'].values):
    res[n] = data[data['n'] == n]
  return res


def make_printable(s):
  res = ''
  for i in s:
    if is_printable(i):
      res += chr(i)
    else:
      res += '?'
  return res


def get_incidence_freq(s):
  x = defaultdict(lambda: 0)
  for a in s:
    x[a] += 1
  return x


def get_sorted_incidence_freq(s, norm=0):
  if norm:
    n = len(s)
    res = list([(v / n, k) for k, v in get_incidence_freq(s).items()])
  else:
    res = list([(v, k) for k, v in get_incidence_freq(s).items()])
  res.sort()
  res.reverse()
  return res


def compute_ngrams(s, n):
  e = cmisc.defaultdict(lambda: 0)
  nx = len(s) - n + 1
  for i in range(len(s) - n + 1):
    e[s[i:i + n]] += 1 / nx
  return cmisc.asq_query(e.items()).order_by_descending(lambda x: x[1])


class Scoring:

  def __init__(self):
    self.freq_en = {}
    for x in freq_en_raw.splitlines():
      a, b = x.split('=')
      self.freq_en[a] = float(b) / 100.

    freq_sorted = sorted(self.freq_en.values())
    freq_sorted.reverse()

    self.freq_sorted = cmisc.defaultdict(lambda: 0)
    for i, v in enumerate(freq_sorted):
      self.freq_sorted[i] = v
    self.expected_random_score = 0
    for v in self.freq_en.values():
      self.expected_random_score += v * v

  def compute_score_en(self, s):
    x = DictWithDefault(lambda: 0)
    nspace = 0
    nalpha = 0
    nbad = 0
    n = len(s)
    for c in s:
      if isspace(c) or c == '_':
        nspace += 1
      elif isalpha(c):
        nalpha += 1
        x[c | 32] += 1
      elif not isgoodchar(c):
        nbad += 1

    tb = sorted(x.values())
    tb.reverse()
    score = 0
    if nalpha == 0:
      score += 1000
    else:
      for c, v in self.freq_en.items():
        score += (v - x[c] / nalpha)**2
    #score += (nspace / n - self.freq_sorted[0])**2
    score += (300 * nbad / n)**2
    return Attributize(score=score, nbad=nbad)

  def compute_score_freq(self, s):
    tb = get_sorted_incidence_freq(s, norm=1)
    score = 0
    for i, (v, _) in enumerate(tb):
      score += (self.freq_sorted[i] - v)**2
    return score

  def compute_score(self, s):
    return self.compute_score_en(s)

  def compute_score_random_eq(self, x):
    ix = get_sorted_incidence_freq(x)
    random_score = 0
    for v, _ in ix:
      random_score += v * (v - 1)
    random_score /= len(x) * (len(x) - 1)
    return random_score


def xorpad(a, b):
  assert len(a) >= len(b)
  bl = len(b)
  return type(a)([a[i] ^ b[i % bl] for i in range(len(a))])


def xor(a, b):
  if isinstance(b, int):
    return [a[i] ^ b for i in range(len(a))]
  assert len(a) <= len(b)
  return [a[i] ^ b[i] for i in range(len(a))]


class XorpadSolver:

  def __init__(self, data, pad=None):
    self.data = [bytearray(x) for x in data]
    self.n = len(data)
    self.maxl = max([len(x) for x in self.data])
    self.pad = bytearray(self.maxl)
    self.scoring = Scoring()
    self.left_margin = 6

    if pad is None:
      self.guess_best()
    else:
      self.pad = pad

  def guess_best(self):
    for i in range(self.maxl):
      sx = bytearray()
      for x in self.data:
        if len(x) <= i:
          continue
        sx.append(x[i])

      tb = []
      for k in range(256):
        tmp = xor(sx, k)
        tb.append((self.scoring.compute_score(tmp), k))
      tb.sort(key=lambda x: x[0].score)
      self.pad[i] = tb[0][1]

  def disp_cols_helper(self):
    n = 1
    while n < self.maxl:
      sx = ' ' * self.left_margin
      for j in range(self.maxl // n):
        sx += '{val:<{width}d}'.format(width=n, val=j % 10)
      print(sx)
      n *= 10

  def display(self):
    for i in range(len(self.data)):
      if i % 10 == 0:
        self.disp_cols_helper()
      u = xorpad(self.data[i], self.pad)
      sanitized = bytearray()
      for c in u:
        if not isprint(c):
          sanitized.append(ord('?'))
        else:
          sanitized.append(c)
      print(
          '{row:{width}d}{space:{space_width}}{content}'.format(
              width=self.left_margin - 2,
              space_width=2,
              space='',
              row=i,
              content=sanitized.decode()
          )
      )

  def solve(self):
    while True:
      try:
        self.display()
        tmp = input('row col newchar? ')
        print(self.pad)
        tmp.rstrip()
        m = re.match('(?P<row>[0-9]+)\W(?P<col>[0-9]+)\W(?P<nc>.+)', tmp)
        row = int(m.group('row'))
        col = int(m.group('col'))
        nc = m.group('nc')

        for i in range(len(nc)):
          nv = ord(nc[i]) ^ self.data[row][col + i]
          self.pad[(col + i) % len(self.pad)] = nv
      except KeyboardInterrupt:
        raise
      except Exception as e:
        print(e)
        pass
    pass


freq_en_raw = """E=12.02
T=9.10
A=8.12
O=7.68
I=7.31
N=6.95
S=6.28
R=6.02
H=5.92
D=4.32
L=3.98
U=2.88
C=2.71
M=2.61
F=2.30
Y=2.11
W=2.09
G=2.03
P=1.82
B=1.49
V=1.11
K=0.69
X=0.17
Q=0.11
J=0.10
Z=0.07"""


class Mt19937:

  tamp_inv = [
      270681289, 2, 2165448768, 263304, 16, 67670322, 64, 274877641, 8392962, 2165317636, 33571848,
      67405969, 201953586, 8196, 807814344, 1074299152, 2148598304, 2165449284, 270943305, 67666194,
      2298684000, 270926024, 4196369, 76054832, 2165318212, 308415681, 75534610, 2299600932,
      302138440, 604014609, 1275170866, 2148540932
  ]
  MATRIX_A = 0x9908b0df
  N = 624
  M = 397
  MATRIX_A = 0x9908b0df
  UPPER_MASK = 0x80000000
  LOWER_MASK = 0x7fffffff
  MAG01 = [0, MATRIX_A]

  def __init__(self):
    self.mt = [0] * 624
    self.index = 0

  def do_tamp(x):
    x ^= x >> 11
    x ^= (x << 7) & 0x9d2c5680
    x ^= (x << 15) & 0xefc60000
    x ^= x >> 18
    return x

  def do_itamp(x):
    res = 0
    for i in range(32):
      if x >> i & 1:
        res ^= Mt19937.tamp_inv[i]
    return res

  def refresh(self, i):
    y = (self.mt[i] & self.UPPER_MASK) | (self.mt[(i + 1) % self.N] & self.LOWER_MASK)
    self.mt[i] = self.mt[(i + self.M) % self.N] ^ y >> 1
    self.mt[i] ^= self.MAG01[y & 1]

  def next(self):

    if self.index >= 624:
      for i in range(self.N):
        self.refresh(i)

      self.index = 0

    y = self.mt[self.index]
    self.index += 1
    return Mt19937.do_tamp(y)

  def reset(self, tb, need_itamp=False):
    assert len(tb) == self.N

    if need_itamp:
      self.mt = [Mt19937.do_itamp(x) for x in tb]
    else:
      self.mt = list(tb)

    self.index = self.N

  def getstate(self):
    return (3, tuple(self.mt + [self.index]), None)


class VigenereSolver:

  def __init__(self, data):
    self.data = data
    self.scoring = Scoring()

  def get_group(self, i, m):
    return list([self.data[j] for j in range(i, len(self.data), m)])

  def get_n_guess_score(self, n):
    vals = []
    for i in range(n):
      cur = self.get_group(i, n)
      vals.append(self.scoring.compute_score_freq(cur))
    return vals

  def solve_single_pad(self, x, only_good_key=0):
    scores = []
    for i in range(256):
      if not only_good_key or isgoodchar(i):
        scores.append((self.scoring.compute_score(xor(x, i)), i))
    return scores

  def sub(self, mp):
    res = ''
    for i in self.data:
      res += mp.get(i, '.')
    return res


def tsf(s, x, default=0):
  res = bytearray()
  for c in s:
    nc = x.get(c, default)
    if isinstance(nc, str):
      nc = nc.encode()[0]
    res.append(nc)
  return res


class SubSolver:

  def __init__(self, data, tsf=None):
    self._data = data
    self.data = Format(data).bucket(160).v
    self.n = len(data)
    self.maxl = max([len(x) for x in self.data])
    self.scoring = Scoring()
    self.left_margin = 6
    if tsf is None: tsf = cmisc.defaultdict(lambda: 0)
    self.tsf = tsf

  def disp_cols_helper(self):
    n = 1
    tb = []
    while n < self.maxl:
      sx = ' ' * self.left_margin
      for j in range(self.maxl // n):
        sx += '{val:<{width}d}'.format(width=n, val=j % 10)
      tb.append(sx)
      n *= 10
    print('\n'.join(reversed(tb)))

  def display(self):
    for i in range(len(self.data)):
      if i % 10 == 0:
        self.disp_cols_helper()
      u = tsf(self.data[i], self.tsf)
      sanitized = make_printable(u)
      print(
          '{row:{width}d}{space:{space_width}}{content}'.format(
              width=self.left_margin - 2, space_width=2, space='', row=i, content=sanitized
          )
      )

  def itsf(self):
    return {chr(v): k for k, v in self.tsf.items()}

  def update(self, row, col, nc):
    for i in range(len(nc)):
      self.tsf[self.data[row][col + i]] = ord(nc[i])

  def solve(self):
    while True:
      try:
        self.display()
        tmp = input('row col newchar? ')
        tmp.rstrip()
        m = re.match('(?P<row>[0-9]+)\W(?P<col>[0-9]+)\W(?P<nc>.+)', tmp)
        row = int(m.group('row'))
        col = int(m.group('col'))
        nc = m.group('nc')

        for i in range(len(nc)):
          self.tsf[self.data[row][col + i]] = ord(nc[i])
      except KeyboardInterrupt:
        raise
      except Exception as e:
        tb.print_exc()
        print(e)
        pass
    pass

  def compute_score(self, baseline, data, l):
    maxn = baseline['n'].max()

  def guess(self):
    ng_baseline = ngram_data()
    ng_data = {}
    for n in range(1, 5):
      ng_data[n] = compute_ngrams(self._data, n).take(100).to_list()
    proced = set()

    self.tsf[space] = space_chr
    proced.add(space_chr)

    while True:
      scores = []
      for l in letters:
        if l in proced: continue
        score = self.compute_score(ng_baseline, ng_data, l)
        scores.append((score, l))


class Solver:

  def __init__(self, data):
    self.data = data
    ng_data = {}
    for n in range(1, 7):
      ng_data[n] = compute_ngrams(self.data, n).take(100).to_list()
    self.ng_data = ng_data
    self.ng_baseline = ngram_data()
    self.mp = {}
    self.cols = (
        dict(name='col_all', start=0, end=0),
        dict(name='col_minus1', start=0, end=1),
        dict(name='col_1', start=1, end=0),
        dict(name='word', start=1, end=1),
    )
    self.letters = list(range(ord('A'), ord('Z') + 1))
    space = self.ng_data[1][0][0][0]
    self.space = space
    self.ng_with_space = self.compute_ng_with_space()

  def measure_err(self, a, b, n):
    x = 0
    for ax, bx in zip(a, b):
      x += ax**1.5 * (ax - bx)**2
    return x

  def compute_score(self, attr):
    rmp = {v: k for k, v in attr.items()}
    rem = [l for l in self.letters if l not in rmp]
    score = 0
    for n in self.ng_data.keys():
      n1 = self.ng_data[n]
      n2 = self.ng_baseline[n]

      cv = n2['col_all'].values
      nv = n2['ngram_col_all'].values
      defaultv = n2['default_col_all'].values[0]
      groups = cmisc.defaultdict(list)
      for cx, nx in zip(cv, nv):
        sx = ''
        for l in nx:
          if l in rmp:
            sx += l
          else:
            sx += '_'
        groups[sx].append(cx)

      tmpsum = 0
      for group, groupv in groups.items():

        def filter_actual(e):
          ngram, prob = e
          for l1, l2 in zip(group, ngram):
            if l1 != '_' and l1 != attr.get(l2, None): return 0
          return 1

        group_act = cmisc.asq_query(n1).where(filter_actual).select(lambda x: x[1]).to_list()
        nn = max(len(group_act), len(groupv))

        groupv = Format(groupv).pad(nn, defaultv, force_size=0).v
        group_act = Format(group_act).pad(
            nn, (defaultv if not group_act else group_act[-1] / 2), force_size=0
        ).v
        err = self.measure_err(groupv, group_act, n)
        score += err
        tmpsum += err
        #if len(group) == 1: print(group, tmpsum, rmp.get(group, -1), err, groupv[:2], group_act[:2])
      #print(n, tmpsum)
    return score

  def compute_next(self, attr, t):
    n_try = 10
    for s, p in self.ng_data[1]:
      s = s[0]
      if p < 0.05: break
      if s in attr: continue
      n_try -= 1
      nattr = dict(attr)
      nattr[s] = t
      yield (nattr, self.compute_score(nattr))
      if n_try == 0: break

  def solve_search(self):
    mp = self.solve_most_likely()

    states = [(mp, 0)]
    max_states = 1000

    for i in range(10):
      print(i, len(states))
      nstates = []
      for attr, _ in states:
        iattr = {v: k for k, v in attr.items()}
        n_try = 1
        for l in self.ng_baseline[1]['ngram_col_all'].values:
          if l in iattr: continue
          nstates.extend(self.compute_next(attr, l))
          n_try -= 1
          if n_try == 0: break

      nstates.sort(key=lambda x: x[1])
      nstates = nstates[:max_states]
      for i in range(len(nstates)):
        if nstates[i][1] < nstates[0][1] / 3:
          nstates = nstates[:i]
          break
      if len(nstates) == 0:
        return states
      states = nstates

  def compute_score1(self, p1, p2, p2_unnormed, p1_best, p2_best, n):
    p2_unnormed = max(p2_unnormed, 1e-8)
    err = abs(p1 - p2) / max(p1, p2)
    err *= -math.log10(p2_unnormed**2)
    err *= n**0.5
    err *= (p1_best / p1) ** 0.5 + (p2_best / p2) ** 0.5
    return err

  def solve_most_likely(self):
    done = set()
    mp = {}
    rmp = {}
    mp[space] = ' '

    states = [(mp, 0, None)]
    seen = set()
    for step in range(6):
      nstates = []
      for mp, _, _ in states:
        ncnds = self.get_next_cnds(mp, get_all=1)
        maybe = [self.cnd_to_state(mp, x) for x in ncnds]
        for cnd in maybe:
          h = hash(frozenset(cnd[0].items()))
          if h in seen: continue
          seen.add(h)
          nstates.append(cnd)

      nstates.sort(key=lambda x: x[1])
      #cmisc.ppx.pprint(nstates[:20])
      states = nstates[:100]
    return states

  def compute_ng_with_space(self):
    space = self.space
    res = cmisc.defaultdict(lambda: cmisc.defaultdict(list))
    for n in range(1, 6):
      tmp = cmisc.defaultdict(list)
      res[n] = tmp
      for col in self.cols:
        if col['start'] and col['end']:
          if n + 2 not in self.ng_data: continue
          have = cmisc.asq_query(
              self.ng_data[n + 2]
          ).where(lambda x: (space not in x[0][1:-1] and x[0][0] == space and x[0][-1] == space))
        elif col['start']:
          if n + 1 not in self.ng_data: continue
          have = cmisc.asq_query(self.ng_data[n + 1]
                                ).where(lambda x: (space not in x[0][1:] and x[0][0] == space))
        elif col['end']:
          if n + 1 not in self.ng_data: continue
          have = cmisc.asq_query(self.ng_data[n + 1]
                                ).where(lambda x: (space not in x[0][:-1] and x[0][-1] == space))
        else:
          have = cmisc.asq_query(self.ng_data[n]).where(lambda x: space not in x[0])

        if col['start']: have = have.select(lambda x: (x[0][1:], x[1]))
        if col['end']: have = have.select(lambda x: (x[0][:-1], x[1]))
        have = have.to_list()

        probs = [x[1] for x in have]
        probs = normalize_probs(probs)
        have = [cmisc.Attributize(s=x[0], p_norm=p, p_raw=x[1]) for x, p in zip(have, probs)]
        tmp[col['name']] = have
      cmisc.ppx.pprint(res[n])

    return res

  def get_next_cnds(self, mp, get_all=0):
    rmp = {v: k for k, v in mp.items()}
    space = self.space
    lst = []
    for n in self.ng_data.keys():
      if n == 1: continue
      for col in self.cols:
        target_str_list = self.ng_baseline[n]['ngram_' + col['name']].values[:7]
        target_prob_list = self.ng_baseline[n][col['name']].values[:7]
        for (target_str, target_prob) in zip(target_str_list, target_prob_list):
          cur_coldata = self.ng_with_space[n][col['name']]
          for cx in cur_coldata:

            stats,_,_ = self.compute_nxt_mp_stats(mp, rmp, target_str, cx.s)
            if stats.conflict or not stats.new: continue

            lst.append(
                cmisc.Attributize(
                    score=self.compute_score1(
                        target_prob,
                        cx.p_norm,
                        cx.p_raw,
                        target_prob_list[0],
                        cur_coldata[0].p_norm,
                        n,
                    ),
                    target_str=target_str,
                    have_str=cx.s,
                    target_prob=target_prob,
                    have_prob=cx.p_raw,
                    have_norm_prob=cx.p_norm,
                    col_name=col['name'],
                )
            )

    lst.sort(key=lambda x: x.score)
    if get_all: return lst

    if not lst: return None
    lbd = lst[0].score * 2
    probs = [math.exp(x.score / lbd) for x in lst]
    return self.compute_next_mp(mp, random.choices(lst, weights=probs))

  def cnd_to_state(self, mp, cnd):
    nmp = self.compute_next_mp(mp, cnd)
    return nmp, self.compute_mp_score(nmp), cnd

  def compute_next_mp(self, mp, cnd):
    nmp = dict(mp)
    for target_c, have_c in zip(cnd.target_str, cnd.have_str):
      nmp[have_c] = target_c
    return nmp

  def compute_nxt_mp_stats(self, mp, rmp, target_str, have_str):
    res = cmisc.Attributize(conflict=0, new=0)
    nmp = dict(mp)
    nrmp = dict(rmp)
    for target_c, have_c in zip(target_str, have_str):
      if have_c in nmp and nmp[have_c] != target_c: res.conflict = 1
      if target_c in nrmp and nrmp[target_c] != have_c: res.conflict = 1
      if res.conflict: break
      if have_c not in nmp:
        nmp[have_c] = target_c
        nrmp[target_c] = have_c
        res.new = 1
    return res, nmp, nrmp

  def norm_mp(self, mp):
    res = {}
    for k, v in mp.items():
      if isinstance(k, str): res[ord(k)] = v
      else:
        res[k] = v
    return res

  def s2o(self, rmp, s):
    res = []
    for c in s:
      if not c in rmp: return None
      res.append(rmp[c])
    return bytes(res)

  def compute_mp_score(self, mp):
    score = 0
    rmp = {v: k for k, v in mp.items()}
    for n in self.ng_baseline.keys():
      for col in self.cols:
        target_str_list = self.ng_baseline[n]['ngram_' + col['name']].values[:7]
        target_prob_list = self.ng_baseline[n][col['name']].values[:7]
        for (target_str, target_prob) in zip(target_str_list, target_prob_list):
          o_str = self.s2o(rmp, target_str)
          if not o_str: continue
          cur_coldata = self.ng_with_space[n][col['name']]
          for cx in cur_coldata:
            assert type(cx.s) == type(o_str)
            if cx.s != o_str: continue
            cscore = self.compute_score1(
                target_prob,
                cx.p_norm,
                cx.p_raw,
                target_prob_list[0],
                cur_coldata[0].p_norm,
                n,
            )
            #print(mp, cscore, target_str, cx.s, col['name'], cx, target_prob, target_prob_list[0], cur_coldata[0].p_norm)
            score += cscore
            break
    return score

  def solve_bruteforce(self):
    space = self.ng_data[1][0][0][0]
    mp = {}
    rmp = {}
    mp[space] = ' '
    rmp[' '] = space
    self.space = space
    self.ng_with_space = self.compute_ng_with_space()




    target_list = []
    map_key_to_src = cmisc.defaultdict(list)

    for n in self.ng_baseline.keys():
      tmp = []
      for col in self.cols:
        want = 7 if n == 1 else 4
        key = (n, col['name'])

        for i in range(want):
          target_str= self.ng_baseline[n]['ngram_' + col['name']].values[i]
          target_prob= self.ng_baseline[n][col['name']].values[i]
          target_global_prob = self.ng_baseline[n]['p_global_'+col['name']].values[i]
          score = target_prob ** 3 * target_global_prob
          tmp.append((score, target_str, key))

        cur_coldata = self.ng_with_space[n][col['name']]
        map_want = min(int(want * 1.5), len(cur_coldata))
        for i in range(map_want):
          cx = cur_coldata[i]
          map_key_to_src[key].append(cx)
      tmp.sort(key=lambda x: x[0], reverse=1)
      target_list.extend(tmp[:8])

    target_list.sort(key=lambda x: x[0], reverse=1)
    curated = []
    known = set()
    for target in target_list:
      new = set(target[1])
      if new <= known: continue
      known.update(new)

      curated.append(target)

    cmisc.ppx.pprint(curated[:30])
    cmisc.ppx.pprint(target_list[:30])


    res = []
    def rec(pos, mp, rmp):
      if pos == 7:
        res.append((mp, self.compute_score_mp2(mp)))
        return
      e_, s, key = curated[pos]
      for cx in map_key_to_src[key]:
        stats, nmp, nrmp = self.compute_nxt_mp_stats(mp, rmp, s, cx.s)
        if stats.conflict: continue
        rec(pos+1, nmp, nrmp)
      rec(pos+1, mp, rmp)

    rec(0, mp, rmp)
    res.sort(key=lambda x: x[1])
    return res[:10]

  def compute_score_mp2(self, mp):
    rmp = {v: k for k, v in mp.items()}
    score = 0
    for n in self.ng_data.keys():
      for col in self.cols:
        cur_coldata = self.ng_with_space[n][col['name']]
        for i in range(len(self.ng_baseline[n])):
          target_str= self.ng_baseline[n]['ngram_' + col['name']].values[i]
          target_prob= self.ng_baseline[n][col['name']].values[i]
          target_global_prob = self.ng_baseline[n]['p_global_'+col['name']].values[i]



          src_str = tsf_or_none(target_str, rmp)
          p = 0
          coeff = target_prob ** 0.5 * target_global_prob
          if src_str is not None:
            src_str = bytes(src_str)
            src = next(filter(lambda x: x.s == src_str, cur_coldata), None)
            if src is not None: p = src.p_norm
            #print(target_str, src_str, score, p, target_prob, coeff, col['name'])

          err = (p - target_prob) ** 2
          score += coeff * err
    return score



def tsf_or_none(v, mp):
  res = []
  for x in v:
    if x not in mp: return None
    res.append(mp[x])
  return res

