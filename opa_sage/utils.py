#!/usr/bin/env python

try:
  from sage.all import *
  import gmpy2
except:
  pass

from chdrft.cmds import *
import ast
import argparse
import inspect
import subprocess as sp
import itertools
import binascii
import re
import chdrft.utils.cmdify as cmdify


def vec_to_poly(row, var):
  res = 0
  for j in range(len(row)):
    res = res + (var**j) * row[j]
  return res


def get_matrix_for_op(f, n):
  m = Matrix(GF(2), n)
  for i in range(n):
    x = f(1 << i)
    for j in range(n):
      m[j, i] = x >> j & 1
  return m


def mt19937_inv_tampering():
  F = GF(2)
  n = 32
  m = identity_matrix(F, n)

  f1 = lambda y: y ^ (y >> 11)
  f2 = lambda y: y ^ (y << 7) & 0x9d2c5680
  f3 = lambda y: y ^ (y << 15) & 0xefc60000
  f4 = lambda y: y ^ (y >> 18)
  m = get_matrix_for_op(f1, n) * m
  m = get_matrix_for_op(f2, n) * m
  m = get_matrix_for_op(f3, n) * m
  m = get_matrix_for_op(f4, n) * m
  m = m.inverse()
  return m


def disp_mt13997_inv():
  m = mt19937_inv_tampering()
  tb = []
  for i in range(32):
    x = 0
    for j in range(32):
      x |= int(m[j, i]) << j
    tb.append(x)
  print(tb)


def gcd_polyring(a, b):
  if a.degree() < b.degree():
    return gcd_polyring(b, a)

  N = int(parent(a).modulus())
  x = parent(a).gens()[0]

  while b.degree() != -1:
    la = int(a.leading_coefficient())
    lb = int(b.leading_coefficient())
    ilb = gmpy2.invert(lb, N)
    ilb = int(ilb)

    diff = a.degree() - b.degree()
    tmp = a - b * ilb * la * x**diff
    a = b
    b = tmp

  return a


def coppersmith_small_root(N, poly, sol=None):
  N = Integer(N)

  if sol:
    assert poly(10, sol) % N == 0
  x = poly.parent().gens()[0]
  print(poly.variable_name())

  t = 20
  D = 30
  NT = N**t
  print(poly.degree())
  print(int(N).bit_length())
  B = N.nth_root(poly.degree(), truncate_mode=1)[0]
  print('B>>> ', B)

  if sol:
    print(poly(0, sol) % N)

  fl = []
  for i, j in itertools.product(range(D), range(1, D / poly.degree())):
    if i + j * poly.degree() > D:
      continue
    fij = (x**i) * (poly**j) * (N**max(0, t + 1 - j))

    if sol:
      tmp = fij(10, sol)
      assert tmp % NT == 0
    fl.append(fij)

  M = Matrix(ZZ, len(fl), D)
  n, m = M.dimensions()
  for i in range(len(fl)):
    lst = fl[i].padded_list(m)
    for j in range(m):
      M[i, j] = int(lst[j]) * (B**j)

  print('start lll')
  res = M.LLL()
  print('done lll')
  best = None
  for i in range(n):
    for j in range(m):
      if res[i, j] != 0:
        best = res.row(i)
        break
    else:
      continue
    break
  else:
    print('fail')
    assert False
  #best = res.row(n - 1)
  for j in range(m):
    best[j] = best[j] / (B**j)

  vx = 0
  B = 1
  for i in range(len(best)):
    vx += abs(best[i]) * (B**i)

  x2 = ZZ['x'].gens()[0]
  best = vec_to_poly(best, x2)
  if sol:
    assert best(sol) % NT == 0
    print('sol >> ', sol)
    print(best(sol) % (NT))
  return [x[0] for x in best.roots()]


def coppersmith_small_diff(N, E, c1, c2, sol=None):

  x, y = ZZ['x', 'y'].gens()
  p1 = x**E - c1
  p2 = (x + y)**E - c2
  res = p1.resultant(p2)
  res = res.univariate_polynomial()

  roots = coppersmith_small_root(N, res, sol)
  print('N', N)
  print('E', E)
  print('c1', c1)
  print('c2', c2)
  print('sol', sol)
  #roots=[sol]
  for root in roots:

    x, = Zmod(N)['x'].gens()
    p1 = x**E - c1
    p2 = (x + root)**E - c2
    res = gcd_polyring(p1, p2)
    if res.degree() < 1:
      continue
    x0 = int(res[0])
    x1 = int(-res[1])
    x0 = x0 * gmpy2.invert(x1, N) % N
    return x0
  else:
    print('FAILED')


def coppersmith_small_root_entry(N, coeffs, X, beta):
  coeffs = ast.literal_eval(coeffs)
  x, = Zmod(N)['x'].gens()
  poly = x.parent(0)
  print(poly)
  for i in range(len(coeffs)):
    poly = poly + (x**i) * coeffs[i]
  res = poly.small_roots(X, beta)
  return res


base_dir = './misc/confidence_2015/rsa1_crypto_400'


def get_flag(ia):
  return int(open(base_dir + '/resources/out/flag%d' % ia).read())


def get_key():
  with open(base_dir + '/resources/out/n1') as f:
    return int(f.read()), 3


def test():
  n, e = get_key()

  va = get_flag(0)
  vb = get_flag(1)
  coppersmith_small_diff(n, e, va, vb)


def test2():
  set_random_seed(0)
  q = int(random_prime(2**512))
  p = int(random_prime(2**512))
  n = p * q
  e = 3

  m = int(random_prime(2**900))
  r = int(random_prime(2**32))
  print('start with >> ', hex(m))

  m1, m2 = m, m + r
  c1 = (m1**e) % n
  c2 = (m2**e) % n
  print(c1, c2)

  coppersmith_small_diff(n, e, c1, c2, r)


def test3():
  c = get_flag(0)
  n = get_key()
  e = 3

  x, = ZZ['x'].gens()


def test4():
  fx = open('/tmp/data.in', 'r')
  data = '\n'.join(fx.readlines())
  lst = ast.literal_eval(data)
  #coeffs='[1589798829309068671127749569798632386565877700471683468991083004691620730152974734863210648285568889648985744632563497019514127092876354219501380135120700323194288306197634607007097220387951235931508441020847970689409, -1535126813620037027529348839561346852040710364363025630735164323804062309920845854904897334050878, 818165036523831719374470562231161318768287340550870791753799779913837602782623599327465948499029]'
  #n=515005405643932092863509501092126359476792027268461708353931079049341634928657037702291475541239
  #res=coppersmith_small_root_entry(n, coeffs)
  #print(res)
  for idx, x in enumerate(lst):
    res = coppersmith_small_root_entry(*x)
    if len(res) > 0:
      print(idx, res)


def solve(ia, ib):
  n, e = get_key()

  va = get_flag(ia)
  vb = get_flag(ib)
  print(n, e, va, vb)
  res = sage_utils.call_argparsified(
      ['sage', '-python'], sage_utils.coppersmith_small_diff, n, e, va, vb)

  print(res)

#print('DATA >> ', data)
#return re.search('RES_CALL=(.*)', data).group(1)

def lll(tb):
  X = Matrix(ZZ, len(tb), len(tb[0]))
  for i in range(len(tb)):
    for j in range(len(tb[0])):
      X[i,j]=tb[i][j]
  L=X.LLL()
  res=[]
  for i in range(len(tb)):
    cur=[]
    for j in range(len(tb[0])):
      cur.append(int(L[i,j]))
    res.append(cur)
  return res

Cmds.add(func=lll, typ='pickle')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  sp = parser.add_subparsers()
  #argparsify(sp, coppersmith_small_diff, [int, int, int, int, int])
  #argparsify(sp, coppersmith_small_root_entry, [int, str, int, float])
  cmdify.argparsify_pickle(sp, lll)

  args = parser.parse_args()
  args.func(args)
