#!/usr/bin/env python

import math
import gmpy2
import numpy as np
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.types import *
from scipy.spatial.transform import Rotation as R

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def make_norm(a):
  return a / np.linalg.norm(a)


def make_orth_c(a, b):
  a = np.array(a)
  b = make_norm(np.array(b))
  return a - np.dot(a, np.conj(b)) * b


def make_orth(a, b):
  a = np.array(a)
  b = make_norm(np.array(b))
  return a - np.dot(a, b) * b


def make_orth_norm(a, b):
  return make_norm(make_orth(a, b))


def orth_v2(a):
  return np.array((-a[1], a[0]))


@cmisc.to_numpy_decorator
def rotate_vec(x, z, angle):
  y = np.cross(z, x)
  return np.cos(angle) * x + np.sin(angle) * y


def rot_look_at(dz, dy):
  dz = make_norm(np.array(dz))
  dy = make_orth_norm(dy, dz)
  dx = np.cross(dy, dz)

  m = np.matrix([dx, dy, dz]).T
  return R.from_matrix(m)


def perspective(fovy, aspect, n, f):
  s = 1.0 / math.tan(np.deg2rad(fovy) / 2.0)
  sx, sy = s / aspect, s
  zz = (f + n) / (n - f)
  zw = 2 * f * n / (n - f)
  return np.matrix([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, zz, zw], [0, 0, -1, 0]])


def rad2deg(a):
  return a * (180 / math.pi)


def deg2rad(a):
  return a * (math.pi / 180)


def numpy_full_len(x):
  return np.product(x.shape)


def unit_vec(i, n):
  res = np.zeros((n,))
  res[i] = 1
  return res


def numpy_vec_decorator():

  def decorator(f):

    def wrap(*args):
      orig_shape = cmisc.SingleValue()
      args = list(args)
      for i in range(len(args)):
        cur = args[i]
        if isinstance(cur, np.ndarray):
          orig_shape.set(cur.shape)
          if len(cur.shape) == 1: args[i] = cur.reshape((-1, len(cur)))
      res = f(*args)
      if orig_shape.val is not None:
        res = res.reshape(orig_shape.val)
      return res

    return wrap

  return decorator


class Normalizer:

  def __init__(self, pts, same_factor=0):
    self.pts = pts
    self.u = np.mean(pts, axis=0)
    tmp = pts - self.u
    self.std = np.std(tmp, axis=0)
    if same_factor: self.std = np.max(self.std)

    self.normalized = tmp / self.std

  def whiten(self, a):
    return (a - self.u) / self.std

  def unwhiten(self, a, vec=0):
    if vec: return a * self.std
    return a * self.std + self.u


class MatHelper:

  def IdMat4():
    return np.identity(4)

  def matmul(a, b):
    return np.matmul(MatHelper.mat4(a), MatHelper.mat4(b))

  def mulnorm(a, b):
    return MatHelper.vec3(np.matmul(MatHelper.mat4(a), MatHelper.vec4(b)))

  def vec3(a):
    if len(a) == 3: return a
    return a[:3] / a[3]

  def vec4(a):
    if len(a) == 4: return a
    return np.array(list(a) + [1])

  def mat4(a):
    if len(a) == 4: return a
    na = np.zeros((4, 4))
    na[:3, :3] = a
    na[3, 3] = 1
    return na

  def matn(a, n=None):
    if n is None: n = max(a.shape)
    if a.shape == (n, n): return a

    na = np.zeros((n, n))
    na[:a.shape[0], :a.shape[1]] = a
    na[n - 1, n - 1] = 1
    return na

  def mat_translate(v):
    res = np.identity(len(v) + 1)
    res[:-1, -1] = v
    return res

  def mat_scale(v, n=2):
    if not is_list(v): v = [v] * n
    return np.diag(MatHelper.make_affine(v))

  def mat_rot(v):
    if isinstance(v, R): v=v.as_matrix()
    n = len(v)
    mat = np.identity(n + 1)
    mat[:n, :n] = v
    return mat

  def simple_mat(offset=None, scale=None, rot=None, n=3):
    mats = []
    if offset is not None:
      mats.append(MatHelper.mat_translate(offset))
    if scale is not None:
      mats.append(MatHelper.mat_scale(scale, n))
    if rot is not None:
      mats.append(MatHelper.mat_rot(rot))
    return MatHelper.mat_apply_nd(*mats, n=n)

  def mat_rotz(rot_z):
    return np.array(
        (
            (math.cos(rot_z), -math.sin(rot_z), 0),
            (math.sin(rot_z), math.cos(rot_z), 0),
            (0, 0, 1),
        )
    )

  def mat_apply2d(*params, point=None, vec=0, affine=0):
    b = params[-1]
    nb = numpy_full_len(b)
    if nb == 3:
      b = b.reshape((3,))
    elif nb == 2:
      if point is None and not vec: point = 1
      b = np.array([b[0], b[1], 1])
    elif len(b.shape) == 2 and b.shape[0] == 2:
      b = np.insert(b, 2, 1, axis=0)
    if point is None: point = 0

    res = b
    for i in reversed(range(len(params) - 1)):
      print(params[i].shape, res.shape)
      res = np.matmul(params[i], res)

    ravel = 0
    if len(res.shape) == 1:
      ravel = 1
      res = res[:, np.newaxis]
    if point: return MatHelper.get_2d_point(res)
    if vec: return MatHelper.get_2d_vec(res)
    if affine: return MatHelper.get_2d_affine(res)
    if ravel:
      res = np.ravel(res)
    return res

  def mat_apply_nd(*params, point=None, vec=0, n=None):
    params = [np.array(x) for x in params]
    b = params[-1]
    if len(params) == 1: return b
    if n is None: n = len(params[0]) - 1

    nb = numpy_full_len(b)
    if nb == n + 1:
      b = b.reshape((n,))
    elif nb == n:
      if point is None and not vec: point = 1
      b = MatHelper.make_affine(b, point=point)
    elif len(b.shape) == 2 and b.shape[0] == n:
      b = np.insert(b, n, 1, axis=0)  # insert 1 on last row
    if point is None: point = 0

    res = b
    for i in reversed(range(len(params) - 1)):
      res = np.matmul(params[i], res)

    ravel = 0
    if len(res.shape) == 1:
      ravel = 1
      res = res[:, np.newaxis]
    if point: res = MatHelper.affine2point(res, n)
    elif vec: res = MatHelper.affine2vec(res, n)

    if ravel: res = np.ravel(res)
    return res

  def make_affine(a, point=True):
    a = np.array(a)
    v = 1 if point else 0
    if len(a.shape) > 1:
      a = np.vstack([a, np.ones_like(a[0]) * v])
    else:
      a = np.array(list(a) + [v])
    return a

  def affine2point(a, n):
    if a.shape[0] == n + 1: return a[:-1, :] / a[-1, :]
    return a

  def affine2vec(a, n):
    if a.shape[0] == n + 1: return a[:-1, :]
    return a

  def get_2d_affine(a):
    assert a.shape == (3, 3), a
    return a[:2, :]

  def get_2d_point(a):
    if a.shape[0] == 3: return a[:2, :] / a[2, :]
    return a

  def get_2d_vec(a):
    if a.shape[0] == 3: return a[:2, :]
    return a


class MathUtils:

  def __init__(self, n, mod=None, pw=1):
    self.n = n
    self.mod = mod
    self.fact = [1] * (n + 1)
    self.ifact = [1] * (n + 1)
    self.isp = set()
    self.pw = pw

    for i in range(1, n + 1):
      self.fact[i] = self.fact[i - 1] * int(gmpy2.powmod(i, self.pw, self.mod))
      if mod is not None:
        self.fact[i] %= mod

    if mod is None:
      self.ifact = None
    else:
      for i in range(1, n + 1):
        self.ifact[i] = int(gmpy2.invert(self.fact[i], mod))

  def cnk(self, n, k):
    if n < k or k < 0: return 0
    if self.mod is not None:
      return self.fact[n] * self.ifact[k] % self.mod * self.ifact[n - k] % self.mod
    return self.fact[n] // self.fact[k] // self.fact[n - k]


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
      for j in range(i * i, n, i):
        self.prev[j] = j // i


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
