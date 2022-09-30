from __future__ import print_function
from datetime import datetime, timedelta
import base64
import binascii
import hashlib
import inspect
import inspect
import json
import os
import re
import struct
import threading
import traceback as tb
import pprint as pp
import sys
import glog
from collections import OrderedDict, defaultdict
import csv
import fnmatch
import glob
import numpy as np
import copy
import functools
import shlex
from chdrft.utils.types import is_num, is_list
import tempfile
from reprlib import recursive_repr
from pydantic import BaseModel, Field, Extra
import pydantic

from typing import no_type_check
from copy import deepcopy

pydantic.BaseConfig.copy_on_model_validation = False

def yield_wrapper(f):

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    return list(f(*args, **kwargs))

  return wrapper


def logged_f(filename=None):
  return lambda f: logged_f_internal(f, filename)


def logged_f_internal(f, filename):

  def x(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except:
      if filename is None:
        tb.print_exc()
      else:
        with open(filename, 'w') as fx:
          fx.write(tb.format_exc())

      raise

  return x


def logged_failsafe(f):

  def x(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except:
      tb.print_exc()

  return x


def splitext_full(a):
  a = os.path.splitext(a)[0]
  if a.endswith('.tar'):
    a = os.path.splitext(a)[0]
  return a


def to_str(x):
  if isinstance(x, str):
    return x
  return x.decode()


def list2dictvec(l):
  res = defaultdict(int)
  for x in l:
    res[x] = 1
  return res


def to_numpy(x):
  if x is None:
    return None
  if isinstance(x, np.ndarray):
    return x
  elif is_num(x):
    return x
  return np.array(x)


def firstor(v, val=None):
  if len(v) == 0:
    return val
  return v[0]


def numpy_round_int(x):
  return np.round(x).astype(int)


def iterate_pairs(lst):
  lst = list(lst)
  for i in range(len(lst)):
    for j in range(i):
      yield lst[i], lst[j]


def to_numpy_args(*args):
  for arg in args:
    yield to_numpy(arg)


class List:

  @staticmethod
  def Get(lst, e):
    if not is_list(e): return lst[e]
    return [lst[x] for x in e]


def to_numpy_decorator_cl(f):

  def wrap(self, *args):
    return f(self, *to_numpy_args(*args))

  return wrap


def to_numpy_decorator(f):

  def wrap(*args):
    return f(*to_numpy_args(*args))

  return wrap


def cl_norm_decorator(normf):

  def wrap0(f):

    def wrap(self, *args):
      return f(self, *list(map(normf, args)))

    return wrap

  return wrap0


try:
  from enum import Enum
  from contextlib import ExitStack
  import ctypes
  from asq.initiators import query as asq_query
except:
  pass

devnull = open(os.devnull, 'r')


def identity(x):
  return x


def none_func():
  return None


def list_func():
  return list


def dict_func():
  return dict


import itertools
try:
  from progressbar import ProgressBar
except:
  pass
is_python2 = sys.version_info < (3, 0)
if is_python2:
  try:
    from builtins import *
  except:
    pass

misc_backend = None
chdrft_executor = None
try:
  from concurrent.futures import ThreadPoolExecutor, as_completed, Future

  class CustomExecutor(ThreadPoolExecutor):

    def submit_log(self, fn, *args, **kwargs):

      def logged(*args, **kwargs):
        try:
          return fn(*args, **kwargs)
        except Exception as e:
          tb.print_exc()
          raise

      return self.submit(logged, *args, **kwargs)

  chdrft_executor = CustomExecutor(max_workers=100)
  chdrft_executor = chdrft_executor.__enter__()
except:
  pass

if sys.version_info >= (3, 0):
  misc_backend = None
  import jsonpickle
  from jsonpickle import handlers
  misc_backend = jsonpickle.backend.JSONBackend()
  misc_backend.set_encoder_options('json', sort_keys=True, indent=4)
  misc_backend.set_preferred_backend('json')

  class BinaryHandler(jsonpickle.handlers.BaseHandler):

    def flatten(self, obj, data):
      data['data'] = base64.b64encode(obj).decode('ascii')
      return data

    def restore(self, obj):
      return base64.b64decode(obj['data'].encode('ascii'))

  BinaryHandler.handles(bytes)

  #import jsonpickle.ext.numpy as jsonpickle_numpy
  #jsonpickle_numpy.register_handlers()
  def json_dumps(*args, **kwargs):
    return jsonpickle.dumps(*args, backend=misc_backend, unpicklable=0, **kwargs)

  def json_loads(*args, **kwargs):
    return jsonpickle.loads(*args, backend=misc_backend, **kwargs)


def ordered_groupby(lst, key=None):
  res = defaultdict(list)
  for x in lst:
    res[key(x)].append(x)
  return list([res[x] for x in sorted(res)])


def argmax(iterable, f=lambda x: x):
  return max(enumerate(iterable), key=lambda x: f(x[1]))[0]


def argmin(iterable, f=lambda x: x):
  return min(enumerate(iterable), key=lambda x: f(x[1]))[0]


def align(v, pw):
  if pw <= 0:
    pw = 1
  return ((v - 1) | (pw - 1)) + 1


def mask(pw):
  return ((1 << pw) - 1)


def imask(pw, modpw=None):
  v = ~((1 << pw) - 1)
  if modpw is not None:
    v = v & mask(modpw)
  return v


def bit(pw):
  return 1 << pw


def ibit(pw):
  return ~bit(pw)


def mod1(v, mod):
  v %= mod
  if v == 0:
    v = mod
  return v


def xorlist(a, b):
  res = type(a)([x ^ y for x, y in zip(a, b)])
  return res


def setb(cur, v, b):
  if v:
    return cur | bit(b)
  return cur & ibit(b)


def lowbit(b):
  return (b & b - 1) ^ b


def highbit_log2(b):
  for i in itertools.count(0):
    if b == 0:
      return i - 1
    b >>= 1


def sign(x):
  return 1 if x >= 0 else -1


def bit2sign(x):
  return [1, -1][x]


def cntbit(x):
  if x == 0:
    return 0
  return 1 + cntbit(x & (x - 1))


def ror(v, n, sz):
  n %= sz
  v1 = (v >> n)
  v2 = (v << (sz - n)) % (2**sz)
  return v2 | v1


def rol(v, n, sz):
  n %= sz
  v1 = (v << n) % (2**sz)
  v2 = v >> (sz - n)
  return v2 | v1


def cycle_arr(a, n):
  return np.concatenate((a[n:], a[n:]))

def loop(x):
  ix = iter(x)
  a = next(ix)
  yield a
  yield from ix
  yield a



def flatten(a, explode=False, depth=-1):
  if depth == 0:
    return a
  if isinstance(a, np.ndarray):
    assert depth <= -1
    return flatten(a.flatten().tolist())
  if not is_list(a):
    return to_list(a, explode)

  lst = []
  for x in to_list(a):
    lst.extend(flatten(x, depth=depth - 1))
  return lst


def zip_safe(*args):

  elems = []
  for x in args:
    elems.append(list(x))
    assert len(elems[0]) == len(elems[-1])
  return zip(*elems)


def to_list(a, explode=False, sep=' '):
  if isinstance(a, list):
    return a
  elif isinstance(a, (tuple, range)):
    return list(a)
  elif isinstance(a, str):
    return a.split(sep)
  return [a]


def csv_list(a):
  if len(a) == 0:
    return []
  return to_list(a, sep=',')


def concat_flat(*lst):
  res = []
  for x in lst:
    res += to_list(x)
  return res


def kv_to_dict(data, sep='='):
  res = {}
  for x in data:
    k = None
    v = None
    if isinstance(x, list):
      k, v = x
    elif isinstance(x, str):
      k, v = x.split(sep, maxsplit=1)
    else:
      assert 0, str(data)

    res[k] = v
  return res


class OpaExecutor:

  def __init__(self, num):
    self.executor = ThreadPoolExecutor(num)
    self.tasks = []
    self.ex = None

  def __enter__(self):
    self.ex = self.executor.__enter__()
    return self

  def __exit__(self, typ, value, tcb):
    self.do_wait()
    self.executor.__exit__(typ, value, tcb)

  def launcher(self, data):
    try:
      return data[0](*data[1:])
    except:
      tb.print_exc()

  def submit(self, func, *args):
    data = [func]
    data.extend(args)
    res = self.ex.submit(self.launcher, data)
    self.tasks.append(res)
    return res

  def map(self, func, data):
    for x in data:
      self.tasks.append(self.ex.submit(self.launcher, [func, x]))

  def results(self):
    return list([x.result() for x in self.tasks])

  def do_wait(self):
    mv = len(self.tasks)
    pb = ProgressBar(max_value=mv).start()
    pos = 0
    pb.update(0)
    for out in as_completed(self.tasks):
      pos += 1
      pos = min(pos, mv)
      pb.update(pos)
    pb.finish()
    print('')


class JsonUtils:

  def __init__(self, backend=misc_backend):
    self.backend = backend

  def encode(self, obj):
    return jsonpickle.encode(obj, backend=self.backend, keys=True)

  def decode(self, data):
    return jsonpickle.decode(data, backend=self.backend, keys=True)


def opa_print(x):
  print(jsonpickle.encode(x, backend=misc_backend))


def proc_path(path):
  if path.startswith('~'):
    path = os.path.join(os.getenv('HOME'), path[2:])
  return os.path.normpath(path)


def cwdpath(path):
  return proc_path(os.path.join(os.getcwd(), path))


def cwdpath_abs(path):
  return os.path.abspath(proc_path(os.path.join(os.getcwd(), path)))


def add_to_n_globals(n, **kwargs):
  _, globs = get_n2_locals_and_globals(n + 1)
  print('abc' in globs)
  globs.update(kwargs)


def add_to_parent_globals(**kwargs):
  add_to_n_globals(1, **kwargs)


def get_n2_locals_and_globals(n=0):
  p2 = inspect.currentframe().f_back.f_back
  for i in range(n):
    p2 = p2.f_back
  return p2.f_locals, p2.f_globals


def path_here(*path):
  frame = inspect.currentframe().f_back
  while not '__file__' in frame.f_globals:
    frame = frame.f_back
  return proc_path(
      os.path.join(os.path.split(os.path.realpath(frame.f_globals['__file__']))[0], *path)
  )


def path_from_script(script, *path):
  return proc_path(os.path.join(os.path.realpath(os.path.split(script)[0]), *path))


def get_or_set_attr(self, attr, default):
  if not hasattr(self, attr):
    setattr(self, attr, default)
  return getattr(self, attr)


def normalize_path(basepath, path):
  return os.path.normpath(os.path.join(basepath, path))


class Dict(dict):

  def __getattr__(self, name):
    if not super().__contains__(name):
      raise AttributeError
    return super().__getitem__(name)

  def __setattr__(self, name, val):
    super().__setitem__(name, val)

  def __add__(self, x):
    tmp = Dict(self)
    tmp.update(x)
    return tmp


class CustomClass:

  def __init__(self, setitem=None, getitem=None, **kwargs):
    self.setitem = setitem
    self.getitem = getitem
    self.__dict__.update(**kwargs)

  def __setitem__(self, k, v):
    self.setitem(k, v)

  def __getitem__(self, k):
    return self.getitem(k)


class DictWithDefault(OrderedDict):

  def __init__(self, default=None, **kwargs):
    super().__init__(**kwargs)
    self._default = default

  def __getitem__(self, name):
    if not name in self:
      self[name] = self._default()
    return super().__getitem__(name)


def to_dict(lst):
  return {a: b for a, b in lst}


class NormHelper:

  @staticmethod
  def lowercase_norm(x):
    if isinstance(x, str):
      return x.lower()
    return x

  @staticmethod
  def byte_norm(x):
    if isinstance(x, bytes):
      return x.decode()
    return x

  @staticmethod
  def numpy_norm(x):
    if isinstance(x, (np.ndarray, list)):
      return tuple(x)
    return x


lowercase_norm = NormHelper.lowercase_norm
byte_norm = NormHelper.byte_norm


class DictUtil:

  @staticmethod
  def get_or_insert(d, key, val):
    if key in d:
      return d[key]
    d[key] = val
    return val


class Attributize(dict):

  @staticmethod
  def FromElems(elems, key=None, value=identity, **kwargs):
    assert key is not None
    d = {}
    return Attributize(_merge={key(x): value(x) for x in elems}, **kwargs)

  def __new__(cls, *args, **kwargs):
    instance = dict.__new__(cls, *args, **kwargs)
    dict.__setattr__(instance, '_key_norm', None)
    dict.__setattr__(instance, '_elem', OrderedDict())
    dict.__setattr__(instance, '_other', None)

    # _handler returns None or tuple(object, should_insert)
    dict.__setattr__(instance, '_handler', None)
    dict.__setattr__(instance, '_affect_elem', True)
    dict.__setattr__(instance, '_repr_blacklist_keys', [])
    return instance

  def __init__(
      self,
      elem=None,
      other=None,
      default=None,
      default_none=False,
      handler=None,
      key_norm=None,
      affect_elem=True,
      repr_blacklist_keys=[],
      _merge={},
      nopickle=0,
      *args,
      **rem
  ):
    if elem is None:
      if (default is not None) or default_none:
        if not callable(default):
          ndefault = lambda: default
        else:
          ndefault = default
        elem = DictWithDefault(ndefault, **rem)
      else:
        elem = OrderedDict(**rem)
    elem.update(_merge)

    super().__init__()
    super().__init__(attributize_main=elem)
    object.__setattr__(self, '_key_norm', key_norm)
    object.__setattr__(self, '_elem', elem)
    object.__setattr__(self, '_other', other)
    object.__setattr__(self, '_handler', handler)  # returns None or tuple(object, should_insert)
    object.__setattr__(self, '_affect_elem', affect_elem)
    object.__setattr__(self, '_repr_blacklist_keys', repr_blacklist_keys)
    object.__setattr__(self, '_nopickle', nopickle)

  def _do_clone(self, elem=None):
    if elem is None: elem = self._elem
    return Attributize(
        elem=dict(self._elem),
        key_norm=self._key_norm,
        _other=self._other,
        handler=self._handler,
        affect_elem=self._affect_elem,
        repr_blacklist_key=self._repr_blacklist_keys,
        nopickle=self._nopickle
    )

  def __deepcopy__(self, memo):
    cl = self._do_clone(deepcopy(self._elem, memo))
    return cl

  def __iter__(self):
    return iter(self._elem)

  #def __reduce__(self):
  #  return (self.__class__, (self._elem,))
  def __getstate__(self):
    if getattr(self, '_nopickle', 0): return dict()
    return dict(_elem=self._elem, _key_norm=self._key_norm, _other=self._other)

  def __reduce__(self):
    if getattr(self, '_nopickle', 0): return (A, (), None, None, None)
    return (Attributize, (), self.__getstate__(), None, None, None)

  def __setstate__(self, x):
    for k, v in x.items():
      super().__setattr__(k, v)

  def __hash__(self):
    return hash(id(self))

  def to_dict(self):
    return dict(self._elem)

  def items(self):
    return self._elem.items()

  def get(self, key, default=None):
    if key in self:
      return self[key]
    return default

  def get_or_insert(self, key, default):
    return DictUtil.get_or_insert(self, key, default)

  @yield_wrapper
  def get_multiple(self, keys):
    for key in keys:
      yield self[key]

  def keys(self):
    return self._elem.keys()

  def __delitem__(self, key):
    key = self.norm_key(key)
    del self._elem[key]

  def asq(self):
    return asq_query(self)

  def sorted_keys(self):
    a = list(self.keys())
    a.sort()
    return a

  def values(self):
    return self._elem.values()

  def find_if_str(self, k):
    if isinstance(k, str):
      return self[k]
    return k

  def norm_key(self, v):
    x = self._key_norm
    if x:
      return x(v)
    return v

  def update(self, *args, **kwargs):
    self._elem.update(*args, **kwargs)
    return self

  def __len__(self):
    return len(self._elem)

  def __getattr__(self, name):
    name = self.norm_key(name)
    try:
      return self._elem.__getattribute__(name)
    except AttributeError:
      pass

    try:
      return self._elem[name]
    except KeyError:
      if self._handler:
        res = self._handler(name)
        if res is not None:
          if res[1]:
            self._elem[name] = res[0]
          return res[0]
      raise AttributeError('Shit attribute %s' % name)

  def __setattr__(self, name, val):
    self.__setitem__(name, val)

  def __getitem__(self, name):
    name = self.norm_key(name)
    try:
      return self._elem[name]
    except KeyError:
      if self._other:
        return self.__add_key(name, self._other(name))
      if self._handler:
        res = self._handler(name)
        if res is not None:
          if res[1]:
            self._elem[name] = res[0]
          return res[0]
      raise KeyError('Shit key %s' % name)

  def __setitem__(self, name, val):
    name = self.norm_key(name)
    self.__add_key(name, val)

  def __add_key(self, name, val):
    if self._affect_elem:
      self._elem[name] = val
    else:
      super().__setattr__(name, val)
    return val

  def __contains__(self, name):
    name = self.norm_key(name)
    return name in self._elem

  def __eq__(self, other):
    if id(self) == id(other): return True
    return self._elem == other

  def __str__(self):
    return str(dict(self._elem))

  @recursive_repr()
  def __repr__(self):
    return repr(dict(self._elem))
    #return yaml.dump(dict(self._elem), default_flow_style=False)

  #def __repr__(self):
  #  tmp = dict(self._elem)
  #  for k in self._repr_blacklist_keys:
  #    if k in tmp:
  #      del tmp[k]
  #  return repr(tmp)

  def to_yaml(self):
    return yaml.dump(dict(self._elem), default_flow_style=False)

  def _str_oneline(self):
    return str(self)
    #return yaml.dump(dict(self._elem), default_flow_style=True)

  def deepmerge(self, src):
    for k, v in src.items():
      if k not in self:
        self[k] = v
      elif isinstance(v, Attr):
        self[k].deepmerge(v)
        self[k]

  @staticmethod
  def Skeleton(e):
    if is_list(e):
      res = Attr(OPA_LIST=1, item=Attr())
      for obj in e:
        res.item.deepmerge(Attributize.Skeleton(obj))
    elif isinstance(e, dict):
      res = Attr()
      for k, v in e.items():
        res[k] = Attributize.Skeleton(v)
    else:
      res = Attr()
    return res

  @staticmethod
  def ToDict(self, rec=0):
    if isinstance(self, Attr) or (rec and isinstance(self, dict)):
      res = dict()
      for k, v in self.items():
        res[k] = Attr.ToDict(v, rec)
      return res
    if not rec: return self

    if is_list(self):
      return [Attr.ToDict(x, rec) for x in self]
    return self

  @staticmethod
  def RecursiveImport(e, force=1, **params):
    res = e
    if not force and isinstance(e, A): return e
    if isinstance(e, dict):
      res = Attributize(**params)
      for k, v in e.items():
        res[k] = Attributize.RecursiveImport(v, force=force, **params)
    elif isinstance(e, (list, tuple)):
      res = []
      for x in e:
        res.append(Attributize.RecursiveImport(x, force=force, **params))
      res = type(e)(res)
    return res

  @staticmethod
  def FromYaml(s, **params):
    res = yaml.load(s)
    return Attributize.RecursiveImport(res, **params)

  @staticmethod
  def FromJson(s, **params):
    if isinstance(s, str):
      s = open(s, 'r').read()
    res = json_loads(s)
    return Attributize.RecursiveImport(res, **params)

  @staticmethod
  def Numpy(**kwargs):
    return Attr(key_norm=NormHelper.numpy_norm, **kwargs)

  @staticmethod
  def MakeId(**kwargs):
    return Attr(key_norm=id, **kwargs)


Attr = Attributize
A = Attributize


class IntMapper:

  def __init__(self):
    self.tb = []

  def add(self, x):
    idx = len(self.tb)
    self.tb.append(x)
    return idx


class Remap:

  @staticmethod
  def Numpy(**kwargs):
    return Remap(rmp=Attr(key_norm=NormHelper.numpy_norm), **kwargs)

  def __init__(self, tb=[], rmp=None, key=None):
    if rmp is None:
      rmp = {}
    self.rmp = rmp
    self.inv = []
    self.key = key
    for x in tb:
      self.get(x)

  def get_key(self, x):
    if self.key:
      k = self.key(x)
    else:
      k = x
    return k

  def get(self, x, assert_in=0):
    k = self.get_key(x)

    if k in self.rmp:
      return self.rmp[k]
    assert assert_in == 0

    cid = len(self.inv)
    self.rmp[k] = cid
    self.inv.append(x)
    return cid

  def rget(self, id):
    return self.inv[id]

  @property
  def n(self):
    return len(self.inv)

  def __contains__(self, x):
    return self.get_key(x) in self.rmp


class PatternMatcher:

  def __init__(self, check_method, start_method=None, get_method=None):
    self.check = check_method
    self.start = start_method
    self.get = get_method
    self.result = None

  def check(self, data):
    return self.check(data)

  def __call__(self, data):
    return self.check(data)

  @staticmethod
  def Before(pattern, data):
    if not isinstance(pattern, PatternMatcher):
      pattern = PatternMatcher.frombytes(pattern)
      return data[:pattern.start(data)]

  @staticmethod
  def Normalize(matcher):
    if isinstance(matcher, str):
      matcher = matcher.encode()
    if isinstance(matcher, bytes):
      matcher = PatternMatcher.frombytes(matcher)
    return matcher

  @staticmethod
  def smart(matcher, re=0):
    if re:
      return PatternMatcher.fromre(matcher)

    if isinstance(matcher, str):
      return PatternMatcher.fromstr(matcher)
    elif isinstance(matcher, bytes):
      return PatternMatcher.frombytes(matcher)
    return matcher

  @staticmethod
  def frombytes(s):

    def checker(data):
      x = data.find(s)
      if x == -1:
        return None
      return x + len(s)

    def start(data):
      x = data.find(s)
      if x == -1:
        return None
      return x

    return PatternMatcher(checker, start)

  @staticmethod
  def fromstr(s):
    # does not change :)
    return PatternMatcher.frombytes(s)

  @staticmethod
  def fromre(reg):
    r = re.compile(reg)

    pm = PatternMatcher(None)

    def get(data):
      return r.search(data)

    def checker(data):
      x = get(data)
      if x is None:
        return None
      pm.result = x
      return x.end()

    def start(data):
      x = get(data)
      if x is None:
        return None
      return x.start()

    pm.get = get
    pm.check = checker
    pm.start = start
    return pm

  @staticmethod
  def fromglob(glob):

    def checker(data):
      return fnmatch.fnmatch(data, glob)

    return PatternMatcher(checker, None)

  @staticmethod
  def fromstr(reg):
    r = re.compile(reg)

    def checker(data):
      x = r.search(data)
      if x is None:
        return None
      return x.end()

    def start(data):
      x = r.search(data)
      if x is None:
        return None
      return x.start()

    return PatternMatcher(checker, start)

  @staticmethod
  def CreatePattern(pattern):
    if isinstance(pattern, PatternMatcher):
      return pattern(s)

    m = re.match('(?P<typ>\w+):(?P<arg>.*)', pattern)
    if not m:
      return PatternMatcher.fromstr(pattern)
    else:
      typ, arg = m['typ'], m['arg']
      if typ == 'glob':
        return PatternMatcher.fromglob(arg)
      elif typ == 're':
        return PatternMatcher.fromre(arg)
      return PatternMatcher.fromstr(arg)

  @staticmethod
  def IsMatching(pattern, s):
    if pattern is None:
      return True
    pattern = PatternMatcher.CreatePattern(pattern)
    return pattern(s)


class TwoPatternMatcher:

  def __init__(self, a, b):
    self.a = PatternMatcher.Normalize(a)
    self.b = PatternMatcher.Normalize(b)
    self.a_check = False
    self.b_check = False

  def check(self, data):
    ar = self.a(data)
    if ar is not None:
      self.a_check = True
      return ar

    br = self.b(data)
    if br is not None:
      self.b_check = True
      return br
    return None

  def __call__(self, data):
    return self.check(data)


class SeccompFilters:
  SIZE = 8

  def __init__(self, mem):
    pos = 0

    self.code = struct.unpack('<H', mem[pos:pos + 2])[0]
    pos += 2

    self.jt = struct.unpack('<B', mem[pos:pos + 1])[0]
    pos += 1

    self.jf = struct.unpack('<B', mem[pos:pos + 1])[0]
    pos += 1

    self.k = struct.unpack('<I', mem[pos:pos + 4])[0]
    pos += 4

  def disp(self):
    print("code=%x, jt=%x, jf=%x, k=%x" % (self.code, self.jt, self.jf, self.k))


class Timer:

  def __init__(self):
    self._start = None

  def start(self):
    self._start = datetime.now()

  def get(self):
    return Timespan(datetime.now() - self._start)


class TimeoutException(Exception):

  def __init__(self):
    super().__init__('timeout')


class Timeout:
  _TimeoutException = TimeoutException

  def __init__(self, td=None):
    self.start = datetime.now()
    self.never = td is None
    if self.never:
      self.end = None
    else:
      self.end = self.start + td

  @staticmethod
  def from_sec(x):
    return Timeout(timedelta(seconds=x))

  def expired(self):
    return not self.never and self.end <= datetime.now()

  def get_sec(self):
    if self.never:
      return None
    return (self.end - self.start).total_seconds()

  def raise_if_expired(self):
    if self.expired():
      raise TimeoutException()

  @staticmethod
  def Build(x=None):
    if (isinstance(x, int) or isinstance(x, float)):
      if x > 0:
        x = timedelta(seconds=x)
      else:
        x = None
    return Timeout(x)

  @staticmethod
  def Normalize(v):
    if isinstance(v, Timeout):
      return v
    return Timeout.Build(v)


class Timespan:
  sec_mul = 1000000
  day_mul = 24 * 3600 * sec_mul

  # take timedelta
  def __init__(self, td):
    self.td = td

  def tot_usec(self):
    return self.td.microseconds + self.td.seconds * \
            self.sec_mul + self.td.days * self.day_mul

  def tot_msec(self):
    return self.tot_usec() / 1000

  def tot_sec(self):
    return self.tot_usec() / 1000000


def failsafe_or(action, alt=None):
  try:
    return action()
  except:
    return alt


def failsafe(action):
  try:
    action()
  except:
    pass


makedirs = lambda cur_dir: failsafe(lambda: os.makedirs(cur_dir))


def change_extension(x, ext=None):
  tmp = os.path.splitext(x)[0]
  if ext:
    tmp += '.' + ext
  return tmp


class OpaStrVersion:

  def __str__(self):
    m = hashlib.md5()
    classes = inspect.getmro(self.__class__)
    m.update(str(classes).encode())
    for x in classes:
      if x == object:
        continue
      m.update(inspect.getsource(x).encode())
    return 'OpsStrVersion.__str__=' + binascii.hexlify(m.digest()).decode()

  def __repr__(self):
    return str(self)


def opa_serialize(swig_struct):
  return ctypes.string_at(swig_struct.get_ptr(), swig_struct.get_size())


def to_filename(x):
  assert isinstance(x, str)
  return x.replace(' ', '_').lower()


def OpaInit(var, func):
  if var['__name__'] == '__main__':
    func()


def to_bytes(x):
  if isinstance(x, int):
    x = bytes([x])
  if isinstance(x, str):
    x = x.encode()
  else:
    x = bytes(x)
  return x


def to_int(x, base=None):
  if isinstance(x, int):
    return x
  elif isinstance(x, str):
    if base == 16 or x.startswith('0x'):
      return int(x, 16)
    return int(x, 10)
  else:
    assert 0, 'bad shit %s' % x


def to_int_or_none(x):
  try:
    return to_int(x)
  except:
    return None


class BitMapper:

  def __init__(self, desc):
    self.desc = desc
    self.field_to_pos = dict({v: i for i, v in enumerate(desc)})

  def from_value(self, value):
    res = Attributize(default=lambda: 0)
    for pos, e in enumerate(self.desc):
      if e is None:
        continue
      res[e] = (value >> pos & 1)
    #glog.info('BitMapper %s >> %s', value, res)
    return res

  def to_value(self, status):
    res = 0
    for pos, e in enumerate(self.desc):
      if e is None:
        continue
      res = res | (status[e] << pos)
    return res


def read_file_or_buf(filename=None, data=None):
  if filename is not None:
    return open(filename, 'rb').read()
  return data


class StructHelper:

  def __init__(self):
    types = []
    types.append('Q 64')
    types.append('I 32')
    types.append('H 16')
    types.append('B 8')
    self.types = types
    self.little_endian = '><'

    self.s2c = {}
    for x in types:
      l, val = x.split(' ')
      val = int(val)
      self.s2c[val // 8] = l

  def tobytes(self, *args, **kwargs):
    return self.set(*args, **kwargs)

  def frombytes(self, *args, **kwargs):
    return self.get(*args, **kwargs)

  def get(self, v, size=None, nelem=None, little_endian=True):
    auto_size = nelem is None
    if size is None and nelem is None:
      size = len(v)

    if size is not None:
      assert len(v) % size == 0
      nelem = len(v) // size

    if nelem is not None:
      assert len(v) % nelem == 0
      size = len(v) // nelem

    assert size in self.s2c
    pattern = '%s%d%s' % (self.little_endian[little_endian], nelem, self.s2c[size])
    res = struct.unpack(pattern, v)
    if len(res) == 1 and auto_size:
      res = res[0]
    return res

  def set(self, v, size=None, little_endian=True):
    v = to_list(v)
    assert size in self.s2c
    pattern = '%s%d%s' % (self.little_endian[little_endian], len(v), self.s2c[size])
    return struct.pack(pattern, *v)


class CircularBufferView:

  def __init__(self, buf, pos=None, stride=1, stop=None):
    self.buf = buf
    self.n = len(buf)
    if stop is None:
      stop = self.n
    else:
      stop = stop % self.n
    self.stop = stop

    if pos is None:
      pos = 0
      if stride == -1:
        pos = self.n - 1
    self.i = -1
    self.pos = pos
    self.stride = stride

  def __iter__(self):
    return self

  def __next__(self):
    self.i += 1
    if self.stop == self.i:
      raise StopIteration
    return self

  def id(self, p=0):
    return (self.pos + (self.i + p) * self.stride) % self.n

  def get(self, p=0):
    return self.buf[self.id(p=p)]


def dict_oneline(d):
  res = []
  for k, v in d.items():
    res.append(k + '=' + str(v))
  s = ', '.join(res)
  return s


def vars_to_dict(var_lst, n=0):
  var_lst = to_list(var_lst)
  ploc, pglob = get_n2_locals_and_globals(n=n)
  res = {}
  for var in var_lst:
    res[var] = eval(var, pglob, ploc)
  return res


def display_vars_str(var_lst, n=0):
  var_lst = to_list(var_lst)
  return dict_oneline(vars_to_dict(var_lst, n=n + 1))


def display_vars(var_lst, n=0, **kwargs):
  print(display_vars_str(var_lst, n=n + 1), **kwargs)


#TODO: switch to
# from string import Template
# Template('${TESTABC} {not chaing}').substitute(TESTABC=12)


def multi_replace(s, d):
  return re.sub(r'\b(' + '|'.join(d.keys()) + r')\b', lambda x: str(d[x.group()]), s)


def build_template_class(_opa_delimiter):
  from string import Template
  if _opa_delimiter is None:
    return Template
  else:

    class MyTemplate(Template):
      delimiter = _opa_delimiter

      def __init__(self, template):
        super().__init__(template)

    return MyTemplate


def template_replace(s, _opa_delimiter=None, **d):
  CL = build_template_class(_opa_delimiter)
  return CL(s).substitute(**d)


def template_replace_safe(s, _opa_delimiter=None, **d):
  CL = build_template_class(_opa_delimiter)
  return CL(s).safe_substitute(**d)


def is_interactive():
  import __main__ as main
  return not hasattr(main, '__file__')


def list_files_rec(base, *paths):
  lst = []
  if len(paths) == 0: paths = ['./']
  for path_entry in paths:
    for root, dirs, files in os.walk(os.path.join(base, path_entry)):
      for f in files:
        lst.append(os.path.join(root, f))
  return asq_query(lst)


def filter_glob_list(lst, globs, blacklist_default=True, key=lambda x: x, blacklist=0):
  if isinstance(globs, str):
    globs = [globs]
  globs = list(globs)
  for entry in lst:
    if len(globs) == 0 and (blacklist or not blacklist_default):
      yield entry
      continue
    kk = key(entry)

    for pattern in globs:
      if fnmatch.fnmatch(kk, pattern):
        if not blacklist:
          yield entry
        break
      else:
        if blacklist:
          yield entry


def whitelist_blacklist_filter(lst, whitelist, blacklist, ikey=lambda x: x[1]):
  ilst = list(enumerate(lst))
  if not ilst:
    return []

  if whitelist:
    ilst = list(filter_glob_list(ilst, whitelist, key=ikey))

  if blacklist:
    ilst = list(filter_glob_list(ilst, blacklist, key=ikey, blacklist=1))
  assert len(ilst) > 0
  return ilst


class InfGenerator:

  def __init__(self, cb, colrange=None):
    from chdrft.struct.base import Range1D
    self.cb = cb
    if colrange is None: colrange = Range1D.Range01()
    self.colrange = colrange
    self.acc = self.gen()

  def __call__(self, v=None):
    if v is None: v = next(self.acc)
    pos = self.colrange.from_local(v)
    return self.cb[pos]

  def gen(self):
    yield 0
    yield 1
    n = 2
    while True:
      for i in range(1, n, 2):
        yield i / n
      n *= 2


class KeyedCGen:

  def __init__(self, cmap):
    self.cmap = cmap
    self.cgen = InfGenerator(cmap)
    self.mp = dict()

  def __call__(self, x):
    col = self.mp.get(x, None)
    if col is not None: return col
    col = self.cgen()
    self.mp[x] = col
    return col


def get_input(globs):
  res = set()
  for pattern in globs:
    res.update(glob.glob(pattern))

  x = list(res)
  x.sort()
  return x


struct_helper = StructHelper()

BitOps = Attributize(
    align=align,
    mask=mask,
    imask=imask,
    bit=bit,
    ibit=ibit,
    setb=setb,
    lowbit=lowbit,
    highbit_log2=highbit_log2,
    cntbit=cntbit,
    bit2sign=bit2sign,
    mod1=mod1,
    sign=sign,
    xorlist=xorlist,
    ror=ror,
    rol=rol,
)

Ops = Attributize(
    BitOps=BitOps,
    argmin=argmin,
    argmax=argmax,
)

if sys.version_info >= (3, 0):

  class Arch(Enum):
    x86 = 'x86'
    x86_16 = 'x86_16'
    x86_64 = 'x86_64'
    mips = 'mips'
    arm = 'arm'
    arm64 = 'arm64'
    thumb = 'thumb'

  class CsvWriterStream(ExitStack):

    def __init__(self, filename, log=False):
      super().__init__()
      self.filename = filename
      self.writer = None
      self.fil = None
      self.cur_vars = {}
      self.log = log

    def __enter__(self):
      fil = open(self.filename, 'w')
      self.enter_context(fil)
      self.fil = fil

    def push_args(self, *args):
      if self.writer is None:
        #, delimiter=',', quotechar='|'
        self.writer = csv.writer(self.fil)
      self.writer.writerow(args)

    def push_dict(self, data):
      for k in data.keys():
        if k in self.cur_vars:
          self.do_push(self.cur_vars)
          self.cur_vars = {}
          break
      self.cur_vars.update(data)

    def do_push(self, data):
      if not data:
        return
      if self.writer is None:
        self.writer = csv.DictWriter(self.fil, data.keys())
        self.writer.writeheader()
      if self.log:
        glog.info('Pushing %s', dict_oneline(data))
      self.writer.writerow(data)

    def push_vars(self, var_lst, n=0):
      lst = vars_to_dict(to_list(var_lst), n=n + 1)
      self.push_dict(lst)


def gcd_list(lst):
  import gmpy2
  res = 0
  for i in lst:
    res = gmpy2.gcd(res, i)
  return res


def make_uniq(lst, key=lambda x: x):
  mp = {}
  for e in lst:
    mp[key(e)] = e
  return list(mp.values())


def get_uniq(x):
  x = list(x)
  assert len(x) == 1, str(x)
  return x[0]


def find_in_list(lst, sublist):
  for i in range(len(lst) - len(sublist) + 1):
    for j in range(len(sublist)):
      if lst[i + j] != sublist[j]:
        break
    else:
      yield i


def int_to_bytes(a, byteorder='big'):
  a = int(a)
  return a.to_bytes(a.bit_length() + 7 >> 3, byteorder=byteorder)


class FormatPrinter(pp.PrettyPrinter):

  def __init__(self, formats, *args, **kwargs):
    super().__init__(*args, **kwargs)
    #self._dispatch[Attributize.__repr__] = self._pprint_dict
    self.formats = formats

  def format(self, obj, ctx, maxlvl, lvl):
    if type(obj) in self.formats:
      return self.formats[type(obj)] % obj, 1, 0
    return super().format(obj, ctx, maxlvl, lvl)


class TempOverride(ExitStack):

  def __init__(self):
    super().__init__()
    self.data = []
    self.to_remove = []

  def cleanup(self):
    for obj, name in self.to_remove:
      delattr(obj, name)
    for obj, name, old_v in self.data:
      setattr(obj, name, old_v)

  def __enter__(self):
    super().__enter__()
    self.callback(self.cleanup)
    return self

  def override_attr(self, obj, name, v):
    if hasattr(obj, name):
      self.data.append((obj, name, getattr(obj, name)))
    else:
      self.to_remove.append((obj, name))
    setattr(obj, name, v)


def safe_select(a, x, default=None):
  if len(a) <= x:
    return default
  return a[x]


def is_array_constant(x):
  for i in range(1, len(x)):
    if x[i] != x[0]:
      return 0
  return 1


ppx = FormatPrinter({float: '%.4f', np.float64: "%.4f", int: "%06X"}, compact=1, width=200)


class IdGen:

  def __init__(self, v=0):
    self.v = v

  def __call__(self):
    return self.next()

  def next(self):
    res = self.v
    self.v += 1
    return res


class AttributizeHandler(handlers.BaseHandler):

  def flatten(self, obj, data):
    return self.context.flatten(obj._elem)

  def restore(self, data):
    return Attributize(self.context.restore(data['data'], reset=False))


def numpytojson(_, obj, data):
  return _.context.flatten(obj.tolist())


def numpygenerictojson(_, obj, data):
  return _.context.flatten(obj.item())


def tupletojson(_, obj, data):
  return _.context.flatten(list(obj))


handlers = [
    Attr(tojson=numpytojson, type=np.ndarray, name='numpyhandler'),
    Attr(tojson=numpygenerictojson, type=np.generic, name='numpygenerichandler'),
    Attr(tojson=tupletojson, type=tuple, name='tuplehandler'),  # NOT WORKING , FING ORDER
    #Attr(tojson=lambda _, obj, data: _.context.flatten(dict(obj)), type=OrderedDict, name='oredreddict_handler'), # NOT WORKING , FING ORDER
    #Attr(tojson=lambda _, obj, data: _.context.flatten(dict(obj)), type=defaultdict, name='defaultdict_handler'), # NOT WORKING , FING ORDER
]

handler_list = []


class SingleValue:

  def __init__(self, val=None):
    self.val = val

  def set(self, v):
    assert self.val is None or self.val == v
    self.val = v


class ParserUtil:

  @staticmethod
  def GetCtx(parser, cmdline=None):
    args = shlex.split(cmdline)
    flags = parser.parse_args(args=args)
    return Attr(vars(flags))


def asq_gen(x):
  return lambda: asq_query(x)


def define_json_handler(handler):
  x = type(handler.name, (jsonpickle.handlers.BaseHandler,), {})
  if 'tojson' in handler:
    x.flatten = handler.tojson
  if 'fromjson' in handler:
    x.restore = handler.fromjson

  #x.handles(handler.type)
  jsonpickle.handlers.register(handler.type, x, base=True)
  handler_list.append(x)


for handler in handlers:
  define_json_handler(handler)

jsonpickle.handlers.register(Attr, AttributizeHandler, base=True)

try:
  import yaml
  represent_dict_order = lambda self, data: self.represent_mapping(
      'tag:yaml.org,2002:map', data.items()
  )

  class CustomDump(yaml.Dumper):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

  represent_numpy_int = lambda self, data: self.represent_int(int(data))
  yaml.Dumper.ignore_aliases = lambda *args: True
  yaml.add_representer(A, lambda self, data: self.represent_dict(dict(data)), Dumper=CustomDump)
  yaml.add_representer(OrderedDict, represent_dict_order)
  yaml.add_representer(np.int64, represent_numpy_int)

  CustomDump.ignore_aliases = lambda *args: True

  yaml.add_representer(OrderedDict, represent_dict_order, Dumper=CustomDump)
  yaml.add_representer(np.int64, represent_numpy_int, Dumper=CustomDump)
  yaml.add_representer(np.int32, represent_numpy_int, Dumper=CustomDump)
  yaml.add_representer(np.float64, lambda self, data: self.represent_float(data), Dumper=CustomDump)
  yaml.add_representer(np.float32, lambda self, data: self.represent_float(data), Dumper=CustomDump)
  yaml.add_representer(np.int64, represent_numpy_int, Dumper=CustomDump)
  yaml.add_representer(
      np.ndarray, lambda self, data: self.represent_list(list(data)), Dumper=CustomDump
  )

  def yaml_dump_custom(x, **kwargs):
    return yaml.dump(x, Dumper=CustomDump, **kwargs)

  def yaml_load_custom(x, **kwargs):
    return yaml.load(x, Loader=yaml.Loader, **kwargs)

except Exception as e:
  print('FUU ', e)
  glog.error(e)
  pass


class PatchedModel(BaseModel, extra=Extra.forbid):

  def __eq__(self, peer): return id(self) == id(peer)
  def __hash__(self):
    return hash(id(self))



  @no_type_check
  def __setattr__(self, name, value):
    """
        To be able to use properties with setters
        """
    try:
      super().__setattr__(name, value)
    except ValueError as e:
      import inspect
      setters = inspect.getmembers(
          self.__class__, predicate=lambda x: isinstance(x, property) and x.fset is not None
      )
      for setter_name, func in setters:
        if setter_name == name:
          object.__setattr__(self, name, value)
          break
      else:
        raise e
