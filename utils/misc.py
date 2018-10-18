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

def to_str(x):
  if isinstance(x, str): return x
  return x.decode()

try:
  from enum import Enum
  from contextlib import ExitStack
  import ctypes
  from asq.initiators import query as asq_query
except:
  pass

devnull = open(os.devnull, 'r')

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
  from concurrent.futures import ThreadPoolExecutor, as_completed
  chdrft_executor = ThreadPoolExecutor(max_workers=100)
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

try:
  import yaml
  represent_dict_order = lambda self, data: self.represent_mapping('tag:yaml.org,2002:map', data.items())
  yaml.add_representer(OrderedDict, represent_dict_order)
except:
  pass

def argmax(iterable, f=lambda x: x):
  return max(enumerate(iterable), key=lambda x: f(x[1]))[0]
def argmin(iterable, f=lambda x: x):
  return min(enumerate(iterable), key=lambda x: f(x[1]))[0]

def align(v, pw):
  if pw <= 0: pw = 1
  return ((v - 1) | (pw - 1)) + 1


def mask(pw):
  return ((1 << pw) - 1)


def imask(pw, modpw=None):
  v= ~((1 << pw) - 1)
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
  res = [x ^ y for x, y in zip(a, b)]
  return res


def setb(cur, v, b):
  if v:
    return cur | bit(b)
  return cur & ibit(b)


def lowbit(b):
  return (b & b - 1) ^ b


def sign(x):
  return 1 if x >= 0 else -1

def bit2sign(x):
  return [1, -1][x]

def cntbit(x):
  if x == 0: return 0
  return 1 + cntbit(x & (x - 1))

def ror(v, n, sz):
    n %= sz
    v1 = (v >> n)
    v2 = (v << (sz - n)) % (2 ** sz)
    return v2 | v1


def rol(v, n, sz):
    n %= sz
    v1 = (v << n) % (2 ** sz)
    v2 = v >> (sz - n)
    return v2 | v1


def is_list(x):
  if isinstance(x, list):
    return True
  elif isinstance(x, tuple):
    return True
  return False


def flatten(a, explode=False, depth=-1):
  if depth==0: return a
  if not is_list(a):
    return to_list(a, explode)

  lst = []
  for x in to_list(a):
    lst.extend(flatten(x, depth=depth-1))
  return lst


def to_list(a, explode=False, sep=' '):
  if isinstance(a, list):
    return a
  elif isinstance(a, tuple):
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

def get_n2_locals_and_globals(n=0):
  p2 = inspect.currentframe().f_back.f_back
  for i in range(n): p2= p2.f_back
  return p2.f_locals, p2.f_globals


def path_here(*path):
  frame = inspect.currentframe().f_back
  while not '__file__' in frame.f_locals:
    frame = frame.f_back
  return proc_path(os.path.join(os.path.split(frame.f_locals['__file__'])[0], *path))


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


class DictWithDefault(OrderedDict):

  def __init__(self, default, **kwargs):
    super().__init__(**kwargs)
    self.__default = default

  def __getitem__(self, name):
    if not name in self:
      self[name] = self.__default()
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

  def __init__(self,
               elem=None,
               other=None,
               default=None,
               default_none=False,
               handler=None,
               key_norm=None,
               affect_elem=True,
               repr_blacklist_keys=[],
               **rem):
    if elem is None:
      if (default is not None) or default_none:
        elem = DictWithDefault(lambda: default, **rem)
      else:
        elem = OrderedDict(**rem)

    super().__init__(attributize_main=elem)
    super().__setattr__('_key_norm', key_norm)
    super().__setattr__('_elem', elem)
    super().__setattr__('_other', other)
    super().__setattr__('_handler', handler)
    super().__setattr__('_affect_elem', affect_elem)
    super().__setattr__('_repr_blacklist_keys', list(repr_blacklist_keys))

  def __reduce__(self):
    return (self.__class__, (self._elem,))

  def to_dict(self):
    return dict(self._elem)

  def items(self):
    return self._elem.items()

  def get(self, key, default=None):
    if key in self:
      return self[key]
    return default

  def keys(self):
    return self._elem.keys()

  def __delitem__(self, key):
    del self._elem[key]

  def sorted_keys(self):
    a = list(self.keys())
    a.sort()
    return a

  def values(self):
    return self._elem.values()


  def find_if_str(self, k):
    if isinstance(k, str): return self[k]
    return k

  def norm_key(self, v):
    x = self._key_norm
    if x:
      return x(v)
    return v

  def update(self, *args, **kwargs):
    self._elem.update(*args, **kwargs)

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

  def _do_clone(self):
    return Attributize(elem=dict(self._elem))

  def __eq__(self, other):
    return self._elem == other

  def __str__(self):
    return str(dict(self._elem))
    #return yaml.dump(dict(self._elem), default_flow_style=False)

  def __repr__(self):
    tmp = dict(self._elem)
    for k in self._repr_blacklist_keys:
      if k in tmp: del tmp[k]
    return repr(tmp)

  def to_yaml(self):
    return yaml.dump(dict(self._elem), default_flow_style=False)

  def _str_oneline(self):
    return str(self)
    #return yaml.dump(dict(self._elem), default_flow_style=True)

  def skeleton(self):
    res=Attributize()
    for k,v in self.items():
      if isinstance(v, Attributize):
        res[k] = v.skeleton()
      else:
        res[k] = ''
    return res

  @staticmethod
  def RecursiveImport(e, **params):
    res = e
    if isinstance(e, dict):
      res = Attributize(**params)
      for k, v in e.items():
        res[k] = Attributize.RecursiveImport(v)
    elif is_list(e):
      res = []
      for x in e:
        res.append(Attributize.RecursiveImport(x))
    return res

  @staticmethod
  def FromYaml(s, **params):
    res = yaml.load(s)
    return Attributize.RecursiveImport(res, **params)


Attr = Attributize


class Remap:
  def __init__(self):
    self.rmp = {}
    self.inv = []
  def get(self, x):
    if x in self.rmp:
      return self.rmp[x]

    cid = len(self.inv)
    self.rmp[x] = cid
    self.inv.append(x)
    return cid
  def rget(self, id): return self.inv[id]

  @property
  def n(self):
    return len(self.inv)

  def __contains__(self, x):
    return x in self.rmp

if sys.version_info >= (3, 0):

  class AttributizeHandler(handlers.BaseHandler):

    def flatten(self, obj, data):
      data['data'] = self.context.flatten(obj._elem)
      return data

    def restore(self, data):
      return Attributize(self.context.restore(data['data'], reset=False))

  AttributizeHandler.handles(Attr)


class PatternMatcher:

  def __init__(self, check_method, start_method=None):
    self.check_method = check_method
    self.start = start_method

  def check(self, data):
    return self.check_method(data)

  def __call__(self, data):
    return self.check_method(data)

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
  def fromre(reg):
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

class TwoPatternMatcher:
  def __init__(self, a,b):
    self.a = PatternMatcher.Normalize(a)
    self.b = PatternMatcher.Normalize(b)
    self.a_check = False
    self.b_check = False

  def check(self, data):
    ar = self.a(data)
    if ar is not None:
      self.a_check = True
      return ar

    br =  self.b(data)
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
    if self.never: return None
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


def failsafe(action):
  try:
    action()
  except:
    pass


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
    self.field_to_pos = dict({v:i for i,v in enumerate(desc)})

  def from_value(self, value):
    res = Attributize(default=lambda: 0)
    for pos, e in enumerate(self.desc):
      if e is None: continue
      res[e] = (value >> pos & 1)
    #glog.info('BitMapper %s >> %s', value, res)
    return res

  def to_value(self, status):
    res = 0
    for pos, e in enumerate(self.desc):
      if e is None: continue
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
  def __init__(self,  buf, pos=None, stride=1, stop=None):
    self.buf = buf
    self.n = len(buf)
    if stop is None: stop = self.n
    else: stop = stop % self.n
    self.stop = stop

    if pos is None:
      pos = 0
      if stride == -1: pos = self.n - 1
    self.i = -1
    self.pos = pos
    self.stride =stride

  def __iter__(self):
    return self

  def __next__(self):
    self.i += 1
    if self.stop == self.i: raise StopIteration
    return self

  def id(self, p=0):
    return (self.pos + (self.i + p) * self.stride)% self.n

  def get(self, p=0):
    return self.buf[self.id(p=p)]


def dict_oneline(d):
  res = []
  for k,v in d.items():
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
  return dict_oneline(vars_to_dict(var_lst, n=n+1))

def display_vars(var_lst, n=0, **kwargs):
  print(display_vars_str(var_lst, n=n+1), **kwargs)




#TODO: switch to
# from string import Template
# Template('${TESTABC} {not chaing}').substitute(TESTABC=12)

def multi_replace(s, d):
  return  re.sub(r'\b(' + '|'.join(d.keys()) + r')\b', lambda x: str(d[x.group()]), s)


def template_replace(s, **d):
  from string import Template
  return Template(s).substitute(**d)

def template_replace_safe(s, **d):
  from string import Template
  return Template(s).safe_substitute(**d)

def is_interactive():
  import __main__ as main
  return not hasattr(main, '__file__')

def filter_glob_list(lst, globs, blacklist_default=True):
  if isinstance(globs, str): globs = [globs]
  globs = list(globs)
  for entry in lst:
    if len(globs) == 0 and not  blacklist_default: yield entry

    for pattern in globs:
      if fnmatch.fnmatch(entry, pattern):
        yield entry
        break

class InfGenerator:
  def __init__(self, cb):
    self.cb = cb
    self.acc = self.gen()

  def __call__(self):
    pos=next(self.acc)
    return self.cb[pos]

  def gen(self):
    yield 0
    yield 1
    n = 2
    while True:
      for i in range(1, n, 2):
        yield i / n
      n *= 2



def get_input(globs):
  res = set()
  for pattern in globs:
    res.update(glob.glob(pattern))

  x = list(res)
  x.sort()
  return x


struct_helper = StructHelper()

BitOps = Attributize(align=align,
                     mask=mask,
                     imask=imask,
                     bit=bit,
                     ibit=ibit,
                     setb=setb,
                     lowbit=lowbit,
                     cntbit=cntbit,
                     bit2sign=bit2sign,
                     mod1=mod1,
                     sign=sign,
                     xorlist=xorlist,
                     ror=ror,
                     rol=rol,
                     )

Ops = Attributize(BitOps=BitOps, argmin=argmin, argmax=argmax,)

if sys.version_info >= (3, 0):

  class Arch(Enum):
    x86 = 'x86'
    x86_16 = 'x86_16'
    x86_64 = 'x86_64'
    mips = 'mips'
    arm = 'arm'
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
        print('FUU U ', self.fil)
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
      if not data: return
      if self.writer is None:
        self.writer = csv.DictWriter(self.fil, data.keys())
        self.writer.writeheader()
      if self.log:
        glog.info('Pushing %s', dict_oneline(data))
      self.writer.writerow(data)

    def push_vars(self, var_lst, n=0):
      lst = vars_to_dict(to_list(var_lst), n=n+1)
      self.push_dict(lst)


