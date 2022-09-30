import glog
from chdrft.utils.misc import struct_helper, to_int, BitOps, to_bytes, flatten, yaml_load_custom, yaml_dump_custom
import chdrft.utils.misc as cmisc
import json
import pandas
import yaml
from asq.initiators import query
import pickle
import io

def bit2num(tb):
  res = 0
  for i in reversed(tb):
    res= res << 1 | i
  return res

class Format(object):

  def __init__(self, v=None):
    super().__init__()
    self.__dict__['v'] = v

  def __getattr__(self, name):
    if name.startswith('v_'):
      name = name[2:]
      return lambda x: getattr(fmt(x), name)().v
    raise AttributeError('NAME >> %s'%name)

  def __call__(self, v):
    return Format(v)

  def toint(self, base=None):
    self.v = to_int(self.v, base)
    return self

  def from_conf(self):
    from chdrft.utils.arg_gram import LazyConf
    self.v = LazyConf.ParseFromString(self.v)
    return self

  def to_strio(self):
    self.v = io.StringIO(self.v)
    return self

  def to_pickle(self):
    self.v = pickle.dumps(self.v)
    return self

  def from_pickle(self):
    self.v = pickle.loads(self.v)
    return self

  def from_yaml(self):
    self.v = yaml_load_custom(self.v)
    return self

  def to_yaml(self):
    self.v = yaml_dump_custom(self.v)
    return self

  def to_attr(self):
    self.v = cmisc.Attributize.RecursiveImport(self.v)
    return self

  def from_json(self):
    self.v = cmisc.Attr.RecursiveImport(cmisc.json_loads(self.v))
    return self

  def from_csv(self, **kwargs):
    self.v = pandas.read_csv(self.v, **kwargs)
    return self

  def pretty(self, **kwargs):
    from chdrft.emu.trace import Display
    self.v = Display.disp(self.v, **kwargs)
    return self

  def to_json(self):
    self.v = cmisc.json_dumps(self.v)
    return self

  def to_csv(self, **kwargs):
    if isinstance(self.v, pandas.DataFrame):
      self.v=  self.v.to_csv(**kwargs)
    elif isinstance(self.v, dict):
      self.v = pandas.DataFrame.from_dict(self.v).to_csv(**kwargs)
    else:
      import csv
      assert 0
    return self

  def flatten(self):
    self.v = flatten(self.v)
    return self

  def tobytes(self):
    self.v = to_bytes(self.v)

    return self
  def to_bytes(self):
    self.v = to_bytes(self.v)
    return self

  @staticmethod
  def ToInt(v):
    return Format(v).toint().v

  @staticmethod
  def ToBytes(v):
    return Format(v).tobytes().v

  def tobytearray(self):
    self.v = bytearray(self.v)
    return self


  def lpad(self, n, fill):
    v = self.v
    oldn = n
    n = max(0, n - len(v))
    if isinstance(v, list):
      self.v = ([fill] * n) + list(v)
    elif isinstance(v, bytes):
      self.v = bytes([fill] * n) + v
    elif isinstance(v, bytearray):
      self.v = bytearray([fill] * n) + v
    elif isinstance(v, str):
      self.v = fill * n + v
    else:
      assert 0, 'cant pad on %s' % type(self.v)

    self.v = self.v[:oldn]

    return self
  def pad(self, n, fill, force_size=1):
    v = self.v
    oldn = n
    import numpy as np
    n = max(0, n - len(v))
    if isinstance(v, (list, tuple)):
      self.v = type(v)(list(v) + ([fill] * n))
    elif isinstance(v, bytes):
      self.v = v + bytes([fill] * n)
    elif isinstance(v, bytearray):
      self.v = v + bytearray([fill] * n)
    elif isinstance(v, str):
      self.v = v + fill * n
    elif isinstance(v, np.ndarray):
      self.v = np.pad(v, (0,n), 'constant', constant_values=(0,0))
    else:
      assert 0, 'cant pad on %s, content=%s' % (type(self.v), self.v)

    if force_size: self.v = self.v[:oldn]

    return self

  def same_type(self, val):
    self.v=type(val)(self.v)
    return self

  def lmodpad(self, mod, fill):
    rem = -len(self.v) % mod
    return self.lpad(rem + len(self.v), fill)

  def modpad(self, mod, fill):
    rem = -len(self.v) % mod
    return self.pad(rem + len(self.v), fill)

  def bucket(self, grouplen, default=0):
    if default is not None: self.modpad(grouplen, default)
    res = []
    for i in range(0, len(self.v), grouplen):
      res.append(self.v[i:i + grouplen])
    self.v = res
    return self

  def inv_buckets(self):
    self.v = list([x[::-1] for x in self.v])
    return self

  def bitlist(self, size=None, le=True, bitorder_le=True):

    res=[]
    if isinstance(self.v, bytes) or isinstance(self.v, bytearray) or isinstance(self.v, list):
      for v in self.v:
        for j in range(8):
          if bitorder_le:
            res.append(v >> j & 1)
          else:
            res.append(v >> (7-j) & 1)
    else:
      if size is None:
        size = 0
        while self.v > 0:
          res.append(self.v &1)
          self.v >>= 1
          size += 1

      else:
        for j in range(size):
          res.append(self.v >> j & 1)
      if not bitorder_le:
        res=res[::-1]

    if size is None:
      size = len(res)

    if le: res=res[:size]
    else: res=res[len(res)-size:]

    assert len(res) == size
    self.v = res
    return self

  def bit(self, group=None, inv_group=False, sep=' '):
    if group is None:
      self.v = ''.join([str(x) for x in self.v])
      if inv_group:
        self.v = self.v[::-1]
    else:
      self.bucket(group, default=0)
      self.v = sep.join([Format(x).bit(inv_group=inv_group).v for x in self.v])
    return self

  def bit2list(self):
    if isinstance(self.v, list):
      pass
    elif isinstance(self.v, str):
      self.v = list([int(x) for x in self.v])
    else:
      assert 0
    return self

  def bit2num(self, blk_size=-1):
    cur = self.v
    if blk_size == -1:
      self.v = bit2num(cur)
    else:
      self.v = []
      for i in range(0, len(cur), blk_size):
        self.v.append(bit2num(cur[i:i+blk_size]))
    return self


  def bin2byte(self, lsb=False):
    if isinstance(self.v, str):
      return self.bit2list().bin2byte(lsb=lsb)

    self.modpad(8, 0)
    res = []
    for i in range(0, len(self.v), 8):
      val = 0
      for j in range(8):
        b = self.v[i + j]
        if lsb:
          val += b << j
        else:
          val = (val << 1) + b
      res.append(val)
    self.v = res
    return self

  def byte2num(self, little_endian=True):
    if isinstance(self.v, list):
      self.v = bytes(self.v)
    assert isinstance(self.v, bytes)
    self.v = struct_helper.get(self.v, little_endian=little_endian)[0]
    return self

  def byte_str(self):
    self.v = ' '.join([hex(x)[2:] for x in self.v])
    return self

  def resize_bits(self, n):
    assert isinstance(self.v, bytearray)
    nx = len(self.v) * 8 - n
    if nx // 8 != 0:
      del self.v[-(nx // 8):]

    if nx > 0:
      nx &= 7
      self.v[-1] &= BitOps.mask(8 - (nx & 7))
    return self

  def shiftr(self, n):
    assert n < 8
    n2 = 8 - n
    if n < 0:
      return self.shiftl(-n)
    prev = 0
    for i in reversed(range(len(self.v))):
      x = self.v[i] >> n
      # may have broken shit here, see diff history
      x |= (prev << n2 ) & 0xff
      prev = self.v[i]
      self.v[i] = x
    return self

  def shiftl(self, n):
    if n < 0:
      return self.shiftr(-n)
    assert n < 8
    n2 = 8 - n
    prev=0
    for i in range(len(self.v)):
      x = (self.v[i] << n) & 0xff
      x |= (prev >> n2)
      prev = self.v[i]
      self.v[i] = x
    return self

  def split_by_size(self, sz, pos=False):
    #assert len(self.v)%sz == 0
    if not pos:
      self.v=list([self.v[i:i+sz] for i in range(0, len(self.v), sz)])
    else:
      self.v=list([(i, self.v[i:i+sz]) for i in range(0, len(self.v), sz)])
    return self

  def strip_prefix(self, prefix):
    if self.v.startswith(prefix):
      self.v = self.v[len(prefix):]
    else:
      self.v = None
    return self

  def asq(self):
    return query(self)

class Comparator:
  @staticmethod
  def bytes_eq(a, b):
    return Format(a).tobytes().v==Format(b).tobytes().v

fmt = Format()
