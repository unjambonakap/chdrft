#!/usr/bin/env python

import glog
import glog
import numpy as np
import numpy as np
import pandas as pd
import re

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.struct.base import Range1D, Range2D
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.opa_types import *
from chdrft.utils.rx_helpers import WrapRX
from numpy._typing import (_8Bit, _16Bit, _32Bit, _64Bit)
import shapely.geometry

import chdrft.dsp.dataop as DataOp

sizes = {8:_8Bit, 16: _16Bit, 32: _32Bit, 64: _64Bit}
def get_typ(base, size, endian):
  d= None
  if base == 's': base='i'
  return f'{endian}{base}{size//8}'



class DataFile:

  def __init__(self, filename, typ=None, samp_rate=None, **kwargs):
    self.filename = filename
    typ_map = {'adcpro': 'adcpro', 'float32': np.float32, 'float16': np.float16}
    if typ in typ_map: typ = typ_map[typ]

    if filename.endswith('.csv'): typ = 'csv'
    assert typ is not None

    if typ == 'adcpro':
      data, samp_rate = Dataset.load_adcpro(filename)
    elif typ == 'csv':
      data = pd.read_csv(filename, **kwargs)
    elif not isinstance(typ, str):
      data = np.fromfile(filename, typ)
    else:
      fmt = '(?P<endian>(<|>))?(?P<typ>(f|s))(?P<size>\d+)(?P<complex>_c)?'

      m = re.match(fmt, typ)
      endian = m['endian'] or '<'
      base_typ = get_typ(m['typ'], int(m['size']), endian)
      is_complex = m['complex']

      tmp = np.fromfile(filename, base_typ)
      if is_complex:
        data = tmp[::2] + 1j * tmp[1::2]
      else:
        data = tmp

    if not isinstance(data, pd.DataFrame):
      data = pd.DataFrame(data)

    self.data = data
    self.samp_rate = samp_rate

  @property
  def size(self):
    return self.data.shape[1]

  def col(self, i):
    if isinstance(i, int): i = self.data.columns[i]
    return self.data[i].values

  def col_name(self, i):
    return self.data.columns.values[i]

  def to_ds(self, i=0, **kwargs):
    tmp = dict(kwargs)
    if 'samp_rate' not in tmp:
      tmp['samp_rate'] = self.samp_rate
    return Dataset(self.col(i), **tmp)

  @staticmethod
  def load_adcpro(filename):
    x = []
    params = Attributize()
    reg = re.compile('([^\t]*)\t([^\t]+)')
    with open(filename, 'r') as f:
      for line in f.readlines():
        if line.startswith('\t'):
          pass
        m = reg.match(line.rstrip())
        if not m:
          continue
        name = m.group(1)
        data = m.group(2)
        if len(name) == 0:
          x.append(int(data))
        else:
          params[name] = data.replace(',', '.')
    low_in = float(params['Min Code'])
    high_in = float(params['Max Code'])
    low_out = float(params['Min Voltage'])
    high_out = float(params['Max Voltage'])

    coeff = (high_out - low_out) / (high_in - low_in)
    x = [(e - low_in) * coeff + low_out for e in x]
    return [list(x)], float(params['Sampling Frequency'])


class Sampler1D:

  def __init__(self, x=None, samp_rate=None):
    if x is None: x = 0
    if samp_rate is None: samp_rate = 1

    if isinstance(x, Sampler1D):
      self.x = x.x
      self.samp_rate = x.samp_rate
    else:
      self.x = x
      self.samp_rate = samp_rate

  def get_range(self, npoints):
    if isinstance(self.x, Range1D): return self.x
    return Range1D(self.x, self.x + self.samp_rate * npoints)


class Dataset2d:

  def __init__(self, y, x0=None, x1=None):
    self.y = y

    self.x0 = Sampler1D(x0)
    self.x1 = Sampler1D(x1)

  def transpose(self):
    norder = list(range(self.y.ndim))
    norder[0:2] = [1, 0]
    return Dataset2d(np.transpose(self.y, norder), self.x1, self.x0)

  def set_box(self, box):
    self.x0 = Sampler1D(box.xr)
    self.x1 = Sampler1D(box.yr)

  def shift(self, p):
    self.set_box(self.box + p)
    return self

  def new_y(self, ny):
    return Dataset2d(ny, self.x0, self.x1)

  @property
  def box(self):
    return Range2D(self.x0.get_range(self.y.shape[1]), self.x1.get_range(self.y.shape[0]))


class Dataset:

  def __init__(
      self,
      y=None,
      x=None,
      xy=None,
      samp_rate=None,
      orig_dataset=None,
      name='',
      offset=None,
  ):
    if xy is not None:
      xy = np.array(xy)
      if len(xy) == 0: x, y = [], []
      else:
        x = xy[:, 0]
        y = xy[:, 1]
    self.offset = offset
    self.y = y
    self.pub_data_changed = WrapRX.Subject()
    if not samp_rate:
      samp_rate = None

    self.n = len(self.y)
    self.x = x
    if name is None and orig_dataset is not None:
      name = orig_dataset.name + "'"

    self.name = name
    if orig_dataset is not None and samp_rate is None:
      samp_rate = orig_dataset.samp_rate

    if x is None:
      if orig_dataset is not None:
        if len(y) == orig_dataset.n:
          self.x = orig_dataset.x

      if x is None:
        if samp_rate is None: samp_rate = 1
        glog.info(f'Building x with samp rate {samp_rate}')
        self.reset_x(samp_rate)
    self.samp_rate = samp_rate

    if self.samp_rate is None:
      if len(self.x) < 2:
        self.samp_rate = 1
      else:
        assert len(self.x) >= 2
        diff = self.x[-1] - self.x[0]
        self.samp_rate = len(self.x) / diff
    if isinstance(self.x, list):
      self.x = np.array(self.x)
    if isinstance(self.y, list):
      self.y = np.array(self.y)
    glog.info('Creating dataset name=%s, samp_rate=%s', self.name, self.samp_rate)

  def sub_datasets(self):
    return [self.make(x=self.x, y=self.y[:,i], name = f'{self.name}[{i}]') for i in range(self.y.shape[1])]

  def __len__(self):
    return self.n

  def is_complex(self):
    return np.iscomplexobj(self.y)

  def reset_x(self, samp_rate=None, shift=0):
    if samp_rate is not None:
      self.samp_rate = samp_rate
    self.x = np.linspace(shift, (self.n - 1) / self.samp_rate + shift, self.n)
    assert len(self.x) == self.n
    return self


  @property
  def xlast(self) -> float | None:
    if len(self.x) == 0:
      return None
    return self.x[-1]

  @property
  def xy(self):
    return np.stack([self.x, self.y], axis=-1)

  def min_x(self):
    return self.x[0]

  def max_x(self):
    return self.x[-1]

  def get_x(self):
    return self.x

  def get_y(self):
    return self.y

  def downsample(self, rate):
    return Dataset(
        x=self.x[::rate],
        y=self.y[::rate],
        samp_rate=self.samp_rate / rate,
        name='downsample(%s, %d)' % (self.name, rate)
    )

  def sample_range(self, v, a, b):
    a = max(a, 0)
    b = min(b, self.n)
    return v[a:b]

  def sample_at(self, x, nmean=0):
    pos = np.searchsorted(self.x, x)
    if nmean != 0: return self.sample_idx(pos, nmean)

    if pos <= 0: return self.y[0]
    if pos > len(self.y) - 1: return self.y[-1]
    l = self.x[pos] - self.x[pos - 1]
    return self.y[pos - 1] * (x - self.x[pos - 1]) / l + self.y[pos] * (self.x[pos] - x) / l

  def sample_idx(self, idx, nmean=0):
    tmpy = self.sample_range(self.y, idx - nmean, idx + nmean + 1)
    return np.mean(tmpy)

  def extract_by_idx_array(self, arr):
    return Dataset(
        x=self.x[list(arr)], y=self.y[list(arr)], name='sample_by_idx_array(%s)' % self.name
    )

  def xr2idx(self, r):
    ilow = max(0, np.searchsorted(self.x, r.low))
    ihigh = np.searchsorted(self.x, r.high) + 1
    return Range1D(ilow, ihigh, is_int=1)

  def extract_by_idx(self, *args):
    # not precise
    range1d = Range1D(*args).clamp(0, len(self.x))
    return Dataset(
        x=self.x[range1d.low:range1d.high],
        y=self.y[range1d.low:range1d.high],
        samp_rate=self.samp_rate,
        name='sample_idx(%s)' % self.name
    )

  def extract_by_x(self, *args):
    # not precise
    range1d = Range1D(*args)
    ilow = max(0, np.searchsorted(self.x, range1d.low))
    ihigh = np.searchsorted(self.x, range1d.high) + 1
    return Dataset(
        x=self.x[ilow:ihigh],
        y=self.y[ilow:ihigh],
        offset=ilow,
        samp_rate=self.samp_rate,
        name='sample(%s)' % self.name
    )

  def shift(self, pt):
    return Dataset(
        x=self.x + pt[0], y=self.y + pt[1], samp_rate=self.samp_rate, name=self.name + '_shift'
    )

  def plot(self, **kwargs):
    from chdrft.display.service import g_plot_service, PlotTypes
    g_plot_service.plot(self, PlotTypes.Graph, **kwargs)

  @property
  def range1d(self):
    return Range1D(0, len(self) - 1)

  @property
  def rangex(self):
    return Range1D(self.x[0], self.x[-1])

  def find_poly(self, deg):
    return np.poly1d(np.polyfit(self.get_x(), self.get_y(), deg))

  def extend_x(self, destv=None, ratio=None):
    npoints = None
    if destv is not None:
      npoints = int((destv - self.max_x()) / (self.max_x() - self.min_x()) * len(self.x))
    else:
      npoints = int(len(self.x) * ratio)
      destv = self.max_x() + (self.max_x() - self.min_x()) * ratio
    nx = np.append(self.x, np.linspace(self.max_x(), destv, npoints, endpoint=False))
    return nx

  def find_next_intersection(self, selfpoly, interpoly):
    tmp = selfpoly - interpoly
    roots = np.roots(tmp)
    roots = np.real(roots[np.where(np.isreal(roots))])
    roots.sort()
    pos = np.searchsorted(roots, self.max_x())
    if pos == len(roots): return None
    return roots[pos]

  def eval_at(self, x):
    pos = np.searchsorted(self.x, x)
    if pos == self.n: return self.y[-1]
    if pos == 0: return self.y[0]
    d = self.x[pos] - self.x[pos - 1]
    return self.y[pos] * (x - self.x[pos - 1]) / d + self.y[pos - 1] * (self.x[pos] - x) / d

  def get_closest_point(self, pt):
    return shapely.geometry.Point(pt.x(), self.eval_at(pt.x()))

  def mean(self):
    return self.make_new('mean', y=DataOp.Center(self.y))

  @property
  def center_xy(self):
    xy = self.xy
    return self.make_new('center_xy', xy=xy - np.mean(xy, axis=0))

  def select_edge_idx(self, rising=False):
    rising = int(rising)
    res = []
    for i in range(1, len(self.y)):
      if self.y[i - 1] != self.y[i] and (self.y[i] > self.y[i - 1]) == rising:
        res.append(i)
    return res

  def select_edge(self, rising=False):
    return self.x[self.select_edge_idx(rising=rising)]

  def apply(self, func, ctx=None, **kwargs):
    tmp = dict(globals())
    if ctx is None:
      from IPython.utils.frame import extract_module_locals
      _, ctx = extract_module_locals(1)
    tmp.update(ctx)
    tmp.update(kwargs)
    tmp.update(DataOp.__dict__)
    res = eval(func, tmp, self.__dict__)

    params = cmisc.A()
    if len(res) == len(self): params.x = self.x
    return self.make_new(str(func), y=res, **params)

  def make(self, **kwargs):
    kwargs = dict(kwargs)
    for k, v in list(kwargs.items()):
      if v is None:
        kwargs[k] = getattr(self, k)
    return Dataset(**kwargs)


  def tsf_y(self, fy):
    return self.make(y=fy(self.y))

  def make_new(self, op_name, **kwargs):
    return Dataset(orig_dataset=self, name='%s(%s)' % (op_name, self.name), **kwargs)

  def __getitem__(self, x):
    if not isinstance(x, slice):
      assert False
    else:
      assert x.step is None or x.step == 1

      nx = self.x[x]
      ny = self.y[x]
      res = self.make_new('select_%s' % x, x=nx, y=ny)
      return res

  def to_file(self, filename, typ='float32'):
    DataOp.ToFile(self.y, filename, typ)

  @staticmethod
  def FromPoints(pts):
    pts = np.array(pts).reshape((-1, 2))
    return Dataset(pts[:, 1], pts[:, 0])

  @staticmethod
  def FromImpulse(at, eps=1e-4, maxy=1e5, orig_dataset=None):
    yl = []
    xl = []
    for x in at:
      yl.extend([0, maxy, 0])
      xl.extend([x, x, x])
    return Dataset(yl, x=xl, orig_dataset=orig_dataset)

  @staticmethod
  def FromBitstream(bitstream):
    eps = 1e-4
    yl = []
    xl = []
    prev = bitstream[0]
    for i, v in enumerate(bitstream):
      yl.extend([prev, v])
      xl.extend([i, i])
      prev = v
    return Dataset(yl, x=xl)

  @property
  def mod(self):
    #TODO (ease with dataop)
    pass

  def __add__(self, peer):
    if peer is None: return self
    if isinstance(peer, Dataset): peer = peer.y
    return self.make(y=self.y + peer, x=self.x)

  def __mul__(self, peer):
    if peer is None: return self
    if isinstance(peer, Dataset): peer = peer.y
    return self.make(y=self.y * peer, x=self.x)


class DynamicDataset(Dataset):

  def __init__(self, y, manual_notify_change: bool = False, **kwargs):
    self.manual_notify_change = manual_notify_change
    super().__init__(y, **kwargs)


  def notify_change(self):
      self.pub_data_changed.on_next(None)

  def update_data(self, nx, ny):
    n = len(nx)
    assert n == len(ny)
    self.x = np.append(self.x, nx)
    self.y = np.append(self.y, ny)
    if not self.manual_notify_change:
      self.notify_change()






global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  import chdrft.utils.Z as Z
  a = np.ones((3, 1))
  b = np.zeros((5, 5))
  b[2:, 3] = 1
  Z.pprint(b)

  x = CorrelHelper(a, b, norm=0, mode='valid')
  x.compute_full_correl()
  Z.pprint(x.res)
  res = x.compute_best()
  print(res)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
