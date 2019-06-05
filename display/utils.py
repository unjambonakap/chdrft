import chdrft.display.base
from IPython.utils.frame import extract_module_locals
from chdrft.utils.swig import swig
from asq.initiators import query as asq_query
from chdrft.dsp.gnuradio import GnuRadioFile
from chdrft.struct.base import Range1D, Range2D, Intervals, ListTools
from chdrft.utils.colors import ColorPool
from chdrft.utils.misc import to_list, Attributize, proc_path
from chdrft.utils.fmt import Format
from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE, USE_PYQT5
from scipy import fftpack
from scipy import optimize
from scipy import signal
from scipy import cluster
from scipy import stats
import glog
import math
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import scipy.ndimage as ndimage
import sys
import tempfile
from scipy.stats.mstats import mquantiles

def norm_angle(x):
  while abs(x) > math.pi:
    if x > 0:
      x -= 2 * math.pi
    else:
      x += 2 * math.pi
  return x

def norm_angle_from(src, dest):
  return src + norm_angle(dest - src)


class DataFile:
  def __init__(self, filename, typ=None, samp_rate=None, **kwargs):
    self.filename = filename
    typ_map = {'adcpro': 'adcpro', 'float32': np.float32}
    if typ in typ_map:
      typ = typ_map[typ]
    if filename.endswith('.csv'):
      typ = 'csv'
    assert typ is not None

    if typ == 'adcpro':
      data, samp_rate = DataSet.load_adcpro(filename)
    elif typ == 'csv':
      data = pd.read_csv(filename, **kwargs)
    elif typ == 'complex8':
      tmp = np.fromfile(filename, np.int8)
      data = tmp[::2] + 1j * tmp[1::2]
    else:
      data = np.fromfile(filename, typ)

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
      tmp['samp_rate']=  self.samp_rate
    return DataSet(self.col(i), **tmp)

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

class Dataset2D:
  def __init__(self, y, x0, x1):
    self.y = y
    self.x0 = x0
    self.x1 = x1

  def transpose(self):
    norder = list(range(self.y.ndim))
    norder[0:2] = [1,0]
    return Dataset2D(np.transpose(self.y, norder), self.x1, self.x0)

  def new_y(self, ny):
    return Dataset2D(ny, self.x0, self.x1)

  @property
  def range2d(self):
    return Range2D((self.x0[0], self.x0[-1]), (self.x1[0], self.x1[-1]))

class DataSet:

  def __init__(self, y, x=None, samp_rate=None, sig_replot_cb=None, orig_dataset=None, name=''):
    self.y = y
    self.sig_replot_cb = sig_replot_cb
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
        self.samp_rate = float(diff) / (len(self.x) - 1)
    if isinstance(self.x, list):
      self.x = np.array(self.x)
    if isinstance(self.y, list):
      self.y = np.array(self.y)
    glog.info('Creating dataset name=%s, samp_rate=%s', self.name, self.samp_rate)

  def __len__(self):
    return self.n

  def is_complex(self):
    return np.iscomplexobj(self.y)

  def reset_x(self, samp_rate=None, shift=0):
    if samp_rate is not None:
      self.samp_rate = samp_rate
    self.x = np.linspace(shift, self.n / self.samp_rate + shift, self.n)
    assert len(self.x) == self.n
    return self

  def min_x(self):
    return self.x[0]

  def max_x(self):
    return self.x[-1]

  def get_x(self):
    return self.x

  def get_y(self):
    return self.y

  def downsample(self, rate):
    return DataSet(x=self.x[::rate],
                    y=self.y[::rate],
                   samp_rate=self.samp_rate // rate,
                   name='downsample(%s, %d)' % (self.name, rate))

  def sample_range(self, v, a, b):
    a = max(a, 0)
    b = min(b, self.n)
    return v[a:b]

  def sample_at(self, x, nmean=0):
    pos = np.searchsorted(self.x, x)
    return self.sample_idx(pos, nmean)

  def sample_idx(self, idx, nmean=0):
    tmpy = self.sample_range(self.y, pos - nmean, pos + nmean + 1)
    return np.mean(tmpy)

  def extract_by_idx(self, *args):
    # not precise
    range1d = Range1D(*args).clamp(0, len(self.x))
    return DataSet(x=self.x[range1d.low:range1d.high],
                   y=self.y[range1d.low:range1d.high],
                   samp_rate=self.samp_rate,
                   name='sample_idx(%s)' % self.name)

  def extract_by_x(self, *args):
    # not precise
    range1d = Range1D(*args)
    ilow = max(0, np.searchsorted(self.x, range1d.low))
    ihigh = np.searchsorted(self.x, range1d.high) + 1
    return DataSet(x=self.x[ilow:ihigh],
                   y=self.y[ilow:ihigh],
                   samp_rate=self.samp_rate,
                   name='sample(%s)' % self.name)

  def shift(self, pt):
    return DataSet(x=self.x + pt[0],
                   y=self.y + pt[1],
                   samp_rate=self.samp_rate,
                   name=self.name + '_shift')

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
      destv = self.max_x() + (self.max_x() -self.min_x()) * ratio
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
    return pg.Point(pt.x(), self.eval_at(pt.x()))

  def mean(self):
    return self.make_new('center', y=DataOp.Center(self.y))

  def select_edge(self, rising=False):
    rising = int(rising)
    res = []
    for i in range(1, len(self.y)):
      if self.y[i - 1] != self.y[i] and self.y[i] == rising:
        res.append(i)
    return res

  def apply(self, func, ctx=None, **kwargs):
    tmp = dict(globals())
    if ctx is None:
      _, ctx = extract_module_locals(1)
    tmp.update(ctx)
    tmp.update(kwargs)
    tmp.update(DataOp.__dict__)
    res = eval(func, tmp, self.__dict__)
    return self.make_new(str(func), y=res)

  def make_new(self, op_name, **kwargs):
    return DataSet(orig_dataset=self, name='%s(%s)' % (op_name, self.name), **kwargs)

  def __getitem__(self, x):
    if not isinstance(x, slice):
      assert False
    else:
      assert x.step is None or x.step == 1

      print(x)
      print(type(self.x))
      print(type(self.y))
      nx = self.x[x]
      ny = self.y[x]
      res = self.make_new('select_%s' % x, x=nx, y=ny)
      return res

  def to_file(self, filename, typ='float32'):
    DataOp.ToFile(self.y, filename, typ)

  @staticmethod
  def FromImpulse(at, eps=1e-4, maxy=1e5, orig_dataset=None):
    yl = []
    xl = []
    for x in at:
      yl.extend([0, maxy, 0])
      xl.extend([x, x, x])
    return DataSet(yl, x=xl, orig_dataset=orig_dataset)

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
    return DataSet(yl, x=xl)

  @property
  def mod(self):
    #TODO (ease with dataop)
    pass



class DynamicDataset(DataSet):
  def __init__(self, y, **kwargs):
    super().__init__(y, **kwargs)

  def update_data(self, nx, ny):
    n = len(nx)
    assert n == len(ny)
    self.x = np.append(self.x, nx)
    self.y = np.append(self.y, ny)
    self.sig_replot_cb()


class DataOp:
  @staticmethod
  def MakeOrthogonal(a, b, debug=0):
    k = np.dot(a,b) / np.dot(b,b)
    res = a-k*b
    if debug:
      return res,k
    return res

  @staticmethod
  def GetBPSKBits(bits):
    bits = np.array(bits)
    soft_trans = np.abs(np.diff(np.unwrap(bits))) % (2 * np.pi) / np.pi
    hard_trans = soft_trans >= 0.5
    return np.array([0] + list(np.cumsum(hard_trans))) % 2

  @staticmethod
  def Power(data):
    return np.linalg.norm(data)**2 / len(data)

  @staticmethod
  def Power_Db(data):
    return 10*np.log10(DataOp.Power(data))

  @staticmethod
  def Hysteresis2(data, ratio=None, low=None, high=None, avg=None):
    if ratio is not None:
      low = data.min()
      high = data.max()
      diff = (high - low) * (1 - ratio)
      low += diff
      high -= diff
    assert low is not None
    assert high is not None

    y = data.copy()
    if avg is not None:
      b = [1. / avg for i in range(avg)]
      y = signal.lfilter(b, [1], y)

    ny = []
    cur = 0
    ny.append(int(y[0] > high))
    for i in range(1, len(y)):
      if y[i] > y[i - 1] and y[i] > low:
        cur = 1
      if y[i] < y[i - 1] and y[i] < high:
        cur = 0
      ny.append(cur)
    ny = np.array(ny)
    return ny

  @staticmethod
  def Hysteresis(data, ratio=0.8, low=None, high=None, avg=None):

    if ratio is not None:
      low = data.min()
      high = data.max()
      diff = (high - low) * (1 - ratio)
      low += diff
      high -= diff
    assert low is not None
    assert high is not None

    y = data.copy()
    if avg is not None:
      b = [1. / avg for i in range(avg)]
      y = signal.lfilter(b, [1], y)

    ny = []
    cur = 0
    for i in range(len(y)):
      if y[i] > high:
        cur = 1
      if y[i] < low:
        cur = 0
      ny.append(cur)
    ny = np.array(ny)
    return ny

  @staticmethod
  def Remap(data, low, high):
    cur_low = data.min()
    cur_high = data.max()
    ratio = (high - low) / (cur_high - cur_low)
    return (data - cur_low) * ratio + low

  @staticmethod
  def RemapBinary(data, low, high):
    centers, _ = cluster.vq.kmeans(data, 2)
    cur_low, cur_high = min(centers), max(centers)
    data = np.clip(data, cur_low, cur_high)
    ratio = (high - low) / (cur_high - cur_low)
    return (data - cur_low) * ratio + low

  @staticmethod
  def RemapBinaryClosest(data, lowv=None, highv=None):
    if lowv is None:
      centers, _ = cluster.vq.kmeans(data, 2)
      lowv, highv = min(centers), max(centers)
    data = (abs(data-lowv) > abs(data-highv)).astype(int)
    return data

  CenterUnit = lambda x: DataOp.Remap(x, -0.5, 0.5)

  @staticmethod
  def Center(data):
    return data - np.mean(data)

  @staticmethod
  def Orthogonalize(data, base_vec):
    v_ab = np.inner(data, base_vec)
    v_aa = np.inner(base_vec, base_vec)
    res = data - base_vec * (v_ab / v_aa)
    return res

  @staticmethod
  def HighPass(data, n=50, wn=0.3):
    fx_b, fx_a = signal.iirfilter(n, wn, btype='highpass')
    return signal.lfilter(fx_b, fx_a, data)

  @staticmethod
  def LowPass(data, n=10, wn=0.3):
    fx_b, fx_a = signal.iirfilter(n, wn, btype='lowpass')
    return signal.lfilter(fx_b, fx_a, data)

  @staticmethod
  def BinaryTransitionPos(data, double_edge=1):
    a = data[:-1]
    b = data[1:]
    res = a != b
    if not double_edge: res = res & (a < b)
    return np.where(res)[0] + 1

  @staticmethod
  def OOKMessages(data, debug=False):
    # assuming data starts with zero
    transitions = DataOp.BinaryTransitionPos(data)
    durations = DataOp.InvCumSum(transitions)
    zeros, ones = durations[::2], durations[1::2]

    bit0, bit1 = cluster.vq.kmeans(DataOp.Remove1DOutliers(ones), 2)[0]
    if bit0>bit1: bit0,bit1=bit1,bit0


    clk_zero = np.median(zeros)
    stop_ids = np.where(zeros > clk_zero * 5)[0]
    msgs = ListTools.SplitIntoBlocks(ones, stop_ids)
    res=[]
    for msg in msgs:
      res.append(DataOp.RemapBinaryClosest(msg, bit0, bit1))

    if debug:
      return Attributize(locals())
    return res

  @staticmethod
  def Remove1DOutliers(data, low_q=0.05):
    low,high=stats.mstats.mquantiles(data, [low_q, 1-low_q])
    data = data[(data>low) & (data<high)]
    return data



  @staticmethod
  def InvCumSum(data):
    return signal.lfilter([1., -1.], [1.], data)

  @staticmethod
  def ToFile(data, filename, typ='float32'):
    np.asarray(data, typ).tofile(filename)

  @staticmethod
  def FromFile(filename, typ='float32'):
    return np.fromfile(filename, typ)

  @staticmethod
  def GnuRadioOp(data_in, target, params={}):
    params = dict(params)

    runner = GnuRadioFile(target)
    print(runner.variables)
    in_param_name, in_type = runner.get_input_data()
    out_param_name, out_type = runner.get_output_data()
    with tempfile.NamedTemporaryFile(mode='rb',
                                     prefix='grc_output_',
                                     delete=False) as out_file, tempfile.NamedTemporaryFile(
                                         mode='wb',
                                         prefix='grc_input_',
                                         delete=False) as in_file:
      print('Got ', in_file.name, out_file.name)
      if not in_param_name in params:
        params[in_param_name] = '"%s"' % in_file.name
        DataOp.ToFile(data_in, in_file.name, typ=in_type)
      params[out_param_name] = '"%s"' % out_file.name
      runner.run(params)

      out_data = DataOp.FromFile(out_file.name, typ=out_type)
      return out_data

  @staticmethod
  def ClockRecovery(data, sps_guess, sps_dev=0.1, nsps=None, final_drift=0.1, nphase=20):

    sps_min, sps_max = sps_guess * (1 - sps_dev), sps_guess * (1 + sps_dev)

    if nsps is None:
      # a = clock precision =  (sps_max - sps_min) / (nsps - 1) / 2
      # n clock cycles considered -> nmax = N / sps_min
      # clock_drift = a * nmax
      # max_drift = final_drift * sps

      # clock_drift     < max_drift
      # a * N / sps_min < final_drift * sps_min
      # (sps_max - sps_min) / 2 / final_drift * N / sps_min ^2 + 1 < nsps
      nsps = math.ceil(sps_dev * sps_guess * len(data) / final_drift / (sps_min**2) + 1)

    fast = True
    if fast:
      x = swig.opa_or_swig

      bounds = swig.opa_or_swig.GridSearchBounds()
      tmp = np.arange(sps_min, sps_max, (sps_max - sps_min) / nsps)
      assert nsps * nphase < 1e6, f'{nsps} {nphase} {sps_guess} {len(data)} {sps_min} {sps_max}'
      bounds.bounds.append(np.linspace(sps_min, sps_max, nsps))
      bounds.bounds.append(np.linspace(0., 1., nphase))

      data_tmp = data.astype(np.float32)
      solver = swig.opa_or_swig.DspBinaryMatcher(data_tmp)
      res = swig.opa_or_swig.DoGridSearch(bounds, solver.get_grid_search_func())
      return res.state
    else:
      sps_range = slice(sps_min, sps_max, (sps_max - sps_min) / nsps)
      phase_range = slice(0, 1, 1 / nphase)

      cumsum = np.cumsum(data)

      def get_correl(l, r):
        l = round(l) - 1
        r = round(r)
        r = min(r, len(cumsum) - 1)
        n = r - (l - 1)
        v = cumsum[r]
        if l > 0:
          v -= cumsum[l - 1]
        v = abs(v)
        # want to maximize this v/n
        return -v / n

      def cost_func(state):
        sps, phase = state
        phase = sps * phase
        v = get_correl(0, phase)
        pos = phase
        while pos < len(data):
          v += get_correl(pos, pos + sps)
          pos += sps
        return v

      res = optimize.brute(cost_func, (sps_range, phase_range), finish=None)
      return res

  @staticmethod
  def PulseToClock(data):
    val = -0.5
    res = np.ones_like(data) * val
    transitions = DataOp.BinaryTransitionPos(data, double_edge=0)
    for i in range(1, len(transitions)):
      val *= -1
      res[transitions[i-1]:transitions[i]] = val
    return res

  @staticmethod
  def ClockPos(period, phase, start_pos, end_pos):
    pos = start_pos + phase * period
    return np.arange(pos, end_pos, period)

  @staticmethod
  def GetRange(data, a, b):
    a = max(a, 0)
    b = min(b, len(data))
    return v[a:b]

  @staticmethod
  def SampleAt(data, pos, nmean=0):
    tmpy = self.sample_range(data, pos - nmean, pos + nmean + 1)
    return np.mean(tmpy)

  @staticmethod
  def ExtractAtPos(data, pos_list, nmean=0):
    if nmean == 0:
      return data[pos_list.astype(int)]

    res = []
    for pos in pos_list:
      res.append(DataOp.ExtractAtPos(pos, nmean=nmean))
    return np.array(res)

  @staticmethod
  def RetrieveMessages(data):

    power = np.log(DataOp.Power(DataOp.HighPass(data, 20, 0.4)) + 1e-20)
    message_level, transition_level = stats.mstats.mquantiles(power, [0.9, 0.999])

    message_ids = np.where(abs(power - message_level) < abs(power - transition_level))[0]

    message_inter = Intervals.FromIndices(message_ids, merge_dist=100)
    message_inter = message_inter.expand(40)
    message_inter = message_inter.filter(lambda x : len(x)>300)
    msg_list = message_inter.split_data(data)

    return msg_list

  @staticmethod
  def CleanBinaryDataSimple(data):
    cleaned_data = DataOp.CenterUnit(DataOp.Hysteresis(data))
    return cleaned_data

  @staticmethod
  def CleanBinaryData(data):
    cleaned_data = DataOp.CenterUnit(DataOp.Hysteresis(DataOp.RemapBinary(data, -0.5, 0.5)))
    return cleaned_data

  @staticmethod
  def BinaryClockFirstGuess(data, double_edge=1):
    pos = DataOp.BinaryTransitionPos(data, double_edge=double_edge)
    dx = DataOp.InvCumSum(pos)
    return stats.mstats.mquantiles(dx, 0.3)[0]
    print(dx)
    centers, _ = cluster.vq.kmeans(dx, 2)
    print('GOT CENTERS, ',centers)
    sps = min(centers)
    return sps

  @staticmethod
  def GetBinaryClock(data, clock_guess=None):
    if clock_guess is None:
      # requires cleaned data
      clock_guess = DataOp.BinaryClockFirstGuess(data)
    glog.info('clock guess %s', clock_guess)
    sps, phase = DataOp.ClockRecovery(data, clock_guess, sps_dev=0.01, final_drift=0.01, nphase=50)
    return sps, phase

  @staticmethod
  def RetrieveBinaryData(data, clock_guess=None, clean=1):
    # TODO: not only binary
    if clean: cleaned_data = DataOp.CleanBinaryData(data)
    else: cleaned_data = data
    sps, phase = DataOp.GetBinaryClock(cleaned_data, clock_guess)
    sample_phase = (phase + 0.5) % 1
    pos_list = DataOp.ClockPos(sps, sample_phase, 0, len(cleaned_data))

    samples = DataOp.ExtractAtPos(cleaned_data, pos_list)
    binary_data = DataOp.Remap(samples, 0, 1).astype(bool)
    return binary_data, pos_list

  @staticmethod
  def FreqWaterfall(data, window_size):
    hann = signal.hanning(window_size)
    fall_data = []
    data = Format(data).modpad(window_size, 0).v
    for i in range(0, len(data), window_size):
      cur = data[i:i + window_size]
      cur = np.multiply(cur, hann)

      now = np.abs(fftpack.fft(cur))
      now = 20 * np.log(now)
      fall_data.append(now)

    fall_data = np.array(fall_data)
    return fall_data

  @staticmethod
  def FreqWaterfall_DS(ds, window_size):
    hann = signal.hanning(window_size)
    fall_data = []
    data = Format(ds.y).modpad(window_size, 0).v
    axis_t = []

    samp_rate = ds.samp_rate
    axis_f = np.linspace(-0.5 * samp_rate, 0.5 * samp_rate, window_size)
    print('gogo', ds.samp_rate)
    for i in range(0, len(data), window_size):
      axis_t.append(i / ds.samp_rate)
      cur = data[i:i + window_size]
      cur = np.multiply(cur, hann)

      now = np.abs(fftpack.fft(cur))
      now = 20 * np.log(now / window_size)
      fall_data.append(now)

    fall_data = np.array(fall_data)
    fall_data = np.roll(fall_data, window_size // 2, axis=1)
    return Dataset2D(fall_data, axis_t, axis_f)

  @staticmethod
  def FreqSpectrum(data, window_size, agg='max'):
    res = DataOp.FreqWaterfall(data, window_size)
    if agg == 'max':
      res = np.amax(res, axis=0)
    elif agg == 'mean':
      res = np.mean(res, axis=0)
    return res

  @staticmethod
  def SolveMatchedFilter_Acquisition(data, filter):
    nfilter = len(filter)
    pre_data = data[:nfilter*11]
    correlation = np.abs(signal.correlate(pre_data, filter, mode='valid'))
    correl_level = np.max(correlation)

    cleaned = DataOp.CleanBinaryDataSimple(correlation)
    clock = DataOp.PulseToClock(cleaned)
    transitions = DataOp.BinaryTransitionPos(clock)
    diffs = np.diff(transitions)
    period = np.mean(diffs)
    return transitions[0] / period, period, np.var(diffs)


  @staticmethod
  def SolveMatchedFilter(data, filter, min_jitter=1):
    phase, period, var = DataOp.SolveMatchedFilter_Acquisition(data, filter)
    jitter = math.ceil(max(var*2, min_jitter))
    vals = []
    sample_times = []

    last_pos = None
    nfilter = len(filter)
    pos = int(phase * period)
    print(nfilter, pos, period)
    while pos+nfilter < len(data):
      rd =data[pos-jitter:pos+jitter+nfilter]

      correlation = signal.correlate(rd, filter, mode='valid')
      best_pos = np.argmax(np.abs(correlation))
      vals.append(correlation[best_pos])
      best_pos += + pos - jitter
      sample_times.append(best_pos)

      if last_pos is not None:
        period = 0.9*period + 0.1 * (best_pos - last_pos)

      last_pos = best_pos
      pos = best_pos + int(period)
    return np.array(vals), np.array(sample_times)



def linearize(x, a, b, na, nb):
  ratio = (nb - na) / (b - a)
  return (x - a) * ratio + na

def linearize_clamp(x, a, b, na, nb):
  tmp = linearize(x, a, b, na, nb)
  return np.clip(tmp, na, nb)
class FFTHelper:
  def __init__(self):
    from vispy.color import get_colormap
    self.cmap = get_colormap('viridis')

  def map0(self, f, maxv=0, minv=-60, quant=None):
    f = np.ravel(f)
    if quant is not None:
      minv,maxv = mquantiles(f, [quant, 1-quant])

    return linearize_clamp(f, minv, maxv, 0, 1)

  def map(self, f, **kwargs):
    shape = np.shape(f)
    cf =  self.cmap.map(self.map0(f, **kwargs))
    res =  np.reshape(cf, shape + (4,))
    return res

g_fft_helper = FFTHelper()


def compute_normed_correl(haystack, needle):
  needle = needle - np.mean(needle)
  needle = needle / np.linalg.norm(needle)
  haystack = haystack - np.mean(haystack)
  haystack = haystack / np.linalg.norm(haystack) * len(haystack) / len(needle)

  return signal.correlate(haystack, needle, mode='valid')
