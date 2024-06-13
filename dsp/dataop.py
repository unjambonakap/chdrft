#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
import glog
import math, sys, os
import numpy as np
from chdrft.utils.types import *
from enum import Enum
from chdrft.utils.path import FileFormatHelper
from chdrft.utils.fmt import Format
import chdrft.display.base
from IPython.utils.frame import extract_module_locals
from chdrft.utils.swig import swig
from asq.initiators import query as asq_query
from chdrft.dsp.opa_gnuradio import GnuRadioFile
from chdrft.struct.base import Range1D, Range2D, Intervals, ListTools, Box
from chdrft.utils.colors import ColorPool
from chdrft.utils.misc import to_list, Attributize, proc_path
import chdrft.utils.misc as cmisc
from chdrft.utils.fmt import Format
from pyqtgraph.Qt import QtGui, QtCore
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
import scipy.ndimage as ndimage
import sys
import tempfile
from scipy.stats.mstats import mquantiles
import itertools
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import cv2

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)



def Power(data, alpha=0.1):
  data = np.abs(data)**2
  return signal.lfilter([alpha], [1, -(1 - alpha)], data)


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


def Remap(data, low, high):
  cur_low = data.min()
  cur_high = data.max()
  ratio = (high - low) / (cur_high - cur_low)
  return (data - cur_low) * ratio + low


def RemapBinary(data, low, high):
  centers, _ = cluster.vq.kmeans(data, 2)
  cur_low, cur_high = min(centers), max(centers)
  data = np.clip(data, cur_low, cur_high)
  ratio = (high - low) / (cur_high - cur_low)
  return (data - cur_low) * ratio + low


def RemapBinaryClosest(data, lowv=None, highv=None):
  if lowv is None:
    centers, _ = cluster.vq.kmeans(data, 2)
    lowv, highv = min(centers), max(centers)
  data = (abs(data - lowv) > abs(data - highv)).astype(int)
  return data

CenterUnit = lambda x: Remap(x, -0.5, 0.5)


def Center(data):
  return data - np.mean(data)


def Orthogonalize(data, base_vec):
  v_ab = np.inner(data, base_vec)
  v_aa = np.inner(base_vec, base_vec)
  res = data - base_vec * (v_ab / v_aa)
  return res


def HighPass(data, n=50, wn=0.3):
  fx_b, fx_a = signal.iirfilter(n, wn, btype='highpass')
  return signal.lfilter(fx_b, fx_a, data)


def LowPass(data, n=10, wn=0.3):
  fx_b, fx_a = signal.iirfilter(n, wn, btype='lowpass')
  zi = signal.lfilter_zi(fx_b, fx_a)

  res = signal.lfilter(fx_b, fx_a, data, zi=zi)[0]
  return res


def BinaryTransitionPos(data, double_edge=1):
  a = data[:-1]
  b = data[1:]
  res = a != b
  if not double_edge: res = res & (a < b)
  return np.where(res)[0] + 1


def OOKMessages(data, debug=False):
  # assuming data starts with zero
  transitions = BinaryTransitionPos(data)
  durations = InvCumSum(transitions)
  zeros, ones = durations[::2], durations[1::2]

  bit0, bit1 = cluster.vq.kmeans(Remove1DOutliers(ones), 2)[0]
  if bit0 > bit1: bit0, bit1 = bit1, bit0

  clk_zero = np.median(zeros)
  stop_ids = np.where(zeros > clk_zero * 5)[0]
  msgs = ListTools.SplitIntoBlocks(ones, stop_ids)
  res = []
  for msg in msgs:
    res.append(RemapBinaryClosest(msg, bit0, bit1))

  if debug:
    return Attributize(locals())
  return res


def Remove1DOutliers(data, low_q=0.05):
  low, high = stats.mstats.mquantiles(data, [low_q, 1 - low_q])
  data = data[(data > low) & (data < high)]
  return data


def InvCumSum(data):
  return signal.lfilter([1., -1.], [1.], data)


def ToFile(data, filename, typ='float32'):
  np.asarray(data, typ).tofile(filename)


def FromFile(filename, typ='float32'):
  return np.fromfile(filename, typ)


def GnuRadioOp(data_in, target, params={}):
  params = dict(params)

  runner = GnuRadioFile(target)
  print(runner.variables)
  in_param_name, in_type = runner.get_input_data()
  out_param_name, out_type = runner.get_output_data()
  with tempfile.NamedTemporaryFile(mode='rb', prefix='grc_output_',
                                    delete=False) as out_file, tempfile.NamedTemporaryFile(
                                        mode='wb', prefix='grc_input_', delete=False
                                    ) as in_file:
    print('Got ', in_file.name, out_file.name)
    if not in_param_name in params:
      params[in_param_name] = '"%s"' % in_file.name
      ToFile(data_in, in_file.name, typ=in_type)
    params[out_param_name] = '"%s"' % out_file.name
    runner.run(params)

    out_data = FromFile(out_file.name, typ=out_type)
    return out_data


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


def PulseToClock(data):
  val = -0.5
  res = np.ones_like(data) * val
  transitions = BinaryTransitionPos(data, double_edge=0)
  for i in range(1, len(transitions)):
    val *= -1
    res[transitions[i - 1]:transitions[i]] = val
  return res


def ClockPos(period, phase, start_pos, end_pos):
  pos = start_pos + phase * period
  return np.arange(pos, end_pos, period)


def GetRange(data, a, b):
  a = max(a, 0)
  b = min(b, len(data))
  return v[a:b]


def SampleAt(data, pos, nmean=0):
  tmpy = self.sample_range(data, pos - nmean, pos + nmean + 1)
  return np.mean(tmpy)


def ExtractAtPos(data, pos_list, nmean=0):
  if nmean == 0:
    return data[pos_list.astype(int)]

  res = []
  for pos in pos_list:
    res.append(ExtractAtPos(pos, nmean=nmean))
  return np.array(res)


def RetrieveMessages(data):

  power = np.log(Power(HighPass(data, 20, 0.4)) + 1e-20)
  message_level, transition_level = stats.mstats.mquantiles(power, [0.9, 0.999])

  message_ids = np.where(abs(power - message_level) < abs(power - transition_level))[0]

  message_inter = Intervals.FromIndices(message_ids, merge_dist=100)
  message_inter = message_inter.expand(40)
  message_inter = message_inter.filter(lambda x: len(x) > 300)
  msg_list = message_inter.split_data(data)

  return msg_list


def CleanBinaryDataSimple(data):
  cleaned_data = CenterUnit(Hysteresis(data))
  return cleaned_data


def CleanBinaryData(data):
  cleaned_data = CenterUnit(Hysteresis(RemapBinary(data, -0.5, 0.5)))
  return cleaned_data


def BinaryClockFirstGuess(data, double_edge=1):
  pos = BinaryTransitionPos(data, double_edge=double_edge)
  dx = InvCumSum(pos)
  return stats.mstats.mquantiles(dx, 0.3)[0]
  print(dx)
  centers, _ = cluster.vq.kmeans(dx, 2)
  print('GOT CENTERS, ', centers)
  sps = min(centers)
  return sps


def GetBinaryClock(data, clock_guess=None):
  if clock_guess is None:
    # requires cleaned data
    clock_guess = BinaryClockFirstGuess(data)
  glog.info('clock guess %s', clock_guess)
  sps, phase = ClockRecovery(data, clock_guess, sps_dev=0.01, final_drift=0.01, nphase=50)
  return sps, phase


def RetrieveBinaryData(data, clock_guess=None, clean=1):
  # TODO: not only binary
  if clean: cleaned_data = CleanBinaryData(data)
  else: cleaned_data = data
  sps, phase = GetBinaryClock(cleaned_data, clock_guess)
  sample_phase = (phase + 0.5) % 1
  pos_list = ClockPos(sps, sample_phase, 0, len(cleaned_data))

  samples = ExtractAtPos(cleaned_data, pos_list)
  binary_data = Remap(samples, 0, 1).astype(bool)
  return binary_data, pos_list


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


def FreqWaterfall_DS(ds, window_size):
  from chdrft.dsp.datafile import Sampler1D, Dataset2d
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
  return Dataset2d(fall_data, Sampler1D(0, samp_rate=ds.samp_rate / window_size), Range1D(-0.5 * samp_rate, 0.5*samp_rate))


def FreqSpectrum(data, window_size, agg='max'):
  res = FreqWaterfall(data, window_size)
  if agg == 'max':
    res = np.amax(res, axis=0)
  elif agg == 'mean':
    res = np.mean(res, axis=0)
  return res


def SolveMatchedFilter_Acquisition(data, filter):
  nfilter = len(filter)
  pre_data = data[:nfilter * 11]
  correlation = np.abs(signal.correlate(pre_data, filter, mode='valid'))
  correl_level = np.max(correlation)

  cleaned = CleanBinaryDataSimple(correlation)
  clock = PulseToClock(cleaned)
  transitions = BinaryTransitionPos(clock)
  diffs = np.diff(transitions)
  period = np.mean(diffs)
  return transitions[0] / period, period, np.var(diffs)



def SolveMatchedFilter(data, filter, min_jitter=1):
  phase, period, var = SolveMatchedFilter_Acquisition(data, filter)
  jitter = math.ceil(max(var * 2, min_jitter))
  vals = []
  sample_times = []

  last_pos = None
  nfilter = len(filter)
  pos = int(phase * period)
  print(nfilter, pos, period)
  while pos + nfilter < len(data):
    rd = data[pos - jitter:pos + jitter + nfilter]

    correlation = signal.correlate(rd, filter, mode='valid')
    best_pos = np.argmax(np.abs(correlation))
    vals.append(correlation[best_pos])
    best_pos += +pos - jitter
    sample_times.append(best_pos)

    if last_pos is not None:
      period = 0.9 * period + 0.1 * (best_pos - last_pos)

    last_pos = best_pos
    pos = best_pos + int(period)
  return np.array(vals), np.array(sample_times)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
