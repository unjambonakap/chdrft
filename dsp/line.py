#!/usr/bin/env python

from chdrft.utils.misc import cwdpath, Attributize, kv_to_dict
from chdrft.utils.arg_gram import LazyConf
import numpy as np
from scipy import signal
from chdrft.cmds import Cmds
import os
import math
from chdrft.display.utils import *
import re
import itertools
import fnmatch


class PlotDb:

  def __init__(self, helper=None):
    self.plot_map = {}
    self.plots = []
    self.gconf = LazyConf()
    self.helper = helper

  def filter_plots(self, keys):
    res=set()
    for key in keys:
      e=re.compile(fnmatch.translate(key))
      for k,v in self.plot_map.items():
        if e.match(k):
          res.add(v.id_name)
    self.plots =  list([self.plot_map[x] for x in res])

  def find_plot(self, name):
    if name not in self.plot_map:
      glog.warning('Could not find plot %s, available names are %s', name,
                   self.plot_map.keys())
      return None
    return self.plot_map[name]

  def set_preferred_name(self, key, name):
    plot = self.find_plot(key)
    assert plot is not None, 'Could not find plot %s %s' % (key, name)
    assert self.maybe_add_plot(name, plot), 'name already taken for %s %s' % (
        key, name)
    plot.ds.name = name

  def modify_plot(self, plot_conf):
    ds = plot_conf.ds
    print('OFFSET > >', plot_conf.offset)
    if plot_conf.offset:
      ds.y = ds.y + plot_conf.offset
    if plot_conf.reset:
      assert plot_conf.samp_rate
      ds = ds.reset_x(plot_conf.samp_rate)

    plot_conf.ds = ds

  def add_plot(self, plot_conf):
    self.plots.append(plot_conf)
    plot_conf.aliases = []
    self.modify_plot(plot_conf)

    self.maybe_add_plot(plot_conf.ds.name, plot_conf)

    file_cnd = [plot_conf.file_pos]
    if plot_conf.file_desc:
      file_cnd.append(plot_conf.file_desc)

    if plot_conf.prefix:
      file_cnd.append(plot_conf.prefix)

    plot_cnd = [plot_conf.plot_pos, plot_conf.ds.name]
    join_cnd = '_.'
    for file_str, plot_str, join_char in itertools.product(file_cnd, plot_cnd,
                                                           join_cnd):
      file_str = str(file_str)
      plot_str = str(plot_str)
      if file_str and plot_str:
        key = join_char.join([file_str, plot_str])
        if file_str == '-1':
          key = plot_str

        self.maybe_add_plot(key, plot_conf)

    assert len(plot_conf.aliases)>0, 'Could not add plot successfully'
    plot_conf.id_name = plot_conf.aliases[0]

  def maybe_add_plot(self, name, plot_conf):
    if name not in self.plot_map:
      self.plot_map[name] = plot_conf
      glog.info('Adding dataset with name=%s', name)
      plot_conf.aliases.append(name)
      return True
    return False

  def load_file_list(self, file_list):
    if len(file_list)==1:
      self.load_input(file_list[0], '')
    else:
      for count, inp in enumerate(file_list):
        self.load_input(inp, '', count)

  def load_input(self, desc, prefix='', file_pos=-1):
    file_re = re.compile(
        '^((?P<format>[^:,]+):)?(?P<filename>[^,]+)(,(?P<params>.+))?')
    m = file_re.match(desc)
    assert m, 'Bad file format %s' % desc

    tb = m.groupdict()
    filename = tb['filename']
    fmt = tb.get('format', 'csv')

    conf = LazyConf()
    conf.all._merge(self.gconf)
    conf._merge(self.gconf)
    if tb['params']:
      print(tb['params'])
      lc=LazyConf.ParseFromString(tb['params'])
      conf._merge(lc)
      conf.all._merge(lc)
    conf.all.file_prefix = prefix
    conf.all.file_pos = file_pos
    fmt=conf.get('format', fmt)


    data = DataFile(filename, typ=fmt, samp_rate=conf.samp_rate, **conf.get('file_params', {}))


    if not conf.type:
      default_typ = 'xy'
      if data.size == 1:
        default_typ = 'y'
      conf.type = default_typ

    # some simple scheme
    if conf.type == 'xy' or conf.type == 'y':
      x = None
      xpos = conf.get('xpos', -1)
      if conf.type == 'xy':
        if xpos == -1: xpos = 0
        x = data.col(xpos)

      ylist = []
      if conf.ypos:
        if isinstance(conf.ypos, str): ylist = [(conf.ypos, data.col(conf.ypos))]
        else: ylist = [(x, data.col(x)) for x in conf.ypos]
      else:
        for i in range(data.size):
          if conf.get('skip', -1) == i: continue
          if i != xpos:
            ylist.append((data.col_name(i), data.col(i)))


      for pos, (yname, y) in enumerate(ylist):
        plot_conf = conf[pos]
        plot_conf._merge(conf.all)
        plot_conf.plot_pos = pos
        basename='y%s%d_%s' % (prefix, pos, yname)

        ds = DataSet(y,
                     x,
                     samp_rate=plot_conf.samp_rate,
                     name=basename,)

        if plot_conf.apply:
          apply_list=to_list(plot_conf.apply, sep='#')
          for entry in apply_list:
            ny=eval(entry, globals(), dict(x=x, y=y, ds=ds))
            ds_name = '%s_%s'%(basename, entry)

            if isinstance(ny, DataSet): new_ds = ny
            else: new_ds=DataSet(ny, x, samp_rate=plot_conf.samp_rate, name=name)
            print(len(new_ds.y))
            print(len(ds.y))
            new_conf = LazyConf(plot_conf)
            new_conf.ds = new_ds
            self.add_plot(new_conf)
        else:
          plot_conf.ds = ds
          self.add_plot(plot_conf)
    else:
      assert 0

  def plot_ds_list(self, ds_list):
    fig = self.helper.create_plot()
    for ds in ds_list:
      fig.add_plot(ds)
    return fig


def extract_bitstream_2wire(clk, data, ratio=0.9, rising=1):
  clk = clk.hysteresis(ratio=ratio)
  data = data.hysteresis(ratio=ratio)
  edges = clk.select_edge(rising=rising)
  bitstream = list(data.y[edges])
  return [edges, bitstream]


def find_stop_bits(clk, data, want_stop=True, want_start=True):
  clk = clk.hysteresis2(ratio=0.9)
  data = data.hysteresis2(ratio=0.9)
  n = len(clk.y)
  res = []
  for i in range(1, n):
    if want_stop and (clk.y[i - 1] == 1 and clk.y[i] == 1) and (
        data.y[i - 1] == 0 and data.y[i] == 1):
      res.append(i)
    if want_start and (clk.y[i - 1] == 1 and clk.y[i] == 1) and (
        data.y[i - 1] == 1 and data.y[i] == 0):
      res.append(i)
  return res


def extract_i2c_comm(scl, sda, ignore_stopbits=False):
  if ignore_stopbits:
    cur = extract_bitstream_2wire(scl, sda)
    return cur
  else:
    stopbits = find_stop_bits(scl, sda)
    stopbits.insert(0, -1)
    stopbits.append(scl.n + 1)

    res = []
    for i in range(len(stopbits) - 1):
      interval = [stopbits[i] + 1, stopbits[i + 1]]
      edges, cur = extract_bitstream_2wire(
          scl.extract_by_idx(interval), sda.extract_by_idx(interval))
      if i == 0:
        cur = cur[1:]
      if i != len(stopbits) - 2:
        cur = cur[:-1]
      if len(cur) % 8 != 0:
        glog.error('fail at %d, %s %s', i, cur, interval)
      res.append((scl.x[interval[0]], cur))
    return res


class StatsHelper:

  def __init__(self, ds):
    self.ds = ds.mean()

  def num_edges(self):
    v = self.ds.y
    cnt = 0
    for i in range(1, self.ds.n):
      if v[i] * v[i - 1] < 0 or v[i] == 0:
        cnt += 1
    glog.info('found %d edges', cnt)
    return cnt


def compute_stats(ds):
  stats = Attributize()
  stats.freq = 123
  return stats


def compute_i2c_data(data_list):
  data = [Attributize(data=x, stats=StatsHelper(x)) for x in data_list]

  if data[0].stats.num_edges() < data[1].stats.num_edges():
    data[0], data[1] = data[1], data[0]
  scl, sda = data
  return extract_i2c_comm(scl.data, sda.data, ignore_stopbits=True)
