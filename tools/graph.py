#!/usr/bin/env python

from chdrft.utils.misc import cwdpath, Attributize, kv_to_dict
from chdrft.utils.arg_gram import LazyConf
import numpy as np
from scipy import signal
from chdrft.cmds import Cmds
import os
import math
from chdrft.display.utils import *
from chdrft.display.ui import *
import chdrft.dsp.line as dsp_line
import re


def graph(flags):
  helper = GraphHelper(create_kernel=flags.create_kernel)

  to_plot = dict()
  gconf = LazyConf()
  gconf.samp_rate = None
  gconf._merge(LazyConf.ParseFromString(flags.params))
  db = dsp_line.PlotDb(helper=helper)
  db.gconf = gconf
  db.load_file_list(flags.input)

  glog.info('Graph input files >> %s', flags.input)
  if gconf.select:
    selected_keys = gconf.select.split('-')
    db.filter_plots(selected_keys)

  if flags.mode == 'align':
    assert len(db.plots) == 2, db.plots
    data = []
    for plot in db.plots:
      plot.ds = plot.ds.mean().reset_x(gconf.samp_rate)
    p = [x.ds for x in db.plots]
    correl = signal.fftconvolve(p[0].get_y(), p[1].get_y()[::-1])
    shiftv = (np.argmax(correl) - p[1].n) / p[0].samp_rate

    yoff = 10
    p[1] = p[1].shift([shiftv, yoff])

    plot = helper.create_plot()
    for x in p:
      plot.add_plot(PlotEntry(x))
  elif flags.mode == '2wire':
    clk = db.find_plot(gconf.clk)
    data = db.find_plot(gconf.data)
  elif flags.mode == 'plot':
    figs = []
    if not gconf.plots:
      figs.append(db.plots)
    else:
      fig_list = gconf.plots.split('/')
      for fig_desc in fig_list:
        lst = []
        for plot_desc in fig_desc.split('-'):
          lst.append(db.find_plot(plot_desc))
        figs.append(lst)

    for fig_desc in figs:
      db.plot_ds_list([x.ds for x in fig_desc])
  else:
    assert 0, 'unknown mode %s' % flags.mode

  helper.run()
  glog.info('Done graph')

