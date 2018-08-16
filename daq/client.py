#!/usr/bin/env python

import zerorpc
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import glog
from chdrft.display.utils import DynamicDataset
from chdrft.utils.arg_gram import LazyConf
from chdrft.display.ui import GraphHelper
from threading import Thread
import time
import numpy as np

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(query).add(graph)
  ActionHandler.Prepare(parser, clist.lst)
  parser.add_argument('--host', type=str, default='localhost')
  parser.add_argument('--port', type=int, default=12345)
  parser.add_argument('--desc', type=LazyConf.ParseFromString)
  parser.add_argument('--period_sec', type=float)

class DAQClient:

  def __init__(self, host=None, port=None, **kwargs):
    if host is None: host = app.flags.host
    if port is None: port = app.flags.port
    self.host = host
    self.port = port
    self.c = None
    self.cbs = []


  def init_client(self):
    self.c = zerorpc.Client()
    self.c.connect("tcp://%s:%d" % (self.host, self.port))
    glog.info('Connecting to %s:%d', self.host, self.port)
    self.id = self.c.register_new()

  def get_new_data(self):
    if self.c is None: self.init_client()

    nd =  self.c.pull(self.id)
    self.handle(nd)
    for cb in self.cbs: cb()

  def handle(self, nd):
    print(nd)

class DAQGraphClient(DAQClient):
  def __init__(self, ghelper, desc, **kwargs):
    super().__init__(**kwargs)
    self.ghelper = ghelper
    self.desc = desc
    self.is_init = False
    self.figs = []
    self.plot = None
    self.id = 0

  def init(self, nd):
    self.is_init = True
    self.plot = self.ghelper.create_plot()
    self.dds_list = []
    print(self.desc.plots)
    for x,y in self.desc.plots:
      dds = DynamicDataset(nd[:,y], x=nd[:,x], name='daq(%s,%s)'%(x,y))
      self.dds_list.append(dds)
      fig = self.plot.add_plot(dds)
      self.figs.append(fig)



  def handle(self, nd):
    nd = np.array(nd, np.float)
    if not self.is_init:
      self.init(nd)
    else:
      self.update(nd)

  def update(self, nd):
    if nd.shape[0] == 0: return
    for (x,y), dds in zip(self.desc.plots, self.dds_list):
      dds.update_data(nd[:,x], nd[:,y])


class TimedAction(Thread):
  def __init__(self, period, cb):
    super().__init__()
    self.period = period
    self.cb = cb

  def run(self):
    while True:
      self.cb()
      time.sleep(self.period)


def query(ctx):
  x = DAQClient(flags.host, flags.port)
  print(x.get_new_data());

def start_timer(ghelper, func, period_sec):
  from pyqtgraph.Qt import QtCore, USE_PYQT5
  timer = QtCore.QTimer()
  func()
  timer.timeout.connect(func)
  timer.start(int(1000*period_sec))
  ghelper.register_cleanup_cb(lambda: timer.stop())
  return timer

def graph(ctx):
  print(flags.desc)
  ghelper = GraphHelper(create_kernel=False)
  x = DAQGraphClient(ghelper, flags.desc, host=flags.host, port=flags.port)
  if flags.period_sec:
    start_timer(ghelper, x.get_new_data, flags.period_sec)
  ghelper.run()

def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
