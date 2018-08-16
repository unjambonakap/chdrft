#!/usr/bin/env python
import time
import zerorpc
from contextlib import ExitStack
import binascii
import csv
from queue import Queue
import threading

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import glog
from chdrft.utils.arg_gram import LazyConf
import subprocess as sp

global flags, cache
flags = None
cache = None


class Callback(LazyConf):

  def __init__(self, data):
    super().__init__()
    self.set_from_string(data)
    self.code_cache = None

  def __call__(self):
    if self.typ == 'shell':
      return sp.check_output(self.cmd, shell=True)
    elif self.typ == 'python':
      if self.code_cache is None:
        if self.cmd: self.code_cache = compile(self.cmd, '<string>', 'exec')
        else: self.code_cache = compile(open(self.filename, 'r').read(), self.filename, 'exec')
      if self.args: ns = dict(DATA=self.args)
      eval(self.code_cache, globals(), ns)
      return ns['res']
    else:
      assert 0, self


def args(parser):
  parser.add_argument('--port', type=int, default=12345)
  parser.add_argument('--outfile', type=str)
  parser.add_argument('--daq_period_s', type=float, required=True)
  parser.add_argument('--test', action='store_true')
  parser.add_argument('--fullcpu', action='store_true')
  parser.add_argument('--noserver', action='store_true')
  parser.add_argument('--cb', type=Callback, required=True)


class DAQ(ExitStack):

  def __init__(self, period_sec=None, outfile=None, push_cb=None, want_queue=False,
               time_speedup=1.):
    assert period_sec is not None
    super().__init__()
    self.outfile = outfile
    self.period_sec = period_sec
    self.push_cbs = []
    self.q = None
    self.stop = False
    self.thread = None
    self.time_speedup = time_speedup

    if push_cb is not None: self.push_cbs.append(push_cb)
    if want_queue:
      self.q = Queue()
      self.push_cbs.append(lambda *args: self.q.put(args))

  def __enter__(self):
    super().__enter__()
    if self.outfile is not None:
      fil = open(self.outfile, 'w')
      self.enter_context(fil)
      writer = csv.writer(fil, delimiter=',', quotechar='|')
      self.push_cbs.append(lambda *args: writer.writerow(args))
    return self

  def thread_stop(self):
    if self.stop: return
    self.stop = True
    self.thread.join()

  def pull_queue_data(self):
    res = []
    try:
      while True:
        res.append(self.q.get_nowait())
    except:
      pass
    return res

  def start(self, cb, fullcpu=True, threaded=False):
    self.stop = False
    run_func = self.go_cpu if fullcpu else self.go
    if not threaded:
      run_func(cb)
    else:
      self.thread = threading.Thread(target=run_func, args=(cb,))
      self.thread.start()
      self.callback(self.thread_stop)

  def go(self, cb):
    startt = self.get_time()
    curtime = startt
    want_sleep = self.period_sec
    while not self.stop:
      res = cb()
      self.dump(curtime - startt, res)
      time.sleep(want_sleep)

      now = self.get_time()
      slept = now - curtime
      want_sleep = max(self.period_sec / 2, 2 * self.period_sec - slept)
      curtime = now

  def go_cpu(self, cb):
    startt = self.get_time()
    curtime = startt
    while not self.stop:
      res = cb()
      #print(curtime, self.period_sec)
      self.dump(curtime - startt, res)

      while True:
        now = self.get_time()
        if now - curtime >= self.period_sec:
          curtime = now
          break

  def get_time(self):
    return time.time() * self.time_speedup

  def dump(self, *args):
    glog.info('Dump: %s', args)
    for cb in self.push_cbs:
      cb(*args)


class MultiClientServer:

  def __init__(self):
    self.id = 0
    self.ids_to_pos = {}
    self.items = []

  def __call__(self, *data):
    self.items.append(data)

  def register_new(self):
    nid = self.id
    self.id += 1
    self.ids_to_pos[nid] = 0
    return nid

  def pull(self, id):
    have = len(self.items)
    curpos = self.ids_to_pos[id]
    self.ids_to_pos[id] = have
    return self.items[curpos:have]

def run_daq_server(outfile=None, period_sec=None, cb=None, fullcpu=None, threaded=None, port=None):
  daq = DAQ(outfile=outfile, period_sec=period_sec)
  with daq:
    server = MultiClientServer()
    daq.push_cbs.append(server)
    daq.start(cb, fullcpu=fullcpu, threaded=True)

    s = zerorpc.Server(server)
    s.bind('tcp://0.0.0.0:%s' % port)
    print('Starting server')
    s.run()


def main():
  if flags.noserver:
    daq = DAQ(outfile=flags.outfile, period_sec=flags.daq_period_s)
    with daq:
      if flags.noserver:
        daq.start(flags.cb, fullcpu=flags.fullcpu, threaded=False)
  else:
    run_daq_server(outfile=flags.outfile, period_sec=flags.daq_period_s, cb=flags.cb, fullcpu=flags.fullcpu, threaded=True, port=flags.port)

app()
