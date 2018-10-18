#!/usr/bin/env python

from . import tube
import subprocess as sp
import os
import tempfile
import time
import errno
from chdrft.utils import proc
from chdrft.utils.misc import failsafe
from chdrft.utils.misc import chdrft_executor
import threading
import contextlib
import traceback as tb
import glog
import fcntl


class Fifo(tube.Tube):

  def __init__(self, read_fifo=None, write_fifo=None):
    super().__init__()
    self.read_fifo = read_fifo
    self.write_fifo = write_fifo
    self.rfile = None
    self.wfile = None

  def _enter(self):
    try:
      t_w = None
      t_r = None

      from concurrent.futures import ThreadPoolExecutor
      with ThreadPoolExecutor(4) as executor:
        if self.write_fifo:

          def t1():
            glog.info('opening  write %s', self.write_fifo)
            a = open(self.write_fifo, 'wb')
            #w = os.open( 'my_fifo', os.O_WRONLY )
            glog.info('done opening write')
            return a

          t_w = executor.submit(t1)


        if self.read_fifo:

          def t2():
            glog.info('opening  read %s', self.read_fifo)
            a = open(self.read_fifo, 'rb')
            fl = fcntl.fcntl(a.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(a.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)
            #a = os.open(self.read_fifo, os.O_RDONLY  | os.O_NONBLOCK | os.O_CREAT, 0o644)
            glog.info('done opening read')
            return a

          t_r = executor.submit(t2)

        if t_w:
          self.wfile = t_w.result()
        if t_r:
          self.rfile = t_r.result()
    except:
      tb.print_exc()
      raise

  def _shutdown(self):
    if self.rfile:
      self.rfile.close()
    if self.wfile:
      self.wfile.close()

  def _send(self, data):
    if not self.wfile:
      assert False, "No fifo for write"
    self.wfile.write(data)

  def _recv(self, n, timeout):
    if not self.rfile:
      assert False, "No fifo for read"

    print('TRY RECV ', n)
    if not self._recv_ready(self.rfile.fileno(), timeout):
      return None
    res = self.rfile.read(n)
    print('has read ', res)
    if len(res) == 0:
      self.rfile.close()
    return res


class ManagedBidirectionalFifo(contextlib.ExitStack):

  def __init__(self):
    super().__init__()
    self.read_fifo = None
    self.write_fifo = None
    self.fifo = None

  def __enter__(self):
    super().__enter__()
    self.tempdir = tempfile.mkdtemp(prefix='chdrft_bififo_')
    try:
      self.read_fifo = os.path.join(self.tempdir, "read_fifo")
      self.write_fifo = os.path.join(self.tempdir, "write_fifo")

      os.mkfifo(self.read_fifo)
      os.mkfifo(self.write_fifo)
      self.fifo = Fifo(self.read_fifo, self.write_fifo)
      return self
    except OSError as e:
      if e.errno == errno.EEXIST:
        assert False, "Something spooky"
      raise

  def activate(self):
    chdrft_executor.submit(lambda: self.enter_context(self.fifo))


  def __exit__(self, typ, value, tb):
    self.fifo._shutdown()
    failsafe(lambda: os.remove(self.read_fifo))
    failsafe(lambda: os.remove(self.write_fifo))
    failsafe(lambda: os.removedirs(self.tempdir))
