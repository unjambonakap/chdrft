#!/usr/bin/env python

from . import tube
import subprocess as sp
import os
import serial
import tempfile
import time
from ..utils import proc
import glog
from contextlib import ExitStack


class Serial(tube.Tube):

  def __init__(self, port=None, **kwargs):
    super(Serial, self).__init__()
    glog.debug('Creating serial port with params %s', kwargs)
    #not setting port here because don't want to open write away
    self.serial = serial.Serial(**kwargs)
    self.serial.port = port

    self.closed = True

  def _enter(self):
    glog.info('OPENGING port ', self.serial.port)
    self.serial.open()
    self.closed = False

  def _shutdown(self):
    glog.info('Serial shutting down')
    self.serial.close()
    self.closed = True

  def _send(self, data):
    if self.closed:
      raise EOFError
    return self.serial.write(data)

  def _recv(self, n, timeout):
    if self.closed:
      glog.info('Serial closed, raising eof')
      raise EOFError

    self._recv_ready(self.serial, timeout)

    try:
      want = min(n, self.serial.inWaiting())
      res = self.serial.read(want)
      if len(res) == 0:
        glog.info('Serial read zero size')
        self._shutdown()
        raise EOFError
    except Exception as e:
      raise EOFError
    return res

  def flush(self):
    self.serial.flush()


class SerialFromProcess(Serial):
  tty_filename = 'tty_fake'

  def __init__(self, cmd):
    super(SerialFromProcess, self).__init__()
    self.cmd = cmd

  def get_process_pid(self):
    proc_tree = proc.ProcTree()
    proc_tree.refresh()
    return proc_tree.nodes[self.socat_proc.pid]['children'][0]

  def _enter(self):
    self.tempdir = tempfile.mkdtemp()
    self.ptyfile = os.path.join(self.tempdir, self.tty_filename)
    self.serial.port = self.ptyfile

    self.socat_proc = sp.Popen(['socat', '-d', 'pty,link={0},raw,wait-slave'.format(self.ptyfile),
                                'exec:{0},pty,raw,echo=0'.format(self.cmd)])

    while not os.path.exists(self.ptyfile):
      time.sleep(1e-3)

    super(SerialFromProcess, self)._enter()

  def _shutdown(self):
    super(SerialFromProcess, self)._shutdown()
    try:
      if self.socat_proc.poll():
        self.socat_proc.terminate()
        if self.socat_proc.poll():
          self.socat_proc.kill()
    except:
      pass

    try:
      os.remove(self.ptyfile)
    except:
      pass
    os.rmdir(self.tempdir)


class TunneledFiles(ExitStack):

  def __init__(self, conn_f1=False, conn_f2=False, ctx=None):
    super().__init__()
    self.f1 = None
    self.f2 = None
    self.conn_f1 = conn_f1
    self.conn_f2 = conn_f2
    self.cmd = None
    self.ctx=ctx



  def __create(self):
    self.f1 = tempfile.mktemp(prefix='opa_tunnel_f1_')
    self.f2 = tempfile.mktemp(prefix='opa_tunnel_f2_')
    def pty_fmt(x):
      return 'PTY,raw,echo=0,link=%s'%x

    self.cmd = sp.Popen(['socat', pty_fmt(self.f1), pty_fmt(self.f2)])
    while not os.path.exists(self.f1) and self.cmd.poll() is None:
      time.sleep(0.1)

    if self.cmd.poll() is not None:
      glob.error('Socat failed for soem reason, err_code is %s', self.cmd.returncode)
      assert False

    if self.conn_f1:
      self.f1=Serial(self.f1)
      self.enter_context(self.f1)
    if self.conn_f2:
      self.f2=Serial(self.f2).__enter__()
      self.enter_context(self.f2)

  def __shutdown(self):
    if self.f1 is None:
      return

    print('KILLING SHIT')
    self.cmd.kill()
    self.f1 = None
    self.f2 = None

  def __enter__(self):
    self.__create()
    return self.f1, self.f2

  def __exit__(self, typ, value, tb):
    self.__shutdown()
