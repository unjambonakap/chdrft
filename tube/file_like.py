#!/usr/bin/env python

from chdrft.tube import tube

import select
import errno
import os
import pty
import tempfile
import subprocess as sp
import time
from chdrft.utils.misc import Attributize
import glog


class FileLike(tube.Tube):

  def __init__(self, fileobj=None, **kwargs):
    super().__init__(**kwargs)
    self.closed = {'recv': True, 'send': True}
    self.fileobj = None
    if fileobj:
      self.set_fileobj(fileobj)

  def set_fileobj(self, fileobj):
    self.fileobj = fileobj

  def _connect(self):
    self.closed = {'recv': False, 'send': False}

  def _enter(self):
    self._connect()
    return self

  def _shutdown(self):
    self.close_dir('send')
    self.close_dir('recv')
    self._do_close()

  def _send(self, data):
    if self.closed['send']:
      raise EOFError

    try:
      written = self._do_write(data)
    except IOError as e:
      eof_numbers = [errno.EPIPE, errno.ECONNRESET, errno.ECONNREFUSED]
      if e.errno in eof_numbers:
        self.close_dir('send')
        raise EOFError
      else:
        raise
    return written

  def _recv(self, n, timeout):
    if self.closed['recv']:
      raise EOFError

    if not self._recv_ready(self.fileobj, timeout):
      self._raise_timeout()

    res = self._do_read(n)
    if len(res) == 0:
      glog.info('ZERO LENGTH RECV -> connection closed')
      self.close_dir('recv')
    return res

  def close_dir(self, direction):
    self.closed[direction] = True

  # what's may be to overwrite
  def _do_read(self, n):
    return self.fileobj.read(n)

  def _do_write(self, data):
    return self.fileobj.write(data)

  def _do_close(self):
    self.fileobj.close()


class FdTube(FileLike):

  def __init__(self, fd=None):
    super().__init__(fd)

  def _do_read(self, n):
    res= os.read(self.fileobj, n)
    return res

  def _do_write(self, data):
    return os.write(self.fileobj, data)

  def _do_close(self):
    os.close(self.fileobj)


class FileTube(FileLike):

  def __init__(self, filename=None, fd=None, fileobj=None):
    if filename:
      fileobj= open(filename)
    if fd:
      fileobj = os.fdopen(fd)
    super().__init__(fileobj)


def create_pty():
  master, slave = pty.openpty()
  slave_filename = os.ttyname(slave)
  # Shit won't be working unless ECHO is turned off :(
  return slave_filename, FdTube(fd=master)

