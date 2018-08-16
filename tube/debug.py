#!/usr/bin/env python

from . import tube
import subprocess as sp
from chdrft.utils.fmt import Format


class DebugTube(tube.Tube):
  def __init__(self):
    super().__init__()
    self.buf = b''

  def _enter(self):
    pass

  def _send(self, data):
    assert 0

  def push_data(self, data):
    self.buf += Format.ToBytes(data)

  def _recv(self, n, timeout):
    while True:
      if len(self.buf)!=0:
        res=self.buf[:n]
        self.buf = self.buf[n:]
        return res
      if timeout is not None:
        timeout.raise_if_expired()

  def _shutdown(self):
    self.proc.terminate()
    if self.proc.poll():
      self.proc.kill()
