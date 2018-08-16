from contextlib import ExitStack, contextmanager
import os
import mmap
from chdrft.utils.misc import Timeout
import time


class SysfsAttribute(ExitStack):

  def __init__(self, filename, timeout = None):
    super().__init__()
    self.filename = filename
    self.timeout = timeout
    self.fd = None
    self.flush()

  def __enter__(self):
    self.fd = os.open(self.filename, os.O_RDWR)
    self.callback(self._close)
    return self

  def _reset(self):
    os.lseek(self.fd, 0, os.SEEK_SET)

  def read(self):
    with self:
      return os.read(self.fd, mmap.PAGESIZE)

  def nonzero_read(self, timeout=None):
    if timeout is None: timeout = self.timeout
    timeout = Timeout.Normalize(timeout)
    while not timeout.expired():
      res = self.read()
      if len(res)>0: return res
      time.sleep(1e-2)
    timeout.raise_if_expired()

  def write(self, data):
    with self:
      return os.write(self.fd, data)

  def _close(self):
    os.close(self.fd)

  def flush(self):
    while len(self.read())!=0:
      pass
