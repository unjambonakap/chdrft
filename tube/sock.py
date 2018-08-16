#!/usr/bin/env python

from chdrft.tube.file_like import FileLike
import select
import errno


class Sock(FileLike):
  def __init__(self):
    super().__init__()

  def _do_read(self, n):
    return  self.fileobj.recv(n)

  def _do_write(self, data):
    return self.fileobj.send(data)


