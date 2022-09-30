#!/usr/bin/env python

from chdrft.main import app
import zerorpc
import glog
from chdrft.utils.misc import struct_helper
import os

global flags, cache
flags = None
cache = None


def args(parser):
  parser.add_argument('--port', type=int, default=12345)
  parser.add_argument('--host', type=str, default=os.environ.get('DEFAULT_HOST', 'fpga.local'))


class FPGAClient:

  def __init__(self, host=None, port=None, ctx=None):
    if host is None: host = ctx.host
    if port is None: port = ctx.port

    self.c = zerorpc.Client()
    self.c.connect("tcp://%s:%d" % (host, port))
    glog.info('Connecting to %s:%d', host, port)

    self.buf_client = RemoteBufClient(self.c)

  def run_cmd(self, *args, **kwargs):
    cmd_dbg = args[0]
    if isinstance(args[0], list):
      cmd_dbg = '#array: ' + ' '.join(args[0])
    glog.info('Running cmd=%s, args=%s, kwargs=%s', cmd_dbg, args[1:], kwargs)
    self.c.run_cmd(args, kwargs)

  @staticmethod
  def Args(parser):
    args(parser)


class RemoteBufClient:

  def __init__(self, c):
    self.c = c
    self.filename = '/dev/mem'
    assert self.c.check()
    self.c.set_active_file(self.filename)

  def add_mmap(self, pos, n):
    self.c.mmap_file(self.filename, pos, n)

  def mmap_struct_children(self, s):
    for x in s._children.values():
      self.add_mmap(x.byte_offset, x.byte_size)

  def rm_mmap(self, pos):
    self.c.unmmap_file(self.filename, pos)

  def read(self, pos, n=None):
    glog.info('read normal %x %x', pos, n)
    if n is None:
      return self.c.read_buf(pos, 1)[0]
    else:
      return self.c.read_buf(pos, n)

  def write(self, pos, content):
    glog.info('Write normal %x %x', pos, len(content) )
    return self.c.write_buf(pos, bytes(content))

  def read_atom(self, pos, n):
    res = self.c.read_atom(pos, n)
    glog.info('Read atom %x %x', pos,n)
    return struct_helper.tobytes(res, size=n)

  def write_atom(self, pos, content):
    glog.info('Write atom %x %s', pos, content)
    self.c.write_atom(pos, struct_helper.frombytes(content, size=len(content)), len(content))

  def __getitem__(self, x):
    if not isinstance(x, slice):
      return self.__getitem__(slice(x, x + 1))[0]
    else:
      assert x.step is None or x.step == 1
      assert x.stop > 0
      res = self.c.read_buf(x.start, x.stop - x.start)
      return res

  def __setitem__(self, x, v):
    if not isinstance(x, slice):
      return self.__setitem__(slice(x, x + 1), bytes([v]))
    else:
      assert x.step is None or x.step == 1
      assert x.stop > 0
      return self.c.write_buf(x.start, bytes(v))


def main():
  filename = '/tmp/test1.txt'
  print(c.mmap_file(filename, 0, 100))
  c.set_active_file(filename)
  print(c.write_buf(0, b'abcdef'))
  print(c.read_buf(0, 100))


app()
