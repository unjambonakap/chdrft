#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
from chdrft.emu.base import NonRandomMemory

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst)


class BufferParser(NonRandomMemory):
  def __init__(self, data, *args, **kwargs):
    self.data = data
    self.pos = 0
    super().__init__(reader=lambda addr, size: self._read(size), *args, **kwargs)

  def _read(self, size):
    res = self.data[self.pos:self.pos+size]
    self.pos += size
    return res

class BufferBuilder(NonRandomMemory):
  def __init__(self):
    self.data = bytearray()
    self.pos = 0
    super().__init__(writer=lambda addr, content: self._write(content))


  def _write(self, content):
    self.data += content



def test(ctx):
  res=BufferParser(b'abcdefkapapa'*100)
  print(res.read_u8())
  print(res.read_u32())
  print(res.read(5))

  res=BufferBuilder()
  print(res.write_u8(123))
  print(res.write(b'abc'))
  print(res.data)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
