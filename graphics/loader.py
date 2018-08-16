#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, read_file_or_buf
import glog
from chdrft.utils.parser import BufferParser
import stl #numpy-stl

global flags, cache
flags = None
cache = None

def args_def(parser):
  parser.add_argument('--file', type=str)
  parser.add_argument('--outfile', type=str)

def args(parser):
  args_def(parser)
  clist = CmdsList().add(parse).add(to_binary_stl)
  ActionHandler.Prepare(parser, clist.lst)


def parse(ctx):
  return stl_parser(flags.file)


def to_binary_stl(ctx):
  mesh =stl.mesh.Mesh.from_file(ctx.file)
  mesh.save(ctx.outfile, mode=stl.Mode.BINARY)


def stl_parser(**kwargs):
  data = read_file_or_buf(**kwargs)
  parser = BufferParser(data)
  header = parser.read(80)
  ntr = parser.read_u32()
  tb = []
  for i in range(ntr):
    norm = parser.read_nf32(3)
    vx = [parser.read_nf32(3) for j in range(3)]
    attr = parser.read_u16()
    tb.append(Attributize(norm=norm, vx=vx, attr=attr))
  return tb


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
