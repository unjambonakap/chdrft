#!/usr/bin/env python

from __future__ import annotations
from chdrft.tube.tube import Tube
from chdrft.tube.connection import Connection, Server
from chdrft.tube.serial import Serial
import re
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import A
import glog
import numpy as np
from pydantic import Field, BaseModel
from chdrft.utils.path import FileFormatHelper
import subprocess as sp
import time

global flags, cache
flags = None
cache = None

def int_or_none(x):
  return None if x is None else int(x)

def get_tube_smart(uri: str) -> Tube:
  if m:= re.match(r'serial:(?P<path>[^,;]*)(,b(?P<baudrate>\d+))', uri):
    mg = m.groupdict()
    return Serial(port=mg['path'], baudrate=int_or_none(mg.get('baudrate', None)))
  if m:= re.match(r'udp:(?P<host>[^:]*)(?P<port>\d+))', uri):
    mg = m.groupdict()
    return Connection(mg['host'], port=int(mg['port']), udp=True)
  if m:= re.match(r'tcp:(?P<host>[^:]*)(?P<port>\d+))', uri):
    mg = m.groupdict()
    return Connection(mg['host'], port=int(mg['port']), udp=False)
  assert 0



def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  for lst in [
    'serial:/dev/ttyUSB0,b926100',
      'tcp:192.168.1.1:2222',
      'udp:192.168.1.1:2222',
  ]:

    a = get_tube_smart(lst)
    print(lst, a)


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()



