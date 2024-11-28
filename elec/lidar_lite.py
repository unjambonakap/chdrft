#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from pydantic import Field
import pandas as pd
import polars as pl
import re
import os.path
import os
import sys
import time
import chdrft.utils.Z as Z
from chdrft.elec.i2c_con import I2CCon
from chdrft.elec.bus_pirate import bus_pirate_manager, BusPirateModeManager
from chdrft.tube.serial import Serial


global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--tty')
  BusPirateModeManager.SetFlags(parser)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  conn = Serial(ctx.tty)
  conn.serial.baudrate = 115200
  with bus_pirate_manager(conn) as bp:
    mode = bp.get().mode_binary().mode_i2c()

    #mode.action_modify_state(power=1)
    #mode.action_modify_conf(output_3v3=1)
    #mode.action_write(0x62, bytearray([0, 0]))
    #mode.action_write(0x62, bytearray([0, 4]))
    #print(mode.action_write_and_read(0x62, bytearray([1]), 1))
    print(mode.action_write_and_read(0x62, bytearray([0x16]), 2))



def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
