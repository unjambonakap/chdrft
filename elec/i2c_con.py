#!/usr/bin/env python

from __future__ import annotations
from enum import Enum
from chdrft.main import app
from chdrft.tube.serial import Serial
from chdrft.utils.misc import PatternMatcher, Attributize, to_bytes
import time
import glog
import struct
from contextlib import ExitStack
from chdrft.elec.bus_pirate import bus_pirate_manager
from chdrft.tube.tube import TubeWrapper


class I2CConLine:

  def __init__(self, conn: I2CCon, addr: int):
    self.conn = conn
    self.addr = addr

  def get_wrapper(self):
    return TubeWrapper(
        send=lambda data: self.conn.i2c_con_send(self.addr, data),
        recv=lambda n, timeout: self.conn.i2c_con_recv(self.addr, n, timeout),
      enter=self.conn.__enter__
    )

class I2CCon(ExitStack):

  def __init__(self, conn):
    super().__init__()
    self.conn = conn
    self.manager = bus_pirate_manager(conn)
    self.mode = None

  def __enter__(self):
    self.enter_context(self.manager)
    self.mode = self.manager.get().mode_binary().mode_i2c()
    self.mode.action_modify_state(power=1, pullup=1, aux=0, cs=0)
    print('mode is ', self.mode)
    return self

  def reset(self):
    self.mode.action_modify_state(power=0, pullup=0, aux=0, cs=0)
  def activate(self):
    self.mode.action_modify_state(power=1, pullup=1, aux=0, cs=0)

  def __exit__(self, typ, value, tb):
    glog.info('Exit i2ccon')
    super().__exit__(typ, value, tb)

  def __call__(self, addr):
    return I2CConLine(self, addr).get_wrapper()

  def i2c_con_send(self, addr, data):
    self.mode.action_i2c_write(addr, data)

  def i2c_con_recv(self, addr, n, timeout):
    while True:
      avail=self.mode.action_i2c_read(addr, 1)
      assert avail is not None
      if avail[0]==0:
        time.sleep(1e-3)
        continue
      print(avail, n)
      avail=min(avail, n)
      res = self.mode.action_i2c_read(addr, avail+1)
      return res[1:]

