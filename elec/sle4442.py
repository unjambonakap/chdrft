#!/usr/bin/env python

from enum import Enum
from chdrft.main import app
from chdrft.tube.serial import Serial
from chdrft.utils.misc import PatternMatcher, Attributize, BitMapper
import time
import glog
from intelhex import IntelHex
from chdrft.elec.bus_pirate import *
from chdrft.utils.fmt import fmt


class SLE4442:

  def __init__(self, manager):
    self.manager = manager
    self.mode = self.manager.get()

  def start(self):
    self.mode = self.manager.get().mode_binary().mode_binary_raw_wire()
    self.mode.action_speed(BusPirateSpeedMode.SPEED_50KHZ)
    self.mode.action_modify_conf(lsb=1, prot_3wire=0, output_3v3=0)
    self.mode.action_modify_state(power=0, pullup=0, cs=0)
    self.mode.action_set_clock(0)
    self.mode.action_set_val(0)
    time.sleep(0.5)
    self.mode.action_modify_state(power=1, pullup=1, cs=0)
    time.sleep(0.2)

  def atr(self):
    self.mode.action_modify_state(cs=1)
    self.mode.action_nclock(1)
    self.mode.action_modify_state(cs=0)
    return self.mode.action_read_nbyte(4)

  def send_outgoing(self, data, n):
    self.send_cmd(data)
    res = self.mode.action_read_nbyte(n)
    self.mode.action_read_bit()
    return res

  def send_processing(self, data):
    self.send_cmd(data)
    while not self.mode.action_read_bit():
      pass

  def send_cmd(self, data):
    data = bytes(fmt(data).pad(3, 0).v)
    self.mode.action_i2c_start()
    self.mode.action_write(data)
    self.mode.action_i2c_stop()
    self.mode.action_set_clock(0)

  def read_main(self, addr, nbytes):
    need = 256 - addr
    assert nbytes <= need
    res= self.send_outgoing([0b00110000, addr], need)
    glog.info('Read >> %s', res)
    return res[:nbytes]

  def read_protection(self):
    return self.send_outgoing([0b00110100], 4)

  def read_security(self):
    res=self.send_outgoing([0b00110001], 4)
    glog.info('security read >> %s', res)
    return res

  def update_main(self, addr, word):
    self.send_processing([0b00111000, addr, word])

  def update_protection(self, addr, word):
    self.send_processing([0b00111100, addr, word])

  def update_security(self, addr, word):
    self.send_processing([0b00111001, addr, word])

  def compare_verification_data(self, addr, word):
    self.send_processing([0b00110011, addr, word])

  def unlock(self, passwd):
    assert len(passwd) == 3
    res = self.read_security()
    assert res[0] != 0
    # unset one bit
    ntry = res[0] & (res[0] - 1)
    assert (ntry  & 0xff) > 0, 'Wont lock the card dumdum'

    self.update_security(0, ntry)
    for i in range(3):
      self.compare_verification_data(i + 1, passwd[i])

    self.update_security(0, 0xff)
    res = self.read_security()
    return res
