#!/usr/bin/env python

from os.path import splitext, basename
from chdrft.tube.serial import TunneledFiles
import sys

import pysimulavr


class XPin(pysimulavr.Pin):

  def __init__(self, dev, name, state=None):
    pysimulavr.Pin.__init__(self)
    self.name = name
    if state is not None: self.SetPin(state)
    # hold the connecting net here, it have not be destroyed, if we leave this method
    self.__net = pysimulavr.Net()
    self.__net.Add(self)
    self.__net.Add(dev.GetPin(name))
    self.__dev = dev

  def SetInState(self, pin):
    pysimulavr.Pin.SetInState(self, pin)
    #print("%s='%s' (t=%dns)" % (self.name, pin.toChar(), sim.getCurrentTime()))

  def get(self):
    return self.__net.CalcNet()

  def connect(self, other_pin):
    self.__net.Add(other_pin)

  def release(self):
    self.__net.Delete(self)
    self.__net.Delete(self.__dev.GetPin(self.name))

  def set_high(self):
    self.SetPin('h')

  def set_low(self):
    self.SetPin('L')

  def set(self, v):
    if v == 0:
      self.set_low()
    else:
      self.set_high()


class SimulavrAdapter(object):

  def __init__(self):
    self.pins = []

  def loadDevice(self, t, e, clock):
    self.__sc = pysimulavr.SystemClock.Instance()
    self.__sc.ResetClock()
    dev = pysimulavr.AvrFactory.instance().makeDevice(t)
    dev.Load(e)
    dev.SetClockFreq(int(1e9//clock))
    self.__sc.Add(dev)
    self.__time = 0
    self.__dev = dev
    return dev

  def run(self, nt):
    self.__time += nt
    self.doRun(self.__time)

  def doRun(self, n):
    ct = self.__sc.GetCurrentTime
    while ct() < n:
      res = self.__sc.Run(int(n))
    return 0

  def doStep(self, stepcount=1):
    while stepcount > 0:
      res = self.__sc.Step()
      if res is not 0: return res
      stepcount -= 1
    return 0

  def getCurrentTime(self):
    return self.__sc.GetCurrentTime()

  def getAllRegisteredTraceValues(self):
    os = pysimulavr.ostringstream()
    pysimulavr.DumpManager.Instance().save(os)
    return filter(None, [i.strip() for i in os.str().split("\n")])

  def dmanSingleDeviceApplication(self):
    pysimulavr.DumpManager.Instance().SetSingleDeviceApp()

  def start(self):
    pysimulavr.DumpManager.Instance().start()

  def stop(self):
    pysimulavr.DumpManager.Instance().stopApplication()

  def setVCDDump(self, vcdname, signals, rstrobe=False, wstrobe=False):
    dman = pysimulavr.DumpManager.Instance()
    sigs = ["+ " + i for i in signals]
    dman.addDumpVCD(vcdname, "\n".join(sigs), "ns", rstrobe, wstrobe)

  def getWordByName(self, dev, label):
    addr = dev.data.GetAddressAtSymbol(label)
    v = dev.getRWMem(addr)
    addr += 1
    v = (dev.getRWMem(addr) << 8) + v
    return v

  def __enter__(self):
    pass
    return self

  def release(self):
    for pin in self.pins:
      pin.release()

  def __exit__(self, typ, value, tb):
    self.release()

  def add_pin(self, pin_name):
    if isinstance(pin_name, XPin): return pin_name
    pin = XPin(self.__dev, pin_name)
    self.pins.append(pin)
    return pin

  @property
  def dev(self):
    return self.__dev

  def add_serial_rx(self, tx_pin, fil, baudrate):
    serial_rx = pysimulavr.SerialRxFile(fil)
    serial_rx.SetBaudRate(baudrate)
    tx_pin = self.add_pin(tx_pin)
    tx_pin.connect(serial_rx.GetPin('rx'))
    self.serial_rx = serial_rx
    self.serial_tx_pin = tx_pin

  def add_serial_tx(self, rx_pin, fil, baudrate):
    serial_tx = pysimulavr.SerialTxFile(fil)
    serial_tx.SetBaudRate(baudrate)
    rx_pin = self.add_pin(rx_pin)
    rx_pin.connect(serial_tx.GetPin('tx'))
    self.serial_tx = serial_tx
    self.serial_rx_pin = rx_pin

  # rx, tx: avr view
  def setup_serial_con(self, exit_stack, baudrate, rx_pin, tx_pin):
    print('gogo ', rx_pin, tx_pin)
    conn_f1,f2 = exit_stack.enter_context(TunneledFiles(conn_f1=True))
    self.add_serial_rx(tx_pin, f2, baudrate)
    self.add_serial_tx(rx_pin, f2, baudrate)
    return conn_f1

class I2CHelper:

  def __init__(self, sim, sda, scl):
    self.sim = sim
    self.sda = sda
    self.scl = scl
    self.ts = 1e6

    self.sda.set_high()
    self.scl.set_high()

  def half(self):
    self.sim.run(self.ts)

  def quarter(self):
    self.sim.run(self.ts // 2)

  def write_bit(self, v):
    self.sda.set(v)
    self.quarter()
    self.scl.set_high()
    self.half()
    self.scl.set_low()
    self.half()

  def read_bit(self):
    self.sda.set_high()
    self.quarter()
    self.scl.set_high()
    self.half()
    v = self.sda.get()
    self.scl.set_low()
    self.half()
    return v

  def start(self):
    self.sda.set_low()
    self.half()
    self.scl.set_low()
    self.half()

  def stop(self):
    self.scl.set_high()
    self.half()
    self.sda.set_high()
    self.half()
    return self.sda.get()

  def set_addr(self, addr):
    self.start()
    return self.write_word(addr)

  def write_word(self, v):
    for i in range(8):
      b = v >> (7 - i) & 1
      self.write_bit(b)
    ack = self.read_bit() == 0
    return ack

  def read_word(self, last=False):
    v = 0
    for i in range(8):
      b = self.read_bit()
      v = v << 1 | b
    self.write_bit(last)
    return v


class SimpleSim:

  def __init__(self, model, elffile, clock):
    self.model = model
    self.elffile = elffile
    self.clock = clock
    self.serial_con = None

  def __enter__(self):
    sim = SimulavrAdapter()
    sim.dmanSingleDeviceApplication()
    sim.loadDevice(self.model, self.elffile, self.clock)
    self.sim = sim
    return self.sim

  def __exit__(self, typ, value, tb):
    self.sim.stop()
    self.sim.release()

#def go(sim):
#
#  sda = sim.add_pin('A0')
#  scl = sim.add_pin('A1')
#
#  sim.start()
#  #if doVCD:
#  #  print("all registrered trace values:\n ", end=' ')
#  #  print("\n  ".join(sim.getAllRegisteredTraceValues()))
#  #  sigs = ("IRQ.VECTOR9", "PORTA.PIN", 'PORTC.PORT', 'PORTC.C0-Out', 'PORTC.C1-Out',
#  #          'PORTC.C2-Out', 'PORTC.C3-Out', 'PORTC.C4-Out', 'PORTC.C5-Out', 'PORTC.C6-Out',
#  #          'PORTC.C7-Out')
#  #  sim.setVCDDump(splitext(basename(argv[0]))[0] + ".vcd", sigs)
#  #  print("-" * 20)
#
#  print("simulation start: (t=%dns)" % sim.getCurrentTime())
#  ts = int(2e6)
#  sim.run(ts)
#  i2c = I2CHelper(sim, sda, scl)
#  i2c.half()
#  i2c.half()
#  i2c.half()
#
#
#
#if __name__ == "__main__":
#
#  model, elffile = argv[1].split(":")
#  #dev.SetClockFreq(100)
#
#
#  with SimpleSim(simmodel, elffile):
#    go(sim)
#
#  print('laa')
#
#  # EOF
#
