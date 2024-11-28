#!/usr/bin/env python

from enum import Enum
from chdrft.main import app
from chdrft.tube.serial import Serial
from chdrft.utils.misc import PatternMatcher, Attributize, to_bytes
import time
import glog
import struct
from contextlib import ExitStack, contextmanager


class BusPirateModes(Enum):
  TEXT_MODE = 'text'
  BINARY_MODE = 'binary'
  BINARY_I2C = 'i2c'
  BINARY_SPI = 'spi'
  BINARY_RAW_WIRE = 'binary_raw_wire'
  PARENT = 'parent'


class BusPirateSpeedMode(Enum):
  SPEED_5KHZ = 0
  SPEED_50KHZ = 1
  SPEED_100KHZ = 2
  SPEED_400KHZ = 3

class BusPirateSpeedModeSPI(Enum):
  SPEED_30KHZ = 0
  SPEED_125KHZ = 1
  SPEED_250KHZ = 2
  SPEED_1MHZ = 3



class BusPirateAction:

  def __init__(self,
               name=None,
               cmd=None,
               expected_output=None,
               output_size=None,
               func=None,
               output_type=None):
    self.cmd = cmd
    self.name = name
    self.output_size = output_size
    self.func = func
    self.output_type = output_type
    self.expected_output = expected_output

  @staticmethod
  def Make(v):
    if isinstance(v, BusPirateAction):
      return v
    elif isinstance(v, list):
      name = v[0]
      v = v[1:]
      if len(v) == 2:
        return BusPirateAction(name, cmd=v[0], output_size=v[1])
      else:
        return BusPirateAction(name, func=v[0])


class BusPirateModeDesc:

  def __init__(self, name=None, enter_expect=None, actions=None, submodes=[]):
    self.name = name
    self.enter_expect = enter_expect
    self.submodes = set(submodes)
    self.parent = None

    self.actions = {}
    if actions:
      for action in actions:
        tmp = BusPirateAction.Make(action)
        self.actions[tmp.name] = tmp


class BusPirateMode:

  def __init__(self, manager, desc):
    self.serial = None
    self.manager = manager
    self.desc = desc

  def __getattr__(self, name):
    if name.startswith('mode_'):
      mode_name = name[5:]
      mode_name = BusPirateModes(mode_name)
      return lambda: self.manager.change_mode(self, mode_name)
    if name.startswith('action_'):
      action_name = name[7:]
      action = self.desc.actions[action_name]
      return self.manager.proc_action(action)


class BusPirateModeManager(ExitStack):
  REPLY_OK = b'\x01'

  @staticmethod
  def SetFlags(parser):
    parser.add_argument('--from_text', action='store_true')
    parser.add_argument('--default_mode',
                        type=BusPirateModes,
                        default=BusPirateModes.TEXT_MODE.value)

  def __init__(self, modes):
    super().__init__()
    self.modes = {}
    self.setup(modes)

    root_mode_name = BusPirateModes.TEXT_MODE
    self.update_submodes(self.modes[root_mode_name])
    self.state = Attributize(aux=1, pullup=1, power=0, cs=0)
    self.conf = Attributize(output_3v3=0, prot_3wire=0, lsb=0, ckp_idle=0, cke_active=1, sample_time=0)
    self.serial = None
    self.flags = None
    self.default_mode = None
    self.active_mode = None

  def setup(self, modes):
    for mode in modes:
      assert not mode.name in self.modes
      self.modes[mode.name] = mode

  def update_submodes(self, mode):
    mode.submodes = {x: self.modes[x] for x in mode.submodes}
    for submode_name in mode.submodes:
      submode = self.modes[submode_name]
      self.update_submodes(submode)
      submode.submodes[mode.name] = mode
      submode.submodes[BusPirateModes.PARENT] = mode

  def get_mode(self, v):
    if isinstance(v, str):
      v = BusPirateModes(v)

    if isinstance(v, BusPirateModes):
      assert v in self.modes, 'Fail for ' + v
      v = self.modes[v]
    if isinstance(v, BusPirateMode):
      return v
    return BusPirateMode(self, v)

  @contextmanager
  def _cleanup_on_error(self):
    with ExitStack() as stack:
      stack.push(self)
      yield
      # The validation check passed and didn't raise an exception
      # Accordingly, we want to keep the resource, and pass it
      # back to our caller
      stack.pop_all()

  def __enter__(self):
    self.enter_context(self.serial)
    glog.info('Bus pirate enter %s', self.flags.from_text)
    with self._cleanup_on_error():
      self.enter(self.serial, self.flags, self.default_mode)
    return self

  def __call__(self, serial, flags=None, default_mode=None):
    self.serial = serial
    if flags is None:
      flags = app.flags
    self.flags = flags
    self.default_mode = default_mode
    return self


  def enter(self, serial, flags, default_mode=None):
    self.flags = flags
    self.active_mode = self.get_mode(self.flags.default_mode)
    self.serial = serial
    if self.active_mode.desc.name == BusPirateModes.TEXT_MODE and self.flags.from_text:
      #self.serial.send(b'\n' * 20)
      #self.serial.send('#')
      for i in range(20):
        self.serial.send(b'\x00')
        time.sleep(1e-3)
        self.serial.trash()

      self.serial.trash(0.3)
      self.change_mode(self.active_mode, BusPirateModes.BINARY_MODE)

  def get(self):
    self.active_mode.serial = self.serial
    return self.active_mode

  def change_mode_adj(self, from_mode, to_name):
    assert to_name in from_mode.desc.submodes
    to_mode = self.get_mode(to_name)
    self.execute_enter_expect(to_mode.desc.enter_expect)
    self.active_mode = to_mode
    return self.to_mode

  def change_mode(self, from_mode, to_name):
    to_mode = self.get_mode(to_name)
    self.execute_enter_expect(to_mode.desc.enter_expect)
    self.active_mode = to_mode
    return to_mode

  def normalize_cmd(self, cmd):
    return to_bytes(cmd)

  def execute_enter_expect(self, enter_expect):
    cmd, expect = enter_expect
    cmd = self.normalize_cmd(cmd)
    return self.serial.send_and_expect(cmd, expect)

  def write_raw_wire(self, data):
    if isinstance(data, int):
      data = bytes([data])
    if isinstance(data, list):
      data = bytes(data)

    cmd = bytes([16 + len(data) - 1]) + data
    #fuck your doc
    self.execute_enter_expect(
        [cmd, BusPirateModeManager.REPLY_OK * (len(data) + 1)])
    self.set_val(1)

  def send_nclock(self, n):
    cmd = bytes([32 + n - 1])
    self.execute_enter_expect([cmd, BusPirateModeManager.REPLY_OK])

  def set_speed(self, speed_mode):
    cmd = bytes([0b01100000 + speed_mode.value])
    self.execute_enter_expect([cmd, BusPirateModeManager.REPLY_OK])

  def set_clock(self, val):
    cmd = bytes([0b000001010 + val])
    self.execute_enter_expect([cmd, BusPirateModeManager.REPLY_OK])

  def set_val(self, val):
    cmd = bytes([0b000001100 + val])
    self.execute_enter_expect([cmd, BusPirateModeManager.REPLY_OK])

  def modify_state(self, **kwargs):
    for k, v in kwargs.items():
      assert k in self.state, 'Bad key ' + k
      self.state[k] = v
    new_state = self.get_state()
    cmd = new_state | 0b01000000
    glog.info('setting state to %s', hex(new_state))
    self.execute_enter_expect([cmd, BusPirateModeManager.REPLY_OK])

  def get_state(self):
    x = self.state
    return 8 * x.power + 4 * x.pullup + 2 * x.aux + 1 * x.cs
  def get_conf(self):
    x = self.conf
    if self.get_mode(BusPirateModes.BINARY_SPI) == self.active_mode:
      return 8 * x.output_3v3 + 4*x.ckp + 2*  x.cke + 1 * x.sample_time
    else:
      return 8 * x.output_3v3 + 4 * x.prot_3wire + 2 * x.lsb

  def modify_conf(self, **kwargs):
    for k, v in kwargs.items():
      assert k in self.conf, 'Bad key ' + k
      self.conf[k] = v
    new_conf = self.get_conf()
    cmd = new_conf | 0b10000000
    glog.info('setting conf to %s', hex(new_conf))
    self.execute_enter_expect([cmd, BusPirateModeManager.REPLY_OK])

  def proc_action(self, action):
    func = action.func
    if func is None:

      def built_func(*args):

        if action.expected_output:
          out = self.execute_enter_expect([action.cmd, action.expected_output])
          return
        else:
          self.serial.send(self.normalize_cmd(action.cmd))
          out = self.serial.recv_fixed_size(action.output_size)
        if action.output_type == int:
          assert len(out) == 1
          return out[0]
        assert 0

      func = built_func

    return lambda *args, **kwargs: func(self, *args, **kwargs)

  def i2c_read_addr(self, addr):
    return bytes([addr << 1 | 1])

  def i2c_write_addr(self, addr):
    return bytes([addr << 1 | 0])

  def raw_i2c_write_and_read(self, data, nread):
    data = to_bytes(data)
    cmd = struct.pack('>BHH', 0x08, len(data), nread) + data
    self.serial.send(cmd)
    res = self.serial.recv_fixed_size(1)
    if res[0] == 0:
      return None
    return self.serial.recv_fixed_size(nread)

  def i2c_write(self, addr, data):
    write_cmd = self.i2c_write_addr(addr) + to_bytes(data)
    if self.raw_i2c_write_and_read(write_cmd, 0) is None:
      return False
    return True

  def i2c_read(self, addr, nread):
    read_cmd = self.i2c_read_addr(addr)
    return self.raw_i2c_write_and_read(read_cmd, nread)

  def i2c_write_and_read(self, addr, data, nread):
    if not self.i2c_write(addr, data):
      return None
    return self.i2c_read(addr, nread)



  def spi_write_and_read(self, write=None, nread=None):
    cmd = 0b00010000
    if write is None:
      write=bytearray([0]*nread)
    write = to_bytes(write)
    if nread is None:
      nread = len(write)
    assert nread >= 1  and nread <=16, nread
    cmd += nread - 1
    self.serial.send(bytearray([cmd] + write))
    return self.serial.recv_fixed_size(nread)

  def spi_write_and_read_fast(self, write=b'', nread=0):
    cmd = 0b00000100
    write = to_bytes(write)
    full_cmd = struct.pack('>BHH', cmd, len(write), nread) + write
    self.serial.send(full_cmd)
    res = self.serial.recv_fixed_size(1)
    if res[0] == 0:
      return None
    return self.serial.recv_fixed_size(nread)


binary_raw_wire_actions = [
    BusPirateAction(name='modify_state',
                    func=lambda x, **kwargs: x.modify_state(**kwargs)),
    BusPirateAction(name='modify_conf',
                    func=lambda x, **kwargs: x.modify_conf(**kwargs)),
    BusPirateAction(name='read_byte',
                    cmd=0b00000110,
                    output_size=1,
                    output_type=int),
    BusPirateAction(
        name='read_nbyte',
        func=lambda x, n: list([x.active_mode.action_read_byte() for i in range(n)]),),
    BusPirateAction(name='peek',
                    cmd=0b00001000,
                    output_size=1,
                    output_type=int),
    BusPirateAction(name='read_bit',
                    cmd=0b00000110,
                    output_size=1,
                    output_type=int),
    BusPirateAction(name='write',
                    func=lambda x, data: x.write_raw_wire(data)),
    BusPirateAction(name='nclock',
                    func=lambda x, n: x.send_nclock(n)),
    BusPirateAction(name='speed',
                    func=lambda x, mode: x.set_speed(mode)),
    BusPirateAction(name='set_clock',
                    func=lambda x, val: x.set_clock(val)),
    BusPirateAction(name='set_val',
                    func=lambda x, val: x.set_val(val)),
    BusPirateAction(name='one_clock',
                    cmd=0b00001001,
                    expected_output=BusPirateModeManager.REPLY_OK,),
    BusPirateAction(name='i2c_start',
                    cmd=0b00000010,
                    expected_output=BusPirateModeManager.REPLY_OK,),
    BusPirateAction(name='i2c_stop',
                    cmd=0b00000011,
                    expected_output=BusPirateModeManager.REPLY_OK,),
]

i2c_actions = [
    BusPirateAction(name='modify_state',
                    func=lambda x, **kwargs: x.modify_state(**kwargs)),
    BusPirateAction(name='modify_conf',
                    func=lambda x, **kwargs: x.modify_conf(**kwargs)),
    BusPirateAction(name='nclock',
                    func=lambda x, n: x.send_nclock(n)),
    BusPirateAction(name='speed',
                    func=lambda x, mode: x.set_speed(mode)),
    BusPirateAction(name='write_and_read',
                    func=BusPirateModeManager.i2c_write_and_read),
    BusPirateAction(name='write',
                    func=BusPirateModeManager.i2c_write),
    BusPirateAction(name='read',
                    func=BusPirateModeManager.i2c_read),
]

spi_actions = [
    BusPirateAction(name='modify_state',
                    func=lambda x, **kwargs: x.modify_state(**kwargs)),
    BusPirateAction(name='cs',
                    func=lambda x, n: x.set_cs(n)),
    BusPirateAction(name='speed',
                    func=lambda x, mode: x.set_speed(mode)),
    BusPirateAction(name='write_and_read',
                    func=BusPirateModeManager.spi_write_and_read),
    BusPirateAction(name='write_and_read_fast',
                    func=BusPirateModeManager.spi_write_and_read_fast),
    BusPirateAction(name='modify_conf',
                    func=lambda x, **kwargs: x.modify_conf(**kwargs)),
]

bus_pirate_modes_desc = [
    BusPirateModeDesc(name=BusPirateModes.TEXT_MODE,
                      submodes=[BusPirateModes.BINARY_MODE]),
    BusPirateModeDesc(name=BusPirateModes.BINARY_MODE,
                      submodes=[BusPirateModes.BINARY_RAW_WIRE],
                      enter_expect=[b'\x00', 'BBIO1']),
    BusPirateModeDesc(name=BusPirateModes.BINARY_I2C,
                      enter_expect=[b'\x02', 'I2C1'],
                      actions=i2c_actions),
    BusPirateModeDesc(name=BusPirateModes.BINARY_SPI,
                      enter_expect=[b'\x01', 'SPI1'],
                      actions=spi_actions),
    BusPirateModeDesc(name=BusPirateModes.BINARY_RAW_WIRE,
                      enter_expect=[b'\x05', 'RAW1'],
                      actions=binary_raw_wire_actions),
]
bus_pirate_manager = BusPirateModeManager(bus_pirate_modes_desc)
