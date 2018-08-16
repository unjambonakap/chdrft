#!/usr/bin/env python

from enum import Enum
from chdrft.main import app
from chdrft.tube.serial import Serial
from chdrft.utils.misc import PatternMatcher, Attributize, BitMapper, BitOps
import time
import glog
from intelhex import IntelHex
from chdrft.elec.bus_pirate import *
from chdrft.emu.binary import MCS51Compiler
from chdrft.utils.swig import swig
from chdrft.utils.fmt import Format


class CC1110Consts:
  FLASH_WORD_SIZE = 2
  FLASH_PAGE_SIZE = 1024
  WORDS_PER_FLASH_PAGE = FLASH_PAGE_SIZE // FLASH_WORD_SIZE
  FLASH_SIZE = 32 * FLASH_PAGE_SIZE
  FREF = 26e6
  FOSC = 26e6


class MCS51_InsBuilder(object):

  def __init__(self):
    super().__setattr__('v', bytearray())

  def append(self, name, *args, **kwargs):
    attr = getattr(mcs51, name)
    self.v += bytearray(attr(*args, **kwargs))
    return self

  def __getattr__(self, name):
    return lambda *args, **kwargs: self.append(name, *args, **kwargs)


class MCS51_Assembler:

  def __init__(self):
    self.consts = CC1110Consts
    self.compiler = MCS51Compiler()
    self.status = BitMapper(['stack', 'osc', 'dbg', 'halt', 'power', 'cpu', 'pcon', 'erase',])
    self.config = BitMapper(['flash_info', 'timer_suspend', 'dma_pause', 'timers_off'])

  def disable_cache_prefetch(self):
    return bytes([0x75, 0xc7, 0x51])

  def erase_flash_cmd(self):
    cmd = []
    #// ; Wait for flash erase to complete
    cmd += [0x75, 0xAE, 0x01]  # MOV FLC, #01H; // ERASE
    cmd += [0xE5, 0xAE]  # eraseWaitLoop: MOV A, FLC;
    cmd += [0x20, 0xE7, 0xFB]  # JB ACC_BUSY, eraseWaitLoop;
    return bytes(cmd)

  def write_flash_page_cmd(self,
                           address=None,
                           data_ptr=0xf000,
                           data_len=CC1110Consts.FLASH_PAGE_SIZE,
                           erase=False,
                           halt=False):
    data_len //= CC1110Consts.FLASH_WORD_SIZE
    asm = []
    if address is not None:
      assert (address & 0x1) == 0, 'Address not aligned on a page'
      asm.append('mov FADDRH, #%d' % (((address >> 8) // CC1110Consts.FLASH_WORD_SIZE) & 0x7E))

    erase_code = ''
    if erase:
      erase_code = '''
  mov FCTL, #01;
eraseWaitLoop:
  mov a, FCTL;
  JB ACC_BUSY, eraseWaitLoop;
'''

    asm.append('''
    mov FADDRL, #00
    {erase_code}
    mov DPTR, #{data_ptr}
     MOV R7, #{data_len_h};
     MOV R6, #{data_len_l};
     MOV FCTL, #02;
     inc R7
writeLoopExt:
      dec R7
writeLoop:
     MOV R5, #{FLASH_WORD_SIZE};
writeWordLoop:
     MOVX A, @DPTR;
     INC DPTR;
     MOV FWDATA, A;
     DJNZ R5, writeWordLoop;
writeWaitLoop:
     MOV A, FCTL;
     JB ACC_SWBSY, writeWaitLoop;
     DJNZ R6, writeLoop;
     CJNE R7, #0, writeLoopExt;
    '''.format(data_ptr=data_ptr,
               data_len_h=data_len >> 8,
               data_len_l=data_len & 0xff,
               FLASH_WORD_SIZE=CC1110Consts.FLASH_WORD_SIZE,
               erase_code=erase_code,))
    res = self.compiler.assemble(asm).bin()
    if halt:
      res += mcs51.halt()

    #cmd = []
    #if address is not None:
    #  cmd += [0x75, 0xAD, ((address >> 8) // CC1110Consts.FLASH_WORD_SIZE) &
    #          0x7E]  ## MOV FADDRH, #imm;
    #cmd += [0x75, 0xAC, 0x00]  # # MOV FADDRL, #00;
    ## ; Initialize the data pointer
    #cmd += [0x90, data_ptr>>8, data_ptr&0xff]  # MOV DPTR, #0F000H;
    ## ; Outer loops
    #cmd += [0x7F, data_len >> 8]  # MOV R7, #imm;
    #cmd += [0x7E, data_len & 0xff]  # MOV R6, #imm;
    #cmd += [0x75, 0xAE, 0x02]  # MOV FLC, #02H; # WRITE
    ## ; Inner loops
    #cmd += [0x7D, CC1110Consts.FLASH_WORD_SIZE]  # writeLoop: MOV R5, #imm;
    #cmd += [0xE0]  # writeWordLoop: MOVX A, @DPTR;
    #cmd += [0xA3]  # INC DPTR;
    #cmd += [0xF5, 0xAF]  # MOV FWDATA, A;
    #cmd += [0xDD, 0xFA]  # DJNZ R5, writeWordLoop;
    ## ; Wait for completion
    #cmd += [0xE5, 0xAE]  # writeWaitLoop: MOV A, FLC;
    #cmd += [0x20, 0xE6, 0xFB]  # JB ACC_SWBSY, writeWaitLoop;
    #cmd += [0xDE, 0xF1]  # DJNZ R6, writeLoop;
    #cmd += [0xDF, 0xEF]  # DJNZ R7, writeLoop;
    ## ; Done, fake a breakpoint
    #cmd += [0xA5]  # DB 0xA5;
    #return bytes(cmd)
    return res

  def get_debug_ins(self, args):
    tb = {1: 0x55, 2: 0x56, 3: 0x57,}
    code = tb[len(args)]
    return [code] + list(args)

  def mova_imm(self, imm):
    return [0x74, imm]

  def inc_a(self):
    return bytes([0x4])

  def halt(self):
    return bytes([0xa5])

  def nop(self):
    return bytes([0])

  def set_pc(self, addr):
    return [0x2, addr >> 8 & 0xff, addr & 0xff]

  def builder(self):
    return MCS51_InsBuilder()

  def get_file_buf(self, filename):
    fil = open(filename, 'r')
    hexfile = IntelHex(fil)
    glog.info('Min addr >> ', hexfile.minaddr())
    assert hexfile.minaddr() == 0
    return hexfile.tobinstr()


mcs51 = MCS51_Assembler()


class CCHelper:

  def __init__(self, sendx_func, ins_func):
    self.sendx_func = sendx_func
    self.ins_func = ins_func

  def get_chip_id(self):
    #chip_id, chip_rev
    return list(self.sendx_func(0x68, 2))

  def debug_instr(self, *args):
    if len(args) == 1 and not isinstance(args[0], int):
      args = list(args[0])

    return self.ins_func(args)

  def get_pc(self):
    res = self.sendx_func(0x28, 2)
    return (res[0] << 8) + res[1]

  def set_pc(self, addr):
    self.debug_instr(mcs51.set_pc(addr))

  def halt(self):
    return self.sendx_func(0x44, 1)

  def resume(self):
    return self.sendx_func(0x4c, 1)

  @property
  def status(self):
    return mcs51.status.from_value(self.sendx_func(0x34, 1))

  @property
  def config(self):
    return mcs51.config.from_value(self.sendx_func(0x24, 1))

  @config.setter
  def config(self, value):
    return self.sendx_func([0x1d, mcs51.config.to_value(value)], 1)

  def erase(self):
    return self.sendx_func(0x14, 1)


class CC1110Debugger:

  def __init__(self, manager):
    self.manager = manager
    if manager is not None: self.mode = self.manager.get()

  def start(self):
    self.mode = self.manager.get().mode_binary().mode_binary_raw_wire()
    self.mode.action_modify_state(power=1, pullup=1, aux=1)
    self.mode.action_speed(BusPirateSpeedMode.SPEED_400KHZ)
    self.mode.action_modify_conf(lsb=0, prot_3wire=0, output_3v3=1)

  def reset(self):
    time.sleep(0.5)
    self.mode.action_modify_state(aux=1)
    time.sleep(0.5)
    self.mode.action_modify_state(aux=0)
    time.sleep(0.1)
    self.mode.action_nclock(2)
    time.sleep(0.1)
    self.mode.action_modify_state(aux=1)
    time.sleep(0.1)

  def get_chip_id(self):
    #chip_id, chip_rev
    return self.send_and_read(0x68, 2)

  def debug_instr(self, *args):
    if len(args) == 1 and not isinstance(args[0], int):
      args = list(args[0])

    return self.send_and_read(mcs51.get_debug_ins(args), 1)

  def get_pc(self):
    res = self.send_and_read(0x28, 2)
    return (res[0] << 8) + res[1]

  def set_pc(self, addr):
    self.debug_instr(mcs51.set_pc(addr))

  def halt(self):
    return self.send_and_read(0x44, 1)

  def resume(self):
    return self.send_and_read(0x4c, 1)

  @property
  def status(self):
    return mcs51.status.from_value(self.send_and_read(0x34, 1))

  @property
  def config(self):
    return mcs51.config.from_value(self.send_and_read(0x24, 1))

  @config.setter
  def config(self, value):
    return self.send_and_read([0x1d, mcs51.config.to_value(value)], 1)

  def erase(self):
    return self.send_and_read(0x14, 1)

  def execute_ins(self, ins):
    return self.debug_instr(ins)

  def send_and_read(self, data, nbytes):
    if isinstance(data, int):
      data = [data]
    self.mode.action_write(data)
    while self.mode.action_peek() != 0:
      pass
    tb = list([self.mode.action_read_byte() for i in range(nbytes)])
    if len(tb) == 1: return tb[0]
    return tb

  def read_xdata(self, addr, count):
    self.debug_instr(0x90, addr >> 8 & 0xff, addr & 0xff)
    tb = []
    for n in range(count):
      v = self.debug_instr(0xe0)
      tb.append(v)
      self.debug_instr(0xa3)
    return tb

  def write_xdata(self, addr, data):
    self.debug_instr(0x90, addr >> 8 & 0xff, addr & 0xff)
    for x in data:
      self.debug_instr(0x74, x)
      self.debug_instr(0xf0)
      self.debug_instr(0xa3)

  def wait_cpu(self):
    while not self.status.cpu:
      pass

  def execute_egg(self, egg_addr, egg_content):
    self.write_xdata(egg_addr, egg_content)
    self.set_pc(egg_addr)
    self.resume()
    self.wait_cpu()

  def read_flash_page(self, addr):
    return self.read_xdata(addr, mcs51.consts.FLASH_PAGE_SIZE)

  def write_flash_page(self, dest_addr, content):
    assert isinstance(content, bytes)
    content = content.ljust(mcs51.consts.FLASH_PAGE_SIZE, b'\x00')
    cmd = mcs51.write_flash_page_cmd(dest_addr, halt=True)
    #ram address
    tmp_addr = 0xf000
    glog.info('Writing xdata')
    self.write_xdata(tmp_addr, content)

    self.execute_ins(mcs51.disable_cache_prefetch())
    egg_addr = tmp_addr + CC1110Consts.FLASH_PAGE_SIZE
    glog.info('Executing egg')
    self.execute_egg(egg_addr, cmd)

  def flash_file(self, filename, addr=0, verify=False):
    hexfile = IntelHex(filename)
    print('Min addr >> ', hexfile.minaddr())
    assert hexfile.minaddr() == 0
    data = hexfile.tobinstr()
    page_size = mcs51.consts.FLASH_PAGE_SIZE

    for i in range(0, len(data), page_size):
      cur = data[i:i + page_size]
      glog.info('Writing page %d of %d', i // page_size, len(data) // page_size)
      self.write_flash_page(addr + i, cur)
      if verify:
        glog.info('Verifying page')
        written = bytes(self.read_flash_page(addr + i))
        written = written[:len(cur)]
        assert cur == written, '%s VS %s' % (cur, written)


class BufferConsumer:

  def __init__(self, data):
    self.pos = 0
    self.data = data

  def get(self, n=None):
    if n is None:
      n = len(self.data) - self.pos
    assert n > 0 and self.pos + n <= len(self.data)
    npos = self.pos + n
    res = self.data[self.pos:npos]
    self.pos = npos
    return res

  def rem(self):
    return self.data[self.pos:]

  def rem_len(self):
    return len(self.data) - self.pos


class RadioPacket:

  def __init__(self, msg, params):
    self.msg = msg
    self.params = params
    self.valid = False
    if msg is None: return

    for i in range(-1,2):
      consumer = BufferConsumer(Format(msg).tobytearray().shiftr(i).v)
      self.content = Attributize()

      self.content.preamble = consumer.get(params.npreamble.get())
      self.content.sync = consumer.get(params.nsync.get())
      print(self.content.sync)
      if self.check_sync():
        break
    else:
      return

    self.content.white_data = None
    if params.white_data.get():
      self.content.whitened_data = consumer.rem()
      consumer = BufferConsumer(self.unwhiten(self.content.whitened_data))

    self.content.raw_data = consumer.rem()
    self.content.length = consumer.get(1)[0]
    print('GOT LENGTH ', self.content.length)
    try:

      if not params.addr_check.is_no:
        self.content.address = consumer.get(1)[0]
      self.data_pos = consumer.pos
      self.content.data = consumer.get(self.content.length)
      self.content.crc = consumer.get(2)
      self.valid = True
    except:
      self.content.data = consumer.data[self.data_pos:]
      pass

  def get_white_seq(self, n):
    m = swig.opa_math_common_swig
    c = swig.opa_crypto_swig
    init_state = [0,0,0,0,0,1,1,1,1]
    poly = Format(0x221).bitlist(10).v
    init_state = m.Poly_u32(m.cvar.PR_GF2, m.v_u32(init_state))
    poly = m.Poly_u32(m.cvar.PR_GF2, m.v_u32(poly))
    lfsr = c.LFSR_u32(m.cvar.GF2, init_state, poly)

    lst = [lfsr.get_next() for _ in range(8 * n)]
    lst=Format(lst).bin2byte(lsb=1).v
    return lst

  def unwhiten(self, whitened_data):
    res = BitOps.xorlist(whitened_data, self.get_white_seq(len(whitened_data)))

    return Format.ToBytes(res)

  def check_preamble(self):
    nx = self.params.npreamble.get()
    want = b'\xaa' * nx
    return self.content.preamble == want

  def check_sync(self):
    nsync = self.params.nsync.get()
    expect_sync = bytes([self.params.SYNC1.get(), self.params.SYNC0.get()])
    if nsync == 4:
      expect_sync += expect_sync
    return self.content.sync == expect_sync
