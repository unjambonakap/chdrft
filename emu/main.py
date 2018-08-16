#!/usr/bin/env python

from chdrft.main import app
from chdrft.utils.misc import cwdpath, Attributize, Arch, opa_print, to_list, lowercase_norm
from chdrft.utils.math import rotlu32, modu32
from chdrft.emu.elf import ElfUtils, MEM_FLAGS
from chdrft.emu.binary import X86Machine, cs_print_x86_insn, regs, Memory, Regs
from chdrft.emu.syscall import SyscallDb
from chdrft.emu.structures import StructureReader, Structure, BufferMemReader
from chdrft.emu.trace import Tracer, TraceDiff, WatchedMem, TraceEvent, VmOp
from chdrft.emu.kernel import Kernel, load_elf
from chdrft.tube.sock import Sock
import multiprocessing as mp

import binascii
import ctypes
import pprint as pp
import threading
import time

import struct
from capstone.x86_const import *
import traceback as tb
import jsonpickle
import sys

global flags, cache
flags = None
cache = None

#!/usr/bin/env python
# Sample code for X86 of Unicorn. Nguyen Anh Quynh <aquynh@gmail.com>

from unicorn import *
from unicorn.x86_const import *


def args(parser):
  parser.add_argument('--file', type=cwdpath)
  parser.add_argument('--mode', type=int, default=0)
  parser.add_argument('--runid', type=str, default='0')
  parser.add_argument('--nstep', type=int, default=10)
  parser.add_argument('--buf', type=str, default='')
  parser.add_argument('--action', type=int, default=0)



W1_C1 = 0xb00bface
W1_C2 = 0xdeadbeef
W1_CMP = 0xfc999c2c


def enc_w1(w):
  s1 = w ^ W1_C1
  s2 = s1 - W1_C2
  s3 = ctypes.c_uint32(s2).value
  return rotlu32(s3, 0x11)


def dec_w1(w):
  s1 = rotlu32(w, -0x11)
  s2 = s1 + W1_C2
  s3 = ctypes.c_uint32(s2).value
  return s3 ^ W1_C1


def check1(b):

  tb = []
  for i in range(len(b) // 8):
    tb.append(int(b[i * 8:i * 8 + 8], 16))
  x = 0
  for e in tb:
    print(hex(e))
    x ^= e
  x = enc_w1(x)
  print(hex(x), hex(W1_CMP))
  assert x == W1_CMP


def load1(buf, pos):
  mu = Uc(UC_ARCH_X86, UC_MODE_64)
  elf = load_elf(mu, flags.file)
  log_file = '/tmp/info_{}.out_{}'.format(pos, flags.runid)
  kern = Kernel(mu, log_file)
  handlers = open('/home/benoit/programmation/comm/handlers.sigaction.delroth', 'r').read()
  handlers = jsonpickle.decode(handlers)
  for handler in handlers:
    kern.do_sigaction(*handler)

  if 1:
    code_low = 0x13370000
    code_high = 0x13390000
    stack_low = 0x7ffff7dc6000
    stack_high = 0x7ffff7dd7400
  else:
    code_low = 1
    code_high = 0
    stack_low = 1
    stack_high = 0

  mu.hook_add(UC_HOOK_INSN, safe_hook(kern.hook_syscall), None, UC_X86_INS_SYSCALL)
  mu.hook_add(UC_HOOK_INTR, safe_hook(kern.hook_intr))
  mu.hook_add(UC_HOOK_MEM_FETCH_UNMAPPED, safe_hook(kern.hook_unmapped))


  if 0:
    trigger_code=0x1337c131
    global triggered
    triggered=False
    def hook_trigger(mu, address, size, _2):
      print('TRIGGER HERE')
      global triggered
      if triggered:
        return
      triggered=True
      mu.hook_add(UC_HOOK_CODE, safe_hook(kern.hook_code), None, code_low, code_high)
      mu.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, safe_hook(kern.hook_mem_access), None,
                  stack_low, stack_high)

    mu.hook_add(UC_HOOK_CODE, hook_trigger, None, trigger_code, trigger_code+1)
  else:
    mu.hook_add(UC_HOOK_CODE, safe_hook(kern.hook_code), None, code_low, code_high)
    mu.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, safe_hook(kern.hook_mem_access), None,
                stack_low, stack_high)

  if flags.mode == 0:
    print(hex(kern.regs.rsi), hex(kern.regs.rdx))
    print(hex(kern.regs.rax))
    print('gogo for buf ', buf)
    if len(buf)>0:
      kern.mem.write(kern.regs.rsi, buf)
      kern.regs.rdx = len(buf)
      print(kern.mem.read(kern.regs.rsi, kern.regs.rdx))
  else:
    kern.regs.rax = kern.syscalls.entries.__rt_sigreturn.syscall_num
    kern.regs.rip = kern.regs.rip - 2
    print(hex(kern.regs.rip))
    for i in range(40):
      addr = kern.regs.rsp + 8 * i - 8
      print(hex(addr), hex(kern.mem.get_u64(addr)))
    if flags.mode == 2:
      return
  return kern


def event_handle(e, fevent, fop):
  if isinstance(e, TraceEvent):
    fevent.write(str(e) + '\n\n')
  elif isinstance(e, VmOp):
    fevent.write(e.gets(regs=True) + '\n\n\n')
    fop.write(str(e) + '\n\n\n')
    fop.flush()
    fevent.flush()


def start_kern(q, buf, i):
  kern = load1(buf, i)
  kern.sigret_count_lim = flags.nstep

  runid = '{}_{}'.format(flags.runid, i)
  event_log = open('/tmp/evens_{}.out'.format(runid), 'w')
  vmop_log = open('/tmp/vmop_{}.out'.format(runid), 'w')
  kern.tracer.cb = lambda x: event_handle(x, event_log, vmop_log)

  if q is not None:
    sys.stdout = open('/tmp/kern_{}.out'.format(i), 'w')

  try:
    print('starting >> ', hex(kern.regs.rip))
    kern.start()
  except UcError as e:
    print('%s' % e)
  return


def test_diff():
  kx = []
  bx = [b'1ccfc7f400000001', b'1ccfc7f500000000']
  ni = 2

  print('LAAA')

  processes = []
  qs = []
  for i in range(ni):
    qs.append(mp.Queue())
    processes.append(mp.Process(target=start_kern, args=[qs[-1], bx[i], i]))

  trace_diff = TraceDiff(qs)

  for x in processes:
    x.start()

  done = False
  while True:
    while trace_diff.can():
      trace_diff.step()

    if done:
      break

    done = True
    for x in processes:
      if x.is_alive():
        done = False
  for x in processes:
    x.join()


def test1():
  if flags.buf:
    buf = flags.buf.encode()
  else:
    buf = b'dd7d9509abcdefaa3333334499933551'
  start_kern(None, buf, 0)


class Solver:

  def __init__(self):
    mu = Uc(UC_ARCH_X86, UC_MODE_64)
    elf = load_elf(mu, flags.file)
    kern = Kernel(mu, None)

    perms = []
    base = 0x7ffff7dc70ac
    for i in range(4):
      tb = []
      for j in range(0x20):
        x = kern.mem.get_u8(base)
        tb.append(x)
        base += 1
      perms.append(tb)
    self.op_params = []
    self.op_params.append(Attributize(xorv=0x20bfdf15,
                                      addv=0x762defc2,
                                      shiftv=0x7,
                                      perm=perms[0],
                                      add0=0x19bf70ca))
    self.op_params.append(Attributize(xorv=0x83e2c81b,
                                      addv=0x4525e7b8,
                                      shiftv=0xd,
                                      perm=perms[1],
                                      add0=0xabb456d0))
    self.op_params.append(Attributize(xorv=0xd4d084ba,
                                      addv=0x3f5be319,
                                      shiftv=0x11,
                                      perm=perms[2],
                                      add0=0xdd1d76ca))
    self.op_params.append(Attributize(xorv=0xa6872cbd,
                                      addv=0xf59e62f,
                                      shiftv=0x19,
                                      perm=perms[3],
                                      add0=0x17d7d01d))

    in_test = [0x368f38be, 0xabb456d1, 0xdd1d76ca, 0x17d7d01d,]
    out_test = [0x16aacfd6, 0x2fc42297, 0xf045384d, 0x2722cb0f]
    assert out_test == self.full(in_test)
    assert in_test == self.ifull(out_test)
    test_full = [0x1ccfc7f4, 1, 0, 0]
    res = self.tsf(test_full)
    assert test_full == self.itsf(res)

    self.ans = [0xdd78363f, 0x2f19e578, 0x3ad92c4f, 0x2b6917a5]
    res = self.itsf(self.ans)

    res[3] = self.find_s3(res)
    assert self.checksum(res)==W1_CMP
    res = ['{:08x}'.format(x) for x in res]
    print(res)
    print(''.join(res))

    buf = '1ccfc7f400000001'+('0'*16)
    buf=flags.buf
    if buf:
      lst=list(struct.unpack('>4I', binascii.unhexlify(buf)))
      print([hex(x) for x in lst])
      print([hex(x) for x in self.tsf(lst)])
      print([hex(x) for x in self.tsf(lst, 0)])
      print([hex(x) for x in self.tsf(lst, 1)])


  def checksum(self, tb):
    v=(0xb00bface^tb[0]^tb[1])-(0xdeadbeef^tb[2]^tb[3])
    v=rotlu32(modu32(v), 0x11)
    return v

  def find_s3(self, tb):
    need = modu32(-rotlu32(W1_CMP, -0x11) + (0xb00bface ^ tb[0] ^ tb[1]))
    need = need ^ 0xdeadbeef ^ tb[2]
    return need

  def do_round(self, v, k):
    nv = 0
    op = self.op_params[k]
    for i in range(0x20):
      b = v >> op.perm[i] & 1
      nv = nv | (b << i)
    nv = nv ^ op.xorv
    nv = rotlu32(nv, op.shiftv)
    nv = modu32(nv + op.addv)
    return nv

  def do_iround(self, v, k):
    op = self.op_params[k]

    v = modu32(v - op.addv)
    v = rotlu32(v, -op.shiftv)
    v = v ^ op.xorv
    nv = 0
    for i in range(0x20):
      b = v >> i & 1
      nv = nv | (b << op.perm[i])
    return nv

  def _full(self, vals, cb):
    res = []
    for i in range(len(vals)):
      res.append(cb(vals[i], i))
    return res

  def full(self, vals):
    return self._full(vals, self.do_round)

  def ifull(self, vals):
    return self._full(vals, self.do_iround)

  def tsf(self, vals, nr=0x80):
    vals[0]^=vals[2]
    vals[1]^=vals[3]
    res = []
    for i in range(len(vals)):
      v = vals[i]
      v = modu32(v + self.op_params[i].add0)
      for j in range(nr):
        v = self.do_round(v, i)
      res.append(v)
    return res

  def itsf(self, vals):
    res = []
    for i in range(len(vals)):
      v = vals[i]
      for j in range(0x80):
        v = self.do_iround(v, i)
      v = modu32(v - self.op_params[i].add0)
      res.append(v)
    res[0]^=res[2]
    res[1]^=res[3]
    return res


def solve():
  solver = Solver()


def main():
  action = flags.action
  if action == 0:
    solve()
  elif action == 1:
    test_diff()
  else:
    test1()


app()
