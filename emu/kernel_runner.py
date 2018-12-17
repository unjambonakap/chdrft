#!/usr/bin/env python


from chdrft.cmds import CmdsList
from chdrft.emu.binary import arch_data, Arch
from chdrft.emu.elf import ElfUtils
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.fmt import Format
from chdrft.utils.misc import Attributize
import binascii
import chdrft.emu.kernel as kernel
import chdrft.emu.structures as Structures
import chdrft.emu.trace as trace
import chdrft.utils.misc as cmisc
import os
import re
import struct
import traceback as tb
import unicorn as uc
import glog
from chdrft.emu.func_call import AsyncMachineCaller, FuncCallWrapperGen, FunctionCaller, AsyncFunctionCaller, SimpleBufGen
from chdrft.emu.structures import StructBuilder, CodeStructExtractor, Structure, StructBackend, g_data, MemBufAccessor

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=0)



class KernelRunner:

  def __init__(self, kern):
    self.kern = kern

  def __call__(self, end_pc):
    pass

class KernBaseImports:

  def __init__(self, kern):
    self.kern = kern
    self.mem = kern.mem
    self.regs = kern.regs
    self.mc = kern.mc
    for x in KernBaseImports.__dict__.keys():
      if not isinstance(x, str) or not x.endswith('_hook'): continue
      self.kern.defined_hooks[x[:-5]] = getattr(self, x)
    self.buffer = b''
    self.f = None

  def fseek_hook(self, _):
    stream = self.regs.rdi
    offset = self.regs.rsi
    whence = self.regs.rdx
    self.f.seek(offset, whence)
    return 0

  def ftell_hook(self, _):
    return self.f.tell()

  def fread_hook(self, _):
    buf = self.regs.rdi
    sz = self.regs.rsi
    n = self.regs.rdx
    read = self.f.read(sz * n)
    self.mem.write(buf, read)
    return len(read)

  def fclose_hook(self, _):
    print('FCLOSE')
    self.f.close()
    return 0

  def malloc_hook(self, _):
    n = self.regs.rdi
    return self.kern.heap.alloc(n)

  def fopen_hook(self, _):
    pos = self.regs.rdi
    fname = self.mem.get_str(pos)
    self.f = open(fname, 'rb')
    assert 0

    return 4

  def strlen_hook(self, _):
    pos = self.regs.rdi
    res = len(self.mem.get_str(pos))
    return res

  def rand_hook(self, _):
    return libc.x.rand()

  def srand_hook(self, _):
    print('ON SRAND')
    libc.x.srand(self.regs.rdi)
    return 0

  def puts_hook(self, _):
    print('PUTS >> ', self.mem.get_str(self.regs.rdi))
    return 0

  def printf_hook(self, _):
    fmt = self.mem.get_str(self.regs.rdi)
    print('PRINTF >> ', fmt)
    return 0

  def memcmp_hook(self, _):
    a = self.regs.rdi
    b = self.regs.rsi
    n = self.regs.rdx
    if self.mem.read(a, n) == self.mem.read(b, n):
      return 0
    return 1

  def memcpy_hook(self, _):
    n = self.regs.rdx
    self.mem.write(self.regs.rdi, self.mem.read(self.regs.rsi, n))
    return n

  def fgets_hook(self, _):
    buf = self.regs.rdi
    n = self.regs.rsi

    print('FGETS ', n)
    assert 0
    cnd = self.buf[:n - 1]
    firstnl = self.buf.find(b'\n')
    if firstnl != -1:
      cnd = cnd[firstnl + 1]
    cnd += b'\x00'
    print('GETS ', cnd)
    self.mem.write(buf, cnd)
    if len(cnd) == 1: return 0
    return buf


class BaseSolver:

  def __init__(self, kern, handle=None, hook_intr=0):
    self.kern = kern
    self.regs = kern.regs
    self.mem = kern.mem
    self.stack = kern.stack
    self.kern.notify_hook_code = self.notify_hook_code

    if handle is None: handle=self.handle
    self.handler = handle(self)

    if hook_intr:
      kern.mu.hook_add(uc.UC_HOOK_INTR, kernel.safe_hook(self.hook_intr0))
      kern.hook_intr_func = self.hook_intr


  def enable_full_hook(self):
    code_low = 1
    code_high = 0
    stack_low = 1
    stack_high = 0
    self.kern.mu.hook_add(uc.UC_HOOK_CODE, kernel.safe_hook(self.kern.hook_code), None, code_low, code_high)
    self.kern.mu.hook_add(
        uc.UC_HOOK_MEM_READ | uc.UC_HOOK_MEM_WRITE, kernel.safe_hook(self.kern.hook_mem_access), None, 1, 0
    )

  def hook_addr(self, addr, handler=None):
    if handler is None: handler = self
    self.kern.hook_addr(addr, handler)

  def failure_hook(self, *args):
    print('ON FAILURE')
    self.kern.stop()

  def notify_hook_code(self, address):
    pass

  def hook_intr(self, intno, addr):
    pass

  def __call__(self, *args, **kwargs):
    next(self.handler)

  def handle(self, _):
    self.kern.stop()
    yield

  def run(self, ip):
    try:
      self.kern.start(ip=ip)
    except uc.UcError as e:
      tb.print_exc()


def event_handle(e, fevent, fop):
  glog.info('Got event %s', e)
  if isinstance(e, trace.TraceEvent):
    fevent.write(str(e) + '\n\n')

  elif isinstance(e, trace.VmOp):
    fevent.write(e.gets(regs=True) + '\n\n\n')
    fop.write(str(e) + '\n\n\n')
    fop.flush()
    fevent.flush()


def trace_event_handler(e):
  if isinstance(e, TraceEvent):
    print(str(e) + '\n\n')



def create_kern(solver_class=BaseSolver, runid='default', solver_kwargs={}, **kwargs):
  kern, elf = kernel.Kernel.LoadFrom(**kwargs)

  #kern.mu.hook_add(uc.UC_ERR_WRITE_PROT, kernel.safe_hook(kern.hook_bad_mem_access), None, 0, 2**32-1)
  kern.mu.hook_add(
      uc.UC_HOOK_MEM_WRITE_UNMAPPED | uc.UC_HOOK_MEM_FETCH_UNMAPPED,
      kernel.safe_hook(kern.hook_unmapped)
  )
  kern.tracer.ignore_unwatched = True
  #kern.tracer.watched.append(WatchedRegs('regs', regs, cmisc.to_list('rip rcx rdx rsp')))
  #kern.tracer.watched.append(WatchedMem('all', kern.mem, 0, n=2**32, sz=1))

  kern.tracer.diff_mode = False
  event_log = open(f'/tmp/evens_{runid}.out', 'w')
  vmop_log = open(f'/tmp/vmop_{runid}.out', 'w')
  kern.tracer.cb = lambda x: event_handle(x, event_log, vmop_log)

  solver = solver_class(kern, **solver_kwargs)
  kern.ignore_mem_access = False

  return solver


def test(ctx):
  g_data.set_m32(True)
  g_code = StructBuilder()
  #g_code.add_extractor(SimpleStructExtractor('./test.h', ''))
  g_code.add_extractor(CodeStructExtractor('int test(int a);', ''))
  g_code.build(extra_args=StructBuilder.opa_common_args())

  s = create_kern(arch='thumb')
  code = s.kern.machine.get_disassembly('''
nop
FUNC:
add %r0, #1
mov %pc, %lr

  adr %r2, START
  mov %r1, #12;
  mov %r1, #12;
  mov %r1, #12;
  mov %r1, #12;
  mov %r1, #12;
  .align
START:
  mov %r1, #0xff;
  ''')

  scratch_start = s.kern.scratch[0]
  s.mem.write(scratch_start, code)
  s.enable_full_hook()

  def handle(s):
    print('HANDLE LA')
    yield s.kern.fcaller.test(123)
    print('RESULT IS ', s.kern.fcaller.result())
    yield
    s.kern.stop()
    yield

  s.kern.fcaller.fcgen.code_db=g_code
  s.kern.fcaller.fcgen.name_to_addr= {'test': scratch_start+2}
  s.kern.forward_ret_hook = handle(s)
  s.hook_addr(scratch_start + len(code))



  s.run(scratch_start+1)

def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
