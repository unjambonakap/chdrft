import re
import binascii
import sys
import struct
import subprocess as sp
import os
import glog
from contextlib import ExitStack

#def fix_environ():
#  if 'VIRTUAL_ENV' in os.environ:
#    sys.path.append(os.path.join(os.environ['VIRTUAL_ENV'], 'lib/python3.5/site-packages'))
#  sys.path.extend(os.environ['PYTHONPATH'].split(':'))
#
#fix_environ()

from .debuggercommon import Status
from chdrft.utils.misc import SeccompFilters, failsafe, lowercase_norm
import chdrft.utils.misc as cmisc
from chdrft.emu.binary import guess_arch
from chdrft.emu.base import Memory, Regs
from chdrft.emu.elf import ElfUtils
from chdrft.emu.trace import WatchedRegs, WatchedMem, Tracer
from chdrft.emu.func_call import MachineCaller


class GdbReg(Regs):

  def __init__(self, arch, x):
    super().__init__(arch, x=x)

  def get_register(self, name):
    return self.x.get_register(name)

  def set_register(self, name, val):
    return self.x.set_register(name, val)


class GdbDebugger(ExitStack):

  def __init__(self, tracer_cb=None, diff_mode=False):
    super().__init__()
    import gdb
    self.statuses = Status
    self.gdb = gdb
    self.entry_regex = re.compile('Entry point: 0x([0-9a-f]+)', re.MULTILINE)
    self.stop_handler = None
    self.set_arch()
    self.reg = GdbReg(self.arch, self)
    self.mem = Memory(self.get_memory, self.set_memory)
    self.reason = ''
    self.set_stop_handler(lambda x: None)
    self.reg_watch = WatchedRegs('all', self.reg, self.arch.regs)
    self.tracer = None
    if tracer_cb is not None:
      self.tracer = Tracer(
          self.arch, self.reg, self.mem, [self.reg_watch], cb=tracer_cb, diff_mode=diff_mode
      )

  def __enter__(self):
    super().__enter__()
    self.callback(lambda: self.do_execute('disconnect'))

  def set_arch(self):
    s = self.do_execute('show architecture')
    self.arch = guess_arch(s)

  def set_aslr(self, val=True):
    self.do_execute('set disable-randomization %s' % (['on', 'off'][val]))

  def del_bpts(self):
    if self.gdb.breakpoints() is not None:
      for bpt in self.gdb.breakpoints():
        bpt.delete()

  def del_bpt(self, bpt):
    bpt.delete()

  def get_pid(self):
    return self.gdb.selected_thread().ptid[0]

  def do_execute(self, s):
    try:
      return self.gdb.execute(s, to_string=True)
    except KeyboardInterrupt as e:
      raise e
    except Exception as e:
      print('Execute exception >> ', e)
      return None

  def get_entry_address(self):
    if 1:
      return ElfUtils(self.get_file_path()).get_entry_address()
    else:
      s = self.do_execute('info files')
      print(s)
      addr = self.entry_regex.search(s).group(1)
      return int(addr, 16)

  def get_file_path(self):
    return self.gdb.current_progspace().filename
    #s = self.do_execute('info files')
    #return re.search("`([^']*)', file type", s).group(1)

  def bp_generator(self, cb=None):
    mainSelf = self

    class BpWithCallback(self.gdb.Breakpoint):

      def stop(self):
        print('ON BPt CALLBSC')
        if cb:
          return cb(mainSelf)

    return BpWithCallback

  def set_bpt(self, ea, cb=None):
    return self.bp_generator(cb)("*0x%x" % ea)

  def set_hard_bpt(self, ea, cb=None):
    return self.bp_generator(cb)("*0x%x" % ea, self.gdb.BP_WATCHPOINT, self.gdb.WP_ACCESS)

  def wait(self):
    pass

  def stop_debugger(self):
    self.do_execute('kill')

  def add_entry_bpt(self, cb=None):
    addr = self.get_entry_address()
    return self.set_bpt(addr, cb)

  def load_debugger(self):
    pass

  def set_stop_solib_events(self, val):
    self.do_execute('set stop-on-solib-events {0:d}'.format(val))

  def run(self, args='', silent=False):
    x = "r %s" % args
    if silent:
      x += ' 1>/dev/null'
    return self.do_execute(x)

  def run_managed_fifo(self, managed_fifo, args='', silent=False):
    return self.run_with_fifo(
        input=managed_fifo.write_fifo,
        output=managed_fifo.read_fifo,
      silent=silent,
        args=args,
    )

  def run_with_fifo(self, input=None, output=None, silent=False, args=''):
    x = "r %s" % args
    if input:
      x += ' <{}'.format(input)
    if output:
      x += ' >{}'.format(output)
    elif silent:
      x += ' 1>/dev/null'
    return self.do_execute(x)

  def resume(self):
    self.reason = self.do_execute('c')
    if self.reason is None:
      self.reason = ''
    return self.reason

  def kill(self):
    return self.do_execute('kill')

  def trace(self):
    if not self.tracer: return
    assert self.arch.ins_size != -1
    self.tracer.notify_ins(self.get_pc(), self.arch.ins_size)

  def step_into(self, nsteps=1):
    for i in range(nsteps):
      res = self.do_execute('si')
      self.trace()
    return res

  def show_context(self):
    print(self.do_execute('context'))

  def get_status(self):
    if len(self.gdb.inferiors()[0].threads()) > 0:
      th = self.gdb.inferiors()[0].threads()[0]
      if th.is_stopped():
        return Status.STOPPED
      elif th.is_exited():
        return Status.TERMINATED
      else:
        return Status.RUNNING
    else:
      return Status.TERMINATED

  def get_pc(self):
    res = self.gdb.parse_and_eval('$' + self.arch.reg_pc)
    res = str(res).split()[0]
    return int(res, 16)

  def get_sym_address(self, sym):
    res = self.gdb.parse_and_eval('&{0}'.format(sym))
    return int(str(res).split(' ')[0], 16)

  def get_register(self, reg):
    cmd = '$%s' % reg
    res = self.gdb.parse_and_eval(cmd)
    return int(str(res), 16)

  def set_register(self, reg, val):
    self.gdb.parse_and_eval('$%s = 0x%x' % (reg, val))

  def get_instruction(self, pos=None):
    if pos == None:
      pos = self.gdb.selected_frame().pc()
    res = self.gdb.selected_frame().architecture().disassemble(pos)[0]['asm']
    return res

  def reg(self):
    return lambda x: self.get_register(x)

  def get_memory(self, addr, l):
    res = self.gdb.inferiors()[0].read_memory(addr, l)
    res = [res[i] for i in range(l)]
    res = b''.join(res)
    #read-write buffer object
    return res

  def set_memory(self, addr, data):
    self.gdb.inferiors()[0].write_memory(addr, data)

  def get_eflags(self):
    res = self.gdb.parse_and_eval('$eflags')
    return int(res)

  def set_stop_handler(self, stop_handler):
    if self.stop_handler is not None:
      self.remove_stop_handler()

    def tmp(x):
      self.stop_event = x
      return stop_handler(self)

    self.stop_handler = tmp
    self.gdb.events.stop.connect(tmp)

  def remove_stop_handler(self):
    if self.stop_handler is not None:
      self.gdb.events.stop.disconnect(self.stop_handler)
      self.stop_handler = None

  def is_bpt_active(self, bp):
    if not self.is_bpt():
      return False
    for b in self.stop_event.breakpoints:
      if b == bp:
        return True
    return False

  def is_signal(self):
    return isinstance(self.stop_event, self.gdb.SignalEvent)

  def is_bpt(self):
    return isinstance(self.stop_event, self.gdb.BreakpointEvent)

  def get_signal_type(self):
    if not self.is_signal():
      return None
    return self.stop_event.stop_signal

  def runner(self):

    def func(end_pc):
      self.run_to(end_pc)

    return func

  def caller(self):
    mc = MachineCaller(self.arch, self.reg, self.mem, self.runner())
    return mc

  def run_to(self, addr):
    bpt = self.set_bpt(addr, lambda _: True)
    self.resume()
    assert self.is_bpt_active(bpt)
    return

  def extract_seccomp_filters(self, addr):
    l = self.mem.get_u16(addr)
    # only works for x86 atm (alignment)
    ptr = self.mem.get_ptr(addr + 4)
    tb = []

    sz = SeccompFilters.SIZE
    mem = x.get_memory(ptr, sz * l)
    for i in range(l):
      tb.append(SeccompFilters(mem[i * sz:i * sz + sz]))
    return tb

  def is_sigtrap_int(self):
    return self.has_signal('SIGTRAP')

  def has_signal(self, sig=None):
    m = re.search('Program received signal ([^,]*),', self.reason)
    if not m:
      return False
    return sig is None or m.group(1) == sig

  def is_syscall_trap(self, syscall=None):
    m = re.search('(call to|returned from) syscall ([^)]*)\),', self.reason)
    if not m:
      return False
    return syscall is None or m.group(2) == syscall


def launch_gdb(module, func, dbg_file=None, args=[], gdb_cmd='gdb', gdb_args=[]):
  try:
    python_cmd = """python
import sys
import os
import traceback as tb
sys.path.extend({path})
sys.path.append(os.getcwd())

import {module}
try:
    {module}.{func}(*{args});
except Exception as e:
    tb.print_exc()
finally:
    gdb.execute('q')
    pass
""".format(
        module=module, func=func, path=str(sys.path), args=str(args)
    )

    cmd = [gdb_cmd] + gdb_args
    cmd += ['-ex', python_cmd]
    if dbg_file: cmd += [dbg_file]

    print(cmd)
    devnull = open(os.devnull, 'r')
    p = sp.Popen(cmd, stdin=devnull)
    p.wait()

  except Exception as e:
    try:
      pid = p.pid
      tb.print_exc()
      print('>>>> TRY KIOLL', pid)
      failsafe(lambda: p.kill())
      failsafe(lambda: sp.call(['kill', '-9', str(pid)]))
      print('DONE KILL')
    except:
      pass
