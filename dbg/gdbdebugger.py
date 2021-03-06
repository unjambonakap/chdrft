import re
import binascii
import sys
import struct
import subprocess as sp
import os
import glog
from contextlib import ExitStack
import traceback as tb

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
from chdrft.emu.base import Memory, Regs, Stack
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
    self.regs = GdbReg(self.arch, self)
    self.mem = Memory(self.get_memory, self.set_memory, arch=self.arch)
    self.stack = Stack(self.regs, self.mem, self.arch)
    self.reason = ''
    self.set_stop_handler(lambda x: None)
    self.regs_watch = WatchedRegs('all', self.regs, self.arch.regs)
    self.tracer = None
    if tracer_cb is not None:
      self.tracer = Tracer(
          self.arch, self.regs, self.mem, [self.regs_watch], cb=tracer_cb, diff_mode=diff_mode
      )

  def __enter__(self):
    super().__enter__()
    self.callback(lambda: self.do_execute('disconnect'))

  def set_arch(self):
    s = self.do_execute('show architecture')
    print('arch >> ', s)
    self.arch = guess_arch(s)

  def set_aslr(self, val=True):
    self.do_execute('set disable-randomization %s' % (['on', 'off'][val]))

  def del_bpts(self):
    if self.gdb.breakpoints() is not None:
      for bpt in self.gdb.breakpoints():
        bpt.delete()

  def del_bpt(self, bpt):
    if isinstance(bpt, int):
      self.do_execute(f'del {bpt}')
    else:
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

  def get_elf(self):
    return ElfUtils(self.get_file_path())
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

  def bp_generator(self, cb=None, should_stop=True):
    mainSelf = self

    class BpWithCallback(self.gdb.Breakpoint):
      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.silent = True

      def stop(self):
        if should_stop: super().stop()
        if cb: return cb(mainSelf)

    return BpWithCallback

  def set_bpt(self, ea, cb=None, hard=False):
    if hard: return self.set_hard_bpt_exe(ea)
    return self.bp_generator(cb)("*0x%x" % ea)

  def set_hard_bpt(self, ea, cb=None):
    return self.bp_generator(cb)("*0x%x" % ea, self.gdb.BP_WATCHPOINT, self.gdb.WP_ACCESS)

  def set_hard_bpt_exe(self, ea, cb=None):
    res = self.do_execute('hbreak *0x%x' % ea)
    return int(re.search('Hardware assisted breakpoint (\d+) at', res).group(1))


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

  def get_ins_data(self, pc=None):
    if pc is None: pc = self.get_pc()
    data = self.mem.read(pc, self.arch.ins_size)
    return data

  def get_ins_str(self, pc=None):
    if pc is None: pc = self.get_pc()
    return self.arch.mc.ins_str(self.get_ins_data(pc), addr=pc)

  def get_ins_str_gdb(self, pc=None):
    if pc is None: pc = self.get_pc()
    return self.do_execute(f'x/i {pc}').strip()

  def trace_to(self, target_pc, disp_ins=0, regs=[], max_ins=-1, mem=[], funcs={}, want_prev_regs=1):
    if isinstance(target_pc, int):
      target_pc = set([target_pc])

    inslist = []
    prev_regs = []
    while True:
      if max_ins == 0: break
      max_ins -= 1
      self.step_into()
      cpc = self.regs.pc

      if disp_ins:

        u = self.get_ins_info(cpc, regs=regs, mem=mem,funcs=funcs, prev_regs=prev_regs)
        if want_prev_regs: prev_regs = u.prev_regs
        inslist.append(u.res)
      else: inslist.append(cpc)

      if cpc in target_pc:
        break
    return inslist

  def get_ins_info(self, cpc, regs=[], mem=[], funcs={}, prev_regs=[]):
    ins_data = self.get_ins_data(cpc)
    ins_str = self.arch.mc.ins_str(ins_data, addr=cpc)
    ins_str2 = self.get_ins_str_gdb(pc=cpc)

    res = cmisc.Attr(pc=cpc, ins_str2=ins_str2, ins_data=ins_data, failed=0, data={})

    try:
      ins = self.arch.mc.get_one_ins(ins_data, addr=cpc)
      regcheck= self.arch.mc.get_reg_ops(ins)
    except:
      regcheck=['v0', 'v1', 'v2']
      res.failed = 1

    nregs = []
    for reg in regcheck:
      if reg in ('fp',): continue
      if reg[0] == 'v':
        nregs.append(reg + '.q.s[0]')
        nregs.append(reg + '.q.u[0]')
      else:
        nregs.append(reg)

    cregs = list(nregs) + regs
    cregs.extend(prev_regs)
    prev_regs = nregs


    cmem = list(mem)
    if cpc in funcs:
      cf = funcs[cpc]
      cmem.extend(cf.get('mem', []))
      cregs.extend(cf.get('regs', []))
      res.name = cf.name

    res.regs=  self.snapshot_regs(cregs, main_obj=res.data, failsafe=1)
    res.mem=  self.snapshot_mem(cmem, res.data)

    return cmisc.Attr(res=res, prev_regs=prev_regs)

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

  def set_reg(self, reg ,v):
    return self.regs.set(reg, v)

  def get_reg(self, reg):
    return self.regs.get(reg)

  def get_regs(self):
    return self.regs

  def get_register(self, reg):
    cmd = '$%s' % reg
    res = self.gdb.parse_and_eval(cmd)
    try:
      return int(str(res), 16)
    except:
      glog.info('Failed to get %s'%reg)
      assert False, 'Failed to get reg %s'%reg
      raise

  def set_register(self, reg, val):
    self.gdb.parse_and_eval('$%s = 0x%x' % (reg, val))

  def get_instruction(self, pos=None):
    if pos == None:
      pos = self.gdb.selected_frame().pc()
    res = self.gdb.selected_frame().architecture().disassemble(pos)[0]['asm']
    return res

  def reg(self):
    return lambda x: self.get_register(x)

  def snapshot_mem(self, q, main_obj=None):
    res = {}
    for x in q:
      count = 1
      if isinstance(x, int):
        res[f'mem_{m:x}'] = cmisc.failsafe_or(lambda : self.mem.get_ptr(x), 0)
      else:
        reg, n, off, count  = x.reg, x.size, x.offset, x.get('count', 1)

        addr=self.regs[reg]+off
        ans = list(self.mem.read_n_u(addr, n, count))
        res[f'mem_{reg}+0x{off:x}(={addr:x}'] = ans
        if main_obj is not None and 'name' in x: main_obj[x.name] = ans

    return res

  def snapshot_regs(self, q, failsafe=0, main_obj=None):
    tmp = set()
    regname_to_name = {}
    for r in q:
      if isinstance(r, str):
        tmp.add(r)
      else:
        tmp.add(r.reg)
        if 'name' in r: regname_to_name[r.reg] = r.name
    regs = self.regs.snapshot(tmp, failsafe=failsafe)

    if main_obj is not None:
      for r, name in regname_to_name.items():
        main_obj[name] = regs[r]
    return regs

  def get_snapshot(self, nstack=0):
    regs = self.regs.snapshot_all()
    mem = []
    for i in range(nstack):
      mem.append(self.stack.get(i))
    return cmisc.Attr(regs=regs, mem=mem)


  def get_memory(self, addr, l):
    ntry = 5
    for i in range(ntry):
      try:
        res = self.gdb.inferiors()[0].read_memory(addr, l)
      except Exception as e:
        if i == ntry-1:
          raise

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
    if isinstance(bp, int):
      assert 0
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
    mc = MachineCaller(self.arch, self.regs, self.mem, self.runner())
    return mc

  def run_to(self, addr, **kwargs):
    bpt = self.set_bpt(addr, lambda _: True, **kwargs)
    self.resume()
    assert self.is_bpt_active(bpt)
    self.del_bpt(bpt)
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


def launch_gdb(module, func, dbg_file=None, args=[], gdb_cmd='gdb', gdb_args=[], nowait=False):
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

    devnull = open(os.devnull, 'r')
    p = sp.Popen(cmd, stdin=devnull)
    if nowait:return p
    p.wait()

  except:
    try:
      pid = p.pid
      tb.print_exc()
      print('>>>> TRY KIOLL', pid)
      failsafe(lambda: p.kill())
      failsafe(lambda: sp.call(['kill', '-9', str(pid)]))
      print('DONE KILL')
    except:
      tb.print_exc()
      pass
