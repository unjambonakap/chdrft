from capstone.x86_const import *
from chdrft.emu.base import Regs, Memory, BufferMemReader
from chdrft.emu.code_db import code
from chdrft.emu.elf import ElfUtils, MEM_FLAGS
from chdrft.emu.structures import Structure
#from chdrft.emu.syscall import SyscallDb, g_sys_db
from chdrft.emu.trace import Tracer, Display, WatchedMem, WatchedRegs
from chdrft.utils.math import rotlu32, modu32
from chdrft.utils.misc import cwdpath, Attributize, Arch, opa_print, to_list, lowercase_norm
import unicorn as uc
from unicorn.x86_const import *
import capstone.x86_const as cs_x86_const
import binascii
import ctypes
import jsonpickle
import pprint as pp
import struct
import sys
import traceback as tb
import glog


def safe_hook(func):

  def hook_func(uc, *args):
    try:
      func(uc, *args)
    except Exception as e:
      tb.print_exc()
      print('FUUU')
      sys.exit(0)
      print('Got exceptino , stopping emu>> ', e)
      uc.emu_stop()

  return hook_func

def load_elf(mu, fil):
  assert fil
  elf = ElfUtils(fil, core=True)
  need_load = []
  for seg in elf.elf.iter_segments():
    s = Attributize(seg.header)
    if s.p_type == 'PT_LOAD' and s.p_memsz > 0:
      need_load.append(s)
  flag_mp = [(MEM_FLAGS.PF_X, uc.UC_PROT_EXEC),
             (MEM_FLAGS.PF_R, uc.UC_PROT_READ),
             (MEM_FLAGS.PF_W, uc.UC_PROT_WRITE),]
  for s in need_load:
    flag_mem = 0
    for seg_flag, uc_flag in flag_mp:
      if seg_flag & s.p_flags:
        flag_mem = flag_mem | uc_flag
    print('MAPPING ', s, flag_mem)
    addr = s.p_vaddr
    sz = s.p_memsz
    align = s.p_align
    assert addr % align == 0
    sz = (sz + align - 1) & ~(align - 1)

    print('mapping ', hex(addr), hex(addr + sz), hex(sz))
    mu.mem_map(addr, sz, flag_mem)

    content = elf.get_seg_content(s)
    mu.mem_write(addr, content)
  for note in elf.notes:
    if note.n_type != 'NT_PRSTATUS':
      continue

    for r, v in note.status.pr_reg.items():
      if r not in regs:
        print('could not load reg ', r)
        continue
      mu.reg_write(regs[r], v._val)

      print('load ', r, hex(v._val), hex(mu.reg_read(regs[r])))

  return elf


class UCRegReader(Regs):

  def __init__(self, arch, mu):
    super().__init__(arch, mu=mu)

  def reg_code(self, name):
    return self.arch.uc_regs[name]

  def get_register(self, name):
    return self.mu.reg_read(self.reg_code(name))

  def set_register(self, name, val):
    return self.mu.reg_write(self.reg_code(name), val)


def round_down(v, p):
  return (v // p) * p


class Kernel:
  @staticmethod
  def LoadFromElf(arch, elf, stack_params=None, heap_params=None):
    mu = uc.Uc(arch.uc_arch, arch.uc_mode)
    elf = load_elf(mu, elf)
    kern = Kernel(mu, arch)
    kern.regs[arch.reg_pc] = elf.get_entry_address()

    if stack_params:
      print('MAPPING stack ', stack_params)
      if stack_params[1] < 0: 
        stack_params = list(stack_params)
        stack_params[1] = -stack_params[1]
        stack_params[0] -= stack_params[1]
      kern.regs[arch.reg_stack] = stack_params[0] + stack_params[1]
      mu.mem_map(stack_params[0], stack_params[1], uc.UC_PROT_READ | uc.UC_PROT_WRITE)

    if heap_params:
      print('MAPPING heap ', heap_params)
      mu.mem_map(heap_params[0], heap_params[1], uc.UC_PROT_READ | uc.UC_PROT_WRITE)
    return kern, elf


  def __init__(self, mu, arch, sigret_count_lim=100, read_handler=None):
    self.sigret_count_lim = sigret_count_lim
    self.log_file = None
    self.mem = Memory(self.byte_reader, self.write_mem)
    self.mu = mu
    self.arch = arch
    self.regs = UCRegReader(self.arch, self.mu)
    #self.syscalls = g_sys_db.data[Arch.x86_64]
    #self.syscall_handlers = {'mmap': self.mmap_handler,
    #                         #'__rt_sigreturn': self.sigreturn_handler,
    #                         'write': self.write_handler}
    #if read_handler:
    #  self.syscall_handlers['read']=read_handler

    #self.syscall_conv = {}
    #self.syscall_conv[Arch.x86_64] = [X86_REG_RDI, X86_REG_RSI, X86_REG_RDX, X86_REG_RCX,
    #                                  X86_REG_R8, X86_REG_R9]
    #self.structure_reader = StructureReader(self.byte_reader)
    self.sig_handlers = {}
    self.kill_in = -1
    self.sigret_count = 0
    self.save_buf = None
    self.info = []
    self.ret_map = {}
    self.want_stop = 0
    self.ignore_mem_access = False
    self.prev_ins=None

    watched = []
    sz = 4
    watched.append(WatchedMem('all',
                              self.mem,
                              0,
                              n=10,
                              all=True))
    watched.append(WatchedMem('input',
                              self.mem,
                              0x10ec00,
                              n=30,
                              sz=1))
    #watched.append(WatchedMem('stack3', self.mem, 0x7ffff7dc7000, n=0x200 // sz, sz=sz))
    #watched.append(WatchedRegs('regs', self.regs, to_list('r12 rip rbx rsp xmm0 xmm1 xmm2')))
    self.watchers = watched

    #for k, v in self.syscall_handlers.items():
    #  assert k in self.syscalls.entries, k

    self.tracer = Tracer(self.arch, self.regs, self.mem, watched)

  def dump_log(self):
    return
    self.info.append(str(self.ret_map))
    res = '\n'.join(self.info)
    open(self.log_file, 'w').write(res)

  def start(self):
    glog.info('Starting emulator at pc=%x', self.regs[self.arch.reg_pc])
    self.mu.emu_start(self.regs[self.arch.reg_pc], -1)

  def stop(self):
    print('stopping emu')
    self.want_stop = 1
    self.mu.emu_stop()

  def write_mem(self, addr, buf):
    self.mu.mem_write(addr, buf)

  def byte_reader(self, addr, n):
    return self.mu.mem_read(addr, n)

  def qword_read(self, addr):
    return struct.unpack('<Q', self.byte_reader(addr, 8))[0]

  def hook_code(self, mu, address, size, _2):
    glog.info('hook code %s %s', address, size)

    if self.want_stop:
      self.stop()
    mem = self.byte_reader(address, size)
    self.ignore_mem_access=False
    if self.prev_ins == mem and mem==b'\x00\x00':
      self.ignore_mem_access = True
      return
    self.prev_ins=mem
    self.tracer.notify_ins(address, size)
    glog.info('\n\n')
    mem = bytes(mem)
    glog.info('CODE at %s %s', hex(address), binascii.hexlify(mem))
    if self.kill_in == 0:
      self.stop()
    self.kill_in -= 1

    data = self.arch.mc.get_ins(mem, address)
    ins0 = data[0]
    for ins in data:
      glog.info("0x%x:\t%s\t%s %s" % (ins.address, ins.mnemonic, ins.op_str, ins.bytes))


  def hook_unmapped(self, mu, access, address, size, value, _2):
    print('BAD access at ', hex(address), access, size, value)
    return False

  def hook_mem_access(self, mu, access, address, size, value, user_data):
    if self.ignore_mem_access:
      return
    cur = self.regs[self.arch.reg_pc]
    s = ''
    if access == uc.UC_MEM_WRITE:
      self.tracer.notify_write(address, size, value)
      s = 'hook_access_write: ip={:x} addr={:x}, size={}, val={:x}'.format(cur, address, size,
                                                                           value)
    else:
      self.tracer.notify_read(address, size)
      s = 'hook_access_read: ip={:x} addr={:x}, size={}, content={:x}'.format(
          cur, address, size, self.mem.read_u(address, size))
    self.info.append(s)

  #def hook_intr(self, _1, intno, _2):
  #  if intno != 3:
  #    return
  #  self.tracer.end_vm_op()
  #  cur = self.regs[self.arch.reg_pc] + 3
  #  self.proc_signal(code.g_code.consts.SIGTRAP, intno)

  def hook_syscall(self, _1, _2):
    res = self.get_syscalls_args()
    print('GOT HOOK >> ', res.func.func_name)
    if res.func.func_name in self.syscall_handlers:
      self.syscall_handlers[res.func.func_name](res)
    else:
      assert 0, res.func.func_name

  def get_syscalls_args(self):
    syscall_num = self.mu.reg_read(uc.UC_X86_REG_RAX)
    syscall = self.syscalls.by_num[syscall_num]
    args = []
    for i in range(len(syscall.args)):
      args.append(self.mu.reg_read(self.syscall_conv[Arch.x86_64][i]))
    print(args)

    res = self.structure_reader.parse(syscall, args)
    return Attributize(func=syscall, args=res)

  def mmap_handler(self, data):
    print('mmap handler', opa_print(data.args))

  def print_regs(self):
    print(self.watched_regs.summary())

  def do_sigaction(self, signum, action, old_action, size_sigset):
    print('DODO sigaction', signum, action)
    self.sig_handlers[signum._val] = action

  def write_handler(self, data):
    print('write handler KAPAPA', data, self.sigret_count)
    buf = self.mem.read(data.args[1].val._val, data.args[2].val._val)
    print('WRITING >> ', buf)
    self.info.append('WRITING >> ' + bytes(buf).decode())
    self.dump_log()

    #self.stop()

  def disp_watchers(self):
    s = []
    for watcher in self.watchers:
      s.append(Display.disp(watcher.snapshot(),
                            name=watcher.diff_params.name,
                            params=watcher.diff_params))
    s = '\n'.join(s)
    print(s)

  def get_sigframe_ptr(self):
    sp = self.regs.rsp
    sp -= 128
    fpstate = 0
    math_size = 0

    if 1:
      buf_fx = 0
      #consider use_xsave=1
      xstate_size = 128 + 0x2c0
      magic2_size = 4
      xstate_sigframe_size = magic2_size + xstate_size
      math_size = xstate_sigframe_size
      sp = round_down(sp - math_size, 64)
      fpstate = sp
    sp -= code.g_asm.typs.rt_sigframe.size

    sp = (sp // 16) * 16 - 8
    return sp, fpstate, math_size

  def setup_sigcontext(self, context, fpstate, regs, mask, trapno):
    context.rdi._val = regs.rdi
    context.rsi._val = regs.rsi
    context.rbp._val = regs.rbp
    context.rsp._val = regs.rsp
    context.rbx._val = regs.rbx
    context.rdx._val = regs.rdx
    context.rcx._val = regs.rcx
    context.r8._val = regs.r8
    context.r9._val = regs.r9
    context.r10._val = regs.r10
    context.r11._val = regs.r11
    context.r12._val = regs.r12
    context.r13._val = regs.r13
    context.r14._val = regs.r14
    context.r15._val = regs.r15
    context.rax._val = regs.rax

    err = 0
    context.trapno._val = trapno
    context.err._val = err
    context.rip._val = regs.rip

    context.eflags._val = regs.eflags
    context.cs._val = regs.cs
    context['__pad2']._val = 0
    context['__pad1']._val = 0

    context.fpstate._val = fpstate
    context.oldmask._val = mask
    #pagefault address of last sigsegv?
    cr2 = 0x1338adac
    context.cr2._val = cr2

  def proc_signal(self, signum, trapno):
    assert signum in self.sig_handlers
    #print('gogo proc signal ', signum, hex(self.regs.rsp))
    handler = self.sig_handlers[signum].pointee
    #print(hex(handler.sa_restorer._val))
    self.prev_rip = self.regs.rip

    sigset_struct = Structure(code.g_asm.typs.sigset_t)

    #restorer at
    siginfo_struct = code.g_asm.typs.siginfo_t
    #print(siginfo_struct)

    sigframe_ptr, fpstate, math_size = self.get_sigframe_ptr()
    #print('math size >> ', math_size)
    sigframe_typ = code.g_asm.typs.rt_sigframe
    sigframe_v = Structure(sigframe_typ)

    # copy siginfo
    sigframe_v.info.si_signo._val = signum
    sigframe_v.info.si_errno._val = 0
    sigframe_v.info.si_code._val = code.g_asm.consts.SI_KERNEL

    #xstate stays at 0 for the moment
    sigframe_v.uc.uc_stack._val = self.regs.sp
    #oh god sigset multiple defs rofl
    sigframe_v.pretcode._val = handler.sa_restorer._val

    sigframe_v.uc.uc_flags._val = code.g_asm.consts.UC_FP_XSTATE
    sigframe_v.uc.uc_link._val = 0
    sigframe_v.uc.uc_stack.ss_flags._val = code.g_asm.consts.SS_DISABLE

    self.setup_sigcontext(sigframe_v.uc.uc_mcontext, fpstate, self.regs, sigset_struct._val, trapno)
    sigframe_v.uc.uc_sigmask = sigset_struct

    buf = sigframe_v.to_buf()
    self.save_buf = buf
    #print(type(buf))
    self.write_mem(sigframe_ptr, buf)

    self.regs.rdi = signum
    self.regs.rax = 0

    self.regs.rsi = sigframe_ptr + sigframe_typ.fields.info.off
    self.regs.rdx = sigframe_ptr + sigframe_typ.fields.uc.off
    self.regs.rip = handler.sa_handler._val
    #print('GOT RIP >> ', hex(self.regs.rip))
    self.regs.rsp = sigframe_ptr
    #print('GOT rsp >> ', hex(self.regs.rsp))
    #for r in self.want_regs:
    #  print(r, hex(self.regs[r]))
    #sys.exit(0)

    #=> 0x7ffff7a31680:      mov    rax,0xf
    #   0x7ffff7a31687:      syscall


  def sigreturn_handler(self, args):
    self.sigret_count += 1
    frame_ptr = self.regs.rsp - 8
    s1 = Display.mem_summary(self.mem, self.regs.rsp, 30, word_size=8, istart=-1)
    #print(s1)
    #self.info.append('STACK STATE')
    #self.info.append(s1)

    #print('sigreturn handler', hex(frame_ptr))
    frame = self.structure_reader.parse_ptr(code.g_asm.typs.rt_sigframe, frame_ptr)
    #print('READING UC MASK XX ', frame.uc.uc_sigmask._val)
    retv = self.restore_sigcontext(frame.uc.uc_mcontext)
    self.restore_altstack(frame.uc.uc_stack)

    #oregs = Display.regs_summary(self.regs)
    #print('REGS>>> ')
    #print(oregs)
    #self.info.append('RESTORE REGS ')
    #self.info.append(oregs)

    if self.regs.rip in self.ret_map:
      #assert self.regs.rax == self.ret_map[self.regs.rip], self.sigret_count
      pass
    self.ret_map[self.regs.rip] = self.regs.rax
    #print(hex(self.regs.rip), 'was called at ', hex(self.prev_rip))

    #self.info.append('\n\n\n')
    self.tracer.start_vm_op()
    print('ON ', self.sigret_count)

    if self.sigret_count == self.sigret_count_lim:
      self.dump_log()
      print('EXIT BECAUSE COUNT', self.sigret_count)
      self.stop()
    #print('Status of watchers === ')
    #self.disp_watchers()

    #print('\n\n\n\nDONE SIGRETURN')
    #pp.pprint(frame)
    #print(frame.typ.gets())

    #if self.regs.rip==0x1337295d:
    #  sys.exit(0)
    #self.kill_in=1000

  def restore_altstack(self, stack):
    pass
    #print('restoring alt stack to ', stack)
    #print('is this shit used?')
    #asmlinkage long sys_rt_sigreturn(void)
    #{
    #       struct pt_regs *regs = current_pt_regs();
    #       struct rt_sigframe __user *frame;
    #       unsigned long ax;
    #       sigset_t set;
    #
    #       frame = (struct rt_sigframe __user *)(regs->sp - sizeof(long));
    #       if (!access_ok(VERIFY_READ, frame, sizeof(*frame)))
    #               goto badframe;
    #       if (__copy_from_user(&set, &frame->uc.uc_sigmask, sizeof(set)))
    #               goto badframe;
    #
    #       set_current_blocked(&set);
    #
    #       if (restore_sigcontext(regs, &frame->uc.uc_mcontext, &ax))
    #               goto badframe;
    #
    #       if (restore_altstack(&frame->uc.uc_stack))
    #               goto badframe;
    #
    #       return ax;
    #
    #badframe:
    #       signal_fault(regs, frame, "rt_sigreturn");
    #       return 0;
    #}

  def restore_sigcontext(self, sc):
    to_copy = to_list('rdi rsi rbp rsp rbx rdx rcx rip r8 r9 r10 r11 r12 r13 r14 r15 rax')
    for r in to_copy:
      self.regs[r] = sc[r]._val
    # orig_rax=-1 ?
    self.regs.eflags = sc.eflags._val
    #print('Got rax with ', self.regs.rax)
    #print('ret at ', hex(self.regs.rip))

    return sc.rax
##define FIX_EFLAGS     (X86_EFLAGS_AC | X86_EFLAGS_OF | \
#                        X86_EFLAGS_DF | X86_EFLAGS_TF | X86_EFLAGS_SF | \
#                        X86_EFLAGS_ZF | X86_EFLAGS_AF | X86_EFLAGS_PF | \
#                        X86_EFLAGS_CF | X86_EFLAGS_RF)

#int restore_sigcontext(struct pt_regs *regs, struct sigcontext __user *sc,
#                      unsigned long *pax)
#{
#       void __user *buf;
#       unsigned int tmpflags;
#       unsigned int err = 0;
#
#       /* Always make any pending restarted system calls return -EINTR */
#       current_thread_info()->restart_block.fn = do_no_restart_syscall;
#
#       get_user_try {
#
##ifdef CONFIG_X86_32
#               set_user_gs(regs, GET_SEG(gs));
#               COPY_SEG(fs);
#               COPY_SEG(es);
#               COPY_SEG(ds);
##endif /* CONFIG_X86_32 */
#
#               COPY(di); COPY(si); COPY(bp); COPY(sp); COPY(bx);
#               COPY(dx); COPY(cx); COPY(ip);
#
##ifdef CONFIG_X86_64
#               COPY(r8);
#               COPY(r9);
#               COPY(r10);
#               COPY(r11);
#               COPY(r12);
#               COPY(r13);
#               COPY(r14);
#               COPY(r15);
##endif /* CONFIG_X86_64 */
#
##ifdef CONFIG_X86_32
#               COPY_SEG_CPL3(cs);
#               COPY_SEG_CPL3(ss);
##else /* !CONFIG_X86_32 */
#               /* Kernel saves and restores only the CS segment register on signals,
#                * which is the bare minimum needed to allow mixed 32/64-bit code.
#                * App's signal handler can save/restore other segments if needed. */
#               COPY_SEG_CPL3(cs);
##endif /* CONFIG_X86_32 */
#
#               get_user_ex(tmpflags, &sc->flags);
#               regs->flags = (regs->flags & ~FIX_EFLAGS) | (tmpflags & FIX_EFLAGS);
#               regs->orig_ax = -1;             /* disable syscall checks */
#
#               get_user_ex(buf, &sc->fpstate);
#
#               get_user_ex(*pax, &sc->ax);
#       } get_user_catch(err);
#
#       err |= restore_xstate_sig(buf, config_enabled(CONFIG_X86_32));
#
#       return err;
#}

  def hook_signal(self):
    pass
