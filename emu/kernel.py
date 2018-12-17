from capstone.x86_const import *
from chdrft.emu.base import Regs, Memory, BufferMemReader, RegContext, DummyAllocator, Stack
from chdrft.emu.code_db import code
from chdrft.emu.elf import ElfUtils, MEM_FLAGS
from chdrft.emu.structures import Structure, MemBufAccessor
from chdrft.emu.binary import norm_arch
from chdrft.emu.trace import Tracer, Display, WatchedMem, WatchedRegs
from chdrft.utils.opa_math import rotlu32, modu32
from chdrft.utils.misc import cwdpath, Attributize, Arch, opa_print, to_list, lowercase_norm
import chdrft.utils.misc as cmisc
import unicorn as uc
from unicorn.x86_const import *
import capstone.x86_const as cs_x86_const
import binascii
import jsonpickle
import pprint as pp
import struct
import sys
import traceback as tb
import glog
from chdrft.emu.func_call import MachineCaller, mem_simple_buf_gen, AsyncMachineCaller, FunctionCaller, AsyncFunctionCaller
import re
from collections.abc import Iterable


def safe_hook(func):

  def hook_func(uc, *args):
    try:
      func(uc, *args)
    except KeyboardInterrupt as kb:
      tb.print_exc()
      uc.emu_stop()
    except Exception as e:
      tb.print_exc()
      print('FUUU')
      sys.exit(0)
      print('Got exceptino , stopping emu>> ', e)
      uc.emu_stop()

  return hook_func


def load_elf(kern, fil):
  assert fil
  if not isinstance(fil, ElfUtils):
    elf = ElfUtils(fil)
  else:
    elf = fil
  need_load = []
  for seg in elf.elf.iter_segments():
    s = Attributize(seg.header)
    if s.p_type == 'PT_LOAD' and s.p_memsz > 0:
      need_load.append(s)
  flag_mp = [
      (MEM_FLAGS.PF_X, uc.UC_PROT_EXEC),
      (MEM_FLAGS.PF_R, uc.UC_PROT_READ),
      (MEM_FLAGS.PF_W, uc.UC_PROT_WRITE),
  ]

  for s in need_load:
    flag_mem = 0
    for seg_flag, uc_flag in flag_mp:
      if seg_flag & s.p_flags:
        flag_mem = flag_mem | uc_flag

    flag_mem |= uc.UC_PROT_EXEC
    addr = s.p_vaddr
    sz = s.p_memsz
    align = s.p_align
    align = max(align, 4096)

    if s.p_paddr != 0:
      #base_addr = addr - s.p_offset
      base_addr = addr
      base_addr = base_addr - base_addr % align
    else:
      addr =base_addr = s.p_vaddr
    #base_addr = base_addr - addr % align
    seg_sz = sz + addr % align
    seg_sz = (sz + align - 1) & ~(align - 1)
    seg_sz = (seg_sz +4095) & ~4095

    print('LOADING ', flag_mem, hex(base_addr), hex(addr), seg_sz, hex(s.p_offset))
    kern.mem_map(base_addr, seg_sz, flag_mem)

    content = elf.get_seg_content(s)
    kern.mu.mem_write(addr, content)

  kern.post_load()

  regs = kern.regs
  for note in elf.notes:
    if note.n_type != 'NT_PRSTATUS':
      continue

    if 'status' in note:
      for r in note.status.pr_reg._fields:
        vx = r
        if not vx in regs:
          glog.info('could not load reg %s', r)
          continue
        v = note.status.pr_reg[r].get()
        regs[r] = v
      if kern.arch.typ == Arch.x86_64:
        fsbase = note.status.pr_reg.fs_base.get()
        gsbase = note.status.pr_reg.gs_base.get()
        kern.set_fs_base(fsbase)
        kern.set_gs_base(gsbase)
        assert kern.get_gs_base() == gsbase
        assert kern.get_fs_base() == fsbase

    print(note.status.pr_reg)

  return elf


class UCRegReader(Regs):

  def __init__(self, arch, mu):
    super().__init__(arch, mu=mu)

  def reg_code(self, name):
    return self.arch.uc_regs[name]

  def _has(self, name):
    return name in self.arch.uc_regs

  def get_register(self, name):
    return self.mu.reg_read(self.reg_code(name))

  def set_register(self, name, val):
    return self.mu.reg_write(self.reg_code(name), val)


def round_down(v, p):
  return (v // p) * p


def load_pe(kern, lib, base=0):
  prefix = 'IMAGE_SCN_'
  section_flags = pefile.retrieve_flags(pefile.SECTION_CHARACTERISTICS, prefix)
  mp = dict(MEM_READ=uc.UC_PROT_READ, MEM_WRITE=uc.UC_PROT_WRITE, MEM_EXECUTE=uc.UC_PROT_EXEC)

  for section in lib.sections:

    flags = []
    for flag in sorted(section_flags):
      if getattr(section, flag[0]):
        flags.append(flag[0])
    ucflags = 0
    for flag in flags:
      x = flag[len(prefix):]
      if x in mp:
        ucflags |= mp[x]

    rva = section.VirtualAddress + base
    sz = max(section.Misc_VirtualSize, section.SizeOfRawData)
    sz = cmisc.align(sz, 4096)

    print(hex(rva), sz, ucflags, flags, hex(rva + sz))
    kern.mem_map(rva, sz, ucflags)
    kern.mu.mem_write(rva, section.get_data())
  kern.post_load()
  return lib


def load_qemu_state(kern, qemu_state):
  memfile, regsfile = qemu_state

  memc = open(memfile, 'rb').read()
  kern.mem_map(0, len(memc), uc.UC_PROT_ALL)
  kern.mem.write(0, memc)

  mpstr = 'FPR/FP EFL/EFLAGS XMM0/XMM'
  rules = list([x.split('/') for x in mpstr.split()])
  kern.regs.fs = 0

  data = []
  for line in open(regsfile, 'r').readlines():
    for token in re.finditer('([A-Z0-9]+)\s*=([^A-Z[\]]+)', line):
      name = token.group(1).strip()
      v = [int(x, 16) for x in token.group(2).strip().split()]

      for src, dst in rules:
        name = re.sub(src, dst, name)
      data.append((name, v))

  for name, v in data:
    if len(v) == 1:
      if name in kern.regs:
        kern.regs[name] = v[0]
      else:
        print('skipping ', name)

  for name, v in data:
    if name in cmisc.to_list('FS GS DS ES SS CS'):
      kern.regs[name] = v[0]


def load_bochs_state(kern, bochs_state):
  memfile, regsfile = bochs_state

  memc = open(memfile, 'rb').read()
  kern.mem_map(0, len(memc), uc.UC_PROT_ALL)
  kern.mem.write(0, memc)

  mpstr = 'FPR/FP EFL/EFLAGS XMM0/XMM'
  rules = list([x.split('/') for x in mpstr.split()])
  kern.regs.fs = 0
  # in order:
  # regs
  # fp
  # dreg
  # creg
  # sreg

  data = []
  for line in open(regsfile, 'r').readlines():
    for token in re.finditer('^(\w+):0x([0-9a-f_]+)', line):
      data.append((token.group(1), token.group(2)))
    for token in re.finditer('(\w+)\s*:\s+([0-9a-f_]+)', line):
      data.append((token.group(1), token.group(2).replace('_', '')))
    for token in re.finditer('(eflags) 0x([0-9a-f_]+)', line):
      data.append((token.group(1), token.group(2)))
    for token in re.finditer('^(\w+)=0x([0-9a-f_]+)', line):
      data.append((token.group(1), token.group(2)))

  kern.set_real_mode()
  blacklist = cmisc.to_list('ldtr gdtr idtr tr')
  for name, v in data:
    v = int(v, 16)
    print(name, v)

    name = name.strip()
    if name in blacklist: continue
    name = kern.regs.normalize(name)
    name = kern.arch.reg_rel.get_base(name)

    if name in kern.regs:
      kern.regs[name] = v


class Kernel:

  @staticmethod
  def LoadFrom(
      pe=None,
      elf=None,
      stack_params=None,
      heap_params=None,
      arch=None,
      hook_imports=False,
      orig_elf=None,
      qemu_state=None,
      bochs_state=None,
      base=0,
      **kwargs
  ):

    arch = norm_arch(arch)
    lib = None
    if elf:
      lib = ElfUtils(elf)
      if arch is None:
        arch = lib.arch
    elif qemu_state:
      assert arch is not None
      lib = qemu_state
    elif bochs_state:
      assert arch is not None
      lib = bochs_state
    elif pe:
      lib = pefile.PE(pe)
      assert arch is not None
    else:
      lib = None

    mu = uc.Uc(arch.uc_arch, arch.uc_mode)
    kern = Kernel(mu, arch, **kwargs)
    if elf:
      assert base == 0
      lib = load_elf(kern, lib)
      if not lib.core: kern.regs[arch.reg_pc] = lib.get_entry_address()
    elif qemu_state:
      lib = load_qemu_state(kern, lib)
    elif bochs_state:
      lib = load_bochs_state(kern, lib)
    elif pe:
      lib = load_pe(kern, lib, base=base)
    else:
      kern.post_load()

    kern.lib = lib

    if stack_params:
      print('MAPPING stack ', stack_params)
      if stack_params[1] < 0:
        stack_params = list(stack_params)
        stack_params[1] = -stack_params[1]
        stack_params[0] -= stack_params[1]
      self.mem_map(stack_params[0], stack_params[1], uc.UC_PROT_READ | uc.UC_PROT_WRITE)
      self.stack_seg = tuple(stack_params)

    if heap_params:
      print('MAPPING heap ', heap_params)
      self.heap_seg = tuple(self.heap_params)
      self.mem_map(heap_params[0], heap_params[1], uc.UC_PROT_READ | uc.UC_PROT_WRITE)

    if hook_imports:
      celf = elf
      if orig_elf: celf = ElfUtils(orig_elf)
      kern.hook_imports(celf.plt)
    kern.post_init()
    return kern, lib

  def mem_map(self, start, sz, flags):
    print('MAPPING ', hex(start), hex(sz), flags)
    self.mu.mem_map(start, sz, flags)
    self.maps.append((start, sz, flags))

  def alloc_seg(self, n, prot):
    self.maps.sort()
    nmaps = list(self.maps)
    nmaps.append((256**self.arch.reg_size, 1, None))
    space = 0x1000
    pos = space
    target = None
    for start, sz, _ in nmaps:
      if pos + n + space <= start:
        target = pos + n
        break
      pos = start + sz + space
    assert target, self.maps
    self.mem_map(target, n, prot)
    return (target, n)

  def ret_hook(self, *args):
    if isinstance(self.forward_ret_hook, Iterable):
      next(self.forward_ret_hook)
    elif self.forward_ret_hook:
      self.forward_ret_hook(*args)

  def post_init(self):
    self.stack = Stack(self.regs, self.mem, self.arch, self)

  def post_load(self):
    self.set_scratch()

    self.forward_ret_hook = None
    self.ret_hook_addr = self.scratch[0]  # maybe not use this address
    self.hook_addr(self.ret_hook_addr, self.ret_hook)

    self.mc = MachineCaller(
        self.kern_runner_interface(),
        self.arch,
        self.regs,
        self.mem,
        ret_hook_addr=self.ret_hook_addr
    )
    self.amc = AsyncMachineCaller(
        self.kern_runner_interface(),
        self.arch,
        self.regs,
        self.mem,
        ret_hook_addr=self.ret_hook_addr
    )

    self.bufcl = mem_simple_buf_gen(self.mem)
    if not self.heap_seg:
      heap_size = 0x1000000
      self.heap_seg = self.alloc_seg(heap_size, uc.UC_PROT_READ | uc.UC_PROT_WRITE)
    self.heap = DummyAllocator(*self.heap_seg, bufcl=self.bufcl)

    if not self.stack_seg:
      stack_size = 0x100000
      self.stack_seg = self.alloc_seg(stack_size, uc.UC_PROT_READ | uc.UC_PROT_WRITE)
      self.regs.stack_pointer = self.stack_seg[0] + self.stack_seg[1]

    self.fcaller = AsyncFunctionCaller(self.heap)
    self.fcaller.fcgen.caller=self.amc


  def set_scratch(self):
    scratch_size = 0x1000
    self.scratch = self.alloc_seg(scratch_size, uc.UC_PROT_ALL)
    print('scratch', self.scratch)

  def __init__(self, mu, arch, sigret_count_lim=100, read_handler=None):
    self.sigret_count_lim = sigret_count_lim
    self.log_file = None
    self.mem = Memory(self.byte_reader, self.write_mem, arch=arch)
    self.mu = mu
    self.arch = arch
    self.machine = arch.mc
    self.regs = UCRegReader(self.arch, self.mu)
    self.maps = []
    self.scratch = None
    from chdrft.emu.syscall import SyscallDb, g_sys_db
    try:
      self.syscalls = g_sys_db.data[arch.typ]
      self.syscall_handlers = {}
    except:
      pass
    #self.syscall_handlers = {'mmap': self.mmap_handler,
    #                         #'__rt_sigreturn': self.sigreturn_handler,
    #                         'write': self.write_handler}
    #if read_handler:
    #  self.syscall_handlers['read']=read_handler

    #self.structure_reader = StructureReader(self.byte_reader)
    self.sig_handlers = {}
    self.kill_in = -1
    self.sigret_count = 0
    self.save_buf = None
    self.info = []
    self.ret_map = {}
    self.want_stop = 0
    self.ignore_mem_access = False
    self.prev_ins = None
    self.heap_seg = None
    self.heap = None
    self.stack_seg = None
    self.real_mode = False
    self.ins_count = 0
    self.hook_intr_func = None
    self.notify_hook_code = None

    watched = []
    sz = 4
    #watched.append(WatchedMem('stack3', self.mem, 0x7ffff7dc7000, n=0x200 // sz, sz=sz))
    self.watchers = watched

    self.default_import_hook = self.default_fail_import_hook
    self.defined_hooks = {}
    self.plt_addr_to_func = {}

    #for k, v in self.syscall_handlers.items():
    #  assert k in self.syscalls.entries, k

    self.tracer = Tracer(self.arch, self.regs, self.mem, watched)
    self.change_eflags = None

  def kern_runner_interface(self):

    class KernelRunner:

      def __init__(self, kern):
        self.kern = kern

      def __call__(self, end_pc):
        pass

    return KernelRunner(None)

  def default_fail_import_hook(self, func):
    print(hex(self.regs.ins_point))
    assert 0, func

  def func_hook_import(self, mu, address, size, _2):
    func = self.plt_addr_to_func[address]

    if func in self.defined_hooks: res = self.defined_hooks[func](func)
    else: res = self.default_import_hook(func)
    self.mc.ret_func(res)

  def hook_addr(self, addr, func):
    self.mu.hook_add(uc.UC_HOOK_CODE, safe_hook(func), None, addr, addr + 1)

  def hook_imports(self, plt):
    for func, addr in plt.items():
      self.mock_func(addr, func)

  def mock_func(self, addr, func):
      self.plt_addr_to_func[addr] = func
      self.hook_addr(addr, func)

  def dump_log(self):
    return
    self.info.append(str(self.ret_map))
    res = '\n'.join(self.info)
    open(self.log_file, 'w').write(res)

  def set_real_mode(self):
    self.real_mode = True
    self.regs.ds = 0

  def start(self, end=-1, count=0, ip=None):
    self.want_stop = 0
    if ip is None: ip = self.regs.ins_pointer
    glog.info('Starting emulator at pc=%x', self.regs.ins_pointer)
    self.mu.emu_start(ip, end, count=count)

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

  def hook_intr(self, intno, addr):
    if self.hook_intr_func:
      self.hook_intr_func(intno, addr)

  def context_str(self):
    ip = self.regs.ins_pointer
    mem = self.mem.read(ip, 10)
    data = self.arch.mc.ins_str(self.arch.mc.get_one_ins(mem, ip))
    s = f'INS {data}\n'
    s += Display.regs_summary(self.regs, self.regs)

    return s

  def hook_code(self, mu, address, size, _2):
    self.ins_count += 1
    if self.notify_hook_code:
      self.notify_hook_code(address)

    if self.change_eflags is not None:
      self.regs.eflags = self.change_eflags
      self.change_eflags = None

    glog.info('hook code %s %s %d', hex(address), hex(size), self.ins_count)

    if self.want_stop:
      self.stop()
    mem = self.byte_reader(address, size)
    self.ignore_mem_access = False

    if mem[0] == 0xcd:
      self.has_int = True
      self.hook_intr(mem[1], address+size)

    if self.prev_ins == mem and mem == b'\x00\x00':
      self.ignore_mem_access = True
      return
    self.prev_ins = mem
    self.tracer.notify_ins(address, size)
    glog.info('\n\n')
    mem = bytes(mem)
    glog.info('CODE at %s %s %d', hex(address), binascii.hexlify(mem), self.ins_count)
    if self.kill_in == 0:
      self.stop()
    self.kill_in -= 1

    data = self.arch.mc.get_ins(mem, address)
    ins0 = data[0]
    for ins in data:
      glog.info("0x%x:\t%s\t%s %s" % (ins.address, ins.mnemonic, ins.op_str, ins.bytes))

  def hook_unmapped(self, mu, access, address, size, value, _2, *args, **kwargs):
    print('BAD access at rip=', hex(self.regs.ins_pointer), hex(address), access, size, value)
    return False

  def hook_bad_mem_access(self, mu, access, address, size, value, user_data):
    print('bad mem access ', access, address, size, value)

  def hook_mem_access(self, mu, access, address, size, value, user_data):
    if self.ignore_mem_access:
      return
    cur = self.regs.ins_pointer
    s = ''
    if access == uc.UC_MEM_WRITE:
      self.tracer.notify_write(address, size, value)
      s = 'hook_access_write: ip={:x} addr={:x}, size={}, val={:x}'.format(
          cur, address, size, value
      )
    else:
      self.tracer.notify_read(address, size)
      s = 'hook_access_read: ip={:x} addr={:x}, size={}, content={:x}'.format(
          cur, address, size, self.mem.read_u(address, size)
      )
    glog.info(s)
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

  def stack_lin(self, addr):
    return addr + 16 * self.regs.ss

  def get_syscalls_args(self):
    syscall_num = self.regs.rax
    syscall = self.syscalls.by_num[syscall_num]
    args = []
    if 'args' in syscall:
      for i in range(len(syscall.args)):
        args.append(self.regs[self.arch.syscall_conv[i]])
    else:
      args = None

    #res = self.structure_reader.parse(syscall, args)
    res = None
    return Attributize(func=syscall, args=res, raw_args=args)

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
      s.append(
          Display.disp(
              watcher.snapshot(), name=watcher.diff_params.name, params=watcher.diff_params
          )
      )
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

  def set_msr(self, msr, value):
    '''
    set the given model-specific register (MSR) to the given value.
    this will clobber some memory at the given scratch address, as it emits some code.
    '''
    # save clobbered registers
    with self.regs.context('rax rdx rcx rip') as ctx:

      buf = b'\x0f\x30'
      self.regs.rax = value & 0xffffffff
      self.regs.rdx = value >> 32
      self.regs.rcx = msr
      self.execute_code_on_scratch(buf)

  def get_msr(self, msr):
    '''
    fetch the contents of the given model-specific register (MSR).
    this will clobber some memory at the given scratch address, as it emits some code.
    '''
    with self.regs.context('rax rdx rcx rip') as ctx:

      buf = b'\x0f\x32'
      self.regs.rcx = msr
      self.execute_code_on_scratch(buf)
      return self.regs.edx << 32 | self.regs.eax

  def set_fs_base(self, val):
    self.set_msr(self.arch.FSMSR, val)

  def get_fs_base(self):
    return self.get_msr(self.arch.FSMSR)

  def set_gs_base(self, val):
    self.set_msr(self.arch.GSMSR, val)

  def get_gs_base(self):
    return self.get_msr(self.arch.GSMSR)

  def execute_code_on_scratch(self, buf):
    scratch_addr = self.scratch[0] + 0x100
    self.mu.mem_write(scratch_addr, buf)
    self.mu.emu_start(scratch_addr, scratch_addr + len(buf))
