import re
import sys
import subprocess as sp
import struct
import mmap
import array
import binascii
import tempfile
from collections import OrderedDict, defaultdict
from asq.initiators import query as asq_query
import io
import os
import glog
import chdrft.emu.binary_consts as binary_consts
from chdrft.utils.fmt import Format

from chdrft.utils.misc import Attributize, lowercase_norm, Arch, DictUtil
import chdrft.utils.misc as cmisc
from chdrft.emu.base import RegExtractor
import chdrft.emu.base as ebase
import traceback
import shutil
from chdrft.gen.types import types_helper_by_m32
from chdrft.utils.cmdify import ActionHandler
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.path import FileFormatHelper

uc =  None
try:
  import capstone as cs
  from capstone.x86_const import *
  import capstone.x86_const as cs_x86_const
  import capstone.arm_const as cs_arm_const
  import capstone.arm64_const as cs_arm64_const
  import unicorn.x86_const as x86_const
  import unicorn.arm_const as arm_const
  import unicorn.arm64_const as arm64_const

  import unicorn.mips_const as mips_const
  import capstone.mips_const as cs_mips_const
  import unicorn as uc
except Exception as e:
  exc_type, exc_value, exc_traceback = sys.exc_info()
  print('Got exception', e)
  traceback.print_tb(exc_traceback)
  pass


class UCX86Regs(RegExtractor):

  def __init__(self, *args):
    super().__init__(x86_const, 'UC_X86_REG_')


class UCARM64Regs(RegExtractor):

  def __init__(self, *args):
    super().__init__(arm64_const, 'UC_ARM64_REG_')


class UCARMRegs(RegExtractor):

  def __init__(self, *args):
    super().__init__(arm_const, 'UC_ARM_REG_')


class CSX86Regs(RegExtractor):

  def __init__(self, *args):
    super().__init__(cs_x86_const, 'X86_REG_')


class CSARM64Regs(RegExtractor):

  def __init__(self, *args):
    super().__init__(cs_arm64_const, 'ARM64_REG_')


class CSARMRegs(RegExtractor):

  def __init__(self, *args):
    super().__init__(cs_arm_const, 'ARM_REG_')


class UCMipsRegs(RegExtractor):

  def __init__(self):
    super().__init__(mips_const, 'UC_MIPS_REG_')


class CSMipsRegs(RegExtractor):

  def __init__(self):
    super().__init__(cs_mips_const, 'MIPS_REG_')

import capstone as cs
x86_64_regs = cmisc.to_list('rax rbx rcx rdx rsi rdi rsp rbp rip r8 r9 r10 r11 r12 r13 r14 r15')
x86_64_regs = x86_64_regs + list(['xmm{}'.format(i) for i in range(16)])
x86_regs = ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp']
x86_16_regs = cmisc.to_list('ax bx cx dx si di sp bp ip ss cs es fs gs ds')

mips_regs = cmisc.to_list(
    'zero at v0 v1 a0 a1 a2 a3 t0 t1 t2 t3 t4 t6 t7 s0 s1 s2 s3 s4 s5 s6 s7 t8 t9 k0 k1 gp sp fp ra pc hi lo pc'
)
x86_redirects = dict()
#x86_redirects = dict(fs_base='fs', gs_base='gs')
arm_regs = list(['r%d' % i for i in range(13)] + cmisc.to_list('sp lr pc cpsr'))
arm64_regs = list(['x%d' % i for i in range(31)] + cmisc.to_list('sp pc lr cpacr_el1 tpidr_el0'))
#arm64_regs += list(['q%d' % i for i in range(31)])

class RegRelationX86:

  def __init__(self):
    self.par = Attributize(default_none=True)
    for i in 'abcde':
      self.par[i + 'l'] = [i + 'x', 1]
      self.par[i + 'h'] = [i + 'x', 1]
      self.par[i + 'x'] = ['e' + i + 'x', 2]
      self.par['e' + i + 'x'] = ['r' + i + 'x', 4]

    for suffix in ('ip', 'sp', 'bp', 'si', 'di'):
      self.par[suffix] = ['e' + suffix, 2]
      self.par['e' +suffix] = ['r' + suffix, 4]


    for i in range(8, 16):
      base = 'r' + str(i)
      self.par[base + 'd'] = [base, 4]
    for i in range(0, 16):
      self.par['xmm{}'.format(i)] = [None, 16]
    self.base = Attributize(default_none=True)
    for x0 in x86_16_regs:
      x = x0
      while x is not None:
        self.base[x] = x0
        x = self.get_par(x)

  def get_base(self, x):
    if x in self.base:
      return self.base[x]
    return x

  def get_par(self, reg):
    if reg not in self.par:
      return None
    res = self.par[reg]
    if res is not None:
      return res[0]
    return None

  def get_size(self, reg):
    assert None not in self.par
    if reg in self.par:
      return self.par[reg][1]
    return 8

  def find(self, reg, d):
    orig_reg = reg
    while reg is not None:
      if reg in d:
        return d[reg]
      reg = self.get_par(reg)
    return None


def norm_ins(code):
  if isinstance(code, str):
    code = [code]

  tsf = []
  for i in code:
    tsf.extend(i.split(';'))

  x = '\n'.join(map(str.strip, tsf))
  x += '\n'
  return x



class CallData:
  def __init__(self, arch, x):
    if not isinstance(x, tuple): x= tuple((x,))
    self.arch = arch
    self.cur= x[0]
    self.cnds = x
  def set_active(self, typ):
    for v in self.cnds:
      if v.typ == typ:
        self.cur = v
        break
    else:
      assert 0, typ

  def __getattr__(self, v):
    res =  getattr(self.cur, v)
    print("QUERY ", v, res)
    return res
  def norm_func_addr(self, addr):
    if self.arch.typ == Arch.thumb:
      return addr | 1
    return addr


def guess_arch(s):
  s = s.lower()
  if s.find('x86-64') != -1 or s.find('x86_64') != -1: return arch_data[Arch.x86_64]
  if s.find('x64') != -1: return arch_data[Arch.x86_64]
  if s.find('i386') != -1: return arch_data[Arch.x86]
  if s.find('aarch64') != -1: return arch_data[Arch.arm64]
  for v in Arch:
    if s == v.name: return arch_data[v]
  for v in Arch:
    if s.find(v.name) != -1: return arch_data[v]
  return None

def norm_arch(arch):
  if isinstance(arch, str): arch = guess_arch(arch)
  if isinstance(arch, Arch): arch = arch_data[arch]
  return arch


class Machine:

  def __init__(self, arch):
    self.regs = {}
    self.reg_mapping = {}
    self.cs_arch = arch_data[arch].cs_arch
    self.cs_mode = arch_data[arch].cs_mode
    self.arch = arch_data[arch]

    self.init_capstone()

  def get_reg_ops(self, ins):
    return None

  def set_reg(self, reg, val):
    self.regs[self.reg_mapping[reg]] = val

  def init_capstone(self):
    md = None
    md = cs.Cs(self.cs_arch, self.cs_mode)

    self.md = md
    if md is None:
      return
    self.md.detail = True

  def get_ins(self, data, addr=0):
    data = bytes(data)
    res = list(self.md.disasm(data, addr))
    return res

  def get_one_ins(self, data, addr=0):
    tmp = self.get_ins(data, addr)
    assert len(tmp) > 0, 'Bad turd for %s %x' % (data, addr)
    return tmp[0]

  def disp_ins(self, data, addr=0):
    for x in self.get_ins(data, addr):
      self.print_insn(x)

  def print_insn(self, ins):
    print("0x%x: %s %s" % (ins.address, ins.mnemonic, ins.op_str), bytes(ins.bytes))

  def ins_str(self, ins, addr=0):
    if isinstance(ins, (bytes, bytearray)):
      ins = self.get_ins(ins, addr=addr)
    if isinstance(ins, list):
      return ' ; '.join([self.ins_str(x) for x in ins])
    return "0x%x: %s %s %s" % (
        ins.address, ins.mnemonic, ins.op_str, binascii.hexlify(ins.bytes[::-1])
    )

  def get_disassembly(self, code, addr=0, **kwargs):
    assert 0

class ArmMachine(Machine):

  def __init__(self):
    super().__init__(Arch.arm)

  def get_disassembly(self, code, addr=0, **kwargs):
    compiler = ArmCompiler(thumb=0, **kwargs)
    return compiler.get_assembly(code)

class ThumbMachine(Machine):

  def __init__(self):
    super().__init__(Arch.thumb)

  def get_disassembly(self, code, addr=0, **kwargs):
    compiler = ArmCompiler(thumb=1, **kwargs)
    return compiler.get_assembly(code)

class Arm64Machine(Machine):

  def __init__(self):
    super().__init__(Arch.arm64)

  def get_reg_ops(self, ins):
    lst = []
    for op in ins.operands:
      if op.type == cs.arm64_const.ARM64_OP_REG:
        lst.append(self.arch.cs_regs.get_name(op.reg))
      elif op.type == cs.arm64_const.ARM64_OP_MEM:
        if op.value.mem.base != 0:
          lst.append(ins.reg_name(op.value.mem.base))
        if op.value.mem.index != 0:
          lst.append(ins.reg_name(op.value.mem.index))

    return lst

class MipsMachine(Machine):

  def __init__(self):
    super().__init__(Arch.mips)


class X86Machine(Machine):

  def __init__(self, x64=False, bit16=False):
    arch = Arch.x86
    if bit16: arch=Arch.x86_16
    if x64: arch = Arch.x86_64
    super().__init__(arch)
    self.x64 = x64
    self.bit16 = bit16
    self.regs = {}
    self.reg_mapping = {}
    self.reg_mapping['rip'] = cs.x86.X86_REG_RIP
    self.nop = b'\x90'

  def get_jmp_indirect(self, ins):
    if ins.id != cs.x86.X86_INS_JMP:
      return None
    if ins.bytes[0:2] == b'\xff\x25':
      v = struct.unpack('<I', ins.bytes[2:])[0] + ins.size
      return self.regs[cs.x86.X86_REG_RIP] + v
    return None

  def get_call(self, ins, addr=0):
    if ins.id != cs.x86.X86_INS_CALL:
      return None
    if len(ins.operands) != 1:
      return None

    op0 = ins.operands[0]
    if op0.type != cs.x86.X86_OP_IMM:
      return None
    return op0.imm

  def get_disassembly(self, code, addr=0):
    f = tempfile.NamedTemporaryFile()
    f2 = tempfile.NamedTemporaryFile()

    x = norm_ins(code)

    if not self.x64:
      x = f'BITS 32\nORG {addr}\n' + x
    else:
      x = f'BITS 64\nORG {addr}\n' + x
    f.write(x.encode())
    f.flush()

    sp.check_call('nasm %s -o %s' % (f.name, f2.name), shell=True)

    ops = None
    with open(f2.name, 'rb') as g:
      ops = g.read(100)
    return ops

  def get_reg_ops(self, ins):
    lst = []
    for op in ins.operands:
      if op.type == cs.x86_const.X86_OP_REG:
        lst.append(self.arch.cs_regs.get_name(op.reg))
    return lst


if not uc:
  arch_data = None
else:
  x86_16_arch = Attributize(
      regs=x86_16_regs,
      ins_size=-1,
      reg_size=2,
      reg_pc='ip',
      reg_stack='sp',
      cs_arch=cs.CS_ARCH_X86,
      cs_mode=cs.CS_MODE_16,
      uc_arch=uc.UC_ARCH_X86,
      uc_mode=uc.UC_MODE_16,
      cs_regs=CSX86Regs(),
      uc_regs=UCX86Regs(),
      reg_rel=RegRelationX86(),
      call_data=Attributize(reg_return='ax', reg_call=(), has_link=False),
      redirect=x86_redirects,
    flags_desc=(ebase.eflags, 'eflags'),
  )

  x86_arch = Attributize(
      regs=x86_regs,
      ins_size=-1,
      reg_size=4,
      reg_pc='eip',
      reg_stack='esp',
      cs_arch=cs.CS_ARCH_X86,
      cs_mode=cs.CS_MODE_32,
      uc_arch=uc.UC_ARCH_X86,
      uc_mode=uc.UC_MODE_32,
      cs_regs=CSX86Regs(),
      uc_regs=UCX86Regs(),
      reg_rel=RegRelationX86(),
      call_data=Attributize(reg_return='eax', reg_call=(), has_link=False),
      redirect=x86_redirects,
  )

  x86_64_arch = Attributize(
      regs=x86_64_regs,
      ins_size=-1,
      reg_size=8,
      reg_pc='rip',
      reg_stack='rsp',
      cs_arch=cs.CS_ARCH_X86,
      cs_mode=cs.CS_MODE_64,
      uc_arch=uc.UC_ARCH_X86,
      uc_mode=uc.UC_MODE_64,
      cs_regs=CSX86Regs(),
      uc_regs=UCX86Regs(),
      reg_rel=RegRelationX86(),
      FSMSR = 0xC0000100,
      GSMSR = 0xC0000101,
      call_data=(
        Attributize(typ='linux',reg_return='rax', reg_call=cmisc.to_list('rdi rsi rdx'), has_link=False),
        Attributize(typ='win', reg_return='rax', reg_call=cmisc.to_list('rcx rdx r8 r9'), has_link=False),
        )
        ,
      syscall_conv=cmisc.to_list('rdi rsi rdx rcx r8 r9'),
      redirect=x86_redirects,
  )

  mips_arch = Attributize(
      regs=mips_regs,
      ins_size=4,
      reg_size=4,
      reg_pc='pc',
      reg_link='ra',
      reg_stack='sp',
      cs_arch=cs.CS_ARCH_MIPS,
      cs_mode=cs.CS_MODE_MIPS32 | cs.CS_MODE_LITTLE_ENDIAN,
      call_data=Attributize(reg_return='v0', reg_call=cmisc.to_list('a0 a1 a2 a3')),
      uc_arch=uc.UC_ARCH_MIPS,
      uc_mode=uc.UC_MODE_MIPS32 | uc.UC_MODE_LITTLE_ENDIAN,
      cs_regs=CSMipsRegs(),
      uc_regs=UCMipsRegs(),
  )

  arm_arch = Attributize(
      regs=arm_regs,
      ins_size=4,
      reg_size=4,
      reg_pc='pc',
      reg_link='lr',
      reg_stack='sp',
      cs_arch=cs.CS_ARCH_ARM,
      cs_mode=0,
      call_data=Attributize(reg_return='r0', reg_call=cmisc.to_list('r0 r1 r2 r3')),
      uc_arch=uc.UC_ARCH_ARM,
      uc_mode=0,
      cs_regs=CSARMRegs(),
      uc_regs=UCARMRegs(),
  )


  arm64_arch = Attributize(
      regs=arm64_regs,
      ins_size=4,
      reg_size=8,
      reg_pc='pc',
      reg_link='lr',
      reg_stack='sp',
      cs_arch=cs.CS_ARCH_ARM64,
      cs_mode=0,
      uc_arch=uc.UC_ARCH_ARM64,
      uc_mode=0,
      cs_regs=CSARM64Regs(),
      uc_regs=UCARM64Regs(),
      do_normalize=0,
  )

  thumb_arch = Attributize(
      regs=arm_regs,
      ins_size=2,
      reg_size=4,
      reg_pc='pc',
      reg_link='lr',
      reg_stack='sp',
      cs_arch=cs.CS_ARCH_ARM,
      cs_mode=cs.CS_MODE_THUMB,
      uc_arch=uc.UC_ARCH_ARM,
      uc_mode=uc.UC_MODE_THUMB,
      cs_regs=CSARMRegs(),
      uc_regs=UCARMRegs(),
      call_data=Attributize(reg_return='r0', reg_call=cmisc.to_list('r0 r1 r2 r3')),
  )

  arch_data = {
      Arch.x86_64: x86_64_arch,
      Arch.x86: x86_arch,
      Arch.x86_16: x86_16_arch,
      Arch.mips: mips_arch,
      Arch.arm: arm_arch,
      Arch.arm64: arm64_arch,
      Arch.thumb: thumb_arch,
  }

  for k, v in arch_data.items():
    v.typ = k
    if v.reg_size == 4: v.typs_helper = types_helper_by_m32[True]
    elif v.reg_size == 8: v.typs_helper = types_helper_by_m32[False]

    if 'call_data' in v: v.call_data=CallData(v, v.call_data)

    v.typs = None

  x64_mc = X86Machine(True)
  x86_mc = X86Machine(False)
  x86_16_mc = X86Machine(bit16=True)
  arm_mc = ArmMachine()
  thumb_mc = ThumbMachine()
  mips_mc = MipsMachine()
  arm64_mc = Arm64Machine()

  arch_data[Arch.x86_16].mc = x86_16_mc
  arch_data[Arch.x86].mc = x86_mc
  arch_data[Arch.x86_64].mc = x64_mc
  arch_data[Arch.mips].mc = mips_mc
  arch_data[Arch.arm].mc = arm_mc
  arch_data[Arch.arm64].mc = arm64_mc
  arch_data[Arch.thumb].mc = thumb_mc

import sys


def to_hex(s):
  return " ".join("0x{0:02x}".format(c) for c in s)  # <-- Python 3 is OK


def to_hex2(s):
  r = "".join("{0:02x}".format(c) for c in s)  # <-- Python 3 is OK
  while r[0] == '0':
    r = r[1:]
  return r


def to_x(s):
  from struct import pack
  if not s:
    return '0'
  x = struct.pack(">q", s)
  while x[0] in ('\0', 0):
    x = x[1:]
  return to_hex2(x)


def to_x_32(s):
  from struct import pack
  if not s:
    return '0'
  x = struct.pack(">i", s)
  while x[0] in ('\0', 0):
    x = x[1:]
  return to_hex2(x)


def cs_print_x86_insn(insn):
  # Capstone Python bindings, by Nguyen Anh Quynnh <aquynh@gmail.com>
  def print_string_hex(comment, str):
    print(comment, end=' '),
    for c in str:
      print("0x%02x " % c, end=''),
    print()

  # print address, mnemonic and operands
  print("0x%x:\t%s\t%s" % (insn.address, insn.mnemonic, insn.op_str))

  # "data" instruction generated by SKIPDATA option has no detail
  if insn.id == 0:
    return

  # print instruction prefix
  print_string_hex("\tPrefix:", insn.prefix)

  # print instruction's opcode
  print_string_hex("\tOpcode:", insn.opcode)

  # print operand's REX prefix (non-zero value is relavant for x86_64
  # instructions)
  print("\trex: 0x%x" % (insn.rex))
  print('\tid: {}'.format(insn.id))

  # print operand's address size
  print("\taddr_size: %u" % (insn.addr_size))

  # print modRM byte
  print("\tmodrm: 0x%x" % (insn.modrm))

  # print displacement value
  print("\tdisp: 0x%s" % to_x_32(insn.disp))

  # SSE CC type
  if insn.sse_cc != X86_SSE_CC_INVALID:
    print("\tsse_cc: %u" % (insn.sse_cc))

  # AVX CC type
  if insn.avx_cc != X86_AVX_CC_INVALID:
    print("\tavx_cc: %u" % (insn.avx_cc))

  # AVX Suppress All Exception
  if insn.avx_sae:
    print("\tavx_sae: TRUE")

  # AVX Rounding Mode type
  if insn.avx_rm != X86_AVX_RM_INVALID:
    print("\tavx_rm: %u" % (insn.avx_rm))

  count = insn.op_count(X86_OP_IMM)
  if count > 0:
    print("\timm_count: %u" % count)
    for i in range(count):
      op = insn.op_find(X86_OP_IMM, i + 1)
      print("\t\timms[%u]: 0x%s" % (i + 1, to_x(op.imm)))

  if len(insn.operands) > 0:
    print("\top_count: %u" % len(insn.operands))
    c = -1
    for i in insn.operands:
      c += 1
      if i.type == X86_OP_REG:
        print("\t\toperands[%u].type: REG = %s" % (c, insn.reg_name(i.reg)))
      if i.type == X86_OP_IMM:
        print("\t\toperands[%u].type: IMM = 0x%s" % (c, to_x(i.imm)))
      if i.type == X86_OP_FP:
        print("\t\toperands[%u].type: FP = %f" % (c, i.fp))
      if i.type == X86_OP_MEM:
        print("\t\toperands[%u].type: MEM" % c)
        if i.mem.segment != 0:
          print("\t\t\toperands[%u].mem.segment: REG = %s" % (c, insn.reg_name(i.mem.segment)))
        if i.mem.base != 0:
          print("\t\t\toperands[%u].mem.base: REG = %s" % (c, insn.reg_name(i.mem.base)))
        if i.mem.index != 0:
          print("\t\t\toperands[%u].mem.index: REG = %s" % (c, insn.reg_name(i.mem.index)))
        if i.mem.scale != 1:
          print("\t\t\toperands[%u].mem.scale: %u" % (c, i.mem.scale))
        if i.mem.disp != 0:
          print("\t\t\toperands[%u].mem.disp: 0x%s" % (c, to_x(i.mem.disp)))

      # AVX broadcast type
      if i.avx_bcast != X86_AVX_BCAST_INVALID:
        print("\t\toperands[%u].avx_bcast: %u" % (c, i.avx_bcast))

      # AVX zero opmask {z}
      if i.avx_zero_opmask:
        print("\t\toperands[%u].avx_zero_opmask: TRUE" % (c))

      print("\t\toperands[%u].size: %u" % (c, i.size))


class ArmCompiler:
  def __init__(self, compiler=None, arch='', cpu='cortex-m0', thumb=1):
    for cnd in ('arm-none-eabi-gcc', 'arm-linux-androideabi-gcc'):
      if compiler is not None: break
      compiler = shutil.which(cnd)
    assert compiler is not None

    self.compiler = compiler
    self.thumb = thumb
    opts = []
    if thumb: opts.append('-mthumb')
    if arch: opts.append(f'-march={arch}')
    if cpu: opts.append(f'-mcpu={cpu}')
    self.opts = ' '.join(opts)

  def get_assembly(self, code):
    fil = tempfile.mkstemp(suffix='.S')[1]
    ofil = tempfile.mkstemp()[1]
    F1 = 'MAINKAPPA'
    F2 = 'MAINKAPPA_END'

    code = norm_ins(code)
    s = f'''
{'.THUMB' if self.thumb else ''}
.global {F1}
{F1}:
{code}

.global {F2}
{F2}:
'''

    open(fil, 'w').write(s)
    print('fil', fil, 'ofile', ofil)
    sp.check_call(
        f'{self.compiler} -c {self.opts} -o {ofil} {fil}'.format(**locals()),
        shell=True
    )
    from chdrft.emu.elf import ElfUtils
    x = ElfUtils(ofil)
    f1 = x.get_symbol(F1)
    f2 = x.get_symbol(F2)
    data = x.get_range(f1, f2)
    return data


class FilePatcher:

  def __init__(self, fname, mc, ofile=None, dry=False):
    from chdrft.emu.elf import ElfUtils
    self.elf_file = ElfUtils(fname)
    self.fname = fname
    self.mc = mc
    self.patches = []
    self.dry = dry
    self.ofile = ofile

  def apply(self):
    tgt_file = self.fname
    if self.ofile is not None:
      shutil.copy2(self.fname, self.ofile)
      tgt_file = self.ofile


    with open(tgt_file, 'r+b') as f:
      mm = mmap.mmap(f.fileno(), 0)
      for patch in self.patches:
        pos = patch.pos
        content = patch.content
        if self.dry:
          before = mm[pos:pos + len(content)]
          before_ins = self.mc.ins_str(self.mc.get_ins(before, patch.addr))
          next_ins = self.mc.ins_str(self.mc.get_ins(content, patch.addr))
          print('fuu ', next_ins, content)
          print('Applying patch ', patch.to_dict(), next_ins, 'replaceing >> ', before_ins)
        else:
          mm[pos:pos + len(content)] = content
      mm.close()

  def add_patch(self, pos, content, addr=0):
    assert pos is not None
    self.patches.append(Attributize(pos=pos, content=content, addr=addr))

  def patch(self, addr, content):
    self.add_patch(self.elf_file.get_pos(addr), content, addr=addr)

  def patch_one_ins(self, addr, content):
    one_ins = self.elf_file.get_one_ins(addr)

    if isinstance(content, str):
      content = self.mc.get_disassembly(content, addr=addr)

    assert len(content) <= len(one_ins.bytes
                              ), 'Cannot replace %s with %s(%s)' % (one_ins.bytes, one_ins.bytes, content)
    self.patch(addr, Format(content).pad(len(one_ins.bytes), self.mc.nop[0]).v)

  def nop_ins(self, addr):
    return self.patch_one_ins(addr, b'')


def patch_file(fname, pos, patch):
  from chdrft.emu.elf import EF
  off = None
  vaddr = None
  sz = None

  with open(fname, 'rb') as f:
    elf = EF.ELFFile(f)
    for s in elf.iter_sections():
      if s.name == '.text':
        vaddr = s['sh_addr']
        off = s['sh_offset']
        sz = s['sh_size']

  assert off is not None
  with open(fname, 'r+b') as f:
    mm = mmap.mmap(f.fileno(), 0)
    pos = pos - vaddr + off
    mm[pos:pos + len(patch)] = patch
    mm.close()


class DebugRegion:

  @staticmethod
  def CreateIns():
    attr_list = 'addr label mnemonic line opcode'.split(' ')
    res = Attributize(elem={k: None for k in attr_list})
    res['tostr'] = lambda: DebugRegion.InsStr(res)
    res['label'] = ''
    return res

  @staticmethod
  def InsStr(ins):
    res = dict(ins._elem)
    res['opcode'] = binascii.hexlify(res['opcode']).decode()
    return '(l {line:03d}) {addr:08x} ({label}) -> {mnemonic} \t({opcode})'.format(**res)

  def __init__(self):
    self.data = dict()
    self.labels = dict()

  def finalize(self):
    for k, v in self.data.items():
      if v.label:
        self.labels[v.label] = v
    self.ins = asq_query(self.data.values())

  def query(self, offset=0):
    return DebugRegionQuery(self, offset)


class DebugRegionQuery:

  def __init__(self, region, offset=0, q=None):
    self.region = region
    self.offset = 0
    if q is None:
      q = asq_query(self.region.data.values())
    self.q = q

  def set_offset(self, offset):
    self.offset = offset
    return self

  def minv(self, label):
    cur = self.region.labels[label]
    self.q = self.q.where(lambda x: x.addr >= cur.addr)
    return self

  def maxv(self, label):
    cur = self.region.labels[label]
    self.q = self.q.where(lambda x: x.addr < cur.addr)
    return self

  def clone(self):
    tmp = self.q.to_list()
    self.q = asq_query(tmp)
    res = DebugRegionQuery(self.region, asq_query(tmp))
    return res

  def bin(self):
    res = self.clone().q.order_by(lambda x: x.addr).select(lambda x: x.opcode or b'').to_list()
    res = b''.join(res)
    return res

  def str(self, addr):
    addr -= self.offset
    return DebugRegion.InsStr(self.region.data[addr])
    #return DebugRegion.InsStr(self.clone().q.where(lambda x: x.addr == addr).first())


class DebugFile:

  def __init__(self):
    self.region = DebugRegion()
    self.start = None
    self.end = None

  def setup(self, fileobj=None, filename=None, content=None):
    if filename is not None:
      fileobj = open(filename, 'r')
    elif content is not None:
      fileobj = io.StringIO(content)

    for linenum, line in enumerate(fileobj.readlines()):
      self.add_line(line, linenum)
    self.region.finalize()
    return self

  def get(self, addr):
    return DictUtil.get_or_insert(self.region.data, addr, DebugRegion.CreateIns())

  def add_line(self, line, linenum):
    assert 0


class REDebugFile(DebugFile):

  def __init__(self):
    super().__init__()
    self.re_list = []
    self.ctx_list = []
    self.ctx = Attributize()
    self.hooks = defaultdict(lambda: [])

  def setup(self, **kwargs):
    super().setup(**kwargs)
    return self

  def add_ctx_action(self, reg, action):
    self.ctx_list.append((re.compile(reg), action))

  def add_re(self, v, cond=None):
    if isinstance(v, list):
      pattern = []
      for e in v:
        k, v = e.split('=', maxsplit=1)
        pattern.append('(?P<%s>%s)' % (k, v))
      pattern = '\s+'.join(pattern)
      v = '\s*' + pattern + '\s*'

    if isinstance(v, str):
      #glog.info('Adding re %s', v)
      v = re.compile(v)
    self.re_list.append((v, cond))

  def add_hook(self, fieldname, hook):
    self.hooks[fieldname].append(hook)

  def add_line(self, line, linenum):
    for e, action in self.ctx_list:
      if re.match(e, line):
        action(self.ctx)

    for e, cond in self.re_list:
      if cond is not None and not cond(self.ctx):
        continue

      res = re.match(e, line)
      if not res: continue
      res = Attributize(res.groupdict())
      for k, v in res.items():
        for hook in self.hooks[k]:
          v = hook(v)
        res[k] = v

      ins = self.get(res.addr)
      ins.line = linenum
      for k, v in res.items():
        if k in ins:
          ins[k] = v
      break


class PRUDebugFile(REDebugFile):

  def __init__(self):
    super().__init__()
    self.add_re(
        '(?P<file>[^(]*)\([^)]*\) : (?P<addr>\S*) = (?P<opcode>0x\S+)\s+:\s+(?P<mnemonic>.*)\s*'
    )
    self.add_re('(?P<file>[^(]*)\([^)]*\) : (?P<addr>\S*) = Label\s+:\s+(?P<label>.*):\s*')
    #example: pru_cc1110.p(  225) : 0x0000 = Label      : main:

    self.add_hook('opcode', lambda x: binascii.unhexlify(x))
    self.add_hook('addr', lambda x: Format(x).toint().v)


class PRULstDebugFile(REDebugFile):

  #00000014                   main:
  #00000014     051ee2e2      SUB R2, R2, 30
  def __init__(self):
    super().__init__()
    addr_fmt = '(?P<addr>[0-9a-f]+)'
    opcode_fmt = '(?P<opcode>[0-9a-f]+)'
    self.ctx.text = False
    text_cond = lambda x: x.text
    self.add_re(
        '{addr_fmt}\s+{opcode_fmt}\s+(?P<mnemonic>.*)\s*'.format(
            addr_fmt=addr_fmt, opcode_fmt=opcode_fmt
        ),
        cond=text_cond
    )
    self.add_re('{addr_fmt}\s+(?P<label>.*):\s*'.format(addr_fmt=addr_fmt), cond=text_cond)
    #example: pru_cc1110.p(  225) : 0x0000 = Label      : main:

    self.add_ctx_action('TEXT Section', lambda x: x.update(text=True))
    self.add_ctx_action('DATA Section', lambda x: x.update(text=False))
    self.add_hook('opcode', lambda x: binascii.unhexlify(x))
    self.add_hook('addr', lambda x: Format(x).toint(16).v)


class MCS51DebugFile(REDebugFile):

  #000000                          1 main:
  #000000 74 12            [12]    2         mov A, #0x12
  #000002 04               [12]    3 inc A

  def __init__(self):
    super().__init__()
    re_list = []
    addr = 'addr=\S*'
    fileline = 'fileline=\d+'

    self.add_re([addr, 'opcode=\S+( \S+)*', 'cyclecount=\[\d+\]', fileline, 'mnemonic=.*'])
    self.add_re([addr, fileline, 'label=\S+:'])
    self.add_hook('opcode', lambda x: binascii.unhexlify(x.replace(' ', '')))
    self.add_hook('addr', lambda x: Format(x).toint(base=16).v)
    self.add_hook('label', lambda x: x[:-1])


class MCS51Compiler:

  def __init__(self):
    self.binary = 'sdas8051'

  def compile_file(self, filename, cwd):
    res_prefix = 'res'
    res_file = os.path.join(cwd, res_prefix + '.out')
    res_dbg_file = os.path.join(cwd, res_prefix + '.lst')
    sp.check_call([self.binary, '-l', '-s', '-o', res_file, filename], cwd=cwd)
    return res_dbg_file

  def assemble(self, code):
    code = norm_ins(code)

    with tempfile.TemporaryDirectory(prefix='tmp_mcs51_compile_') as temp_dir:
      glog.info('Assembling in directory %s', temp_dir)

      F1 = 'MAINKAPPA'
      F2 = 'MAINKAPPA_END'

      s = '''{prefix}

  {F1}:
  {code}
  {F2}:
  '''.format(
          prefix=binary_consts.cc1110_prefix, **locals()
      )

      asm_file = os.path.join(temp_dir, 'raw.s')
      open(asm_file, 'w').write(s)

      res_dbg_file = self.compile_file(asm_file, cwd=temp_dir)
      debug_file = MCS51DebugFile().setup(filename=res_dbg_file)
    return debug_file.region.query().minv(F1).maxv(F2)

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  parser.add_argument('--arch', type=guess_arch)
  parser.add_argument('--data', type=FileFormatHelper.Read)
  parser.add_argument('--addr', type=cmisc.to_int, default=0)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  mc = ThumbMachine()
  res = mc.get_disassembly('''
  adr %r2, START
  mov %r1, #12;
  mov %r1, #12;
  mov %r1, #12;
  mov %r1, #12;
  mov %r1, #12;
  .align
START:
  ''')
  import codecs
  mc.disp_ins(res, addr=ctx.addr)
  print(codecs.encode(res, 'hex'))



def str_to_bytes(s):
  try:
    return bytes(exec(s))
  except:
    pass

  res = []
  for line  in s.split('\n'):
    dx  = line.find(':')
    if dx!=-1: line = line[dx+1:]
    res.extend( map(cmisc.to_int, line.split()))
  return bytes(res)

def disassemble(ctx):
  ctx.arch.mc.disp_ins(str_to_bytes(ctx.data), addr=ctx.addr)

def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
