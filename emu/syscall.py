#!/usr/bin/env python

from chdrft.utils.misc import path_here, to_list, Attributize, Arch, opa_print
from chdrft.gen.opa_clang import OpaIndex, InputOnlyFilter
from chdrft.emu.structures import StructBuilder, SimpleStructExtractor
from clang.cindex import CursorKind
from chdrft.emu.binary import X86Machine

#shared with unicorn
from capstone.x86_const import *
import re
import sys

class SyscallData():

  def __init__(self, arch):
    self.arch = arch
    self.vals = {}
    self.entries = Attributize()
    self.by_num = {}
    self.consts = Attributize()
    self.builder=StructBuilder()

    self.builder.content+= '''
#include <unistd.h>
'''

    self.builder.add_extractor(SimpleStructExtractor('asm/signal.h', 'SIG sigset'))
    self.builder.add_extractor(SimpleStructExtractor('asm/mman.h', 'MAP_ PROT_'))
    self.builder.add_extractor(SimpleStructExtractor('asm/prctl.h', 'PR_'))
    self.builder.add_extractor(SimpleStructExtractor('asm/prctl.h', 'ARCH_'))
    self.builder.add_extractor(SimpleStructExtractor('unistd.h', 'write stat read'))
    self.builder.add_extractor(SimpleStructExtractor('fcntl.h', 'open openat'))
    self.builder.setup_includes()

  def add_syscall_num(self, name, val):
    self.vals[name] = val

  def add_entry(self, data):
    assert data.syscall_name in self.vals, 'Fail for arch={}, func={}'.format(self.arch,
                                                                              data.func_name)
    data.syscall_num = self.vals[data.syscall_name]
    data = data._do_clone()
    self.entries[data.func_name] = data
    self.entries[data.syscall_name] = data
    self.by_num[data.syscall_num] = data

  def build(self):

    # why is that existing?
    def build_func_name(name):
      return name + '__OPA_DECL'

    to_proc = {}
    for func_name, entry in self.entries.items():
      if len(entry.params_list) == 0:
        entry.args = []
        continue
      nfunc_name = build_func_name(func_name)

      self.builder.content += '''
{return_val} {func_name}({func_args});
  '''.format(return_val=entry.return_val,
             func_name=nfunc_name,
             func_args=','.join(entry.params_list))
      to_proc[nfunc_name] = entry

    self.builder.build(extra_args=['-std=c++11'])

    for func_name, func in self.builder.res.functions.items():
      # updating entry with args from function
      func_name=build_func_name(func_name)
      if not func_name in to_proc:
        continue
      entry = to_proc[func_name]
      entry.args = func.args
      entry.function=func


class SyscallDb():

  def __init__(self, desc_filename=path_here('./data/syscall_desc.txt')):
    self.syscall_filename = {Arch.x86: path_here('./data/unistd_32.h'),
                             Arch.x86_64: path_here('./data/unistd_64.h')}

    self.desc_filename = desc_filename
    self.re = re.compile("""\
(?P<return_val>\S+)\s+(?P<func_name>\w+)\
(\|(?P<alias_list>\w+))?\
(:(?P<syscall_name>\w+)\
(:(?P<socketcall_id>\w+))?)?\
\((?P<params>[^)]*)\)\s+\
(?P<arch_list>\S+)\
""")

    self.all_archs = [Arch.x86, Arch.x86_64]
    self.data = Attributize()
    for arch in self.all_archs:
      self.data[arch] = SyscallData(arch)

    self.setup_syscalls_num()
    self.analyze()
    #only support x64 atm
    self.data[Arch.x86_64].build()
    self.data[Arch.x86].build()

  def setup_syscalls_num(self):
    cur_re = re.compile('#define\s+__NR_(?P<syscall_name>\w+)\s+(?P<syscall_val>\d+)')
    for arch, filename in self.syscall_filename.items():
      with open(filename, 'r') as f:
        for line in f.readlines():
          m = cur_re.match(line)
          if not m: continue
          data = Attributize(m.groupdict())
          self.data[arch].add_syscall_num(data.syscall_name, int(data.syscall_val))

  def proc_entry(self, e):
    m = re.match(self.re, e)
    if not m:
      return
    extract = to_list('return_val func_name alias_list syscall_name socketcall_id params arch_list')
    tb = m.groupdict()
    data = Attributize(tb)
    data.params_list = [] if len(data.params) == 0 else data.params.split(',')

    data.arch_list = data.arch_list.split(',')
    if data.arch_list[0] == 'all':
      data.arch_list = self.all_archs
    else:
      data.arch_list = [Arch(x) for x in data.arch_list if x in Arch.__members__]
    inter = set(data.arch_list).intersection(self.all_archs)
    data.arch_list = inter
    if not data.syscall_name:
      data.syscall_name = data.func_name

    for arch in inter:
      self.data[arch].add_entry(data)

  def analyze(self):
    with open(self.desc_filename, 'r') as f:
      for line in f.readlines():
        self.proc_entry(line)

g_sys_db=SyscallDb()
#g_sys_db=None
g_syscall_conv={}
g_call_conv={}
g_syscall_conv[Arch.x86_64] = [X86_REG_RDI, X86_REG_RSI, X86_REG_RDX,
                                  X86_REG_RCX, X86_REG_R8, X86_REG_R9]
g_call_conv[Arch.x86_64] = [X86_REG_RDI, X86_REG_RSI, X86_REG_RDX,
                                  X86_REG_R10, X86_REG_R8, X86_REG_R9]

class SyscallGen:

  def __init__(self):
    self.mx = X86Machine(True)

  def get_reg_name(self, v):
    pass

  def do_simple_syscall(self, syscall, *args):

    asm = []
    asm.append('mov rax, qword {}'.format(syscall.syscall_num))
    for i in range(len(args)):
      asm.append('mov {}, qword {}'.format(regs.get_name(g_syscall_conv[Arch.x86_64][i]), args[i]))
    asm.append('syscall')
    res = self.mx.get_disassembly(asm)
    return res

g_sys_gen = SyscallGen()



if __name__ == '__main__':
  a = SyscallDb()
  x=a.data[Arch.x86_64]
  print(x.entries.read)
  print(x.entries.write)
  print(x.entries.__open)
  print(x.entries.__openat)
