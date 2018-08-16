import pyopa
import argparse
import argcomplete
import time
from asq.initiators import query
import ctypes
import struct
import errno

from chdrft.utils.misc import path_from_script, cwdpath, opa_print, PatternMatcher, RopBuilder, opa_serialize, Dict
from chdrft.utils.elf import ElfUtils
from chdrft.utils.proc import ProcHelper
from chdrft.km.ksyms import KSyms
from chdrft.utils.binary import X86Machine, cs_print_x86_insn

from chdrft.utils.cache import Cachable
import capstone as cs


class Hooking(Cachable):

    handle_msg_sym = 'opa_handle_msg'
    dummy_sym = 'opa_dummy'
    ptr_size = 8

    def __init__(self, x, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.ksysm = KSyms()
        self.asm = X86Machine(x64=True)

    @Cachable.cached
    def get_orig_num_syscalls(self):
        val = 0
        for k, v in pyopa.__dict__.items():
            if k.startswith('__NR_'):
                val = max(val, v)
        return val+1

    @Cachable.cached
    def get_syscall_code(self):
        syscall_entry_addr = self.ksysm.get_sym_addr('system_call_after_swapgs')
        buf_len = 0x100
        buf = self.x.peek2(syscall_entry_addr, buf_len)
        assert len(buf) == buf_len
        return [buf, syscall_entry_addr]

    @Cachable.cached
    def get_syscall_entry_insn(self):
        buf, addr = self.get_syscall_code()
        insn = self.asm.get_ins(buf, addr)

        cnd = []
        for x in insn:
            if x.id != cs.x86_const.X86_INS_CMP:
                continue
            if len(x.operands) != 2:
                continue

            op0, op1 = x.operands[0:2]
            if op0.type != cs.x86_const.X86_OP_REG or op0.reg != cs.x86_const.X86_REG_RAX:
                continue
            if op1.type != cs.x86_const.X86_OP_IMM:
                continue
            print('op val >> {:x}'.format(op1.imm))
            cnd.append(x)
        assert len(cnd) == 1
        cs_print_x86_insn(cnd[0])
        return Dict(
            address=cnd[0].address,
            size=cnd[0].size,
            bytes=cnd[0].bytes)

    @Cachable.cached
    def get_syscall_table_addr(self):
        buf, addr = self.get_syscall_code()
        insn = self.asm.get_ins(buf, addr)

        cnd = []
        for x in insn:
            ok = True
            ok &= x.id == cs.x86_const.X86_INS_CALL
            ok &= len(x.operands) == 1
            if not ok:
                continue
            op = x.operands[0]
            ok &= op.type == cs.x86_const.X86_OP_MEM
            ok &= op.mem.index == cs.x86_const.X86_REG_RAX
            ok &= op.mem.scale == self.ptr_size
            if not ok:
                continue
            cnd.append(op.mem.disp)
        assert len(cnd) == 1
        return cnd[0]

    def set_num_syscalls(self, num):
        ins = self.get_syscall_entry_insn()

        new = self.asm.get_disassembly('cmp rax, 0x{:x}'.format(num-1))
        #-1 because it's followed by a ja
        for x in self.asm.get_ins(new, 0):
            cs_print_x86_insn(x)

        assert len(new) == ins.size
        addr = ins.address

        old = ins.bytes
        while len(new) and new[0] == old[0]:
            new = new[1:]
            old = old[1:]
            addr += 1

        while len(new) and new[-1] == old[-1]:
            new = new[:-1]
            old = old[:-1]

        if len(new):
            print('writing {} at {:x}'.format(new, addr))
            res = self.x.poke2(addr, new)
            assert res == len(new)

    def set_syscall(self, syscall_num, sym):
        func_addr = self.ksysm.get_sym_addr(sym)
        buf = struct.pack('<Q', func_addr)
        target_addr = self.get_syscall_table_addr()
        print('syscall table addr should be at {:x}'.format(target_addr))

        target_addr += syscall_num*self.ptr_size
        res = self.x.poke2(target_addr, buf)
        assert len(buf) == res
