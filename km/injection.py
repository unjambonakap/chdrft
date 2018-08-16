import subprocess as sp
import chdrft.tube.process as process
import pyopa
import argparse
import argcomplete
import time
from asq.initiators import query
import ctypes
import struct
import errno

from chdrft.utils.misc import path_from_script, cwdpath, opa_print, PatternMatcher, RopBuilder, opa_serialize
from chdrft.utils.elf import ElfUtils
from chdrft.utils.proc import ProcHelper
from chdrft.km.hooking import Hooking


class Injector:

    def __init__(self, ctrl, stage0_elf, stage0_sym, stage1_elf, stage1_sym):
        self.pid = None
        self.stage0_sym = stage0_sym
        self.stage1_sym = stage1_sym
        self.ctrl = ctrl
        self.hooking = Hooking(self.ctrl)

        self.stage0_elf = ElfUtils(stage0_elf)
        self.stage1_elf_name = stage1_elf
        self.target_addr_off = 0x400

        self.test_syscall = self.hooking.get_orig_num_syscalls()
        self.hooking.set_num_syscalls(self.test_syscall+1)
        self.hooking.set_syscall(self.test_syscall, Hooking.dummy_sym)
        tmp = pyopa.SyscallAccessor(self.test_syscall)
        tmp.dummy(0x12340000)
        print('done call dummy')

    def load(self, pid):
        self.pid = pid

        helper = ProcHelper(self.pid)
        regions = helper.get_memory_regions()

        ld_res = regions.get_elf_entry(PatternMatcher.fromre('/ld'))
        libdl_res = regions.get_elf_entry(PatternMatcher.fromre('/libdl'))
        libc_res = regions.get_elf_entry(PatternMatcher.fromre('/libc'))
        target_region = query(
            regions.regions).where(
            lambda u: u.typ == 'stack').order_by(
            lambda u: u.size).then_by_descending().to_list()[0]
        self.page_start = target_region.start_addr
        self.target_addr = self.page_start + self.target_addr_off

        self.ld_elf = ElfUtils(ld_res.file, offset=ld_res.start_addr)
        self.libdl_elf = ElfUtils(libdl_res.file, offset=libdl_res.start_addr)
        self.libc_elf = ElfUtils(libc_res.file, offset=libc_res.start_addr,
                                 load_sym=False)
        self.inject_section = self.stage0_elf.get_section('.inject')
        self.stage0_entry_off = self.stage0_elf.get_symbol(self.stage0_sym)

    def get_stage0_data(self):
        stage0_data = pyopa.opa_stage0_data_t()
        stage0_data.stage1_main_sym = self.stage1_sym
        stage0_data.stage1_elf = self.stage1_elf_name
        stage0_data.dl_allocate_tls_addr = self.ld_elf.get_dyn_symbol(
            '_dl_allocate_tls')
        stage0_data.dlopen_addr = self.libdl_elf.get_dyn_symbol('dlopen')
        stage0_data.dlsym_addr = self.libdl_elf.get_dyn_symbol('dlsym')
        stage0_data.dummy_sysnum = self.test_syscall
        print('dl_allocate_tls >> ', hex(stage0_data.dl_allocate_tls_addr))
        print('dlopen >> ', hex(stage0_data.dlopen_addr))
        print('dlsym >> ', hex(stage0_data.dlsym_addr))
        return opa_serialize(stage0_data)

    def get_rop(self):
        self.stage0_sym_addr = self.stage0_elf.get_symbol(
            self.stage0_sym) - self.inject_section.sh_addr
        g_sys = self.libc_elf.find_gadget('syscall; ret')
        self.g_ret = self.libc_elf.find_gadget('ret')
        g_rdi = self.libc_elf.find_gadget('pop rdi; ret')
        g_rsi = self.libc_elf.find_gadget('pop rsi; ret')
        g_rax = self.libc_elf.find_gadget('pop rax; ret')
        g_mark = self.libc_elf.find_gadget(
            'push rax; add rsp, 0x18 ; xor eax, eax ; pop rbx ; pop rbp ; ret')

        builder = RopBuilder(self.target_addr, ptr_size=8)
        self.builder = builder

        builder.add('QQQ', g_rax, self.ctrl.get_sysnum('mmap'), g_sys)
        self.ref_mmap_addr = 'ref_mmap_addr'
        builder.add('{#%s}Q' % self.ref_mmap_addr, g_mark)
        ntrash = 4  # see g_mark
        builder.add('%dQ' % ntrash, *([0]*ntrash))

        for i in range(1):
            builder.add('QQ', g_rsi, 0)
            builder.add('Q{Q:_ref_sleep_data}', g_rdi)
            builder.add('QQ', g_rax, self.ctrl.get_sysnum('nanosleep'))
            builder.add('Q', g_sys)

        builder.add('QQQ', g_rax, self.test_syscall, g_sys)#debug

        print('ret buffer storm at addr ', hex(self.g_ret))
        print('g_rdi >> ', hex(g_rdi))
        print('g_mark >>', hex(g_mark))
        builder.add('Q{Q:_ref_stage0_data}', g_rdi)

        self.ref_stage0_entry = 'target_stage0_entry'
        builder.add('{#%s}Q' % self.ref_stage0_entry, 0xdeadbeefb00bface)

        sleep_struct = pyopa.opa_timespec()
        sleep_struct.tv_sec = 8
        sleep_struct.tv_nsec = 0

        print(opa_serialize(sleep_struct))
        builder.add('{#sleep_data}{S:sleep_struct}',
                    sleep_struct=opa_serialize(sleep_struct))
        builder.add('{#stage0_data}{S:0}', self.get_stage0_data())
        print('rop at ', hex(self.target_addr))

        rop = builder.get()
        print('content:', rop)
        return rop

    def get_regs(self):
        regs = pyopa.opa_Regs()
        #no rax set because gonna be trashed on around sysret
        regs.rdi = 0x12340000 #for debugging
        regs.rsi = len(self.inject_section.data)
        regs.rdx = pyopa.PROT_READ | pyopa.PROT_EXEC | pyopa.PROT_WRITE
        regs.r10 = pyopa.MAP_ANONYMOUS | pyopa.MAP_PRIVATE
        regs.r8 = 0
        regs.r9 = 0
        regs.sp = self.target_addr
        regs.ip = self.g_ret
        regs.bp = self.target_addr+0x200
        return regs

    def wait_inject_mmap(self, check_addr, orig_content):
        print(
            'wait inject mmap, checkaddr={:x}, orig_content={}'.format(
                check_addr,
                orig_content))
        while True:
            content = ctypes.string_at(check_addr, 8)
            if content != orig_content:
                break
            time.sleep(1e-3)
        print('injected mmap done, gogogo')

        time.sleep(1e-3)
        content = ctypes.string_at(check_addr, 8)

        target_elf0_addr = struct.unpack('<Q', content)[0]
        target_elf0_addr = ctypes.c_int64(target_elf0_addr).value
        if target_elf0_addr < 0:
            print(
                'failed inject mmap, ret={}, errno={}'.format(target_elf0_addr,
                                                              errno.errorcode[-target_elf0_addr]))
            raise Exception('failed')
        print('gota addr >> addr={:x} raw={}'.format(target_elf0_addr, content))
        payload = self.inject_section.data
        mmaped_addr = self.add_mapping(target_elf0_addr, len(payload))
        ctypes.memmove(mmaped_addr, payload, len(payload))

        entry = struct.pack('<Q', target_elf0_addr+self.stage0_entry_off)
        ret_sp_addr = self.rop_mmaped_addr + self.builder.get_ref_off(
            self.ref_stage0_entry)
        ctypes.memmove(ret_sp_addr, entry, len(entry))
        print('should be done')

    def add_mapping(self, addr, n):
        res, mmaped_addr = self.ctrl.add_mapping(
            self.pid, addr, n)
        assert pyopa.opa_is_success(res)
        return mmaped_addr

    def remote_inject(self, pid):
        self.load(pid)
        rop_len = 0x200

        rop = self.get_rop()

        input('go mmap?')
        self.rop_mmaped_addr = self.add_mapping(
            self.page_start, rop_len+self.target_addr_off)+self.target_addr_off

        input('do memmove?')
        ctypes.memmove(self.rop_mmaped_addr, rop, len(rop))
        check_addr = self.rop_mmaped_addr + \
            self.builder.get_ref_off(self.ref_mmap_addr)
        orig_content = ctypes.string_at(check_addr, 8)

        input('go inject thread?')
        self.ctrl.inject_thread(self.pid, self.get_regs())

        self.wait_inject_mmap(check_addr, orig_content)

    def local_inject(self, proc):

        self.load(proc.proc.pid)
        payload = self.get_rop()
        regs = self.get_regs()

        check_addr = self.builder.get_ref(self.ref_mmap_addr)
        # orig_content = ctypes.string_at(check_addr, 8)
        serialized_regs = pyopa.opa_serialize_regs(regs).encode(
            'utf-8', errors='surrogateescape')

        print('proc pdi >> ', proc.proc.pid)
        input('go?')
        proc.send(struct.pack('<I', len(payload)))
        proc.send(payload)
        proc.send(serialized_regs)
        proc.send(struct.pack('<Q', check_addr))
        proc.send(struct.pack('<I', len(self.inject_section.data)))
        proc.send(self.inject_section.data)
        proc.send(struct.pack('<I', self.stage0_entry_off))
        proc.send(
            struct.pack(
                '<Q',
                self.builder.get_ref(
                    self.ref_stage0_entry)))

        proc.proc.stdin.flush()
        proc.proc.wait()
