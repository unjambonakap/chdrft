#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, cwdpath
import glog
from chdrft.emu.structures import StructBuilder, SimpleStructExtractor, Structure, StructBackend, BufAccessor, BufAccessorBase

from contextlib import ExitStack
import ctypes


class FuncCallWrapper:

  def __init__(self, typ=None, addr=None, caller=None):
    self.typ = typ
    self.addr = addr
    self.caller = caller

  def __call__(self, *args):
    return self.caller(self.addr, *args)


class FuncCallWrapperGen:

  def __init__(self, elffile=None, lib=None, code_db=None, caller=None):
    self.lib = lib
    self.elffile = elffile
    self.code_db = code_db
    self.caller = caller

  def find_address(self, name):
    if self.lib is not None:
      a = getattr(self.lib, name)
      a = ctypes.cast(a, ctypes.c_void_p)
      return a.value
    else:
      return self.elffile.get_symbol(name)

  def find_typ(self, name):
    return self.code_db.functions[name]

  def get(self, name):
    return FuncCallWrapper(
        typ=self.find_typ(name), addr=self.find_address(name), caller=self.caller)


class MachineCaller:
  def __init__(self, arch, regs, mem, runner):
    self.arch = arch
    self.regs = regs
    self.mem = mem
    self.runner = runner

  def __call__(self, func, *args):
    ret_pc = self.regs[self.arch.reg_pc]
    print('CALLING ', func, args)
    self.regs[self.arch.reg_pc] = func
    self.regs[self.arch.reg_link] = ret_pc

    for i, arg in enumerate(args):
      self.regs[self.arch.call_data.reg_call[i]] = arg
    self.runner(ret_pc)
    res = self.regs[self.arch.call_data.reg_return]
    return res

class AsyncMachineCaller:
  def __init__(self, arch, regs, mem, runner):
    self.arch = arch
    self.regs = regs
    self.mem = mem
    self.runner = runner

  def __call__(self, func, *args):
    ret_pc = self.regs[self.arch.reg_pc]
    print('CALLING ', func, args)
    self.regs[self.arch.reg_pc] = func
    self.regs[self.arch.reg_link] = ret_pc

    for i, arg in enumerate(args):
      self.regs[self.arch.call_data.reg_call[i]] = arg
    self.runner(ret_pc)
    yield
    res = self.regs[self.arch.call_data.reg_return]
    print('RETURNING FINAL RESULT ', res)
    yield res


class SimpleBuf(BufAccessorBase):

  def __init__(self, addr, sz, read, write, **kwargs):
    super().__init__(**kwargs)
    self.addr = addr
    self.sz = sz
    self.read_f = read
    self.write_f = write

  def _read(self, pos, n):
    assert pos + n <= self.sz
    return self.read_f(self.addr + pos, n)

  def _write(self, pos, content):
    assert pos + len(content) <= self.sz
    return self.write_f(self.addr + pos, content)


class SimpleBufGen:

  def __init__(self, read=None, write=None):
    self.read = read
    self.write = write

  def __call__(self, addr, sz, **kwargs):
    return SimpleBuf(addr, sz, self.read, self.write, **kwargs)


class CTypesBuf(BufAccessor):

  def __init__(self, addr, sz, **kwargs):
    typ = ctypes.c_char * sz
    buf = typ.from_address(addr)
    super().__init__(sz, buf, **kwargs)
    self.addr = addr

  def _read(self, pos, n):
    return self.buf[pos:pos + n]

  def _write(self, pos, content):
    self.buf[pos:pos + len(content)] = content


class CTypesAllocator(ExitStack):

  def __init__(self):
    super().__init__()
    self.objs = []

  def __call__(self, sz):
    obj = ctypes.create_string_buffer(sz)
    self.objs.append(obj)
    addr = ctypes.addressof(obj)
    return addr, CTypesBuf(addr, sz, allocator=self)

  def __enter__(self):
    super().__enter__()
    self.callback(self.objs.clear)
    return self


class SimpleAllocator(ExitStack):

  def __init__(self, alloc=None, free=None, bufcl=CTypesBuf):
    super().__init__()
    self.alloc = alloc
    self.free = free
    self.bufcl = bufcl

  def __call__(self, sz):
    addr = self.alloc(sz)
    self.callback(self.free, addr)
    return addr, self.bufcl(addr, sz, allocator=self)

class DummyAllocator(SimpleAllocator):
  def __init__(self, addr, sz, bufcl):
    super().__init__(alloc=self.alloc_func, free=self.free_func, bufcl=bufcl)
    self.addr = addr
    self.sz = sz
    self.pos = 0

  def alloc_func(self, sz):
    ret = self.addr + self.pos
    self.pos += sz
    assert self.pos <= self.sz
    return ret

  def free_func(self, addr):
    print('KAPPA freeing ', addr)




class FunctionCaller:

  def __init__(self, allocator, fcgen):
    self.allocator = allocator
    self.fcgen = fcgen

  def __getattr__(self, name):
    func = self.fcgen.get(name)
    return lambda *args, **kwargs: self.call_func(func, *args, **kwargs)

  def call_func(self, func, *args, **kwargs):
    assert len(args) == 0 or len(kwargs) == 0
    if len(kwargs) != 0:
      args = []
      for i, arg in enumerate(func.typ.args):
        args.append(kwargs[arg.name])

    assert len(args) == len(func.typ.args)
    cur_allocator = ExitStack()
    if self.allocator is not None: cur_allocator = self.allocator

    with cur_allocator:
      assert cur_allocator is not None
      data = []
      for farg, arg in zip(func.typ.args, args):
        s = Structure(farg.typ, default_backend=False)
        s.set_backend(StructBackend(buf=BufAccessor(s.bytesize, allocator=cur_allocator)))
        s.smart_set(arg)
        data.append(s.get_for_call())
      result = func(*data)
      return result

  def __call__(self, *args, **kwargs):
    return self.call_func(*args, **kwargs)

class AsyncFunctionCaller:

  def __init__(self, allocator, fcgen):
    self.allocator = allocator
    self.fcgen = fcgen
    self.coroutine = None

  def __getattr__(self, name):
    func = self.fcgen.get(name)
    return lambda *args, **kwargs: self.do_call(func, *args, **kwargs)

  def do_call(self, func, *args, **kwargs):
    self.coroutine = self.call_func(func, *args, **kwargs)
    next(self.coroutine)
  
  def result(self):
    return next(self.coroutine)

  def call_func(self, func, *args, **kwargs):
    assert len(args) == 0 or len(kwargs) == 0
    if len(kwargs) != 0:
      args = []
      for i, arg in enumerate(func.typ.args):
        args.append(kwargs[arg.name])

    cur_allocator = ExitStack()
    if self.allocator is not None: cur_allocator = self.allocator
    assert len(args) == len(func.typ.args)
    with cur_allocator:
      assert cur_allocator is not None
      data = []
      for farg, arg in zip(func.typ.args, args):
        s = Structure(farg.typ, default_backend=False)
        s.set_backend(StructBackend(buf=BufAccessor(s.bytesize, allocator=cur_allocator)))
        s.smart_set(arg)
        data.append(s.get_for_call())
      cx = func(*data)
      tmp_res = next(cx) # enter function
      print('FUNCTION READY TO BE CALLED')
      yield tmp_res
      print('NOW SETTING RESULT OF FUNCTIOn')
      yield next(cx) #now ready to set result

  def __call__(self, *args, **kwargs):
    return self.call_func(*args, **kwargs)
