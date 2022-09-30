#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, cwdpath
import glog
from chdrft.emu.structures import StructBuilder, SimpleStructExtractor, Structure, StructBackend, BufAccessor, BufAccessorBase, g_data, CodeStructExtractor
from chdrft.emu.base import Stack

from contextlib import ExitStack
import ctypes
import chdrft.utils.misc as cmisc

class FuncCallWrapper:

  def __init__(self, typ=None, addr=None, caller=None):
    self.typ = typ
    self.addr = addr
    self.caller = caller
    assert caller is not None

  def __call__(self, *args):
    return self.caller(self.addr, *args)


class FuncCallWrapperGen:

  def __init__(self, elffile=None, lib=None, code_db=None, caller=None, name_to_addr={}):
    self.lib = lib
    self.elffile = elffile
    self.code_db = code_db
    self.caller = caller
    self.name_to_addr = name_to_addr

  def find_address(self, name):
    if self.lib is not None:
      a = getattr(self.lib, name)
      a = ctypes.cast(a, ctypes.c_void_p)
      return a.value
    elif name in self.name_to_addr:
      return self.name_to_addr[name]
    else:
      assert self.elffile is not None, name
      return self.elffile.get_symbol(name)

  def find_typ(self, name):
    return self.code_db.functions[name]

  def get(self, name):
    return FuncCallWrapper(
        typ=self.find_typ(name), addr=self.find_address(name), caller=self.caller)


class MachineCallerBase:
  def __init__(self, runner, arch, regs, mem, ret_hook_addr=None):
    self.runner = runner
    self.arch = arch
    self.regs = regs
    self.mem = mem
    self.stack =Stack(self.regs, self.mem, self.arch)
    self.ret_hook_addr = ret_hook_addr


  def get_arg(self, i):
    nregcall=len(self.arch.call_data.reg_call)
    if nregcall >= i:
      return self.regs[self.arch.call_data.reg_call[i]]
    return self.stack.get(i - nregcall)

  def set_arg(self, i, v):
    nregcall=len(self.arch.call_data.reg_call)
    if nregcall >= i:
      self.regs[self.arch.call_data.reg_call[i]] = v
    else: self.stack.set(i - nregcall, v)


  @property
  def args(self):
    return cmisc.CustomClass(setitem=self.set_arg, getitem=self.get_arg)

  def prepare(self, func, *args, ret_to_pad=True):
    if ret_to_pad: ret_pc = self.ret_hook_addr
    else: ret_pc = self.regs.ins_pointer
    ret_pc = self.arch.call_data.norm_func_addr(ret_pc)
    self.regs.ins_pointer = self.arch.call_data.norm_func_addr(func)

    rem = []
    for i, arg in enumerate(args):
      if len(self.arch.call_data.reg_call) > i:
        self.regs[self.arch.call_data.reg_call[i]] = arg
      else: rem.append(arg)
    for e in rem[::-1]:
      self.stack.push(e)

    if self.arch.call_data.get('has_link', True): self.regs[self.arch.reg_link] = ret_pc
    else: self.stack.push(ret_pc)

    self.runner(ret_pc)

  def ret_func(self, retv=None):
    if retv is not None:
      self.regs[self.arch.call_data.reg_return] = retv
    if self.arch.call_data.has_link: self.regs.ins_pointer = self.regs[self.arch.reg_link]
    else:
      self.regs.ins_pointer = self.stack.pop()


  def get_ret(self):
    return self.regs[self.arch.call_data.reg_return]

class MachineCaller(MachineCallerBase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __call__(self, func, *args, **kwargs):
    self.prepare(func, *args, **kwargs)
    return self.get_ret()

class AsyncMachineCaller(MachineCallerBase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __call__(self, func, *args, **kwargs):
    self.prepare(func, *args)
    yield
    res = self.get_ret()
    yield res


class CTypesCaller:
  def __init__(self, gdata):
    self.gdata = gdata
    pass
  def __call__(self, func, *args):
    base_typ = self.gdata.types_helper.by_name.ptr.ctype

    func_obj = ctypes.CFUNCTYPE(base_typ,*( (base_typ,) *len(args)))
    f = func_obj(func)
    res =  f(*args)
    return res

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

def mem_simple_buf_gen(mem):
  return SimpleBufGen(read=mem.read, write=mem.write)


class CTypesBuf(BufAccessorBase):

  def __init__(self, addr=None, sz=None, **kwargs):
    super().__init__(**kwargs)
    buf = None
    if sz is not None:
      typ = ctypes.c_char * sz
      buf = typ.from_address(addr)
    self.buf = buf
    self.addr = addr

  def _read(self, pos, n):
    if self.buf is not None:
      return self.buf[pos:pos + n]
    else:
      typ = ctypes.c_char * n
      buf = typ.from_address(pos)
      return buf[:]

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
      self.cur_allocator = cur_allocator
      assert cur_allocator is not None
      data = []
      for farg, arg in zip(func.typ.args, args):
        s = self.get_structure(farg.typ)
        s.smart_set(arg)
        data.append(s.get_for_call())
      result = func(*data)

      restype =self.get_structure(func.typ.restype)
      restype.set_child_backend(StructBackend(buf=CTypesBuf()))
      restype.smart_set(result)
      return restype

  def get_structure(self, typ):
    s = Structure(typ, default_backend=False)
    s.set_backend(StructBackend(buf=BufAccessor(s.bytesize, allocator=self.cur_allocator)))
    return s


  def __call__(self, *args, **kwargs):
    return self.call_func(*args, **kwargs)

class AsyncFunctionCaller:

  def __init__(self, allocator, fcgen=None):
    self.allocator = allocator
    if fcgen is None: fcgen = FuncCallWrapperGen()
    self.fcgen = fcgen
    self.coroutine = None

  def reset_fcgen(self, **kwargs):
    self.fcgen = FuncCallWrapperGen(**kwargs)

  def __getattr__(self, name):
    func = self.fcgen.get(name)
    assert func is not None, name
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
      yield tmp_res
      yield next(cx) #now ready to set result

  def __call__(self, *args, **kwargs):
    return self.call_func(*args, **kwargs)



def create_ctypes_caller(ctypes_lib, code):
  allocator =CTypesAllocator()
  caller = CTypesCaller(g_data)
  fcgen = FuncCallWrapperGen(code_db=code, lib=ctypes_lib, caller=caller)
  fc = FunctionCaller(allocator, fcgen)
  return fc

def test(ctx):
  #fona_target = swig_unsafe.elec_fona_test_target_swig
  import ctypes
  l1 = ctypes.cdll.LoadLibrary('libc.so.6')
  g_data.set_m32(False)

  test_code = '''
int getpid();
char *get_current_dir_name();
char *strdup(const char *s);
'''
  g_code = StructBuilder()
  g_code.add_extractor(CodeStructExtractor(test_code, ''))
  g_code.build(extra_args=[], want_sysincludes=False)

  fc =create_ctypes_caller(l1, g_code)
  print(fc.getpid().get())
  print(fc.get_current_dir_name().get_ptr_as_str())
  print(fc.strdup('jambonkapap').get_ptr_as_str())

def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)



def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
