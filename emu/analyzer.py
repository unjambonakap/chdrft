from chdrft.emu.syscall import g_sys_db, g_syscall_conv, g_call_conv
from chdrft.emu.binary import x86_64_arch
from chdrft.emu.structures import Structure, StructBackend
from chdrft.utils.misc import Attributize


class Analyzer:

  def __init__(self, reader):
    self.reader = reader
    self.arch = x86_64_arch
    self.sys_conv = g_syscall_conv[self.arch.typ]
    self.call_conv = g_call_conv[self.arch.typ]

  def analyze_syscall(self, syscall_num, regs):
    func = g_sys_db.data[self.arch.typ].by_num[syscall_num]
    res=self.analyze_function(func.function, regs, self.sys_conv)
    res.num = syscall_num
    res.func = func
    return res

  def analyze_function(self, func, regs, conv):
    res = Attributize()
    tb = []
    for i, arg in enumerate(func.args):
      vx = regs[self.arch.cs_regs.get_name(conv[i])]


      v = Structure(arg.typ, off=0, child_backend=StructBackend(self.reader))
      v.set_scalar_backend(vx & v.mask)
      tb.append(v)
      res[arg.name] = v
    res.by_order = tb
    return res
