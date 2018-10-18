#!/usr/bin/python

import traceback as tb
import idaapi
import idc
import idautils
import time
import struct
import re
import binascii
import json
import base64

start = 0x401c48

def main():
  def wait_susp():
    while True:
      res = idc.GetDebuggerEvent(idc.WFNE_SUSP, -1)
      if res == idc.WFNE_NOWAIT:
        break


  def go(addr):
    while True:
      idc.GetDebuggerEvent(idc.WFNE_CONT | idc.WFNE_SUSP, -1)
      if idc.GetRegValue('eip') == addr:
        break


  def get_str(addr):
    s = ''
    n = 0x100
    while True:
      x = idc.DbgRead(addr, n)
      for j in range(len(x)):
        if ord(x[j]) == 0:
          s += x[:j]
          return s
      s += x
      addr += len(x)
    return s


  def step():
    go(f1_addr)
    res = get_str(idautils.cpu.ebp)
    f = open('/tmp/res.out', 'a+')
    f.write(res)
    f.close()
    return res

  def push(data):
    idautils.cpu.esp-=4
    write_u32(idautils.cpu.esp, data)


  def setup_buf_bpt(ptr, n, enable):
    for i in range(n):
      u = ptr + i
      if enable:
        idc.AddBptEx(u, 1, idc.BPT_RDWR)
      else:
        idc.DelBpt(u)

    return read_data

  def write_u32(addr, v):
    idc.DbgWrite(addr, struct.pack('<I', v))
  def write_u16(addr, v):
    idc.DbgWrite(addr, struct.pack('<H', v))

  def read_u32(addr):
    return struct.unpack('<I', idc.DbgRead(addr, 4))[0]


  def req_web(host, path):
    x = 'http://'+host+path
    import subprocess as sp
    return sp.check_output(['curl', x])




  def disable_trace():
    idc.EnableTracing(idc.TRACE_INSN, 0)


  def start_trace():
    idc.ClearTraceFile('')
    idc.EnableTracing(idc.TRACE_INSN, 1)
    idc.SetStepTraceOptions(idc.ST_OVER_LIB_FUNC)


  def ida_continue():
    idc.GetDebuggerEvent(idc.WFNE_CONT, 0)


  def do_ret():
    retaddr=idc.DbgDword(idautils.cpu.esp)
    idautils.cpu.esp+=4
    idautils.cpu.eip=retaddr

  class Hooker(idaapi.DBG_Hooks):

    def __init__(self, data):
      super(Hooker, self).__init__()
      self.done = False
      self.exited = False
      self.data = data


    def prepare(self):
      self.rw_seg_desc = idc.LocByName('rw_seg_desc')
      self.rx_seg_desc = idc.LocByName('rx_seg_desc')
      self.rw_seg_count = idc.LocByName('rw_seg_count')
      self.rx_seg_count = idc.LocByName('rx_seg_count')

      self.start_code_ea = idc.LocByName('STARTHANDLER')
      self.ret_pad_ea = idc.LocByName('RETPAD')
      self.decode_seg_ea = idc.LocByName('decode_seg')
      self.virtual_protect_ea = read_u32(idc.LocByName('VirtualProtect'))

      idc.AddBpt(self.start_code_ea)
      idc.AddBpt(self.ret_pad_ea)
      self.hx = self.handler()
      self.segs=  []

    def call(self, addr, *args):
      for arg in reversed(args):
        push(arg)
      push(self.ret_pad_ea)
      idautils.cpu.eip = addr

    def decode_seg(self, seg_addr, mem_seg):
      sz = read_u32(seg_addr + 0)
      addr = read_u32(seg_addr + 0x18)
      print('HAS SEG >> ', seg_addr, hex(addr), hex(sz), mem_seg)
      inspos_ptr = read_u32(seg_addr + 12)
      pos = 0
      inslst = []
      content = bytearray()
      self.call(self.virtual_protect_ea, addr, sz, 4, idautils.cpu.esp)
      yield

      while pos < sz:
        ipos = read_u32(inspos_ptr)
        if pos > ipos: break
        assert pos == ipos, '%s %s'%(pos, ipos)
        csize = read_u32(inspos_ptr + 4)
        inslst.append((pos, csize))
        pos += csize
        inspos_ptr += 8
      assert pos <= sz

      for ins, sz in inslst:
        target_addr = addr + ins
        push(mem_seg)
        push(target_addr)
        push(seg_addr-8)
        self.call(self.decode_seg_ea, seg_addr-8, target_addr, mem_seg)
        yield
        assert idautils.cpu.eip == self.ret_pad_ea
        content[ins:ins+sz] = idc.DbgRead(target_addr, sz)
      self.segs.append(dict(sz=sz, addr=addr, content=base64.b64encode(content), mem_seg=mem_seg, seg_addr=seg_addr))




    def handler(self):
      if 1:
        b1 = 'sn00gle-fl00gle-p00dlekins'
        dst = idautils.cpu.esp + 0x100
        dstsize = 0x102
        idc.DbgWrite(dst, '\x00'*dstsize)

        buf = dst - len(b1)-10
        idc.DbgWrite(buf, b1+'\x00')

        gen_perm_ea = idc.LocByName('generate_perm')
        mix_ea = idc.LocByName('mix_things')
        pass1_ea = idc.LocByName('PASSWORD1')
        pass1_len = read_u32(idc.LocByName('PASSWORD1_LEN'))

        finalsize = pass1_len
        finaldest = idautils.cpu.esp - 2*finalsize
        idc.DbgWrite(finaldest, '\x00'*(finalsize+1))

        self.call(gen_perm_ea, dst, buf, len(b1))
        yield


        print(hex(dst), hex(pass1_ea), hex(finaldest), hex(pass1_len))
        #self.done = 1
        #return
        self.call(mix_ea, dst, pass1_ea, finaldest, pass1_len)
        yield
        with open('./final.data', 'wb') as f:
          f.write(idc.DbgRead(finaldest, pass1_len))

      else:
        stride = 0x24

        nseg = read_u32(self.rx_seg_count)
        base_addr = self.rx_seg_desc
        print('HANDLER', nseg)
        for i in range(nseg):
          seg_addr = base_addr + stride * i
          for data in self.decode_seg(seg_addr, False):
            yield

        nseg = read_u32(self.rw_seg_count)
        base_addr = self.rw_seg_desc
        for i in range(nseg):
          seg_addr = base_addr + stride * i
          for data in self.decode_seg(seg_addr, True):
            yield

        print('dumping handler')
        json.dump(self.segs, open('./dump.json', 'w'))
      self.done = 1


    def dbg_bpt(self, tid, ea):
      try:
        if self.done:
          print('should be done')
          return 0
        if ea == self.start_code_ea:
          next(self.hx)

        elif ea == self.ret_pad_ea:
          next(self.hx)
        else:
          print('where the fuck')
          self.done = 1
          assert 0

      except Exception as e:
        tb.print_exc()
        self.done = 1
        print('FAILURE')

      return 0

    def dbg_process_exit(self, pid, tid, ea, code):
      print('exited >> ', pid, tid, ea, code)
      self.exited = True

    def off(self):
      idc.StopDebugger()
      print('gogo exit')

      while not self.exited:
        idc.GetDebuggerEvent(idc.WFNE_ANY, 1)

      self.unhook()
      idc.DelBpt(self.start_code_ea)
      idc.DelBpt(self.ret_pad_ea)

    def run(self):
      while not self.done:
        ida_continue()
    def get_time(self, t):
      return t/3600, t/60%60, t%60


  def stop_debugger():
    idc.StopDebugger()
    print('gogo exit')

    while not self.exited:
      idc.GetDebuggerEvent(idc.WFNE_ANY, 1)


  def setup():
    args = r''
    exe = r'C:\Users\benoit\work\leet\leet_editr.exe'
    path = r'C:\Users\benoit\work\leet'
    idc.StopDebugger()
    idc.AddBpt(idc.LocByName('main'))

    res = idc.StartDebugger(exe, args, path)
    print('starting dbugger')
    time.sleep(1)
    wait_susp()


  data = dict()

  h = Hooker(data)
  print('HOOKER SETUP')
  setup()
  print('RUNNIG NOW')
  wait_susp()
  h.prepare()
  try:
    try:
      assert h.hook()
      h.run()
    except:
      tb.print_exc()
      pass
    h.off()
  except Exception as e:
    tb.print_exc()
  finally:
    print('UNHOOK')
    h.unhook()
  print('finished running')

# import sys; sys.path.append(r'C:\Users\benoit\work'); from leet import test; test=reload(test); test.main()
# vim: tabstop=2 shiftwidth=2 expandtab softtabstop=2
