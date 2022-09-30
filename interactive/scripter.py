#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import threading
from contextlib import ExitStack
import os
import time
from chdrft.tube.serial import Serial
from chdrft.tube.process import Process

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class InterfaceScripter(ExitStack):

  def __init__(self, cmd_args=None, p=None, done_check_pattern=None, pattern_size=20, **cmd_kwargs):
    super().__init__()
    self.p = p
    self.cmd_args = cmd_args
    self.cmd_kwargs = cmd_kwargs
    self.done_check_pattern = done_check_pattern
    self.pattern_size = pattern_size
    self.pending = None

  def run(self, data):
    pattern = self.generate_pattern()
    send_done_cmd = self.done_check_pattern.format(pattern)
    self.p.send(data + send_done_cmd)
    self.ev = threading.Event()
    self.pending = pattern

  def check_waiting(self):
    return self.pending is not None

  def run_and_wait(self, data):
    self.run(data)
    return self.wait()

  def wait(self):
    glog.info(f'Waiting for pattern {self.pending}')
    res = self.p.recv_until(self.pending)
    a = res[:-len(self.pending)]
    self.pending = None
    return a

  def __call__(self, data):
    return self.run_and_wait(data)

  def generate_pattern(self):
    return os.urandom(self.pattern_size).hex()

  def __enter__(self):
    super().__enter__()
    if self.p is None:
      self.p = Process(self.cmd_args, **self.cmd_kwargs)
    self.enter_context(self.p)
    time.sleep(0.1)
    return self


class EnterCtxRetry(ExitStack):

  def __init__(self, obj, try_count=1, retry_wait_s=1):
    super().__init__()
    self.obj = obj
    self.try_count = try_count
    self.retry_wait_s = retry_wait_s

  def __enter__(self):
    super().__enter__()
    for i in range(self.try_count):
      try:
        self.obj.__enter__()
      except KeyboardInterrupt:
        raise
      except Exception:
        glog.info(f'On try {i}, exception: {tb.format_exc()}')
        continue

      self.push(self.obj)
      break
    else:
      raise Exception('max retry for EnterCtxRetry')

    return self.obj


def get_qemu_interface(cmd=None, p=None):
  interface = InterfaceScripter(
      cmd=cmd, p=p, done_check_pattern='\np /x 0x11{}\n', shell=1, pattern_size=7
  )
  app.global_context.enter_context(interface)
  return interface


def get_bash_scripter(**kwargs):
  return InterfaceScripter('/bin/bash', done_check_pattern=';echo {};\n', shell=1, **kwargs)

def get_shelllike_interface(cmd, **kwargs):
  interface = InterfaceScripter(cmd, done_check_pattern=';echo {};\n', shell=1)
  app.global_context.enter_context(EnterCtxRetry(interface, **kwargs))
  return interface


def get_gdb_interface(cmd):
  interface = InterfaceScripter(cmd, done_check_pattern='\np /x 0x11{}\n', shell=1, pattern_size=7)
  return app.global_context.enter_context(interface)


def find_qemu_pts(tube):
  m = b'char device redirected to (/dev/pts/[0-9]+)\s'
  pm = cmisc.PatternMatcher.fromre(m)
  res = tube.recv_until(pm)
  cdev = pm.result.group(1)
  return cdev


def get_qemu(pts=None):
  if pts is None:
    flash_file = cmisc.path_here('../flash.bin')
    rom_file = cmisc.path_here('../rom.bin')
    cmd = f'{qemu_bin} -nographic -machine virt,secure=on -cpu max -smp 1 -m 1024 -bios {rom_file} -semihosting-config enable,target=native -device loader,file={flash_file},addr=0x04000000 -netdev user,id=network0,hostfwd=tcp:127.0.0.1:5555-192.168.200.200:22 -net nic,netdev=network0 -serial stdio -monitor pty 2>&1'
    px = Process(cmd, shell=1)
    app.global_context.enter_context(px)
    pts = find_qemu_pts(px)
    print('GOOOT PTS >> ', pts)
  return get_qemu_interface(p=Serial(pts))


def get_ssh():
  return get_shelllike_interface(
      'sshpass -p sstic ssh root@0.0.0.0 -p 5555', try_count=10, retry_wait_s=1
  )


def get_gdb():
  elf_aarch64 = cmisc.path_here('./resources/decrypted_file')
  return get_gdb_interface(f'aarch64-linux-gnu-gdb -q {elf_aarch64}')


def get_gdb_res(x):
  import re
  res = int(re.search('= 0x([0-9a-f]+)', x.decode()).group(1), 16)
  glog.info(f'GOT res {res:x}')
  return res


def test_interface(ctx):
  if 1:
    with Serial('/dev/pts/1') as sx:
      sx.send('help\n')
      print(sx.recv(10))
      print(sx.trash())
      return
  if 1:
    ssh = get_ssh()
    print(ssh('ls'))
    return
  if 0:
    ix = get_shelllike_interface('/bin/bash')
    print(ix('ls'))

  if 1:
    qx = get_qemu(flags.qemu_pts)
    print(qx('help').decode())

  if 0:
    qx = get_gdb()
    print(qx('help').decode())



def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
