#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.types import *
import queue
from concurrent.futures import Future
import threading
import traceback as tb

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)

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
    px = Z.Process(cmd, shell=1)
    app.global_context.enter_context(px)
    pts = find_qemu_pts(px)
    print('GOOOT PTS >> ', pts)
  return get_qemu_interface(p=Z.Serial(pts))


def get_ssh():
  return get_shelllike_interface(
      'sshpass -p sstic ssh root@0.0.0.0 -p 5555', try_count=10, retry_wait_s=1
  )


def get_gdb():
  elf_aarch64 = cmisc.path_here('./resources/decrypted_file')
  return get_gdb_interface(f'aarch64-linux-gnu-gdb -q {elf_aarch64}')


class EventLoop:

  def __init__(self):
    self.q = queue.Queue()
    self.res = {}
    self.id = 0
    self.cur_future = None

  def get_id(self):
    with threading.Lock() as lock:
      r = self.id
      self.id += 1
    return r

  def create_action(self, func):
    fx = Future()
    x = cmisc.Attr(func=func, future=fx, id=self.get_id())
    self.res[x.id] = x
    self.q.put(x.id)
    return x

  def finish(self):
    self.q.put(None)

  def do_sync(self, func):
    x = self.create_action(func)
    return x.future.result()

  def do_async(self, func):
    x = self.create_action(func)
    return x.future

  def run(self):
    while True:
      e = self.q.get()
      if e is None: break
      x = self.res[e]
      r = self.execute_action(x)
      x.future.set_result(r)

  def yield_run(self):
    while True:
      e = self.q.get()
      if e is None: break
      x = self.res[e]
      r = yield from self.yield_execute_action(x)
      x.future.set_result(r)

  def yield_execute_action(self, x):
    try:
      yield from x.func()
    except Exception as e:
      glog.error(f'Failed executing {e}')
      tb.print_exc()
      raise


  def execute_action(self, x):
    try:
      return x.func()
    except Exception as e:
      glog.error(f'Failed executing {e}')
      tb.print_exc()
      raise


class SyncWrapper:

  def __init__(self, obj, el):
    self.obj = obj
    self.el = el

  def __getattr__(self, name):

    def f(*args, **kwargs):
      f = getattr(self.obj, name)
      func = lambda: f(*args, **kwargs)
      x = self.el.create_action(func)
      return x.future.result()

    return f


class SyncGetterWrapper:

  def __init__(self, obj, el):
    self.obj = obj
    self.el = el

  def __getattr__(self, name):

    def f():
      return getattr(self.obj, name)

    x = self.el.create_action(f)
    return x.future.result()


class ASyncWrapper:

  def __init__(self, obj, el):
    self.obj = obj
    self.el = el

  def __getattr__(self, name):

    def f(*args, **kwargs):
      f = getattr(self.obj, name)
      func = lambda: f(*args, **kwargs)
      x = self.el.create_action(func)
      return x.future

    return f



def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
