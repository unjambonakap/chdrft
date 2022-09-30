#!/usr/bin/env python

import os
import time
import threading
import subprocess as sp
from IPython.utils.frame import extract_module_locals
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
from chdrft.interactive.kernel_jup import OpaKernelSpecManager, OpaKernelApp, list_kernels
import pprint

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--console', action='store_true')
  parser.add_argument('--conn-dir')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def run_notebook(
    *args, notebook_dir='./', background=False, current_pid=-1, console=0, extra_args=[],
  conn_dir=''
):
  res = list_kernels(conn_dir)
  pprint.pprint(res)

  print('Running notebook for ', current_pid)
  os.environ['QT_API'] = 'pyqt5'
  common_opts = [
      '--NotebookApp.kernel_manager_class=chdrft.interactive.kernel_jup.OpaEntryKernelManager',
      '--NotebookApp.kernel_spec_manager_class=chdrft.interactive.kernel_jup.OpaKernelSpecManager',
      '--JupyterConsoleApp.kernel_manager_class=chdrft.interactive.kernel_jup.OpaConsoleKernelManager',
  ]
  cmd = [
      'jupyter',
      'notebook',
      '--Session.key=b""',
      '--NotebookApp.token=',
      '--no-browser',
      f'--OpaEntryKernelManager.caller_kid={current_pid}',
      f'--OpaKernelSpecManager.conn_dir={conn_dir}',
      f'--notebook-dir={notebook_dir}',
  ]
  print(cmd)

  if console:
    cmd = [
        'jupyter',
        'console',
        '--existing',
    ]

  proc = sp.Popen(cmd + common_opts + extra_args)
  if not background:
    while True:
      try:
        if proc.wait() is not None: break
      except KeyboardInterrupt:
        pass


@cmisc.logged_f()
def create_kernel(runid=None, do_run_notebook=False, in_thread=False, blender=False, conn_dir=None, **kwargs):
  if runid is None: runid = app.flags.runid
  if conn_dir is not None: cmisc.makedirs(conn_dir)

  par_locals, par_globals = cmisc.get_n2_locals_and_globals()
  par_globals = dict(par_globals)
  par_globals.update(par_locals)

  user_module, _ = extract_module_locals(1)
  user_ns = par_globals

  ipkernel = OpaKernelApp.instance(user_ns=user_ns)
  if conn_dir: ipkernel.connection_dir = conn_dir
  ipkernel.runid = runid

  kwargs.update(current_pid=os.getpid())
  if do_run_notebook:
    kwargs = dict(kwargs)
    kwargs['runid'] = app.flags.runid
    threading.Thread(target=run_notebook, kwargs=kwargs).start()

  print('LA create kernel')

  if blender:
    import chdrft.interactive.blender as blender
    blender.register()
    blender.JupyterKernelLoop.kernelApp = ipkernel
    print('jupyter is setup')
    return

  #ipkernel.initialize(['python', '--matplotlib=qt5'])
  #ipkernel.shell.user_global_ns.update(user_ns)
  #ipkernel.user_ns = user_ns
  #ipkernel.user_module = user_module

  ipkernel.initialize(['python', '--matplotlib=qt5'])

  time.sleep(1)

  def startit():
    glog.warn('Start it')
    try:
      ipkernel.start()
    except Exception as e:
      glog.warn(f'failed, got exception {e}')

  if in_thread:
    threading.Thread(target=startit).start()

  else:
    startit()


def test_try_kernel(ctx):
  a = 123
  u = cmisc.Attributize()
  create_kernel(runid=ctx.runid)


def test_run_notebook(ctx):
  a = 123
  u = cmisc.Attributize()
  run_notebook(extra_args=ctx.other_args, console=ctx.console, conn_dir=ctx.conn_dir)


def start_blender(ctx):
  sp.check_call(
      f'blender --debug-python --python-use-system-env --python-expr "from chdrft.interactive.base import create_kernel; create_kernel(runid=\'blender1\', blender=1, conn_dir=\'{ctx.conn_dir}\') "',
      shell=1
  )


def test_blender(ctx):
  sp.check_call(
      'blender --debug-python --python-use-system-env --python-expr "from chdrft.interactive.base import test_blender_internal; test_blender_internal()"',
      shell=1
  )


def test_blender_internal():
  import bpy
  import asyncio
  import sys
  from bpy.app.handlers import persistent

  @persistent
  @cmisc.logged_f(filename='/tmp/res.out')
  def loadHandler(dummy):
    glog.warn('load handler')
    # If call tmp_timer here instead, kernel doesn't work on successive files if have used kernel in current file.

  bpy.app.handlers.load_post.append(loadHandler)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
