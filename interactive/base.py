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
from chdrft.interactive.kernel_jup import OpaKernelSpecManager, OpaKernelApp

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)

def run_notebook(*args, notebook_dir='./', background=False, current_pid=-1):

  print('Running notebook for ', current_pid)
  proc = sp.Popen(
      [
          'jupyter',
          'notebook',
          '--NotebookApp.kernel_manager_class=chdrft.interactive.kernel_jup.OpaEntryKernelManager',
          '--NotebookApp.kernel_spec_manager_class=chdrft.interactive.kernel_jup.OpaKernelSpecManager',
          '--Session.key=b""',
          '--NotebookApp.token=',
          '--no-browser',
          f'--OpaKernelManager.caller_kid={current_pid}',
          f'--notebook-dir={notebook_dir}',
      ]
  )
  if not background:
    while True:
      try:
        if proc.wait() is not None: break
      except KeyboardInterrupt:
        pass

def create_kernel(runid=None, do_run_notebook=False, **kwargs):
  if runid is None: runid = app.flags.runid
  ipkernel = OpaKernelApp.instance()
  ipkernel.runid = runid



  kwargs.update(current_pid=os.getpid())
  if do_run_notebook:
    kwargs = dict(kwargs)
    kwargs['runid'] = app.flags.runid
    threading.Thread(target=run_notebook, kwargs=kwargs).start()

  ipkernel.initialize(['python', '--matplotlib=qt'])
  ipkernel.user_module, ipkernel.user_ns = extract_module_locals(1)
  ipkernel.shell.set_completer_frame()

  time.sleep(1)
  ipkernel.start()


def test_try_kernel(ctx):
  a = 123
  u = cmisc.Attributize()
  create_kernel(runid=ctx.runid)

def test_run_notebook(ctx):
  a = 123
  u = cmisc.Attributize()
  run_notebook()



def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
