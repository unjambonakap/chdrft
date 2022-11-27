"""
Kernel manager for connecting to a IPython kernel started outside of Jupyter.
Use this kernel manager if you want to connect a Jupyter notebook to a IPython
kernel started outside of Jupyter.
"""

import glob
import os
import os.path
import re
import traceback as tb

import time
from jupyter_core.paths import jupyter_runtime_dir, jupyter_path
from ipykernel.kernelapp import IPKernelApp
from notebook.services.kernels.kernelmanager import MappingKernelManager
from jupyter_client.kernelspec import KernelSpecManager, KernelSpec, NoSuchKernel
from tornado import gen
from tornado.concurrent import Future
from jupyter_client import write_connection_file
import datetime
from collections import defaultdict
from ipykernel.ipkernel import IPythonKernel
import uuid
from ipython_genutils.py3compat import unicode_type
from jupyter_client.manager import KernelManager

from jupyter_client.ioloop import IOLoopKernelManager
import asyncio

from traitlets import default, Integer, Unicode, Instance
import json
from chdrft.utils.path import FileFormatHelper
import subprocess as sp
import shlex
import os, signal
from enum import Enum

import chdrft.utils.misc as cmisc
class KernelKind(str, Enum):
  BLENDER='blender'
  OTHER='other'


kRunId = 'runid'
kDate = 'date'
kData = 'data'
kKind = 'kind'
kPid = 'pid'
kKernelFile = 'kernel_file'
kNamePrefix = 'kernel_'


class ProcHelper:

  def __init__(self, pid):
    self.pid = pid
    base = f'/proc/{pid}'
    try:
      self.cmdline = FileFormatHelper.Read(f'{base}/cmdline').strip(b'\x00').split(b'\x00')
      self.cwd = os.path.realpath(f'{base}/cwd')
      self.active = True
    # TODO: environ
    except:
      self.active = False

  @cmisc.logged_failsafe
  def duplicate(self):
    print(self.cmdline)
    sp.Popen([x.decode() for x in self.cmdline], cwd=self.cwd, stdout=sp.PIPE, stderr=sp.STDOUT)

  @cmisc.logged_failsafe
  def kill(self):
    assert self.active
    os.kill(self.pid, signal.SIGKILL)

  @cmisc.logged_failsafe
  def interrupt(self):
    assert self.active
    os.kill(self.pid, signal.SIGINT)


class OpaEntryKernelManager(MappingKernelManager):
  caller_kid = Integer(config=True, default_value=-1)

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def _attach_to_latest_kernel(self, kernel_id):
    self.log.info(f'Attaching {kernel_id} to an existing kernel...')
    kernel = self._kernels[kernel_id]
    port_names = ['shell_port', 'stdin_port', 'iopub_port', 'hb_port', 'control_port']
    port_names = kernel._random_port_names if hasattr(kernel, '_random_port_names') else port_names
    for port_name in port_names:
      setattr(kernel, port_name, 0)
    kernel.load_connection_file(connection_fname)


  def get_kernel(self, kid):
    return self._kernels[kid]

  @gen.coroutine
  def start_kernel(self, kernel_name=None, **kwargs):
    print('STARTTING ', kernel_name)
    if kernel_name.startswith('kernel_'):
      kernel_id = kwargs.pop('kernel_id', unicode_type(uuid.uuid4()))
      constructor_kwargs = {}
      if self.kernel_spec_manager:
        constructor_kwargs['kernel_spec_manager'] = self.kernel_spec_manager

      km = OpaKernelManager(
          parent=self,
          log=self.log,
          kernel_name=kernel_name,
          **constructor_kwargs,
      )
      km.kernel_spec_manager = OpaKernelSpecManager(parent=self.parent)
      km.start_kernel(**kwargs)
      self._kernels[kernel_id] = km
      self.start_watching_activity(kernel_id)
      self._kernel_connections[kernel_id] = 0
      raise gen.Return(kernel_id)

    else:
      return super().start_kernel(kernel_name=kernel_name, **kwargs)

  async def restart_kernel(self, kernel_id=None):
    try:
      k = self._kernels.pop(kernel_id) 
      k.request_shutdown(restart=True)
      self.start_kernel(k.kernel_name)
    except Exception as e:
      tb.print_exception(e)
      print('failed to restart kernel')
      #self._kernel_connections.pop(kernel_id, None)

  def shutdown_kernel(self, kernel_id, now=False, restart=False):
    try:

      print('SHUTDOWN')
      super().shutdown_kernel(kernel_id, now, restart=False)
      print('DONE SHUTDOWN')
      k = self._kernels.pop(kernel_id) 
      k.ph.duplicate()
      self.start_kernel(k.kernel_name)

    except Exception as e:
      tb.print_exception(e)
      print('failed to shutdown normally')
      self._kernel_connections.pop(kernel_id, None)


#loop.remove_timeout(timeout)
#kernel.remove_restart_callback(on_restart_failed, 'dead')
#    def finish():
#      """Common cleanup when restart finishes/fails for any reason."""
#      if not channel.closed():
#        channel.close()
#      loop.remove_timeout(timeout)
#      kernel.remove_restart_callback(on_restart_failed, 'dead')
#      self._attach_to_latest_kernel(kernel_id)
#
#    def on_reply(msg):
#      self.log.debug("Kernel info reply received: %s", kernel_id)
#      finish()
#      if not future.done():
#        future.set_result(msg)
#
#    def on_timeout():
#      self.log.warning("Timeout waiting for kernel_info_reply: %s", kernel_id)
#      finish()
#      if not future.done():
#        future.set_exception(gen.TimeoutError("Timeout waiting for restart"))
#
#    def on_restart_failed():
#      self.log.warning("Restarting kernel failed: %s", kernel_id)
#      finish()
#      if not future.done():
#        future.set_exception(RuntimeError("Restart failed"))
#
#    kernel.add_restart_callback(on_restart_failed, 'dead')
#    kernel.session.send(channel, "kernel_info_request")
#    channel.on_recv(on_reply)
#    loop = IOLoop.current()
#    timeout = loop.add_timeout(loop.time() + 30, on_timeout)
#    return future

  def list_kernels(self):
    res = super().list_kernels()
    print('KERNEL LIST >> ', res)
    return res


def norm_date(date):
  if date.endswith('0Z'): date = date[:-2]
  return datetime.datetime.fromisoformat(date)


def list_kernels(connection_dir):
  kv = {}
  kernels_by_runid = defaultdict(lambda: {kDate: datetime.datetime.min})
  conn_fnames = glob.glob(f'{connection_dir}/kernel-*.json')
  for conn_fname in conn_fnames:
    with open(conn_fname, 'r') as f:
      con = json.load(f)
      if not kRunId in con: continue
      con[kDate] = norm_date(con[kDate])
      con[kKernelFile] = conn_fname
      con['filename'] = conn_fname
      runid = con[kRunId]
      if kernels_by_runid[runid][kDate] < con[kDate]:
        kernels_by_runid[runid] = con

  for runid, kernel in kernels_by_runid.items():
    kernel[kDate] = kernel[kDate].isoformat()
    kernel[kRunId] = runid
    name = f'{kNamePrefix}{runid}'
    kv[name] = kernel
  return kv


class OpaKernelSpecManager(KernelSpecManager):

  conn_dir = Unicode(config=True, default_value=jupyter_runtime_dir())

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @property
  def mod_kernels(self):
    return list_kernels(self.conn_dir)

  def get_all_specs(self):
    res = super().get_all_specs()

    for name, kernel in self.mod_kernels.items():
      res[name] = dict(spec={'display_name': name, kData: kernel}, resource_dir='')

    return res

  def get_kernel_spec(self, kernel_name):
    if not kernel_name.startswith(kNamePrefix):
      return super().get_kernel_spec(kernel_name)

    if kernel_name not in self.mod_kernels:
      raise NoSuchKernel(kernel_name)

    cur = self.mod_kernels[kernel_name]

    res = KernelSpec()
    res.display_name = kernel_name
    res.metadata = cur
    return res


from ipykernel.ipkernel import IPythonKernel
from traitlets import Type


class OKernel(IPythonKernel):

  def __init__(*args, **kwargs):
    print('CREATE OKERNEL')
    super().__init__(*args, **kwargs)

  def do_shutdown(self, restart):
    print('OK restart down', os.getpid())
    assert 0
    return dict(status="ok", restart=restart)


class OpaKernelApp(IPKernelApp):

  #kernel_class = Type(
  #    OKernel,
  #    klass="ipykernel.kernelbase.Kernel",
  #    help="The Kernel subclass to be used."
  #)

  def write_connection_file(self):
    """write connection info to JSON file"""
    cf = self.abs_connection_file
    super().write_connection_file()

    with open(cf, 'r+') as f:
      con = json.load(f)
      f.seek(0)
      con[kPid] = os.getpid()
      con[kDate] = datetime.datetime.now().isoformat()
      con[kRunId] = self.runid
      con[kKind] = self._kind
      json.dump(con, f, indent=2)
      f.truncate()


class FakeKernel:

  def __init__(self, pid):
    self.pid = pid

  def send_signal(self, signum):
    print('SEND SIG ', self.pid, signum)

    pass


class OpaKernelManager(IOLoopKernelManager):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def start_restarter(self):
    super().start_restarter()

  def start_kernel(self, **kw):
    self.start_restarter()
    self.ready.set_result(True)
    self.load_connection_file(self.kernel_spec.metadata[kKernelFile])

    self._connect_control_socket()
    info = self.kernel_spec.metadata
    print('starting kernel', info)
    self._pid = info.get(kPid, -1)
    self._kind = info.get(kKind, "")
    self.kernel = FakeKernel(self._pid)
    self.ph = ProcHelper(self._pid)

  def interrupt_kernel(self):
    self.ph.interrupt()

  def request_shutdown(self, restart=False):
    if self._kind == KernelKind.BLENDER:
      self.ph.kill()
      if not restart: return
      self.ph.duplicate()
      for i in range(10):
        ksm = OpaKernelSpecManager(parent=self.parent)
        spec = ksm.get_kernel_spec(self.kernel_name)
        print('TRY ', i)
        time.sleep(2)
        ph = ProcHelper(spec.metadata[kPid])
        if ph.active: break

    pass

  def finish_shutdown(self, waittime=None, pollinterval=0.1, **kwargs):
    pass

  def cleanup(self, connection_file=True):
    self.cleanup_ipc_files()
    self._close_control_socket()


class OpaConsoleKernelManager(KernelManager):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._opa_kernel = OpaKernelSpecManager(data_dir=self.data_dir, parent=self)

  @property
  def kernel_spec(self):
    if self._kernel_spec is None and self.kernel_name != '':
      self._kernel_spec = self._opa_kernel.get_kernel_spec(self.kernel_name)
    return self._kernel_spec


#from ipykernel.eventloops import register_integration
#class MyKernel:
#  def __init__(self):
#    self.app = None
#    self.cnt = 0
#  def run(self, kernel):
#    if self.app is not None:
#      print('Refresh here')
#      K.vispy_utils.vispy.app.process_events()
#    kernel.do_one_iteration()
#  def __call__(self, kernel):
#    self.run(kernel)
#
#mk = MyKernel()
#register_integration('mykernel')(mk)

# % gui my_kernel

# get_ipython() # -> available from locals

#from traitlets.config.application import Application
#kernel = Application.instance().kernel
#print(kernel.__dict__.keys())
