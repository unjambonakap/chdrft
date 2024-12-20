#!/usr/bin/env python

import os
import time
import threading
import subprocess as sp

import glob
import os
import os.path
import re

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

from jupyter_client.ioloop import IOLoopKernelManager

from traitlets import default, Integer, Unicode, Instance
import json

kName = 'name'
kDate = 'date'
kData = 'data'
kKernelFile = 'kernel_file'
kNamePrefix = 'kernel_'

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

  @gen.coroutine
  def start_kernel(self, kernel_name=None, **kwargs):
    print('STARTTING ', kernel_name)
    if kernel_name.startswith('kernel_'):
      kernel_id = kwargs.pop('kernel_id', unicode_type(uuid.uuid4()))
      constructor_kwargs = {}
      if self.kernel_spec_manager:
        constructor_kwargs['kernel_spec_manager'] = self.kernel_spec_manager

      km = OpaKernelManager(parent=self, log=self.log, kernel_name=kernel_name,
                  **constructor_kwargs ,)
      km.kernel_spec_manager = OpaKernelSpecManager(parent=self.parent)
      km.start_kernel(**kwargs)
      self._kernels[kernel_id] = km
      self.start_watching_activity(kernel_id)
      self._kernel_connections[kernel_id] = 0
      raise gen.Return(kernel_id)

    else:
      return super().start_kernel(kernel_name=kernel_name, **kwargs)
  def shutdown_kernel(self, kernel_id, now=False, restart=False):
    self._kernel_connections.pop(kernel_id, None)
    pass

  async def restart_kernel(self, kernel_id, now=False):
    print('RESTART aSK')
    assert 0
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


class OpaKernelSpecManager(KernelSpecManager):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    print('WE ARE HERE')


  @property
  def mod_kernels(self):
    kv = {}
    kernels_by_runid = defaultdict(lambda: {kDate:datetime.datetime.min})
    connection_dir = '/tmp/kernels/'
    conn_fnames = glob.glob(f'{connection_dir}/kernel*.json')
    print('COnnection dir >> ', connection_dir)
    for conn_fname in conn_fnames:
      with open(conn_fname, 'r') as f:
        con = json.load(f)
        try:
          print('TRY', conn_fname)
          if 0: # only valid on windows to check if file is opened
            with open(conn_fname, 'a') as f2:
              continue
        except PermissionError:
          pass
        if not kName in con: continue
        con[kKernelFile] = conn_fname
        kv[kNamePrefix + con[kName]] = con
    print('GOOOT KERNEL ', kv)

    return kv

  def get_all_specs(self):
    res = super().get_all_specs()

    for name, kernel in self.mod_kernels.items():
      res[name] = dict(spec={'display_name':name, kData:kernel}, resource_dir='')

    return res

  def get_kernel_spec(self, kernel_name):
    print('GEt kernel spec', kernel_name)
    if not kernel_name.startswith(kNamePrefix):
      return super().get_kernel_spec(kernel_name)

    if kernel_name not in self.mod_kernels:
      raise NoSuchKernel(kernel_name)

    cur = self.mod_kernels[kernel_name]
    print(cur)

    res = KernelSpec()
    res.display_name = kernel_name
    res.metadata = cur
    return res


class OpaKernelApp(IPKernelApp):

  def write_connection_file(self):
    """write connection info to JSON file"""
    cf = self.abs_connection_file
    super().write_connection_file()

    with open(cf, 'r+') as f:
      con = json.load(f)
      f.seek(0)
      con['date'] = datetime.datetime.now().isoformat()
      con['runid'] = self.runid
      json.dump(con, f, indent=2)
      f.truncate()

class OpaKernelManager(IOLoopKernelManager):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def start_kernel(self, **kw):
    self.start_restarter()
    print(self.kernel_spec_manager.get_kernel_spec)
    print(self.kernel_spec_manager.get_kernel_spec(self.kernel_name))
    print(self.kernel_spec)
    self.load_connection_file(self.kernel_spec.metadata[kKernelFile])
    print('CONNECTNIG')
    self._connect_control_socket()
    print('done connect control')
    print(self._control_socket)

  def request_shutdown(self, restart=False):
    pass

  def finish_shutdown(self, waittime=None, pollinterval=0.1):
    pass

  def cleanup(self, connection_file=True):
    self.cleanup_ipc_files()
    self._close_control_socket()
def run_notebook(*args, notebook_dir='./', background=False, current_pid=-1):
  proc = sp.Popen(
      [
          'jupyter',
          'notebook',
          '--NotebookApp.kernel_manager_class=chdrft.runkernel.OpaEntryKernelManager',
          '--NotebookApp.kernel_spec_manager_class=chdrft.runkernel.OpaKernelSpecManager',
          '--NotebookApp.token=',
          '--no-browser',
          f'--OpaKernelManager.caller_kid={current_pid}',
          f'--notebook-dir={notebook_dir}',
      ]
          #'--Session.key=b""',
  )
  if not background:
    while True:
      try:
        if proc.wait() is not None: break
      except KeyboardInterrupt:
        pass

if __name__=='__main__': run_notebook()


