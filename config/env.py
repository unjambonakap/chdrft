#!/usr/bin/env python

from __future__ import annotations
import os
import chdrft.utils.misc as cmisc
import tempfile
import numpy as np
import random
import sys
import glog

g_pyqt4 = 'pyqt4'
g_pyqt5 = 'pyqt5'
g_environ_pyqt = 'QT_API'
g_environ_nojax = 'OPA_NOJAX'
g_environ_opa_slim = 'OPA_SLIM'
kSlim = os.environ.get(g_environ_opa_slim, 0) == '1'
kNoJax = os.environ.get(g_environ_nojax, 0) == '1' or kSlim

if not kNoJax:
  import jax

if not kSlim:
  from PyQt5 import QtCore

class IPythonHandler(cmisc.PatchedModel):
  tapp: object = None
  in_jupyter_at_startup: bool = False

  def __post_init__(self):
    self.in_jupyter_at_startup = cmisc.is_interactive() or sys.argv[0].endswith('/ipython') or (
        isinstance(__builtins__, dict) and '__IPYTHON__' in __builtins__
    )

  @property
  def in_ipython(self):
    return self.tapp is not None

  @property
  def in_jupyter(self) -> bool:
    return self.in_jupyter_at_startup or self.in_ipython

  def get_ipython(self):
    if self.tapp is not None:
      return self.tapp.shell
    else:
      return get_ipython() # try jupyter magic stuff



  def drop_to_shell(self, n):
    from IPython.terminal import ipapp
    self.tapp = ipapp.TerminalIPythonApp.instance()
    self.tapp.initialize()
    g_env.run_magic(force2=True)

    locs, g = cmisc.get_n2_locals_and_globals(n=n)
    self.tapp.shell.user_global_ns.update(g | locs)
    self.tapp.start()
    self.tapp = None


g_ipython = IPythonHandler()

class Env:

  def __init__(self):
    self.slim = kSlim
    self.qt5 = None
    self.loaded = False
    self.vispy_app = None
    self.set_qt5(os.environ.get(g_environ_pyqt, g_pyqt5) == g_pyqt5)
    self.qt_imports = None
    self.ran_magic = 0
    self.app = None
    self.rx_qt_sched = None
    self.jax_dump_dir = None


  def create_app(self, ctx=dict()):
    app = qt_imports.QApplication.instance()
    if app is None:
      # needs to be stored otherwise GCed
      self.app = qt_imports.QApplication(ctx.get('other_args', []))
    else:
      assert self.app is not None  # app should be created through g_env?
    return app


  @cmisc.cached_property
  def qt_sched(self):
    # sucks when observable generated from thread not started by qthread
    self.create_app()
    from reactivex.scheduler.mainloop import QtScheduler
    return QtScheduler(self.qt_imports.QtCore)

  def set_qt5(self, v=1, load=0):
    if self.loaded:
      assert self.qt5 == v
      return
    self.vispy_app = [g_pyqt4, g_pyqt5][v]
    os.environ[g_environ_pyqt] = self.vispy_app
    os.environ['PYQTGRAPH_QT_LIB'] = ['PyQt4', 'PyQt5'][v]
    self.qt5 = v
    if load:
      print('loading')
      self.get_qt_imports()
      self.loaded = 1

  def run_magic(self, force=False, force2=False):
    if force2 or (not self.ran_magic and (g_ipython.in_jupyter or force)):

      self.ran_magic = 1
      magic_name = ['qt4', 'qt5'][self.qt5]
      print('Runnign magic', magic_name)
      self.create_app()
      try:
        g_ipython.get_ipython().run_line_magic('gui', magic_name)
      except Exception as  e:
        glog.exception(e)


  def get_qt_imports(self):

    if self.qt5:
      from PyQt5 import QtGui, QtCore, QtWidgets, QtTest, QtWebEngineWidgets
      QWidget, QApplication = QtWidgets.QWidget, QtWidgets.QApplication  # Compat
    else:
      from PyQt4 import QtGui, QtCore, QtTest
      QWidget, QApplication = QtGui.QWidget, QtGui.QApplication
      QtWebEngineWidgets = None
    import vispy.app
    vispy.app.use_app(self.vispy_app)

    self.run_magic()

    return cmisc.Attr(
        QWidget=QWidget,
        QApplication=QApplication,
        QtCore=QtCore,
        QtGui=QtGui,
        QtWidgets=QtWidgets,
        QtWebEngineWidgets=QtWebEngineWidgets,
    )

  def get_qt_imports_lazy(self):

    def handler(name):
      if self.qt_imports is None:
        self.qt_imports = self.get_qt_imports()
      return self.qt_imports.get(name), 1

    return cmisc.Attr(handler=handler)

  def setup(self, app):
    np.random.seed(0)
    random.seed(0)
    os.environ['SAGE_LOCAL'] = '/usr'
    np.set_printoptions(edgeitems=10, linewidth=280)
    np.set_printoptions(formatter={'int_kind': '{:}'.format, 'float_kind': '{:.07f}'.format})

    self.jax_dump_dir = tempfile.TemporaryDirectory(prefix='jaxdump')
    app.global_context.enter_context(self.jax_dump_dir)

    os.environ[
        "XLA_FLAGS"
    ] = f'--xla_force_host_platform_device_count=16 --xla_embed_ir_in_executable --xla_dump_to={self.jax_dump_dir.name}'

    if not kNoJax:
      import jax
      jax.config.update("jax_enable_x64", False)


g_env = Env()
qt_imports = g_env.get_qt_imports_lazy()


def init_jupyter(run_app=False, run_magic=True):
  if run_magic: g_env.run_magic(force=True)
  from IPython.core.display import display, HTML
  display(HTML("<style>.container { width:90% !important; }</style>"))
  from chdrft.main import app
  import atexit
  atexit.register(app.exit_jup)
  if run_app:
    app(force=True, argv=[])

  import numpy as np
  import sys
  import math
  import traceback
  import pandas as pd
  try:
    import chdrft.utils.Z as Z
    import chdrft.utils.K as K
  except:
    print(traceback.print_exc())
    raise
  Z.cmisc.add_to_parent_globals(
      Z=Z,
      np=np,
      sys=sys,
      math=math,
      app=app,
      cmisc=Z.cmisc,
      os=os,
      K=K,
      A=Z.cmisc.A,
      itertools=cmisc.itertools,
      functools=Z.functools,
      pd=pd,
      oplt=K.g_plot_service,
      g_env=g_env,
      qt_imports=qt_imports,
  )
