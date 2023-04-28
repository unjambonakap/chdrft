#!/usr/bin/env python

import os
import chdrft.utils.misc as cmisc
import jax
from chdrft.main import app
import tempfile
import numpy as np
import random


np.random.seed(0)
random.seed(0)
np.set_printoptions(edgeitems=10, linewidth=180)
np.set_printoptions(formatter={'int_kind': '{:},'.format, 'float_kind': '{:.07f},'.format})

g_pyqt4 = 'pyqt4'
g_pyqt5 = 'pyqt5'
g_environ_pyqt = 'QT_API'
os.environ['SAGE_LOCAL'] = '/usr'
jax_dump_dir = app.global_context.enter_context(tempfile.TemporaryDirectory())


os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count=16 --xla_embed_ir_in_executable --xla_dump_to={jax_dump_dir}'

import jax
from jax.config import config
config.update("jax_enable_x64", False)

class Env:

  def __init__(self):
    self.qt5 = None
    self.loaded = False
    self.vispy_app = None
    self.set_qt5(os.environ.get(g_environ_pyqt, g_pyqt5) == g_pyqt5)
    self.qt_imports = None
    self.ran_magic = 0

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

  def get_qt_imports(self):
    import vispy.app
    vispy.app.use_app(self.vispy_app)

    if self.qt5:
      magic_name = 'qt5'
      from PyQt5 import QtGui, QtCore, QtWidgets, QtTest
      QWidget, QApplication = QtWidgets.QWidget, QtWidgets.QApplication  # Compat
    else:
      magic_name = 'qt4'
      from PyQt4 import QtGui, QtCore, QtTest
      QWidget, QApplication = QtGui.QWidget, QtGui.QApplication

    if not self.ran_magic and cmisc.is_interactive():
      self.ran_magic = 1
      print('Runnign magic', magic_name)
      try: get_ipython().run_line_magic('gui', magic_name)
      except:pass
      print('done')

    return cmisc.Attr(
        QWidget=QWidget,
        QApplication=QApplication,
        QtCore=QtCore,
        QtGui=QtGui,
        QtWidgets=QtWidgets,
    )

  def get_qt_imports_lazy(self):

    def handler(name):
      if self.qt_imports is None:
        self.qt_imports = self.get_qt_imports()
      return self.qt_imports.get(name), 1

    return cmisc.Attr(handler=handler)


g_env = Env()
qt_imports = g_env.get_qt_imports_lazy()


def init_jupyter(run_app=False):
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
