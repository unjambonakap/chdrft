import os
from chdrft.utils.misc import proc_path, cwdpath
import chdrft.utils.misc as cmisc
from chdrft.cmds import Cmds
from contextlib import ExitStack
from chdrft.utils.fmt import Format
import glog

def ProgDir(*args):
  return os.path.join(proc_path('~/programmation'), *args)

def OpaDir(*args):
  return ProgDir('opa', *args)
def ElecDir(*args):
  return ProgDir('elec', *args)
def ToolsDir(*args):
  return ProgDir('tools', *args)
def PinDir(*args):
  return ToolsDir('pin-3.7-97619-g0d0c92f4f-gcc-linux', *args)

def CC1110Dir(*args):
  return ElecDir('cc1110', *args)
def FonaDir(*args):
  return ElecDir('fona', *args)
def BBBDir(*args):
  return ElecDir('bbb', *args)
def BBBPythonDir(*args):
  return BBBDir('python/bbb', *args)


def FindEnv(path):
  path=os.path.realpath(path)

  while path!='/':
    cnd=os.path.join(path, 'env.sh')
    if os.path.exists(cnd):
      return cnd
    path=os.path.split(path)[0]

  return ''

Cmds.add(func=ProgDir, args=[])
Cmds.add(func=ElecDir, args=[])
Cmds.add(func=CC1110Dir, args=[])
Cmds.add(func=BBBPythonDir, args=[])
Cmds.add(func=FindEnv, args=[str])

class FileFormatHelper(ExitStack):
  def __init__(self, filename=None, mode='', write=True):
    super().__init__()
    if not mode: mode, filename = FileFormatHelper.GetParams(filename)
    self.write_ = write
    self.filename = filename
    self.mode = mode

  @staticmethod
  def GetParams(path):
    before = path[:path.find('/')]
    mode = ''
    pos = before.find(':')
    if pos != -1:
      mode = path[:pos]
      path = path[pos+1:]
    path = cwdpath(path)
    return mode, path

  @staticmethod
  def Read(filename, mode=''):
    glog.info('Reading file %s', filename)
    with FileFormatHelper(filename, write=False, mode=mode) as f:
      return f.read()

  @staticmethod
  def Write(filename, content, mode=''):
    glog.info('Writing file %s', filename)
    with FileFormatHelper(filename, write=True, mode=mode) as f:
      f.write(content)


  @property
  def binary_mode(self):
    return not self.mode in ('json', 'attr_yaml', 'yaml', 'csv')

  def binary_mode_str(self):
    return 'b' if self.binary_mode else ''

  def __enter__(self):
    super().__enter__()
    if self.write_: self.file = open(self.filename, 'w'+self.binary_mode_str())
    else: self.file = open(self.filename, 'r'+self.binary_mode_str())
    self.enter_context(self.file)
    return self

  def write(self, x):
    x = Format(x)
    if self.mode=='json': x = x.to_json()
    elif self.mode=='csv': x = x.to_csv()
    elif self.mode=='yaml': x = x.to_yaml()
    elif self.mode=='attr_yaml': assert 0
    if self.binary_mode:
      x = x.tobytes()
    self.file.write(x.v)

  def read(self):
    res = self.file.read()
    x = Format(res)
    if self.mode=='json': x = x.from_json()
    elif self.mode=='yaml': x = x.from_yaml()
    elif self.mode=='attr_yaml': x = x.from_yaml().to_attr()
    return x.v

