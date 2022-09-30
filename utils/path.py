import os
from chdrft.utils.misc import proc_path, cwdpath
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
from chdrft.cmds import Cmds
from contextlib import ExitStack
from chdrft.utils.fmt import Format
import glog
import sys


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
  path = os.path.realpath(path)

  while path != '/':
    cnd = os.path.join(path, 'env.sh')
    if os.path.exists(cnd):
      return cnd
    path = os.path.split(path)[0]

  return ''


Cmds.add(func=ProgDir, args=[])
Cmds.add(func=ElecDir, args=[])
Cmds.add(func=CC1110Dir, args=[])
Cmds.add(func=BBBPythonDir, args=[])
Cmds.add(func=FindEnv, args=[str])


class FormatParams:

  def __init__(self, mode=None, path=None, conf=None):
    self.mode = mode
    if conf is None: conf = cmisc.Attr()
    self.conf = conf
    self.path = path


class FileFormatHelper(ExitStack):

  def __init__(self, filename=None, mode='', write=True, **kwargs):
    super().__init__()
    if mode: params = FormatParams(mode=mode, path=filename)
    else: params = FileFormatHelper.GetParams(filename, **kwargs)
    self.write_ = write
    self.params = params

  @staticmethod
  def GetParams(path, default_mode=''):
    before = path[:path.find('/')]
    mode = ''
    pos = before.find(':')
    if pos != -1:
      mode = path[:pos]
      path = path[pos + 1:]
    if not path.startswith('@'): path = cwdpath(path)
    conf = None

    if mode == '':
      ext = os.path.splitext(path)[1]
      ext2mode = {'.yaml': 'yaml', '.json': 'json', '.pickle': 'pickle', '.csv': 'csv'}
      mode = ext2mode.get(ext, default_mode)
    else:
      print(mode)
      split = mode.split(',', 1)
      mode = split[0]
      if len(split) == 2:
        print(split[1])
        conf = Format(split[1]).from_conf().v

    return FormatParams(mode, path, conf)

  @staticmethod
  def Read(filename, mode='', **kwargs):
    glog.info('Reading file %s', filename)
    with FileFormatHelper(filename, write=False, mode=mode, **kwargs) as f:
      return f.read()

  @staticmethod
  def Write(filename, content, mode='', **kwargs):
    glog.info('Writing file %s %s', filename, kwargs)
    with FileFormatHelper(filename, write=True, mode=mode, **kwargs) as f:
      f.write(content)

  @property
  def binary_mode(self):
    return not self.mode in ('json', 'attr_yaml', 'yaml', 'csv', 'conf', 'txt')

  def binary_mode_str(self):
    return 'b' if self.binary_mode else ''

  @property
  def mode(self):
    return self.params.mode

  @property
  def filename(self):
    return self.params.path

  def __enter__(self):
    super().__enter__()
    if self.filename == '@stdin': self.file = sys.stdin
    elif self.filename == '@stdout': self.file = sys.stdout
    else:
      if self.write_: self.file = open(self.filename, 'w' + self.binary_mode_str())
      else: self.file = open(self.filename, 'r' + self.binary_mode_str())
      self.enter_context(self.file)
    return self

  def write(self, x):
    x = Format(x)
    if self.mode == 'json': x = x.to_json()
    elif self.mode == 'csv': x = x.to_csv()
    elif self.mode == 'yaml': x = x.to_yaml()
    elif self.mode == 'pickle': x = x.to_pickle()
    elif self.mode == 'txt': x = x
    elif self.mode == 'attr_yaml': assert 0
    if self.binary_mode:
      x = x.tobytes()
    self.file.write(x.v)

  def read(self):
    if self.mode == 'cache':
      from chdrft.utils.cache import FileCacheDB
      with FileCacheDB(filename=self.filename, ro=1) as cache:
        return cache[self.params.conf.key]

    res = self.file.read()
    x = Format(res)
    if self.mode == 'json': x = x.from_json()
    elif self.mode == 'yaml': x = x.from_yaml()
    elif self.mode == 'conf': x = x.from_conf()
    elif self.mode == 'csv': x = x.to_strio().from_csv(**self.params.conf)
    elif self.mode == 'txt': x = x
    elif self.mode == 'pickle': x = x.from_pickle()
    elif self.mode == 'attr_yaml': x = x.from_yaml().to_attr()
    return x.v


class AnnotatedFiles(cmisc.ExitStack):

  def __init__(self, path, key, w=0):
    super().__init__()
    cmisc.makedirs(path)
    self.path = path
    self.key = key
    self.id = 0
    self.entries = []
    self.conf_file = f'{self.path}/chx_conf_{self.key}.yaml'

  def __enter__(self):
    super().__enter__()
    self.callback(self.write)
    return self

  def load(self):
    conf = FileFormatHelper.Read(self.conf_file)
    self.id = conf.id
    self.entries = conf.entries
    for x in self.entries:
      self.norm(x)

  def query(self):
    return cmisc.asq_query(self.entries)

  def write(self):
    FileFormatHelper.Write(self.conf_file, A(entries=self.entries, id=self.id))

  def make_fname(self, data):
    return cmisc.yaml_dump_custom(data, default_flow_style=True
                                 ).replace('/', '_').replace('\n', '').replace(' ', '')

  def add_file(self, data, ext=None):
    idx = self.id
    self.id += 1
    e = A(data=data, id=idx, fname=f'chx_file_{self.key}_{self.make_fname(data)}_{idx:04}{ext}')
    self.norm(e)
    self.entries.append(e)
    return e.path

  def norm(self, e):
    e.path = f'{self.path}/{e.fname}'
    return e.path
#with  AnnotatedFiles('./data', 'rots') as af:
#    mesh_in =meshio.read('test.stl')
#    for rot in gen_rot_grid(3):
#        p = af.add_file(A(rot=rot.as_euler('ZYX')), '.stl')
#        mout = transform_mesh(mesh_in, mat=centered_rot(rot.as_matrix()))
#        mout.write(p)

#af = AnnotatedFiles('./data', 'rots')
#af.load()
#for x in af.entries:
#    Z.shutil.copyfile(x.path, f'./tmp/t{x.id}.vtk')
