#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import numpy as np
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import A
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import numpy as np
import chdrft.display.vispy_utils as vispy_utils
import sys
import rx
import rx.core
import rx.subject
from pydantic.v1 import Field
from typing import ClassVar

try:
  import av
except:
  print('failed to import av', file=sys.stderr)

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class ImageIO(cmisc.PatchedModel):
  kEndStream: ClassVar[object] = object()
  kNop: ClassVar[object] = object()

  src: rx.subject.BehaviorSubject = None
  dest: rx.subject.BehaviorSubject = None
  f: object = None
  last: object = None
  last_in: object = None
  res: object = None
  disposed: bool = False

  def dispose(self):
    if self.disposed: return
    self.disposed = True
    self.send_endstream()

  @staticmethod
  def Connect(a, b):
    if a is None or b is None: return
    if isinstance(a, ImageIO): a.connect_to(b)
    else: a.subscribe(b.push)

  def __init__(self, f=None, **kwargs):
    super().__init__(**kwargs)
    self.src = rx.subject.BehaviorSubject(None)
    self.dest = rx.subject.BehaviorSubject(None)
    if f is None:
      f = lambda x: x
    self.f = f
    self.src.subscribe(self.proc)

  def register_ctx(self, ctx):
    self._ctx = ctx
    for k, v in ctx.items():
      if isinstance(v, rx.core.Observable):
        v.subscribe(lambda _: self.rerun())

  def rerun(self):
    self.proc(self.last_in)

  def send_endstream(self):
    self.push(ImageIO.kEndStream)

  def end_stream(self):
    self.dispose()

  def proc_impl(self, i):
    return self.f(i)

  @cmisc.logged_failsafe
  def proc(self, i):
    self.last_in = i
    if i is None:
      return
    is_end = False
    if id(i) == id(ImageIO.kEndStream):
      is_end = True
      r = i
    else:
      r = self.proc_impl(i)
    self.last = r
    if r is not None:
      self.dest.on_next(r)

    if is_end:
      self.res = self.end_stream()

  def push(self, e):
    self.src.on_next(e)

  def push_nop(self):
    self.src.on_next(ImageIO.kNop)

  def connect_to(self, nxt):
    self.dest.subscribe(nxt.push)


class VideoWriter(ImageIO):

  def __init__(self, fname, format='rgb24'):
    super().__init__()
    self.fname = fname
    self.container = av.open(fname, 'w')
    self.ostream = self.container.add_stream('mpeg4')
    self.format = format
    self.first = 1

  def proc_impl(self, i):
    self.feed(i)

  def end_stream(self):
    self.finish()

  def feed(self, img):
    if img is None:
      return
    if self.first:
      self.ostream.width = img.img_box.width
      self.ostream.height = img.img_box.height
      self.first = 0
    frame = av.VideoFrame.from_ndarray(img.img, format=self.format)
    self.container.mux(self.ostream.encode(frame))

  def finish(self):
    self.container.close()


class RollingEvent(ImageIO):
  state: list = Field(default_factory=list)
  k: int

  def proc_impl(self, x):
    self.state = [x] + self.state
    self.state = self.state[:self.k]
    return self.state


class SceneControllerInput(cmisc.PatchedModel):
  torques: dict = cmisc.defaultdict(float)
  speed: float = 0
  scale: float = 0


def gamepad2scenectrl_io() -> ImageIO:

  def mixaxis(v1, v2):
    v = (max(v1, v2) + 1) / 2
    if v1 > v2: return -v
    return v

  mp = {
      ('LT', 'RT'): lambda s, v: s.torques.__setitem__(0, mixaxis(*v)),
      'RIGHT-X': lambda s, v: s.torques.__setitem__(1, v),
      'RIGHT-Y': lambda s, v: s.torques.__setitem__(2, v),
      'LEFT-X': lambda s, v: s.__setattr__('speed', v),
      ('LB', True): lambda s: s.__setattr__('scale', max(s.scale - 1, 0)),
      ('RB', True): lambda s: s.__setattr__('scale', s.scale + 1),
  }

  state = SceneControllerInput()

  def proc(tb):
    for k, cb in mp.items():
      if isinstance(k, tuple):
        if isinstance(k[-1], bool):
          if tb.get(k):
            cb(state)
        else:
          cb(state, (tb[x] for x in k))
      else:
        val = tb.get(k)
        cb(state, val)
    return state

  return ImageIO(proc)


class IdIO(ImageIO):

  def __init__(self):
    super().__init__(self)

  def proc_impl(self, i):
    return i


def pipe_connect(*args):
  args = cmisc.flatten(args)
  for i in range(len(args) - 1):
    args[i].connect_to(args[i + 1])
  return args[-1]


class ArraySink(ImageIO):

  def __init__(self):
    super().__init__(self)
    self.tb = []

  def end_stream(self):
    return self.tb

  def proc_impl(self, obj):
    self.tb.append(obj)


def pipe_process(inlist, *args, want_array=0):
  args = cmisc.flatten(args)
  res = None
  args = list(args)
  if want_array:
    args.append(ArraySink())
  for i in range(len(args) - 1):
    args[i].connect_to(args[i + 1])

  rx.from_(inlist).subscribe(args[0].push)
  args[0].send_endstream()
  return args[-1].res


def tree_connect(root, tree_desc):
  if isinstance(tree_desc, list):
    if not tree_desc:
      return root
    root = tree_connect(root, tree_desc[0])
    return tree_connect(root, tree_desc[1:])
  elif isinstance(tree_desc, cmisc.A):
    for child in tree_desc.children:
      tree_connect(root, child)
  else:
    root.connect_to(tree_desc)
    return tree_desc


def tree_process(inlist, tree_desc):
  src = IdIO()
  tree_connect(src, tree_desc)
  rx.from_(inlist).subscribe(src.src)


class ImageProcessor(ImageIO):

  def get_hint_size(self, insize):
    return insize

  def process(self, i):
    raise NotImplementedError()

  def proc_impl(self, i):
    return self(i)

  def __call__(self, i):
    if i is None:
      return None
    i = vispy_utils.ImageData.Make(i)
    r = self.process(i)
    return vispy_utils.ImageData.Make(r)

  def chain(self, other):
    return ChainedImageProcessor([self, other])


class PixProcessor(ImageProcessor):

  def __init__(self, f):
    super().__init__()
    self.f = f

  def process(self, i):
    i = i.img
    if len(np.shape(i)) == 2:
      vec = i.reshape(-1, 3)
    else:
      vec = i.reshape(-1, np.shape(i)[2])
    res = self.f(vec)
    return res.reshape(i.shape[:2])


class ChainedImageProcessor(ImageProcessor):

  def __init__(self, tb=[]):
    super().__init__()
    self.tb = list(tb)

  def get_hint_size(self, insize):
    for proc in self.tb:
      insize = proc.get_hint_size(insize)
    return insize

  def process(self, i):
    for proc in self.tb:
      i = proc(i)
    return i


class FuncImageProcessor(ImageProcessor):
  hint_size: int = None
  endf: object = None

  def get_hint_size(self, insize):
    return self.hint_size

  def end_stream(self):
    if self.endf:
      return self.endf()

  def process(self, i):
    return self.f(i)


class FuncCtxImageProcessor(FuncImageProcessor):

  ctx: A = Field(default_factory=A)

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.register_ctx(self.ctx)

  def get_ctx(self):
    res = A()
    for k, v in self.ctx.items():
      if isinstance(v, rx.core.Observable):
        v = v.value
      res[k] = v
    return res

  def process(self, i):
    return self.f(i, self.get_ctx())


kIdImageProc = FuncImageProcessor(f=lambda x: x)


def jupyter_print(x):
  from IPython.display import clear_output
  clear_output()
  print(cmisc.json_dumps(x))


def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
