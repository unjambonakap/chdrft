#!/usr/bin/env python

from __future__ import annotations
from chdrft.config.env import g_env
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
import numpy as np
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import A
import chdrft.utils.misc as cmisc
import numpy as np
import sys
import reactivex.operators
import reactivex.abc
import reactivex as rx
import reactivex.subject.innersubscription
from chdrft.config.env import qt_imports
import glog
from reactivex.subject import BehaviorSubject, Subject
import abc
import typing

from pydantic.v1 import Field
from typing import ClassVar

try:
  import av
except:
  print(f'failed to import av in {__file__}', file=sys.stderr)

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
  internal: object = None
  hint_size: object = None

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
      if isinstance(v, rx.Observable):
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
    from chdrft.dsp.image import ImageData
    i = ImageData.Make(i)
    r = self.process(i)
    return ImageData.Make(r)

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
      if isinstance(v, rx.Observable):
        v = v.value
      res[k] = v
    return res

  def process(self, i):
    return self.f(i, self.get_ctx())


kIdImageProc = FuncImageProcessor(f=lambda x: x)
kErrObj = object()


class CustomSubject(rx.Subject):

  def __init__(self, notify_sub=None):
    super().__init__()
    self._notify_subscriber = notify_sub

  def _subscribe_core(self, observer, scheduler):
    with self.lock:
      self.check_disposed()
      if not self.is_stopped:
        self._notify_subscriber(observer)
        self.observers.append(observer)
        return rx.subject.innersubscription.InnerSubscription(self, observer)

      if self.exception is not None:
        observer.on_error(self.exception)
      else:
        observer.on_completed()
      return Disposable()


if not g_env.slim:

  class QtSig(qt_imports.QtCore.QObject):
    obj = qt_imports.QtCore.pyqtSignal(object)


class WrapRX:

  @classmethod
  def Subject(cls):
    return cls(rx.Subject())

  @classmethod
  def CustomSubject(cls, **kwargs):
    return cls(CustomSubject(**kwargs))

  def __init__(self, obj: rx.Observable):
    rx.operators.replay
    self.obj = obj
    self.value = None
    self.done = cmisc.threading.Event()

  def __getattr__(self, name):
    if hasattr(self.obj, name): return getattr(self.obj, name)
    a = getattr(rx.operators, name, None)
    if a is None:
      if name.endswith('_safe'):
        a = getattr(rx.operators, name[:-5], None)
        if a is None:
          raise AttributeError(name)

        def wrap_f(f, *args, **kwargs):

          @cmisc.logged_failsafe
          def fsafe(*args, **kwargs):
            try:
              return f(*args, **kwargs)
            except:
              cmisc.tb.print_exc()
              return kErrObj

          return WrapRX(
              self.obj.pipe(
                  a(fsafe, *args, **kwargs), rx.operators.filter(lambda x: x is not kErrObj)
              )
          )

        return wrap_f

      else:
        raise AttributeError(name)

    def wrap(*args, **kwargs):
      return WrapRX(self.obj.pipe(a(*args, **kwargs)))

    return wrap

  def set_value(self, x):
    self.value = x

  def subscribe_safe(self, f, qt_sig=False, **subscribe_kwargs):

    @cmisc.logged_failsafe
    def call(*args, **kwargs):
      return f(*args, **kwargs)

    subscribe_kwargs = subscribe_kwargs | dict(
        on_error=glog.exception, on_completed=lambda: self.done.set()
    )
    if qt_sig:
      sig = QtSig()
      sig.obj.connect(call)
      self.obj.subscribe(on_next=lambda x: sig.obj.emit(x), **subscribe_kwargs)
    else:
      self.obj.subscribe(on_next=call, **subscribe_kwargs)

  def listen_value(self):
    self.subscribe_safe(self.set_value)
    return self


kAddAction = True
kRemoveAction = False
class ObservableSet(cmisc.PatchedModel):
  data: set = cmisc.pyd_f(set)
  ev: WrapRX = None

  def __post_init__(self):
    self.ev = WrapRX.CustomSubject(notify_sub=self.notify_sub)


  def merge(self, peer: ObservableSet):
    peer.subscribe(self.add, self.remove)

  def proc_event(self, action_and_item):
    action, item = action_and_item
    if action is kAddAction:
      self.add(item)
    else:
      self.remove(item)

  def notify_sub(self, obs):
    for obj in self.data:
      obs.on_next((kAddAction, obj))

  def add(self, item):
    if item not in self.data:
      self.data.add(item)
      self.ev.on_next((kAddAction, item))

  def remove(self, item):
    self.data.remove(item)
    self.ev.on_next((kRemoveAction, item))

  def subscribe(self, add_action, remove_action):
    self.ev.subscribe_safe(lambda action_and_item: {kAddAction: add_action, kRemoveAction: remove_action}[action_and_item[0]](action_and_item[1]))


  def update(self, items):
    for x in items:
      self.add(x)
    return self

  def __ior__(self, peer):
    for p in peer:
      self.add(p)
    return self


def jupyter_print(x):
  from IPython.display import clear_output
  clear_output()
  print(cmisc.json_dumps(x))


def test(ctx):
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
