#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import A
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.types import *
import chdrft.display.vispy_utils as vispy_utils
from chdrft.struct.base import Box, g_unit_box, Range1D, GenBox
import sys
import datetime
from concurrent.futures import Future
import time
from enum import Enum
import shapely.geometry as geometry
import chdrft.utils.geo as geo_utils
import rx
import rx.core
import rx.subject
from rx import operators as ops
import chdrft.math.sampling as sampling
import cv2
import sortedcontainers
import chdrft.display.grid as grid
from vispy.plot import Fig

from PyQt5 import QtCore
from PyQt5.QtCore import Qt

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSpinBox, QComboBox, QGridLayout, QVBoxLayout,
    QSplitter
)
try:
  import av
except:
  print('failed to import av', file=sys.stderr)

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--infile')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class ImageIO:
  kEndStream = object()

  @staticmethod
  def Connect(a, b):
    if isinstance(a, ImageIO): a.connect_to(b)
    a.subscribe(b.push)

  def __init__(self, f=None):
    self.src = rx.subject.BehaviorSubject(None)
    self.dest = rx.subject.BehaviorSubject(None)
    if f is None:
      f = lambda x: x
    self.f = f
    self._last = None
    self._last_in = None
    self.src.subscribe(self.proc)
    self.res = None


  def register_ctx(self, ctx):
    self._ctx = ctx
    for k, v in ctx.items():
      if isinstance(v, rx.core.Observable):
        v.subscribe(lambda _ : self.rerun())

  def rerun(self):
    self.proc(self._last_in)

  def send_endstream(self):
    self.push(ImageIO.kEndStream)

  def end_stream(self):
    pass

  def proc_impl(self, i):
    return self.f(i)

  def proc(self, i):
    self._last_in = i
    if i is None:
      return
    if id(i) == id(ImageIO.kEndStream):
      self.res = self.end_stream()
      r = i
    else:
      r = self.proc_impl(i)
    self._last = r
    if r is not None:
      self.dest.on_next(r)

  def push(self, e):
    self.src.on_next(e)

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


class VideoReader:

  def __init__(self, fname, format=None):
    self.fname = fname
    self.format = format

    self.container = av.open(fname)

    self.stream = next(s for s in self.container.streams if s.type == 'video')
    self.box = Box.FromSize((self.stream.width, self.stream.height))

    self.time_base = float(self.stream.time_base)
    self.duration_s = self.stream.duration * self.time_base
    self.nframes = self.stream.frames
    self.averate_rate_is = float(self.stream.guessed_rate)
    self.r = Range1D(self.stream.start_time, n=self.stream.duration, is_int=1)
    self._gen = None

  def rel2ts(self, rel):
    rel = TimeSpec.Make(rel, typ='rel')
    if rel.is_ts(): return rel
    return TimeSpec.MakeTS(self.r.from_local(rel.v))

  def ts2rel(self, ts):
    return self.r.to_local(ts)

  def ts2time(self, ts):
    return ts * self.time_base

  def rel2time(self, rel):
    return self.ts2time(self.rel2ts(rel))

  @property
  def fps(self):
    return self.stream.rate

  def frame2numpy(self, frame):
    return frame.to_ndarray(format=self.format)

    #return frame.to_rgb().to_ndarray()[::-1]

  def __next__(self):
    if self._gen is None:
      self._gen = self.gen()
    return next(self._gen)

  def gen(self, keyframe_only=0, out_imgs=False):
    for packet in self.container.demux(self.stream):
      if keyframe_only and not packet.is_keyframe:
        continue
      for frame in packet.decode():
        fo = self.get_frame_out(frame)
        if not out_imgs:
          yield fo
        else:
          yield fo.img

  def get_frame_out(self, frame):
    return cmisc.Attr(
        ts=frame.pts,
        time=self.ts2time(frame.pts),
        rel=self.ts2rel(frame.pts),
        img=vispy_utils.ImageData(self.frame2numpy(frame), inv=0),
    )

  def get_next(self):
    return next(self._gen)

  def seek(self, rel, **kwargs):
    ts = self.r.clampv(self.rel2ts(rel).v)
    self.container.seek(ts, stream=self.stream)

    self._gen = self.gen(**kwargs)

class TimeSpecType(Enum):
  Rel=1,
  TS=2,
  S=3,
  FID=4
class TimeSpec:
  def __init__(self, v, typ):
    self.v = v
    self.typ = typ

  def is_ts(self): return self.typ == TimeSpecType.TS

  @staticmethod
  def Make(v, typ):
    if isinstance(v, TimeSpec): return v
    return TimeSpec(v, typ)

  @staticmethod
  def MakeTS(v): return TimeSpec.Make(v, TimeSpecType.TS)

class VideoFrameDb:
  def __init__(self, fname, **kwargs):
    vs = VideoReader(fname, **kwargs)
    self.id2ts = {}
    self.ts2id = sortedcontainers.SortedDict()
    for i, frame in enumerate(vs.gen()):
      self.id2ts[i] =frame.ts
      self.ts2id[frame.ts] = i


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

  def __init__(self):
    super().__init__()

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

  def __init__(self, f, hint_size=None, endf=None):
    super().__init__()
    self.f = f
    self.hint_size = hint_size
    self.endf = endf

  def get_hint_size(self, insize):
    return self.hint_size

  def end_stream(self):
    if self.endf:
      return self.endf()

  def process(self, i):
    return self.f(i)

class FuncCtxImageProcessor(FuncImageProcessor):

  def __init__(self, f, ctx=A(), **kwargs):
    super().__init__(f, **kwargs)
    self.ctx = ctx
    self.register_ctx(ctx)
  def get_ctx(self):
    res = A()
    for k, v in self.ctx.items():
      if isinstance(v, rx.core.Observable):
        v = v.value
      res[k] = v
    return res


  def process(self, i):
    return self.f(i, self.get_ctx())

kIdImageProc = FuncImageProcessor(lambda x: x)


class RectAnnotator:

  def __init__(self, polygon, model):
    self.polygon = polygon
    self.samples = sampling.sample(polygon, 10)
    self.model = model

  def predict(self, i):
    vals = []
    for v in self.samples:
      vv = i.img_box.to_box_space(v)
      vals.append(i.get_at(vv))
    return np.sum(self.model.predict(vals)) > len(self.samples) / 2

  def __call__(self, i):
    return self.predict(i)


class SegWatcher:

  def __init__(self, polygon, model):
    self.polygon = polygon
    self.annotator = RectAnnotator(polygon, model)

  def __call__(self, i):
    v = self.annotator.predict(i)
    return cmisc.Attr(lines=[self.polygon], color='rg'[int(v)])


class OverlaySink(ImageIO):

  def __init__(self, gw, fgen):
    self.gw = gw
    self.fgen = fgen
    self.objs = []
    super().__init__()

  def proc_impl(self, i):
    self.gw.vctx.remove_objs(self.objs)
    meshes = self.fgen(i)
    res = self.gw.vctx.plot_meshes(meshes, temp=1)
    if self.gw.box is None and res.vb_hint:
      self.gw.box = res.vb_hint
      self.gw.vctx.set_viewbox(res.vb_hint)

    self.objs = res.objs
    return None


class DrawContext:

  def __init__(self, mw, start_key):
    self.mw = mw
    self.canvas = mw.canvas
    self.vctx = mw.vctx
    self.run = 0
    self.start_key = start_key
    self.idx = 0

    self.data = []
    self.objs = []

  def end(self, cancel=0):
    self.run = 0
    self.clear_objs()
    self.finish(cancel)

  def clear_impl(self):
    cx = cmisc.flatten([x.objs for x in self.data])
    self.vctx.remove_objs(cx)
    self.data = []

  def clear(self):
    self.clear_impl()

  def handle_key(self, ev):
    if ev.key == 'Escape' and self.run:
      self.end(cancel=1)
    if ev.key == 'Enter' and self.run:
      self.end(cancel=0)
    if ev.key == 'c' and self.run:
      self.clear()

    elif ev.key == self.start_key:
      self.run = 1
      self.start()
    elif self.run == 1:
      return self.handle_key_impl(ev)
    else:
      return 0
    return 1

  def handle_key_impl(self, ev):
    return 0

  def start(self):
    self.run = 1
    self.start_impl()

  def start_impl(self):
    pass

  def finish(self, cancel=0):
    self.idx += 1
    self.run = 0
    self.finish_internal(cancel)

  def finish_internal(self, cancel):
    raise NotImplementedError()

  def handle_click(self, wpos):
    raise NotImplementedError()

  def build_current_impl(self, wpos):
    raise NotImplementedError()

  def clear_objs(self):
    self.vctx.remove_objs(self.objs)
    self.objs = []


  def create_meshes(self, *args, **kwargs):
    return self.vctx.plot_meshes(*args, **kwargs).objs

  def notify_mouse(self, ev, move=0):
    if not self.run:
      return
    if not move:
      self.handle_click(ev.wpos)

    self.build_current(ev)

  def build_current(self, ev=None, new_objs=None):
    self.clear_objs()
    if new_objs is None:
      new_objs = self.build_current_impl(ev.wpos)
    self.objs = new_objs


class BoxContext(DrawContext):

  def __init__(self, mw):
    super().__init__(mw, 'b')
    self.low = None

  def start_impl(self):
    self.low = None

  def finish_internal(self, cancel):
    pass

  def add_box(self, box, **kwargs):
    obj = cmisc.Attr(geo=box.shapely, box=box, idx=self.idx, typ='box', **kwargs)
    dx = cmisc.Attr(polyline=box.poly_closed(), obj=obj)
    dx.objs = self.vctx.plot_meshes(
        cmisc.Attr(
            lines=[dx],
            color='g',
            points=vispy_utils.POINTS_FROM_LINE_MARKER,
        )
    ).objs
    self.data.append(dx)
    self.finish()

  def handle_click(self, wpos):
    if self.low is None:
      self.low = wpos
    else:
      self.add_box(Box.FromPoints([self.low, wpos]))

  def build_current_impl(self, wpos):
    if self.low is None:
      return []
    box = Box.FromPoints([self.low, wpos])
    return self.create_meshes(cmisc.Attr(lines=[box], color='r'), temp=1)


class PolyContext(DrawContext):

  def __init__(self, mw):
    super().__init__(mw, 'p')
    self.curpoints = []

  def start_impl(self):
    self.curpoints = []

  def add_poly(self, pts, **kwargs):
    if isinstance(pts, geometry.Polygon):
      pts = pts.convex_hull.exterior.coords[:-1]

    obj = cmisc.Attr(geo=geometry.Polygon(pts), idx=self.idx, typ='poly', **kwargs)
    dx = cmisc.Attr(polyline=pts + pts[0:1], obj=obj)
    dx.objs = self.vctx.plot_meshes(
        cmisc.Attr(
            lines=[dx],
            color='r',
            points=vispy_utils.POINTS_FROM_LINE_MARKER,
        )
    ).objs
    self.data.append(dx)

  def finish_internal(self, cancel):

    if cancel:
      return
    if len(self.curpoints) < 3:
      return
    pts = list(geometry.MultiPoint(self.curpoints).convex_hull.exterior.coords)
    self.add_poly(pts)

  def handle_click(self, wpos):
    self.curpoints.append(wpos)

  def build_current_impl(self, wpos):
    tgt = self.curpoints + [wpos]
    return self.create_meshes(
        cmisc.Attr(
            lines=[tgt],
            color='r',
            points=vispy_utils.POINTS_FROM_LINE_MARKER,
        ), temp=1
    )


class RectContext(DrawContext):

  def __init__(self, mw):
    super().__init__(mw, 'r')
    self.base = None
    self.l = None
    self.orth = 0

  def start_impl(self):
    self.base = None
    self.l = None

  def handle_key_impl(self, ev):
    if ev.key == 'o':
      self.orth ^= 1
      return 1

  def finish_internal(self, cancel):
    pass

  def add_rect(self, rect, **kwargs):
    obj = cmisc.Attr(geo=rect.shapely, rect=rect, idx=self.idx, typ='rect', **kwargs)
    dx = cmisc.Attr(polyline=obj.geo, obj=obj)

    dx.objs = self.vctx.plot_meshes(cmisc.Attr(
        lines=[dx],
        color='g',
    )).objs
    self.data.append(dx)
    self.finish()

  def handle_click(self, wpos):
    if self.base is None:
      self.base = wpos
    elif self.l is None:
      self.l = wpos
    else:
      self.add_rect(GenBox.FromPoints(self.base, self.l, self.get_peer(wpos)))

  def get_peer(self, other):
    if not self.orth:
      return other

    v = geo_utils.make_orth(other - self.base, self.l - self.base)
    return self.base + v

  def build_current_impl(self, wpos):
    lines = []
    if self.l is not None:
      lines = [GenBox.FromPoints(self.base, self.l, self.get_peer(wpos)).shapely]
    elif self.base is not None:
      lines.append((self.base, wpos))

    return self.create_meshes(
        cmisc.Attr(
            lines=lines,
            color='r',
            points=vispy_utils.POINTS_FROM_LINE_MARKER,
        ), temp=1
    )


class ClickSlider(QtWidgets.QSlider):

  def __init__(self, *args):
    super().__init__(*args)
    a = 0
    b = 2**30
    self.setMinimum(a)
    self.setMaximum(b)
    self.r = Range1D(a, b + 1, is_int=1)
    self.pub = rx.subject.BehaviorSubject(0)
    self.valueChanged.connect(self.slider_valuechanged)
    self.set_rel(0)

    self.setFocusPolicy(QtCore.Qt.NoFocus)

  def slider_valuechanged(self):
    rel = self.get_rel()
    self.pub.on_next(rel)

  def set_rel(self, rel):
    rx = self.r.clampv(self.r.from_local(rel))
    self.setValue(rx)

  def set_nosig(self, rel):
    prev_state = self.blockSignals(True)
    self.set_rel(rel)
    self.blockSignals(prev_state)

  def get_rel(self):
    return self.r.to_local(self.value())

  def mousePressEvent(self, ev):
    ev.accept()
    r = Range1D(self.minimum(), self.maximum() + 1, is_int=1)
    if self.orientation() == Qt.Horizontal:
      rel = ev.x() / self.width()
    else:
      rel = 1 - ev.y() / self.height()
    self.set_rel(rel)

    super().mousePressEvent(ev)


class SingleChildW(QWidget):

  def __init__(self):
    super().__init__()

    self.layout = QtWidgets.QGridLayout()
    self.setLayout(self.layout)
    self.child = None

  def set_single_child(self, child):
    self.layout.addWidget(child, 0, 0)
    self.child = child

  def installEventFilter(self, ev):
    self.child.installEventFilter(ev)


class FigWidget(SingleChildW):
  def __init__(self):
    super().__init__()
    self.fig=  Fig()
    self.fig.create_native()
    self.set_single_child(self.fig.native)


class GraphWidget(SingleChildW):

  def __init__(self, hint_box=None):
    super().__init__()

    self.box = hint_box
    if hint_box is None:
      hint_box = g_unit_box
    vctx = vispy_utils.render_for_meshes(run=0)
    self.vctx = vctx
    self.canvas = vctx.canvas
    self.canvas.create_native()
    self.key_cbs = {}

    self.canvas.events.key_press.connect(lambda ev: self.key_event(ev, 1))
    self.canvas.events.key_release.connect(lambda ev: self.key_event(ev, 0))

    self.canvas.events.mouse_press.connect(self.mouse_press)
    self.canvas.events.mouse_move.connect(self.mouse_move)

    self.dcs = cmisc.Attr()
    dcs_classes = [PolyContext, BoxContext, RectContext]
    for cl in dcs_classes:
      self.dcs[cl.__name__] = cl(self)

    self.set_single_child(self.canvas.native)
    self.wpos = None

    #self.canvas.native.setParent(self)

  def key_event(self, ev, press):
    if not press:
      return
    self.augment_ev(ev)

    for dc in self.dcs.values():
      if dc.run and dc.handle_key(ev):
        ev.native.accept()
        return

    for dc in self.dcs.values():
      if dc.handle_key(ev):
        ev.native.accept()
        return

    for k, v in self.key_cbs.items():
      if ev.key == k:
        v(ev)
        return

  def mouse_press(self, ev):
    self.wpos = self.vctx.screen_to_world(ev.pos)
    self.augment_ev(ev)
    for dc in self.dcs.values():
      dc.notify_mouse(ev)

  def augment_ev(self, ev):
    ev.src = self
    ev.wpos = self.wpos

  def mouse_move(self, ev):
    self.wpos = self.vctx.screen_to_world(ev.pos)
    self.augment_ev(ev)
    for dc in self.dcs.values():
      dc.notify_mouse(ev, move=1)


class VideoSource(SingleChildW):

  def __init__(self, infile, **kwargs):
    super().__init__()

    self.video = VideoReader(infile, **kwargs)
    self.setWindowTitle(f'Floating {infile}')

    splitter = QSplitter(Qt.Vertical)

    s2 = QtWidgets.QHBoxLayout()
    self.slider = ClickSlider(Qt.Horizontal)
    self.text = QtWidgets.QLabel()
    self.text.setText('abcdef')

    s2.addWidget(self.slider, 8)
    s2.addWidget(self.text, 2)

    self.graphic = GraphWidget(self.video.box)
    self.graphic.key_cbs['Right'] = lambda _: self.next_frame()
    self.graphic.key_cbs['G'] = lambda _: self.goto()
    vx = QtWidgets.QWidget()
    vx.setLayout(s2)

    splitter.addWidget(self.graphic)
    splitter.addWidget(vx)

    self.slider.valueChanged.connect(self.slider_valuechanged)

    self.task = Future()
    self.task.set_result(None)
    self.cur_objs = []

    self.source = ImageIO()

    self.set_single_child(splitter)
    self.frame_db = None

  def installEventFilter(self, ev):
    super().installEventFilter(ev)
    self.graphic.installEventFilter(ev)

  def slider_valuechanged(self):
    if not self.task.done():
      return
    rel = self.slider.get_rel()
    self.task = cmisc.chdrft_executor.submit_log(self.set_pos, rel)

  def set_pos(self, rel, **kwargs):
    self.video.seek(rel, **kwargs)
    frame = self.video.get_next()
    self.set_at_frame(frame)

  def goto(self):
    if self.frame_db is None:
      print('frame db not set')
      return

    fid = grid.qt_dialog(self.graphic, 'Frame id', typ=int)
    ts = self.frame_db.id2ts.get(fid, None)
    if ts is None:
      print(f'{fid} not found in frame db')
      return

    self.set_pos(TimeSpec.MakeTS(ts))

  def set_at_frame(self, frame):
    p1 = time.perf_counter()

    td = datetime.timedelta(seconds=frame.time)
    self.text.setText(str(td))

    self.graphic.vctx.free_objs()
    self.slider.set_nosig(frame.rel)
    self.graphic.vctx.remove_objs(self.cur_objs)
    self.cur_frame = frame
    self.cur_objs = self.graphic.vctx.plot_meshes(cmisc.Attr(images=[frame.img]).objs)
    self.source.push(frame.img)
    p2 = time.perf_counter()
    #print('set took ', p2 - p1)

  def next_frame(self):
    self.set_at_frame(self.video.get_next())


def sink_graph():
  gw = GraphWidget()
  main_objs = []

  def proc(i):
    gw.vctx.remove_objs(main_objs)
    if gw.box is None:
      gw.vctx.set_viewbox(i.box)
      gw.box = i.box
      nsink.hint_size = i.box
    main_objs.clear()
    main_objs.extend(gw.vctx.plot_meshes(cmisc.Attr(images=[i])).objs)
    return i

  nsink = ImageIO(f=proc)
  nsink._obj = gw
  return nsink


canny = FuncImageProcessor(lambda x: cv2.Canny(x.u8.img, 100, 140))


def dilate(n=3):
  kernel = np.ones((n, n))
  return FuncImageProcessor(lambda x: cv2.dilate(x.img, kernel))


def erode(n=3):
  kernel = np.ones((n, n))
  return FuncImageProcessor(lambda x: cv2.erode(x.img, kernel))


def to_gray8():
  return FuncImageProcessor(lambda x: x.u8)


def to_grayfloat():
  return FuncImageProcessor(lambda x: x.float)


def stream_mean():
  dx = A(s=None, cnt=0)

  def acc(x):
    dx.cnt += 1
    if dx.s is None:
      dx.s = x.img
    else:
      dx.s += x.img
    return x

  def endf():
    return dx.s / dx.cnt

  return FuncImageProcessor(acc, endf=endf)


class MetricWidget(QtWidgets.QLabel, ImageIO):
  def __init__(self, name):
    super().__init__()
    self.setAutoFillBackground(True)
    self.name = name
    self.v = None
    self.refresh()

  def refresh(self):
    self.setText(f'{self.name}: {self.v}')
    self.setAlignment(QtCore.Qt.AlignCenter)

  def proc_impl(self, v):
    self.v = v
    self.refresh()
  @staticmethod
  def Make(name=None, obs=None):
    res =  MetricWidget(name)
    ImageIO.Connect(obs, res)
    return res

def test(ctx):
  win, app = create_window(ctx)
  win.show()
  app.exec_()


def timestamp_to_frame(timestamp, stream):
  fps = stream.rate
  time_base = stream.time_base
  start_time = stream.start_time
  frame = (timestamp - start_time) * float(time_base) / float(fps)
  return frame


def test_av(ctx):
  import av
  container = av.open(ctx.infile)
  from av.video import VideoStream

  video_stream = next(s for s in container.streams if s.type == 'video')
  total_frame_count = 0

  time_base = float(video_stream.time_base)
  duration = video_stream.duration * time_base
  nframes = video_stream.frames

  target_frame = nframes // 2
  rate = float(video_stream.average_rate)
  target_sec = target_frame * 1 / rate
  a = time.perf_counter()
  container.seek(int(video_stream.start_time + duration / 2 / time_base), stream=video_stream)

  current_frame = None
  frame_count = 0

  for packet in container.demux(video_stream):
    for frame in packet.decode():
      if current_frame is None:
        current_frame = timestamp_to_frame(frame.pts, video_stream)
      else:
        current_frame += 1
      # start counting once we reach the target frame
      if current_frame is not None and current_frame >= target_frame:
        frame_count += 1
      dx = frame.to_rgb().to_ndarray()
      print(time.perf_counter() - a)
      vispy_utils.render_for_meshes(cmisc.Attr(images=[vispy_utils.ImageData(dx)]))
      return 0


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
