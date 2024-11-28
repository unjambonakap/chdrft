#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import A
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import numpy as np
from chdrft.utils.opa_types import *
import chdrft.display.vispy_utils as vispy_utils
from chdrft.struct.base import Box, g_unit_box, Range1D, GenBox
import sys
import datetime
from concurrent.futures import Future
import time
from enum import Enum
import shapely.geometry as geometry
import chdrft.utils.geo as geo_utils
import reactivex as rx
import chdrft.math.sampling as sampling
import cv2
import sortedcontainers
import chdrft.display.grid as grid
from vispy.plot import Fig
import chdrft.utils.colors as ocolors

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from chdrft.utils.rx_helpers import ImageIO, FuncImageProcessor

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QSplitter
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


class VideoReader:

  def __init__(self, fname, format='rgb24'):
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
    self._gen = self.gen()

  def rel2ts(self, rel):
    rel = TimeSpec.Make(rel, typ=TimeSpecType.Rel)

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
    fx = self.frame2numpy(frame)
    return cmisc.Attr(
        base=frame,
        ts=frame.pts,
        time=self.ts2time(frame.pts),
        rel=self.ts2rel(frame.pts),
        img=vispy_utils.ImageData(fx, inv=1),
    )

  def get_next(self):
    return next(self._gen, None)

  def seek(self, rel, **kwargs):
    ts = self.r.clampv(self.rel2ts(rel).v)
    self.container.seek(int(ts), stream=self.stream)

    self._gen = self.gen(**kwargs)


class TimeSpecType(Enum):
  Rel = 1
  TS = 2
  S = 3
  FID = 4


class TimeSpec(cmisc.PatchedModel):

  v: float
  typ: TimeSpecType

  def is_ts(self):
    return self.typ == TimeSpecType.TS

  @staticmethod
  def Make(v, typ):
    if isinstance(v, TimeSpec): return v
    return TimeSpec(v=v, typ=typ)

  @staticmethod
  def MakeTS(v):
    return TimeSpec.Make(v, TimeSpecType.TS)


class VideoFrameDb:

  def __init__(self, fname, **kwargs):
    vs = VideoReader(fname, **kwargs)
    self.id2ts = {}
    self.ts2id = sortedcontainers.SortedDict()
    for i, frame in enumerate(vs.gen()):
      self.id2ts[i] = frame.ts
      self.ts2id[frame.ts] = i


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
    self.vctx: vispy_utils.VispyCtx = mw.vctx
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
    cx = cmisc.flatten([x.meshes for x in self.data])
    self.vctx.remove_objs(cx)
    self.data = []

  def clear(self):
    self.clear_impl()

  def handle_key(self, ev, last_mouse):
    if ev.key == 'Escape' and self.run:
      self.end(cancel=1)
    if ev.key == 'Enter' and self.run:
      self.end(cancel=0)
    if ev.key == 'Delete' and self.run:
      self.delete(last_mouse)
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

  def add_item(self, entry, *descs):
    descs = [x.update(fromobj=entry) for x in descs]
    entry.meshes = self.create_meshes(*descs)
    entry.src = self
    self.data.append(entry)

  def delete(self, last_mouse):
    ql = self.vctx.do_query(last_mouse.wpos)

    for cnd in ql.cnds:
      u = cnd.obj.obj
      if u.get('src') != self: continue
      self.data.remove(u)
      self.vctx.remove_objs(u.meshes)
      break

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
    pass

  def handle_click(self, wpos):
    pass

  def build_current_impl(self, wpos):
    return []

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


class PointContext(DrawContext):

  def __init__(self, mw):
    super().__init__(mw, 'u')
    self.wpos = None
    self.clear()

  def clear_impl(self):
    super().clear_impl()
    self.cmap = ocolors.ColorPool(looping=True)

  def build_current_impl(self, wpos):
    return self.create_meshes(cmisc.Attr(points=[wpos], color=self.cmap(remove=False)), temp=1)

  def handle_click(self, wpos):
    self.wpos = wpos
    col = self.cmap()
    self.add_item(
        A(obj=self.wpos, geo=geometry.Point(list(wpos)), color=col),
        A(
            points=[self.wpos],
            color=col,
        )
    )

  def finish_internal(self, cancel):
    pass


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
    self.add_item(
        dx, cmisc.Attr(
            lines=[dx],
            color='g',
            points=vispy_utils.POINTS_FROM_LINE_MARKER,
        )
    )
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
    self.add_item(dx, A(
        lines=[dx],
        color='r',
        points=vispy_utils.POINTS_FROM_LINE_MARKER,
    ))

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
    self.fig = Fig()
    self.fig.create_native()
    self.set_single_child(self.fig.native)


class OMouseEvent(cmisc.PatchedModel):
  raw: object
  wpos: np.ndarray
  pos: np.ndarray
  modifiers: list
  is_left: bool
  src: object
  move: bool = False

  @classmethod
  def Make(cls, vctx: vispy_utils.VispyCtx, raw: object, src: object, move: bool = False):
    return OMouseEvent(
        raw=raw,
        wpos=vctx.screen_to_world(raw.pos),
        pos=raw.pos,
        modifiers=raw.modifiers,
        is_left=raw.button == 1,
        src=src,
        move=move,
    )


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
    dcs_classes = [PolyContext, BoxContext, RectContext, PointContext]
    for cl in dcs_classes:
      self.dcs[cl.__name__] = cl(self)

    self.set_single_child(self.canvas.native)
    self.wpos = None
    self.last_mouse = None

    #self.canvas.native.setParent(self)

  def key_event(self, ev, press):
    if not press:
      return
    self.augment_ev(ev)

    for dc in self.dcs.values():
      if dc.run and dc.handle_key(ev, self.last_mouse):
        ev.native.accept()
        return

    for dc in self.dcs.values():
      if dc.handle_key(ev, self.last_mouse):
        ev.native.accept()
        return

    for k, v in self.key_cbs.items():
      if ev.key == k:
        v(ev)
        return

  def mouse_press(self, ev):
    ev = OMouseEvent.Make(self.vctx, ev, self, move=False)
    self.wpos = ev.wpos
    self.last_mouse = ev

    if not vispy_utils.vispy_keys.SHIFT in ev.modifiers: return
    if not ev.is_left: return
    for dc in self.dcs.values():
      dc.notify_mouse(ev)

  def augment_ev(self, ev):
    ev.src = self
    ev.wpos = self.wpos

  def mouse_move(self, ev):
    ev = OMouseEvent.Make(self.vctx, ev, self, move=True)
    self.last_mouse = ev
    self.wpos = ev.wpos
    for dc in self.dcs.values():
      dc.notify_mouse(ev, move=1)


class VideoSource(SingleChildW):

  def __init__(self, infile, set_frame_db=False, **kwargs):
    super().__init__()
    self.frame_db = None
    self.first = True

    self.video = VideoReader(infile, **kwargs)
    self.setWindowTitle(f'Floating {infile}')
    if set_frame_db:
      self.frame_db = VideoFrameDb(infile)

    splitter = QSplitter(Qt.Vertical)

    s2 = QtWidgets.QHBoxLayout()
    self.slider = ClickSlider(Qt.Horizontal)
    self.text = QtWidgets.QLabel()
    self.text.setText('abcdef')

    s2.addWidget(self.slider, 8)
    s2.addWidget(self.text, 2)

    self.graphic = GraphWidget(self.video.box)
    self.graphic.key_cbs['Right'] = lambda _: self.next_frame()
    self.graphic.key_cbs['Left'] = lambda _: self.prev_frame()
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
    print('seeek', rel)
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
    if frame is None: return
    p1 = time.perf_counter()

    td = datetime.timedelta(seconds=frame.time)
    self.text.setText(str(td))

    self.graphic.vctx.free_objs()
    print('FUUU ', frame.rel)
    self.slider.set_nosig(frame.rel)
    self.graphic.vctx.remove_objs(self.cur_objs)
    self.cur_frame = frame
    self.cur_objs = self.graphic.vctx.plot_meshes(cmisc.Attr(images=[frame.img])).objs

    if self.first:
      self.first = False
      self.graphic.vctx.set_viewbox()

    self.source.push(frame.img)
    p2 = time.perf_counter()
    #print('set took ', p2 - p1)

  def next_frame(self):
    self.set_at_frame(self.video.get_next())

  def prev_frame(self):

    id = self.frame_db.ts2id.get(self.cur_frame.ts, None)
    if id is None:
      return
    ts = self.frame_db.id2ts.get(id - 1, None)
    if ts is None: return
    self.set_pos(TimeSpec.MakeTS(ts))


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
  nsink.internal = gw
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
