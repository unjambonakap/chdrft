#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
import glog
import math, sys, os
import numpy as np
from chdrft.utils.types import *
from enum import Enum
from chdrft.utils.path import FileFormatHelper
from chdrft.utils.fmt import Format
from chdrft.struct.base import Box
import rx
import rx.subject

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSpinBox, QComboBox, QGridLayout, QVBoxLayout,
    QSplitter
)
from PyQt5.QtCore import (QSize)
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
from chdrft.display.ui import PlotManager, OpaPlot

global flags, cache
flags = None
cache = None

g_data = A()

def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def qt2py(v):
  if isinstance(v, (QtCore.QSize, QtCore.QSizeF)):
    return (v.width(), v.height())
  if isinstance(v, (QtCore.QPoint, QtCore.QPointF)):
    return (v.x(), v.y())
  return v



class GridPlacerPolicy(Enum):
  Dispatch = 'dispatch'
  Normal = 'normal'

class GridPlacer:
  kMoveDrag = 'move'
  kSizeDrag = 'size'

  def __init__(self, dc, w, size=(1, 1), policy=GridPlacerPolicy.Dispatch):
    self.w = w
    self.policy = policy
    self.grid_box = Box.FromSize(size, is_int=1)
    self.zorder = []
    self.w2e = A.MakeId()
    self.box = Box.FromSize((800, 600), is_int=1)
    self.dc = dc
    self.dc.mods.append(
        DragMode('left', self.dragstart, self.dragmove, self.dragend, GridPlacer.kMoveDrag)
    )
    self.dc.mods.append(
        DragMode('right', self.dragstart, self.dragmove, self.dragend, GridPlacer.kSizeDrag)
    )

    self.dragdata = None
    self.actions = {
        QtCore.Qt.Key_F: self.toggle_full,
    }
    self.toggle_state = None

  def toggle_full(self, pos):
    if self.toggle_state is None:
      hit = self.get_first_hit(pos)
      if hit is None:
        return
      self.toggle_state = A(old_box=hit.box, e=hit)
      self.change_req(hit, self.grid_box)
      self.make_top(hit, force=1)
    else:
      self.change_req(self.toggle_state.e, self.toggle_state.old_box)
      self.toggle_state = None

  def dragstart(self, smouse):
    hit = self.get_first_hit(smouse.pos)
    if not hit:
      return 0
    self.make_top(hit)
    self.dragdata = A(hit=hit, orig_box=hit.box, start_pos=self.to_space(smouse.pos))
    if smouse.dragtype == GridPlacer.kSizeDrag:
      corner_id = hit.box.closest_corner(self.dragdata.start_pos)
      self.dragdata.fixed_corner = hit.box.get_grid(corner_id ^ 3)  # opposite corner
      self.dragdata.move_corner = hit.box.get_grid(corner_id)
    return 1

  def dragmove(self, smouse):
    obox = self.dragdata.orig_box
    curpos = self.to_space(smouse.pos)

    delta = curpos - self.dragdata.start_pos
    reqbox = None
    if smouse.dragtype == GridPlacer.kMoveDrag:
      reqbox = (obox.to_double() + delta)
    else:
      reqbox = Box.FromPoints([self.dragdata.fixed_corner, self.dragdata.move_corner + delta])
    reqbox = reqbox.to_int_round()
    self.change_req(self.dragdata.hit, reqbox)

  def dragend(self, smouse):
    self.dragdata = None

  def update_all(self):
    for e in reversed(self.zorder):
      self.update(e)
    self.w.update()

  def change_req(self, e, box=None, pos=None, size=None):
    if pos is None:
      pos = e.box.low
    if size is None:
      size = e.box.size
    if box is None:
      box = Box(low=pos, size=size, is_int=1)
    if not box.empty and self.grid_box.contains(box):
      e.box = box
      self.update(e)
    return e

  def resize(self, newsize):
    self.box = Box.FromSize(newsize, is_int=1)
    self.update_all()

  def covered(self, box):
    return bool(self.get_by_z(lambda x: x.box.intersects(box)))

  def get_overlapping(self, e):
    return self.get_by_z(lambda x: x != e and x.box.intersects(e.box))

  def move_above(self, obj, target):
    if obj == target:
      return
    obj.w.stackUnder(target.w)
    obj.w.raise_()
    self.zorder.remove(obj)
    idx = self.zorder.index(target)
    self.zorder.insert(idx, obj)

  def move_below(self, obj, target):
    if obj == target:
      return
    obj.w.stackUnder(target.w)
    self.zorder.remove(obj)
    idx = self.zorder.index(target)
    self.zorder.insert(idx + 1, obj)

  def make_top(self, e, force=0):
    if force:
      ov = self.zorder
    else:
      ov = self.get_overlapping(e)
    if not ov:
      return
    self.move_above(e, ov[0])

  @cmisc.yield_wrapper
  def get_by_z(self, pred):
    for i, e in enumerate(self.zorder):
      if pred(e):
        yield e

  def get_hit_list(self, pos):
    pos = self.box.to_box_space(pos) * self.grid_box.size
    tb = self.get_by_z(lambda x: x.box.contains(pos))
    return tb

  def get_first_hit(self, pos):
    return cmisc.firstor(self.get_hit_list(pos))

  def cycle(self, pos):
    tb = self.get_hit_list(pos)
    if len(tb) < 2:
      return

    tg = tb[0]
    self.move_below(tg, self.zorder[-1])

  def find_empty_place(self, box):
    while True:
      for pos in self.grid_box:
        cnd = box.make(dim=None, low=pos)
        if not self.covered(cnd): return cnd
      self.resize_grid((1,1), double=0)

  def make(self, widget, box=None, pos=None, size=None, label=None, **kwargs):
    if box is None:
      if size is None: size = (1, 1)
      if pos is None: pos = (0, 0)
      box = Box(low=pos, dim=size)
      if self.policy == GridPlacerPolicy.Dispatch:
        box = self.find_empty_place(box)


    e = A(
        w=widget,
        box=box.to_int_round().intersection(self.grid_box),
        gw=self,
      label=label,
    )
    e.change_req = lambda **kwargs: self.change_req(e, **kwargs)
    return e


  @property
  def asq(self): return cmisc.asq_query(self.zorder)

  def find_by_label(self, label):
    if label is None: return None
    return self.asq.where(lambda x: x.label == label).first_or_default(None)

  def place(self, widget, **kwargs):
    e = self.make(widget, **kwargs)
    e.update(kwargs)
    self.zorder.insert(0, e)
    self.w2e[widget] = e
    self.update(e)
    return e

  def remove_pos(self, pos):
    self.remove(e=self.get_first_hit(pos))
  def remove(self, w=None, e=None):
    if e is None and w is None: return
    if e is None: e = self.w2e[w]
    del self.w2e[e.w]
    self.zorder.remove(e)
    e.w.setParent(None)
    e.w.deleteLater()

  def to_space(self, pos, e=None):
    pos = self.box.to_box_space(pos) * self.grid_box.size
    if e is None:
      return pos
    if not isinstance(e, Box):
      e = e.box
    return e.to_box_space(pos)

  def from_space(self, pos, e=None):
    if e is not None:
      if not isinstance(e, Box):
        e = e.box
      pos = e.from_box_space(pos) / self.grid_box.size
    return self.box.from_space(pos)

  def update(self, e):
    nbox = self.box.from_box_space(e.box.to_double() / self.grid_box.size).to_int()
    e.w.move(*nbox.low)
    e.w.resize(*nbox.size)
    e.w.update()

  def resize_grid(self, delta, double=1):
    used = Box.Union([e.box for e in self.zorder])
    if double: delta = self.grid_box.size * delta
    nbox = self.grid_box.make(low=None, size=self.grid_box.size + delta)
    if not nbox.contains(used):
      return
    self.grid_box = nbox
    self.update_all()


class QtUtil:

  kModMapping = [
      [QtCore.Qt.ShiftModifier, 'shift'],
      [QtCore.Qt.ControlModifier, 'ctrl'],
      [QtCore.Qt.AltModifier, 'alt'],
      [QtCore.Qt.MetaModifier, 'meta'],
  ]
  kMouseMapping = [
      [QtCore.Qt.MouseButton.LeftButton, 'left'],
      [QtCore.Qt.MouseButton.RightButton, 'right'],
  ]

  def GetState(v, mapping):
    res = cmisc.A()
    v = int(v)
    for key, name in mapping:
      res[name] = v & key
    return res

  def GetModState(w):
    mstate = QtUtil.GetState(QtWidgets.QApplication.keyboardModifiers(), QtUtil.kModMapping)
    mstate.pos = np.array(qt2py(w.mapFromGlobal(QtGui.QCursor().pos())))
    return mstate

  def GetMouseState(w, ev):
    smouse = QtUtil.GetState(ev.buttons(), QtUtil.kMouseMapping)
    smouse.pos = np.array(qt2py(w.mapFromGlobal(QtGui.QCursor().pos())))
    return smouse


class DragMode:

  def __init__(self, button, startcb, movecb, endcb, type):
    self.button = button
    self.startcb = startcb
    self.movecb = movecb
    self.endcb = endcb
    self.type = type

  def state_check(self, smouse):
    return smouse[self.button]


class DragContext:

  def __init__(self):
    self.mods = []
    self.active_mode = None
    self.scur = None

  def check_active(self):
    if self.active_mode is None:
      return 0
    if self.active_mode.state_check(self.scur):
      return 1
    self.finish()
    return 0

  def finish(self):
    self.active_mode.endcb(self.scur)
    self.active_mode = None

  def start(self, mod):
    self.active_mode = mod
    self.scur.dragtype = mod.type
    if not self.active_mode.startcb(self.scur):
      self.active_mode = None

  def set_state(self, smouse):
    self.scur = smouse
    if self.active_mode:
      smouse.dragtype = self.active_mode.type

  def mouse_press(self, smouse):
    self.set_state(smouse)
    if self.check_active():
      return
    for mod in self.mods:
      if mod.state_check(self.scur):
        self.start(mod)
        break

  def mouse_release(self, smouse):
    self.set_state(smouse)
    self.check_active()

  def mouse_move(self, smouse):
    self.set_state(smouse)
    if not self.check_active():
      return
    self.active_mode.movecb(smouse)


class GridWidget(QWidget):

  def __init__(self, **kwargs):
    super().__init__()
    self.dc = DragContext()
    self.gp = GridPlacer(self.dc, self, **kwargs)

  @cmisc.logged_failsafe
  def resizeEvent(self, ev):
    self.gp.resize(qt2py(ev.size()))

  def setup(self, parent):
    self.setParent(parent)
    self.gp.setup(self.size)

  def add(self, w, **kwargs):
    w.setParent(self)
    e = self.gp.place(w, **kwargs)
    w.installEventFilter(self)
    w.show()
    return e

  def remove(self, w):
    self.gp.remove(w)

  @cmisc.logged_failsafe
  def eventFilter(self, obj, ev):

    if not self.get_mods().ctrl:
      if isinstance(ev, QtGui.QKeyEvent):
        if ev.key() == QtCore.Qt.Key_Tab:
          self.handle_tab(ev)
          return True
      return False

    if ev.type() == QtCore.QEvent.MouseButtonPress:
      self.mousePressEvent(ev)
    elif ev.type() == QtCore.QEvent.MouseMove:
      self.mouseMoveEvent(ev)
    elif ev.type() == QtCore.QEvent.MouseButtonRelease:
      self.mouseReleaseEvent(ev)
    elif ev.type() == QtCore.QEvent.KeyPress:
      self.keyPressEvent(ev)
    elif ev.type(
    ) in (QtCore.QEvent.KeyRelease, QtCore.QEvent.ContextMenu, QtCore.QEvent.GraphicsSceneMove):
      return True
    elif isinstance(ev, QtWidgets.QGraphicsSceneMouseEvent):
      return True
    else:
      #print('Forward ', ev, ev.type())
      return False
    return True

  def handle_tab(self, ev):
    #QtWidget.QApplication.sendEvent(self, ev)
    pass

  def childEvent(self, ev):
    #print('event >> ', ev, ev.child(), ev.added(), ev.polished(), ev.removed())
    pass

  @cmisc.logged_failsafe
  def keyPressEvent(self, ev):
    curpos = self.get_mods().pos
    if ev.key() == QtCore.Qt.Key_C or ev.key() == QtCore.Qt.Key_Tab:
      self.gp.cycle(curpos)
    if ev.key() == QtCore.Qt.Key_Delete:
      self.gp.remove_pos(curpos)
    elif ev.key() == QtCore.Qt.Key_Up:
      self.gp.resize_grid((0, 1))
    elif ev.key() == QtCore.Qt.Key_Down:
      self.gp.resize_grid((0, -1))
    elif ev.key() == QtCore.Qt.Key_Left:
      self.gp.resize_grid((-1, 0))
    elif ev.key() == QtCore.Qt.Key_Right:
      self.gp.resize_grid((1, 0))
    if ev.key() == QtCore.Qt.Key_Escape:
      self.setFocus()

    action = self.gp.actions.get(ev.key(), None)
    if action:
      action(curpos)

  def get_mods(self):
    return QtUtil.GetModState(self)

  def get_mouse_state(self, ev):
    return QtUtil.GetMouseState(self, ev)

  @cmisc.logged_failsafe
  def mousePressEvent(self, ev):
    smouse = self.get_mouse_state(ev)
    if self.get_mods().ctrl:
      self.dc.mouse_press(smouse)

  @cmisc.logged_failsafe
  def mouseReleaseEvent(self, ev):
    smouse = self.get_mouse_state(ev)
    if self.dc.active_mode or self.get_mods().ctrl:
      self.dc.mouse_release(smouse)

  @cmisc.logged_failsafe
  def mouseMoveEvent(self, ev):
    smouse = self.get_mouse_state(ev)
    if self.dc.active_mode or self.get_mods().ctrl:
      self.dc.mouse_move(smouse)


class MainWindow(QMainWindow):

  def __init__(self, main=None, title='Floating test'):
    super().__init__()
    self.ctx = cmisc.ExitStack().__enter__()
    self.resize(800, 600)
    self.setWindowTitle(title)
    self.closed = rx.subject.BehaviorSubject(False)
    if main is not None:
      self.setCentralWidget(main)

  def closeEvent(self, ev):
    print('CLOSING ', ev)
    self.closed.on_next(True)
    self.ctx.close()
    super().closeEvent(ev)

  def setup(self, main=None):
    if main is not None:
      self.setCentralWidget(main)
    self.show()


class GridWidgetHelper:

  def __init__(self, gw=None, **kwargs):
    if gw is None:
      win = MainWindow()
      gw = GridWidget(**kwargs)
      win.setup(gw)
      self.win = win

    self.gw = gw
    self.pm = PlotManager(A(w=self.gw, add=gw.add, remove=gw.remove))

  def create_plot(self, *args, **kwargs):
    plot = self.pm.create_plot(OpaPlot, *args, **kwargs)
    return plot

  def remove_plot(self, plot):
    self.pm.remove_plot(plot)


class MetricStoreWidget(QWidget):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.layout = QGridLayout()
    self.setLayout(self.layout)
    self.children = []
    self.setAutoFillBackground(True)

  def get_by_w(self, w):
    return cmisc.firstor([x for x in self.children if x.w == w])

  def add(self, w, label=None):
    e = cmisc.A(w=w, label=None)
    self.children.append(e)
    self.update()
    return e

  def asq(self): return cmisc.asq_query(self.children)

  def find_by_label(self, label):
    if label is None: return None
    return self.asq.where(lambda x: x.label == label).first_or_default(None)

  def remove(self, w):
    self.children.remove(self.get_by_w(w))
    w.setParent(None)
    self.update()

  def update(self):
    n = len(self.children)
    if n == 0:
      return
    ncol = int(math.ceil(n**0.5))
    nrow = int(math.ceil(n / ncol))
    sz = np.array(qt2py(self.size()))
    esize = sz // (ncol, nrow)
    for i, child in enumerate(self.children):
      r = i // ncol
      c = i % ncol
      child.w.resize(*esize)
      self.layout.addWidget(child.w, r, c, QtCore.Qt.AlignCenter)
    for i in range(r):
      self.layout.setRowStretch(i, 1)
    for i in range(c):
      self.layout.setColumnStretch(i, 1)
    super().update()

def qt_dialog(parent, name, default='', typ=None):
  res, ok = QtGui.QInputDialog.getText(parent, name, name,
                                                    QtWidgets.QLineEdit.Normal,
                                                    default)
  if not ok: return None
  if typ is not None: res = typ(res)
  return res

def create_app(ctx=dict()):
  print('CREATING AP')
  app = QApplication.instance()
  if app is None:
    # needs to be stored otherwise GCed
    g_data.app = QApplication(ctx.get('other_args', []))
  return app


def create_window(ctx):
  app = create_app(ctx)
  win = MainWindow(ctx)
  return win, app


def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
