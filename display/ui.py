#!/usr/bin/env python

from chdrft.display.base import qt_imports
import sys
import numpy as np
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')  # fuckoff

import pyqtgraph.ptime as ptime
import scipy.ndimage as ndimage
from scipy import signal
import glog
import pandas as pd
from vispy.color import Color, get_colormap

from chdrft.utils.misc import to_list, Attributize, proc_path, is_interactive
from chdrft.utils.colors import ColorPool
from asq.initiators import query as asq_query
from chdrft.display.dsp_ui import DspTools
from chdrft.struct.base import Range2D, Intervals

from ipykernel.kernelapp import IPKernelApp
from IPython.utils.frame import extract_module_locals

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
from chdrft.config.env import g_env

from chdrft.dsp.datafile import Dataset2d, Dataset

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class EventHelper:

  def __init__(self, ev=None, modifiers=None, pos=None):
    self.ev = ev
    if ev is not None and modifiers is None:
      modifiers = int(ev.modifiers())
    self.mods = modifiers

    if pos is None and hasattr(ev, 'pos'):
      pos = ev.pos()

    if pos is not None:
      pos = pg.Point(pos)
    self._pos = pos

  def has_meta(self):
    return (self.mods & qt_imports.QtCore.Qt.MetaModifier) != 0

  def has_shift(self):
    return (self.mods & qt_imports.QtCore.Qt.ShiftModifier) != 0

  @staticmethod
  def FromModifiers():
    return qt_imports.QtWidget.QApplication.keyboardModifiers()

  def pos(self):
    return self._pos

  def key(self):
    return self.ev.key()


class SamplingRegion(pg.LinearRegionItem):

  def __init__(self, plot):
    super().__init__()
    self.hide()
    self.plot = plot
    plot.addItem(self)
    self.removed = False
    #self.contextMenu.addAction('test').triggered.connect(self.action1)

  def __del__(self):
    self.plot.removeItem(self)

  def remove(self):
    self.plot.removeItem(self)
    self.removed = True

  def mousePressEvent(self, ev):
    glog.debug('clicked')
    super().mousePressEvent(ev)
    glog.debug('%s %s', self.lines[0].pos(), self.lines[1].pos())

  def keyPressEvent(self, ev):
    ev.accept()
    glog.debug('sampling press')
    super().keyPressEvent(ev)

  def set_left(self, pos):
    self.lines[0].setPos(pos)

  def set_right(self, pos):
    self.lines[1].setPos(pos)

  def notify_key_press(self, h):
    h.ev.accept()
    if h.key() == qt_imports.QtCore.Qt.Key_H:
      self.set_left(h.pos().x())
    elif h.key() == qt_imports.QtCore.Qt.Key_L:
      self.set_right(h.pos().x())
    else:
      h.ev.ignore()

  def notify_mouse_press(self, ev):
    if self.isVisible():
      return
    self.show()
    view_coord = self.plot.to_view_coord(ev.pos())
    glog.debug('put at %s', view_coord)
    self.lines[0].setPos(view_coord)
    self.lines[1].setPos(view_coord)
    self.lines[1].setEnabled(True)

  def notify_mouse_release(self, ev):
    pass

  def notify_mouse_drag(self, ev):
    glog.debug('notify drag')
    if ev.isAccepted():
      return
    if self.isVisible():
      return
    self.show()


class QContainer(list):

  def query(self):
    return asq_query(self)


class PlotMenu:

  def __init__(self, plot):
    self.plot = plot
    self.menu = qt_imports.QtWidgets.QMenu()
    self.submenus = QContainer()

    glog.debug('set menu')

  def add_menu(self, name):
    submenu = self.menu.addMenu(name)
    self.submenus.append(Attributize(name=name, obj=submenu))
    return submenu

  def remove_menu(self, name):
    target = self.submenus.query().where(lambda x: x.name == name).to_list()[0]
    self.submenus.remove(target)
    self.menu.removeMenu(target)


class RegionManager:
  """Manage a bunch of select regions on one plot"""

  def __init__(self, plot):
    self.plot = plot
    self.regions = []
    self.last = None
    self.set_menu()
    self.create_keys = [qt_imports.QtCore.Qt.Key_H, qt_imports.QtCore.Qt.Key_L]
    self.active_region = None

  def set_menu(self):
    region_menu = self.plot.menu.add_menu('regions')
    region_menu.addAction('clear', self.clear)

  def clear(self):
    for r in self.regions:
      r.remove()
    self.regions = []

  def add_region(self):
    region = SamplingRegion(self.plot)
    glog.debug('new region')
    self.regions.append(region)
    self.last = region
    return region

  def remove_region(self, region):
    region.remove()
    self.plot.notify_remove_region(region)
    for i in range(len(self.regions)):
      if self.regions[i] == region:
        del self.regions[i]
        break
    else:
      assert 0
    self.last = None
    if len(self.regions) > 0:
      self.last = self.regions[-1]

  def key_press(self, h):
    if h.key() in self.create_keys:
      if not self.last:
        self.add_region()
      self.last.notify_key_press(h)
      return

    if h.key() == qt_imports.QtCore.Qt.Key_Escape or h.key() == qt_imports.QtCore.Qt.Key_Delete:
      r = self.find_hovering()
      if r:
        self.remove_region(r)

  def notify_press(self):
    self.active_region = None

  def mouse_press(self, ev):
    self.active_region = self.find_hovering()
    if not EventHelper(ev).has_meta():
      return
    if ev.button() != qt_imports.QtCore.Qt.LeftButton:
      return
    region = self.add_region()
    region.notify_mouse_press(ev)

  def filter(self, dataset):
    manager = self.plot.manager
    xlow, ylow = self.getRegion()
    ndata = self.plot.data.extract_range(xlow, ylow)
    manager.create_plot(OpaPlot, ndata)

  def regions_list(self):
    return self.regions

  def find_hovering(self):
    for region in self.regions:
      if region.mouseHovering:
        return region
    return None


class Marks:

  def __init__(self, plot, data):
    self.data = data
    self.plot = plot
    self.lines = []

  def clear(self):
    for x in self.lines:
      self.plot.removeItem(x)
    self.lines = []

  def recompute(self, start, period):
    self.clear()

    pos = start
    while pos >= self.data.min_x():
      pos -= period
    pos += period
    while pos <= self.data.max_x():
      self.add_line(pos)
      pos += period

  def add_line(self, pos):
    line = pg.InfiniteLine(pos, 90, movable=False)
    line.show()
    self.plot.addItem(line)
    #line.sigPositionChangeFinished.connect(lambda: self.drag_line(line))
    self.lines.append(line)

  def build_dataset(self, filter):
    if not self.lines:
      return
    nx = []
    ny = []
    for line in self.lines:
      x = line.getXPos()
      if not filter.should_keep(x):
        continue
      nx.append(x)
      ny.append(self.data.sample_at(x))

    return Dataset(x=nx, y=ny, orig_dataset=self.data)


class Sampler:

  def __init__(self, plot):
    self.plot = plot
    self.regions = self.plot.regions
    self.marks = Marks(self.plot, self.plot.data)

  def setup_menu(self, menu):
    sampler_menu = menu.add_menu('sampler')
    sampler_menu.addAction('sample', self.sample)
    sampler_menu.addAction('sample_regions', self.sample_regions)
    marks_menu = menu.add_menu('marks')
    marks_menu.addAction('mark', self.mark)
    marks_menu.addAction('sample', self.sample_marks)
    marks_menu.addAction('sample_regions', self.sample_marks_regions)
    marks_menu.addAction('clear', self.marks.clear)

  def mark(self):
    r = self.plot.last_hovered
    n, ok = qt_imports.QtWidgets.QInputDialog.getInt(self.plot, 'Count', 'count')
    if not ok or n < 2:
      glog.debug('bad choice %s %s', ok, n)
      return
    xl, xh = r.getRegion()
    if xl > xh:
      xl, xh = xh, xl
    self.marks.recompute(xl, (xh - xl) / (n - 1))

  def sample_marks(self):
    self.do_sample_marks()

  def sample_marks_regions(self):
    self.do_sample_marks(self.regions.regions_list())

  def do_sample_marks(self, regions=None):
    a = Intervals(regions=regions)
    nd = self.marks.build_dataset(a)
    self.plot.manager.create_plot(OpaPlot, nd)

  def sample(self):
    region = self.plot.last_hovered

    glog.debug('hovered > > %s', region)
    if region:
      self.do_sample_regions([region])

  def sample_regions(self):
    self.do_sample_regions(self.plot.regions.regions_list())

  def do_sample_regions(self, regions):
    if len(regions) == 0:
      return
    xl = [r.getRegion() for r in regions]
    a = Intervals(xl)
    fig = self.plot.manager.create_plot(OpaPlot)
    for plot in self.plot.opa_plots:
      nd = a.filter_dataset(plot.data)
      fig.add_plot(PlotEntry(nd))


class ImageEntry:

  def __init__(self, data, **kwargs):
    self.data = data
    self.kwargs = kwargs
    self.obj = None
    self.cmap = kwargs.get('cmap', None)
    self.plot_widget = None

  def register(self, plot_widget):
    self.plot_widget = plot_widget
    ndata = self.data
    self.obj = pg.ImageItem(ndata.y)
    if self.cmap:
      self.obj.setLookupTable(self.cmap.getLookupTable())

    self.obj.setCompositionMode(qt_imports.QtGui.QPainter.CompositionMode_Plus)

    self.plot_widget.viewbox.addItem(self.obj)
    r = ndata.box.to_qrectf()
    self.obj.setRect(r)

  def unregister(self):
    assert 0


class PlotEntry:

  def __init__(self, data, **kwargs):
    self.data = data
    self.kwargs = kwargs
    self.obj = None
    self.plot_widget = None
    self.data.sig_replot_cb = self.update

  def register(self, plot_widget):
    self.plot_widget = plot_widget
    self.obj = self.plot_widget.plot(
        self.data.get_x(), self.data.get_y(), name=self.data.name, **self.kwargs
    )
    self.plot_widget.sigRangeChanged.connect(self.sig_range_changed)
    self.obj.curve.setClickable(True)
    self.obj.curve.sigClicked.connect(self.clicked_plot)

  def unregister(self):
    self.plot_widget.sigRangeChanged.disconnect(self.sig_range_changed)
    self.plot_widget.removeItem(self.obj)
    if self.plot_widget.legend: self.plot_widget.plotItem.legend.removeItem(self.data.name)

  def sig_range_changed(self, new_view_range):
    self.obj.curve._mouseShape = None

  def clicked_plot(self):
    self.plot_widget.action_manager.notify_clicked_curve(self)

  def update(self):
    self.obj.setData(self.data.get_x(), self.data.get_y())


class ShadowPlotEntry:

  def __init__(self, main_entry, view_range, plot_widget, **kwargs):
    self.main_entry = main_entry
    self.active_range = None
    self.cur_data = None
    self.shift = pg.Point(0, 0)
    self.plot_widget = plot_widget
    self.obj = self.plot_widget.plot([], [], **kwargs)
    self.plot_widget.sigRangeChanged.connect(self.sig_range_changed)

    self.update_view_range(view_range)

  def update_view_range(self, new_view_range):
    new_view_range = Range2D(new_view_range)
    self.cur_view_range = new_view_range
    self.maybe_update_data()

  def maybe_update_data(self):
    if self.active_range is None or not self.active_range.shift(self.shift
                                                               ).contains(self.cur_view_range):
      self.view_range = self.cur_view_range.double()
      self.update()
      return True
    return False

  def sig_range_changed(self, _, new_view_range):
    glog.debug('RNAGE CHANGED >> %s', new_view_range)
    self.update_view_range(new_view_range)

  def update(self):
    self.cur_data = self.main_entry.data.extract_by_x(self.cur_view_range.xr.shift(-self.shift[0])
                                                     ).shift(self.shift)
    self.obj.setData(x=self.cur_data.get_x(), y=self.cur_data.get_y())

  def update_shift(self, new_shift):
    self.shift = new_shift
    if not self.maybe_update_data():
      self.update()


class DragCurve:

  def __init__(self, plot):
    self.entry = None
    self.shadow_entry = None
    self.plot = plot
    self.plot_entries = self.plot.opa_plots

    self.start_pos = None
    self.end_pos = None
    self.drag = False
    self.tot_shift = pg.Point()

  def find_best_curve(self, pos):
    tb = [(1e100, None)]
    view_dim = self.plot.get_view_range().length()

    for entry in self.plot_entries:
      pt = entry.data.get_closest_point(pos)
      normalized_dist = (pt - pos) / view_dim
      #glog.debug(pt, pos, view_dim, normalized_dist, normalized_dist.length())
      tb.append((normalized_dist.length(), entry))
    tb.sort(key=lambda x: x[0])

    precision_threshold = 1. / 20
    if tb[0][0] > precision_threshold:
      return None

    if len(tb) == 1 or tb[0][0] * 3 > tb[1][0]:
      return None
    return tb[0][1]

  def is_active(self):
    return self.entry is not None

  def should_activate_on_click(self, h):
    return h.has_shift()

  def maybe_activate(self, h):
    if not self.should_activate_on_click(h):
      return False
    if self.entry is not None:
      return False
    best_curve = self.find_best_curve(h.pos())
    if best_curve is None:
      return False

    self.do_activate(h, best_curve)
    return True

  def do_activate(self, h, curve):
    self.entry = curve
    self.start_pos = h.pos()
    self.end_pos = self.start_pos
    self.tot_shift = pg.Point()
    self.entry.obj.setAlpha(0.3, False)
    self.shadow_entry = ShadowPlotEntry(
        self.entry, self.plot.get_view_range(), self.plot, pen={'color': self.plot.get_color()}
    )
    self.plot.addItem(self.shadow_entry.obj)
    self.drag = True

  def cancel(self):
    self.plot.sigRangeChanged.disconnect(self.shadow_entry.sig_range_changed)
    self.plot.removeItem(self.shadow_entry.obj)
    if self.entry is not None:
      self.entry.obj.setAlpha(1, False)
    self.entry = None
    self.shadow_entry = None

  def notify_release(self, h):
    if self.drag:
      self.drag = False
      self.tot_shift += self.end_pos - self.start_pos
      self.start_pos = self.end_pos
      return True
    return False

  def notify_press(self, h):
    if not self.should_activate_on_click(h):
      return False
    self.drag = True
    self.start_pos = h.pos()
    return True

  def notify_key(self, key):
    if key == qt_imports.QtCore.Qt.Key_Escape:
      self.cancel()
    elif key == qt_imports.QtCore.Qt.Key_Enter or key == qt_imports.QtCore.Qt.Key_Return:
      self.validate()
    else:
      return False

    return True

  def get_shift(self):
    return self.end_pos - self.start_pos + self.tot_shift

  def notify_mov(self, pos):
    if self.drag:
      self.end_pos = pg.Point(pos)
      #glog.debug(self.end_pos, type(self.end_pos), type(self.end_pos - self.start_pos))
      self.shadow_entry.update_shift(self.get_shift())

  def validate(self):
    self.plot.add_plot(PlotEntry(self.entry.data.shift(self.get_shift()), **self.entry.kwargs))
    self.plot.remove_plot(self.entry)
    self.entry = None
    # remove shadow one
    self.cancel()


class ActionManager:

  def __init__(self, plot):
    self.plot = plot
    self.pt_plot = None
    self.setup()
    self.drag_curve = DragCurve(plot)

  def setup(self):
    pass

  def notify_clicked_curve(self, plot_entry):
    if self.pt_plot is not None:
      self.plot.removeItem(self.pt_plot)
    cur_pos = self.plot.get_cursor_pos()
    glog.debug('Creating point at %s', cur_pos)
    self.pt_plot = self.plot.plot(x=[cur_pos.x()], y=[cur_pos.y()], symbolPen='w')

  def key_press(self, h):
    if self.drag_curve.is_active():
      self.drag_curve.notify_key(h.key())
      return True
    else:
      self.plot.regions.key_press(h)
    return False

  def mouse_press(self, h):
    glog.debug('mouse press')
    if self.drag_curve.is_active():
      if self.drag_curve.notify_press(h):
        return True
    elif self.drag_curve.maybe_activate(h):
      return True
    return False

  def mouse_release(self, h):
    glog.debug('mouse release')
    if self.drag_curve.is_active():
      if self.drag_curve.notify_release(h):
        return True
    return False

  def mouse_mov(self, h):
    #print('mouse mov', h.pos())
    if self.drag_curve.is_active():
      if self.drag_curve.drag:
        self.drag_curve.notify_mov(h.pos())
        return True
    return False


class OpaPlot(pg.PlotWidget):

  def __init__(self, plots=[], images=[], legend=0, *args, **kwargs):
    super().__init__(*args, viewBox=ViewBox2(), **kwargs)
    self.manager = None
    self.menu = PlotMenu(self)
    self.regions = RegionManager(self)
    self.sampler = Sampler(self)
    self.last_hovered = None
    self.names = set()
    self.dsp_tools = DspTools(self)
    self.sigRangeChanged.connect(self.sig_range_changed)
    self.viewbox = self.getPlotItem().getViewBox()
    self.legend = legend

    if legend: self.addLegend()

    self.getPlotItem().getViewBox().menu = self.menu.menu
    self.setup_menu()
    self.opa_plots = QContainer()
    self.color_pool = ColorPool()
    self.cpool2 = cmisc.InfGenerator(get_colormap('viridis'))

    for plot in to_list(plots):
      self.add_plot(plot)

    for image in to_list(images):
      self.add_image(image)

    self.used_color = None

    self.action_manager = ActionManager(self)


  def installEventFilter(self, ev):
    #super().installEventFilter(ev)
    qt_imports.QtWidgets.QGraphicsView.installEventFilter(self, ev)
    self.getPlotItem().installEventFilter(ev)
    self.getPlotItem().getViewBox().installEventFilter(ev)
    self.getPlotItem().getViewBox()
    self.viewport().installEventFilter(ev)

  def show_grid(self):
    self.getPlotItem().showGrid(x=True, y=True)

  def get_color(self):
    if 0:
      self.used_color = self.color_pool.get_rgb()
    else:
      self.used_color = self.cpool2().rgb[0] * 255
    return self.used_color

  def get_view_range(self):
    return Range2D(self.getPlotItem().getViewBox().viewRange())

  def get_free_name(self, base_name):
    count = 0
    name = base_name
    while name in self.names:
      name = '%s_%d' % (base_name, count)
      count += 1
    self.names.add(name)
    return name

  def add_plot(self, plot_entry, **kwargs):
    if isinstance(plot_entry, (tuple, list, np.ndarray)):
      plot_entry = Dataset(plot_entry)

    if isinstance(plot_entry, Dataset):
      plot_entry = PlotEntry(plot_entry, **kwargs)
    glog.info('Adding plot with name=%s', plot_entry.data.name)

    if plot_entry.data.name in self.names:
      plot_entry.data.name = self.get_free_name(plot_entry.data.name)

    if not 'pen' in plot_entry.kwargs:
      plot_entry.kwargs['pen'] = {}
    pen = plot_entry.kwargs['pen']
    if pen is not None and 'color' not in pen:
      if 'color' in plot_entry.kwargs:
        color = Color(plot_entry.kwargs['color']).RGB
      else:
        color = self.get_color()
      pen['color'] = color

    self.opa_plots.append(plot_entry)
    plot_entry.register(self)
    return plot_entry

  def add_image(self, image_entry):
    if isinstance(image_entry, np.ndarray):
      image_entry = Dataset2d(image_entry)
    if isinstance(image_entry, Dataset2d):
      image_entry = ImageEntry(image_entry)
    glog.info('Adding image ')

    image_entry.register(self)
    return image_entry

  def save_figure(self):
    path, ok = qt_imports.QtWidgets.QInputDialog.getText(self, 'Path', 'path')
    if not ok:
      return
    cols = []
    data = {}
    for plot in self.opa_plots:
      assert not plot.data.name in data, 'already have name %s in %s' % (
          plot.data.name, data.keys()
      )
      data[plot.data.name] = plot.data.y
    df = pd.DataFrame(data, index=self.opa_plots[0].data.x)
    df.to_csv(path, index=True, index_label='x')

  def remove_plot(self, plot_entry):
    if 0:
      self.color_pool.release(self.used_color)
    assert plot_entry in self.opa_plots
    plot_entry.unregister()
    self.opa_plots.remove(plot_entry)

  def getContextMenus(self, event):
    return self.menu.menu

  def setup_menu(self):
    file_menu = self.menu.add_menu('file')
    file_menu.addAction('save', self.save_figure)

    transform_menu = self.menu.add_menu('transform')
    transform_menu.addAction('filter', self.transform_filter)
    transform_menu.addAction('fft', self.transform_fft)

    self.sampler.setup_menu(self.menu)
    self.dsp_tools.setup_menu(self.menu)

  def transform_filter(self):
    glog.debug('transfomrfilter')

  def transform_fft(self):
    glog.debug('transfomrft')

  def to_view_coord(self, pos):
    return self.getPlotItem().getViewBox().mapSceneToView(pg.Point(pos))

  def get_cursor_pos(self):
    pos = qt_imports.QtGui.QCursor.pos()
    pos = self.mapFromGlobal(pos)
    pos = self.to_view_coord(pos)
    return pos

  def keyPressEvent(self, ev):
    self.update_all()
    pos = self.get_cursor_pos()
    glog.debug('got key press event %s', ev.key())
    helper = EventHelper(ev, pos=pos)
    if self.action_manager.key_press(helper):
      ev.accept()
    else:
      super().keyPressEvent(ev)

  def mouseDragEvent(self, ev):
    glog.debug('drag >> %s', ev.isAccepted())
    self.region.notify_mouse_drag(ev)
    super().mouseDragEvent(ev)
    if ev.isAccepted():
      return
    glog.debug('drag la')

  def mousePressEvent(self, ev):
    glog.debug('START MOUSE PRESS %s', ev.isAccepted())

    self.update_all()
    helper = EventHelper(ev, pos=self.get_cursor_pos())
    self.regions.notify_press()
    if self.action_manager.mouse_press(helper):
      return

    glog.debug('PASSING MOUSE PRESS %s', ev.isAccepted())
    super().mousePressEvent(ev)
    if ev.isAccepted():
      return

    self.regions.mouse_press(ev)

  def mouseReleaseEvent(self, ev):
    helper = EventHelper(ev, pos=self.get_cursor_pos())
    if self.action_manager.mouse_release(helper):
      return
    super().mouseReleaseEvent(ev)

  def mouseMoveEvent(self, ev):
    helper = EventHelper(ev, pos=self.get_cursor_pos())
    if self.action_manager.mouse_mov(helper):
      return
    super().mouseMoveEvent(ev)

  def update_all(self):
    self.last_hovered = self.regions.find_hovering()

  def sig_range_changed(self, _, new_view_range):
    glog.debug('RNAGE CHANGED >> %s', new_view_range)

  def notify_remove_region(self, region):
    self.dsp_tools.notify_remove_region(region)


class ViewBox2(pg.ViewBox):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def mouseDragEvent(self, ev, axis=None):
    super().mouseDragEvent(ev, axis)


class PlotManager:

  def __init__(self, parent):
    self.parent = parent
    self.plots = []

  def create_plot(self, cl, *args, **kwargs):
    plot = cl(*args, parent=self.parent.w, **kwargs)
    self.plots.append(plot)
    plot.manager = self
    self.parent.add(plot)
    return plot

  def remove_plot(self, plot):
    self.plots.remove(plot)
    self.parent.remove(plot)


class MainWindow(qt_imports.QtWidgets.QMainWindow):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setWindowTitle('dsp display')
    if g_env.qt5:
      from chdrft.display.test_ui import Ui_MainWindow
    else:
      from chdrft.display.test_ui4 import Ui_MainWindow
    self.ui = Ui_MainWindow()
    self.ui.setupUi(self)

  #def closeEvent(self, event):
  #  glog.info('Main window close event')
  #  event.accept()


class OpaMainWindow(MainWindow):

  def __init__(self, close_cb):
    super().__init__()
    self.close_cb = close_cb

  def closeEvent(self, ev):
    self.close_cb()
    super().closeEvent(ev)


class GraphHelper:

  def __init__(self, create_kernel=False, run_in_jupyter=0):

    self.run_in_jupyter = run_in_jupyter
    self.app = qt_imports.QtCore.QCoreApplication.instance()
    if self.app is None:
      self.app = qt_imports.QApplication([])
    self.app.setQuitOnLastWindowClosed(True)
    self.app.aboutToQuit.connect(self.about_to_quit)
    #QtCore.QObject.connect(self.app, Qt.SIGNAL("lastWindowClosed()"),
    #                  self.app, Qt.SLOT("quit()"))

    self.ipkernel = None
    if create_kernel:
      self.ipkernel = IPKernelApp.instance()
      self.ipkernel.initialize(['python', '--matplotlib=qt'])
      glog.info('stuff >> %s', self.ipkernel.connection_file)
    self.cleanups_cb = []

    self.win = OpaMainWindow(self.close_main_window)
    self.win.show()

    #ui.graphicsView.useOpenGL()  ## buggy, but you can try it if you need extra speed.

    #self.vb = ViewBox2()
    #self.vb.setAspectLocked()
    layout = self.win.ui.verticalLayout
    self.manager = PlotManager(
        cmisc.A(
            w=self.win,
            add=layout.addWidget,
            remove=layout.removeWidget,
        )
    )

  def close_main_window(self):
    for cb in self.cleanups_cb:
      cb()

  def register_cleanup_cb(self, cb):
    self.cleanups_cb.append(cb)

  def about_to_quit(self):
    #print('about to quit kappa')
    pass

  def create_plot(self, *args, **kwargs):
    plot = self.manager.create_plot(OpaPlot, *args, **kwargs)
    return plot

  def remove_plot(self, plot):
    self.manager.remove_plot(plot)

  def run(self):

    if self.run_in_jupyter:
      print('LAAAAA')
      from IPython.lib.guisupport import start_event_loop_qt4
      start_event_loop_qt4(self.app)
      return

    #if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    if not is_interactive():
      if self.ipkernel is not None:
        self.ipkernel.user_module, self.ipkernel.user_ns = extract_module_locals(1)
        self.ipkernel.shell.set_completer_frame()
        self.ipkernel.start()
      else:
        qt_imports.QtGui.QApplication.instance().exec_()


def test1(ctx):
  g = GraphHelper()
  p1 = Dataset(y=-np.array(range(10)))
  g.create_plot(plots=[p1])
  g.run()


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
