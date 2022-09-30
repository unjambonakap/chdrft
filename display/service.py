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
import chdrft.display.grid as grid
from chdrft.dsp.datafile import Dataset
from chdrft.display.ui import PlotEntry, OpaPlot
from chdrft.dsp.image import ImageData
import chdrft.display.video_helper as vh
from rx import operators as ops
from chdrft.config.env import g_env
from vispy.plot import Fig

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class PlotTypes(Enum):
  Graph = 'graph'
  Vispy = 'vispy'
  Fig = 'fig'
  Metric = 'metric'


# updating oplt >>
#gw = oplt.windows[0].gw.gp.zorder[0]['w']
#update_axis_visual(gw.vctx)
#gw.vctx.view.camera = ArcballCamera()
#gw.vctx.update_axis_visual = lambda: update_axis_visual(gw.vctx)


class PlotService:

  def __init__(self):
    self.windows = []

  def create_window(self, **kwargs):
    grid.create_app()
    gwh = grid.GridWidgetHelper(**kwargs)
    gwh.win.closed.pipe(ops.filter(lambda x: x == True)
                       ).subscribe(lambda _: self.windows.remove(gwh))
    self.windows.append(gwh)
    return gwh

  def find_window(self):
    if not self.windows: return None
    return self.windows[-1]

  def guess_typ(self, obj, typ):
    if typ is not None:
      if isinstance(typ, str):
        for e in PlotTypes:
          if typ == e.value: return e
        else:
          assert 0, typ

      return typ

    if isinstance(obj, (Dataset, np.ndarray, PlotEntry)): return PlotTypes.Graph
    return PlotTypes.Vispy

  def plot(self, obj, typ=None, new_window=0, label=None, o=0, gwh=None, **kwargs):
    if not new_window: gwh = self.find_window()
    if not gwh: gwh = self.create_window()
    typ = self.guess_typ(obj, typ)
    entry = gwh.gw.gp.find_by_label(label)
    if entry is not None: entry = entry.w

    res = None
    if typ == PlotTypes.Vispy: res = self.plot_vispy(obj, entry=entry, **kwargs)
    elif typ == PlotTypes.Metric: res = self.plot_metric(obj, entry=entry, **kwargs)
    elif typ == PlotTypes.Fig: res = self.plot_fig(obj, entry=entry, **kwargs)
    else: res = self.plot_graph(obj, entry=entry, **kwargs)

    if entry is None: gwh.gw.add(res.w, label=label, **kwargs)
    if o: return res

  def plot_vispy(self, obj, entry=None, update_vb=1, **kwargs):
    if entry is None: entry = vh.GraphWidget()
    data = entry.vctx.plot_meshes(obj, **kwargs)
    if update_vb: entry.vctx.set_viewbox(vb_hint=data.vb_hint)
    return A(w=entry, data=data)

  def plot_fig(self, obj, entry=None, **kwargs):
    if entry is None: entry = vh.FigWidget()
    return A(w=entry)

  def plot_graph(self, obj, entry=None, gwh=None, **kwargs):
    if entry is None: entry = OpaPlot(**kwargs)
    data = entry.add_plot(obj, **kwargs)
    return A(w=entry, data=data)

  def plot_metric(self, obj, entry=None, gwh=None, **kwargs):
    if entry is None: entry = grid.MetricStoreWidget()
    data = entry.add(vh.MetricWidget.Make(obj))
    return A(w=entry, data=data)


g_plot_service: PlotService = PlotService()
oplt = g_plot_service


def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
