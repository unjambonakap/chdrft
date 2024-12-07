#!/usr/bin/env python

import chdrft.utils.misc as cmisc
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
import numpy as np
from enum import Enum
import chdrft.display.grid as grid
from chdrft.dsp.datafile import Dataset
from chdrft.display.ui import PlotEntry, OpaPlot, MetricWidget
import chdrft.display.video_helper as vh
import reactivex.operators as ops
from chdrft.config.env import g_env
import typing
import chdrft.display.plot_req as oplot_req

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class PlotTypeDesc(cmisc.PatchedModel):
  create_widget: type
  type: grid.PlotTypesEnum
  add_to_widget: typing.Callable


descs = [
    PlotTypeDesc(
        type=grid.PlotTypesEnum.Graph, create_widget=OpaPlot, add_to_widget=OpaPlot.add_plot
    ),
    PlotTypeDesc(
        type=grid.PlotTypesEnum.Metric,
        create_widget=grid.MetricStoreWidget,
        add_to_widget=grid.MetricStoreWidget.add
    ),
    PlotTypeDesc(
        type=grid.PlotTypesEnum.Vispy,
        create_widget=vh.GraphWidget,
        add_to_widget=lambda w, x: w.vctx.plot_meshes(x)
    ),
    PlotTypeDesc(
        type=grid.PlotTypesEnum.Fig, create_widget=vh.FigWidget, add_to_widget=lambda w, x: None
    ),
]
g_plot_type2desc = {x.type: x for x in descs}

# updating oplt >>
#gw = oplt.windows[0].gw.gp.zorder[0]['w']
#update_axis_visual(gw.vctx)
#gw.vctx.view.camera = ArcballCamera()
#gw.vctx.update_axis_visual = lambda: update_axis_visual(gw.vctx)


def guess_typ(obj, typ):
  if typ is not None:
    if isinstance(typ, str):
      for e in grid.PlotTypesEnum:
        if typ == e.value: return e
      else:
        assert 0, typ

    return typ

  if isinstance(obj, (Dataset, np.ndarray, PlotEntry)): return grid.PlotTypesEnum.Graph
  if isinstance(obj, (vh.QWidget, MetricWidget)): return grid.PlotTypesEnum.Metric
  return grid.PlotTypesEnum.Vispy


class PlotService:

  def __init__(self):
    self.windows = []

  def setup(self):
    g_env.create_app()

  def create_window(self, **kwargs) -> grid.GridWidgetHelper:
    g_env.create_app()
    gwh = grid.GridWidgetHelper(**kwargs)
    gwh.win.closed.pipe(ops.filter(lambda x: x == True)
                       ).subscribe(lambda _: self.windows.remove(gwh))
    self.windows.append(gwh)
    return gwh

  def find_window(self) -> grid.GridWidgetHelper | None:
    if not self.windows: return None
    return self.windows[-1]

  def plot(self, req: oplot_req.PlotRequest, o=1) -> oplot_req.PlotResult | None:
    gwh = None
    if not req.new_window: gwh = self.find_window()
    if gwh is None: gwh = self.create_window()

    typ = guess_typ(req.obj, req.type)
    entry: GridEntry = gwh.gw.gp.find(typ, req.label, req.attach_to_existing)

    attach_w = None if entry is None else entry.w

    res = None
    typ_desc = g_plot_type2desc[typ]
    if attach_w is None:
      attach_w = typ_desc.create_widget()
    add_res = typ_desc.add_to_widget(attach_w, req.obj)

    if entry is None:
      entry = grid.GridEntry(w=attach_w, label=req.label, type=typ)
      gwh.attach(entry)

    res = oplot_req.PlotResult(ge=entry, req=req, input_obj=req.obj, ge_item=add_res, type=typ)
    if o: return res


g_plot_service: PlotService = PlotService()
oplt = g_plot_service


def test(ctx):
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
