#!/usr/bin/env python

from __future__ import annotations
import chdrft.utils.misc as cmisc
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
import numpy as np
from enum import Enum
import chdrft.display.grid as grid

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class PlotRequest(cmisc.PatchedModel):
  target_ge: grid.GridEntry = None
  label:str | None =None

  attach_to_existing: bool = True
  new_window: bool= False
  type: grid.PlotTypesEnum = None
  gwh: grid.GridWidgetHelper = None
  obj: object = None

  def plot(self, obj, o=1) -> PlotResult:
    from chdrft.display.service import g_plot_service
    return g_plot_service.plot(self.model_update(obj=obj), o=o)


class PlotResult(cmisc.PatchedModel):
  ge: grid.GridEntry
  req: PlotRequest
  type: grid.PlotTypesEnum
  input_obj: object
  ge_item: object | None


