#!/usr/bin/env python

import chdrft.utils.K as K
import numpy as np
from chdrft.sim.base import create_earth_actors, Box
from chdrft.display.vtk import TriangleActorVTK
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
from chdrft.utils.misc import Attributize as A
import chdrft.utils.Z as Z
import numpy as np
from chdrft.geo.satsim import gl

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  p = gl.geocode('Paris')
  ll = Z.deg2rad(np.array([p.longitude, p.latitude]))
  md = 10
  u = create_earth_actors(
      TriangleActorVTK,
      max_depth=md,
      tile_depth=md,
      m2u=1e-3,
      ll_box=Box(center=ll, size=(np.pi / 100, np.pi / 100))
  )
  K.oplt.plot(A(images=[K.ImageData(img=x.data[::-1], box=x.box_latlng) for x in u.items]))
  input()


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
