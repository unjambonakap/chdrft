#!/usr/bin/env python

from chdrft.config.env import g_env
g_env.set_qt5(1)
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import calcephpy
from astroquery.jplhorizons import Horizons
from chdrft.display.vtk import vtk_main_obj
import chdrft.display.vtk as opa_vtk
from chdrft.sim.utils import *
from chdrft.sim.base import *
import numpy as np

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  pass

def ok2019_flyby(ctx):
  start = '2019-07-24'
  end = '2019-07-28'
  bodies = [calcephpy.NaifId.MOON, '2019 OK']
  starts = {}
  starts[calcephpy.NaifId.MOON] = '2019-07-01'
  body_center = calcephpy.NaifId.EARTH
  step = '30m'
  steps = {}
  steps[calcephpy.NaifId.MOON] = '5h'
  objs = []
  for body in bodies:
    objs.append(Horizons(id=body,
                  location=f'@{body_center}',
                  epochs={'start':starts.get(body, start),
                          'stop':end,
                          'step':steps.get(body, step)},
                  id_type='majorbody'))
  main = vtk_main_obj()

  for obj in objs:
    eph = obj.ephemerides()
    print(eph)
    pts = eph_to_xyz(eph, U.km)
    dpts = np.diff(pts, axis=0)
    speed = np.linalg.norm(dpts, axis=1)
    dists = np.linalg.norm(pts, axis=1)
    imin=  np.argmin(dists)
    
    print(obj, eph['datetime_str'][imin], speed[imin], dists[imin-1:imin+2])
    print(min(speed), max(speed))
    actor = opa_vtk.create_line_actor(pts)
    main.ren.AddActor(actor)

  tg = TileGetter()
  u = SimpleVisitor(tg, 4)
  u.run_tms()
  earth_assembly = opa_vtk.vtk.vtkAssembly()
  for actor in u.actors:
    earth_assembly.AddPart(actor)
    actor.GetProperty().SetAmbient(1)
    actor.GetProperty().SetDiffuse(0)
  earth_pos = (0,0,0)
  earth_assembly.SetPosition(*earth_pos)
    
  main.ren.AddActor(earth_assembly)
  main.run()

def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
