#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.types import *
from chdrft.sim.moon_sunrise import *
from chdrft.display.vtk import *
from chdrft.sim.base import Renderer

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  moon_sunrise_params(parser)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class MoonSunriseVTK(MoonSunrise):

  def __init__(self, ctx):
    super().__init__(ctx)

  def build(self):

    if self.ctx.offscreen:
      main = opa_vtk.vtk_offscreen_obj(width=self.ctx.width, height=self.ctx.height)
    else:
      main = opa_vtk.vtk_main_obj(width=self.ctx.width, height=self.ctx.height)
    self.ren = main
    super().build()
    self.configure_render(self.ctx.width, self.ctx.height)
    self.post_build()


  def update_display(self):
    self.ren.update()
    self.ren.ren_win.Render()

  def configure_earth_mat(self, **kwargs):
    for k,v in kwargs.items():
      find_node_io(self.earth_grp, k).default_value = v

  def post_build(self):
    earth = self.objs[ActorType.Earth]


  def set_mat(self, obj, mat):
    if obj.obj is None: return

    if obj.type ==  ActorType.Cam:
      a = opa_vtk.vtk.vtkMatrixToHomogeneousTransform()
      obj.obj.SetPosition(0,0,0)
      obj.obj.SetFocalPoint(0, 0, -1)
      obj.obj.SetViewUp(0,1,0)
      mat = np.linalg.inv(mat)
      a.SetInput(numpy_to_vtk_mat(mat))
      obj.obj.SetUserViewTransform(a)
      self.configure_cam()
    else:
      a = opa_vtk.vtk.vtkMatrixToLinearTransform()
      a.SetInput(numpy_to_vtk_mat(mat))
      obj.obj.SetUserTransform(a)


  @property
  def actor_class(self):
    return TriangleActorVTK

  def create_earth_actor_impl(self, u):
    name = 'earth'

    earth_assembly = opa_vtk.vtk.vtkAssembly()
    for x in u.actors:
      actor = x.obj
      earth_assembly.AddPart(actor)
      actor.GetProperty().SetAmbient(1)
      actor.GetProperty().SetDiffuse(0)
    self.ren.ren.AddActor(earth_assembly)
    return earth_assembly

  def create_moon_actor_impl(self, ta):
    moon_actor = ta.obj
    moon_actor.GetProperty().SetAmbient(1)
    moon_actor.GetProperty().SetDiffuse(0)
    self.ren.ren.AddActor(moon_actor)
    return moon_actor

  def create_camera_impl(self, internal):
    cam = self.ren.cam
    return cam
  
  def configure_cam_internal(self, cam):

    #if self.ctx.rot_angle is not None: params.y = rotate_vec(params.y, z, flags.rot_angle)
    #main.cam.SetPosition(*cam_pos)
    #main.cam.SetFocalPoint(*focal_point)
    self.ren.ren_win.SetSize(*cam.internal.res)
    cam.obj.SetViewAngle(cam.internal.view_angle)
    cam.obj.SetClippingRange(1, 1e20)

  def render(self, fname):
    return self.ren.render()


def test1():
  ctx = app.setup_jup(parser_funcs=[args])
  ctx.width = 2000
  ctx.height = 2000
  ctx.moon_model = None
  ctx.earth_depth = 4
  ctx.earth_tile_depth = 3
  ms = MoonSunriseBlender(ctx)
  ms.build()
  print('done build')
  ms.configure_at(ms.tgtime)
  ms.render('/tmp/render1.png')
  app.cache.flush_cache()


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
