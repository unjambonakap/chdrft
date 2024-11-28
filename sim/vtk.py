#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import numpy as np
from chdrft.utils.opa_types import *
from chdrft.sim.moon_sunrise import *
from chdrft.display.vtk import *

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  moon_sunrise_params(parser)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class VTKHelper:

  @staticmethod
  def set_mat(obj, mat, is_cam):
    if is_cam:
      a = opa_vtk.vtk.vtkMatrixToHomogeneousTransform()
      obj.SetPosition(0, 0, 0)
      obj.SetFocalPoint(0, 0, -1)
      obj.SetViewUp(0, 1, 0)
      mat = np.linalg.inv(mat)
      a.SetInput(numpy_to_vtk_mat(mat))
      obj.SetUserViewTransform(a)
    else:
      a = opa_vtk.vtk.vtkMatrixToLinearTransform()
      a.SetInput(numpy_to_vtk_mat(mat))
      obj.SetUserTransform(a)

  @staticmethod
  def SimpleVisitor2Assembly(u, ren):
    ass = opa_vtk.vtk.vtkAssembly()
    for x in u.actors:
      actor = x.obj
      ass.AddPart(actor)
      actor.GetProperty().SetAmbient(1)
      actor.GetProperty().SetDiffuse(0)
    ren.AddActor(ass)
    return ass


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
    for k, v in kwargs.items():
      find_node_io(self.earth_grp, k).default_value = v

  def post_build(self):
    earth = self.objs[ActorType.Earth]

  def set_mat(self, obj, mat):
    if obj.obj is None: return
    is_cam = obj.type == ActorType.Cam
    VTKHelper.set_mat(obj.obj, mat, is_cam)
    if is_cam:
      self.configure_cam()

  @property
  def actor_class(self):
    return TriangleActorVTK

  def create_earth_actor_impl(self, u):
    name = 'earth'

    ass = VTKHelper.SimpleVisitor2Assembly(u, self.ren.ren)
    return ass

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
    #px = qt_imports.QtGui.QPixmap(self.ctx.width, self.ctx.height)
    self.ren.render(fname)


def test1(ctx):
  ctx.width = 2000
  ctx.height = 2000
  ctx.moon_model = None
  ctx.moon_model = 'pic'
  ctx.earth_depth = 4
  ctx.earth_tile_depth = 3
  ms = MoonSunriseVTK(ctx)
  ms.build()
  print('done build')
  ms.configure_at(ms.tgtime)
  if ctx.offscreen:
    ms.render('/tmp/render1.png')
  else:
    ms.ren.run()


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
