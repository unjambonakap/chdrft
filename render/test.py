#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import numpy as np
import chdrft.display.vtk as opa_vtk
import chdrft.struct.base as opa_struct
import cv2
import chdrft.utils.geo as geo_utils
import chdrft.sim.base as sim_base
from chdrft.utils.math import *

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--outfile')
  parser.add_argument('--infile')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def create_grid_image(ctx):
  w, h = (700, 500)
  dd = (10, 30)
  img = np.zeros((w, h))
  for x in np.arange(0, w, dd[0]):
    img[x, :] = 1
  for y in np.arange(0, h, dd[1]):
    img[:, y] = 1
  cv2.imwrite(ctx.outfile, img.T * 255)


def test_skew(ctx):
  main_obj = opa_vtk.vtk_offscreen_obj()

  actor = opa_vtk.create_image_actor_base(
      cmisc.Attr(
          rect=opa_struct.Box(low=(0, 0), high=(1, 1)),
          fname=ctx.infile,
          pos=(0, 0, 0),
      )
  )

  cam_pos = np.array((0, 0, 1))
  focal_point = np.array((0, 0, 0))
  #actor.RotateWXYZ(45, 1,1,0)
  main_obj.ren.AddActor(actor)
  main_obj.cam.SetPosition(*cam_pos)
  main_obj.cam.SetFocalPoint(*focal_point)

  coords = Z.itertools.product((-0.5, 0.5), repeat=2)
  plist = []
  for x, y in coords:
    plist.append(opa_vtk.vtk_mulnorm(actor.GetMatrix(), (x, y, 0)))
  plist = np.array(plist)

  #print(p0, px, py)
  #plane = geo_utils.compute_plane(p0, px, py)
  #inter = opa_vtk.compute_cam_intersection(main_obj.cam, plane, main_obj.aspect)
  #plist = np.array(inter)
  #box= opa_struct.Box.FromPoints(plist[:, :2])

  up = np.array((0, 1, 0))
  params = sim_base.compute_cam_parameters(cam_pos, focal_point, up, plist, aspect=main_obj.aspect)
  print(params)
  #main.cam.SetFocalPoint(*moon_pos)
  angle_y = rad2deg(params.box.height)
  print('ANGLE > > ', angle_y, params.box.height)

  main_obj.cam.SetClippingRange(1, 1e20)
  main_obj.cam.SetViewAngle(angle_y)
  main_obj.cam.SetViewUp(*params.y)

  m = np.identity(4)
  m[0,1] = 1
  mx = sim_base.numpy_to_vtk_mat(m)
  tsf = opa_vtk.vtk.vtkMatrixToHomogeneousTransform()
  tsf.SetInput(mx)
  main_obj.cam.SetUserTransform(tsf)

  main_obj.render(ctx.outfile)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
