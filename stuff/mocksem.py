#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import vtk
import ctypes
import cv2
import numpy as np
import json
import mocksem
import math
from vtk.util.numpy_support import numpy_to_vtk
np.set_printoptions(precision=5)

g = cmisc.Attr()
g.sem_width = 712 # scan_format
g.sem_height = 484
g.sem_width *= 2
g.sem_height *= 2
#g.sem_width = 128 * 1
#g.sem_height = 128 * 1
g.mock_img_file = '/tmp/mock.png'
g.tmp_img_render = '/tmp/mocksem.img.render.png'
g.tmp_img_warp_render = '/tmp/mocksem.img.warp.render.png'

global flags, cache
flags = None
cache = None


def rad2deg(x): return 180*x/np.pi
def deg2rad(x): return x*np.pi/180

def to_grayscale(img): return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def get_z_plane():
  plane = vtk.vtkPlaneSource() # size 1x1, normal z, centered on 0
  plane.SetCenter(0, 0, 0)
  plane.SetNormal(0, 0, -1)
  plane.Update()
  return plane


class MockSEM:
  def __init__(self, ctx):
    self.ctx = ctx
    self.mock_img_file = None
    if ctx.mock_img_base:
      self.base_image = to_grayscale(cv2.imread(ctx.mock_img_base))
      self.mock_img_file = g.mock_img_file
      cv2.imwrite(self.mock_img_file, self.base_image)

    self.chip_size = np.array(ctx.get('chip_size', (1e-3, 1e-3)))
    self.chip_dist = 20e-3
    pos = np.array((self.chip_size[0]/2,self.chip_size[1]/2, self.chip_dist), dtype=np.float64)
    self.view_angle_y = deg2rad(30)


    self.data = cmisc.Attr(pos=pos, rotation=0, flipx=0, flipy=0)
    z0 = self.compute_zoom_for_size_y(self.chip_size[1])
    self.data.mag = self.zoom_to_mag(z0)
    self.enable_warping=0
    if self.enable_warping: self.expand_factor = 1.1
    else: self.expand_factor = 1


  def zoom_to_mag(self, zoom):
    ai = math.tan(self.view_angle_y / 2)
    af = math.tan(self.view_angle_y / zoom /2)
    return ai / af

  def mag_to_zoom(self, mag):
    ai = math.tan(self.view_angle_y / 2)
    return self.view_angle_y / 2 / math.atan(ai / mag)

  def get_image_dim(self):
    return g.sem_width, g.sem_height

  def compute_zoom_for_size_y(self, size_y):
    dist = self.data.pos[2]
    return self.view_angle_y / 2 / Z.math.atan2(size_y / 2, dist)

  def func_get_image_tiff(self, *args):
    img = self.get_image(*self.get_image_dim())
    ok, buffer = cv2.imencode('.tif', img)
    assert ok
    return bytes(buffer)

  def get(self, f, args):
    res = self.do_get(f, args)
    if isinstance(res, np.ndarray): return list(res)
    return res

  def do_get(self, f, args):
    if args is None: args = []
    attrname = f'func_get_{f.endpoint}'
    if hasattr(self, attrname): return getattr(self, attrname)(*args)
    if f.endpoint=='pos_xy': return self.data.pos[:2]
    if f.endpoint =='mag': return self.data.mag
    assert 0, f

  def set(self, f, args):
    if f=='pos_xy': 
      self.data.pos[:2] = args
      print('SETTING POS ', args)
    elif f =='mag': 
      self.data.mag = args[0]
      print('setting mag ', args[0])
    elif f =='rotation': self.data.rotation = args[0]
    elif f =='flipx': self.data.flipx = args[0]
    elif f =='flipy': self.data.flipy = args[0]
    else: assert 0

  @property
  def aspect(self):
    dim = self.get_image_dim()
    return dim[0] / dim[1]

  def func_get_m_per_px(self):
    plist = self.get_chip_rect()
    dim = self.get_image_dim()
    dx = np.linalg.norm(plist[1] - plist[0]) / self.expand_factor / dim[0]
    dy = np.linalg.norm(plist[2] - plist[0]) / self.expand_factor / dim[1]
    return np.array([dx, dy])

  def get_chip_rect(self):
    def mulnorm(m, v):
      v = list(v)+[1]
      v = m.MultiplyPoint(v)
      return np.array(v[:3])/v[-1]

    x = self.get_scene(*self.get_image_dim())
    m = x.cam.GetCompositeProjectionTransformMatrix(self.aspect, -1,1 )
    m.Invert()

    plane =get_z_plane()

    plist = []
    p = vtk.vtkPlane()
    p.SetOrigin(plane.GetOrigin())
    p.SetNormal(plane.GetNormal())
    t = vtk.mutable(0.0)
    interp = [0, 0, 0]

    for py, px in Z.itertools.product((-1, 1), repeat=2):
      p1 = mulnorm(m, [px, py, 1])
      p2 = mulnorm(m, [px, py, -1])

      p.IntersectWithLine(p1, p2, t, interp)
      plist.append(np.array(interp))
    return plist



  def create_image_actor(self):
    pngReader = vtk.vtkPNGReader()
    if self.mock_img_file:
      pngReader.SetFileName(self.mock_img_file)
      pngReader.Update()

    cur = pngReader
    cur = vtk_do_flip(cur, self.data.flipx, self.data.flipy)

    atext = vtk.vtkTexture()
    atext.SetInputConnection(cur.GetOutputPort())
    atext.InterpolateOn()

    plane = get_z_plane()
    planeMapper = vtk.vtkPolyDataMapper()
    planeMapper.SetInputConnection(plane.GetOutputPort())
    planeActor = vtk.vtkActor()
    planeActor.RotateZ(90 * self.data.rotation)
    planeActor.SetMapper(planeMapper)
    planeActor.SetTexture(atext)
    planeActor.SetScale(self.chip_size[0], self.chip_size[1], 1)
    planeActor.SetPosition(self.chip_size[0]/2, self.chip_size[1]/2, 0)
    return planeActor

  def create_renderer(self, width, height, *actors):

    # Create the RenderWindow, Renderer and both Actors
    renWin = vtk.vtkRenderWindow()
    renWin.SetOffScreenRendering(1)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)

    # Add the actors to the renderer, set the background and size
    ren.SetBackground(0.1, 0.2, 0.4)
    renWin.SetSize(width, height)
    print(renWin.GetMultiSamples(), ren.GetUseFXAA())
    #ren.SetUseFXAA(True)
    renWin.SetMultiSamples(16) # not working !
    renWin.SetPointSmoothing(True)
    renWin.SetLineSmoothing(True)
    for actor in actors:
      ren.AddActor(actor)

    ren.ResetCamera()
    cam = ren.GetActiveCamera()

    if self.data.flipx: self.data.pos[0] *= -1
    if self.data.flipy: self.data.pos[1] *= -1

    cam.SetPosition(self.data.pos)
    cam.SetFocalPoint(self.data.pos[0], self.data.pos[1], 0)
    cam.SetViewAngle(rad2deg(self.view_angle_y))
    cam.SetViewUp(0, 1, 0)
    cam.Zoom(self.mag_to_zoom(self.data.mag))
    ren.ResetCameraClippingRange()
    return cmisc.Attr(ren=ren, cam=cam, renWin=renWin, actors=actors)

  def render_to_file(self, renWin, filename=None):
    renWin.Render()
    # screenshot code:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.ReadFrontBufferOff()
    w2if.SetInput(renWin)
    w2if.Update()

    if filename is not None:
      writer = vtk.vtkPNGWriter()
      writer.SetFileName(filename)
      writer.SetInputData(w2if.GetOutput())
      writer.Write()
    return w2if

  def get_scene(self, width, height):

    img_actor  = self.create_image_actor()
    if self.enable_warping:
      mwidth,mheight = int(self.expand_factor*width), int(self.expand_factor*height)
    else:
      mwidth, mheight = width, height

    x = self.create_renderer(mwidth, mheight, img_actor)
    x.mwidth = mwidth
    x.mheight = mheight
    return x

  def get_image(self, width, height):
    dx = self.get_image_raw(width, height)
    if self.enable_warping: img = dx.image
    else: img = dx.image_nowrap
    return to_grayscale(img)

  def get_image_raw(self, width, height):
    x = self.get_scene(width, height)


    w2if = self.render_to_file(x.renWin, g.tmp_img_render)
    self.do_warping(w2if, width, height, x.mwidth, x.mheight, g.tmp_img_warp_render)
    image = cv2.imread(g.tmp_img_warp_render)[::-1,:]
    image_nowrap = cv2.imread(g.tmp_img_render)[::-1,:]
    return cmisc.Attr(image=image, image_nowrap=image_nowrap)

  def do_warping(self, inx, width, height, mwidth, mheight, warped_image_filename):
    wl = vtk.vtkWarpLens()
    wl.SetInputConnection(inx.GetOutputPort())
    wl.SetPrincipalPoint(0, 0)
    wl.SetFormatWidth(4.792)
    wl.SetFormatHeight(3.6)
    wl.SetImageWidth(width)
    wl.SetImageHeight(height)

    if 1:
      wl.SetK1(0)
      wl.SetK2(0)
      wl.SetP1(0)
      wl.SetP2(0)
    else:
      wl.SetK1(0.01307e-5)
      wl.SetK2(0.0003102e-5)
      wl.SetP1(1.953e-008)
      wl.SetP2(-9.655e-008)
    gf = vtk.vtkGeometryFilter()

    gf.SetInputConnection(wl.GetOutputPort())

    tf = vtk.vtkTriangleFilter()
    tf.SetInputConnection(gf.GetOutputPort())

    strip = vtk.vtkStripper()
    strip.SetInputConnection(tf.GetOutputPort())
    strip.SetMaximumLength(250)


    dsm = vtk.vtkPolyDataMapper()
    dsm.SetInputConnection(strip.GetOutputPort())
    dsm.Update()

    acx = vtk.vtkActor()
    acx.SetMapper(dsm)
    #acx.SetPosition(-1-1-0.5+height/width, -1, 0)
    #acx.SetScale(2/height, 2/height, 1)
    acx.SetPosition(0, 0, 0)
    acx.SetScale(1)

    x = self.create_renderer(width, height,acx)

    x.cam.ParallelProjectionOn()
    x.cam.SetPosition(mwidth/2, mheight/2, 1)
    x.cam.SetFocalPoint(mwidth/2, mheight/2, 0)
    x.cam.SetViewUp(0, 1, 0)
    x.cam.SetParallelScale(height/2)
    x.ren.ResetCameraClippingRange()


    self.render_to_file(x.renWin, warped_image_filename)

def test(ctx):
  #ctx.chip_size=(1e-3, 1e-3)
  m = MockSEM(ctx)
  #m.data.mag *= 1.5
  m.data.mag *= 3
  m_per_px = m.func_get_m_per_px()
  print(m_per_px[0], m_per_px[1])
  print(m_per_px * m.get_image_dim())
  return
  img = m.get_image_raw(*m.get_image_dim()).image
  cv2.imwrite('./mock.img.png', img)
  #m.data.pos[0] += m.chip_size[0] / m.data.mag/ 30
  #m.data.pos[0] += m_per_px[0]
  m.data.mag *= 2
  #m.data.flipx = 1

  img = m.get_image_raw(*m.get_image_dim()).image
  cv2.imwrite('./mock.img2.png', img)


def create_image_actor_base(idata, rot_deg=0):
  image_map = idata.rect

  atext = vtk.vtkTexture()
  if 1:
    cur = create_vtk_reader(idata.fname)
    cur = vtk_do_flip(cur, 0, 0)
    atext.SetInputConnection(cur.GetOutputPort())


  atext.InterpolateOn()

  plane = get_z_plane()
  planeMapper = vtk.vtkPolyDataMapper()
  planeMapper.SetInputConnection(plane.GetOutputPort())
  planeActor = vtk.vtkActor()
  planeActor.RotateZ(rot_deg)
  planeActor.SetMapper(planeMapper)
  planeActor.SetTexture(atext)
  planeActor.SetScale(image_map.xr.length, image_map.yr.length, 1)
  planeActor.GetProperty().SetOpacity(app.flags.opacity)
  center=  image_map.center
  planeActor.SetPosition(center[0], center[1], 0)
  return planeActor, image_map


def create_image_composition(images_data, image_dim, out_fname):

  actors = []
  rects = []
  for idata in images_data:
    rect = idata.rect
    actor, nrect = create_image_actor_base(idata)
    actors.append(actor)
    rects.append(nrect)
  viewbox = Z.Range2D.Union(rects).box



  renWin = vtk.vtkRenderWindow()
  renWin.SetOffScreenRendering(1)
  ren = vtk.vtkRenderer()
  renWin.AddRenderer(ren)

  # Add the actors to the renderer, set the background and size
  ren.SetBackground(0.1, 0.2, 0.4)
  renWin.SetSize(image_dim[0], image_dim[1])
  #ren.SetUseFXAA(True)
  #renWin.SetMultiSamples(8) # not working !
  #renWin.SetPointSmoothing(True)
  #renWin.SetLineSmoothing(True)

  for actor in actors:
    ren.AddActor(actor)
    print(actor)

  ren.ResetCamera()
  cam = ren.GetActiveCamera()

  #if self.data.flipx: self.data.pos[0] *= -1
  #if self.data.flipy: self.data.pos[1] *= -1


  viewbox = viewbox.force_aspect(image_dim[0] / image_dim[1])

  viewbox_center = viewbox.center
  cam.ParallelProjectionOn()
  cam.SetPosition(viewbox_center[0], viewbox_center[1], viewbox.width)
  cam.SetFocalPoint(viewbox_center[0], viewbox_center[1], 0)
  cam.SetViewUp(0, 1, 0)
  cam.SetParallelScale(viewbox.height/2)
  ren.ResetCameraClippingRange()

  renWin.Render()
  # screenshot code:
  w2if = vtk.vtkWindowToImageFilter()
  w2if.ReadFrontBufferOff()
  w2if.SetInput(renWin)
  w2if.Update()

  writer =create_vtk_writer(out_fname)
  writer.SetInputData(w2if.GetOutput())
  writer.Write()
  return viewbox


def vtk_do_flip(cur, flipx, flipy):
  for i, f in enumerate((flipx, flipy)):
    if f:
      fx = vtk.vtkImageFlip()
      fx.SetFilteredAxis(i)
      fx.SetInputConnection(cur.GetOutputPort())
      fx.Update()
      cur = fx
  return cur

def create_vtk_reader(fname):
  if fname.endswith('.png'):
    reader = vtk.vtkPNGReader()
  else:
    reader = vtk.vtkTIFFReader()

  reader.SetFileName(fname)
  reader.Update()
  return reader

def create_vtk_writer(fname):
  writer = vtk.vtkPNGWriter()
  writer.SetFileName(fname)
  return writer

def mocksem_args(parser):
  parser.add_argument('--mock-img-base')
  parser.add_argument('--opacity', type=float, default=1.)

def args(parser):
  clist = CmdsList().add(test)
  mocksem_args(parser)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)

def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
