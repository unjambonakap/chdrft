#!/usr/bin/env python

from chdrft.config.env import g_env, qt_imports
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.display.vispy_utils as vispy_utils
import re
import chdrft.struct.base as opa_struct
import cv2
import io
import vtk
import sys
import numpy as np
from IPython.lib import guisupport
from vispy.color import get_colormap, Color
from vtk.util import numpy_support, vtkImageImportFromArray, vtkImageExportToArray
import chdrft.utils.geo as geo_utils
from chdrft.display.base import TriangleActorBase

global flags, cache
flags = None
cache = None


def numpy_to_vtk_mat(mat):
  res = vtk.vtkMatrix4x4()
  res.DeepCopy(mat.flatten())
  return res


def vtk_matrix_to_numpy(mat):
  res = np.zeros((4,4))
  if isinstance(mat, vtk.vtkMatrix4x4):
    n =4
  else:
    n = 3
    res[3,3] = 1
  for i in range(n):
    for j in range(n):
      res[i,j] = mat.GetElement(i,j)

  return res

def vtk_mulnorm(m, v):
  v = list(v)+[1]
  v = m.MultiplyPoint(v)
  return np.array(v[:3])/v[-1]

def vtk_offscreen_obj(width=800, height=600):
  ren_win = vtk.vtkRenderWindow()
  ren_win.SetOffScreenRendering(1)
  ren = vtk.vtkRenderer()
  ren_win.AddRenderer(ren)

  # Add the actors to the renderer, set the background and size
  ren.SetBackground(0.1, 0.2, 0.4)
  ren_win.SetSize(width, height)
  print(ren_win.GetMultiSamples(), ren.GetUseFXAA())
  #ren.SetUseFXAA(True)
  ren_win.SetMultiSamples(16)  # not working !
  ren_win.SetPointSmoothing(True)
  ren_win.SetLineSmoothing(True)
  ren.ResetCamera()
  cam = ren.GetActiveCamera()
  res = cmisc.Attr(ren=ren, ren_win=ren_win, cam=cam, aspect=width/height,
                    render_box=opa_struct.Box(low=(0,0), dim=(width, height)))

  def render(outfile=None):
    ren_win.Render()
    w2if = vtk.vtkWindowToImageFilter()
    w2if.ReadFrontBufferOff()
    w2if.SetInput(ren_win)
    w2if.Update()


    if outfile is not None:
      writer = vtk.vtkPNGWriter()
      writer.SetFileName(outfile)
      writer.SetInputData(w2if.GetOutput())
      writer.Write()

    res = vtkImageExportToArray.vtkImageExportToArray()
    res.SetInputData(w2if.GetOutput())
    data = res.GetArray()
    if data.shape[0] == 1:
      #assert data.shape[1] == res.dim[1]
      data = data[0]

    return data
  res.render = render
  return res



def vtk_main_obj(*args, **kwargs):
  QWidget, QtGui, QtWidgets = qt_imports.QWidget, qt_imports.QtGui, qt_imports.QtWidgets

  class vtkMain(QWidget):

    def run(self, reset=1):
      # reset the camera and set anti-aliasing to 2x

      if reset:
        self.ren.ResetCamera()
        self.cam.SetFocalPoint(0, 0, 0)
      self.show()
      self.window.show()
      self.interactor.Initialize()

      # start the event loop
      if cmisc.is_interactive():
        from IPython.lib.guisupport import start_event_loop_qt4
        start_event_loop_qt4(self.app)
      else:
        self.app.exec_()

    @property
    def aspect(self):
      return self.width / self.height

    def __init__(self, width=800, height=600):
      self.width = width
      self.height = height
      self.render_box=opa_struct.Box(low=(0,0), dim=(width, height))

      #assert g_env.qt5 == 0
      self.app = guisupport.get_app_qt4()
      super().__init__()

      self.window = QtWidgets.QMainWindow()
      self.window.setWindowTitle('opa_vtk')
      self.window.setCentralWidget(self)
      self.window.setFixedSize(self.width, self.height)

      self.interaction_status = cmisc.Attr(rot=0, pan=0, zoom=0)
      self.layout = QtWidgets.QVBoxLayout(self)
      self.layout.setContentsMargins(0, 0, 0, 0)
      self.setLayout(self.layout)

      self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

      self.ren = vtk.vtkRenderer()
      self.ren.SetBackground(1, 1, 1)
      self.ren.SetViewport(0, 0, 1, 1)

      from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
      self.interactor = QVTKRenderWindowInteractor(self)
      self.layout.addWidget(self.interactor)

      self.cam = self.ren.GetActiveCamera()
      self.cam.SetClippingRange(0.1, 1000)

      self.init_interactor()

      self.Rotating = 0
      self.Panning = 0
      self.Zooming = 0

      self.picker = vtk.vtkPropPicker()

      self.outline = vtk.vtkOutlineSource()
      self.outline_mapper = vtk.vtkPolyDataMapper()

      self.outline_actor = vtk.vtkActor()
      self.outline_actor.PickableOff()
      self.outline_actor.DragableOff()
      self.outline_actor.SetMapper(self.outline_mapper)
      self.outline_actor.GetProperty().SetColor(vispy_utils.Color('r').rgb * 255)
      self.outline_actor.GetProperty().SetAmbient(1.0)
      self.outline_actor.GetProperty().SetDiffuse(0.0)
      self.outline_actor.VisibilityOff()
      self.ren.AddActor(self.outline_actor)
      self.outline_mapper.SetInputConnection(self.outline.GetOutputPort())
      self.outline_actor.SetMapper(self.outline_mapper)

      self.init_text_dbg()

    def init_text_dbg(self):

      self.text_mapper = vtk.vtkTextMapper()
      tprop = self.text_mapper.GetTextProperty()
      tprop.SetFontFamilyToArial()
      tprop.SetFontSize(10)
      tprop.BoldOn()
      tprop.ShadowOn()
      tprop.SetColor(1, 0, 0)
      self.text_actor = vtk.vtkActor2D()
      self.text_actor.VisibilityOff()
      self.text_actor.SetMapper(self.text_mapper)
      self.ren.AddActor2D(self.text_actor)

    def set_box(self, actor):
      if actor is None:
        self.outline_actor.VisibilityOff()
        return
      self.outline.SetBounds(actor.GetBounds())
      self.outline_actor.VisibilityOn()

    def init_interactor(self):
      #self.switch = vtk.vtkInteractorStyleSwitch()
      #self.switch.SetCurrentStyleToTrackballCamera()
      #self.interactor.SetInteractorStyle(self.switch)
      #self.interactor.GetRenderWindow().AddRenderer(self.ren)
      self.ren_win = self.interactor.GetRenderWindow()
      self.ren_win.AddRenderer(self.ren)
      self.ren_win.SetSize(self.width, self.height)
      self.interactor.SetInteractorStyle(None)

      self.interactor.AddObserver('LeftButtonPressEvent', self.ButtonEvent)
      self.interactor.AddObserver('LeftButtonReleaseEvent', self.ButtonEvent)
      self.interactor.AddObserver('MiddleButtonPressEvent', self.ButtonEvent)
      self.interactor.AddObserver('MiddleButtonReleaseEvent', self.ButtonEvent)
      self.interactor.AddObserver('RightButtonPressEvent', self.ButtonEvent)
      self.interactor.AddObserver('RightButtonReleaseEvent', self.ButtonEvent)
      self.interactor.AddObserver('MouseMoveEvent', self.MouseMove)
      self.interactor.AddObserver('KeyPressEvent', self.Keypress)
      self.interactor.AddObserver('ConfigureEvent', self.test_event)

    def test_event(self, obj, event):
      self.screen_size = np.array(self.ren.GetSize())
      self.text_actor.SetPosition((10, 10))

    def ButtonEvent(self, obj, event):
      m = re.match('(?P<button>\w+)Button(?P<action>Press|Release)Event', event)

      nstatus = m['action'] == 'Press'
      button = m['button']
      maps = dict(
          rot=cmisc.Attr(button='Left', mod=('ctrl', 0)),
          pan=[
              cmisc.Attr(button='Left', mod=('ctrl', 1)),
              cmisc.Attr(button='Middle'),
          ],
          zoom=cmisc.Attr(button='Right'),
      )

      def match(pattern, button):
        if cmisc.is_list(pattern):
          return any([match(x, button) for x in pattern])
        if button != pattern.button: return 0
        if 'mod' not in pattern: return 1
        mod, modv = pattern.mod

        mod_val = None
        if mod == 'ctrl': mod_val = self.interactor.GetControlKey()
        else: assert 0
        return mod_val == modv

      tgt = None
      for k, v in maps.items():
        if match(v, button):
          tgt = k
      self.interaction_status[tgt] = nstatus

    # General high-level logic
    def MouseMove(self, obj, event):
      lastXYpos = self.interactor.GetLastEventPosition()
      lastX = lastXYpos[0]
      lastY = lastXYpos[1]

      xypos = self.interactor.GetEventPosition()
      x = xypos[0]
      y = xypos[1]

      center = self.ren_win.GetSize()
      centerX = center[0] / 2.0
      centerY = center[1] / 2.0
      ren = self.ren

      if self.interaction_status.rot:
        self.Rotate(ren, ren.GetActiveCamera(), x, y, lastX, lastY, centerX, centerY)
      elif self.interaction_status.pan:
        self.Pan(ren, ren.GetActiveCamera(), x, y, lastX, lastY, centerX, centerY)
      elif self.interaction_status.zoom:
        self.Dolly(ren, ren.GetActiveCamera(), x, y, lastX, lastY, centerX, centerY)

    def Keypress(self, obj, event):
      key = obj.GetKeySym()
      pos = obj.GetEventPosition()

      if key == 'p':
        self.picker.Pick(pos[0], pos[1], 0, self.ren)
        actor = self.picker.GetActor()
        pick_pos = self.picker.GetPickPosition()
        #pick_normal =self.picker.GetPickNormal()
        self.text_mapper.SetInput(f'pick_pos={pick_pos}, actor={actor is not None}')
        self.text_actor.VisibilityOn()
        self.set_box(actor)
        self.ren_win.Render()

      elif key == 'c':
        self.picker.Pick(pos[0], pos[1], 0, self.ren)
        actor = self.picker.GetActor()
        if actor is not None:
          pick_pos = self.picker.GetPickPosition()
          self.cam.SetFocalPoint(*pick_pos)
          print('LAA ', pick_pos)
          self.set_box(actor)
          self.ren_win.Render()

    # Routines that translate the events into camera motions.

    # This one is associated with the left mouse button. It translates x
    # and y relative motions into camera azimuth and elevation commands.
    def Rotate(self, renderer, camera, x, y, lastX, lastY, centerX, centerY):
      camera.Azimuth(lastX - x)
      camera.Elevation(lastY - y)
      camera.OrthogonalizeViewUp()
      self.ren.ResetCameraClippingRange()
      self.ren_win.Render()

    # Pan translates x-y motion into translation of the focal point and
    # position.
    def Pan(self, renderer, camera, x, y, lastX, lastY, centerX, centerY):
      FPoint = camera.GetFocalPoint()
      FPoint0 = FPoint[0]
      FPoint1 = FPoint[1]
      FPoint2 = FPoint[2]

      PPoint = camera.GetPosition()
      PPoint0 = PPoint[0]
      PPoint1 = PPoint[1]
      PPoint2 = PPoint[2]

      renderer.SetWorldPoint(FPoint0, FPoint1, FPoint2, 1.0)
      renderer.WorldToDisplay()
      DPoint = renderer.GetDisplayPoint()
      focalDepth = DPoint[2]

      APoint0 = centerX + (x - lastX)
      APoint1 = centerY + (y - lastY)

      renderer.SetDisplayPoint(APoint0, APoint1, focalDepth)
      renderer.DisplayToWorld()
      RPoint = renderer.GetWorldPoint()
      RPoint0 = RPoint[0]
      RPoint1 = RPoint[1]
      RPoint2 = RPoint[2]
      RPoint3 = RPoint[3]

      if RPoint3 != 0.0:
        RPoint0 = RPoint0 / RPoint3
        RPoint1 = RPoint1 / RPoint3
        RPoint2 = RPoint2 / RPoint3

      camera.SetFocalPoint(
          (FPoint0 - RPoint0) / 2.0 + FPoint0, (FPoint1 - RPoint1) / 2.0 + FPoint1,
          (FPoint2 - RPoint2) / 2.0 + FPoint2
      )
      camera.SetPosition(
          (FPoint0 - RPoint0) / 2.0 + PPoint0, (FPoint1 - RPoint1) / 2.0 + PPoint1,
          (FPoint2 - RPoint2) / 2.0 + PPoint2
      )
      self.ren.ResetCameraClippingRange()
      self.ren_win.Render()

    # Dolly converts y-motion into a camera dolly commands.
    def Dolly(self, renderer, camera, x, y, lastX, lastY, centerX, centerY):
      dollyFactor = pow(1.02, (0.5 * (y - lastY)))
      if camera.GetParallelProjection():
        parallelScale = camera.GetParallelScale() * dollyFactor
        camera.SetParallelScale(parallelScale)
      else:
        camera.Dolly(dollyFactor)
        renderer.ResetCameraClippingRange()

      self.ren_win.Render()


  return vtkMain(*args, **kwargs)


class SphereActor(vtk.vtkActor):

  def __init__(self, rad, res, r, c):
    self.pos = np.array(r)
    self.source = vtk.vtkSphereSource()
    self.source.SetRadius(rad)
    self.source.SetPhiResolution(res)
    self.source.SetThetaResolution(res)
    self.source.SetCenter(r[0], r[1], r[2])
    self.Mapper = vtk.vtkPolyDataMapper()
    self.Mapper.SetInputConnection(self.source.GetOutputPort())
    self.SetMapper(self.Mapper)
    self.GetProperty().SetColor(c)
    self.GetProperty().SetPointSize(30)
    self.GetProperty().SetRenderPointsAsSpheres(1)
    self.GetProperty().SetVertexVisibility(1)
    #self.GetProperty().SetRepresentationToWireframe()
    self.GetProperty().SetEdgeVisibility(1)
    self.GetProperty().SetLineWidth(10)
    self.GetProperty().SetRenderLinesAsTubes(1)

  def move_to(self, r):
    self.pos = np.array(r)
    self.source.SetCenter(r[0], r[1], r[2])

  def set_color(self, color):
    self.GetProperty().SetColor(color)

  def set_rad(self, rad):
    self.source.SetRadius(rad)

  def get_pos(self):
    return self.pos


def create_line_actor(pts, colors=None):
  points = vtk.vtkPoints()
  profileData = vtk.vtkPolyData()

  # Number of points on the spline
  numberOfOutputPoints = len(pts)
  n = len(pts)

  for i, pt in enumerate(pts):
    points.InsertPoint(i, *pt)

  # Setup the colors array
  color_arr = vtk.vtkUnsignedCharArray()
  color_arr.SetNumberOfComponents(3)
  color_arr.SetName('Colors')
  if colors is None:
    cmap = get_colormap('viridis')
    colors = cmap[np.linspace(0, 1, n)]
  for i in range(n):
    col = colors[i].rgb[0] * 255
    color_arr.InsertNextTypedTuple(col.astype('uint8'))

  # Create the polyline.
  lines = vtk.vtkCellArray()
  for i in range(len(pts) - 1):
    aLine = vtk.vtkLine()
    aLine.GetPointIds().SetId(0, i)
    aLine.GetPointIds().SetId(1, i + 1)
    lines.InsertNextCell(aLine)

  profileData.SetPoints(points)
  profileData.SetLines(lines)
  profileData.GetCellData().SetScalars(color_arr)

  profileMapper = vtk.vtkPolyDataMapper()

  # Add thickness to the resulting line.
  if 0:
    profileTubes = vtk.vtkTubeFilter()
    profileTubes.SetNumberOfSides(1)
    profileTubes.SetInputData(profileData)
    profileTubes.SetRadius(.005)
    profileMapper.SetInputConnection(profileTubes.GetOutputPort())
  else:
    profileMapper.SetInputData(profileData)

  profile = vtk.vtkActor()
  profile.SetMapper(profileMapper)
  #profile.GetProperty().SetDiffuseColor(banana)
  #profile.GetProperty().SetSpecular(.3)
  #profile.GetProperty().SetSpecularPower(30)

  profile.GetProperty().SetEdgeVisibility(1)
  #profile.GetProperty().SetEdgeColor(0.9,0.9,0.4);
  profile.GetProperty().SetLineWidth(10)
  profile.GetProperty().SetPointSize(12)
  profile.GetProperty().SetRenderLinesAsTubes(1)
  profile.GetProperty().SetRenderPointsAsSpheres(1)
  profile.GetProperty().SetVertexVisibility(0)
  #profile.GetProperty().SetVertexColor(0.5,1.0,0.8);

  return profile


g_tex_quad = opa_struct.g_unit_box.quad


class TriangleActorVTK(TriangleActorBase):
  def _norm_tex(self, tex):
    return numpy2tex(tex[::-1])


  def _build_impl(self, tex):

    vtk_tex_coords = vtk.vtkFloatArray()
    vtk_tex_coords.SetNumberOfComponents(2)
    vtk_tex_coords.SetName("TextureCoordinates")
    vtk_points = vtk.vtkPoints()
    vtk_trs = vtk.vtkCellArray()

    for pt in self.points:
      vtk_points.InsertNextPoint(*pt)
    for tc in self.tex_coords:
      vtk_tex_coords.InsertNextTuple2(*tc)
    for tr in self.trs:
      vtktr = vtk.vtkTriangle()
      for i in range(3):
        vtktr.GetPointIds().SetId(i, tr[i])
      vtk_trs.InsertNextCell(vtktr)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_trs)
    polydata.GetPointData().SetTCoords(vtk_tex_coords)
    polydata.Modified()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.Update()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(tex)
    actor.GetProperty().BackfaceCullingOn();
    return actor

def create_vtk_reader(fname):
  if fname.endswith('.png'):
    reader = vtk.vtkPNGReader()
  else:
    reader = vtk.vtkTIFFReader()

  reader.SetFileName(fname)
  reader.Update()
  return reader

def vtk_do_flip(cur, flipx, flipy):
  for i, f in enumerate((flipx, flipy)):
    if f:
      fx = vtk.vtkImageFlip()
      fx.SetFilteredAxis(i)
      fx.SetInputConnection(cur.GetOutputPort())
      fx.Update()
      cur = fx
  return cur


def create_image_actor_base(idata):
  image_map = idata.rect

  atext = vtk.vtkTexture()
  cur = create_vtk_reader(idata.fname)
  cur = vtk_do_flip(cur, 0, 0)
  atext.SetInputConnection(cur.GetOutputPort())

  plane = vtk.vtkPlaneSource()  # size 1x1, normal z, centered on 0
  plane.SetCenter(0,0,0)
  plane.SetNormal(0,0,1)
  plane.Update()


  atext.InterpolateOn()

  planeMapper = vtk.vtkPolyDataMapper()
  planeMapper.SetInputConnection(plane.GetOutputPort())
  planeActor = vtk.vtkActor()
  planeActor.SetMapper(planeMapper)
  planeActor.SetTexture(atext)
  planeActor.SetScale(image_map.xr.length, image_map.yr.length, 1)
  #planeActor.GetProperty().SetOpacity(app.flags.opacity)
  planeActor.SetPosition(*idata.pos)
  return planeActor


def compute_cam_intersection(cam, plane, aspect):
  m = cam.GetCompositeProjectionTransformMatrix(aspect, -1, 1)
  m.Invert()

  plist = []
  t = vtk.mutable(0.0)
  interp = [0, 0, 0]

  p = vtk.vtkPlane()
  p.SetOrigin(*plane.center)
  p.SetNormal(*plane.normal)

  for py, px in cmisc.itertools.product((-1, 1), repeat=2):
    p1 = vtk_mulnorm(m, [px, py, 1])
    p2 = vtk_mulnorm(m, [px, py, -1])

    p.IntersectWithLine(p1, p2, t, interp)
    plist.append(np.array(interp))
  return plist


def read_img_gray_from_buf(buf):
  img = cv2.imdecode(np.asarray(bytearray(io.BytesIO(buf).read()), dtype=np.uint8), 0)
  return img

def numpy2tex(data):
  return reader2tex(numpy_to_vtk_image(data))


def numpy_to_vtk_image(data):
  data= cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

  data = np.ascontiguousarray(data.reshape((1,) + data.shape))
  res = vtkImageImportFromArray.vtkImageImportFromArray()
  res.SetArray(data)
  return res

  res = vtk.vtkImageData()
  arr = numpy_support.numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
  # .transpose(2, 0, 1) may be required depending on numpy array order see - https://github.com/quentan/Test_ImageData/blob/master/TestImageData.py

  res.SetDimensions(data.shape)
  res.SetNumberOfSidesz
  res.SetSpacing([1, 1, 1])
  res.SetOrigin([0, 0, 0])
  res.GetPointData().SetScalars(arr)
  return res


def reader2tex(reader):
  tex = vtk.vtkTexture()
  tex.SetInputConnection(reader.GetOutputPort())
  tex.InterpolateOn()
  return tex


def jpeg2tex(fname):
  reader = vtk.vtkJPEGReader()
  reader.SetFileName(fname)
  reader.Update()
  return reader2tex(reader)


def png2tex(fname):
  reader = vtk.vtkPNGReader()
  reader.SetFileName(fname)
  reader.Update()
  return reader2tex(reader)


def test(ctx):

  #create our new Qt MainWindow

  main_obj = vtkMain()
  #create our new custom VTK Qt widget

  t = np.linspace(0, 1, 100)
  pts = np.stack([np.cos(t * 2 * np.pi), np.sin(t * 2 * np.pi), t * 0], axis=-1)
  actor = create_line_actor(pts)
  main_obj.ren.AddActor(actor)

  sp = SphereActor(1e-5, 10, (0, 0, 0), Color('y').rgb)
  sp2 = SphereActor(1e-5, 10, (1, 0, 0), Color('y').rgb)
  main_obj.ren.AddActor(sp)
  main_obj.ren.AddActor(sp2)
  main_obj.run()


def test_triangles(ctx):

  main_obj = vtkMain()
  #create our new custom VTK Qt widget

  t = np.linspace(0, 1, 100)
  pts = np.stack([np.cos(t * 2 * np.pi), np.sin(t * 2 * np.pi), t * 0], axis=-1)
  actor = create_line_actor(pts)
  main_obj.ren.AddActor(actor)

  sp = SphereActor(1e-5, 10, (0, 0, 0), Color('y').rgb)
  sp2 = SphereActor(1e-5, 10, (1, 0, 0), Color('y').rgb)
  main_obj.ren.AddActor(sp)
  main_obj.ren.AddActor(sp2)
  main_obj.run()


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
