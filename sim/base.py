#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import georinex as gr
from chdrft.struct.geo import QuadTree
import pymap3d
import chdrft.display.vtk as opa_vtk
import numpy as np
from chdrft.geo.satsim import TileGetter
import mercantile
from chdrft.utils.math import MatHelper, rad2deg, deg2rad, rot_look_at, perspective
from chdrft.utils.geo import *
from chdrft.display.render import ImageGrid

from chdrft.struct.base import Box, g_unit_box
import chdrft.struct.base as opa_struct
import datetime
import pytz
import scipy.interpolate
from moviepy.editor import VideoClip, ImageClip, concatenate_videoclips
import chdrft.utils.Z as Z
import spiceypy
from chdrft.dsp.image import ImageData

global flags, cache
flags = None
cache = None


def render_params(parser):
  parser.add_argument('--infile')
  parser.add_argument('--outfile')
  parser.add_argument('--offscreen', action='store_true')
  parser.add_argument('--view-angle', default=10.)
  parser.add_argument('--no-render', action='store_true')
  parser.add_argument('--nframes', type=int)
  parser.add_argument('--width', type=int, default=800)
  parser.add_argument('--height', type=int, default=600)
  parser.add_argument('--rot-angle', type=float, default=0)
  parser.add_argument('--zoom-factor', type=float, default=None)


def args(parser):
  clist = CmdsList().add(test)
  render_params(parser)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class Consts:
  EARTH_ELLIPSOID = pymap3d.Ellipsoid('wgs84')
  MOON_ELLIPSOID = pymap3d.Ellipsoid('moon')


class TMSQuad:
  LMAX = 85.05113
  MAX_DEPTH = 20

  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z
    self._children = None

  @property
  def children(self):
    if self._children is None:
      self._children = []
      if self.z + 1 < TMSQuad.MAX_DEPTH:
        for i in range(4):
          self._children.append(TMSQuad(2 * self.x + (i & 1), 2 * self.y + (i >> 1), self.z + 1))
    return self._children

  def __iter__(self):
    return iter(self.children)

  @property
  def box_latlng(self):
    bounds = mercantile.bounds(*self.xyz)
    box = Box(low=(bounds.west, bounds.south), high=(bounds.east, bounds.north))
    return box

  @property
  def quad_ecef(self):
    p = self.box_latlng.poly()
    return opa_struct.Quad(
        np.stack(pymap3d.geodetic2ecef(p[:, 1], p[:, 0], 0, ell=Consts.EARTH_ELLIPSOID), axis=-1) /
        1e3
    )

  @property
  def xyz(self):
    return self.x, self.y, self.z

  def tile(self, tg):
    return tg.get_tile(*self.xyz)

  @staticmethod
  def Root():
    return TMSQuad(0, 0, 0)


def do_visit(obj, func):
  if func(obj):
    for x in obj:
      do_visit(x, func)


class Quad:

  def __init__(self, box, depth):
    self.box = box
    self.depth = depth
    self._children = None

  @property
  def children(self):
    if self._children is None:
      self._children = []
      for qd in self.box.quadrants:
        self._children.append(Quad(qd, self.depth + 1))
    return self._children

  def __iter__(self):
    return iter(self.children)

  @staticmethod
  def Root():
    return Quad(g_unit_box, 0)


def create_moon_data():
  tex = opa_vtk.jpeg2tex(cmisc.path_here('./Moon_LRO_LOLA_global_LDEM_1024.jpg'))
  width, height = tex.GetInput().GetDimensions()[:2]
  cv = CylindricalVisitor(3)
  cv.run()
  actor = cv.ta.build(tex)
  return cmisc.Attr(actor=actor, pts=cv.pts)


def compute_angles(center, focal_point, up, pts):
  center, focal_point, up, pts = cmisc.to_numpy_args(center, focal_point, up, pts)
  pts = pts.reshape((-1, 3))

  dpt = pts - focal_point
  dfocal = focal_point - center
  dfocal_len = np.linalg.norm(dfocal)
  z = dfocal / dfocal_len

  y = make_orth_norm(up, z)
  x = make_norm(np.cross(y, z))
  pt_len = np.linalg.norm(dpt)
  dpc = pts - center
  dproj = np.dot(dpc, z)

  #xp = np.arcsin(np.dot(pts, x) / np.linalg.norm(pts-center, axis=1))
  #yp = np.arcsin(np.dot(dpt, y) / np.linalg.norm(pts-center, axis=1))

  xp = np.arctan2(np.dot(dpc, x), dproj)
  yp = np.arctan2(np.dot(dpc, y), dproj)

  res = np.stack((xp, yp), axis=-1)
  return res, y


def compute_cam_parameters(center, focal_point, up, pts, expand=0.05, aspect=None, nearfar=(1,1e20), blender=False):
  z = make_norm(focal_point - center)
  up = make_orth_norm(up, z)

  angles, y = compute_angles(center, focal_point, up, pts)
  angle_box = Box.FromPoints(angles)
  angle_box = angle_box.center_on(np.zeros((2,)))
  angle_box = angle_box.expand(1 + expand)
  if aspect is not None:
    angle_box = angle_box.set_aspect(aspect)

  z = focal_point-center
  if blender: z = -z
  rot = rot_look_at(z ,y)
  toworld = MatHelper.simple_mat(offset=center, rot=rot)
  persp = None
  if aspect:
    persp = perspective(angle_box.yn, aspect, *nearfar)
  return cmisc.Attr(angle_box=angle_box, y=y, z=z, toworld=toworld, perspective=persp, rot=rot)


def vtk_actor_get_points(e, matrix):
  polydata = e.GetMapper().GetInput()
  for i in range(polydata.GetNumberOfPoints()):
    yield MatHelper.mulnorm(matrix, e.GetMapper().GetInput().GetPoint(i))


def vtk_assembly_get_points(e, matrix):
  e.InitPathTraversal()
  while True:
    x = e.GetNextPath()
    if x is None: break
    px = x.GetLastNode().GetViewProp()
    curmat = opa_vtk.vtk_matrix_to_numpy(x.GetLastNode().GetMatrix())
    yield from vtk_get_points(px, MatHelper.matmul(matrix, curmat))


def vtk_get_points(e, matrix=None):
  if matrix is None: matrix = MatHelper.IdMat4()
  if isinstance(e, opa_vtk.vtk.vtkAssembly):
    return list(vtk_assembly_get_points(e, matrix))
  else:
    return list(vtk_actor_get_points(e, matrix))


def render_it(main, earth_pos, moon_pos, cam_pos):
  main.ren.RemoveAllViewProps()
  up = (0, 0, 1)

  moon_data = create_moon_data()
  moon_actor = moon_data.actor
  moon_actor.GetProperty().SetAmbient(1)
  moon_actor.GetProperty().SetDiffuse(0)
  moon_actor.SetPosition(*moon_pos)

  tg = TileGetter()
  u = SimpleVisitor(tg, 4)
  u.run_tms()
  earth_assembly = opa_vtk.vtk.vtkAssembly()
  for actor in u.actors:
    earth_assembly.AddPart(actor)
    actor.GetProperty().SetAmbient(1)
    actor.GetProperty().SetDiffuse(0)

  earth_assembly.SetPosition(*earth_pos)

  pts_list = np.concatenate((np.array(u.pts) + earth_pos, np.array(moon_data.pts) + moon_pos))
  focal_point = earth_pos
  #pts_list = np.concatenate((np.array(moon_data.pts) + moon_pos, ))
  #focal_point = moon_pos
  #focal_point = earth_pos + (moon_pos - earth_pos)*0.59

  main.ren.AddActor(earth_assembly)
  main.ren.AddActor(moon_actor)

  print('fuu ', main.aspect)
  params = compute_cam_parameters(
      cam_pos,
      focal_point,
      up,
      pts_list,
      expand=0.05,
      aspect=main.aspect,
  )

  #main.cam.SetFocalPoint(*moon_pos)
  main.cam.SetClippingRange(1, 1e20)
  angle_y = rad2deg(params.box.height)
  print('ANGLE > > ', angle_y, params.box.height)

  z = make_norm(focal_point - cam_pos)
  if flags.rot_angle is not None: params.y = rotate_vec(params.y, z, flags.rot_angle)
  if flags.zoom_factor is not None: angle_y /= flags.zoom_factor

  main.cam.SetPosition(*cam_pos)
  main.cam.SetFocalPoint(*focal_point)
  main.cam.SetViewAngle(angle_y)
  main.cam.SetViewUp(*params.y)
  main.cam.SetClippingRange(1, 1e20)
  return cmisc.Attr(earth=earth_assembly, moon=moon_actor, cam=main.cam, params=params)


def build_user_matrix(pos, rot, inv=0):
  mat = np.identity(4)
  mat[:3, 3] = pos
  mat[:3, :3] = rot
  if inv: mat = np.linalg.inv(mat)
  return numpy_to_vtk_mat(mat)


eastern = pytz.timezone('US/Eastern')


def load_edt_date(t):
  t_edt = eastern.localize(datetime.datetime.fromisoformat(t))
  return t_edt.astimezone(pytz.utc)


def spice_time(t_utc):
  if not isinstance(t_utc, datetime.datetime): t_utc = datetime.datetime.utcfromtimestamp(t_utc)
  t_utc = t_utc.astimezone(pytz.utc)
  return spiceypy.str2et(t_utc.strftime('%Y-%m-%dT%H:%M:%S'))


class InterpolatedDF:

  def __init__(self, df, **kwargs):
    self.cols = cmisc.Attr()
    self.df = df
    for column in df.columns:
      y = np.stack(df[column].to_numpy(), axis=0)
      self.cols[column] = scipy.interpolate.interp1d(df.index.values, y, axis=0, **kwargs)

  def __call__(self, t):
    res = cmisc.Attr()
    for k, v in self.cols.items():
      res[k] = v(t)
    return res


class ActorType(cmisc.Enum):
  Earth = 'earth'
  Moon = 'moon'
  Cam = 'cam'
  Light = 'light'


class Actor:

  def __init__(self, typ):
    self.typ = typ
    self.actor = None
    self.main = None
    self.ren = None
    self.cam = None
    self.runt = None
    self.first = True
    self.setup_f = None

  def setup(self, ren):
    self.main = ren.main
    self.ren = ren
    self.cam = self.main.cam
    self.setup_internal()
    if self.setup_f:
      self.setup_f(self)

  def run(self, t, **kwargs):
    self.runt(self, t, first=self.first, **kwargs)
    self.first = False

  def set_pos_and_rot(self, pos, rot):
    self.actor.SetPosition(0, 0, 0)
    self.actor.SetUserMatrix(build_user_matrix(pos, rot))

  def set_pos(self, pos):
    self.actor.SetPosition(*pos)

  def get_pos(self):
    return np.array(self.actor.GetPosition())

  def get_pts_world(self):
    return MatHelper.mat_apply_nd(opa_vtk.vtk_matrix_to_numpy(self.actor.GetMatrix()), np.array(self.pts).T, point=True, n=3).T

  def setup_internal(self):
    return

  def is_normal_actor(self):
    return self.typ not in (ActorType.Light, ActorType.Cam)


class EarthActor(Actor):

  def __init__(self):
    super().__init__(ActorType.Earth)

  def setup_internal(self):
    from chdrft.display.vtk import TriangleActorVTK
    tg = TileGetter()
    u = SimpleVisitor( tg,TriangleActorVTK,  2)
    do_visit(TMSQuad.Root(1e-3), u)
    earth_assembly = opa_vtk.vtk.vtkAssembly()
    for x in u.actors:
      actor =x.obj

      earth_assembly.AddPart(actor)
      actor.GetProperty().SetAmbient(0)
      actor.GetProperty().SetDiffuse(1)
    self.actor = earth_assembly
    self.pts = u.pts


class MoonActor(Actor):

  def __init__(self):
    super().__init__(ActorType.Moon)

  def setup_internal(self):
    moon_data = create_moon_data()
    moon_actor = moon_data.actor
    moon_actor.GetProperty().SetAmbient(0)
    moon_actor.GetProperty().SetDiffuse(1)
    self.actor = moon_actor
    self.pts = moon_data.pts


class CamActor(Actor):

  def __init__(self):
    super().__init__(ActorType.Cam)

  def setup_internal(self):
    self.actor = self.main.cam
    pass

  def set_pos_and_rot(self, pos, rot):
    a = opa_vtk.vtk.vtkMatrixToHomogeneousTransform()
    a.SetInput(build_user_matrix(pos,rot))
    self.actor.SetUserTransform(a)

  def run(self,*args, **kwargs):
    super().run(*args, **kwargs)

  @property
  def proj_mat(self):
    return opa_vtk.vtk_matrix_to_numpy(
        self.actor.GetCompositeProjectionTransformMatrix(self.main.aspect, -1, 1)
    )

  def focus_on_scene(self, focal_point, up, rot_angle=None, zoom_factor=None):
    pts_list = []
    for x in self.ren.normal_actors:
      a = x.get_pts_world()
      pts_list.extend(a)

    self.focus_on_points(focal_point, up, pts_list, rot_angle, zoom_factor)

  def focus_on_points(self, focal_point, up, pts_list, rot_angle=None, zoom_factor=None):
    pos = self.get_pos()
    params = compute_cam_parameters(
        pos,
        focal_point,
        up,
        pts_list,
        expand=0.05,
        aspect=self.main.aspect,
    )
    angle_y = rad2deg(params.box.height)

    z = make_norm(focal_point - pos)
    if rot_angle is not None: params.y = rotate_vec(params.y, z, rot_angle)
    if zoom_factor is not None: angle_y /= zoom_factor
    print('FOCUS >> ', angle_y, params.y)
    self.actor.SetClippingRange(1, 1e20)
    self.actor.SetFocalPoint(*focal_point)
    self.actor.SetViewAngle(angle_y)
    self.actor.SetViewUp(*params.y)


class Renderer:

  def __init__(self, width=None, height=None, offscreen=False, actors=None, dataf=lambda t: None, state_cb=None):
    kwargs = dict(width=width, height=height)
    if offscreen:
      main = opa_vtk.vtk_offscreen_obj(**kwargs)
    else:
      main = opa_vtk.vtk_main_obj(**kwargs)
    if state_cb is None:
      state_cb = lambda data, tdesc: A(label=str(datetime.datetime.utcfromtimestamp(tdesc.t)), want=True, overlay=[])
    self.state_cb = state_cb

    self.main = main
    self.normal_actors = [x for x in actors if x.is_normal_actor()]
    self.cam = cmisc.asq_query([x for x in actors if x.typ == ActorType.Cam]).single()
    self.lights = cmisc.asq_query([x for x in actors if x.typ == ActorType.Light]).to_list()
    self.actors =  self.normal_actors + self.lights + [self.cam]
    self.dataf = dataf
    self.prepare()

  def prepare(self):
    self.main.ren.RemoveAllViewProps()
    for x in self.actors:
      x.setup(self)

    for x in self.normal_actors:
      self.main.ren.AddActor(x.actor)
    for x in self.lights:
      self.main.ren.AddLight(x.actor)

  def render_at(self, t):
    self.configure_at(t)
    return self.main.render()

  def configure_at(self, t):
    data = self.dataf(t)
    self.cur_data = data
    for x in self.actors: x.run(t, data=data)

  def process(self, tl, no_render=None, outfile=None):
    imgs = []
    need_imgs  = outfile or not no_render
    for i, t in enumerate(tl):
      self.configure_at(t)
      state = self.state_cb(self.cur_data, A(t=t, idx=i))

      if need_imgs and state.want:
        res = self.main.render()
        imgs.append(ImageData(img=res, stuff=A(label=state.label, overlay=state.overlay)))

    if not no_render and len(imgs) > 0:
      ig = ImageGrid(images=imgs)
      grid_imgs = ig.get_images()
      mo = cmisc.Attr(images=grid_imgs, misc=[])
      meshes = [mo]

      for e in grid_imgs:
        for ov in e.stuff.overlay:
          ov.transform = e.box.to_vispy_transform()
          meshes.append(ov)
        mo.misc.append(A(text=e.stuff.label, pos=e.pos, zpos=-10))


      import chdrft.utils.K as K
      K.vispy_utils.render_for_meshes(meshes)

    if outfile:
      fps = 10
      video = concatenate_videoclips(list([ImageClip(x[::-1]).set_duration(1 / fps) for x.img in imgs]))
      print(video.duration)
      video.write_videofile(outfile, fps=fps)


def test(ctx):
  nav = gr.load(ctx.infile)
  tmp = nav.sel(sv='G01')
  x, y, z = gr.keplerian2ecef(tmp)

  print(x[-1], y[-1], z[-1])
  print(tmp.to_dataframe().iloc[-1])


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()

class Consts:
  EARTH_ELLIPSOID = pymap3d.Ellipsoid('wgs84')
  MOON_ELLIPSOID = pymap3d.Ellipsoid('moon')

class Quad:

  def __init__(self, box, depth, parent=None):
    self.box = box
    self.depth = depth
    self._children = None
    self.parent= parent

  @property
  def children(self):
    if self._children is None:
      self._children = []
      for qd in self.box.quadrants:
        self._children.append(Quad(qd, self.depth + 1, self))
    return self._children

  def __iter__(self):
    return iter(self.children)

  @staticmethod
  def Root():
    return Quad(g_unit_box, 0)


class TMSQuad:
  LMAX = 85.05113
  MAX_DEPTH = 20

  def __init__(self, x, y, z, u2s, parent=None):
    self.x = x
    self.y = y
    self.z = z
    self.u2s=u2s
    self._children = None
    self.parent = parent
    self.depth = z

  @property
  def children(self):
    if self._children is None:
      self._children = []
      if self.z + 1 < TMSQuad.MAX_DEPTH:
        for i in range(4):
          self._children.append(TMSQuad(2 * self.x + (i & 1), 2 * self.y + (i >> 1), self.z + 1, self.u2s, parent=self))
    return self._children

  def __iter__(self):
    return iter(self.children)

  @property
  def box_latlng(self):
    bounds = mercantile.bounds(*self.xyz)
    box = Box(low=(bounds.west, bounds.south), high=(bounds.east, bounds.north))
    return box

  @property
  def quad_ecef(self):
    p = self.box_latlng.poly()
    return opa_struct.Quad(
        np.stack(pymap3d.geodetic2ecef(p[:, 1], p[:, 0], 0, ell=Consts.EARTH_ELLIPSOID), axis=-1) * self.u2s
    )

  @property
  def xyz(self):
    return self.x, self.y, self.z

  def tile(self, tg):
    return tg.get_tile(*self.xyz)

  @staticmethod
  def Root(u2s):
    return TMSQuad(0, 0, 0, u2s)
  def __str__(self): return f'xyz={self.xyz}'

class CylindricalVisitor:

  def __init__(self, actor_builder, max_depth=2, m2u=1e3):
    self.m2u = m2u
    self.max_depth = max_depth
    self.ta = actor_builder()
    self.real_box = Box(low=(-np.pi, -np.pi / 2), high=(np.pi, np.pi / 2))
    self.pts = []

  def __call__(self, obj):
    if obj.depth < self.max_depth: return 1
    base_p = obj.box.poly()
    p = self.real_box.from_box_space(base_p)

    pmap = np.stack(
        pymap3d.geodetic2ecef(p[:, 1], p[:, 0], 0, ell=Consts.MOON_ELLIPSOID, deg=0), axis=-1
    ) * self.m2u
    self.pts.extend(pmap)
    self.ta.add_quad(pmap, base_p)
    return 0

  def run(self):
    root = Quad.Root()
    do_visit(root, self)


class SimpleVisitor:

  def __init__(self, tg, actor_builder=None, max_depth=2, tile_depth=None):
    self.max_depth = max_depth
    if tile_depth is None: tile_depth= max_depth
    self.tile_depth = tile_depth
    self.actors = []
    self.actor_builder = actor_builder
    self.points = []
    self.tg = tg

  def __call__(self, obj):
    if obj.z < self.max_depth: return 1
    ttile = obj
    while ttile.depth > self.tile_depth:
        ttile = ttile.parent
    tx = ttile.tile(self.tg)

    actor = self.actor_builder()
    actor.name = str(obj)
    actor.full_quad(obj.quad_ecef, uv=ttile.box_latlng.to_box_space(obj.box_latlng).quad.pts).build(tx)
    self.actors.append(actor)
    self.points.extend(obj.quad_ecef.pts)
    return 0




