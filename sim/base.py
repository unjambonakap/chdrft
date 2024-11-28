#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import pymap3d
import chdrft.display.vtk as opa_vtk
import numpy as np
from chdrft.geo.satsim import TileGetter
import mercantile
from chdrft.utils.omath import MatHelper, rad2deg, rot_look_at, perspective
from chdrft.utils.geo import *

from chdrft.struct.base import Box, g_unit_box
import chdrft.struct.base as opa_struct
import datetime
import pytz
import scipy.interpolate
import chdrft.utils.Z as Z
from chdrft.dsp.image import ImageData
from chdrft.sim.rb.base import Transform
from astropy import constants as const
import pandas as pd

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
  EARTH_ELLIPSOID = pymap3d.Ellipsoid.from_name('wgs84')
  MOON_ELLIPSOID = pymap3d.Ellipsoid.from_name('moon')
  EARTH_ROUND_ASTROPY = pymap3d.Ellipsoid(
      semimajor_axis=const.R_earth.value, semiminor_axis=const.R_earth.value
  )


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


def compute_cam_parameters(
    center, focal_point, up, pts, expand=0.05, aspect=None, nearfar=(1, 1e20), blender=False
):
  z = make_norm(focal_point - center)
  up = make_orth_norm(up, z)

  angles, y = compute_angles(center, focal_point, up, pts)
  angle_box = Box.FromPoints(angles)
  angle_box = angle_box.center_on(np.zeros((2,)))
  angle_box = angle_box.expand(1 + expand)
  if aspect is not None:
    angle_box = angle_box.set_aspect(aspect)

  z = focal_point - center
  if blender: z = -z
  rot = rot_look_at(z, y)
  toworld = Transform.From(pos=center, rot=rot)
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
  import spiceypy
  if not isinstance(t_utc, datetime.datetime): t_utc = datetime.datetime.utcfromtimestamp(t_utc)
  t_utc = t_utc.astimezone(pytz.utc)
  return spiceypy.str2et(t_utc.strftime('%Y-%m-%dT%H:%M:%S'))


class InterpolatedDF:

  def __init__(self, df, index=None, **kwargs):
    self.cols = cmisc.Attr()

    if isinstance(df, pd.DataFrame):
      if index is None: index = df.index.values
      df = {column: df[column].to_numpy() for column in df.columns}
    self.index = index

    for column, data in df.items():
      y = np.stack(data, axis=0)
      self.cols[column] = scipy.interpolate.interp1d(index, y, axis=0, fill_value=(y[0], y[-1]), bounds_error=False, **kwargs)

  def __call__(self, t):
    res = cmisc.Attr()
    for k, v in self.cols.items():
      res[k] = v(t)
      if res[k].shape == ():
        res[k] = res[k][()]
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
    return MatHelper.mat_apply_nd(
        opa_vtk.vtk_matrix_to_numpy(self.actor.GetMatrix()), np.array(self.pts).T, point=True, n=3
    ).T

  def setup_internal(self):
    return

  def is_normal_actor(self):
    return self.typ not in (ActorType.Light, ActorType.Cam)


def create_earth_actors(
    actor_class,
    max_depth: int,
    tile_depth: int,
    params: TMSQuadParams,
    ll_box: Box = None
) -> SimpleVisitor:
  tg = TileGetter()
  u = SimpleVisitor(
      tg,
      actor_class,
      max_depth=max_depth,
      tile_depth=tile_depth,
      ll_box=ll_box,
  )
  u.run(TMSQuad.Root(params))
  return u


class EarthActor(Actor):

  def __init__(self):
    super().__init__(ActorType.Earth)

  def setup_internal(self):
    from chdrft.display.vtk import TriangleActorVTK
    u = create_earth_actors(TriangleActorVTK, 2, 2, m2u=TMSQuadParams(m2u=1e-3))
    earth_assembly = opa_vtk.vtk.vtkAssembly()
    for x in u.actors:
      actor = x.obj

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

  def set_pos_and_rot(self, pos, rot):
    a = opa_vtk.vtk.vtkMatrixToHomogeneousTransform()
    a.SetInput(build_user_matrix(pos, rot))
    self.actor.SetUserTransform(a)

  def run(self, *args, **kwargs):
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

  def __init__(
      self,
      width=None,
      height=None,
      offscreen=False,
      actors=None,
      dataf=lambda t: None,
      state_cb=None
  ):
    kwargs = dict(width=width, height=height)
    if offscreen:
      main = opa_vtk.vtk_offscreen_obj(**kwargs)
    else:
      main = opa_vtk.vtk_main_obj(**kwargs)
    if state_cb is None:
      state_cb = lambda data, tdesc: A(
          label=str(datetime.datetime.utcfromtimestamp(tdesc.t)), want=True, overlay=[]
      )
    self.state_cb = state_cb

    self.main = main
    self.normal_actors = [x for x in actors if x.is_normal_actor()]
    self.cam = cmisc.asq_query([x for x in actors if x.typ == ActorType.Cam]).single()
    self.lights = cmisc.asq_query([x for x in actors if x.typ == ActorType.Light]).to_list()
    self.actors = self.normal_actors + self.lights + [self.cam]
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
    for x in self.actors:
      x.run(t, data=data)

  def process(self, tl, no_render=None, outfile=None):
    from moviepy.editor import ImageClip, concatenate_videoclips
    imgs = []
    need_imgs = outfile or not no_render
    from chdrft.display.render import ImageGrid
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
      video = concatenate_videoclips(
          list([ImageClip(x[::-1]).set_duration(1 / fps) for x.img in imgs])
      )
      print(video.duration)
      video.write_videofile(outfile, fps=fps)


def test(ctx):
  import georinex as gr
  nav = gr.load(ctx.infile)
  tmp = nav.sel(sv='G01')
  x, y, z = gr.keplerian2ecef(tmp)

  print(x[-1], y[-1], z[-1])
  print(tmp.to_dataframe().iloc[-1])


class Quad:

  def __init__(self, box, depth, parent=None):
    self.box = box
    self.depth = depth
    self._children = None
    self.parent = parent

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


class TMSQuadParams(cmisc.PatchedModel):
  m2u: float
  ell: pymap3d.Ellipsoid = Consts.EARTH_ELLIPSOID


class TMSQuad:
  LMAX = 85.05113
  MAX_DEPTH = 20

  def __init__(self, x, y, z, params: TMSQuadParams, parent=None):
    self.x = x
    self.y = y
    self.z = z
    self.params = params
    self._children = None
    self.parent = parent
    self.depth = z
    self.data = None

  @property
  def children(self):
    if self._children is None:
      self._children = []
      if self.z + 1 < TMSQuad.MAX_DEPTH:
        for i in range(4):
          self._children.append(
              TMSQuad(
                  2 * self.x + (i & 1), 2 * self.y + (i >> 1), self.z + 1, self.params, parent=self
              )
          )
    return self._children

  def __iter__(self):
    return iter(self.children)

  @property
  def box_lonlat(self):
    bounds = mercantile.bounds(*self.xyz)
    box = Box(low=(bounds.west, bounds.south), high=(bounds.east, bounds.north))
    return box

  @property
  def box_lonlat_rad(self):
    return Z.deg2rad(self.box_lonlat)


  @property
  def ecef_points(self):
    p = self.box_lonlat.poly()
    return np.stack(pymap3d.geodetic2ecef(p[:, 1], p[:, 0], 0, ell=self.params.ell), axis=-1) * self.params.m2u

  @property
  def xyz(self):
    return self.x, self.y, self.z

  def tile(self, tg) -> np.ndarray:
    if self.data is None:
      self.data = tg.get_tile(*self.xyz)
    return self.data

  @staticmethod
  def Root(params: TMSQuadParams):
    return TMSQuad(0, 0, 0, params)

  def __str__(self):
    return f'xyz={self.xyz}'


class VisitorBase:

  @classmethod
  def DoVisit(cls, obj, func):
    if func(obj):
      for x in obj:
        cls.DoVisit(x, func)

  def run(self, obj):
    VisitorBase.DoVisit(obj, self)


class CylindricalVisitor(VisitorBase):

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
    super().run(root)


class SimpleVisitor(VisitorBase):

  def __init__(
      self,
      tg: TileGetter,
      actor_builder=None,
      max_depth=2,
      tile_depth=None,
      ll_box: Box = None, # lon,lat
      coord_map=lambda quad: quad.ecef_points,
  ):
    self.max_depth = max_depth
    if tile_depth is None: tile_depth = max_depth
    self.tile_depth = tile_depth
    self.actors = []
    self.items = []
    self.ll_box = ll_box or Box(xr=[-np.pi, np.pi], yr=[-np.pi / 2, np.pi / 2])
    self.actor_builder = actor_builder
    self.points = []
    self.tg = tg
    self.coord_map = coord_map

  def __call__(self, obj: TMSQuad):
    if not self.ll_box.intersects(obj.box_lonlat_rad): return 0
    if obj.z < self.max_depth: return 1

    ttile = obj
    while ttile.depth > self.tile_depth:
      ttile = ttile.parent
    tx = ttile.tile(self.tg)

    actor = self.actor_builder()
    actor.name = str(obj)
    pts = self.coord_map(obj)
    actor.full_quad(
        opa_struct.Quad(pts), uv=ttile.box_lonlat.to_box_space(obj.box_lonlat).quad.pts
    ).build(tx)
    self.actors.append(actor)
    self.points.extend(pts)
    self.items.append(ttile)
    return 0


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
