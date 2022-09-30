#!/usr/bin/env python

from astroquery.jplhorizons import Horizons
from chdrft.cmds import CmdsList
from chdrft.cmds import CmdsList
from chdrft.display.base import TriangleActorBase
from chdrft.display.vtk import vtk_main_obj
from chdrft.geo.satsim import TileGetter
from chdrft.main import app
from chdrft.main import app
from chdrft.sim.base import *
from chdrft.sim.base import render_params
from chdrft.sim.utils import *
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
from chdrft.utils.misc import Attributize
from chdrft.utils.types import *
import calcephpy
import chdrft.display.vtk as opa_vtk
import chdrft.utils.Z as Z
import chdrft.utils.misc as cmisc
import chdrft.utils.misc as cmisc
import cv2
import glog
import glog
import meshio
import numpy as np
import numpy as np
import spiceypy

global flags, cache
flags = None
cache = None


def moon_sunrise_params(parser):
  parser.add_argument('--moon-model', type=str, default=cmisc.path_here('./t1.small.stl'))
  render_params(parser)


def args(parser):
  clist = CmdsList()
  moon_sunrise_params(parser)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def norm_date(d, tz=pytz.utc):
  return tz.localize(datetime.datetime.fromisoformat(d))


class ActorType(cmisc.Enum):
  Earth = 'earth'
  Moon = 'moon'
  Cam = 'cam'
  Light = 'light'
  Misc = 'misc'


def set_analysis_parameters(ctx, analysis, large=False):
  polar_model = [cmisc.path_here('./notebooks//north.small.stl'), cmisc.path_here('./notebooks/north.large.stl')][large]
  earthrise_model = [cmisc.path_here('./notebooks/t1.small.stl'), cmisc.path_here('./notebooks/t1.large.stl')][large]
  ctx.polar_mode = 0
  ctx.obs_conf = A(obs='LRO', obs_frame='LRO_LROCNACL')
  if analysis == 'polar':
    ctx.polar_mode = 1
    ctx.moon_model = polar_model
    t0 = norm_date('2014-02-01T12:25:00') + datetime.timedelta(seconds=10)
    ctx.view_angle = 41.9
    ctx.rot_angle = -np.pi/2
  elif analysis == 'lro_earthrise':
    ctx.moon_model = earthrise_model
    t0 = norm_date('2015-10-12 12:18:30')
    ctx.view_angle = 5.9
    ctx.rot_angle = 0
  elif analysis == 'kaguya':
    polar = 0
    jst = pytz.timezone('Asia/Tokyo')
    t0 = norm_date('2007-11-07T14:56:26', tz=jst)
    ctx.moon_model = polar_model
    ctx.obs_conf = A(obs='SELENE', obs_frame='SELENE_HDTV_WIDE')
    ctx.rot_angle = -np.pi
    ctx.view_angle = 29.45
  else: assert 0
  ctx.t0 = t0


class MoonSunrise:

  @property
  def km2u(self):
    return self.m2u * 1000

  def __init__(self, ctx):
    self.m2u = 1e-3
    METAKR = '/home/benoit/programmation/science/sim/kernels/lro.km'
    spiceypy.kclear()
    spiceypy.furnsh(METAKR)
    ctx.aspect = ctx.width / ctx.height
    self.ctx = ctx
    self.ctx.get_or_insert('obs_conf', A(obs='LRO', obs_frame='LRO_LROCNACL'))
    self.ctx.get_or_insert('earth_depth', 4)
    self.ctx.get_or_insert('earth_tile_depth', None)
    self.ctx.get_or_insert('moon_details', 6)

    self.tgtime = norm_date('2015-10-12 12:18:33')

  def build(self):
    self.objs = {}
    self.objs[ActorType.Earth] = self.create_earth_actor()
    self.objs[ActorType.Moon] = self.create_moon_actor()
    self.objs[ActorType.Light] = self.create_light()
    self.objs[ActorType.Cam] = self.create_camera()
    for t, x in self.objs.items():
      x.type = t

  def get_data_at_time(self, t_utc):
    if not isinstance(t_utc, datetime.datetime): t_utc = datetime.datetime.utcfromtimestamp(t_utc)
    et = spice_time(t_utc)
    ref_frame = 'MOON_ME_DE421'
    obs_conf = self.ctx.obs_conf
    sun_data, sun_lt = spiceypy.spkezr('SUN', et, ref_frame, 'LT+S', obs_conf.obs)
    earth_data, earth_lt = spiceypy.spkezr('EARTH', et, ref_frame, 'LT+S', obs_conf.obs)
    moon_data, moon_lt = spiceypy.spkezr('moon', et, ref_frame, 'LT+S', obs_conf.obs)
    sat_data, sat_lt = spiceypy.spkezr(obs_conf.obs, et, ref_frame, 'LT+S', obs_conf.obs)

    res = A()
    res[ActorType.Moon] = A(
        pos=moon_data[:3], v=moon_data[3:], rot=spiceypy.pxform(ref_frame, ref_frame, et - moon_lt)
    )
    res[
        ActorType.Cam
    ] = A(pos=sat_data[:3], v=sat_data[3:], rot=spiceypy.pxform(obs_conf.obs_frame, ref_frame, et))
    res[ActorType.Earth] = A(
        pos=earth_data[:3],
        v=earth_data[3:],
        rot=spiceypy.pxform('ITRF93', ref_frame, et - earth_lt)
    )
    res[ActorType.Light] = light = A(
        pos=sun_data[:3],
        v=sun_data[3:],
        rot=Z.opa_math.rot_look_at(np.array(moon_data[:3]) - sun_data[:3], [0, 1, 1]).as_matrix()
    )

    # gives result in km, want m
    for x in res.values():
      x.pos = x.pos * 1000 * self.m2u
    res.t = t_utc.timestamp()
    return res

  def configure_at(self, t=None, data=None):
    if data is None: data = self.get_data_at_time(t)
    for k, v in self.objs.items():
      if v.obj is None: continue
      e = data[k]
      self.configure_obj(v, e)

  def configure_obj(self, obj, e):
    e.mat = MatHelper.simple_mat(offset=e.pos, rot=e.rot)
    mat = self.norm_mat(obj.type, e.mat)
    obj._toworld = mat
    obj._data = e
    self.set_mat(obj, mat)

  def set_mat(self, obj, mat):
    pass

  def norm_mat(self, objtype, mat):
    if objtype in (ActorType.Cam, ActorType.Light):
      mat = MatHelper.mat_apply_nd(
          mat, MatHelper.simple_mat(rot=np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T)
      )
      if objtype == ActorType.Cam and self.ctx.rot_angle is not None:
        mat = mat @ Z.opa_math.MatHelper.mat_rot(
            R.from_rotvec(np.array([0, 0, 1]) * -self.ctx.rot_angle)
        )
      pass
    return mat

  @property
  def actor_class(self):
    return TriangleActorBase

  def configure_cam(self, res=None, aspect=None, view_angle=None):

    cobj = self.objs[ActorType.Cam]
    if res is not None:
      aspect = res[0] / res[1]
      cobj.internal.res = res
    if aspect is not None: cobj.internal.aspect = aspect
    if view_angle is not None: cobj.internal.view_angle = view_angle

    local2clip = Z.opa_math.perspective(cobj.internal.view_angle, cobj.internal.aspect, 1, 1e20)
    cobj.internal.local2clip = local2clip
    self.configure_cam_internal(cobj)

  def configure_cam_internal(self, cobj):
    pass

  def configure_render(self, resx, resy, view_angle=None):
    self.configure_cam((resx, resy), view_angle=view_angle)

  def create_earth_actor_impl(self, u):
    return 'earth'

  def create_earth_actor(self):
    tg = TileGetter()
    u = SimpleVisitor(
        tg, self.actor_class, self.ctx.earth_depth, tile_depth=self.ctx.earth_tile_depth
    )
    do_visit(TMSQuad.Root(self.m2u), u)
    return A(internal=u, obj=self.create_earth_actor_impl(u))

  def create_moon_actor_impl(self, u):
    return 'moon'

  def create_moon_actor(self):
    ta = None
    moon_model = self.ctx.moon_model
    if not moon_model: return None

    if moon_model == 'pic':
      tex = cv2.imread('/home/benoit/programmation/chdrft/sim/Moon_LRO_LOLA_global_LDEM_1024.jpg')
      cv = CylindricalVisitor(self.actor_class, self.ctx.moon_details, m2u=self.m2u)
      cv.run()
      ta = cv.ta
      ta.build(tex)
    else:
      rimg = np.array([[[255, 255, 255]]], dtype=np.uint8)
      trx = self.actor_class()
      m1 = meshio.Mesh.read(moon_model)

      trx.add_points(m1.points * self.km2u)
      for x in m1.points:
        trx.tex_coords.append((0, 0))
      triangles = m1.cells_dict['triangle']
      for x in triangles:
        trx.push_triangle(x)
      trx.build(rimg)
      trx.name = 'dummy'
      ta = trx
    return A(internal=ta, obj=self.create_moon_actor_impl(ta))

  def create_camera_impl(self, u):
    return 'dummy'

  def create_camera(self):

    u = A()
    u.view_angle = self.ctx.view_angle
    u.res = (self.ctx.width, self.ctx.height)
    return A(internal=u, obj=self.create_camera_impl(u))

  def create_light_impl(self, u):
    return None

  def create_light(self):
    u = None
    return A(internal=u, obj=self.create_light_impl(u))


def setup_camera_vtk(cam, tsfmat, view_angle=None, aspect=None):
  cam.SetViewAngle(view_angle)
  cam.SetClippingRange(1, 1e20)

  a = opa_vtk.vtk.vtkTransform()
  a.SetMatrix(numpy_to_vtk_mat(tsfmat))
  cam.ApplyTransform(a)
  if aspect is not None:
    return opa_vtk.vtk_matrix_to_numpy(cam.GetCompositeProjectionTransformMatrix(aspect, -1, 1))


def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
