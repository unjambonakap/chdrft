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
from chdrft.display.blender import *
from chdrft.display.blender import BlenderTriangleActor
from chdrft.sim.rb.rb_gen import RigidBody
from chdrft.sim.rb.base import SurfaceMeshParams, Vec3, AABB, Transform
import chdrft.utils.colors as color_utils

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  moon_sunrise_params(parser)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def find_node_io(nt, name, node_name=None, out=None):
  tg = cmisc.asq_query(nt.nodes).where(lambda x: node_name is None or x.name == node_name)
  return tg.select_many(lambda x: x.outputs if out else x.inputs).where(lambda x: x.name == name
                                                                       ).single_or_default(None)


def find_node(nt, name):
  tg = cmisc.asq_query(nt.nodes).where(lambda x: x.name == name).single_or_default(None)
  return tg


def actors_to_obj(name, col, actors: list[BlenderTriangleActor]):
  res = make_group(name, col, [x.obj for x in actors])
  locs = []
  for x in actors:
    locs.append(Vec3(np.array(x.obj.location)))
    create_material_for_obj(x.obj, x.tex, f'{name}_{x.name}')
  mean_pos = np.mean(locs)
  res.location = mean_pos.vdata
  for p, x in zip(locs, actors):
    x.obj.location = (p - mean_pos).vdata
  return res


class MoonSunriseBlender(MoonSunrise):

  def __init__(self, ctx):
    super().__init__(ctx)
    self.m2u = 1e-6

  def build(self):
    clear_scene()
    bpy.context.scene.world.color = (0, 0, 0)
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    self.obj_col = bpy.data.collections.new('obj_col')
    bpy.context.scene.collection.children.link(self.obj_col)

    self.env_col = bpy.data.collections.new('env_col')
    bpy.context.scene.collection.children.link(self.env_col)

    filepath = cmisc.path_here('./resources.blend')
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
      data_to.node_groups = data_from.node_groups
    self.earth_grp = bpy.data.node_groups['node_earth_mat']

    super().build()
    self.configure_render(self.ctx.width, self.ctx.height)
    self.post_build()

  def configure_cam_internal(self, cobj):
    cobj.obj.data.angle_y = np.deg2rad(cobj.internal.view_angle)
    bpy.context.scene.render.resolution_x = cobj.internal.res[0]
    bpy.context.scene.render.resolution_y = cobj.internal.res[1]
    bpy.context.scene.render.engine = 'CYCLES'

  def configure_earth_mat(self, **kwargs):
    for k, v in kwargs.items():
      find_node_io(self.earth_grp, k).default_value = v

  def post_build(self):

    earth = self.objs[ActorType.Earth]

    for x in earth.obj.children:
      nt = x.active_material.node_tree
      inst_grp_name = 'EarthMatNode'

      if find_node(nt, inst_grp_name): continue
      c = find_node(nt, 'Principled BSDF')
      if c is not None: nt.nodes.remove(c)
      nx = nt.nodes.new('ShaderNodeGroup')
      nx.node_tree = self.earth_grp
      nx.name = inst_grp_name

      surface = find_node_io(nt, 'Surface', out=0)
      tex = find_node_io(nt, 'Color', out=1)
      surface_o = find_node_io(nt, 'BSDF', out=1)
      tex_in = find_node_io(nt, 'Base Color', out=0)

      nt.links.new(surface_o, surface)
      nt.links.new(tex, tex_in)

  def set_mat(self, obj, mat):
    obj.obj.matrix_world = mat.T

  @property
  def actor_class(self):
    return BlenderTriangleActor

  def create_earth_actor_impl(self, u):
    name = 'earth'
    return actors_to_obj(name, self.obj_col, u.actors)

  def create_moon_actor_impl(self, ta):
    moon = ta.obj
    if ta.tex is not None:
      create_material_for_obj(moon, ta.tex, 'moon')
    return make_group('moon', self.obj_col, [moon])

  def create_camera_impl(self, internal):
    cobj = bpy.data.cameras.new('Camera')
    cam = bpy.data.objects.new('Camera', cobj)
    self.env_col.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam

  def create_light_impl(self, internal):
    sun = bpy.data.objects.new('Sun', bpy.data.lights.new('sun', 'SUN'))
    self.env_col.objects.link(sun)
    sun.data.angle = 0
    sun.data.energy = 1
    return sun

  def render(self, fname):
    bpy.context.scene.render.filepath = fname
    bpy.ops.render.render(write_still=True)


class ObjectSync:

  def __init__(self, src, dest):
    self.src = src
    self.dest = dest
    self.src2dest = self.compute_src2dest()

  def compute_src2dest(self):
    return np.linalg.inv(self.dest.mat_world) @ self.src.mat_world

  def sync(self):
    res = self.src.mat_world @ np.linalg.inv(self.src2dest)
    self.dest.mat_world = res


class BlenderPhysHelper:

  def __init__(self, scale: float = 1, shift_p: Vec3 = Vec3.Zero()):
    clear_scene()
    self.scale = scale
    self.shift_p = shift_p
    self.obj2blender: dict[RigidBody, BlenderObjWrapper] = dict()
    self.main_col = bpy.data.collections.new('obj_col')
    self.env_col = bpy.data.collections.new('env_col')

    bpy.context.scene.collection.children.link(self.env_col)
    bpy.context.scene.collection.children.link(self.main_col)
    self.cam = self.create_camera()
    self.sctx: SceneContext = None
    self._objs: RigidBody = []


  def update_context(self):
    bpy.context.view_layer.update()

  def update(self, animation: AnimationSceneHelper = None, pre_cam_cb=None):
    # needed for recomputing matrix_world
    self.update_context()

    updates = []
    for child in self.sctx.obj2name.keys():
      updates.append(
          KeyframeObjData(
              obj=self.obj2blender[child],
              wl=child.self_link.wl.shift(self.shift_p).scale_p(self.scale).data
          )
      )

    for x in updates:
      x.propagate()

    self.update_context()
    if pre_cam_cb is not None:
      pre_cam_cb()


    cam_up = KeyframeObjData(obj=self.cam, wl=self.cam.mat_world)
    cam_up.propagate()
    updates.append(cam_up)
      #osync.sync()
    if animation: animation.push(updates)


  def create_camera(self) -> BlenderObjWrapper:
    cobj = bpy.data.cameras.new('Camera')
    cam = bpy.data.objects.new('Camera', cobj)
    cam.data.clip_start = 1e-4
    cam.data.clip_end = 1e6
    self.env_col.objects.link(cam)
    bpy.context.scene.camera = cam
    return BlenderObjWrapper(cam)

  def create_mat(self, color, name):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat

  def load(self, obj: RigidBody):
    res = self.create_from_obj(obj)
    self.cmap = color_utils.ColorMapper(self.obj2blender.keys())
    self.load_positions()
    self.sctx = obj.ctx
    self._objs.append(obj)

    for obj, bobj, in self.obj2blender.items():
      self.main_col.objects.link(bobj.internal)
      bobj.internal.active_material = self.create_mat(self.cmap.get(obj) + (0.5,), obj.name)

  def create_from_obj(self, obj: RigidBody) -> BlenderTriangleActor:
    actor = BlenderTriangleActor.BuildFrom(
        obj.spec.mesh.surface_mesh(SurfaceMeshParams()),
        name=obj.name,
        scale=self.scale,
    )
    obj_blender = BlenderObjWrapper(actor.blender_obj)
    self.obj2blender[obj] = obj_blender
    for x in obj.move_links:
      child = self.create_from_obj(x.rb)
      child.internal.parent = obj_blender.internal
    return obj_blender

  def load_positions(self):
    for obj, bobj, in self.obj2blender.items():
      bobj.mat_local = obj.self_link.wl.scale_p(self.scale).data

  def set_cam_focus(self, points, dv_rel: Vec3 = None, dv_abs: Vec3 = None, expand=0) -> Transform:

    aabb = AABB.FromPoints(np.array(points))

    if dv_abs is None:
      dv_abs = dv_rel * Vec3(aabb.v)
    cam_loc = Vec3.Pt(aabb.center) + dv_abs
    cam_params = compute_cam_parameters(
        cam_loc.vdata,
        aabb.center,
        Vec3.Z().vdata,
        aabb.points[:, :3],
        blender=True,
        expand=expand,
    )
    self.cam.data.angle_y = cam_params.angle_box.yn
    self.cam.mat_world = cam_params.toworld.data
    return cam_params.toworld


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
