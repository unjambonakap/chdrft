#!/usr/bin/env python

from __future__ import annotations
from bpy_extras.image_utils import load_image
from bpy_extras.node_shader_utils import PrincipledBSDFWrapper
from chdrft.cmds import CmdsList
from chdrft.display.base import TriangleActorBase
from chdrft.dsp.image import ImageData
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
from chdrft.utils.types import *
from chdrft.sim.rb.base import Vec3, Transform, AABB
from scipy.spatial.transform import Rotation as R
import bpy
import chdrft.utils.misc as cmisc
import cv2
import glog
import numpy as np
import tempfile
import chdrft.utils.geo as geo_utils
from mathutils import Matrix

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def blender_look_at(dz, dy):
  dz = -np.array(dz)
  dy = geo_utils.make_orth_norm(dy, dz)
  dx = np.cross(dy, dz)

  m = np.matrix([dx, dy, dz]).T
  rx = R.from_matrix(m)
  return blender_quat(rx)


def blender_quat(r):
  r = r.as_quat()
  return [r[3]] + list(r[:3])


def clear_scene():
  col = bpy.context.scene.collection
  for x in list(bpy.data.collections.values()):
    bpy.data.collections.remove(x)

  clear_collection(col)


def clear_collection(col):
  for x in list(col.children):
    col.children.unlink(x)
  for x in list(col.objects):
    col.objects.unlink(x)


class BlenderTriangleActor(TriangleActorBase):

  def __init__(self, name="object"):
    super().__init__()
    self._name = name

  def _norm_tex(self, tex):
    return tex

  @property
  def blender_obj(self) -> bpy.types.Object:
    return self.obj

  def _build_impl(self, tex):
    edges = []
    faces = []
    new_mesh = bpy.data.meshes.new(f'{self._name}_mesh')
    pts = np.array(self.points)
    center = np.mean(pts, axis=0)
    pts -= center
    new_mesh.from_pydata(pts, edges, list(self.trs))
    new_mesh.update()
    # make object from mesh
    new_object = bpy.data.objects.new(self._name, new_mesh)
    new_object.location = center

    # make collection
    if len(self.tex_coords):
      new_object.data.uv_layers.new(name='uv_map')
      for fid, face in enumerate(new_object.data.polygons):
        for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
          new_object.data.uv_layers.active.data[loop_idx].uv = self.tex_coords[vert_idx]
    return new_object


def create_material_for_obj(obj, img, name):
  if isinstance(img, str):
    tex = bpy.data.images.load(img)
  else:
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      cv2.imwrite(f.name, img)
      f.flush()
      tex = bpy.data.images.load(f.name)
      tex.pack()
  mat = bpy.data.materials.new(name)
  ma_wrap = PrincipledBSDFWrapper(mat, is_readonly=False)
  ma_wrap.base_color_texture.image = tex
  mat.specular_intensity = 0
  mat.roughness = 0.5
  obj.active_material = mat


def make_group(prefix, collection, items) -> bpy.types.Object:
  main = bpy.data.objects.new(prefix, None)
  main_col = bpy.data.collections.new(f'{prefix}_col')
  collection.children.link(main_col)
  main_col.objects.link(main)
  for o in items:
    o.parent = main
    main_col.objects.link(o)
  return main


class MeshExporter:

  def __init__(self, bl=lambda _: False):
    self.data = []
    self.bl = bl

  def walk(self, o, mat, chain=[]):
    chain = list(chain)
    if isinstance(o, bpy.types.Collection):
      for x in cmisc.itertools.chain(o.children, o.objects):
        if x.parent is None: self.walk(x, mat, chain)
      return

    if self.bl(o.name): return
    chain.append(o.name)

    mulm = Matrix.Translation(o.location) @ o.rotation_quaternion.to_matrix().to_4x4()
    mat = mat.copy() @ mulm

    for x in cmisc.itertools.chain(o.children):
      self.walk(x, mat, chain)

    try:
      m = o.to_mesh()
      m.transform(mat)

      x = get_mesh_data(m)
      x.name = o.name
      x.chain = chain
      x.mat = mat
      self.data.append(x)

      o.to_mesh_clear()

    except RuntimeError:
      pass

    #mat = mat @ Matrix.Translation(o.location) @ o.rotation_quaternion.to_matrix().to_4x4()
    #if len(o.children) and 'KK.SPX.Merlin1DV+' in chain:
    #    print(chain, o.name, o.location, o.scale, o.rotation_quaternion, mat)
    #mat = mat @ o.matrix_local.inverted_safe()
    #for x in o.children:
    #    self.walk(x, mat, chain)
    if o.instance_type == 'COLLECTION':
      self.walk(o.instance_collection, mat, chain)


def get_mesh_data(m):
  vl = []
  m.calc_loop_triangles()
  faces = []
  vx = m.vertices
  for x in vx:
    vl.append(list(x.co.copy()))
  for tri in m.loop_triangles:
    faces.append(list(tri.vertices))
  return A(vertices=np.array(vl), faces=np.array(faces))


def merge_meshdata(xl):
  vertices = []
  faces = []
  for x in xl:
    n = len(vertices)
    vertices.extend(x.vertices)
    faces.extend(x.faces + n)
  return A(vertices=vertices, faces=faces)


def blender_obj_to_meshio(obj, bl_func=lambda _: False):
  me = MeshExporter(bl=bl_func)
  me.walk(obj, Matrix())
  e = merge_meshdata(me.data)
  import meshio
  m = meshio.Mesh(e.vertices, [('triangle', e.faces)])
  return m


class BlenderObjWrapper:

  def __init__(self, obj):
    self.internal: bpy.types.Object = obj

  @property
  def data(self):
    return self.internal.data

  @property
  def mat_world(self):
    return np.array(self.internal.matrix_world)

  @property
  def wl(self) -> Transform:
    return Transform(self.mat_world)

  @wl.setter
  def wl(self, v: Transform):
    self.mat_word = v.data

  @mat_world.setter
  def mat_world(self, val):
    self.internal.matrix_world = val.T

  @property
  def mat_local(self):
    if self.internal.parent: return np.array(self.internal.matrix_local)
    return self.mat_world

  @mat_world.setter
  def mat_local(self, val):
    if self.internal.parent: self.internal.matrix_local = val.T
    else: self.mat_world = val

  @property
  def children(self) -> list[BlenderObjWrapper]:
    return [BlenderObjWrapper(x) for x in self.internal.children]

  @property
  def aabb_w(self) -> AABB:
    start = AABB.FromPoints(np.array(self.internal.bound_box))
    return self.wl @ cmisc.functools.reduce(AABB.__or__, [x.aabb_w for x in self.children], start)


class KeyframeObjData(cmisc.PatchedModel):
  obj: BlenderObjWrapper
  wl: np.ndarray

  def propagate(self):
    self.obj.mat_local = np.identity(4)
    rot = self.wl[:3, :3]

    self.obj.internal.rotation_mode = 'QUATERNION'
    self.obj.internal.rotation_quaternion = blender_quat(R.from_matrix(rot))
    self.obj.internal.location = self.wl[:3, 3]


class AnimationSceneHelper(cmisc.PatchedModel):
  fid: int = 0
  frame_step: float = 1

  def clear(self):
    bpy.context.scene.animation_data_clear()
    for o in bpy.context.scene.objects:
      o.animation_data_clear()

  def start(self):
    self.clear()
    self.fid = 0
    bpy.context.scene.frame_set(0)

  def push(self, items: list[KeyframeObjData]):
    for y in items:
      y.propagate()

      y.obj.internal.keyframe_insert(data_path='location', frame=self.fid)
      y.obj.internal.keyframe_insert(data_path='rotation_quaternion', frame=self.fid)
    self.fid += self.frame_step

  def finish(self):
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = int(self.fid)
    bpy.context.scene.frame_set(1)


def test(ctx):
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
