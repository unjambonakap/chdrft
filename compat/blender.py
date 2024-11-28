#!/usr/bin/env python

from dataclasses import dataclass
import sys
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.opa_types import *

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


@dataclass
class BlenderCollections(dict):

  def new(self, name):
    res = self[name] = BlenderCollection(name=name)
    return res


class BlenderObject(A):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.data = None
    self.muproperties = A()
    self.mumodelprops = A()
    self.objects = BlenderList()
    self.vertex_groups= BlenderObjects()

class BlenderMesh(A):

  def __init__(self, name, **kwargs):
    super().__init__(**kwargs)
    self.name = name
    self.uv_layers = A(new=lambda name=None:A(data=A()))
    self.normals_split_custom_set= lambda _: None
    self.materials =  []
  def from_pydata(self, vertices, loops, faces):
    self.vertices = vertices
    self.loops = loops
    self.faces = faces

class BlenderList(list):

  def link(self, a):
    self.append(a)


class BlenderCollection(A):

  def __init__(self, name):
    self.name = name
    self.children = BlenderList()
    self.objects = BlenderList()
    self.mumodelprops = A()


@dataclass
class BlenderObjects(list):

  def new(self, name, base=None):
    res = BlenderObject(name=name)
    self.append(res)
    return res


class MatrixBuilder:
  def __call__(self, x):
    return np.array(x)
  def Identity(self, n):
    return np.identity(n)
  def inverted(self, x):
    return np.linalg.inv(x)


bpy_types = A(
    ArmatureModifier=1,
    PropertyGroup=A,
    Object=BlenderObject,
    Panel=A,
  Mesh=BlenderMesh,
    Collection=BlenderCollection,
    Scene=A,
    Camera=A,
)
scene_obj = A(collection=BlenderCollection('main'))
bpy_obj = A(
  FAKE=1,
    data=A(
        collections=BlenderCollections(),
        objects=BlenderObjects(),
        images=A(
            load=lambda *args:
            A(img_data=args, muimageprop=A(), pixels=[], colorspace_settings=A())
        ), 
        meshes=A(new=BlenderMesh),
      armatures=A(new=lambda name:A(name=name, edit_bones=BlenderObjects())),
    ),
    types=bpy_types,
    context=A(scene=scene_obj, layer_collection=A(collection=BlenderCollection('layer'))),
    props=A(
        BoolVectorProperty=A,
        BoolProperty=A,
        FloatProperty=A,
        StringProperty=A,
        EnumProperty=A,
        PointerProperty=A,
        IntProperty=A,
        FloatVectorProperty=A
    )
)
math_utils_obj = A(Vector=np.array, Quaternion=np.array, Matrix=MatrixBuilder())

bmesh_obj = A()


def register():
  sys.modules['bpy'] = bpy_obj
  sys.modules['bpy_types'] = bpy_types
  sys.modules['bpy.props'] = bpy_obj.props
  sys.modules['mathutils'] = math_utils_obj
  sys.modules['bmesh'] = bmesh_obj


def test(ctx):
  register()
  from io_object_mu.import_craft.import_craft_lib import import_craft
  res = import_craft(
      '/home/benoit/Downloads/tmp/Ships/VAB/SpaceX Falcon 9 Block 5.craft',
      '/home/benoit/work/links/Kerbal Space Program/GameData'
  )
  print(res)


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
