#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, flatten
from vispy.geometry.meshdata import MeshData
import glog
import numpy as np


def triangles_to_xyz_ijk(triangles):
  triangles = list(triangles)
  res = Attributize()
  points = flatten(triangles, depth=1)
  res.x = list([e[0] for e in points])
  res.y = list([e[1] for e in points])
  res.z = list([e[2] for e in points])
  res.i = list(range(0, 3 * len(triangles), 3))
  res.j = list(range(1, 3 * len(triangles), 3))
  res.k = list(range(2, 3 * len(triangles), 3))
  return res


def stl_to_meshdata(stl_data):
  vertices = np.array(flatten([x.vx for x in stl_data], depth=1))
  faces = np.reshape(range(vertices.shape[0]), (vertices.shape[0] // 3, 3))
  return MeshData(vertices=vertices, faces=faces)
