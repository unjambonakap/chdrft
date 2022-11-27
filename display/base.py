#!/usr/bin/env python

import os
from chdrft.config.env import g_env

qt_imports = g_env.get_qt_imports_lazy()
import pyqtgraph as pg
from chdrft.config.env import g_env, qt_imports
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
import glog
import re
import chdrft.struct.base as opa_struct
import sys
import numpy as np
import chdrft.utils.geo as geo_utils
import pymap3d
import meshio


class Consts:
  EARTH_ELLIPSOID = pymap3d.Ellipsoid('wgs84')
  MOON_ELLIPSOID = pymap3d.Ellipsoid('moon')


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
    return Quad(opa_struct.g_unit_box, 0)


class TMSQuad:
  LMAX = 85.05113
  MAX_DEPTH = 20

  def __init__(self, x, y, z, u2s, parent=None):
    self.x = x
    self.y = y
    self.z = z
    self.u2s = u2s
    self._children = None
    self.parent = parent
    self.depth = z

  @property
  def children(self):
    if self._children is None:
      self._children = []
      if self.z + 1 < TMSQuad.MAX_DEPTH:
        for i in range(4):
          self._children.append(
              TMSQuad(
                  2 * self.x + (i & 1), 2 * self.y + (i >> 1), self.z + 1, self.u2s, parent=self
              )
          )
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
        np.stack(pymap3d.geodetic2ecef(p[:, 1], p[:, 0], 0, ell=Consts.EARTH_ELLIPSOID), axis=-1) *
        self.u2s
    )

  @property
  def xyz(self):
    return self.x, self.y, self.z

  def tile(self, tg):
    return tg.get_tile(*self.xyz)

  @staticmethod
  def Root(u2s):
    return TMSQuad(0, 0, 0, u2s)

  def __str__(self):
    return f'xyz={self.xyz}'


class TriangleActorBase:

  def __init__(self):
    self.points = []
    self.trs = []
    self.tex_coords = []
    self.npoints = 0
    self.obj = None
    self.name = None

  def add_points(self, pts):
    tb = []
    for pt in pts:
      self.points.append(pt)
      tb.append(self.npoints)
      self.npoints += 1
    return np.array(tb)

  def add_quad(self, pts, tex_coords=[]):
    if isinstance(pts, opa_struct.Quad): pts = pts.pts
    pids = self.add_points(pts)
    for tc in tex_coords:
      self.tex_coords.append(tc)

    self.push_quad(pids)
    return self

  def add_meshio(self, obj=None, path=None):

    if obj is None:
      obj = meshio.read(path)
    self.points.extend(obj.points)

    for x in obj.cells:
      self.trs.extend(x.data + self.npoints)
    self.npoints += len(obj.points)

  def add_triangle(self, pts, tex_coords=[]):
    pids = self.add_points(pts)
    for tc in tex_coords:
      self.tex_coords.add(tc)
    self.push_triangle(pids)
    return self

  def push_triangle(self, pids):
    self.trs.append(pids)

  def push_quad(self, pids, pow2=False):
    sel = opa_struct.Quad.IdsPow2 if pow2 else opa_struct.Quad.Ids
    self.push_triangle(pids[sel[0]])
    self.push_triangle(pids[sel[1]])

  def full_quad(self, pts, uv=None):
    if uv is None: uv = opa_struct.g_unit_box.quad.pts
    return self.add_quad(pts, uv)

  def full_quad_yinv(self, pts):
    return self.add_quad(pts, np.array([(0, 1), (1, 1), (1, 0), (0, 0)]))

  def build(self, tex=None):
    self.tex = self._norm_tex(tex)
    self.points = np.array(self.points)
    self.tex_coords = np.array(self.tex_coords)
    self.trs = np.array(self.trs)
    self.obj = self._build_impl(self.tex)

  def push_peer(self, peer: "TriangleActorBase"):
    # TODO: decide what to do with tex (would need to merge the texture)
    base = self.npoints
    pts_ids = self.add_points(peer.points)
    self.trs.extend(np.array(peer.trs) + base)
    self.tex_coords.extend(peer.tex_coords)

  @classmethod
  def BuildFrom(cls, peer: "TriangleActorBase", **kwargs) -> "TriangleActorBase":
    self = cls(**kwargs)
    self.tex = peer.tex
    self.points = peer.points
    self.tex_coords = peer.tex_coords
    self.trs = peer.trs
    self.obj = self._build_impl(self.tex)
    return self

  def _norm_tex(self, tex):
    pass

  def _build_impl(self, tex):
    pass

  def tr_line(self, tr):
    return [self.points[tr[i % 3]] for i in range(4)]

  @property
  def lines(self):
    return [self.tr_line(tr) for tr in self.trs]

  @property
  def vispy_data(self):
    return A(points=self.points, lines=self.lines, mesh=self.mesh_data, conf=A(mode='3D'))

  @property
  def mesh_data(self):
    from vispy.geometry.meshdata import MeshData
    assert len(self.points) > 0
    return MeshData(vertices=self.points, faces=self.trs)
