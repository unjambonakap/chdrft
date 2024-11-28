#!/usr/bin/env python

from chdrft.config.env import g_env

qt_imports = g_env.get_qt_imports_lazy()
from chdrft.config.env import g_env
from chdrft.utils.misc import Attributize as A
import chdrft.struct.base as opa_struct
import numpy as np
import meshio



class TriangleActorBase:

  def __init__(self):
    self.points = []
    self.trs = []
    self.tex_coords = []
    self.npoints = 0
    self.obj = None
    self.name = None

  def map_points(self, mp):
    self.points = mp(np.array(self.points))

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
  def BuildFrom(cls, peer: "TriangleActorBase", scale: float = None, **kwargs) -> "TriangleActorBase":
    self = cls(**kwargs)
    self.tex = peer.tex
    self.points = peer.points
    if scale is not None:
      self.points[:,:3] = self.points[:,:3] * scale
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
