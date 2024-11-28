#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from pydantic import Field
import chdrft.display.control_center as cc
import pandas as pd
from chdrft.display.service import oplt
from chdrft.dsp.datafile import Dataset
from chdrft.display.ui import OpaPlot
from chdrft.config.env import g_env
import glog
import chdrft.utils.rx_helpers as rx_helpers
from chdrft.sim.base import InterpolatedDF
import chdrft.utils.geo as ogeo
from chdrft.struct.base import Box
import polars as pl
import astropy
from astropy import units as u

aq = cmisc.asq_query
from shapely import LineString
import shapely

global flags, cache
flags = None
cache = None
import gpxpy
import folium
import jinja2


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def to_pos(x):
  return astropy.coordinates.WGS84GeodeticRepresentation(
      x[1] * u.deg, x[0] * u.deg, x[2] * u.m, 'WGS84'
  ).to_cartesian().xyz.value


def pl_to_numpy(df, **tsf_funcs):
  return A({k: tsf_funcs.get(k, cmisc.identity)(np.array(df[k].to_list())) for k in df.columns})


def cut(line, distance):
  # Cuts a line in two at a distance from its starting point
  if distance <= 0.0:
    return [None, LineString(line)]
  if distance >= line.length:
    return [LineString(line), None]
  coords = list(line.coords)
  pd = 0
  for i, p in enumerate(coords):
    if i == 0: continue
    pd += shapely.euclidean_distance(p, coords[i - 1])
    if pd == distance:
      return [LineString(coords[:i + 1]), LineString(coords[i:])]
    if pd > distance:
      cp = line.interpolate(distance)
      return [LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])]


class InterFolder(cmisc.PatchedModel):
  data: list = cmisc.Field(default_factory=list)

  @staticmethod
  def Push(self, a):
    if not self.data or self.data[-1][1] < a[0]:
      self.data.append(list(a))
    else:
      self.data[-1][1] = max(self.data[-1][1], a[1])
    return self

  @classmethod
  def Proc(cls, data):
    u = cls()
    return cmisc.functools.reduce(cls.Push, data, u).data


class AugmentedLineString:

  def __init__(self, ls: shapely.LineString):
    self.ls = ls
    self.l = ls.length
    pts = np.array(ls.coords)
    self.n = len(pts)
    self.p0 = pts[:-1]
    self.p1 = pts[1:]
    self.norms = np.linalg.norm(self.p1 - self.p0, axis=1)
    self.csums = np.cumsum(self.norms)
    self.kEps = 1e-9

  def subline(self, interval: tuple[float, float]) -> shapely.LineString:
    a = max(interval[0], 0)
    b = min(interval[1], self.l)
    assert a + self.kEps < b, interval

    st = min(self.n - 1, np.searchsorted(self.csums, a))
    nd = min(self.n - 1, np.searchsorted(self.csums, b))
    points = list(self.p0[st:nd])
    if a + self.kEps < self.csums[st]:
      points.insert(0, self.ls.interpolate(a).coords[0])
    if nd == 0 or self.csums[nd - 1] + self.kEps < b:
      points.append(self.ls.interpolate(b).coords[0])
    return shapely.LineString(points)

  def intersection_convex(self, shape):
    box = Box.FromShapely(shape)
    idx = np.arange(len(self.p0))
    pmin = np.minimum(self.p0, self.p1)
    pmax = np.maximum(self.p0, self.p1)
    bad = np.any((pmin > box.high) | (pmax < box.low), axis=1)
    rem = idx[~bad]
    inters = []
    for i in rem:
      l = shapely.LineString([self.p0[i], self.p1[i]])
      if l.length < self.kEps: continue
      inter = l.intersection(shape)
      if len(inter.coords) < 2: continue
      assert len(inter.coords) == 2
      pa = l.project(shapely.Point(inter.coords[0]))
      pb = l.project(shapely.Point(inter.coords[1]))
      if pa > pb: pa, pb = pb, pa
      if i > 0:
        pa += self.csums[i - 1]
        pb += self.csums[i - 1]
      inters.append((pa - self.kEps, pb + self.kEps))

    return InterFolder.Proc(inters)

  def intersection_convex_sublines(self, shape):
    inters = self.intersection_convex(shape)
    return A(inters=inters, lines=list(map(self.subline, inters)))


def lla2ecef(lla):
  return astropy.coordinates.WGS84GeodeticRepresentation(
      lla[0] * u.deg, lla[1] * u.deg, lla[2] * u.m
  ).to_cartesian().xyz.to(u.m).value


def lla_to_local_coord(lla, base_lla=None, xyz_ecef=False):
  if base_lla is None:
    base_lla = lla.mean(axis=0)

  # right hand basis, need lon lat alt

  p0 = lla2ecef(base_lla)
  if xyz_ecef:
    return A(
        xyz=lla2ecef(lla.T).T,
        lla2xyz=lla2ecef,
        xyz2lla='not implemented here',
        base_lla=None,
    )

  cmul = 1e-5
  diff = [
      lla2ecef(base_lla + [cmul, 0, 0]),
      lla2ecef(base_lla + [0, cmul, 0]),
      lla2ecef(base_lla + [0, 0, 1]),
  ]
  scale = np.array([np.linalg.norm(x - p0) for x in diff]) / (cmul, cmul, 1)

  def lla2xyz(lla):
    if len(lla.shape) == 1:
      return (lla - base_lla) * scale
    return (lla - base_lla.reshape((1, 3))) * np.reshape(scale, (1, 3))

  def xyz2lla(xyz):
    if len(xyz.shape) == 1:
      return xyz / scale + base_lla
    return xyz / np.reshape(scale, (1, 3)) + base_lla.reshape((1, 3))

  return A(
      xyz=lla2xyz(lla),
      lla2xyz=lla2xyz,
      xyz2lla=xyz2lla,
      base_lla=base_lla,
  )


def compute_speed(df, wsize):

  u = df.select(
      re=pl.col('pos').shift(-wsize).forward_fill().backward_fill(),
      re_t=pl.col('t').shift(-wsize).forward_fill().backward_fill(),
      rs=pl.col('pos').shift(wsize).forward_fill().backward_fill(),
      rs_t=pl.col('t').shift(wsize).forward_fill().backward_fill(),
  )
  pos = u.with_row_index('i').explode(
      ['re', 'rs']
  ).group_by('i').agg(diff=(pl.col('re') - pl.col('rs')) / (pl.col('re_t') - pl.col('rs_t')).dt.total_seconds())
  return pos['diff']


class GPXData:

  @classmethod
  def FromFile(cls, fname: str, return_raw=False, **kwargs) -> GPXData:
    gpx_file = open(fname, 'r')
    gpx = gpxpy.parse(gpx_file)

    df = pl.from_records(
        cmisc.asq_query(gpx.get_points_data()).select(
            lambda x: dict(
                lat=x.point.latitude,
                lon=x.point.longitude,
                alt=x.point.elevation,
                t=x.point.time,
            )
        ).to_list(),
        orient='row',
    )

    res = GPXData(df, **kwargs)
    if return_raw:
      return gpx, res
    return res
    #df.t = pd.Series(np.array(df.t.dt.), dtype='object')

  def __init__(self, df, dt=None, xyz_ecef=None, speed_window=None, loop=False, tot_duration=None):
    # t, lon, lat, alt
    lla = np.array([df[x].to_list() for x in ['lon', 'lat', 'alt']]).T
    res = lla_to_local_coord(lla, xyz_ecef=xyz_ecef)
    self.lla2xyz = res.lla2xyz
    self.xyz2lla = res.xyz2lla
    self.lla_conv = res

    if speed_window is None: speed_window = 4
    df = df.with_columns(pos=res.xyz, lla=lla)
    has_t = df['t'].is_not_null().all()

    data = pl_to_numpy(df)
    ls = cmisc.shapely.LineString(np.array([data.lon, data.lat]).T)
    als = AugmentedLineString(ls)
    #dist = np.array([0] + list(als.csums))
    dist = np.cumsum(np.linalg.norm(np.diff(data.pos, axis=0, prepend=data.pos[[0]]), axis=1))
    if not has_t:
      if tot_duration is None:
        tot_duration = np.timedelta64(1, 'D')

      t  = np.datetime64('2000-10-13T00:00:00') +  tot_duration.astype('timedelta64[ms]') * (dist / max(dist))
      df = df.with_columns(t=t)

    if loop:

      ts = df['t'][0]
      tt = df['t'][-1]

      dtt = tt - ts + np.linalg.norm(data.pos[0] - data.pos[-1]) * (tt - ts) / (dist[-1] - dist[0])
      df2 = df.with_columns(t=pl.col('t') + dtt)
      df = pl.concat([df, df2])
      self.__init__(df, dt, xyz_ecef, speed_window, loop=False)
      return


    df = df.with_columns(speed=compute_speed(df, speed_window))



    data = pl_to_numpy(df)
    data.dist= dist
    data.speed_v=np.linalg.norm(data.speed, axis=1)
    data.t_abs = data.t.astype(np.datetime64)
    data.t = (data.t_abs - data.t_abs[0]) / np.timedelta64(1, 's')

    df = df.with_columns(speed_v=data.speed_v, dist=data.dist)


    self.idf_t = InterpolatedDF(data, index=data.t)
    self.idf_dist = InterpolatedDF(data, index=data.dist)

    self.df = df
    self.data = data
    self.ls = ls
    self.als = als


class LatLngPush(folium.MacroElement):

  _template = jinja2.Template(
      """
            {% macro script(this, kwargs) %}
                function latLngPop(e) {
                
                fetch( {{ this.endpoint|tojson }},
                {
                  method: "POST",
                  body: JSON.stringify({
                  ll: e.latlng, 
                  bounds: e.sourceTarget.getBounds(),
                  size: e.sourceTarget.getSize(),
                  modifiers: Object.fromEntries("Shift Meta Alt Control".split(" ").map((x) => [x, e.originalEvent.getModifierState(x)])),
                }),
                    });

                }
                {{this._parent.get_name()}}.on('click', latLngPop);
            {% endmacro %}
            """
  )  # noqa

  def __init__(self, endpoint):
    super().__init__()
    self._name = "LatLngPush"
    self.endpoint = endpoint


class ActionController(cmisc.PatchedModel):

  gpx: GPXData
  fh: cc.FoliumHelper
  opp: OpaPlot = None
  sx: cc.RxServer = None

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    obs_push_ll = self.sx.add(
        cc.ObsEntry(name='get_latlng', f=rx_helpers.WrapRX(rx_helpers.rx.Subject())), push=True
    )
    LatLngPush(obs_push_ll.full_path).add_to(self.fh.m)
    obs_push_ll.obs.f.filter(lambda x: x.modifiers.Meta
                            ).subscribe_safe(self.handle_ll_action, qt_sig=True)

  def handle_ll_action(self, v):
    vb = Box(
        low=(v.bounds._southWest.lng, v.bounds._southWest.lat),
        high=(v.bounds._northEast.lng, v.bounds._northEast.lat)
    )
    view_size = (v.size.x, v.size.y)

    pix_size_pointer = 100
    point_size = vb.size / view_size * pix_size_pointer
    b1 = Box(center=(v.ll.lng, v.ll.lat), size=point_size).shapely
    u = self.gpx.als.intersection_convex_sublines(b1)

    self.fh.data['ll_action'] = [dict(feature=x, style=dict(color='red')) for x in u.lines]
    if self.opp:
      for l, h in u.inters:
        self.opp.regions.add_region(self.gpx.idf_dist(l).t, self.gpx.idf_dist(h).t)
