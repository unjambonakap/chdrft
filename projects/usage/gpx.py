#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import chdrft.utils.Z as Z
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

#glog.setLevel('DEBUG')

aq = cmisc.asq_query
from shapely import LineString
import shapely

global flags, cache
flags = None
cache = None
import chdrft.projects.gpx as  ogpx

g_env.run_magic(force=True)


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


#u = Z.FileFormatHelper.Read('/tmp/data.pickle')
##Z.FileFormatHelper.Write('/tmp/data.pickle', A(v=v, ls=ls))
#v = u.v
#vb = Z.Box(
#    low=(v.bounds._southWest.lng, v.bounds._southWest.lat),
#    high=(v.bounds._northEast.lng, v.bounds._northEast.lat)
#)
#view_size = (v.size.x, v.size.y)
#
#pix_size_pointer = 100
#point_size = vb.size / view_size * pix_size_pointer
#b1 = Z.Box(center=(v.ll.lng, v.ll.lat), size=point_size).shapely

print(g_env.qt_imports.QApplication.instance())

#gpx = ogpx.GPXData.FromFile(cmisc.proc_path('~/Downloads/Galibier_revanche_.gpx'))
raw, gpx = ogpx.GPXData.FromFile(cmisc.proc_path(r'~/Downloads/gr54-depuis-le-bourg-d-oisans.gpx') , loop=True, return_raw=True, tot_duration=np.timedelta64(1, 'h'))
#gpx = ogpx.GPXData.FromFile(cmisc.proc_path(r'~/Downloads/gr54-tour-de-loisans-et-des-ecrins-depuis-la-grave-par-le-gr54c-et-les-variantes-alpines.gpx'))
sx = cc.RxServer()
sx.start()

print(g_env.qt_imports.QApplication.instance())
fh = cc.FoliumHelper()
fh.create_folium((gpx.data.lon[0], gpx.data.lat[0]), sat=False)

fh.data['track'] = gpx.ls

fh.ev.on_next(None)
#fh.m.show_in_browser()
g_env.create_app()

idx = gpx.data.dist / 1000

dx = Dataset(x=idx, y=gpx.data.alt, name='z')
opp = OpaPlot(dx, legend=1)

ac = ogpx.ActionController(gpx=gpx, fh=fh, sx=sx, opp=opp)
frt = cc.FoliumRTHelper(fh=fh)
frt.setup(sx)
fh.setup()
gpx.data.dplus = np.cumsum(np.maximum(np.diff(gpx.data.alt, prepend=[gpx.data.alt[0]]), 0))
gpx.data.dminus = np.cumsum(np.maximum(-np.diff(gpx.data.alt, prepend=[gpx.data.alt[0]]), 0))
gpx.df = gpx.df.with_columns(dplus=gpx.data.dplus, dminus=gpx.data.dminus)

oplt.create_window()
#r.obs.subscribe_safe(
gw = oplt.find_window().gw
gw.add(fh.generate_widget())
gw.add(opp, label='data')
opp.add_plot(Dataset(x=idx, y=gpx.data.t, name='t'))

def get_idx(t):
  return min(len(idx) - 1, np.searchsorted(idx, t))
import chdrft.display.ui as oui
class ObservableDict(dict):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._obs = rx_helpers.rx.subject.BehaviorSubject(self)

  def __setitem__(self, a, v):
    super().__setitem__(a, v)
    self._obs.on_next(self)

  def update(self, a):
    super().update(a)
    self._obs.on_next(self)
    return self

odict = ObservableDict()
wx = gw.add(oui.MetricWidget.Make(obs=odict._obs)).w
assert 0

#%%
r = opp.regions.add_region(0, 0)




def t1(x):
  a = get_idx(x[0].x())
  b = get_idx(x[1].x())
  d = gpx.data
  diff = {k:d[k][b] - d[k][a] for k in d.keys()}
  odict['diff']   = diff
  fh.ev.on_next([cmisc.shapely.Point([gpx.data.lon[i], gpx.data.lat[i]]) for i in (a, b)])
  fh.ev.set_value([cmisc.shapely.Point([gpx.data.lon[i], gpx.data.lat[i]]) for i in (a, b)])


r.obs.subscribe_safe(t1)

#%%

l  =opp.sampler.marks.add_line(0, movable=True)
def t2(x):
  a = get_idx(x.x())
  fh.ev.on_next([cmisc.shapely.Point([gpx.data.lon[i], gpx.data.lat[i]]) for i in (a, )])
  odict['mark']   = gpx.df.row(a, named=True)

l.obs.subscribe_safe(t2)
assert 0

#%%

def link_line_obs(e_t, line):
  e_t.subscribe_safe(line.setPos, qt_sig=True)


link_line_obs(e_t, sh.opp.sampler.marks.add_line(0))
sh.at_t.subscribe_safe(cb)
#%%


def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
