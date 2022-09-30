#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import numpy as np
import re
from qgis.core import *

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class AsciiGrid:
  def __init__(self, fname):
    self.fname = fname
    self.headers = self.parse_header()

  def parse_header(self):
    with open(self.fname, 'r') as f:
      lines = [f.readline().strip() for _ in range(6)]
    pairs = [re.split('\s+', x) for x in lines]
    print(pairs)
    kv = cmisc.A({k:float(v) for k,v in pairs})
    box = Z.Box(low=(kv.xllcorner, kv.yllcorner), size=np.array([kv.ncols, kv.nrows])*kv.cellsize)
    kv.box = box
    return kv

class RefSysTypes(Z.Enum):
  Lambert93 = 'EPSG:2154'
  TileService = 'EPSG:3857'

class RefSys:

  def __init__(self, qgs):
    self.qgs = QgsCoordinateReferenceSystem(str(qgs))
    ctx = QgsCoordinateTransformContext()
    self._transformdb = cmisc.A(handler=lambda x: (QgsCoordinateTransform(self.qgs, x.qgs, ctx), True))

  @staticmethod
  def Make(x):
    if isinstance(x, RefSysTypes): x = x.value
    if isinstance(x, str): return RefSys(x)
    return x

  def convert(self, obj, dest):
    if dest == self: return obj
    tsf = self._transformdb[dest]
    print(self.qgs, dest.qgs)
    mapper = lambda x,y: list(tsf.transform(QgsPointXY(x,y)))
    return Z.geo_ops.transform(mapper, obj)


class MappedObject:
  def __init__(self, refsys, obj):
    obj = Z.to_shapely(obj)
    self.refsys = RefSys.Make(refsys)
    self.obj = obj
  def to_sys(self, nsys):
    nsys = RefSys.Make(nsys)
    return MappedObject(nsys, self.refsys.convert(self.obj, nsys))



def test(ctx):
  QgsApplication.setPrefixPath("", True)
  qgs = QgsApplication([], False)
  qgs.initQgis()

  tfile = '/home/benoit/work/data/RGEALTI_1-0_EXT_1M_PPK_ASC_3/RGEALTI_2-0_1M_ASC_LAMB93-IGN69_D005_2020-01-29/RGEALTI/1_DONNEES_LIVRAISON_2021-01-00150/RGEALTI_MNT_1M_ASC_LAMB93_IGN69_D005_20210118//RGEALTI_FXX_0892_6372_MNT_LAMB93_IGN69.asc'
  ag = AsciiGrid(tfile)
  b = ag.headers.box.shapely
  l93 = RefSys.Make(RefSysTypes.Lambert93)
  ts = RefSys.Make(RefSysTypes.TileService)
  bi = MappedObject(l93, b)
  bo = bi.to_sys(ts)
  print(bo.obj)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
