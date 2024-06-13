#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import csv
import requests
import os
import shutil
import subprocess as sp
from contextlib import ExitStack
from chdrft.utils.cache import Cachable
import chdrft.utils.cache as opa_cache
from geopy.geocoders import Nominatim
import cv2
import numpy as np
from osgeo import gdal
from chdrft.dsp.image import ImageData
from chdrft.struct.base import Box
from osgeo import gdal
from osgeo.osr import SpatialReference, CoordinateTransformation
from chdrft.config.creds import get_creds
from requests_cache import CachedSession
import zipfile
import io
import uuid
from chdrft.utils import K

global flags, cache
flags = None
cache = None


def read_img_from_buf(buf):
  img = cv2.imdecode(np.asarray(bytearray(io.BytesIO(buf).read()), dtype=np.uint8), -1)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def args(parser):
  clist = CmdsList()
  parser.add_argument('--infile')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class TileGetter(ExitStack):

  def __init__(self, output_dir=None, verbose=False, nworker=10):
    Cachable.ConfigureCache(self, fields=[])
    super().__init__()
    self.nworker = nworker
    self.verbose = verbose
    self.output_dir = output_dir
    self.exe = cmisc.OpaExecutor(nworker)

  def __enter__(self):
    super().__enter__()
    self.enter_context(self.exe)
    return self

  def do_jobs(self, jobs, wait=False):
    self.exe.map(self.get_tile, jobs)
    if wait: self.exe.do_wait()

  def get_tile_base(self, x=None, y=None, z=None):
    url = f'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
    if self.verbose:
      print(url)
    res = requests.get(url, stream=True)
    return res

  @Cachable.cached()
  def get_tile(self, x=None, y=None, z=None) -> np.ndarray:
    glog.info(f'Querying tile x={x}, y={y}, z={z}')
    res = self.get_tile_base(x, y, z)
    rimg = np.array([[[255, 0, 0]]], dtype=np.uint8)
    read = res.raw.read()
    buf = np.frombuffer(read, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None: return np.zeros((256, 256, 3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  def get_tile_file(self, x=None, y=None, z=None):

    out_fname = os.path.join(self.output_dir, f'tile_{z}_{x}_{y}.jpg')
    out_fname2 = os.path.join(self.output_dir, f'tile_{z}_{x}_{y}.png')
    if os.path.exists(out_fname2): return
    res = self.get_tile_base(x, y, z)

    with open(out_fname, 'wb') as f:
      shutil.copyfileobj(res.raw, f)
    sp.check_output(['convert', out_fname, out_fname2])


gl = opa_cache.Proxifier(Nominatim, user_agent='VGYV7gGlNoWapA==')


def test(ctx):
  tg = TileGetter()
  tl = tg.get_tile(0, 0, 0)
  obj = gl.geocode('Paris')
  print(obj)
  print(len(tl))


import logging
from http.client import HTTPConnection  # py3

log = logging.getLogger('urllib3')
log.setLevel(logging.DEBUG)

# logging from urllib3 to console
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

# print statements from `http.client.HTTPConnection` to console/stdout
HTTPConnection.debuglevel = 1


def gdal_map(ds):
  src = SpatialReference(ds.GetProjectionRef())
  dest = SpatialReference()
  dest.SetWellKnownGeogCS('WGS84')
  ct = CoordinateTransformation(src, dest)
  ict = ct.GetInverse()
  tsf = ds.GetGeoTransform()

  pix2geo = np.array([[tsf[1], tsf[2]], [tsf[4], tsf[5]]])
  c = np.array([tsf[0], tsf[3]])
  geo2pix = np.linalg.inv(pix2geo)

  def w2l(pos):
    tmp = ct.TransformPoint(pos[0], pos[1], 0)[:2]
    return geo2pix @ (tmp - c)

  def l2w(pos):
    tmp = pix2geo @ pos + c
    return ict.TransformPoint(tmp[0], tmp[1], 0)

  return A(w2l=w2l, l2w=l2w)


def read_gdal(fname=None, content=None, dim=None, scale_factor=1):
  if content is not None:
    mmap_name = "/vsimem/" + fname  + '.hgt'
    #gdal.AllRegister();
    gdal.FileFromMemBuffer(mmap_name, content)
    fname = mmap_name

  ds = gdal.Open(fname)
  args = A()
  if dim is not None:
    args.buf_xsize = dim[0]
    args.buf_ysize = dim[1]
  img = np.array(ds.GetRasterBand(1).ReadAsArray(**args)).T
  ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
  b = Box(low=(ulx, uly), size=(ds.RasterXSize * xres, ds.RasterYSize * yres))
  print('fuuu', b, b.xn, b.yn)
  raw = img
  if b.xn < 0:
    b = b.make_new(yr=b.yr, xr=(b.xh, b.xl))
    img = img[::-1]
  if b.yn < 0:
    b = b.make_new(xr=b.xr, yr=(b.yh, b.yl))
    img = img[:, ::-1]

  i = ImageData(img * scale_factor, box=b, yx=0)
  return A(img=i, raw=raw,**gdal_map(ds))


kCacheDir = '~/data/cache_chdrft'


class SRTMGL1Getter(cmisc.ExitStack):

  def __init__(self):
    super().__init__()
    self.session = CachedSession(f'{kCacheDir}/strm', backend='filesystem')
    self.session.should_strip_auth = lambda *args: False
    self.session.auth = get_creds().get('urs_earthdata', as_auth=True)

  def __enter__(self):
    self.enter_context(self.session)
    return self

  def query(self, lat, lon):
    url = f'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/N{lat:02d}E{lon:03d}.SRTMGL1.hgt.zip'
    res = self.session.get(url, auth=self.session.auth)

    data = bytes(res.content)
    zp = zipfile.ZipFile(io.BytesIO(data))
    with open('/tmp/test.zip', 'wb') as fx:
      fx.write(data)
    tg = zp.namelist()[0]
    return A(fname=tg, content=zp.open(tg, 'r').read())


def test2(ctx):
  res = read_gdal(ctx.infile)
  print(res.img.img_box)

  print(res.l2w([0, 0]))
  print(res.l2w(res.img.img_box.dim / 2))
  print(res.w2l([14.5, 37.5]))
  with SRTMGL1Getter() as gx:
    u = gx.query(52, 4)
    data = read_gdal(fname=u.fname, content=u.content)
    tg = [4.8943341, 52.3242155] # lon lat
    tg = [4.88922, 52.32638] # lon lat
    tg = [4.68224, 52.70807] # lon lat
    pix = data.w2l(tg)
    u =tuple(np.round(pix).astype(int))
    print(data.raw[u])
    print(np.max(data.raw))
    K.oplt.plot(A(images=[data.img]))
    input()


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
