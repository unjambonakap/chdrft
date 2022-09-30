#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
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

global flags, cache
flags = None
cache = None

def read_img_from_buf(buf):
  img = cv2.imdecode(np.asarray(bytearray(io.BytesIO(buf).read()), dtype=np.uint8), -1)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def args(parser):
  clist = CmdsList().add(test)
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
    url=f'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
    if self.verbose:
      print(url)
    res = requests.get(url, stream=True)
    return res

  @Cachable.cached()
  def get_tile(self, x=None, y=None, z=None):
    glog.info(f'Querying tile x={x}, y={y}, z={z}')
    res = self.get_tile_base(x, y ,z)
    rimg = np.array([[[255,0,0]]], dtype=np.uint8)
    read = res.raw.read()
    buf = np.frombuffer(read, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


  def get_tile_file(self, x=None, y=None, z=None):

    out_fname=os.path.join(self.output_dir, f'tile_{z}_{x}_{y}.jpg')
    out_fname2=os.path.join(self.output_dir, f'tile_{z}_{x}_{y}.png')
    if os.path.exists(out_fname2): return
    res = self.get_tile_base(x, y ,z)

    with open(out_fname, 'wb') as f:
      shutil.copyfileobj(res.raw, f)
    sp.check_output(['convert', out_fname, out_fname2])


def test(ctx):
  tg = TileGetter()
  tl =tg.get_tile(0,0,0)
  gl = opa_cache.Proxifier(Nominatim, user_agent='VGYV7gGlNoWapA==')
  obj = gl.geocode('Paris')
  print(obj)
  print(len(tl))


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
