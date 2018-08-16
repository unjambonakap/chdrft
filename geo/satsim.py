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


class TileGetter(ExitStack):
  def __init__(self, output_dir, verbose=False, nworker=10):
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

  def get_tile(self, e):
    x=e['x']
    y=e['y']
    z=e['z']

    url=f'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
    if self.verbose:
      print(url)
    out_fname=os.path.join(self.output_dir, f'tile_{z}_{x}_{y}.jpg')
    out_fname2=os.path.join(self.output_dir, f'tile_{z}_{x}_{y}.png')
    if os.path.exists(out_fname2): return

    res = requests.get(url, stream=True)
    with open(out_fname, 'wb') as f:
      shutil.copyfileobj(res.raw, f)
    sp.check_output(['convert', out_fname, out_fname2])
