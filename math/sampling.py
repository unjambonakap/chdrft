#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import numpy as np
import math, sys, os
from chdrft.utils.types import *
import chdrft.utils.geo as geo_utils
import chdrft.struct.base as opa_struct
import shapely.geometry as geo
import shapely.ops as geo_ops

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  pass

def stratified_grid_sampling(nx, ny, overlap=0):
  xs = 1 / nx * (1 + overlap) * 0.5
  ys = 1 / ny * (1 + overlap) * 0.5
  X, Y = np.meshgrid(np.linspace(xs, 1-xs, nx), np.linspace(ys, 1-ys, ny))
  X += np.random.uniform(-xs, xs, (nx, ny))
  Y += np.random.uniform(-ys, ys, (nx, ny))
  return np.stack((X.ravel(), Y.ravel()), axis=-1)

def sample_grid(n):
  nn = int(math.ceil(n**0.5))
  return stratified_grid_sampling(nn, nn)[:n,:]
def sample_triangle(n):
  v= sample_grid(n)
  return np.where((np.sum(v, axis=1) > 1).reshape(-1, 1), np.array([(1,1)]) - v, v)

def sample(item, n):
  if isinstance(item, geo.Polygon) and len(item.exterior.coords) > 4:
    trs = geo_ops.triangulate(item)
    return sample_collection(trs, n)

  if isinstance(item, list):
    return sample_collection(item, n)

  if isinstance(item, opa_struct.Box):
    sx = sample_grid(n)
    return item.as_gen_box().get(sx)

  gb = opa_struct.GenBox.FromPoints(*geo_utils.get_points(item))
  sx = sample_triangle(n)
  return gb.get(sx)

def sample_collection(items, nsamples):
  tot_area = sum(x.area for x in items)
  samples = []
  probs = []
  for item in items:
    cur = sample(item, nsamples)
    samples.extend(cur)
    probs.extend([item.area / tot_area / nsamples] * nsamples)
  sel = np.random.choice(len(samples), size=nsamples, replace=False, p=probs)
  return np.array(samples)[sel]


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
