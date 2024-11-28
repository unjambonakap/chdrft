#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.opa_types import *
from pydantic.v1 import Field
from chdrft.utils.path import FileFormatHelper
import time
from chdrft.utils.rx_helpers import ImageIO
from chdrft.sim.rb.base import Vec3, Transform

from influxdb_client import InfluxDBClient, Point
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.domain.write_precision import WritePrecision
import datetime

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


kDefaultDb = 'db1'
# influx auth  create -o A --all-access
token = 'ONc1EATrNWXMSweHPQ5PIV8NuOpk7LWo66VV3CwOSGbenw0cXdL_ZwiTVtWWu6IRkTqIoTn_caWY1LFPactycw=='


def normalizer(k, v):
  if (isinstance(v, str) or isinstance(v, int) or isinstance(v, bool) or isinstance(v, float)):
    yield (k, v)
  elif isinstance(v, Vec3):
    for i in range(3):
      yield (f'{k}[{i}]', v[i])
  elif isinstance(v, Transform):
    for i in range(3):
      yield (f'{k}.pos[{i}]', v.pos_v[i])
  elif isinstance(v, np.ndarray):
    pass
  else:
    nv = str(v)
    yield (k, nv)


class InfluxDbRun:

  def __init__(self, enable: bool, runid: str):
    self.enable = enable
    if not enable: return
    client = InfluxDBClient('http://localhost:8086', token)
    self.client = client
    self.tags = dict(
        runid=runid,
        xtime=time.time(),
    )
    self.bucket = 'b1'
    self.write_api = client.write_api(write_options=SYNCHRONOUS)

  def normalize(self, data):
    for k, v in data.items():
      yield from normalizer(k, v)

  def push(self, data={}, **kwargs):
    if not self.enable: return
    kwargs.update(data)
    final = dict(self.normalize(kwargs))
    p = Point.from_dict(
        dict(measurement='meas1', tags=self.tags, time=datetime.datetime.utcnow(), fields=final),
        field_types=dict(time='float'),
        write_precision=WritePrecision.MS,
    )
    self.write_api.write(bucket=self.bucket, record=p, org='A')


def influx_connector(runid: str) -> ImageIO:

  infdb = InfluxDbRun(enable=True, runid=runid)

  def push(data):
    infdb.push(data)

  return ImageIO(f=push)


def test(ctx):
  a = InfluxDbRun(enable=True, runid='r1')
  a.push(data=dict(test=555))
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
