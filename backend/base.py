#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import A
import glog
import numpy as np
from pydantic import Field, BaseModel
from chdrft.utils.path import FileFormatHelper
import subprocess as sp
import time
import asyncio
import typing

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
