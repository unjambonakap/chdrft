#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.types import *
from pydantic import Field
from chdrft.utils.path import FileFormatHelper
import os

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


kCredsEnv = 'OPA_CREDS_FILE'


class Creds(cmisc.PatchedModel):
  data: dict
  def get(self, key, as_auth=False):
    v = self.data[key]
    if as_auth: return (v.login, v.password)
    return v

def get_creds(filename = None) -> dict[str,tuple[str, str]]:
  if filename is None: filename = os.environ[kCredsEnv]
  content = FileFormatHelper.Read(filename)
  return Creds(data=A.RecursiveImport({k: dict(login=v[0], password=v[1]) for k,v in content.items()}))




def test(ctx):
  print(get_creds())
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
