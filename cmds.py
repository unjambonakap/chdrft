#!/usr/bin/env python

from chdrft.utils.misc import Attributize, is_python2, cwdpath, csv_list
import types

def ListInput(x):
  if isinstance(x, list): return x
  return x.split(',')


class CmdEntry:

  def __init__(self,
               func=None,
               args=None,
               name=None,
               typ=None,
               parser_builder=None, reqs=[]):
    assert func
    self.func = func
    self.args = args
    self.name = name
    self.typ = typ
    self.parser_builder = parser_builder
    self.reqs = ListInput(reqs)

  def get_name(self):
    return self.name or self.func.__name__


class CmdsList:

  def __init__(self):
    self.lst = []

  def add(self, *args, **kw):
    self.lst.append(CmdEntry(*args, **kw))
    return self
  def add_all(self, kv):
    for k,v in kv.items():
      if isinstance(v, types.FunctionType):
        self.add(v, name=k)
    return self


Cmds = CmdsList()
