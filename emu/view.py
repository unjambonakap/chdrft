from chdrft.emu.structures import Structure, StructureReader, YamlField, CORE_TYPES, accessors_ctx
from chdrft.tube.serial import TunneledFiles, Serial
from chdrft.utils.misc import cwdpath, Attributize, to_bytes, PatternMatcher, DictWithDefault, is_list
from collections import OrderedDict
import glog
import os
import re
import struct
import threading
import time


class YamlViewElement:
  def __init__(self, typ, desc, ctx):
    if is_list(desc):
      name, desc = desc
      desc.name = name
    self.deps=[]

    self.name = desc.name
    self.desc = desc
    self.field = YamlField(desc, None, CORE_TYPES, typ, ctx=ctx)

  def add_to_struct(self, target):
    target.add_field(self.field)

class YamlStructView(object):

  def __init__(self, typ, ctx):
    self.typ = typ
    self.ctx = ctx
    self.ctx.update(accessors_ctx)
    self.elems = OrderedDict()
    self.processed = DictWithDefault(lambda : 0)

  def add_view(self, desc):
    view = YamlViewElement(self.typ, desc, self.ctx)
    self.elems[view.name] = view

  def load_views(self, desc):
    for x in desc.items():
      self.add_view(x)
    return self

  def get(self, *args, **kwargs):
    s = Structure(self.typ, *args, **kwargs)
    self.augment(s)
    return s

  def augment(self, s):
    self.processed.clear()
    vals = list(self.elems.values())
    vals.sort(key=lambda x: x.name)

    for e in vals:
      self._add_element(s, e)

  def _add_element(self, s, e):

    assert self.processed[e.name] != -1
    if self.processed[e.name] == 1: return
    self.processed[e.name] = -1
    for x in e.deps:
      if x in self.elems:
        self.add_element(s, self.elems[x])

    e.add_to_struct(s)
    self.processed[e.name] = 1

class YamlViewBuilder:

  def __init__(self):
    self.typs = Attributize(key_norm=byte_norm)
    self.add_core_types()
    self.typs_to_build = Attributize(key_norm=byte_norm)

  def add_core_types(self):
    for name, typ in types_helper.by_name.items():
      self.typs[name] = OpaCoreType(typ)

  def add_yaml(self, s):
    res = yaml.load(s)
    res = Attributize.RecursiveImport(res)
    self.typs_to_build.update(res._elem)

  def build(self):
    for k, v in self.typs_to_build.items():
      if k == 'global': continue
      self.build_typ(k, v)

  def get_typ(self, typ_name):
    if typ_name in self.typs:
      return self.typs[typ_name]
    assert typ_name in self.typs_to_build
    return self.build_typ(typ_name, self.typs_to_build[typ_name])

  def build_typ(self, name, desc):
    # prevent loops
    self.typs[name] = None
    res = YamlType(name, desc, self)
    self.typs[name] = res
    return res


  @staticmethod
  def FromYamlFile():
    pass



