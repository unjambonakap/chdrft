#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
from chdrft.tools.elf_mod import extract_binary_syms
import chdrft.struct.base as cbase
from chdrft.emu.binary import x64_mc
import base64
from chdrft.utils.fmt import Format
import os
from collections import defaultdict

global flags, cache
flags = None
cache = None

class Intervals:
  def __init__(self, collection):
    self.mp = {}
    self.intervals = []
    for inter, x in collection:
      self.intervals.append(inter)
      self.mp[inter] = x

  def find(self, a):
    for inter in self.intervals:
      if inter.contains(a):
        return  self.mp[inter]
    return None



class BinHelper:
  def __init__(self, fname, sym_data):
    self.fname = fname
    self.f = open(fname, 'rb').read()
    self.sd = sym_data


  def get_func(self, name):
    if isinstance(name, str): f = self.sd[name]
    else: f=name

    data = self.f[f.offset:f.offset+f.realsz]
    return data

  def get_func_ins(self, name):
    data = self.get_func(name)
    return x64_mc.disp_ins(data, addr=f.offset)


  def find_strs_exact(self, x):
    x = Format(x).tobytes().v

    for s in self.sd.s:
      if s.value == x:
        yield s

  def find_strs(self, x):
    x = Format(x).tobytes().v

    for s in self.sd.s:
      if s.value.find(x) !=-1:
        yield s

def outdeg_filter(lst):
  lst = cmisc.to_list(lst)
  tgt = defaultdict(lambda: 0)
  for x in lst:
    tgt[x] += 1
  def check(x):
    for k, v in tgt.items():
      if x.outdeg['sym.imp.'+k] != v: return False
    return True
  return check

def r2_extract_all(filename, offset=0):
  import r2pipe
  p = r2pipe.open(filename)
  p.cmd('aac')
  p.cmd('aa')

  def do_cmd(x):
    return cmisc.Attributize.RecursiveImport(p.cmdj(x))
  res = cmisc.Attributize()

  funcs = cmisc.Attributize({d['name']:d for d in do_cmd('aflj')})
  strs= do_cmd('izj')
  xrefs = do_cmd('axj')

  for x in strs:
    x.value = base64.b64decode(x.string)
    x.xrefs = defaultdict(list)



  for f in funcs.values():
    f.outdeg = defaultdict(lambda: 0)
    f.coderefs_funcs = defaultdict(list)
    f.codexrefs_funcs = defaultdict(list)
    f.coderefs_funcs_list = []
    f.codexrefs_funcs_list = []
    f.refs_str = []

  funcs_inter = Intervals([(cbase.Range1D(offset +x.offset, n=x.size-1), x) for x in funcs.values()])
  strs_inter = Intervals([(cbase.Range1D(offset +x.vaddr, n=x.size-1), x) for x in strs])

  for xref in xrefs:
    xref['addr'] += offset
    xref['from'] += offset
    a =  strs_inter.find(xref['addr'])
    b = funcs_inter.find(xref['from'])
    if not a or not b: continue
    #print(a.value, b.name, hex(v), hex(k))
    ref_obj = cmisc.Attributize(src=b, sink=a, addr=xref.addr, raw=xref)
    a.xrefs[b.name].append(ref_obj)
    b.refs_str.append(ref_obj)

  for f in funcs.values():
    for xref in f.get('codexrefs', []):
      xref.addr += offset
      u = funcs_inter.find(xref.addr)
      if u is None: continue
      ref_obj = cmisc.Attributize(src=u, sink=f, addr=xref.addr, raw=xref)
      f.codexrefs_funcs[u.name].append(ref_obj)
      u.coderefs_funcs[f.name].append(ref_obj)
      f.codexrefs_funcs_list.append(ref_obj)
      u.coderefs_funcs_list.append(ref_obj)
      u.outdeg[f.name] += 1

  return cmisc.Attributize(funcs_inter=funcs_inter, strs_inter=strs_inter, f=funcs, s=strs)

