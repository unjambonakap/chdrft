#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, cwdpath
from chdrft.utils.fmt import Format
import glog
import csv
import base64
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import curses.ascii
import chdrft.graph.render as opa_render
import itertools
import subprocess as sp
import tempfile
import os
import yaml

global flags, cache
flags = None
cache = None


def make_printable(s):
  res = bytearray()
  for x in s:
    if curses.ascii.isprint(x): res.append(x)
    else: res += b'_'
  return res.decode()

class localstr:
  def __init__(self, s, addr=0, shortened=False):
    self.s = base64.b64decode(s)
    self.addr = addr
    self.shortened = shortened

  def __repr__(self):
    return '"' + make_printable(self.s) + '"'

def args(parser):
  clist = CmdsList().add(test).add(test_patch)
  parser.add_argument('--trace', type=cwdpath)
  parser.add_argument('--outfile', type=cwdpath)
  parser.add_argument('--binary', type=cwdpath)
  parser.add_argument('--patchfunc_lib', type=cwdpath)
  parser.add_argument('--patch_func', type=Format.ToInt)
  parser.add_argument('--patch_val', type=Format.ToInt)
  ActionHandler.Prepare(parser, clist.lst)

def read_kv(s):
  pos = s.find('=')
  assert pos != -1, s
  k = s[:pos]
  v = s[pos+1:]
  v = eval(v, globals())
  return k, v


class Event(Attributize):
  def __init__(self, *args, root=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.root = root

  def setup(self):
    self._desc = self.desc

  def set_closing(self, closing_ev):
    if 'args' in closing_ev:
      self.args = self.get('args', {})
      self.args.update(closing_ev.args)

  def key(self):
    if self.type == 'syscall':
      return 'syscall:'+self.name
    return 'func:%s:%s:%s'%(self.name, self.funcid, self.addr)

  @property
  def real_addr(self):
    if self.type == 'syscall':
      if 'syscall' and self.ret:
        return self.ip - 2
      return self.ip
    else:
      return self.ip

  def fullkey(self):
    return self.i

  @property
  def desc(self):
    if self.root: return 'root' 
    if self.type == 'syscall':
      content = f'{self.name}{self.args}'
    else:
      content = self.name
      if 'args' in self:
        content = f'{self.name}({self.args})'

    return f'{self.type} {content}'

class Context:
  def __init__(self, filter_func=None):
    self.imgs = {}
    self.events = []
    self.g = None
    self.root = -1
    self.data = None
    self.filter_func =filter_func

  def read(self, filename):
    for event in read_file(filename):
      event.ctx = self
      event.ignore= False
      if event.type == 'img_load':
        self.imgs[event.name] = event
      else: self.events.append(event)

  def close_vertex(self, v):
    if self.filter_func is None: return
    if self.filter_func(v): return
    v.ignore= True




  def build_graph(self):

    stack = []

    tmp_g =  nx.DiGraph()
    root_event = Event(root=True, parent=None, ignore=False, i=self.root, name='RootNode')
    cur = root_event
    nodes = [root_event]
    plt_name = None
    for i, entry in enumerate(self.events):
      #print(i, len(self.events))
      #print('\n\nProcessing ', entry._str_oneline())
      if entry.name == '.plt': continue
      if '@plt' in entry.name: continue

      if entry.ret:
        found=False
        while len(stack)>0:
          self.close_vertex(stack[-1])
          if stack[-1].key() == entry.key():
            found=True
            break
          #'stack=%s, elem=%s'%(stack, entry)
          #glog.info('Force closing %s, at %s', stack[-1]._str_oneline(), entry._str_oneline())
          stack.pop()

        if not found:
          print('IGNORING RET FUNC ', i, entry)

        else:
          assert found, str(entry)
          entry_event = stack[-1]
          entry_event.set_closing(entry)
          stack.pop()
          cur = root_event
          if len(stack)>0: cur = stack[-1]
      else:
        plt_name = None
        nodes.append(entry)
        entry.parent = cur
        cur = entry
        stack.append(entry)

    self.g = nx.DiGraph()
    for node in nodes:
      node.setup()
      if node.parent is not None:
        print(node.name, node.ignore, ' >> parent ', node.parent.name)
      if node.ignore: continue
      if (node.parent is not None) and node.parent.ignore:
        print('new ignore')
        node.ignore = True
        continue
      self.g.add_node(node.fullkey(), data=Attributize(order=len(self.g), entry=node))
      if node.parent is not None:
        self.g.add_edge(node.parent.fullkey(), node.fullkey())

    final_nodes = list([self.g.node[x]['data'].entry for x in self.g.nodes_iter()])
    def set_depth(x):
      if x.depth is None:
        if x.parent is None: x.depth = 0
        else: x.depth = set_depth(x.parent) + 1
      return x.depth
    for node in final_nodes: node.depth = None
    for node in final_nodes: set_depth(node)



  def render(self):
    tree = opa_render.DrawTree(self.g, self.root)
    opa_render.buchheim(tree)
    for node in self.g:
      nd = self.g.node[node]
      dt = nd['data']
      nd['x'] = dt.renderdata.x
      nd['y'] = dt.renderdata.y

  def setup_data(self):
    self.data = {}
    for node in self.g:
      nd = self.g.node[node]
      nd['desc'] = nd['data'].entry.desc
      nd['name'] = nd['desc']
      self.data[node] = nd['data']
      del nd['data']


class Patcher:
  def __init__(self, binary, patchfunc_lib=None, args=[], pin_binary='pin'):
    if patchfunc_lib is None: patchfunc_lib=flags.patchfunc_lib
    self.args = args
    self.pin_binary = pin_binary

    self.patchfunc_lib = patchfunc_lib
    self.binary = binary

  def run_one(self, func, val):
    with tempfile.NamedTemporaryFile() as tmpfile:
      print('OUTPUT >> ', tmpfile.name)
      cmd = [self.pin_binary, '-t', self.patchfunc_lib, '-opa_inscount_logfile', tmpfile.name, '-opa_patch_func', str(func), '-opa_patch_val', str(val), '--', self.binary]
      cmd += self.args
      proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE,)
      stdout, stderr = proc.communicate()
      retcode = proc.returncode
      return Attributize(retcode=retcode,
          stdout=stdout,
          stderr=stderr,
          log=yaml.load(open(tmpfile.name, 'r')))


  def run_entry(self, entry, retv=None):
    if retv is None: retv = 0 if entry.retv!=0 else 1
    return self.run_one(entry.addr, retv)

  def run_bunch(self, entries):
    for entry in entries:
      yield self.run_entry(entry)



def read_file(filename):
  with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i, entry in enumerate(reader):
      data={'i':i}
      try:
        for kv in entry:
          k,v = read_kv(kv)
          data[k] = v
        yield Event(data)
      except:
        print('Failure on >> ', entry, data)
        raise

def serialize_for_gml(entry):
  print('GOT >> ', entry, type(entry))
  if isinstance(entry, opa_render.DrawTree): return ''
  return entry


def test_patch(ctx):
  patcher = Patcher(ctx.binary, ctx.patchfunc_lib)
  patcher.run_one(ctx.patch_func, ctx.patch_val)



def test(ctx):
  data = read_file(ctx.trace)
  G = trace_to_graph(data)
  for node in G:
    nd = G.node[node]
    nd['desc'] = nd['data'].entry.desc
    del nd['data']
  nx.write_gml(G, ctx.outfile, stringizer=serialize_for_gml)
  return


  pos = nx.spring_layout(G)
  #nx.write_gexf(G, ctx.outfile)
  labels = {k:G.node[k]['data'].desc for k in G.nodes() if k is not None}
  nx.draw_networkx(G, pos=pos, with_labels=True, labels=labels)
  plt.draw()
  plt.show()
  print('Remainer of the stack >> ', stack)





def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
