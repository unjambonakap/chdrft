#!/usr/bin/env python

from chdrft.main import app
from chdrft.utils.misc import to_list, Attributize
from enum import Enum
import chdrft.utils.misc as cmisc
import glog

global flags, cache
flags = None
cache = None


def args(parser):
  pass




def fmt_num(v, n=None):
  if n is None:
    n = 8
  while True:
    cur =  ('{:0%dx}' % (2 * n)).format(v)
    if len(cur)>2*n:
      n*=2
    else:
      return cur


class Display:

  @staticmethod
  def regs_summary(regs, base_regs):

    per = 5
    s = ''
    regs_list = base_regs.reg_list()
    for i in range(0, len(regs_list), per):
      for j in range(per):
        if j + i >= len(regs_list):
          break
        reg = regs_list[i + j]
        reg_size = base_regs.reg_size(reg)
        s += ('{}={:0' + str(reg_size * 2) + 'x}\t').format(reg, regs[reg])
      s += '\n'
    return s.rstrip()

  @staticmethod
  def mem_summary(mem, start, n, word_size=4, istart=None, minv=None, maxv=None):
    if minv is None: minv = getattr(mem, 'minv', 0)
    if maxv is None: maxv = getattr(mem, 'maxv', 0)
    s = ''
    if istart is None:
      istart = 0
    fmt = '{:016x}:\t{:0%dx}\n' % (word_size * 2)
    for i in range(istart, istart + n):
      addr = start + i * word_size
      if minv and addr < minv:
        continue
      if maxv and addr >= maxv:
        continue
      v = mem.read_u(addr, word_size)
      s += fmt.format(addr, v)
    return s

  @staticmethod
  def disp_2str(s1, s2, name=''):
    s = ''
    if name:
      s += '{}\n'.format(name)
    s += '{}\n{}\n'.format(s1, s2)
    return s

  @staticmethod
  def disp(a, name='', n_per_lines=1000,**kwargs):
    params = Display.update_params(**kwargs)
    s = ''
    if a is None:
      pass
    elif isinstance(a, int):
      s = fmt_num(a, params.num_size)
    elif isinstance(a, list):
      s = Display.disp_list(a, params=params)
    elif isinstance(a, dict):
      keys = list(a.keys())
      keys.sort()
      s = []
      for i in range(0, len(keys), n_per_lines):
        s.append(Display.disp_list2([[k, a[k]] for k in keys[i:i+n_per_lines]], params=params))
      s = '\n'.join(s)
    elif isinstance(a, str):
      s = a
    elif isinstance(a, bytes):
      s = str(a)
    else:
      assert 0, type(a)

    if name and len(s) > 0:
      s = name + ' ' + s
    return s

  @staticmethod
  def disp_list2(lst, **kwargs):
    s = ''
    for e in lst:
      s += '{} '.format(Display.disp(e, **kwargs))
    return s

  @staticmethod
  def disp_list(lst, **kwargs):
    return ':'.join([Display.disp(x, **kwargs) for x in lst])

  @staticmethod
  def diff_dicts(t1, t2, name='', **kwargs):
    params = Display.update_params(**kwargs)
    s1 = []
    s2 = []
    s1_rem = []
    s2_rem = []
    keys = list(set(list(t1.keys()) + list(t2.keys())))
    keys.sort()
    for k in keys:
      if not k in t1:
        s2_rem.append([k, fmt_num(t2[k], params.num_size)])
      elif not k in t2:
        s1_rem.append([k, fmt_num(t1[k], params.num_size)])
      elif t1[k] != t2[k]:
        s1.append([k, fmt_num(t1[k], params.num_size)])
        s2.append([k, fmt_num(t2[k], params.num_size)])

    s1 = Display.disp_list2(s1)
    s2 = Display.disp_list2(s2)
    s1_rem = Display.disp_list2(s1_rem)
    s2_rem = Display.disp_list2(s2_rem)

    if len(s1_rem) > 0:
      s1 += '\t###\t' + s1_rem
    if len(s2_rem) > 0:
      s2 += '\t###\t' + s2_rem

    if not s1 and not s2: return ''
    return Display.disp_2str(s1, s2, name)

  @staticmethod
  def diff_lists(t1, t2, name='', restrict_entries=None, **kwargs):
    params = Display.update_params(**kwargs)
    nmin = min(len(t1), len(t2))

    s1 = []
    s2 = []

    for i in range(nmin):
      if restrict_entries is not None and i not in restrict_entries: continue
      if t1[i] != t2[i]:
        s1.append([i, t1[i]])
        s2.append([i, t2[i]])

    if len(t1) > nmin:
      for i in range(nmin, len(t1)):
        s1.append([i, t1[i]])

    if len(t2) > nmin:
      for i in range(nmin, len(t2)):
        s2.append([i, t2[i]])

    def fmt_pos(p):
      if params.addr is not None:
        return '{}'.format(fmt_num(params.addr + params.num_size * p))
      return '#{}'.format(p)

    def fmt_elem(pos, val, params):
      return '{}({}) '.format(fmt_num(val, params.num_size), fmt_pos(pos))

    s1v = ' '.join([fmt_elem(pos, val, params) for pos, val in s1])
    s2v = ' '.join([fmt_elem(pos, val, params) for pos, val in s2])
    return Display.disp_2str(s1v, s2v, name)

  @staticmethod
  def disp_lines(*lst):
    lst2 = [Display.disp(x) for x in lst if len(x) > 0]
    return '\n'.join(lst2)

  @staticmethod
  def update_params(**kwargs):
    if 'params' in kwargs:
      return kwargs['params']

    params = Attributize(default_none=True)
    for k, v in kwargs.items():
      params[k] = v
    return params

  @staticmethod
  def diff(a1, a2, name='', **kwargs):
    params = Display.update_params(**kwargs)

    assert type(a1) == type(a2)
    if isinstance(a1, list):
      return Display.diff_lists(a1, a2, name, params=params)
    elif isinstance(a1, dict):
      return Display.diff_dicts(a1, a2, name, params=params)
    else:
      assert 0


def dict_diff(a, b):
  res = dict()
  for k, v in a.items():
    if k not in b:
      res[k] = [v, None]
    elif b[k] != v:
      res[k] = [v, b[k]]
  for k, v in b.items():
    if k not in a:
      res[k] = [None, v]
  return res


class TraceEvent:

  def __init__(self, arch, regs, ins, addr, short_mode=False, diff_mode=True, info=''):
    self.short_mode = short_mode
    self.arch = arch
    self.mc = arch.mc
    self.read_event = []
    self.write_event = []
    self.diff_regs = dict()
    self.raw_ins = ins
    self.pc = regs.ins_pointer
    self.desc = self.mc.ins_str(self.mc.get_one_ins(ins, addr))
    self.stacks = []
    self.old_stacks = None
    self.diff_mode= diff_mode
    self.base_regs =regs
    self.info = info

    self.i_regs = self.extract_regs(regs)
    self.o_regs = None

  def extract_regs(self, regs):
    res = Attributize()
    if self.short_mode:
      return res

    return regs.snapshot_all()

  def notify_read(self, addr, n, val):
    cur = Attributize(addr=addr, n=n, val=val)
    self.read_event.append(cur)

  def notify_write(self, addr, n, val, nval):
    cur = Attributize(addr=addr, n=n, val=val, nval=nval)
    self.write_event.append(cur)

  def setup_new_regs(self, nregs):
    self.o_regs = self.extract_regs(nregs)
    self.diff_regs = dict_diff(self.i_regs, self.o_regs)

  def get_opreg_str(self):
    regs_data = {}
    ops = self.mc.get_reg_ops(self.mc.get_one_ins(self.raw_ins))
    if ops is None: return None
    if not 'reg_rel' in self.arch: return None
    for reg in ops:
      reg_size = self.arch.reg_rel.get_size(reg)
      res_reg = self.arch.reg_rel.find(reg, self.i_regs)
      if res_reg is None: continue
      assert res_reg is not None, reg + ' ' + str(self.i_regs)
      regs_data[reg] = fmt_num(res_reg, reg_size)
    return Display.disp(regs_data, name='OpReg')

  def __str__(self):
    data = []
    data.append(f'{self.desc} ({self.info})\t\t <<<< TRACEVENT')
    tmp = self.get_opreg_str()
    if tmp: data.append(tmp)
    data.append(Display.disp(self.read_event, name='ReadEvent'))
    data.append(Display.disp(self.write_event, name='WriteEvent'))
    if self.diff_mode: data.append(Display.diff_dicts(self.i_regs, self.o_regs, name='DiffRegs'))
    else: data.append(Display.regs_summary(self.o_regs, self.base_regs))

    return Display.disp_lines(*data)

  def is_rw(self):
    return self.is_r() or self.is_w()

  def is_r(self):
    return len(self.read_event) > 0

  def is_w(self):
    return len(self.write_event) > 0


class VmOp:

  def __init__(self, arch, watched, typ, pos, regs):
    self.arch = arch
    self.pos = pos
    self.watched = watched
    self.start_data = Attributize()
    self.end_data = Attributize()
    self.diff_params = Attributize(elem={x.name: x.diff_params for x in watched})
    self.regs = regs

    self.typ = typ
    self.reads = []
    self.writes = []
    self.header = ''

    self.window = 0x100

  def start(self):
    for x in self.watched:
      self.start_data[x.name] = x.snapshot()

  def end(self):
    for x in self.watched:
      self.end_data[x.name] = x.snapshot()

  def aggregate_rw(self, tb):
    tb.sort()
    last = -self.window - 1
    events = []
    for e in tb:
      if last + self.window < e:
        events.append([e, 1])
      events[-1][1] += 1
      last = e
    return events

  def __str__(self):
    return self.gets()

  def gets(self, regs=False):
    revents = self.aggregate_rw(self.reads)
    wevents = self.aggregate_rw(self.writes)

    data = []
    if self.header: data.append(self.header)
    data.append('VMOP at {} of typ {}, id={}'.format(
        Display.disp(self.regs.ins_pointer), self.typ, self.pos))
    for k in self.start_data.sorted_keys():
      data.append(
          Display.diff(
              self.start_data[k],
              self.end_data[k],
              name=self.diff_params[k].name,
              params=self.diff_params[k]))

      #debug
      if 0:
        data.append(
            Display.disp(
                self.start_data[k],
                name=self.diff_params[k].name + '_START',
                params=self.diff_params[k]))
        data.append(
            Display.disp(
                self.end_data[k],
                name=self.diff_params[k].name + '_END',
                params=self.diff_params[k]))

    data.append('ReadEvents: ' + Display.disp_list2(revents))
    data.append('WriteEvents: ' + Display.disp_list2(wevents))
    if regs:
      rs = Display.regs_summary(self.regs, self.regs)
      data.append('FullRegs:\n' + rs)
      for k in self.end_data.sorted_keys():
        data.append(
            Display.disp(
                self.end_data[k],
                name=self.diff_params[k].name + '_FULL',
                params=self.diff_params[k]))
    s = Display.disp_lines(*data)
    #assert self.end_data.regs['rsp']==self.start_data.regs['rsp'], s
    return s

  def notify_read(self, addr, n):
    self.reads.append(addr)

  def notify_write(self, addr, n, nval):
    self.writes.append(addr)


class WatchedMem:

  def __init__(self, name, mem, addr, end=None, n=None, max_length=None, sz=4, all=False):
    self.name = name
    self.mem = mem
    self.addr = addr
    self.end = end
    if n is None: n = (end - addr) // sz
    self.n = n
    self.sz = sz
    self.all = all
    self.diff_params = Attributize(default_none=True, name=self.name, num_size=sz, addr=self.addr)
    if not max_length:
      max_length = self.get_n()
    self.max_length = max_length

  def summary(self):
    return Display.mem_summary(self.mem, self.addr, self.n, word_size=self.sz)

  def is_in(self, addr):
    if self.all:
      return True
    end_addr = self.addr + self.max_length * self.sz
    return self.addr <= addr < end_addr

  def get_n(self):
    if self.end:
      nv = 0
      if callable(self.end):
        nv = self.end()
      else:
        nv = self.end
      return (nv + self.sz - 1 - self.addr) // self.sz
    if self.n:
      nv = 0
      if callable(self.n):
        nv = self.n()
      else:
        nv = self.n
      return nv
    assert 0

  def snapshot(self):
    nd = self.get_n()
    tb = []
    for i in range(nd):
      addr = self.addr + i * self.sz
      tb.append(self.mem.read_u(addr, self.sz))
    return tb


class WatchedRegs:

  def __init__(self, name, regs, lst):
    self.name = name
    self.regs = regs

    if lst == 'all': lst = regs.arch.regs
    self.lst = lst
    self.diff_params = Attributize(default_none=True, watcher=self, name=self.name, num_size=8)

  def summary(self):
    return Display.regs_summary(self.lst, self.regs)

  def is_in(self, addr):
    return False

  def snapshot(self):
    return dict({x: self.regs[x] for x in self.lst})




class Tracer:

  def __init__(self, arch, regs, mem, watched=[], cb=None, diff_mode=True):
    self.regs = regs
    self.mem = mem
    self.cur = None
    self.cb = cb
    self.filter = None
    self.watched = watched
    self.vm_op = None
    self.load_op = None
    self.count = 0
    self.arch = arch
    self.diff_mode=diff_mode
    self.ignore_unwatched = False

  def get_vm_op(self, typ):
    self.count += 1
    return VmOp(self.arch, self.watched, typ, self.count, self.regs)

  def start_vm_op(self, header=''):
    self.maybe_close_op()
    self.vm_op = self.get_vm_op('normal')
    self.vm_op.header = header
    self.vm_op.start()

    if self.load_op:
      self.load_op.end()
      self.send_event(self.load_op)

  def end_vm_op(self):
    self.load_op = self.get_vm_op('load')
    self.load_op.start()

    self.maybe_close_op()

  def maybe_close_op(self):
    if self.vm_op:
      self.vm_op.end()
      self.send_event(self.vm_op)

  def send_event(self, e):
    if self.cb:
      if self.filter is None or self.filter(e):
        self.cb(e)

  def notify_ins(self, addr, size, short_mode=False, info=''):
    if self.cur is not None:
      self.cur.setup_new_regs(self.regs)
      self.send_event(self.cur)

    ins = self.mem.read(addr, size)
    self.cur = TraceEvent(self.arch, self.regs, ins, addr, short_mode, diff_mode=self.diff_mode, info=info)

  def check_access(self, addr):
    if self.ignore_unwatched: return
    for watcher in self.watched:
      if watcher.is_in(addr):
        return
    assert 0, 'Bad access ' + hex(addr)

  def notify_read(self, addr, n):
    self.check_access(addr)
    buf = self.mem.read2(addr, n)
    if self.cur: self.cur.notify_read(addr, n, buf)

    for x in [self.vm_op, self.load_op]:
      if x:
        x.notify_read(addr, n)

  def notify_write(self, addr, n, nval):
    self.check_access(addr)
    val = self.mem.read2(addr, n)
    if self.cur: self.cur.notify_write(addr, n, val, nval)

    for x in [self.vm_op, self.load_op]:
      if x:
        x.notify_write(addr, n, nval)


class TraceDiff:

  def __init__(self, qs):
    self.qs = qs
    self.n = len(qs)
    self.f = open('/tmp/trace.diff', 'w')

  def diff_stacks(self, addr, s1, s2):
    d1 = []
    d2 = []
    assert len(s1) == len(s2)
    for i in range(len(s1)):
      if s1[i] != s2[i]:
        d1.append('{}({})'.format(fmt_num(s1[i], 4), fmt_num(i * 4, 1)))
        d2.append('{}({})'.format(fmt_num(s2[i], 4), fmt_num(i * 4, 1)))
    if len(d1) > 0:
      self.out('Stack {}:\n{}\n{}'.format(fmt_num(addr), ' '.join(d1), ' '.join(d2)))

  def can(self):
    for q in self.qs:
      if q.empty():
        return False
    return True

  def step(self):
    lst = []
    for q in self.qs:
      lst.append(q.get()[2])
    assert len(lst) == 2
    a, b = lst[0], lst[1]

    assert a.i_regs.rip == b.i_regs.rip
    res = dict_diff(a.o_regs, b.o_regs)
    #self.out('{:016x}: {} ## {}\n'.format(a.i_regs.rip, a.diff_regs, b.diff_regs))

    diff_reg = ''
    disp_regs = dict()
    if len(res) > 0:
      d1 = []
      d2 = []
      for r, v in res.items():
        if v[0] == a.i_regs[r] and v[1] == b.i_regs[r]:
          continue
        disp_regs[r] = 0
        d1.append('{}: {}'.format(r, fmt_num(v[0], 8)))
        d2.append('{}: {}'.format(r, fmt_num(v[1], 8)))

      if len(d1) > 0:
        diff_reg = '{}\n{}'.format(' '.join(d1), ' '.join(d2))
    tmp = ''
    if a.read_event and a.read_event.val != b.read_event.val:
      tmp += 'Read diff: addr={}, v1={}, v2={} ;  '.format(
          fmt_num(a.read_event.addr),
          fmt_num(a.read_event.val, a.read_event.n), fmt_num(b.read_event.val, b.read_event.n))

    if a.write_event and a.write_event.nval != b.write_event.nval:
      tmp += 'Write diff: addr={}, v1={}, v2={}'.format(
          fmt_num(a.write_event.addr),
          fmt_num(a.write_event.nval, a.write_event.n), fmt_num(b.write_event.nval,
                                                                b.write_event.n))
    if diff_reg or tmp:
      regs_str = []
      for reg in self.mc.get_reg_ops(self.mc.get_one_ins(a.raw_ins)):
        if g_reg_rel.find(reg, disp_regs):
          continue
        reg_size = g_reg_rel.get_size(reg)
        regs_str.append('{}:{}'.format(reg, fmt_num(g_reg_rel.find(reg, a.i_regs), reg_size)))
      self.out('\nGOT INS: {}, regs={}'.format(a.desc, ' '.join(regs_str)))

    if diff_reg:
      self.out(diff_reg)

    if tmp:
      self.out(tmp)

    for k in range(len(a.stacks)):
      addr, s1 = a.stacks[k]
      _, s2 = b.stacks[k]
      self.diff_stacks(addr, s1, s2)

    if len(a.stacks) > 0:
      self.out('\n')
      if a.old_stacks:
        self.out('Vs old stacks:')
        for k in range(len(a.stacks)):
          addr, s1 = a.stacks[k]
          _, s2 = a.old_stacks[k]
          print(len(s1), len(s2), a.stacks, a.old_stacks)
          self.diff_stacks(addr, s1, s2)

      self.out('\n')

  def out(self, s):
    self.f.write(s + '\n')
    self.f.flush()


def main():
  pass


app()
