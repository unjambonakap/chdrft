#!/usr/bin/env python

import re
import inspect

def get_n2_locals_and_globals(n=0):
  p2 = inspect.currentframe().f_back.f_back
  for i in range(n):
    p2 = p2.f_back
  return p2.f_locals, p2.f_globals


def do_lazy_import(fname):
  modules = set()
  r1 = re.compile('^(from|import) (?P<module>\S+)\s')
  glob = get_n2_locals_and_globals()
  for line in open(fname, 'r').readlines():
    m = r1.search(line)
    if not m: continue
    glob




