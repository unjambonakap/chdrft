#!/usr/bin/env python
# encoding: utf-8
# Christoph Koke, 2013
"""
Writes the c and cpp compile commands into build/compile_commands.json
see http://clang.llvm.org/docs/JSONCompilationDatabase.html

Usage:

def configure(conf):
conf.load('compiler_cxx')
...
conf.load('clang_compilation_database')
"""

import sys
import os
import json
import shlex
import pipes
from waflib.Node import Node
from waflib import Logs, TaskGen
from waflib.Tools import c, cxx
from chdrft.utils.misc import get_or_set_attr, normalize_path, change_extension
import threading

glock = threading.Lock()

if sys.hexversion >= 0x3030000:
  quote = shlex.quote
else:
  quote = pipes.quote


def addTask(ctx, task):
  clang_db = get_or_set_attr(ctx, 'clang_compilation_database_tasks', [])
  if len(clang_db) == 0:
    ctx.add_post_fun(write_compilation_database)
  clang_db.append(task)


def do_normalize_path(basepath, x):
  if isinstance(x, Node):
    return x.abspath()
  return normalize_path(basepath, x)


def addHeaders(task, headers_list, mapping_src=[]):
  basepath = task.path.abspath()
  mapping_src = [do_normalize_path(basepath, x) for x in mapping_src]
  glock.acquire()
  headers = get_or_set_attr(task.bld, 'headers', {})
  glock.release()
  if isinstance(headers_list, str):
    headers_list = [headers_list]

  for x in headers_list:
    fpath = do_normalize_path(basepath, x)
    cpath = change_extension(fpath, 'c')
    headers[fpath] = [cpath] + mapping_src


@TaskGen.feature('*')
@TaskGen.after_method('process_use')
def collect_compilation_db_tasks(self):
  "Add a compilation database entry for compiled tasks"
  for task in getattr(self, 'compiled_tasks', []):
    if isinstance(task, (c.c, cxx.cxx)):
      addTask(self.bld, task)


def write_compilation_database(ctx):
  "Write the clang compilation database as JSON"
  database_file = ctx.bldnode.make_node('compile_commands.json')
  Logs.info("Build commands will be stored in %s" % database_file.path_from(ctx.path))
  try:
    root = json.load(database_file)
  except IOError:
    root = []
  clang_db = dict((x["file"], x) for x in root)
  for task in getattr(ctx, 'clang_compilation_database_tasks', []):
    try:
      cmd = task.last_cmd
    except AttributeError:
      continue
    directory = getattr(task, 'cwd', ctx.variant_dir)
    f_node = task.inputs[0]
    filename = f_node.abspath()
    cmd = " ".join(map(quote, cmd))
    entry = {
        "directory": directory,
        "command": cmd,
        "file": filename,
    }
    clang_db[filename] = entry

  headers = getattr(ctx, 'headers', {})
  for k, srcs in headers.items():
    for src in srcs:
      if src in clang_db:
        clang_db[k] = clang_db[src].copy()
        clang_db[k]['file'] = k
        break
    else:
      #print("Could not found appropriate params for header {}, list was {}".format(
      #    k, srcs))
      pass
  root = list(clang_db.values())
  database_file.write(json.dumps(root, indent=2))
