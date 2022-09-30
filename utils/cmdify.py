#!/usr/bin/env python

import sys
import inspect
import argparse
from chdrft.utils.misc import Attributize, is_python2, cwdpath, csv_list
import chdrft.utils.misc as cmisc
import pickle
import subprocess as sp
import glog
import os
import pprint

if not is_python2:
  from contextlib import ExitStack

global flags
flags = None


def prepare_parser(sp, cmd):
  name = cmd.get_name()
  parser = sp.add_parser(name)
  return parser

def argparsify(parser, a):

  names = []
  typs=[]
  defaults = []
  if a.args is None:
    for px in inspect.signature(a.func).parameters.values():
      names.append(px.name)
      typs.append(px.annotation)
  else:
    args = inspect.getargspec(a.func)
    names = args.args
    typs = a.args
    defaults = args.defaults


  a2_default = defaults
  if not a2_default:
    a2_default = []
  pos_default = len(names) - len(a2_default)
  a1 = names[:pos_default]
  a1_typs = typs[:pos_default]

  a2 = names[pos_default:]
  a2_typs = typs[pos_default:]

  for name, typ in zip(a1, a1_typs):
    opt = Attributize()
    if isinstance(typ, list):
      assert len(typ) == 1
      opt.type = typ[0]
      opt.nargs = '*'
    else:
      opt.type = typ
    parser.add_argument('--%s' % name, required=True, **opt._elem)

  for name, typ, default in zip(a2, a2_typs, a2_default):
    parser.add_argument('--%s' % name, type=typ, default=default)

  def do_call(data):
    vd = vars(data)
    data = {x: vd[x] for x in names}
    res = a.func(**data)
    if res is not None:
      print(res)

  parser.set_defaults(func=do_call)


def call_argparsified(cmd, func, *args):
  cmd.append(__file__)
  cmd.append(func.__name__)  #argparse action

  func_args = inspect.getargspec(func)
  for name, arg in zip(func_args[0], args):
    cmd.extend(['--%s' % name, str(arg)])

  print(' '.join(cmd))
  data = sp.check_output(cmd).decode().rstrip()
  return data


def argparsify_pickle(parser, f):
  parser.add_argument('--pickle_file', type=str)

  def do_call(data):
    args = pickle.load(open(data.pickle_file, 'rb'))
    res = f.func(*args[0], **args[1])
    fx = open(data.pickle_file, 'wb')
    pickle.dump(res, fx, protocol=2)
    fx.flush()

  parser.set_defaults(func=do_call)


def call_sage(func, *args, **kwargs):
  return call_pickle(func, ['sage', '-python'], *args, **kwargs)


def call_pickle(func, cmd, *args, **kwargs):
  import tempfile
  cmd = list(cmd)
  import chdrft.master
  cmd.append(chdrft.master.__file__)
  cmd.append(func.__name__)
  tmp = tempfile.NamedTemporaryFile(delete=False)
  cmd.append('--pickle_file=%s' % tmp.name)
  pickle.dump([args, kwargs], tmp, protocol=2)
  tmp.flush()
  #print(' '.join(cmd))
  sp.check_call(cmd)
  return pickle.load(open(tmp.name, 'rb'))


def run_cmdify(parser, lst):

  FileCacheDB = None
  cache_argparse = None
  if not is_python2:
    from chdrft.utils.cache import FileCacheDB, cache_argparse

  #if cache_argparse:
  #  cache_argparse(parser)

  sp = parser.add_subparsers()
  sp.required = True
  sp.dest = 'command'
  for x in lst:
    n_parser = prepare_parser(sp, x)
    if x.parser_builder is None:
      if x.typ == 'pickle':
        argparsify_pickle(n_parser, x)
      else:
        argparsify(n_parser, x)
    else:
      n_parser.set_defaults(func=x.func)
      x.parser_builder(n_parser)


class ActionHandler:
  g_handler = None

  def __init__(self, parser, cmds, init=None, global_action=False):
    self.parser = parser
    self.cmds = {x.get_name(): x for x in cmds}
    parser.add_argument('--actions', type=str, default='')
    parser.add_argument('--cleanup_actions', type=str, default='')
    parser.add_argument('--action_output_file', type=str)
    parser.add_argument('--noaction_log_output', action='store_true')
    parser.add_argument('--noctrlc_trace', action='store_true')
    parser.add_argument('--ret_syscode', action='store_true')
    parser.add_argument('--no-local-db', action='store_true')
    parser.add_argument('--local-db-file', type=str, default='.chdrft.db.pickle')

    self.args = None
    self.kwargs = None
    self.flags = None
    self.stack = None
    self.init = init
    self.global_action = global_action

  def do_proc(self, stack, flags, *args, **kwargs):
    self.stack = stack
    output_file = None
    from chdrft.utils.path import FileFormatHelper
    mode = None


    if len(self.args) == 1 and isinstance(self.args[0], Attributize):
      ctx = self.args[0]
      for k, v in vars(self.flags).items():
        if k not in ctx:
          ctx[k] = v

      from chdrft.utils.path import FileFormatHelper
      if not flags.no_local_db and os.path.exists(flags.local_db_file):
        try:
          tmp = FileFormatHelper.Read(flags.local_db_file)
          tmp.update(ctx)
          ctx = tmp
        except:
          pass

      self.args[0] = ctx

    if self.init: self.init(*self.args)

    if flags.action_output_file:
      if stack is not None:
        output_file = FileFormatHelper(filename=flags.action_output_file, write=True)
        stack.enter_context(output_file)
    try:
      for action in self.main_actions:
        res = self.execute_action(action)
        if output_file:
          output_file.write(res)
        if not flags.noaction_log_output:
          glog.info('Action %s results: %s', action, res)
        if flags.ret_syscode:
          if isinstance(res, bool): res = 1 if not res else 0
          sys.exit(res)
    except KeyboardInterrupt as e:
      if flags.noctrlc_trace: pass
      else: raise e
    if not flags.no_local_db:
      FileFormatHelper.Write(flags.local_db_file, ctx)

  def proc(self, flags, caller_ctx, *args, **kwargs):
    self.caller_ctx = caller_ctx
    self.args = list(args)
    self.kwargs = kwargs
    self.flags = flags

    actions = []

    if is_python2:
      self.do_proc(None, flags, *args, **kwargs)
    else:
      with ExitStack() as stack:
        self.do_proc(stack, flags, *args, **kwargs)

    for action in self.cleanup_actions:
      self.execute_action(action)


  def reqs(self, flags):
    self.flags = flags
    res = Attributize(default=False)
    for x in self.all_actions:
      for v in x.reqs:
        res[v] = True
    return res

  @property
  def all_actions(self):
    return self.main_actions + self.cleanup_actions

  @property
  def main_actions(self):
    return self.get_actions(csv_list(self.flags.actions))

  @property
  def cleanup_actions(self):
    return self.get_actions(csv_list(self.flags.cleanup_actions))

  def get_actions(self, action_list):
    actions = []
    for action in action_list:
      actions.append(self.get_action(action))
    return actions

  def get_action(self, action_name):
    if self.global_action: return self.caller_ctx[action_name]
    assert action_name in self.cmds, 'Action not known %s, lst=(%s)' % (
        action_name, self.cmds.keys()
    )
    return self.cmds[action_name]

  def execute_action(self, action):
    if callable(action):
      return action(*self.args, **self.kwargs)
    return action.func(*self.args, **self.kwargs)

  @staticmethod
  def Prepare(*args, **kwargs):
    ActionHandler.g_handler = ActionHandler(*args, **kwargs)

  @staticmethod
  def Run(*args, parent_frame_n=0, **kwargs):
    from chdrft.main import app
    _, caller_ctx = cmisc.get_n2_locals_and_globals(n=parent_frame_n)
    ActionHandler.g_handler.proc(app.flags, caller_ctx, *args, **kwargs)

  @staticmethod
  def Reqs():
    from chdrft.main import app
    return ActionHandler.g_handler.reqs(app.flags)

  @staticmethod
  def Get():
    return ActionHandler.g_handler
