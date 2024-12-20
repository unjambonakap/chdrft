import inspect
import pdb
import argparse
import glog
import logging
import sys
import os
import shlex
from chdrft.config.base import is_python2
from chdrft.config.env import g_env, g_ipython
import numpy as np
import random
import faulthandler
import chdrft.utils.misc as cmisc

import signal, traceback
#def quit_handler(signum,frame):
#    traceback.print_stack()
#signal.signal(signal.SIGQUIT,quit_handler)

if not is_python2:
  from contextlib import ExitStack

import glog

import logging




class App:

  def __init__(self):
    self.flags = None
    self.stack: ExitStack = None
    self.override_flags = {}
    self.setup = False
    self.cache = None
    if not is_python2:
      self.global_context = cmisc.ExitStackWithPush()

    self.env = g_env
    self.env.setup(self)

  def setup_jup(self, cmdline='', argv=None, **kwargs):
    argv = shlex.split(cmdline)
    self(force=1, argv=argv, **kwargs, keep_open_context=1)
    self.setup = True
    return cmisc.A(vars(self.flags))

  def exit_jup(self):
    self.global_context.close()

  def shell(self, n=0):
    g_ipython.drop_to_shell(n + 1)

  def wait_or_interactive(self, n=0):
    if g_ipython.in_jupyter:
      self.shell(n + 1)
    else:
      input('Press to stop')

  def __call__(self, force=False, argv=None, parser_funcs=[], keep_open_context=None):

    f = inspect.currentframe().f_back
    if not force and self.setup: return

    if not force and not f.f_globals['__name__'] == '__main__': return
    self.setup = True

    import sys
    if keep_open_context is None:
      keep_open_context = g_ipython.in_jupyter

    if 'main' not in f.f_globals and not force: return
    parser = None
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbosity', type=str, default='ERROR')
    parser.add_argument('--pdb', action='store_true')
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--runid', type=str, default='default')
    want_cache = force or ('cache' in f.f_globals and not is_python2)
    cache = None
    if want_cache:
      from chdrft.utils.cache import cache_argparse
      cache_argparse(parser)

    if 'args' in f.f_globals:
      args_func = f.f_globals['args']
      args_func(parser)
    for x in parser_funcs:
      x(parser)

    random.seed(0)
    np.random.seed(0)

    faulthandler.register(signal.SIGQUIT)
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    parser.add_argument('other_args', nargs=argparse.REMAINDER, default=['--'])

    flags = parser.parse_args(args=argv)
    if flags.other_args and flags.other_args[0] == '--':
      flags.other_args = flags.other_args[1:]
    self.flags = flags
    for k, v in self.override_flags.items():
      setattr(self.flags, k, v)

    glog.setLevel(flags.verbosity)
    if flags.log_file:
      glog.logger.addHandler(logging.FileHandler(flags.log_file))

    if 'flags' in f.f_globals:
      f.f_globals['flags'] = flags

    if want_cache:
      from chdrft.utils.cache import FileCacheDB
      self.cache = FileCacheDB.load_from_argparse(flags)
      f.f_globals['cache'] = self.cache

    if self.stack is not None:
      self.stack.close()

    main_func = f.f_globals.get('main', None)

    def go():
      try:
        if is_python2:
          main_func()
        else:
          if keep_open_context:
            stack = ExitStack()
            self.run(stack, main_func)
          else:
            with ExitStack() as stack:
              self.run(stack, main_func)
      except Exception as e:
        if flags.pdb:
          pdb.post_mortem()
        raise
      except KeyboardInterrupt:
        raise

    if flags.pdb:
      pdb.runcall(go)
    else:
      go()
      self.stack = None

  def run(self, stack, main_func):
    self.stack = stack
    stack.enter_context(self.global_context)
    script_name = sys.argv[0]
    plog_filename = '/tmp/opa_plog_{}_{}.log'.format(
        os.path.basename(script_name), self.flags.runid
    )

    plog_file = open(plog_filename, 'w')
    stack.enter_context(plog_file)
    self.plog_file = plog_file

    if self.cache:
      stack.enter_context(self.cache)
    if main_func is not None: main_func()


app = App()
