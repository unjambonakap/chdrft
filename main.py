import inspect
import argparse
from chdrft.utils.misc import is_python2
import glog
import logging
import sys
import os

if not is_python2:
  from chdrft.utils.cache import *
  from contextlib import ExitStack
  import argcomplete


class App:
  def __init__(self):
    self.flags = None
    self.stack = None
    if not is_python2:
      self.global_context = ExitStack()

  def __call__(self, force=False):
    f = inspect.currentframe().f_back

    if not force and not f.f_globals['__name__']=='__main__':
      return

    if 'main' in f.f_globals:
      parser = None
      parser = argparse.ArgumentParser(
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      parser.add_argument('--verbosity', type=str, default='ERROR')
      parser.add_argument('--log_file', type=str)
      parser.add_argument('--runid', type=str, default='default')
      want_cache = 'cache' in f.f_globals and not is_python2
      cache = None
      if want_cache:
        cache_argparse(parser)

      if 'args' in f.f_globals:
        args_func = f.f_globals['args']
        args_func(parser)



      parser.add_argument('other_args', nargs=argparse.REMAINDER, default=['--'])

      if not is_python2:
        argcomplete.autocomplete(parser)
      flags = parser.parse_args()
      if flags.other_args and flags.other_args[0] == '--':
        flags.other_args = flags.other_args[1:]
      self.flags = flags

      glog.setLevel(flags.verbosity)
      if flags.log_file:
        glog.logger.addHandler(logging.FileHandler(flags.log_file))

      if 'flags' in f.f_globals:
        f.f_globals['flags'] = flags

      if want_cache:
        cache = FileCacheDB.load_from_argparse(flags)
        f.f_globals['cache'] = cache

      main_func = f.f_globals['main']
      if is_python2:
        main_func()
      else:
        with ExitStack() as stack:
          self.stack = stack
          stack.enter_context(self.global_context)
          script_name = sys.argv[0]
          plog_filename='/tmp/opa_plog_{}_{}.log'.format(os.path.basename(script_name), flags.runid)

          plog_file = open(plog_filename, 'w')
          stack.enter_context(plog_file)
          flags.plog_file = plog_file

          if cache:
            stack.enter_context(cache)
          main_func()

app = App()

