import argparse
import jsonpickle
import chdrft.utils.misc as cmisc
import os
import sys
import shutil
import traceback as tb
import hashlib
import fnmatch

global global_cache
global_cache = None
global_cache_list = []

global_fileless_cache = None

def dump_caches():
  print('DUMP CACHES')
  for x in global_cache_list:
    x.do_exit_entry()


class CacheDB(object):
  default_cache_filename = 'chdrft.cache'
  default_conf = 'default'

  def __getstate__(self):
    return dict()

  def __setstate__(self, state):
    pass

  def __init__(self, conf=None, recompute_set=set()):
    global global_cache
    global_cache = self

    if not conf:
      conf = self.default_conf

    self._conf = conf
    self._cache = None
    self._confobj = None
    self._active = False
    self._recompute_set = recompute_set
    self._json_util = cmisc.JsonUtils(cmisc.misc_backend)

  def do_exit_entry(self):
    if not self._active: return
    self._active = False
    self.do_exit()

  def do_enter(self):
    assert 0

  def do_exit(self):
    assert 0

  def __enter__(self):
    self._active = True
    self._cache = self.do_enter()
    self.set_active_conf(self._conf)
    return self

  def __exit__(self, typ, value, tcb):
    self.do_exit_entry()

  def set_active_conf(self, conf):
    if conf is None:
      return
    self._conf = conf
    if not conf in self._cache:
      self._cache[conf] = cmisc.Dict({})

    self._confobj = self._cache[conf]

  def __getattr__(self, name):
    if name.startswith('_'):
      return object.__getattribute__(self, name)
    else:
      if not name in self._confobj:
        raise AttributeError
      return self._confobj[name]

  def __setattr__(self, name, val):
    if name.startswith('_'):
      super().__setattr__(name, val)
    else:
      self._confobj[name] = val

  def __setitem__(self, name, val):
    return self.__setattr__(name, val)

  def __getitem__(self, name):
    return self._confobj[name]

  def __contains__(self, key):
    return key in self._confobj

  def list_conf(self):
    return self._cache.keys()

  def clean(self):
    self._confobj.clear()

  def get_str2(self, obj):
    res = self._json_util.encode(obj)
    return res

  def get_str(self):
    return self._json_util.encode(self._confobj)

  def get_cachable(self, key, func, *args, **kwargs):
    if not key in self:
      res = func(*args, **kwargs)
      self[key] = res
    return self[key]


class FilelessCacheDB(CacheDB):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def do_enter(self):
    return {}

  def do_exit(self):
    pass


class FileCacheDB(CacheDB):

  @staticmethod
  def load_from_argparse(args):
    return FileCacheDB(
        filename=args.cache_file,
        conf=args.cache_conf,
        recompute=args.recompute_all,
        recompute_set=set(args.recompute))

  def __init__(self, filename=None, recompute=False, **kwargs):
    super().__init__(**kwargs)
    self._recompute = recompute

    if not filename:
      filename = self.default_cache_filename
    self._cache_filename = filename

  def do_enter(self):
    if not os.path.exists(self._cache_filename):
      with open(self._cache_filename, 'w') as f:
        f.write(self._json_util.encode({}))

    with open(self._cache_filename, 'r') as f:
      if self._recompute:
        cache = {}
      else:
        cache = self._json_util.decode(f.read())
    return cache

  def do_exit(self):
    from chdrft.main import app
    if app.flags and app.flags.no_update_cache: return
    shutil.copy2(self._cache_filename, self._cache_filename + '.bak')

    try:
      with open(self._cache_filename, 'w') as f:
        f.write(self._json_util.encode(self._cache))
    except:
      os.remove(self._cache_filename)
      tb.print_exc()


class Cachable:
  ATTR = '__opa_cache'

  def __getstate__(self):
    return dict()

  def __setstate__(self, state):
    pass

  def __init__(self, cache=None):
    if not cache:
      cache = global_cache
    if cache:
      setattr(self, Cachable.ATTR, cache)

  def _cachable_get_cache(self):
    return getattr(self, Cachable.ATTR)

  @staticmethod
  def _cachable_get_key(clazz, func, alt_key):
    if alt_key:
      return alt_key
    return '{}#{}'.format(clazz.__class__.__name__, func.__name__)

  @staticmethod
  def _cachable_get_key_func(func, alt_key):
    if alt_key:
      return alt_key
    return '__func#{}'.format(func.__name__)

  @staticmethod
  def _cachable_get_full_key(cache, key, *args, **kwargs):
    return cache.get_str2([key, args, kwargs])

  @staticmethod
  def proc_cached(cache, key, f, *args, **kwargs):
    full_key = Cachable._cachable_get_full_key(cache, key, *args, **kwargs)
    if (not key in cache._recompute_set) and full_key in cache:
      return cache[full_key]

    #print('NOT CACHED >> ', hashlib.sha1(full_key.encode()).hexdigest())
    res = f(*args, **kwargs)

    if cache:
      from chdrft.main import app
      if not app.flags or not app.flags.disable_cache: cache[full_key] = res
      # if res is modified, can be source of fuck ups

    return res

  @staticmethod
  def cached(alt_key=None):

    def cached_wrap(f):

      def cached_f(self, *args, **kwargs):
        cache = None
        key = Cachable._cachable_get_key(self, f, alt_key)
        if isinstance(self, Cachable) and self._cachable_get_cache():
          cache = self._cachable_get_cache()

        return Cachable.proc_cached(cache, key, f, self, *args, **kwargs)

      return cached_f

    return cached_wrap

  @staticmethod
  def cachedf(alt_key=None, fileless=True):

    def cached_wrap(f):

      def cached_f(*args, **kwargs):
        if fileless:
          global global_fileless_cache
          if global_fileless_cache is None:
            global_fileless_cache = FilelessCacheDB()
            from chdrft.main import app
            app.global_context.enter_context(global_fileless_cache)
          cache = global_fileless_cache
        else: cache = global_cache
        key = Cachable._cachable_get_key_func(f, alt_key)
        return Cachable.proc_cached(cache, key, f, *args, **kwargs)

      return cached_f

    return cached_wrap

  @staticmethod
  def cached2(alt_key=None, cache_filename=None):

    def cached_wrap(f):

      def cached_f(self, *args, **kwargs):
        key = Cachable._cachable_get_key(self, f, alt_key)

        if not hasattr(self, Cachable.ATTR):

          if cache_filename:
            cache = FileCacheDB(
                cache_filename)
          else:
            cache = FilelessCacheDB()

          from chdrft.main import app
          app.global_context.enter_context(cache)
          setattr(self, Cachable.ATTR, cache)
        cache = getattr(self, Cachable.ATTR)
        return Cachable.proc_cached(cache, key, f, self, *args, **kwargs)

      return cached_f

    return cached_wrap


def clean_cache(args):
  with FileCacheDB.load_from_argparse(args) as cache:
    keys = list(cache._confobj.keys())
    if not args.cache_filters: return
    keys = cmisc.filter_glob_list(keys, args.cache_filters)
    for k in keys:
      del cache._confobj[k]


def print_cache(args):
  with FileCacheDB.load_from_argparse(args) as cache:
    keys = cache._confobj.keys()
    if args.cache_filters:
      keys = cmisc.filter_glob_list(keys, args.cache_filters)
    for k in keys:
      print('K={} >> {}'.format(k, cache[k]))


def cache_argparse(parser):
  parser.add_argument('--cache-conf', type=str, default=cmisc.cwdpath(CacheDB.default_conf))
  parser.add_argument(
      '--cache-file', type=cmisc.cwdpath, default=cmisc.cwdpath(CacheDB.default_cache_filename))
  parser.add_argument('--recompute-all', action='store_true')
  parser.add_argument('--recompute', type=lambda x: x.split(','), default=[])
  parser.add_argument('--disable-cache', action='store_true')
  parser.add_argument('--no-update-cache', action='store_true')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  cache_argparse(parser)

  sp = parser.add_subparsers()
  clean_parser = sp.add_parser('clean')
  clean_parser.set_defaults(func=clean_cache)
  clean_parser.add_argument('--cache-filters', nargs='*')

  print_parser = sp.add_parser('print')
  print_parser.set_defaults(func=print_cache)
  print_parser.add_argument('--cache-filters', nargs='*')
  args = parser.parse_args()

  args.func(args)
