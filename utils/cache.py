import argparse
import chdrft.utils.misc as cmisc
import os
import shutil
import traceback as tb
import glog
from chdrft.utils.path import FileFormatHelper

global global_cache
global_cache = None
global_cache_list = []

global_fileless_cache = None

default_cache_filename = '.chdrft.cache.pickle'

def get_default_cache_filename():
  return os.path.join(os.environ.get('OPA_CACHE_DIR', cmisc.cwdpath('./')), default_cache_filename)


def dump_caches():
  print('DUMP CACHES')
  for x in global_cache_list:
    x.do_exit_entry()

def enabled_cache():
  from chdrft.main import app
  return app.flags and app.flags.enable_cache

def update_cache():
  from chdrft.main import app
  return app.flags and app.flags.enable_cache and not app.flags.no_update_cache

class CacheDB(object):
  default_conf = 'default'

  def __getstate__(self):
    return dict()

  def __setstate__(self, state):
    pass

  def __init__(self, conf=None, recompute_set=set()):

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
    if isinstance(name, str) and name.startswith('_'):
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

  def clear(self):
    self._confobj.clear()

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
        recompute_set=set(args.recompute)
    )

  def __init__(self, filename=None, recompute=False, ro=0, **kwargs):
    super().__init__(**kwargs)
    self._recompute = recompute
    self._ro = ro

    if not filename:
      filename = get_default_cache_filename()
    self._cache_filename = filename

  def write_cache(self, content):
    from chdrft.main import app
    if self._ro: return
    if app.flags and not app.flags.enable_cache: return
    glog.info('Writing cache ')

    FileFormatHelper.Write(self._cache_filename, content, default_mode='pickle')
    glog.info('Done Writing cache ')

  def read_cache(self, f):
    glog.info('Reading cache ')
    res = FileFormatHelper.Read(self._cache_filename, default_mode='pickle')
    glog.info('Done reading cache ')
    return res

  def do_enter(self):
    if not os.path.exists(self._cache_filename):
      if self._ro: return {}

      self.write_cache({})

    if not enabled_cache():
      cache = {}
    else:
      with open(self._cache_filename, 'rb') as f:
        if self._recompute:
          cache = {}
        else:
          cache = self.read_cache(f)
    return cache

  def flush_cache(self):

    if not update_cache(): return
    bak = self._cache_filename + '.bak'
    shutil.copy2(self._cache_filename, bak)

    try:
      self.write_cache(self._cache)
    except:
      os.remove(self._cache_filename)
      shutil.copy2(bak, self._cache_filename)
      tb.print_exc()

  def do_exit(self):
    self.flush_cache()


class Cachable:
  ATTR = '__opa_cache'
  ATTR_CACHEOBJ_CONF = '__opa_cacheobj_conf'

  @staticmethod
  def ConfigureCache(obj, fields=None):
    if not hasattr(obj, Cachable.ATTR_CACHEOBJ_CONF):
      setattr(obj, Cachable.ATTR_CACHEOBJ_CONF, cmisc.Attr())
    cacheobj_conf = getattr(obj, Cachable.ATTR_CACHEOBJ_CONF)
    cacheobj_conf.fields = fields

  @staticmethod
  def GetCacheObj(obj):
    return getattr(obj, Cachable.ATTR_CACHEOBJ_CONF, cmisc.Attr())

  def __getstate__(self):
    return dict()

  def __setstate__(self, state):
    pass

  def __init__(self, cache=None):
    if not cache:
      cache = Cachable.GetCache(fileless=False)
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
  def _cachable_get_full_key(
      cache,
      key,
      args,
      kwargs,
      self=None,
      opa_fields=None,
      args_serializer=None,
    key_serializer=None,
  ):
    if self is not None:
      cacheobj = Cachable.GetCacheObj(self)
      if opa_fields is None: opa_fields = cacheobj.get('fields', None)

    if opa_fields is not None:
      selfdata = []
      for field in opa_fields:
        selfdata.append(getattr(self, field))
      return cache.get_str2([key, selfdata, args, kwargs])

    if args_serializer is not None:
      args, kwargs = args_serializer(*args, **kwargs), None
    if self is not None: args = (id(self),) + tuple(args)
    if key_serializer is not None:
      return key_serializer(key, args, kwargs)
    return cache.get_str2([key, args, kwargs])

  @staticmethod
  def proc_cached(
      cache,
      key,
      f,
      args,
      kwargs,
      self=None,
      opa_fields=None,
      opa_fullkey=None,
      id_self=0,
      args_serializer=None,
      key_serializer=None,
  ):
    iself = self
    if self is not None and id_self: iself = id(self)

    full_key = opa_fullkey
    if full_key is None:
      full_key = Cachable._cachable_get_full_key(
          cache,
          key,
          args,
          kwargs,
          self=iself,
          opa_fields=opa_fields,
          args_serializer=args_serializer,
      key_serializer=key_serializer,
      )
    elif not isinstance(full_key, str):
      full_key = cache.get_str2(full_key)

    if (not full_key
        in cache._recompute_set) and (not key in cache._recompute_set) and full_key in cache:
      return cache[full_key]

    if self is not None: res = f(self, *args, **kwargs)
    else: res = f(*args, **kwargs)

    if cache:
      from chdrft.main import app
      if not app.flags or not app.flags.disable_cache:
        cache[full_key] = res
      # if res is modified, can be source of fuck ups

    return res

  @staticmethod
  def cached(alt_key=None, fullkey=None, fields=None):

    def cached_wrap(f):

      def cached_f(self, *args, **kwargs):
        cache = None
        key = Cachable._cachable_get_key(self, f, alt_key)
        if isinstance(self, Cachable):
          cache = self._cachable_get_cache()
        if cache is None:
          cache = Cachable.GetCache(fileless=False)

        return Cachable.proc_cached(
            cache, key, f, args, kwargs, self=self, opa_fields=fields, opa_fullkey=fullkey
        )

      return cached_f

    return cached_wrap

  @staticmethod
  def GetCache(fileless):

    if fileless:
      global global_fileless_cache
      if global_fileless_cache is None:
        global_fileless_cache = FilelessCacheDB()
        from chdrft.main import app
        app.global_context.enter_context(global_fileless_cache)
      cache = global_fileless_cache
    else:
      global global_cache
      if global_cache is None:
        global_cache = FileCacheDB()
        from chdrft.main import app
        app.global_context.enter_context(global_cache)
      cache = global_cache
    return cache

  @staticmethod
  def ResetCache(fileless=True):
    Cachable.GetCache(fileless).clear()

  @staticmethod
  def cachedf(alt_key=None, fileless=True, method=False, args_serializer=None, key_serializer=None):

    def cached_wrap(f):

      def cached_f(*args, **kwargs):
        key = Cachable._cachable_get_key_func(f, alt_key)
        self = None
        if method:
          self, *args = args
        return Cachable.proc_cached(
            Cachable.GetCache(fileless),
            key,
            f,
            args,
            kwargs,
            self=self,
            args_serializer=args_serializer,
          key_serializer=key_serializer,
        )

      return cached_f

    return cached_wrap

  @staticmethod
  def cached_property():
    return Cachable.cached2(id_self=1)

  @staticmethod
  def cached2(alt_key=None, cache_filename=None, fullkey=None):

    def cached_wrap(f):

      def cached_f(self, *args, **kwargs):
        key = Cachable._cachable_get_key(self, f, alt_key)

        if not hasattr(self, Cachable.ATTR):

          if cache_filename:
            cache = FileCacheDB(cache_filename)
          else:
            cache = FilelessCacheDB()

          from chdrft.main import app
          app.global_context.enter_context(cache)
          setattr(self, Cachable.ATTR, cache)
        cache = getattr(self, Cachable.ATTR)
        return Cachable.proc_cached(cache, key, f, args, kwargs, self=self, opa_fullkey=fullkey)

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
    print('HAVE >> ', len(keys))
    for k in keys:
      print('K={} >> {}'.format(k, cache[k]))


class Proxifier:

  def __init__(self, cl, *args, **kwargs):
    super().__setattr__('_opa_obj', cl(*args, **kwargs))
    super().__setattr__('_opa_key', dict(args=args, kwargs=kwargs))

  def __getattr__(self, name):
    res = getattr(self._opa_obj, name)
    if callable(res):

      @Cachable.cached(fullkey=self._opa_key)
      def interceptor(*args, **kwargs):
        return res(*args, **kwargs)

      return interceptor
    return res

  def __setattr__(self, name, val):
    if name.startswith('_'):
      super().__setattr__(name, val)
    else:
      self._confobj[name] = val


def cache_argparse(parser):
  parser.add_argument('--cache-conf', type=str, default=CacheDB.default_conf)
  parser.add_argument(
      '--cache-file', type=cmisc.cwdpath, default=get_default_cache_filename()
  )
  parser.add_argument('--recompute-all', action='store_true')
  parser.add_argument('--recompute', type=lambda x: x.split(','), default=[])
  parser.add_argument('--enable-cache', action='store_true')
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
