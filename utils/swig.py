#!/usr/bin/env python

from chdrft.main import app


class Swig:

  def __getattr__(self, x):
    if not self._initialized:
      self.setup()

    if not self._unsafe and not x in self.attrs:
      raise AttributeError('Bad attr %s, only have %s' % (x, self.attrs))
    res = __import__(x)
    setattr(self, x, res)
    self.init()
    return res

  def __init__(self, unsafe=False):
    super().__setattr__('_initialized', False)
    super().__setattr__('_unsafe', unsafe)

  def init(self):
    from opa_common_swig import opa_init_swig
    args  = ['dummy'] + self.args
    args = list(map(str.encode, args))
    opa_init_swig(args)

  def setup(self, args=None):
    if args is None:
      args = []
      #args = ['--primedb_maxv=1000']
      if app.flags is not None:
        args = app.flags.other_args
    self.args = args

    self.attrs = [
        'opa_or_swig',
        'opa_common_swig',
        'opa_threading_swig',
        'opa_crypto_swig',
        'opa_crypto_linear_cryptanalysis_swig',
        'opa_math_common_swig',
        'opa_math_game_swig',
        'opa_algo_swig',
        'opa_engine_swig',
        'opa_wrapper_swig',
        'opa_dsp_swig',
        'opa_test_swig',
    ]
    self._initialized = True


swig = Swig()
swig_unsafe = Swig(unsafe=True)
