#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, get_n2_locals_and_globals


import glog

global flags, cache
flags = None
cache = None


def args(parser):
  parser.add_argument('--data', type=str)
  clist = CmdsList().add(test_parser).add(test_lazyconf)
  ActionHandler.Prepare(parser, clist.lst)


import decimal


class ConfParser:

  def __init__(self, **kwargs):

    # Build the lexer
    import ply.lex as lex
    self.lexer = lex.lex(module=self, debug=0)
    import ply.yacc as yacc
    self.parser = yacc.yacc(module=self, debug=0)

  def parse(self, data, ctx_locals=None, ctx_globals=None):
    if ctx_locals is None: ctx_locals ={}
    if ctx_globals is None: ctx_globals ={}
    self.ctx_locals = dict(ctx_locals)
    self.ctx_globals = dict(ctx_globals)
    import time
    self.ctx_globals['time'] = time
    self.ctx_globals['Attributize'] = Attributize

    if data is None: return None
    return self.parser.parse(data, lexer=self.lexer)

  tokens = ('NAME', 'FLOAT', 'INTEGER', 'BOOL', 'STRING', 'STRING2', 'STRINGQ', 'LPAR', 'RPAR', 'LSQPAR', 'RSQPAR', 'LBRA', 'RBRA', 'EQUALS', 'DOT', 'COMMA', 'CODE')
  #'PLUS', 'MINUS', 'TIMES', 'DIVIDE',)

  # Tokens

  #t_PLUS = r'\+'
  #t_MINUS = r'-'
  #t_TIMES = r'\*'
  #t_DIVIDE = r'/'
  t_EQUALS = r'='
  t_LPAR = r'\('
  t_RPAR = r'\)'

  t_LBRA = r'\{'
  t_RBRA = r'\}'
  t_LSQPAR = r'\['
  t_RSQPAR = r'\]'
  t_DOT = r'\.'
  t_COMMA = r','
  t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'

  def t_FLOAT(self, t):
    r'\d+[eE][-+]?\d+|(\.\d+|\d+\.\d+)([eE][-+]?\d+)?'
    t.value = float(t.value)  # Conversion to Python float
    return t

  def t_INTEGER(self, t):
    r'(0[Xx]\d+|\d+)'
    # Conversion to a Python int
    if t.value.startswith(('0x', '0X')):
      t.value = int(t.value, 16)
    elif t.value.startswith('0'):
      t.value = int(t.value, 8)
    else:
      t.value = int(t.value)
    return t

  def t_BOOL(self, t):
    r'(True|False)'
    mapping = {"True": True, "False": False}
    t.value = mapping[t.value]
    return t

  def t_STRING(self, t):
    r"'(([^']+|\\'|\\\\)*)*'"  # I think this is right ...
    t.value = t.value[1:-1]
    return t

  def t_STRING2(self, t):
    r'"([^"]+|\\"|\\\\)*"'  # I think this is right ...
    t.value = t.value[1:-1]
    return t

  def t_STRINGQ(self, t):
    r'\#\w+'  # I think this is right ...
    t.value = t.value[1:]
    return t

  def t_CODE(self, t):
    r'\@[^\@]+\@'
    t.value = eval(t.value[1:-1], self.ctx_globals, self.ctx_locals)
    return t

  # Ignored characters
  t_ignore = " \t\n"

  def t_error(self, t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

  # Precedence rules for the arithmetic operators
  #precedence = (('left', 'PLUS', 'MINUS'), ('left', 'TIMES', 'DIVIDE'), ('right', 'UMINUS'),)

  # dictionary of names (for storing variables)
  names = {}

  def p_dict_name(self, p):
    '''dict_name : dict_name DOT NAME
                | NAME'''
    if len(p) == 4: p[0] = p[1] + [p[3]]
    else: p[0] = [p[1]]

  def p_dict(self, p):
    '''dict : LBRA entrylist RBRA'''
    p[0] = p[2]

  def p_entrylist(self, p):
    '''entrylist : entry
                | entrylist COMMA entry
                |
                '''
    if len(p) == 1: p[0] = LazyConf()
    elif len(p) == 2:
      p[0] = LazyConf()
      p[0]._add_rec(*p[1])
    else:
      assert isinstance(p[1], LazyConf)
      p[0] = p[1]
      p[0]._add_rec(*p[3])

  def p_entry(self, p):
    '''entry : dict_name EQUALS value'''
    p[0] = (p[1], p[3])

  def p_value(self, p):
    '''value : BOOL
            | INTEGER
            | FLOAT
            | STRING
            | STRING2
            | STRINGQ
            | CODE
            | list
            | variable
            | tuple
            | dict'''
    p[0] = p[1]


  def p_list(self, p):
    '''list : LSQPAR vararglist RSQPAR
            | LSQPAR RSQPAR'''
    if len(p) == 3: p[0] = []
    else: p[0] = list(p[2])

  def p_variable(self, p):
    '''variable : NAME'''
    p[0] = eval(p[1], self.ctx_globals, self.ctx_locals)


  def p_tuple(self, p):
    '''tuple : LPAR vararglist RPAR
            | LPAR RPAR'''
    if len(p) == 3: p[0] = ()
    else: p[0] = tuple(p[2])

  def p_vararglist(self, p):
    """vararglist : vararglist COMMA value
                  | value"""
    if len(p) == 4: p[0] = p[1] + [p[3]]
    else: p[0] = [p[1]]

  def p_error(self, p):
    print("Syntax error at '%s'" % p.value)

  start = 'entrylist'


g_parser = ConfParser()


class LazyConf(Attributize):

  def __init__(self, conf=None):
    super().__init__(other=lambda x: LazyConf())
    super().__dict__['_has'] = False
    if conf is not None:
      self._merge(conf)

  def _do_clone(self):
    res = LazyConf()
    res._elem = super()._do_clone()
    return res

  def __getattr__(self, name):
    return self.__getitem__(name)

  def __bool__(self):
    return False

  def __add_key(self, key, val):
    self._has = True
    pass

  def _merge(self, other_conf):
    assert isinstance(other_conf, LazyConf), 'Typ is %s, val=%s' % (type(other_conf), other_conf)
    for k, v in other_conf.items():
      if (k in self) and isinstance(self[k], LazyConf):
        self[k]._merge(v)
      else:
        self[k] = v
    return self

  def _add_rec(self, keys, v):
    key = keys[0]
    if len(keys) == 1:
      assert not self[key]
      self[key] = v
    else:
      self[key]._add_rec(keys[1:], v)

  @staticmethod
  def _norm_val(v):
    if isinstance(v, bool):
      return v
    k = to_int_or_none(v)
    if k is None:
      if v.startswith("'"): v = eval(v)
      return v
    return k

  @staticmethod
  def ParseFromString(data, vars_ctx=None):
    res = LazyConf()
    res.set_from_string(data, vars_ctx=vars_ctx)
    return res

  def set_from_string(self, data, vars_ctx=None):
    if not data: return
    ctx_locals, ctx_globals = None, None
    if vars_ctx is not None:
      ctx_locals, ctx_globals = get_n2_locals_and_globals(vars_ctx+1)
    entries = g_parser.parse(data, ctx_locals, ctx_globals)
    self._merge(entries)
    #for keys, v in entries:
    #  self._add_rec(keys, v)


def test_parser(ctx):
  res = g_parser.parse("def='abcdef',kappa.jambon123='lalal33'")
  print(res)
  res = g_parser.parse(ctx.data)
  print(res)
  return res


def test_lazyconf(ctx):
  lc = LazyConf.ParseFromString(ctx.data)
  print(lc)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
