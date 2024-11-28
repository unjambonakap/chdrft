#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
from chdrft.utils.fmt import Format
import re
import tempfile
import glog
import subprocess as sp

global flags, cache
flags = None
cache = None


class GnuRadioFile:

  def __init__(self, filename):
    self.content = open(filename, 'r').read()
    self.filename = filename
    self.var_pattern = re.compile(
        'self.(?P<var_name>\w+) = (?P<var_name2>\w+) = (?P<var_default>.+)')
    self.var_block_pattern = re.compile('# Variables(.*)?\n\n', re.MULTILINE | re.DOTALL)

    self.variables = self.find_variables()

  def find_variables(self):
    res = re.findall(self.var_block_pattern, self.content)

    assert len(res) == 1
    res = res[0]
    x = Attributize()  # base OrderedDict
    for entry in re.finditer(self.var_pattern, res):
      m = Attributize(entry.groupdict())
      assert m.var_name == m.var_name2
      x[m.var_name] = m.var_default
    return x

  @staticmethod
  def FileSuffixToType(suffix):
    if suffix == 'f':
      return 'float32'
    elif suffix == 'c':
      return 'complex64'
    else:
      assert False, 'bad suffix %s' % suffix

  def get_file_data(self, prefix):
    for name, _ in self.variables.items():
      res = Format(name).strip_prefix(prefix).v
      if res is not None:
        return name, GnuRadioFile.FileSuffixToType(res)

  def get_output_data(self):
    return self.get_file_data('output_file_')

  def get_input_data(self):
    return self.get_file_data('input_file_')

  def udpate_vars(self, params):
    print(self.variables)
    for k, v in params.items():
      assert k in self.variables
      self.variables[k] = v

  def get_code(self, params=None):
    if params is not None:
      self.udpate_vars(params)

    def repl_func(x):
      m = Attributize(x.groupdict())
      if m.var_name in self.variables:
        v = self.variables[m.var_name]
        if m.var_name.find('_file') != -1: v= f"'{v}'"
        return 'self.{0} = {0} = {1}'.format(m.var_name, v)
      else:
        return x.group(0)

    res = self.content
    res = re.sub(self.var_pattern, repl_func, res)

    # don't forget to put run to completion instead of prompt for exit
    #res = re.sub('raw_input(.*)', 'pass', res) 
    return res

  def run(self, params=None):
    code = self.get_code(params)
    glog.debug('%s', code)
    with tempfile.NamedTemporaryFile(mode='w') as fil:
      glog.debug('Running gnuradio script at %s', fil.name)
      fil.write(code)
      fil.flush()
      sp.check_call(['python2', fil.name])

  @staticmethod
  def Run(filename, params):
    GnuRadioFile(filename).run(params)


def args(parser):
  clist = CmdsList().add(test)
  parser.add_argument('--file', type=str)
  parser.add_argument('--mod_params', type=Attributize.FromYaml)
  parser.add_argument('--dry_run', action='store_true')
  ActionHandler.Prepare(parser, clist.lst)


def test(ctx):
  fil = GnuRadioFile(flags.file)
  return fil.run(params=flags.mod_params)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
