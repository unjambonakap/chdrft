#!/usr/bin/env python
from chdrft.cmds import Cmds, ListInput
from chdrft.utils.misc import is_python2, cwdpath
import chdrft.utils.misc as cmisc
from chdrft.main import app
from PyQt5 import QtCore, QtGui, QtWidgets


global cache
cache=None

def do_scrape(url, file_re, output_dir, num_tasks=10, recurse=False, dry_run=False):
  from chdrft.web.scrape import Scraper
  file_matcher = cmisc.PatternMatcher.fromre(file_re)
  scraper = Scraper(base_url=url,
                    file_matcher=file_matcher,
                    rec=recurse,
                    only_subfiles=True,
                    dry_run=dry_run,
                    num_tasks=num_tasks,
                    output_dir=output_dir)
  scraper.go()


def fuzzy_tester(query, namelist):
  from chdrft.utils.opa_string import FuzzyMatcher
  matcher = FuzzyMatcher()
  for x in namelist:
    matcher.add_name(x)
  res = matcher.find(query)
  print(query, namelist, res)
  print('debug >> ', matcher.debug)

def fcf_to_csv(in_file):
  from chdrft.conv.dsp import fcf_to_csv
  return fcf_to_csv(in_file)

def fcf_to_array(in_file):
  from chdrft.conv.dsp import fcf_to_array
  return fcf_to_array(in_file)


def graph_args(parser):
  parser.add_argument('--input', type=str, nargs='*', default=[])
  parser.add_argument('--mode', type=str, default='plot')
  parser.add_argument('--params', type=str)
  parser.add_argument('--create_kernel', action='store_true')
  return parser

def graph(flags):
  from chdrft.tools.graph import graph
  graph(flags)

def args(parser):
  if is_python2:
    import chdrft.opa_sage.utils
  else:
    Cmds.add(func=do_scrape, args=[str, str, str, int, bool, bool])
    Cmds.add(func=fuzzy_tester, args=[str, ListInput])

    from chdrft.tools.file_to_img import to_greyscale
    Cmds.add(func=to_greyscale, args=[cwdpath, cwdpath])

    from chdrft.elec.utils.resistor import Resistor
    Cmds.add(func=Resistor.get, args=[[str]], name='Resistor.get')

    Cmds.add(func=graph, parser_builder=graph_args)

    Cmds.add(func=fcf_to_csv, args=[cwdpath])
    Cmds.add(func=fcf_to_array, args=[cwdpath])


    import chdrft.graphics.loader as graphics_loader
    Cmds.add(func=graphics_loader.to_binary_stl, parser_builder=graphics_loader.args_def)

    import chdrft.utils.path

  from chdrft.utils.cmdify import run_cmdify

  run_cmdify(parser, Cmds.lst)

def main():
  app.flags.func(app.flags)

app()

