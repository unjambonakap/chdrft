#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.types import *
import PyFoam
from chdrft.interactive.scripter import get_bash_scripter
from contextlib import ExitStack
import os


global flags, cache
flags = None
cache = None

from PyFoam.Applications.PrepareCase import PrepareCase
from PyFoam.Applications.CloneCase import CloneCase
from PyFoam.Applications.Runner import Runner
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.Execution.ConvergenceRunner import ConvergenceRunner
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer

def args(parser):
  clist = CmdsList()
  parser.add_argument('--ofoam-dir', default=os.environ.get('WM_PROJECT_DIR', None))
  ActionHandler.Prepare(parser, clist.lst, global_action=1)



class Case(ExitStack):
  def __init__(self, ctx, name, dir=None):
    super().__init__()
    self.name = name
    self.ctx = ctx
    if dir is None: dir = cmisc.cwdpath(name)
    self.interface = get_bash_scripter(cwd=dir)
    self.soldir = SolutionDirectory(self.name)

  def __enter__(self):
    super().__enter__()
    self.enter_context(self.interface)
    self.interface.run(f'export PATH=$PATH:{self.ctx.ofoam_dir}/bin/tools')

  def run(self, cmdname, args=''):
    if not isinstance(args, str):
      args = ' '.join(args)
    return self.interface.run_and_wait(f'{cmdname} {args}')

  def clean(self):
    self.soldir.clear()
    cmisc.failsafe(lambda: self.soldir.rmtree(self.get_dir('postProcessing')))

  def refresh():
    self.soldir.reread()

  def get_dir(self, *args):
    return os.path.join(self.soldir.name, *args)
#include        "include/initialConditions"


  def prepare(self):
    self.soldir.rmtree(self.soldir.polyMeshDir())
    self.run('blockMesh')
    self.run('surfaceFeatureExtract')
    self.run('snappyHexMesh', '-overwrite')
    self.soldir.clearResults()

  def run(self, solver='simpleFoam', *args):
    run=ConvergenceRunner(BoundingLogAnalyzer(),argv=[solver,"-case",self.name, *args],silent=True)
    run._writeStopAt('endTime', 'reset write stop')
    run.start()
    self.last_run = run
def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
