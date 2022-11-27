#!/usr/bin/env python

import chdrft.config.env
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.types import *
from pydantic import Field
from chdrft.utils.path import FileFormatHelper
from chdrft.utils.fmt import Format
from chdrft.sim.rb.rb_analysis import SimulHelper, ControlInputState, ControlParameters, RB_PbAnalysis
from chdrft.sim.rb.scenes import get_control_test_data
from chdrft.sim.rb.ctrl import *
import pygmo as pg

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)

def test_mpc(ctx):
  pass

nxx = 3

fspec = get_control_test_data(nxx)
ctrl_spec = fspec.spec
consts = fspec.consts
ctrl_spec.cparams = ControlParameters(
    dt=1e-2,
    ndt_const=10,
    integ_nsteps=1,
    nt=10,
)

i_state = consts.state
end_conds = consts.end_conds

mpc_params = MPCParameters(dt_predict_delay=1e-1, dt_refresh_ctrl=2e-1)
solver = RBSolver(spec=ctrl_spec)
controller = MPCController(params=mpc_params, solver=solver)

params_sim = SystemSimulationParameters(dt_integ=1e-3,)
ss = SystemSimulator(
    controller=controller,
    real_simulator=SimulHelper(spec=ctrl_spec),
    params_sim=params_sim,
)
ss.do_sim_step()



def test_control(ctx):
  nxx = 3
  cparams = ControlParameters(
      dt=2e-2,
      ndt_const=10,
      integ_nsteps=1,
      nt=10,
  )

  fspecs = get_control_test_data(nxx)
  fspecs.spec.cparams = cparams
  #px.integrate(px.args0[0], px.args0[1:], 1)
  consts = fspecs.consts
  fspecs.spec.cparams = ControlParameters(
      dt=2e-2,
      ndt_const=10,
      integ_nsteps=1,
      nt=10,
  )
  i_state = consts.state
  end_conds = consts.end_conds

  solver = RBSolver(spec=fspecs.spec)
  res = solver.solve(
      i_state,
      parallel=1,
      nlopt_params=dict(maxeval=30),
  )

  FileFormatHelper.Write(
      './res.pickle',
      A(ans=res.ans, spec=spec, end_conds=end_conds, i_state=i_state, ctrl=res.ctrl)
  )

  print(solver.fitness(res.raw_ans))
  print(solver.simul_helper.proc(i_state, res.raw_ans))



def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
