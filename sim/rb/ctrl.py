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
from chdrft.sim.rb.rb_gen import SimulHelper, ControlInputState, ControlParameters, ControlSpec, Simulator, NumpyPacker
import pygmo as pg

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class SolverParameters(cmisc.PatchedModel):
  time_ctrl_count: int = 0 # big failure


class RBSolver(cmisc.PatchedModel):

  spec: ControlSpec
  inner_state: np.ndarray = None
  solver_params: SolverParameters = Field(default_factory=SolverParameters)

  @cmisc.cached_property
  def time_ctrl_packer(self) -> NumpyPacker:
    res = NumpyPacker()
    res.add_dict(dt_ctrl=(self.solver_params.time_ctrl_count, 1 + self.spec.mdata.nctrl))
    return res

  @property
  def nctrl(self) -> int:
    return self.spec.mdata.nctrl

  @property
  def use_time_ctrl(self) -> bool:
    return self.solver_params.time_ctrl_count > 0

  @cmisc.cached_property
  def simul_helper(self) -> SimulHelper:
    return SimulHelper.Get(self.spec)

  @property
  def dim(self):
    if self.use_time_ctrl:
      # (dt, nctrl) * timectrlcount + last dt
      return self.time_ctrl_packer.pos

    return self.params.nt * self.nctrl

  @property
  def params(self) -> ControlParameters:
    return self.spec.cparams

  def get_nec(self):
    return 0

  def norm_input_ctrl(self, ctrl, pack=True):
    if self.use_time_ctrl:
      a = self.time_ctrl_packer.unpack(ctrl)
      ctrl = self.params.from_delta_time_annotated(a.dt_ctrl)
    else:
      ctrl = ctrl.reshape((-1, self.nctrl))

    if pack:
      ctrl = self.spec.ctrl_sim_packer.pack(ctrl)
    return ctrl

  def fitness(self, ctrl):
    ctrl = self.norm_input_ctrl(ctrl)
    return self.simul_helper.eval(ctrl, self.inner_state),

  def batch_fitness(self, data):

    data = data.reshape((-1, self.dim))
    res = []
    for x in data:
      res.append(self.norm_input_ctrl(x))
    data = np.array(res)

    if not self.params.enable_batch:
      res = []
      for x in data:
        res.append(self.simul_helper.eval(x, self.inner_state))
      res = np.array(res)
    else:
      s0 = np.broadcast_to(self.inner_state, (len(data),) + self.inner_state.shape)
      res = self.simul_helper.dispatcher(data, s0)
    return res.reshape((-1,))

    assert 0

  def gradient(self, ctrl):
    #return jac_func(ctrl)[0]
    return pg.estimate_gradient_h(lambda x: self.fitness(x), ctrl)

  def get_bounds(self):
    bounds = np.array(self.params.ctrl_bounds)
    if len(bounds.shape) == 1:
      res = (np.full((self.dim,), bounds[0]), np.full((self.dim,), bounds[1]))
      if self.use_time_ctrl:
        res[0][0] = self.params.dt_ctrl
        res[1][0] = self.params.end_t
    else:
      res = np.zeros((2, self.dim))

      if self.use_time_ctrl:
        res[0][0] = self.params.dt_ctrl
        res[1][0] = self.params.end_t
      res[:, 1:] = bounds.T
    return res

  # not reentrant, storign state in inner_state
  def solve(
      self,
      state: ControlInputState | np.ndarray,
      parallel=0,
      nlopt_params={},
      ctrl_startup: np.ndarray = None
  ):
    self.inner_state = self.spec.make_packed(state)

    prob = pg.problem(self)
    nl = pg.nlopt('slsqp')
    algo = pg.algorithm(pg.nlopt("slsqp"))
    nl.ftol_rel = 1e-4
    nl.ftol_abs = 1e-4
    nl.maxeval = 100
    for k, v in nlopt_params.items():
      setattr(nl, k, v)

    ip = pg.ipopt()
    ip.set_numeric_option("tol", 1E-3)  # Change the relative convergence tolerance
    ip.get_numeric_options()

    algo = pg.algorithm(nl)
    algo.set_verbosity(1)

    #uda = pg.mbh(algo, stop = 2, perturb = .2)
    #algo = pg.algorithm(uda = uda)

    prob.c_tol = [1e-3] * prob.get_nc()
    if parallel:
      assert not ctrl_startup, 'not supported'
      print('start')
      archi = pg.archipelago(n=8, algo=algo, prob=prob, pop_size=200, seed=32, b=pg.bfe())
      archi.evolve()
      archi.wait()
      score, raw_ans = min(
          zip(archi.get_champions_f(), archi.get_champions_x()), key=lambda fx: fx[0]
      )

      #isl = pg.island(prob=prob, size=2000, b=pg.bfe(), algo=algo, udi=pg.mp_island())
      #isl.evolve()
      #isl.wait_check()
      #raw_ans = isl.get_population().champion_x
      #pop = isl.get_population()

    else:
      pop = pg.population(prob, size=200, b=pg.bfe())
      if ctrl_startup is not None:
        pop.push_back(self.spec.ctrl_sim_packer.pack(dict(ctrl=ctrl_startup)))

      pop = algo.evolve(pop)
      score, raw_ans = pop.champion_f, pop.champion_x


    ctrl = self.norm_input_ctrl(raw_ans, pack=0)
    if self.use_time_ctrl:
      tctrl = self.time_ctrl_packer.unpack(raw_ans).dt_ctrl
    else:
      tctrl = self.params.to_delta_time_annotated(ctrl)
    return A(algo=algo, prob=prob, raw_ans=raw_ans, ctrl=ctrl, tctrl=tctrl, score=score)


class MPCOutputEntry(cmisc.PatchedModel):
  t: float
  ctrl: np.ndarray


class MPCOutput(cmisc.PatchedModel):
  entries: list[MPCOutputEntry] = Field(default_factory=list)
  ctrl_spec: ControlSpec

  def get_ctrl(self, t: float) -> np.ndarray | None:
    last = np.zeros(self.ctrl_spec.mdata.nctrl)
    for x in self.entries:
      if x.t > t: return last
      last = x.ctrl
    return last

  def get_ctrl_for_sim(self, t: float, cparams: ControlParameters) -> np.ndarray:
    res = []
    for i in range(cparams.nt):
      res.append(self.get_ctrl(t + i * cparams.dt_ctrl))
    return np.array(res)

  def push_sol(self, t: float, ctrl: np.ndarray):
    while len(self.entries) > 0 and self.entries[-1].t >= t:
      self.entries.pop()
    for i in range(self.ctrl_spec.mdata.nctrl):
      self.entries.append(MPCOutputEntry(t=t + i * self.ctrl_spec.cparams.dt_ctrl, ctrl=ctrl[i]))


class MPCModel(cmisc.PatchedModel):
  state: np.ndarray = None
  t: float = None


class MPCParameters(cmisc.PatchedModel):
  dt_predict_delay: float
  dt_refresh_ctrl: float


class MPCController(cmisc.PatchedModel):
  model: MPCModel = Field(default_factory=MPCModel)
  output: MPCOutput = None
  params: MPCParameters
  spec_predict: ControlSpec = None
  sh_predict: SimulHelper = None
  solver: RBSolver
  t_next_refresh: float = -1

  @property
  def spec_ctrl(self) -> ControlSpec:
    return self.solver.spec

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.output = MPCOutput(ctrl_spec=self.solver.spec)
    cparams = self.solver.spec.cparams
    cparams_predict = ControlParameters(
        dt=0,
        ndt_const=4,
        integ_nsteps=2,
        nt=10,
    )
    cparams_predict.end_t = self.params.dt_predict_delay

    nspec = ControlSpec(mdata=self.solver.spec.mdata, cparams=cparams_predict)
    self.spec_predict = nspec
    self.sh_predict = SimulHelper(spec=self.spec_predict)

  def predict_state(self) -> np.ndarray:
    return np.array(
        self.sh_predict.get_end_state(
            self.output.get_ctrl_for_sim(self.model.t, self.spec_predict.cparams), self.model.state
        )[0]
    )

  def compute(self):
    state = self.predict_state()

    t = self.model.t + self.params.dt_predict_delay
    start_ctrl = self.output.get_ctrl_for_sim(t, self.spec_ctrl.cparams)
    res = self.solver.solve(state, parallel=0, ctrl_startup=start_ctrl)
    self.output.push_sol(t, res.ctrl)

  def get_active_output(self) -> np.ndarray:
    return self.output.get_ctrl(self.model.t)

  def cb(self, t: float, state: np.ndarray):
    # should really be push sensor data
    self.model.t = t
    self.model.state = state

    if self.t_next_refresh < t:
      self.compute()
      self.t_next_refresh = t + self.params.dt_refresh_ctrl


class SystemSimulationParameters(cmisc.PatchedModel):
  dt_integ: float


class SystemSimulator(cmisc.PatchedModel):
  controller: MPCController
  real_simulator: SimulHelper
  params_sim: SystemSimulationParameters
  t: float = 0

  def do_sim_step(self):
    # drawback, controller and sim timesteps are interlocked
    dt = self.params_sim.dt_integ
    self.real_simulator.integrate1(dt, (self.controller.get_active_output(),))
    self.t += dt
    self.controller.cb(self.t, self.real_simulator.sim.dump_state0())


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
