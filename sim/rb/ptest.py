#!/usr/bin/env python

import numpy as np
from chdrft.utils.opa_types import *
from chdrft.sim.rb.rb_gen import *
from chdrft.sim.rb.ctrl import *
from chdrft.sim.rb.scenes import *


def checkit(a: Vec3):
  ma = a.exp_rot()
  mb = ma.get_axis_angle().exp_rot()
  diff = ma.rot - mb.rot
  assert np.linalg.norm(diff) < 1e-5


def test_axis_angle():

  np.random.seed(0)
  if 0:
    for mul in (0, -1, 1, np.pi, np.pi * 2, 3 * 2 * np.pi):
      checkit(Vec3.X() * mul)
      checkit(Vec3.Y() * mul)
      checkit(Vec3.Z() * mul)
      for x in np.logspace(-10, 10, 200):
        checkit((Vec3.Z() + Vec3.X() * x) * mul)

  for i in range(10000):
    a = Vec3.Rand()
    checkit(a)


def compare1(sh: SimulHelper, state, fq):
  print(fq, state)
  ra = sh.runner.integrate1(sh.spec.cparams.actual_dt, fq, state, use_jit=0)
  rb = sh.runner.integrate1(sh.spec.cparams.actual_dt, fq, state, use_jit=1)
  return np.linalg.norm(ra - rb)


def test_jit_behavior():
  idx = scene_T_idx
  mdata = ModelData(func_id=idx)

  sim = mdata.create_simulator()
  sctx = sim.sctx
  ss = sctx.sys_spec
  cspec = ControlSpec(
      mdata=mdata,
      cparams=ControlParameters(ndt_const=3, dt=1e-1, integ_nsteps=1, nt=10, use_jit=1),
      end_conds=None
  )
  id = np.identity(6)

  s0 = ControlInputState(q=mdata.ss.q_desc.default, qd=10 * id[3] + 0.5 * id[5])

  sh = SimulHelper.Get(cspec)
  sh.sim.load_state(s0)

  sa = sh.sim.dump_state()
  print(compare1(sh, sa, np.random.random(ss.f_desc.pos)))


def evaluate_integ_mom(sh: SimulHelper, state: ControlInputState, et, dt, rk4) -> float:
  rate = 200
  maxv = 0

  sh.sim.load_state(state)
  mom_0 = sh.sim.root.self_link.compute_momentum(Transform.From(), SpatialVector.Vector()).data
  sh.t = 0

  while sh.t < et:
    sh.integrate1(dt, sh.sim.sctx.sys_spec.f_desc.default, rk4=rk4, use_jit=1)


    if (sh.t-dt) // (et / rate) != sh.t // (et / rate): 
      mom = sh.sim.root.self_link.compute_momentum(Transform.From(), SpatialVector.Vector()).data
      #print(sh.t, 'mom >>> ', mom)
      maxv = max(maxv, np.linalg.norm(mom-mom_0))
  return maxv


def test_momentum():
  idx = scene_T_idx
  mdata = ModelData(func_id=idx)

  cspec = ControlSpec(
      mdata=mdata,
      cparams=ControlParameters(
          ndt_const=3,
          dt=1e-1,
          integ_nsteps=1,
          nt=10,
          use_jit=1,
      ),
      end_conds=None
  )
  sh = SimulHelper.Get(cspec)
  sim = sh.sim
  ss = sim.sctx.sys_spec

  id = np.identity(6)
  s0 = ControlInputState(q=mdata.ss.q_desc.default, qd=50 * id[3] + 0.1 * id[5])


  et = 1
  for dt in np.logspace(-5, -2, 10):
    for rk4 in (0,1):
      print(dt, rk4, evaluate_integ_mom(sh, s0, et, dt, rk4))



def test_fix_momentum():
  idx = scene_T_idx
  mdata = ModelData(func_id=idx)

  cspec = ControlSpec(
      mdata=mdata,
      cparams=ControlParameters(
          ndt_const=3,
          dt=1e-1,
          integ_nsteps=1,
          nt=10,
          use_jit=1,
      ),
      end_conds=None
  )
  sh = SimulHelper.Get(cspec)
  sim = sh.sim
  ss = sim.sctx.sys_spec

  id = np.identity(6)
  s0 = ControlInputState(q=mdata.ss.q_desc.default, qd=50 * id[3] + 0.1 * id[5])

  sh.sim.load_state(s0)
  mom0 = sh.sim.mom

  dt = 1e-2
  for i in range(10):
    sh.integrate1(dt, sh.sim.sctx.sys_spec.f_desc.default, use_jit=0)

  mom1 = sh.sim.mom
  sh.sim.mom_fixer(mom0)
  mom2 = sh.sim.mom
  print((mom0-mom1).norm)
  print((mom0-mom2).norm)






test_fix_momentum()
