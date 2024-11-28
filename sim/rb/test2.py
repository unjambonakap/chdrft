#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
from chdrft.utils.misc import Attributize as A
import numpy as np
from chdrft.utils.opa_types import *
from chdrft.sim.rb.rb_gen import *
from chdrft.sim.rb.ctrl import *
from chdrft.sim.rb.scenes import *
from IPython import embed

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--use-jit', action='store_true')
  parser.add_argument('--use-rk4', action='store_true')
  parser.add_argument('--full-jit', action='store_true')
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--embed', action='store_true')
  parser.add_argument('--scene-name')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test_mpc(ctx):
  pass


def test(ctx):
  sctx = control_simple(1).sctx
  id = InverseDynamicsData.Make(sctx.sys_spec, np.zeros(7))

  ss = sctx.sys_spec
  inverse_dynamics(id, sctx.roots[0].self_link)
  ss.load_vp(np.array([0, 0, 0, 0, 0, 1, 0]), v=1)
  if 0:
    sctx.roots[0].self_link.get_particles(10000).plot(by_col=1)
    input()

  fdata = ForwardDynamicsData(fq=ss.f_desc.zeros_packed)
  res = forward_dynamics(fdata, sctx)
  print(res)


def test_box_ctrl(ctx):

  idx = simple_box
  mdata = ModelData(func_id=idx)
  end_conds = ControlInputEndConds(q=np.array([1]), qd=np.array([0]), qd_weights=0, only_end=0)
  cspec = ControlSpec(
      mdata=mdata,
      cparams=ControlParameters(ndt_const=3, dt=1e-2, integ_nsteps=1, nt=10, use_jit=0),
      end_conds=end_conds
  )

  solver = RBSolver(spec=cspec, ctrl_bounds=[-1, 1])
  i0 = ControlInputState(q=np.zeros(1), qd=np.zeros(1))
  res = solver.solve(i0)
  print(res)

  test = np.ones(cspec.ctrl_sim_packer.pos)
  print(solver.simul_helper.proc(res.raw_ans, i0))
  print(solver.simul_helper.proc(test, i0))


def test_box(ctx):
  print('fuu')
  sctx = box_scene()
  tg = sctx.roots[0].self_link
  ss = sctx.sys_spec
  id = InverseDynamicsData.Make(ss, ss.q_desc.zeros_packed)

  inverse_dynamics(id, tg)
  ss.load_vp(np.array([1]), v=1)
  if 0:
    tg.get_particles(10000).plot(by_col=1)
    input()

  print(ss.f_desc.zeros_packed)
  fdata = ForwardDynamicsData(fq=ss.f_desc.zeros_packed + 3)
  res = forward_dynamics(fdata, sctx)
  print(res)


def prepare_scene(ctx, scene_idx: str, qd0: np.ndarray = None, s0: ControlInputState = None):

  mdata = ModelData(func_id=scene_idx)

  sim = mdata.create_simulator()
  sctx = sim.sctx
  ss = sctx.sys_spec
  cspec = ControlSpec(
      mdata=mdata,
      cparams=ControlParameters(
          ndt_const=5,
          dt=1e-2,
          integ_nsteps=1,
          nt=10,
          use_jit=ctx.use_jit,
          use_rk4=ctx.use_rk4,
          full_jit=ctx.full_jit,
      ),
      end_conds=None
  )

  if qd0 is None: qd0 = mdata.ss.qd_desc.default
  if s0 is None:
    s0 = ControlInputState(q=mdata.ss.q_desc.default, qd=qd0)
  i0 = mdata.ss.state_packer.pack(s0)

  sh = SimulHelper.Get(cspec)
  sh.sim.load_state(s0)
  return sh


def func_test_scene(ctx, scene_idx, qd0: np.ndarray=None, s0: ControlInputState = None):
  sh = prepare_scene(ctx, scene_idx, qd0, s0)
  mdata = sh.spec.mdata

  nx = mdata.ss.qd_desc.pos

  root = sh.sim.rootl
  dt = 1e-2
  ss = sh.sim.ss

  for x, name in sh.sim.sd.sctx.obj2name.items():
    print(name, x.self_link.wl)

  idd = inverse_dynamics(InverseDynamicsData.Make(sh.sim.sd, mdata.ss.qd_desc.default), root).f_map
  print('>>> ', idd)

  fdata = ForwardDynamicsData(fq=ss.f_desc.default)
  res = forward_dynamics(fdata, sh.sim.sd)
  print(ss.f_desc.unpack(res.sd))
  print(res)

  mom0 = sh.sim.mom
  parts = sh.sim.rootl.get_particles(1000000)
  print(parts.linear_momentum())
  print(parts.angular_momentum())
  print('com')
  print(parts.com)
  print(root.agg_com)
  print('mom')
  print(mom0.around(root.agg_com))
  print(mom0)

  if ctx.render:
    root.get_particles(10000).plot(by_col=1)
    input()
    return
  if ctx.embed:
    embed()
    return

  ctrl = np.zeros(sh.spec.mdata.nctrl)
  stats = []
  print(ctrl)
  for i in range(int(100)):
    fq = ctrl
    fq = fq * (1 + np.random.rand(*fq.shape) * 0e-2)
    print('\n' * 5)
    sh.integrate1(dt, fq)
    a = sh.sim.ss.state_packer.unpack(sh.sim.dump_state())
    mom = sh.sim.mom
    entry = A(t=sh.t, com=root.agg_com, mom=mom, v_com=sh.sim.est_v_com)
    print(entry)
    stats.append(entry)
    #sh.fix_mom(mom0)

  return stats


def test_scenes(ctx):
  #return func_test_scene(ctx, two_mass_idx, np.array([1,0,0,0,0,0]))
  return func_test_scene(ctx, scene_T_idx, qd0=np.array([0, 0, 0, 0, 0, 0.01]))
  return func_test_scene(ctx, scene_gyro_wheel_idx)

  if 0:
    return func_test_scene(
        ctx,
        pivot_idx,
        np.array([0]),
        s0=ControlInputState(q=np.array([0.1]), qd=np.array([0]))
    )
  return func_test_scene(
      ctx,
      inv_pendulum_idx,
      np.array([0, 0]),
      s0=ControlInputState(q=np.array([0, 0.1]), qd=np.array([0, 0.0]))
  )
  return func_test_scene(ctx, control_simple_idx3, np.array([0, 0, 0, 0, 1, 1, 1, -3, 2]))
  return func_test_scene(ctx, control_simple_idx1, np.array([0, 0, 0, 0, 0, 1, 1]))


def test_control_simple1(ctx):
  qd0 = np.array([0, 0, 0, 0, 0, 0, 1])
  qd0 = np.array([0, 0, 0, 0, 1, 1, 3])
  qd0 = np.array([0, 0, 0, 10, 0, 0, 0])
  qd0 = np.array([0, 0, 0, 1, -3, -2, 0, 0, 0])
  qd0 = np.array([0, 0, 0, 0, 0, 0, 0.1, -3, 1])
  qd0 = np.array([0, 0, 0, 0, 0, 1, 1])
  idx = control_simple_idx1


def func_control(ctx, scene_idx, s0: ControlInputState, conf_end_conds, cparams = None):
  sh = prepare_scene(ctx, scene_idx, s0=s0)
  ss = sh.sim.ss

  if cparams is None: 
    cparams = ControlParameters(
      ndt_const=3,
      dt=1e-1,
      integ_nsteps=1,
      nt=23,
      use_jit=ctx.use_jit,
      use_rk4=ctx.use_rk4,
      full_jit=ctx.full_jit,
      ctrl_bounds=np.array([-20, 20]) * 2,
    )
  sh.spec.cparams = cparams
  sh.spec.end_conds = conf_end_conds(ss)
  sh.spec.cparams.enable_batch = 0
  print('SCEEENE ', sh.sim.sctx.obj2name.values())
  print('END CONDS >> ', sh.spec.end_conds)
  solver = RBSolver(spec=sh.spec)

  nop_res = sh.proc(sh.spec.ctrl_sim_packer.default, s0)
  print('NOP RES >> ',nop_res)

  if 0:
    from IPython import embed
    embed()

  res = solver.solve(s0, parallel=0, nlopt_params=dict(maxeval=200))
  print(res)
  print('END STATE >> ', sh.get_end_state(res.raw_ans, s0))
  return


def test_ctrl_inv(ctx):

  def conf_end_conds(ss: RBSystemSpec):
    return ControlInputEndConds(
        q=ss.q_desc.pack(dict(axis_000=np.array([np.pi]))),
        qd=ss.qd_desc.default,
        q_weights=ss.qd_desc.pack(dict(axis_000=np.array([1]))) * 100,
        qd_weights=ss.qd_desc.pack(dict(axis_000=np.array([1]))) * 100,
        only_end=1,
    )

  s0 = ControlInputState(q=np.array([0,0]), qd=np.array([0,0.0]))
  func_control(ctx, inv_pendulum_idx, s0, conf_end_conds)

def test_simple(ctx):


  if 0:
    ctrl, s0 = (np.array([-5.7290754, 39.4572496, 4.7939896, 2.1076894, 29.1405284, 14.7750776, -22.5005447, -12.2408426, -3.9486710, -14.0009929, -22.0610298, -10.5987316, -23.4305899,
        37.1241162, -37.6302274, 0.6503233, -19.8971855, 29.3156299, -15.9186024, 15.8728587, -13.5048199, -28.5670490, -21.0467544, 36.0038759, 7.3474934, -8.6449755,
        21.3527248, -10.2556932, 31.9737696, -24.1002612, -24.3950631, 18.5171806, 31.5209668, -33.0122529, 18.8458669, -30.9756901, 36.7844384, 1.5173005,
        -28.9216980, -29.2472531, 7.1319183, -21.4976944, -28.3785091, -34.8260019, 29.8575843, 10.1520970, 18.4668352, -34.9138954, -22.4375205, 26.8571922,
        -10.1944054, 12.8341308, -24.6151833, -14.1105370, 37.7476022, 28.8880683, -26.9165326, -27.0069246, 18.9778214, -24.7244952, 33.2801285, 29.2377440,
        -32.6069796, 23.1108777, 27.2060367, 36.2837984, -8.5897874, -33.2475415, 9.7935807,]), np.array([0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.2000000,
        0.4000000, 0.6000000, 0.0000000, 0.0000000, 0.0000000,]))
    sh = prepare_scene(ctx, control_simple_idx3, s0=s0)
    ss = sh.sim.ss

    cparams = ControlParameters(
      ndt_const=3,
      dt=1e-1,
      integ_nsteps=1,
      nt=23,
      use_jit=ctx.use_jit,
      use_rk4=ctx.use_rk4,
      full_jit=ctx.full_jit,
      ctrl_bounds=np.array([-20, 20]) * 2,
    )
    sh.spec.cparams = cparams
    sh.spec.cparams.enable_batch = 0
    solver = RBSolver(spec=sh.spec)
    print('FUU ', solver.simul_helper.eval(ctrl, s0))
    return
          
  def conf_end_conds(ss: RBSystemSpec):
    return ControlInputEndConds(
        q=ss.qd_desc.pack(dict(root_000=np.array([0, 0, 0, 0, 0, 0, 1]))),
        qd=ss.qd_desc.pack(dict(root_000=np.array([0, 0, 0, 0, 0, 0]))),
        q_weights=ss.q_desc.pack(dict(root_000=np.array([0, 0, 0, 1, 1, 1, 0])*30)),
        qd_weights=ss.qd_desc.pack(dict(root_000=np.array([0,0,0,1,1,1])*3)),
        only_end=1,
    )

  cparams = ControlParameters(
    ndt_const=6,
    dt=2e-2,
    integ_nsteps=2,
    nt=20,
    use_jit=ctx.use_jit,
    use_rk4=ctx.use_rk4,
    full_jit=ctx.full_jit,
    ctrl_bounds=np.array([-1, 1]) * 2,
  )
  s0 = ControlInputState(q=np.zeros(10), qd=np.array([0, 0, 0, 1, 2, 3, 0, 0, 0]) * 0.2)
  func_control(ctx, control_simple_idx3, s0, conf_end_conds, cparams=cparams)



def test_ctrl_box(ctx):
  cparams = ControlParameters(
  ndt_const=5,
  dt=0.5e-1,
  integ_nsteps=20,
  nt=10,
  use_jit=True,
  use_rk4=False,
  ctrl_bounds=np.array([-1, 1]) * 2,
  )
  mdata = ModelData(func_id=box_scene_idx)
  ss = mdata.ss

  end_conds= ControlInputEndConds(
        q=ss.q_desc.pack(dict(root_000=np.array([1]))),
        qd=ss.qd_desc.pack(dict(root_000=np.array([0]))),
        q_weights=100,
        qd_weights=100,
        only_end=1,
    )
  spec = ControlSpec(mdata=mdata, cparams=cparams, end_conds=end_conds)
  solver = RBSolver(spec=spec, solver_params=SolverParameters(time_ctrl_count=0))

  s0 = ss.state_packer.default
  res = solver.solve(s0, nlopt_params=dict(maxeval=300), parallel=0)
  print(solver.simul_helper.proc(res.ctrl, s0))
  print(res.tctrl)
  print(res.ctrl)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
