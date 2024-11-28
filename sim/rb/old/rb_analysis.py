#!/usr/bin/env python

from typing import Tuple, Optional
from dataclasses import dataclass
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.opa_types import *
from pydantic.v1 import BaseModel, Field
from typing import Tuple
import xarray as xr
from typing import Callable, List
from scipy.spatial.transform import Rotation as R
from chdrft.display.base import TriangleActorBase
from chdrft.utils.omath import MatHelper
import itertools
from enum import Enum
import functools
import sympy as sp
import jax.numpy as jnp
import jaxlib
import jax
from chdrft.utils.fmt import Format
from chdrft.utils.path import FileFormatHelper
from typing import Callable
from chdrft.sim.rb.old.rb import *
from chdrft.utils.cache import Cachable

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)




class ControlInputState(cmisc.PatchedModel):
  root_wl: np.ndarray | jnp.ndarray
  root_rotvec_l: np.ndarray | jnp.ndarray
  wh_w: np.ndarray | jnp.ndarray
  wh_ang: np.ndarray | jnp.ndarray


class ControlInputEndConds(cmisc.PatchedModel):
  f_root_rotvec_w: np.ndarray = None
  f_root_rotvec_l: np.ndarray = None
  f_root_wl_rot: np.ndarray = None
  only_end: bool = False


class ControlInput(cmisc.PatchedModel):
  state: ControlInputState
  end_conds: ControlInputEndConds = None


def jax_forctrl(f_u_s, u, s):

  def f(i, us):
    u, s = us
    return u, f_u_s(u[i], s)

  return jax.lax.fori_loop(0, len(u), f, (u, s))[1]


class Simulator(cmisc.PatchedModel):

  class Config:
    arbitrary_types_allowed = True

  state_packer: NumpyPacker
  root: RigidBody = None
  sctx: SceneContext
  name2pos: dict[str, int]
  tc: TorqueComputer = cmisc.pyd_f(TorqueComputer)
  objs: list[RigidBody] = None

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.root = self.sctx.roots[0]
    objs = []
    for name, lst in self.sctx.basename2lst.items():
      if name not in self.name2pos: continue
      a: RigidBody = lst[0]
      pos = self.name2pos[name]
      objs.append(a)
    self.objs = objs

  def load_state0(self, state):
    self.load_state(**self.state_packer.unpack(state))

  def dump_state0(self) -> np.ndarray:
    return self.state_packer.pack(self.dump_state())

  def load_state_input(self, input: ControlInputState):
    self.load_state(**input.dict())

  def load_state(self, root_wl=None, root_rotvec_l=None, wh_w=None, wh_ang=None):

    for i, obj in enumerate(self.objs):
      obj.self_link.link_data.pivot_rotang = wh_ang[i]
      obj.self_link.move_desc.rotspeed = wh_w[i]

    self.tc.set_wl(self.root, A(wl=Transform(data=root_wl)))
    self.root.self_link.rotvec_l = Vec3(root_rotvec_l)

  def update(self, dt: float, u_torques: np.ndarray):
    for i, obj in enumerate(self.objs):
      self.tc.req_link_torque[obj.self_link] = u_torques[i]
    self.tc.setup(self.root.self_link, dt)
    self.tc.process()

  def dump_state(self) -> ControlInputState:
    o_wh_ang = make_array([x.self_link.link_data.pivot_rotang for x in self.objs])
    o_wh_w = make_array([x.self_link.move_desc.rotspeed for x in self.objs])
    return ControlInputState(
        root_wl=self.root.self_link.wl.data,
        root_rotvec_l=self.root.self_link.rotvec_l.vdata,
        wh_w=o_wh_w,
        wh_ang=o_wh_ang
    )

  def do_step(self, dt, controls, state):
    self.load_state0(state)
    for wh_torque in controls:
      self.update(dt, wh_torque)
    return self.dump_state0()


class SimulatorCreator(cmisc.PatchedModel):
  name2pos: dict
  func_id: str

  @cmisc.cached_property
  def func(self):
    from chdrft.sim.rb.old.scenes import g_registar
    return g_registar.get(self.func_id)


class ModelData(cmisc.PatchedModel):
  creator: SimulatorCreator
  nctrl: int

  @cmisc.cached_property
  def nobj(self) -> int:
    return len(self.create_sctx().obj2name) - 1

  def create_sctx(self) -> SceneContext:
    wh_w_params = np.zeros(self.nctrl)
    wh_ang_params = np.zeros(self.nctrl)

    d = self.creator.func(self.nctrl, wh_w_params, wh_ang_params)
    return d.sctx

  def create_simulator(self) -> Simulator:

    sx = Simulator(
        sctx=self.create_sctx(), name2pos=self.creator.name2pos, state_packer=self.state_packer
    )
    return sx

  @cmisc.cached_property
  def state_packer(self) -> NumpyPacker:
    packer = NumpyPacker()
    packer.add_dict(root_wl=(4, 4), root_rotvec_l=3, wh_w=self.nobj, wh_ang=self.nobj)
    return packer


class ControlParameters(cmisc.PatchedModel):
  ndt_const: int
  dt: float
  integ_nsteps: int
  nt: int

  @property
  def actual_dt(self) -> float:
    return self.dt / self.integ_nsteps

  @property
  def dt_ctrl(self) -> float:
    return self.dt * self.ndt_const

  @property
  def dt_tot(self) -> float:
    return self.dt * self.ndt_const * self.nt

  @dt_tot.setter
  def dt_tot(self, v):
    self.dt = v / (self.ndt_const * self.nt)

  @property
  def tot_steps(self) -> int:
    return self.ndt_const * self.integ_nsteps * self.nt


class ControlSpec(cmisc.PatchedModel):
  cparams: ControlParameters = None
  mdata: ModelData
  end_conds: ControlInputEndConds = None

  @cmisc.cached_property
  def ctrl_packer(self) -> NumpyPacker:
    packer = NumpyPacker()
    packer.add_dict(ctrl=(self.cparams.nt, self.mdata.nctrl))
    return packer

  @cmisc.cached_property
  def state_packer(self) -> NumpyPacker:
    return self.mdata.state_packer

  def make_packed(self, state: np.ndarray | ControlInputState) -> np.ndarray:
    if isinstance(state, np.ndarray): return state
    return self.state_packer.pack(state)


class ControlFullSpec(cmisc.PatchedModel):
  consts: ControlInput
  spec: ControlSpec


class RB_PbAnalysis(cmisc.PatchedModel):
  spec: ControlSpec

  @property
  def mdata(self) -> ModelData:
    return self.spec.mdata

  def setup(self, setup_jit=1):
    if setup_jit: self.jit_ctrl
    #self.jac_func

  @property
  def args1(self):
    x = A(
        u=np.zeros((1, self.mdata.nctrl)),
        state=A(
            root_wl=np.identity(4),
            root_rotvec_l=np.identity(3)[1],
            wh_w=np.zeros(self.mdata.nctrl),
            wh_ang=np.zeros(self.mdata.nctrl),
        ),
    )
    x.state = self.state.state_packer.pack(x.state)
    return x

  @property
  def args0(self):
    x = A(
        u=np.zeros((1, self.mdata.nctrl)),
        state=A(
            root_wl=np.identity(4),
            root_rotvec_l=np.zeros(3),
            wh_w=np.zeros(self.mdata.nctrl),
            wh_ang=np.zeros(self.mdata.nctrl),
        ),
    )
    x.state = self.spec.state_packer.pack(x.state)
    return x

  def advance(self, sim: Simulator, dt: float, u: np.ndarray, **kwargs):
    state = sim.dump_state0()
    #print()
    #print()
    #print('INPUT ', dt, sim.dump_state(), u, kwargs)
    res = np.array(self.integrate(dt, np.array((u,)), state, 1, **kwargs))
    sim.load_state0(res)
    #print('FUU ', sim.dump_state())

  @cmisc.cached_property
  def jac_func(self):

    def build_jit(nsteps, *args):
      res = jax.jacrev(lambda *b: px.integrate(*b, nsteps), argnums=(1,))
      return res(*args)

    return jax.jit(build_jit, static_argnums=0)

  def do_control(self, dt, controls, state):
    sx: Simulator = self.mdata.create_simulator()
    return sx.do_step(dt, controls, state)

  @cmisc.cached_property
  def jit_ctrl(self):
    jit_ctrl = jax.jit(lambda *args: self.do_control(*args))
    jit_ctrl(0., self.args0.u, self.spec.state_packer.zeros_packed)  # trigger jit
    return jit_ctrl

  def integrate1(self, dt: float, u: np.ndarray, state: np.ndarray, rk4=False, use_jit=1):
    assert not rk4, "Dont know how to handle this with angular momentum conservation normalization"
    # Would need sympletic integration scheme?
    fx = self.jit_ctrl if use_jit else self.do_control
    return make_array(fx(dt, u, state))

  def integrate(self, dt: float, u: np.ndarray, state: np.ndarray, nsteps, **kwargs):
    for _ in range(nsteps):
      state = self.integrate1(dt, u, state, **kwargs)
    return state

  def eval_obj(self, f_conds: ControlInputEndConds, state: np.ndarray, t=None, end=0) -> np.ndarray:
    if not f_conds: return 0
    if f_conds.only_end and not end: return 0
    state = self.spec.state_packer.unpack(state)

    cur_f_root_rotvec_w = state.root_wl[:3, :3] @ state.root_rotvec_l
    #ans = (last_t, np.linalg.norm(cur_f_root_rotvec_w - f_root_rotvec_w))
    obj = 0

    if f_conds.f_root_rotvec_w is not None:
      obj += g_oph.norm(cur_f_root_rotvec_w - f_conds.f_root_rotvec_w)
    if f_conds.f_root_rotvec_l is not None:
      obj += g_oph.norm(state.root_rotvec_l - f_conds.f_root_rotvec_l)
    if f_conds.f_root_wl_rot is not None:
      obj += g_oph.norm((state.root_wl[:3, :3] - f_conds.f_root_wl_rot).reshape((-1,)))
      #obj += #np.linalg.norm(state.root_wl - f_root_wl)
    return obj


class SimulHelper(cmisc.PatchedModel):
  spec: ControlSpec
  tot_batch_size: int = 16 * 20

  @property
  def end_conds(self) -> ControlInputEndConds:
    return self.spec.end_conds

  def setup(self):
    self.jit_fx
    self._dispatcher_jit

  @cmisc.cached_property
  def sim(self) -> Simulator:
    sim = self.spec.mdata.create_simulator()
    return sim

  def integrate1(self, dt: float, u: np.ndarray):
    ostate = np.array(self.runner.integrate1(dt, u, self.sim.dump_state0()))
    self.sim.load_state0(ostate)

  @cmisc.cached_property
  def runner(self) -> RB_PbAnalysis:
    res = RB_PbAnalysis(spec=self.spec)
    res.setup()
    return res

  @property
  def ctrl_params(self) -> ControlParameters:
    return self.spec.cparams

  @Cachable.cachedf()
  @staticmethod
  def Get(spec: ControlSpec) -> "SimulHelper":
    return SimulHelper(spec=spec)

  def get_end_state(
      self, ctrl: np.ndarray, i_state: ControlInputState, obj_cb=lambda state, **kwargs: 0
  ) -> tuple:
    ctrl = self.spec.ctrl_packer.unpack(ctrl).ctrl
    state = self.spec.make_packed(i_state)
    ntx = len(ctrl)
    obj = 0
    for i in range(ntx):
      state = self.runner.integrate(
          self.ctrl_params.actual_dt,
          (ctrl[i],),
          state,
          self.ctrl_params.ndt_const * self.ctrl_params.integ_nsteps,
      )
      obj += obj_cb(state, end=0, t=(i + 1) / ntx)
    return state, obj

  def proc(self, ctrl, i_state: ControlInputState):
    state, obj = self.get_end_state(
        ctrl, i_state,
        lambda *args, **kwargs: self.runner.eval_obj(self.end_conds, *args, **kwargs)
    )
    obj += self.runner.eval_obj(self.end_conds, state, end=1)
    return (obj,)

  @cmisc.cached_property
  def jit_fx(self):
    func = jax.jit(self.fx)
    func(self.spec.ctrl_packer.zeros_packed, self.spec.state_packer.zeros_packed)
    return func

  def fx(self, ctrl, state):
    ctrl = self.spec.ctrl_packer.unpack(ctrl).ctrl
    ntx = len(ctrl)
    obj = 0

    def fus(u, _s):
      s, obj, sid = _s
      s = jax.lax.fori_loop(
          0, self.ctrl_params.ndt_const * self.ctrl_params.integ_nsteps,
          lambda i, state: self.runner.jit_ctrl(self.ctrl_params.actual_dt, (u,), state), s
      )
      obj += self.runner.eval_obj(self.end_conds, s, t=(sid + 1) / ntx)
      return s, obj, sid + 1

    state, obj, _ = jax_forctrl(fus, ctrl, (state, 0, 0))
    obj += self.runner.eval_obj(self.end_conds, state, end=1)

    return obj

  def _dispatcher(self, u_lst, s_lst):
    nv = len(s_lst)
    res = jnp.zeros(nv)

    def f_usr(i, usr):
      u, s, r = usr
      ri = self.fx(u[i, :], s[i, :])
      r = r.at[i].set(ri)
      return (u, s, r)

    u, s, r = jax.lax.fori_loop(0, nv, f_usr, (u_lst, s_lst, res))
    return r

  @property
  def ndevices(self) -> int:
    return jax.device_count()

  @property
  def batch_size(self) -> int:
    return self.tot_batch_size // self.ndevices

  @property
  def ntake(self) -> int:
    return self.ndevices * self.batch_size

  @Cachable.cachedf(fileless=True, method=True)
  def _dispatcher_jit_pmap(self, batch_size):
    func = jax.jit(self._dispatcher)
    u0 = np.random.rand(*(batch_size, self.spec.ctrl_packer.pos))
    s0_0 = self.runner.args0.state
    s0 = np.broadcast_to(s0_0, (batch_size,) + s0_0.shape)
    return jax.pmap(func)

  @Cachable.cachedf(fileless=True, method=True)
  def make_dispatch_jit(self, batch_size):
    func = self._dispatcher_jit_pmap(batch_size)
    ndevices = jax.device_count()
    u0 = np.random.rand(*(self.ndevices, batch_size, self.spec.ctrl_packer.pos))
    s0_0 = self.runner.args0.state
    s0 = np.broadcast_to(s0_0, (self.ndevices, batch_size) + s0_0.shape)

    _ = np.array(func(u0, s0))
    return func

  @cmisc.cached_property
  def _dispatcher_jit(self):
    return self.make_dispatch_jit(self.batch_size)

  def dispatcher(self, u_lst, s_lst):
    res = []
    for i in range(0, len(u_lst), self.ntake):
      u = u_lst[i:i + self.ntake]
      s = s_lst[i:i + self.ntake]

      ndevices = (len(u) + self.batch_size - 1) // self.batch_size
      target = ndevices * self.batch_size
      add = target - len(u)
      if add:
        u = np.concatenate((u, np.zeros((add, self.spec.ctrl_packer.pos))), axis=0)
        s = np.concatenate((s, np.zeros((add, self.spec.state_packer.pos))), axis=0)

      u = u.reshape((ndevices, self.batch_size, -1))
      s = s.reshape((ndevices, self.batch_size, -1))
      cur = np.array(self._dispatcher_jit_pmap(self.batch_size)(u, s)).reshape((-1,))
      if add:
        cur = cur[:-add]
      res.extend(cur)
    return np.array(res)


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
