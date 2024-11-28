#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.misc import Field
import pandas as pd
import polars as pl
import re
import os.path
import os
import sys
import chdrft.utils.Z as Z
from jaxtyping import Float, Array
import jax
from chdrft.sim.rb.base import g_oph, NumpyPacker

# exercise: tracking app on highly variably curved manifold

global flags, cache
flags = None
cache = None

from jax import tree_util, grad


def grad_pytree(f, data_like):
  shape_out = f(data_like)
  flat, treedef = tree_util.tree_flatten_with_path(data_like)

  def define_func(kout):

    def f_wrap(*u):
      unf = tree_util.tree_unflatten(treedef, u)
      res = f(unf)
      return res[kout]

    res = jax.jacfwd(f_wrap, argnums=list(range(len(flat))))
    return res

  funcs = [define_func(kout) for kout in shape_out.keys()]

  def eval(data):
    data_flat, treedef = tree_util.tree_flatten_with_path(data)
    flat_paths, flat_args = zip(*data_flat)
    res = {}
    for kout, func in zip(shape_out.keys(), funcs):
      cur = func(*flat_args)
      for flat_path, rv in zip(flat_paths, cur):
        flat_path = tuple([x.key for x in flat_path])
        res[(kout, flat_path)] = rv
    return res

  return eval


import dataclasses


@dataclasses.dataclass
class GaussDistribution:
  u: Float[Array, "n"]
  cov: Float[Array, "n n"]

  def marginal(self, ids: Float[Array, "m"]) -> GaussDistribution:
    return GaussDistribution(u=self.u[ids], cov=self.cov[ids, ids])

  @property
  def n(self):
    return len(self.u)

  def conditional(self, ida: slice, idb: slice, valb: Float[Array, "m"]) -> GaussDistribution:
    ua, ub = self.u[ida], self.u[idb]
    caa = self.cov[ida, ida]
    cab = self.cov[ida, idb]
    cbb = self.cov[idb, idb]
    adda = cab @ np.linalg.inv(cbb) @ (valb - ub)
    print('>>> ', ua, adda, valb, ub)

    u = ua + adda
    c = caa - cab @ np.linalg.inv(cbb) @ cab.T
    return GaussDistribution(u=u, cov=c)

  def map_mat(
      self, mat: Float[Array, "n n"], noise: Float[Array, "n n"] = None
  ) -> GaussDistribution:
    res = GaussDistribution(u=mat @ self.u, cov=mat.T @ self.cov @ mat)
    if noise is not None:
      res = res.add_noise(noise)
    return res

  def add_noise(self, noise: Float[Array, "n n"]) -> GaussDistribution:
    return GaussDistribution(u=self.u, cov=self.cov + noise)

  def joint(self, mat: Float[Array, "m n"],
            noise_mat: Float[Array, "m m"]) -> tuple[GaussDistribution, tuple[slice, slice]]:
    m = len(mat)
    cab = self.cov @ mat.T
    cbb = mat @ self.cov @ mat.T + noise_mat

    g = GaussDistribution(
        u=np.block([self.u, mat @ self.u]), cov=np.block([[self.cov, cab], [cab.T, cbb]])
    )
    return g, (slice(0, self.n), slice(self.n, self.n + m))


class KalmanSystem(cmisc.PatchedModel):

  def predict(self, a: GaussDistribution, u: np.ndarray, dt: float) -> GaussDistribution:
    pass

  def refit_obs(self, a: GaussDistribution, obs: np.ndarray) -> GaussDistribution:
    pass

  def run(
      self, a: GaussDistribution, u: np.ndarray, obs: np.ndarray, dt: float
  ) -> GaussDistribution:
    tmp = self.predict(a, u, dt)
    res = self.refit_obs(tmp, obs)
    return res


class EKF_CB(KalmanSystem):
  get_pred: object = None
  get_dpred: object = None
  get_noise_state: object = None
  get_obs: object = None
  get_noise_obs: object = None

  def predict(self, state: GaussDistribution, u: np.ndarray, dt: float) -> GaussDistribution:
    upred = self.get_pred(state.u)
    j = self.get_dpred(state.u)
    return GaussDistribution(u=upred, cov=j @ state.cov @ j.T)

  def refit_obs(self, state: GaussDistribution, obs: np.ndarray) -> GaussDistribution:
    joint, (istate, iobs) = state.joint(self.get_obs(state.u), self.get_noise_obs(state.u))
    return joint.conditional(istate, iobs, obs)


class LinearKalmanSystem(KalmanSystem):
  state2obs: np.ndarray
  noise_state: np.ndarray
  noise_obs: np.ndarray
  pred_state: np.ndarray

  def predict(self, state: GaussDistribution, u: np.ndarray, dt: float) -> GaussDistribution:
    return state.map_mat(self.pred_state, noise_state)

  def refit_obs(self, state: GaussDistribution, obs: np.ndarray) -> GaussDistribution:
    joint, (istate, iobs) = state.joint(self.state2obs, self.noise_obs)
    return joint.conditional(iobs, istate, state.u)


class SymVar(cmisc.PatchedModel):
  name: str
  pos: int
  par: SymVar = None
  child: SymVar = None
  like: np.ndarray
  manifold: object = None
  expr: object = None


def wrapped_jax(f, packer_in, packer_out):

  def fwrap(in_flat):
    return packer_out.pack(f(packer_in.unpack(in_flat).to_dict()))

  return fwrap


def dict_call(tsf):

  def eval(u):
    return {k: v(u) for k, v in tsf.items()}

  return eval


class SymObs(cmisc.PatchedModel):

  name: str
  obs_model: object
  like: np.ndarray
  cov: object


class SysState(cmisc.PatchedModel):
  state: dict[str, np.ndarray] = Field(default_factory=dict)
  pass


class SymSys(cmisc.PatchedModel):
  lst_state: list[SymVar] = Field(default_factory=list)
  ordering: list[SymVar] = Field(default_factory=list)
  lst_obs: list[SymObs] = Field(default_factory=list)
  state_packer: NumpyPacker = None
  obs_packer: NumpyPacker = None

  def create_obs(self, name=None, **kwargs):
    if name is None:
      name = f'var_{len(self.lst_obs):03d}'
    v = SymObs(name=name, **kwargs)
    self.lst_obs.append(v)
    return v

  def create_state(self, name=None, dt_n=1, expr=None, **kwargs):
    if name is None:
      name = f'var_{len(self.lst):03d}'

    lst = [SymVar(name=f'{name}_dt_{i}', pos=i, **kwargs) for i in range(dt_n)]
    for i in range(dt_n - 1):
      lst[i].child = lst[i + 1]
      lst[i + 1].par = lst[i]
    lst[-1].expr = expr

    self.lst_state.extend(lst)

    return lst

  def finish(self):
    state_packer = NumpyPacker()
    for x in self.lst_state:
      state_packer.add(x.name, x.like.shape)

    obs_packer = NumpyPacker()
    for x in self.lst_obs:
      obs_packer.add(x.name, x.like.shape)
    self.state_packer = state_packer
    self.obs_packer = obs_packer

  def compute_tsf_mat(self, ss: SysState, dt: float):

    @jax.grad
    def compute_grad(u):
      res = dict()
      for x in self.ordering:
        pass

  def make_linearize_tsf(self, tsf, packer_in: NumpyPacker, packer_out: NumpyPacker):

    ex = grad_pytree(dict_call(tsf), packer_in.unpack(packer_in.default).to_dict())

    def as_mat_func(state_flat):
      state = packer_in.unpack(state_flat).to_dict()
      res = ex(state)
      blocks = []
      for pout in packer_out.tb:
        blocks.append([])
        for pin in packer_in.tb:
          u = res[(pout.name, (pin.name,))]
          blocks[-1].append(u.reshape((pout.size, pin.size)))
      return g_oph.get_npx(state_flat).block(blocks)

    return as_mat_func

  def make_state_zero(self) -> SysState:
    return SysState(state=self.state_packer.unpack(self.state_packer.default).to_dict())

  def make_kalman(self, tsf, tsf_obs, noise_state, noise_obs):
    dx_tsf = self.make_linearize_tsf(tsf, self.state_packer, self.state_packer)
    dx_tsf_obs = self.make_linearize_tsf(tsf_obs, self.state_packer, self.obs_packer)
    res = EKF_CB(
        get_pred=jax.jit(wrapped_jax(dict_call(tsf), self.state_packer, self.state_packer)),
        get_dpred=dx_tsf,
        get_noise_state=lambda _: noise_state,
        get_obs=dx_tsf_obs,
        get_noise_obs=lambda _: noise_obs
    )

    return res


class WorldState(cmisc.PatchedModel):
  t: float = 0

  def advance(self, dt):
    self.dt += t


def test1(ctx):
  s = SymSys()
  xy, = s.create_state(name='pos', like=np.zeros(2))
  v, = s.create_state(name='v', like=np.zeros(1))
  vdir, = s.create_state(name='vdir', like=np.zeros(1))

  obs_xy = s.create_obs(
      name='pos_obs',
      like=np.zeros(2),
      obs_model=lambda sstate: xy,
      cov=lambda sstate: np.array([0.1, 0.1])
  )
  s.finish()

  dt = 0.1
  tsf_mat = {
      v.name: (lambda u: u[v.name]),
      vdir.name: (lambda u: u[vdir.name]),
      xy.name:
          (
              lambda u: u[xy.name] + g_oph.
              array([g_oph.cos(u[vdir.name]), g_oph.sin(u[vdir.name])]).ravel() * u[v.name] * dt
          ),
  }
  obs_mat = {
      obs_xy.name: (lambda u: u[xy.name]),
  }

  s_a = 1e-2
  s_dir = 1e-3
  q1 = np.multiply(
      np.array([
          [1 / 2 * dt**2, 0, 0],
          [0, 1 / 2 * dt**2, 0],
          [0, dt, 0],
          [0, 0, dt],
      ]),
      np.array([s_a, s_a, s_dir]).reshape((1, -1))
  )
  s_o = 1e-1
  q2 = np.array([
      [s_o, 0],
      [0, s_o],
  ])

  res = s.make_kalman(tsf_mat, obs_mat, q1 @ q1.T, q2 @ q2.T)
  u0 = np.array([0, 0, 0.1, np.pi / 5]).astype(np.float32)
  f1 = s.make_linearize_tsf(tsf_mat, s.state_packer, s.state_packer)
  f1(u0)
  s0 = GaussDistribution(u=u0, cov=np.identity(4))
  for i in range(1, 3000):
    s0 = res.run(s0, None, np.array([i, i]) * 0.1 * dt, dt)
    print(s0.u)


#test1(None)
#%%

from chdrft.sim.base import InterpolatedDF
import scipy.integrate
import scipy.interpolate


class TreeContainer(cmisc.PatchedModel):
  obj2node: dict[object, Node] = cmisc.Field(default_factory=dict)
  node2obj: dict[Node, object] = cmisc.Field(default_factory=dict)
  root: Node = None

  def get(self, node, key):
    if key.startswith('/'):
      node = node.root
      key = key[1:]

    for e in key.split('/'):
      node = node.immediate_child(e)
    return node

  def add(self, obj, obj_par):
    n = Node(name=obj.name, tc=self)
    par = self.obj2node.get(obj_par)
    if par is not None:
      n.parent = par
      par.children[n.name] = n 
    else:
      self.root = n

    self.node2obj[n] = obj
    self.obj2node[obj] = n
    return n


class Node(cmisc.PatchedModel):
  children: dict[str, Node] = cmisc.Field(default_factory=dict)
  name: str
  parent: Node = None
  tc: TreeContainer

  @property
  def root(self):
    return (self if self.parent is None else self.parent.root)

  def immediate_child(self, v):
    if v == '..':
      return self.parent
    return self.children.get(v)

  def get(self, x):
    return self.tc.get(self, x).data



class IntegrationParams(cmisc.PatchedModel):
  phys_integration_period: float = 1e-3


class Integrator(cmisc.PatchedModel):
  params: IntegrationParams = cmisc.Field(default_factory=IntegrationParams)

  def integ1(self, f, x0, ti, dt):
    u = scipy.integrate.RK45(
        f,
        t0=ti,
        y0=x0,
        t_bound=ti + dt,
        max_step=self.params.phys_integration_period,
    )
    states = []
    while u.status == 'running':
      states.append(np.concatenate([(u.t,), u.y]))
      u.step()
    states.append(np.concatenate([(u.t,), u.y]))
    return A(res=states[-1][1:], states=states)

class SysConfig(cmisc.PatchedModel):
  debug:bool = False

class System(cmisc.PatchedModel):
  integrator: Integrator = cmisc.Field(default_factory=Integrator)
  tc: TreeContainer = cmisc.Field(default_factory=TreeContainer)
  config: SysConfig = cmisc.Field(default_factory=SysConfig)

  root: Entity = None

  def __init__(self):
    super().__init__()
    self.root = self.add_entity(Entity(name=''))

  def add_entity(self, e: Entity, par: Entity = None):
    assert e.sys is None
    e.sys = self
    if par is None: par = self.root
    if par is not None:
      e.full_name = f'{par.full_name}/{e.name}'
    else:
      e.full_name = ''
    self.tc.add(e, par)
    return e


class DataContainer(cmisc.PatchedModel):
  mp: dict[object, object] = cmisc.Field(default_factory=dict)

  def get(self, x):
    return self.mp[x]

  def set(self, x, v):
    self.mp[x] = v



class UpdateSpec(cmisc.PatchedModel):
  rate: int = -1
  prio: int = 0


class Entity(cmisc.PatchedModel):
  full_name: str = None
  name: str = None
  sys: System = None
  model: object = None
  update_spec: UpdateSpec = None 
  dc: DataContainer = cmisc.Field(default_factory=DataContainer)

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.postinit()

  def postinit(self):
    pass


class UpdateRequest(cmisc.PatchedModel):
  ti: float
  dt: float


class EqEntity(Entity):
  packer: NumpyPacker = cmisc.Field(default_factory=NumpyPacker)
  states: list = cmisc.Field(default_factory=list)

  
  def to_df(self, prefix=''):
    def packer2columns(packer):
      res = []
      for x in packer.tb:
        if x.size == 1:
          res.append(x.name)
        else:
          for y in range(x.size):
            res.append(f'{x.name}_{y:03d}')
      return res
    cols = ['t'] + packer2columns(self.packer)
    cols = [f'{prefix}{x}' for x in cols]
    return pd.DataFrame.from_records(np.array(c.states), columns=cols)

  def set_state(self, dx: dict):
    for k, v in dx.items():
      self.dc.set(f'state/{k}', v)

  def update(self, req: UpdateRequest):
    f = self.get_integ_func()
    x0 = self.packer.pack(self.state)
    res =  self.sys.integrator.integ1(f, x0, req.ti, req.dt)
    if self.sys.config.debug:
      self.states.extend(res.states)
    self.set_state(self.packer.unpack(res.res))

  def get_integ_func(self):

    init_data = self.init()
    self.packer = NumpyPacker()
    self.packer.add_dict_like(**init_data['state'])


    def integ_func(t, state):
      tmp = dict(state=state, dt=np.zeros(1))
      res = grad_pytree(lambda x: dict(state=self.packer.pack(self.get_f(t, self.packer.unpack(x['state']), init_data['params'], x['dt']))), tmp)(tmp)
      return res[('state', ('dt',))].ravel()
    return jax.jit(integ_func)
    return integ_func


class ConstantVelocity2D(EqEntity):

  @property
  def state(self):
    return dict(xy=self.dc.get('state/xy'))

  def init(self):
    d_xy = self.dc.get('param/d_xy')
    return dict(state=self.state, params=dict(d_xy=d_xy))

  def get_f(self, t, state, params, dt):
    return dict(xy=state['xy'] + params['d_xy'] * dt)


class NoisyAcc2D(EqEntity):

  @property
  def state(self):
    return dict(xy=self.dc.get('state/xy'), d_xy = self.dc.get('state/d_xy'))

  def init(self):
    xy = self.dc.get('state/xy')
    dd_xy = self.dc.get('param/dd_xy')
    norm_noise = self.dc.get('param/norm_noise')

    return dict(state=self.state, params=dict(dd_xy=dd_xy, norm_noise=norm_noise))

  def get_f(self, t, state, params, dt):
    acc = params['dd_xy']
    #acc += np.random.normal(params.norm_noise.u, params.norm_noise.cv)
    res =  dict(xy=state['xy'] + state['d_xy'] * dt + acc * (dt**2), d_xy=state['d_xy'] + acc * dt)
    return res


class ConstantTurn2D(EqEntity):

  @property
  def state(self):
    return dict(xy=self.dc.get('state/xy'), theta=self.dc.get('state/theta'))

  def init(self):
    cturn = self.dc.get('param/cturn')
    self.dc.set('state/theta', np.zeros(1))
    d_xy = self.dc.get('state/d_xy')
    cturn = self.dc.get('param/cturn')
    vnorm = np.linalg.norm(d_xy)

    theta = np.array([np.atan2(d_xy[1], d_xy[0])])
    return dict(state=self.state, params=dict(cturn=cturn, v=vnorm))

  def get_f(self, t, state, params, dt):
    return dict(
        xy=state['xy'] + g_oph.array([g_oph.cos(state['theta']), g_oph.sin(state['theta'])]).ravel() * params['v'] * dt,
        theta=state['theta'] + params['cturn'] * dt
    )

    # v


s = System()
s.config.debug = True
p0 = s.add_entity(Entity(name='p0'))
sel = 2

if sel == 0:
  c = s.add_entity(ConstantVelocity2D(name='controller'), par=p0)

  c.dc.set('state/xy', np.zeros(2))
  c.dc.set('param/d_xy', np.ones(2))
elif sel == 1:
  c = s.add_entity(NoisyAcc2D(name='controller'), par=p0)

  c.dc.set('state/xy', np.zeros(2))
  c.dc.set('state/d_xy', np.zeros(2))
  c.dc.set('param/dd_xy', np.ones(2))
  c.dc.set('param/norm_noise', dict(u=np.zeros(2), std=np.ones((2,2))))
else:
  c = s.add_entity(ConstantTurn2D(name='controller'), par=p0)

  c.dc.set('state/xy', np.zeros(2))
  c.dc.set('state/d_xy', np.ones(2))
  c.dc.set('param/cturn', np.ones(1))

c.update(UpdateRequest(ti=0, dt=4))
c.to_df(prefix='r2').to_parquet('/tmp/c.parquet')




#%%

class Simulator(cmisc.PatchedModel):
  """
  Multi agent simulation -> multiple levels:
    - physics level: integration
    - 
  """
  sys_lcm_period: float
  phys_integration_period: float = 1e-3

  def simulate_phys(self, f, x0, ti):
    u = scipy.integrate.RK45(
        f,
        t0=ti,
        y0=x0,
        t_bound=ti + self.sys_lcm_period,
        max_step=self.phys_integration_period,
    )
    states = []
    while u.status == 'running':
      states.append(np.concatenate([(u.t,), u.y]))
      u.step()

    if 0:
      states = np.array(states)
      res = scipy.interpolate.interp1d(
          states[:, 0],
          states[:, 1:],
          axis=0,
          fill_value=(states[0, 1:], states[-1, 1:]),
          bounds_error=False
      )(tspace)
      res = np.concatenate((tspace[:, np.newaxis], res), axis=1)
      return states

    return states[-1][1:]

  def simulate(self, x0, fsys):
    x = x0
    t = 0
    while True:
      req = fsys(x)
      if req.action is SimResult.FINISH:
        break

      xf = self.simulate_phys(req.f, req.x0, t)
    return res


def test2(ctx):

  def t1():
    a = 123
    raise Exception('aa')

  try:
    t1()
  except Exception as e:
    print(e)
    print(e.__context__)
    import pdb
    pdb.post_mortem()
    pass
  #%%

  ss = SysState(
      state={
          xy.name: np.array([1, 1], dtype=np.float32),
          v.name: np.array([1], dtype=np.float32),
          vdir.name: np.array([0.5], dtype=np.float32),
      }
  )

  f = s.make_linearize_tsf(ss, tsf_mat)
  print(f.as_mat_func(f.packer.pack(ss.state)))

  ff = f.as_mat_func
  ff = jax.jit(f.as_mat_func)

  print(ff(np.array([0, 0, 10, 1], dtype=np.float32)))
  print(ff(np.array([0, 0, 20, 8], dtype=np.float32)))

  cmisc.return_to_ipython()

  #%%

  p0 = np.random.uniform(low=-10, high=10, size=2)
  v = np.random.uniform(0.1, 1)
  vdir = np.random.uniform(0, np.pi * 2)
  fx = lambda t, x: np.array([np.cos(x[3]) * x[2], np.sin(x[3]) * x[2], 0, 0])
  data = do_simulation(fx, np.arange(0, 10, 0.1), np.array([p0[0], p0[1], v, vdir]))


#%%

test1(None)
#%%


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
