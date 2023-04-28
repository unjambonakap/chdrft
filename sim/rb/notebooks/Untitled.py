#!/usr/bin/env python
# coding: utf-8

# In[1]:


from chdrft.sim.rb.scenes import *
init_jupyter()
from chdrft.sim.rb.blender_helper import g_bh
from chdrft.sim.rb.ctrl import RBSolver


# In[8]:


g_bh.stop_sim()


# # Gyroscope

# In[3]:


g_bh.load_scene(scene_gyro_wheel_idx, qd0=np.array([0,0,0,10]), cparams = ControlParameters(ndt_const=1, dt=0.8e-2, integ_nsteps=1, nt=5, use_jit=1))
i02 = g_bh.i0.copy()
i02.qd = np.array([0,0,0,6])
g_bh.sim.load_state(i02)


# In[4]:


g_bh.run_sim(override_ctrl=g_bh.sim.ss.ctrl_packer.default)


# # T handle

# In[6]:


g_bh.load_scene(scene_T_idx, qd0 = np.array([0, 0, 0, 10, 0, 0.01]), cparams = ControlParameters(ndt_const=1, dt=1e-2, integ_nsteps=100, nt=5, use_jit=1))
g_bh.sim.load_state(g_bh.i0)
print(g_bh.ctrl.mom)


# In[7]:


g_bh.i0.qd = np.array([0, 0, 0, 10, 0, 0.01])
g_bh.ctrl.s0.precision = 50
g_bh.fspec.spec.cparams.use_rk4=True
g_bh.sim.load_state(g_bh.i0)
g_bh.ctrl.s0.fix_mom=1
g_bh.ctrl.mom=None
g_bh.run_sim(override_ctrl=g_bh.sim.ss.ctrl_packer.default)


# # Reaction wheels

# In[17]:


g_bh.load_scene(control_simple_idx3, qd0 = np.array([0, 0, 0, 0,0,0,1,2,3]), cparams = ControlParameters(ndt_const=1, dt=1e-2, integ_nsteps=1, nt=5, use_jit=1))


# In[10]:


g_bh.ctrl.mom.v


# In[19]:


g_bh.stop_sim()


# In[20]:


i02 = g_bh.i0.copy()
i02.qd = np.array([0,0,0,1,0,0,1,2,3])
g_bh.sim.load_state(i02)


# In[21]:


g_bh.run_sim(override_ctrl=np.zeros(3))


# In[ ]:


cparams = ControlParameters(
ndt_const=1,
dt=2e-2,
integ_nsteps=2,
nt=50,
use_jit=True,
use_rk4=False,
ctrl_bounds=np.array([-1, 1]) * 2,
)
g_bh.load_scene(box_scene, cparams = cparams)


# In[ ]:


solver = RBSolver(spec=spec, solver_params=SolverParameters(time_ctrl_count=3))


# In[19]:


g_bh.load_scene(control_simple_idx3, qd0 = np.array([0, 0, 0, 0,0,0,1,2,3]), cparams = ControlParameters(ndt_const=1, dt=1e-2, integ_nsteps=1, nt=5, use_jit=0))


# In[3]:


cparams = ControlParameters(
ndt_const=6,
dt=2e-2,
integ_nsteps=2,
nt=20,
use_jit=False,
use_rk4=False,
ctrl_bounds=np.array([-1, 1]) * 2,
)

s0 = ControlInputState(q=np.zeros(10), qd=np.array([0, 0, 0, 1, 2, 3, 0, 0, 0]) * 0.2)

ctrl =  np.array([[-0.3403384, 0.6889292, 0.9156367,],
       [0.8860169, -0.1321909, 1.3484478,],
       [0.0149377, 1.9148841, 1.6285217,],
       [-0.4826536, 1.8254345, 1.9080556,],
       [1.6354564, 0.0681124, 1.9071922,],
       [1.6369922, 0.7074759, 1.6391515,],
       [-0.2738504, 1.1872587, 1.7149707,],
       [1.1209375, 1.6930679, 1.7114263,],
       [1.3459584, 1.5654239, 0.2162160,],
       [-0.8307597, 0.8605069, 0.8491824,],
       [0.2247303, 1.5798573, 0.3491753,],
       [0.9443550, -1.0971220, 0.8664806,],
       [1.0090562, 1.3101106, 0.0362650,],
       [-1.9766337, -1.6639021, -1.1635714,],
       [1.6672500, -1.8289124, 0.2985678,],
       [-0.1447133, 1.4239786, 0.1877080,],
       [-1.4005772, -0.5529979, -0.6441567,],
       [-1.0254344, -1.6479761, -1.7577772,],
       [-1.7048799, -0.4089228, -1.8950150,],
       [1.3518031, -0.5997885, 0.1824498,]]
                )

tctrl = cparams.to_time_annotated(ctrl)
cparams2 = ControlParameters(
ndt_const=3,
dt=1e-3,
integ_nsteps=1,
nt=20,
use_jit=1,
use_rk4=False,
ctrl_bounds=np.array([-1, 1]) * 2,
)
ctrl2 = cparams2.from_time_annotated(tctrl)
ctrl2.shape

spec = ControlSpec(mdata=g_bh.sh.spec.mdata, cparams=cparams2)
solver = RBSolver(spec=spec)


# In[4]:


states = solver.simul_helper.get_response(s0, ctrl2)


# In[5]:


g_bh.make_animation(states)


# In[ ]:


g_bh.load_scene(, use_jit=0)
def ctrl2model(sim, ctrl):
    return (sim.rootl.rw @ SpatialVector.Force(ctrl).around(-sim.rootl.agg_com)).data
    res = sim.rootl.rl @ sim.rootl.lw.tsf_rot @ SpatialVector.Force(ctrl)
    return res.data

g_bh.sim.sd.fm.nctrl_f = lambda a: a
g_bh.sim.sd.fm.ctrl2model = ctrl2model
g_bh.sh.runner.reset_jit()


# In[62]:


g_bh.load_scene(scene_T_idx, qd0 = np.array([0, 0, 0, 10, 0, 0.1]), cparams = ControlParameters(ndt_const=1, dt=1e-2, integ_nsteps=1, nt=5, use_jit=1), use_jit=1)



# In[4]:


g_bh.sh.runner.reset_jit()


# In[11]:


g_bh.running_sim = 0


# In[5]:


g_bh.timer_freq=400
g_bh.ctrl.s0.speed=3


# In[4]:


g_bh.run_sim(override_ctrl=g_bh.sim.ss.ctrl_packer.default)


# In[7]:


g_bh.setup_timer()


# In[10]:


g_bh.sim.load_state(g_bh.i0)
g_bh.run_sim(override_ctrl=np.array([0,0,1]))
    


# In[16]:


ss = g_bh.sim.ss
cparams = ControlParameters(
  ndt_const=5,
  dt=1e-2,
  integ_nsteps=1,
  nt=30,
  use_jit=1,
  use_rk4=0,
  full_jit=0,
  ctrl_bounds=np.array([-5, 5]) * 1,
    enable_batch=0,
)
end_conds= ControlInputEndConds(
    q=ss.qd_desc.pack(dict(root_000=np.array([0, 0, 0, 0, 0, 0, 1]) * 1)),
    qd=ss.qd_desc.pack(dict(root_000=np.array([0, 0, 0, 0, 0, 0]))),
    q_weights=ss.q_desc.pack(dict(root_000=np.array([0, 0, 0, 1, 1, 1]))),
    qd_weights=ss.qd_desc.pack(dict(root_000=np.zeros(6))),
    only_end=1,
)

spec = ControlSpec(mdata=g_bh.sh.spec.mdata, cparams=cparams, end_conds=end_conds)
from chdrft.sim.rb.ctrl import RBSolver

solver = RBSolver(spec=spec)


# In[10]:


solver.simul_helper.runner.integrate1(1e-2,np.array([1,0,0]), s0, use_jit=0)


# In[ ]:


s0 = ControlInputState(q=ss.q_desc.default, qd=ss.qd_desc.pack(dict(root_000=np.array([0,0,0,1,-3,2])*0.1)))
res = solver.solve(s0, parallel=0, nlopt_params=dict(maxeval=100))
res.score


# In[18]:


states = solver.simul_helper.get_response(s0, res.ans)
g_bh.make_animation(states)

