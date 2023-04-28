#!/usr/bin/env python
# coding: utf-8

# In[37]:


from chdrft.sim.rb.scenes import *
init_jupyter()
from chdrft.sim.rb.blender_helper import g_bh
from chdrft.sim.rb.blender_helper import *
from chdrft.sim.rb.ctrl import RBSolver
from chdrft.sim.rb.base import make_array
from chdrft.sim.rb.rb_player import ButtonKind
from chdrft.inputs.controller import SceneControllerInput

from chdrft.dsp.utils import linearize_clamp
def linearize_clamp_abs(v, low, high, ylow, yhigh):
    
    sgn = cmisc.sign(v)
    if abs(v) < low: return ylow * sgn
    return linearize_clamp(abs(v), low, high, ylow, yhigh) * sgn


# In[2]:


def test1():
  sctx = SceneContext()
  tx = RBTree(sctx=sctx, split_rigid=True)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Box(1, 1, 1, 1),
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.FREE,)),
      )
  )

  res = tx.create(root)

  def ctrl2model(sim: Simulator, ctrl):
    ctrl = make_array([ctrl[0], 0, ctrl[1], 0, 0, 0])
    res =  (sim.rootl.rw @ SpatialVector.Force(ctrl).around(-sim.rootl.agg_com)).data
    if 1:
        print()
        print(ctrl, res, sim.rootl.wl @ -sim.rootl.agg_com, sim.rootl.agg_com)
    return res

  def model2ctrl(sim: Simulator, model):
    pass


  fm = ForceModel(
      nctrl_f=lambda n_: 2,
      model2ctrl=model2ctrl,
      ctrl2model=ctrl2model,
      bias_force_f=bias_force_g,
  )

  fm.ctrl2model = ctrl2model
  return SceneData(sctx=sctx, fm=fm)


test1_idx = SceneRegistar.Registar.register('test1', test1,force=1)
from chdrft.utils.cache import Cachable
Cachable.ResetCache(fileless=True)


# In[46]:


def test2():
  sctx = SceneContext()
  tx = RBTree(sctx=sctx, split_rigid=True)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Box(1, 1, 1, 1),
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.FREE,)),
      )
  )

  cone_h = 1
  cone = tx.add(
      RBDescEntry(
          data=RBData(base_name='cone'),
          spec=SolidSpec.Cone(3, 0.5, cone_h),
          link_data=LinkData(
              spec=LinkSpec(
                  type=RigidBodyLinkType.RIGID,
                  wr=Transform.From(pos=[0, 0, -0.5 - cone_h * 3 / 4]),
              )
          ),
          parent=root,
      )
  )

  res = tx.create(root)

  def ctrl2model(sim: Simulator, ctrl):
    ctrl = make_array([ctrl[0], 0, ctrl[1], 0, 0, 0])
    cone = sim.sctx.single('cone').self_link
    res =  (sim.rootl.rl @ cone.wl @ SpatialVector.Force(ctrl)).data
    return sim.ss.f_desc.pack({sim.root.name: res})

  def model2ctrl(sim: Simulator, model):
    pass


  fm = ForceModel(
      nctrl_f=lambda n_: 2,
      model2ctrl=model2ctrl,
      ctrl2model=ctrl2model,
      bias_force_f=bias_force_0,
  )

  fm.ctrl2model = ctrl2model
  return SceneData(sctx=sctx, fm=fm)


test2_idx = SceneRegistar.Registar.register('test2', test2,force=1)
from chdrft.utils.cache import Cachable
Cachable.ResetCache(fileless=True)


# In[47]:


import shapely.geometry
import shapely.ops
max_ang = np.pi/8
max_norm = 40
p0 = np.array([-np.sin(max_ang), np.cos(max_ang)]) * 2
allowed_shape = shapely.geometry.Polygon([(0,0), p0, p0 * [-1,1]])

def getp(v: np.ndarray) -> np.ndarray:
    return v
    nx = np.linalg.norm(v)
    if nx < 1e-5: return np.zeros(2)
    vn = v / nx
    p = shapely.geometry.Point(vn)
    p0 =shapely.ops.nearest_points(allowed_shape, p)[0]
    return np.array([p0.x, p0.y]) * nx * np.linalg.norm(p0)

    
def controller2ctrl(controller: SceneControllerInput) -> np.ndarray:
    cx = controller.scaled_ctrl
    v = np.array([linearize_clamp_abs(cx[ButtonKind.LEFT_X], 0.5, 1, 0, 1),cx[ButtonKind.LEFT_Y]])
    res =  getp(v) * max_norm
    return res
    
g_bh.stop_sim()
g_bh.load_scene(test2_idx, qd0=np.array([0,0,0,0,0,0]), cparams = ControlParameters(ndt_const=1, dt=0.8e-2, integ_nsteps=1, nt=1, 
                                                                                    use_jit=1,
                                                                                   use_rk4=0,),
                      input_params=InputControllerParameters(controller2ctrl=controller2ctrl, ),
                use_jit=1,
                                                            use_gamepad=1,

                
               )
i02 = g_bh.i0.copy()
i02.qd = np.array([0,0,0,0,0,0])
g_bh.sim.load_state(i02)
g_bh.run_sim(override_ctrl=None)


# In[48]:


g_bh.stop_sim()


# In[28]:


g_bh.run_sim(override_ctrl=None)
#g_bh.run_sim(override_ctrl=g_bh.sim.ss.ctrl_packer.default)


# In[26]:


from chdrft.utils.math import rot_look_at
from __future__ import annotations
class BlenderCam(cmisc.PatchedModel):
    cam: object
    
    @classmethod
    def Active(cls) -> BlenderCam:
        return BlenderCam(cam=bpy.context.scene.camera)
    
    def set_wl(self, wl):
        m0  = wl @ Transform.From(rot=np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T)
        self.cam.matrix_world= m0.data.T
        
        
    @property
    def wl(self) -> Transform:
        return Transform.From(data=np.array(self.cam.matrix_world)) @ Transform.From(rot=np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
        
    

cm = BlenderCam.Active()
cm.set_wl(Transform.From(pos=[0,0,10], rot=rot_look_at([0,0,-1], [1,0,0])))

class Trackball(cmisc.PatchedModel):
    cam: BlenderCam
    target: Vec3
        
    @property
    def dist(self) -> float:
        return (self.cam.wl.pos_v - self.target).norm
        
    def dx(self, dx):
        return self.dt(R.from_rotvec(Vec3.Y().vdata * dx))
    def dy(self, dy):
        return self.dt(R.from_rotvec(Vec3.X().vdata * dy))
        
    def dt(self, dt):
        dt = self.cam.wl @ Transform.From(rot=dt)
        dt.pos_v += self.target - dt @ (Vec3.Z() * self.dist).as_pt
        self.cam.set_wl(dt)
        
    def dr(self, dr):
        wl = self.cam.wl
        wl.pos_v = (wl.pos_v - self.target) * max(0.1, 1+dr) + self.target
        self.cam.set_wl(wl)
        
    def set_target(self, tg : Vec3):
        print(self.dist)
        wl = self.cam.wl
        wl.pos_v = (wl.pos_v - self.target) + tg
        self.cam.set_wl(wl)
        self.target = tg
        
        
        
tb = Trackball(cam=cm, target=Vec3.ZeroPt())
tb.dr(-0.5)
tb.set_target(Vec3(np.array([1,0,0]), vec=0))


# In[27]:


from chdrft.sim.rb.rb_player import *

def procit(state: SceneControllerInput):
    def norm(v):
        return linearize_clamp_abs(v, 0.1, 0.9, 0, 1) * 0.05
    tb.dx(norm(state.inputs[ButtonKind.RIGHT_X]))
    tb.dy(norm(state.inputs[ButtonKind.RIGHT_Y]))
    if state.mod1:
        tb.dr(norm(state.inputs[ButtonKind.LTRT]))
    tb.set_target(g_bh.root.agg_com)
g_bh.cbs.clear()
g_bh.cbs.append(lambda: procit(g_bh.ctrl.ctrl_obs.last))    
    


# In[28]:


g_bh.stop_sim()


# In[4]:


def area_of_type(type_name):
    for area in bpy.context.screen.areas:
        if area.type == type_name:
            return area

def get_3d_view():
    return area_of_type('VIEW_3D').spaces[0]

view3d = get_3d_view()


# In[12]:





# In[11]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from mathutils import Matrix
b.region_3d.view_matrix Transform.From(rot=R.from_rotvec(np.array([0,0,1])*2*np.pi/100).as_matrix()).data


# In[ ]:




