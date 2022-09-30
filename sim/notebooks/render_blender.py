#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()
from PyQt5 import QtGui, QtCore, QtWidgets, QtTest
import bpy
from chdrft.sim.moon_sunrise import *
ctx = app.setup_jup('--cache-file=/tmp/render_blender.cache.pickle', parser_funcs=[render_params])

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
import chdrft.sim.blender as ms_blender
from chdrft.display.blender import *
from mathutils import Euler
import mathutils


# In[2]:


#data_blocks = {node for node in bpy.data.node_groups if node.name == 'node_earth_mat'}
#bpy.data.libraries.write('./resources.blend', data_blocks, fake_user=True)


# In[2]:


def build_ms():
    ms = ms_blender.MoonSunriseBlender(ctx)

    ctx.width = 2000
    ctx.height = 2000
    ctx.earth_depth = 4
    ctx.earth_tile_depth = 4
    ms.build()
    earth = ms.objs[ActorType.Earth]
    cam = ms.objs[ActorType.Cam]
    sun = ms.objs[ActorType.Light]
    moon = ms.objs[ActorType.Moon]
    
    ms.configure_earth_mat(Specular=0)
    ms.configure_cam(view_angle=ctx.view_angle, res=(ctx.width, ctx.height))
    ms.configure_at(ctx.t0)
    
    radius_sun = 696340 * ms.km2u
    ang_sun =  math.asin(radius_sun / np.linalg.norm((moon._data.pos - sun._data.pos)))
    print(ang_sun)
    ang_sun = np.deg2rad(2) # increase because 
    sun.obj.data.angle = ang_sun
    np.set_printoptions(precision=8)
    return ms

def render_scene(ms, tl, outpath, dataf=lambda _: None):
    cmisc.makedirs(outpath)
    for i,t in enumerate(tl):
        print(f'on {i=} {t=} {outpath=}')
        ms.configure_at(t, data=dataf(t))
        ms.render(os.path.join(outpath, f'img_{i:04d}.png'))
    Z.FileFormatHelper.Write(os.path.join(outpath, 'data.json'), tl)


# In[4]:


set_analysis_parameters(ctx, 'polar', large=True)
ms = build_ms()
ms.configure_cam(res=(1500, 200))
tl1 = pd.date_range(ctx.t0, ctx.t0  + datetime.timedelta(seconds=300), 100)
render_scene(ms, tl1, 'vid_pole_narrow')
ms.configure_cam(res=(1500, 1500))
render_scene(ms, tl1, 'vid_pole_wide')


# In[7]:


set_analysis_parameters(ctx, 'lro_earthrise', large=True)
ms = build_ms()

ms.configure_cam(res=(1500, 1500))
tl1 = pd.date_range(ctx.t0+datetime.timedelta(seconds=-40), ctx.t0  + datetime.timedelta(seconds=40), 80)
render_scene(ms, tl1, 'vid_earthrise')


# In[8]:


set_analysis_parameters(ctx, 'kaguya', large=True)
ms = build_ms()
ms.configure_at(ctx.t0)


# In[24]:



ctx.obs_conf = A(obs='LRO', obs_frame='LRO_LROCNACL')
ms.configure_at(ctx.t0)


# In[9]:



bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080


# In[20]:


cam = ms.objs[ActorType.Cam]
cam.obj.data.angle_y = np.deg2rad(29.45)
cam.obj.data.angle_x = np.deg2rad(50.12)
bpy.context.scene.render.resolution_x = 1920*2
bpy.context.scene.render.resolution_y = 1080*2


# In[28]:



t0 = ctx.t0+datetime.timedelta(seconds=90)
ms.configure_at(t0)


# In[28]:


bpy.context.scene.animation_data_clear()
for o in bpy.context.scene.objects:
    o.animation_data_clear()
   


# In[29]:



fid = 0
frame_step =1 


so = ms.objs.values()
start = tx0 + datetime.timedelta(seconds=0)
end = tx0 + datetime.timedelta(seconds=300)
nsteps = 500
tl = pd.date_range(start, end, nsteps)


bpy.context.scene.frame_set(0)
for y in so: y.rotation_mode = 'QUATERNION'
for t in tl:
   bpy.context.scene.frame_set(fid)
   if 0: ms.configure_at(data=ms.get_data_at_time(t))
   else: ms.configure_at(t)
   for y in so:
       y.obj.matrix_world = mathutils.Matrix()
       rot = y._toworld[:3,:3]

           
       y.obj.rotation_mode = 'QUATERNION'
       y.obj.rotation_quaternion = blender_quat(R.from_matrix(rot))
       #y.obj.rotation_quaternion = blender_quat(R.identity())
       y.obj.location = y._data.pos
       
       #y.obj.keyframe_insert(data_path='location', index=-1)
       #y.obj.keyframe_insert(data_path='rotation_quaternion', index=-1)
       y.obj.keyframe_insert(data_path='location', frame=fid)
       y.obj.keyframe_insert(data_path='rotation_quaternion', frame=fid)
   fid +=  frame_step
   
bpy.context.scene.frame_set(0)


# In[6]:


app.cache.flush_cache()

