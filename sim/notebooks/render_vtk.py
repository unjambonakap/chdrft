#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()
get_ipython().run_line_magic('gui', 'qt5')
from PyQt5 import QtGui, QtCore, QtWidgets, QtTest
from chdrft.sim.moon_sunrise import *
ctx = app.setup_jup('--cache-file=/tmp/render_vtk.cache.pickle', parser_funcs=[render_params])

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('aimport', 'chdrft.sim.vtk')
get_ipython().run_line_magic('aimport', 'chdrft.display.vtk')
import chdrft.sim.vtk as ms_vtk


ctx.moon_model = 'pic'
ctx.earth_depth = 3
ctx.earth_tile_depth = 3
ctx.moon_details=7
t0  = norm_date('2007-11-07T12:25:00')
ctx.moon_model = 'pic'
ctx.obs_conf = A(obs='SELENE', obs_frame='SELENE_HDTV_WIDE')

ctx.view_angle = 10

np.set_printoptions(8)


# In[77]:


def obj2cspace(cam, obj, clip_bound=1):
    
    o2clip = cam.internal.local2clip @ np.linalg.inv(cam._toworld) @ obj._toworld
    o2world = obj._toworld
    ocspace = MatHelper.mat_apply_nd(o2clip, np.array(obj.internal.points).T, point=1).T
    ocspace = ocspace[np.abs(ocspace)[:,2] < 1][:,:2]
    hull = None
    view_box = Z.g_one_box.shapely
    
    if len(ocspace) > 0:
        hull = Z.geometry.MultiPoint(ocspace).convex_hull
        hull =hull.intersection(view_box)
    ocspace = ocspace[np.max(np.abs(ocspace), axis=1) < clip_bound]
    return A(cspace=ocspace, hull=hull)
    
def compute_state(idx, t, filter_func, expand=0.1):
    ms.configure_at(t)
    earth_wspace = MatHelper.mat_apply_nd(earth._toworld, np.array(earth.internal.points).T, point=1).T
    cam_params = compute_cam_parameters(cam._data.pos, earth._data.pos, [0,1,0], earth_wspace, aspect=cam.internal.aspect, expand=expand)
    cam._data.rot = cam_params.rot
    if 1:
        ms.configure_obj(cam,cam._data)
        ms.configure_cam(view_angle=np.rad2deg(cam_params.angle_box.yn))
    moon_cspace = obj2cspace(cam, moon, clip_bound=1.00)
    earth_cspace = obj2cspace(cam, earth, clip_bound=1.01)
    hulls = [x.hull for x in (moon_cspace, earth_cspace) if x.hull is not None]
    
    tstr = str(datetime.datetime.utcfromtimestamp(t))
    res =  A(label=f'{idx} -> {tstr}', overlay=[A(lines=hulls, color='r' )])
    res.earth = earth_cspace
    res.earth_cspace = earth_cspace
    res.moon_cspace = moon_cspace
    
    res.want = filter_func(res)
    if res.want: res.img = ImageData(ms.render(''), stuff=res, box=Z.g_one_box)
    return res

def plot_states(tl, filter_func=lambda _: True):
    states = [compute_state(i,t, filter_func) for i,t in enumerate(tl)]
    want_states = [x for x in states if x.want]
    if len(want_states) == 0:
        print('no wanted images')
        return
    imgs  = [x.img for x in want_states]
    ig = ImageGrid(images=imgs)
    grid_imgs = ig.get_images()
    
    mo = cmisc.Attr(images=grid_imgs, misc=[])
    meshes = [mo]

    for e in grid_imgs:
      for ov in e.stuff.overlay:
        ov.transform = e.box.to_vispy_transform()* Z.g_one_box.to_vispy_transform().inverse 
        meshes.append(ov)
      mo.misc.append(A(text=e.stuff.label, pos=e.pos, zpos=-10))
    oplt.plot(meshes)
    
def filter_func(data):
    return data.moon_cspace.hull and data.earth_cspace.hull and not data.moon_cspace.hull.contains(data.earth_cspace.hull)


# In[29]:


ctx.offscreen = 1
ms = ms_vtk.MoonSunriseVTK(ctx)
ms.build()
cam = ms.objs[ActorType.Cam]
earth = ms.objs[ActorType.Earth]
moon = ms.objs[ActorType.Moon]


# In[9]:


import poliastro
import poliastro.twobody.orbit as orbit
import poliastro.bodies as bodies
from astropy import units as u

from astropy.time import Time
o = orbit.Orbit.from_vectors(bodies.Moon, moon._data.pos*u.km , moon._data.v * (u.km / u.s))

o.period.to(u.hour)


# In[84]:


ctx.obs_conf.obs_frame = 'SELENE_HDTV_TELE'
ctx.obs_conf.obs_frame = 'SELENE_HDTV_WIDE'
tx = norm_date('2007-11-07T05:53:26')+datetime.timedelta(minutes=10)
ctx.view_angle=100
ms.configure_cam(view_angle=ctx.view_angle)
ms.configure_at(tx)
ms.ren.update()
ms.ren.ren_win.Render()
moon._data.pos, cam._data.rot
#ImageData(ms.render('')).plot()


# In[90]:


t0.astimezone(pytz.utc)


# In[87]:


norm_date('2007-11-07T14:57:00', tz=jst).astimezone(pytz.utc)


# In[73]:





# In[71]:


MatHelper.mat_apply_nd(cam.internal.local2clip @ np.linalg.inv(cam._toworld), earth._data.pos.T, point=1)


# In[24]:


ms.configure_render(1000, 1000)
r = compute_state(0, ms.tgtime.timestamp(), lambda _:True, expand=2)
oplt.plot([A(images=[r.img], points=r.moon_cspace.cspace), A(points=r.earth_cspace.cspace, color='r')])


# In[82]:


ms.configure_render(200, 200)
starttime = norm_date('2015-10-12')
endtime = norm_date('2015-10-13')
    
tl= np.linspace(starttime.timestamp(), endtime.timestamp(), 48)
plot_states(tl)


# In[83]:



norm_date('2007-11-07 14:50', tz=jst).astimezone(pytz.utc)


# In[ ]:





# In[13]:


starttime.timestamp()


# In[14]:



norm_date('2007-11-07').timestamp()


# In[42]:


ctx.obs_conf.obs_frame = 'SELENE_HDTV_WIDE'


# In[17]:


ms.configure_render(400, 400)
starttime = norm_date('2015-10-12 11:44:00')
endtime = norm_date('2015-10-12 13:15:00')
starttime = norm_date('2007-11-07 12:50', tz=jst)
endtime = norm_date('2007-11-07 13:10', tz=jst)
tl= np.linspace(starttime.timestamp(), endtime.timestamp(), 200)
plot_states(tl, filter_func)


# In[16]:


app.cache.flush_cache()

