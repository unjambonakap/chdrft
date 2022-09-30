#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import calcephpy
from astroquery.jplhorizons import Horizons
from chdrft.display.vtk import vtk_main_obj
import chdrft.display.vtk as opa_vtk
from chdrft.sim.utils import *
from chdrft.sim.base import *
import numpy as np
import spiceypy

ctx = app.setup_jup('', parser_funcs=[])


# In[2]:


METAKR = '/home/benoit/programmation/science/sim/kernels/lro.km'
spiceypy.kclear()
spiceypy.furnsh( METAKR )
def norm_date(d):
    return pytz.utc.localize(datetime.datetime.fromisoformat(d))
def get_data_at_time(t_utc):
    if not isinstance(t_utc, datetime.datetime): t_utc = datetime.datetime.utcfromtimestamp(t_utc)
    et = spice_time(t_utc)
    moon_data = spiceypy.spkezr('moon', et, 'GSE', 'LT+S', 'earth')
    sat_data = spiceypy.spkezr('LRO', et, 'GSE', 'LT+S', 'earth')
    
    res = A()
    res.moon_pos = moon_data[0][:3]
    res.sat_pos = sat_data[0][:3]
    res.moon_rot = spiceypy.pxform('MOON_ME_DE440_ME421', 'GSE', et)
    res.sat_rot = spiceypy.pxform('LRO_LROCNACL', 'GSE', et)
    res.earth_rot = spiceypy.pxform('ITRF93', 'GSE', et)
    
    res.earth_pos=np.array((0,0,0))
    res.t =  t_utc.timestamp()
    return res


# In[3]:


tgtime = norm_date('2015-10-12 12:18:40')
dt = get_data_at_time(tgtime)
earth_pos=[0,0,0]
tsf =Z.MatHelper.simple_mat(offset=dt.sat_pos, rot=dt.sat_rot)
Z.MatHelper.mat_apply_nd(np.linalg.inv(tsf), earth_pos, point=True)


# In[4]:


start = norm_date('2015-10-12')
end = norm_date('2015-10-13')


start = tgtime -datetime.timedelta(minutes=0)
end = tgtime +datetime.timedelta(minutes=0.1)
print(datetime.datetime.utcfromtimestamp(tgtime.timestamp()))
print(start, end)
tl = pd.date_range(start, end, 100)

res = []
for t_utc in tl.to_pydatetime():
    res.append(get_data_at_time(t_utc))
rdf= Z.pd.DataFrame(res)
rdf = rdf.set_index('t')

Z.FileFormatHelper.Write('/tmp/res2.pickle', res)


# In[11]:


ctx= A(width=800, height=600, offscreen=True, rot_angle=0, zoom_factor=None, nframes=1, outfile=None)

#dx = Z.FileFormatHelper.Read('/tmp/res2.pickle')
dfx = InterpolatedDF(rdf, kind='cubic')

t0_utc = rdf.index[0]
t1_utc = rdf.index[-1]


earth = EarthActor()
cam = CamActor()
actors = [earth, 
          cam]

def pos_rot_func(poskey, rotkey):
    def f(actor, t, data=None, **kwargs):
      actor.set_pos_and_rot(data[poskey], data[rotkey])
    return f

def func_cam(self, t, first=None, data=None, **kwargs):
    self.set_pos(data.sat_pos)
    if first:
        self.focus_on_points((0,0,0), (0,0,1), earth.get_pts_world(), ctx.rot_angle, ctx.zoom_factor)

        
def state_cb(data, tdesc):
    return A(want=1, label='123', overlay=[  ])
    from_box=Z.opa_struct.g_one_box
    proj_moon = MatHelper.mat_apply_nd(cam.proj_mat, moon.get_pts_world().T, n=3, point=True).T
    proj_earth = MatHelper.mat_apply_nd(cam.proj_mat, earth.get_pts_world().T, n=3, point=True).T
    proj_moon = proj_moon[np.abs(proj_moon[:,2]) < 1]

    target_box = Z.g_unit_box
    proj_moon = from_box.change_rect_space(target_box, proj_moon[:,:2])
    proj_earth = from_box.change_rect_space(target_box, proj_earth[:,:2])
    #pts = list(Z.shapely.geometry.MultiPoint(proj_moon).convex_hull.exterior.coords)
    #print(pts)

    moon_hull = None
    earth_hull = None
    view_box = Z.g_unit_box.shapely
    if len(proj_moon) > 0:
        moon_hull = Z.geometry.MultiPoint(proj_moon).convex_hull
        moon_hull =moon_hull.intersection(view_box)
    if len(proj_earth) > 0:
        earth_hull = Z.geometry.MultiPoint(proj_earth).convex_hull
    
    tstr = str(datetime.datetime.utcfromtimestamp(tdesc.t))
    res =  A(label=f'{tdesc.idx} > {tstr}', overlay=[A(lines=[earth_hull, moon_hull], color='r' )])
    res.want = moon_hull and earth_hull and not moon_hull.contains(earth_hull) and moon_hull.intersects(view_box)
    res.want = True
    return res

ren = Renderer(ctx.width, ctx.height, offscreen=ctx.offscreen, actors=actors, dataf=get_data_at_time, state_cb=state_cb)
    

earth.runt = pos_rot_func('earth_pos', 'earth_rot')
#moon.runt = pos_rot_func('moon_pos', 'moon_rot')
cam.runt = pos_rot_func('sat_pos', 'sat_rot')
cam.runt = func_cam

tl = np.linspace(t0_utc, t1_utc, ctx.nframes)
ren.process(tl, outfile=ctx.outfile)

