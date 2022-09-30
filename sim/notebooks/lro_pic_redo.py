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
import cv2

ctx = app.setup_jup('', parser_funcs=[])


# In[2]:


METAKR = '/home/benoit/programmation/science/sim/kernels/lro.km'
spiceypy.kclear()
spiceypy.furnsh( METAKR )
ctx= A(view_angle=16, width=800, height=600, offscreen=True, rot_angle=0, zoom_factor=None, nframes=100, outfile=None, no_render=0)
ctx.aspect= ctx.width / ctx.height
def norm_date(d):
    return pytz.utc.localize(datetime.datetime.fromisoformat(d))
def get_data_at_time(t_utc):
    if not isinstance(t_utc, datetime.datetime): t_utc = datetime.datetime.utcfromtimestamp(t_utc)
    et = spice_time(t_utc)
    sun_data, sun_lt = spiceypy.spkezr('SUN', et, 'GSE', 'LT+S', 'LRO')
    earth_data, earth_lt = spiceypy.spkezr('EARTH', et, 'GSE', 'LT+S', 'LRO')
    moon_data, moon_lt = spiceypy.spkezr('moon', et, 'GSE', 'LT+S', 'LRO')
    sat_data, sat_lt = spiceypy.spkezr('LRO', et, 'GSE', 'LT+S', 'LRO')
    print(moon_lt, earth_lt, sun_lt, sat_lt)
    
    res = A()
    res.earth_pos = earth_data[:3]
    res.moon_pos = moon_data[:3]
    res.sat_pos = sat_data[:3]
    res.moon_rot = spiceypy.pxform('MOON_ME_DE440_ME421', 'GSE', et-moon_lt)
    res.sat_rot = spiceypy.pxform('LRO_LROCNACL', 'GSE', et)
    res.earth_rot = spiceypy.pxform('ITRF93', 'GSE', et-earth_lt)
    res.sun_pos = sun_data[:3]
    
    res.t =  t_utc.timestamp()
    return res

def setup_camera(cam, tsfmat, view_angle=None, aspect=None):
    cam.SetPosition(0, 0, 0)
    cam.SetFocalPoint([0, 0, 1])
    cam.SetViewAngle(view_angle)
    cam.SetViewUp([0,1,0])
    cam.SetClippingRange(1, 1e20)
    
    a = opa_vtk.vtk.vtkTransform()
    a.SetMatrix(numpy_to_vtk_mat(tsfmat))
    cam.ApplyTransform(a)
    if aspect is not None: return opa_vtk.vtk_matrix_to_numpy(cam.GetCompositeProjectionTransformMatrix(aspect, -1, 1))


# In[3]:


tgtime = norm_date('2015-10-12 12:18:33')
dt = get_data_at_time(tgtime)
dt


# In[17]:


tsf =Z.MatHelper.simple_mat(offset=dt.sat_pos, rot=dt.sat_rot)
cammat = setup_camera(opa_vtk.vtk.vtkCamera(), tsf, ctx.view_angle, ctx.aspect)

latvals = np.linspace(-0.5, 0.5, 1000) * np.pi
lonvals = np.linspace(0, 2, 1000) * np.pi
glat, glon = np.meshgrid(latvals, lonvals)
glat = glat.flatten()
glon = glon.flatten()
pts = np.array(pymap3d.geodetic2ecef(glat, glon, 0, ell=Consts.MOON_ELLIPSOID, deg=0)) / 1e3
moon2world = MatHelper.mat_apply_nd(MatHelper.mat_translate(dt.moon_pos), MatHelper.mat_rot(dt.moon_rot))
projs = MatHelper.mat_apply_nd(cammat, moon2world, pts, point=True).T
sel_ids= np.max(np.abs(projs), axis=1) <= 1
slat = glat[sel_ids]
slon = glon[sel_ids]
sel_pts = np.vstack([slon, slat]).T
boundary =Z.geo_ops.MultiPoint(sel_pts).convex_hull.buffer(0.01)
Z.FileFormatHelper.Write('./boundary.pickle', boundary)


# In[16]:


img = K.ImageData(cv2.imread('./Moon_LRO_LOLA_global_LDEM_1024.jpg'), box=Z.Box(yr=(latvals[0], latvals[-1]), xr=(lonvals[0], lonvals[-1])))
data = A(points=sel_pts, images=[img], lines=[boundary])
oplt.plot(data)


# In[5]:


start = norm_date('2015-10-12')
end = norm_date('2015-10-13')


start = tgtime -datetime.timedelta(minutes=0.5)
end = tgtime +datetime.timedelta(minutes=0.5)
print(datetime.datetime.utcfromtimestamp(tgtime.timestamp()))
print(start, end)
tl = pd.date_range(start, end, 100)

res = []
for t_utc in tl.to_pydatetime():
    res.append(get_data_at_time(t_utc))
rdf= Z.pd.DataFrame(res)
rdf = rdf.set_index('t')

Z.FileFormatHelper.Write('/tmp/res2.pickle', res)


# In[3]:


import meshio
if 0: m1 = meshio.Mesh.read('t1.small.stl')
else: m1 = meshio.Mesh.read('t1.stl')


# In[4]:


#dx = Z.FileFormatHelper.Read('/tmp/res2.pickle')
#dfx = InterpolatedDF(rdf, kind='cubic')

#t0_utc = rdf.index[0]
#t1_utc = rdf.index[-1]


earth = EarthActor()
if 0: moon = MoonActor()
else:
    rimg = np.array([[255,0,0]], dtype=np.uint8)
    trx = opa_vtk.TriangleActor(tex=opa_vtk.numpy2tex(rimg))
    trx.add_points(m1.points)
    for x in m1.points:
        trx.tex_coords.InsertNextTuple2(0, 0)
    triangles = m1.cells_dict['triangle']
    for x in triangles:
        trx.push_triangle(x)

    moon = Actor(ActorType.Moon)
    moon.actor = trx.build()
    moon.pts = m1.points
    moon.setup_internal = lambda *args: None

    
cam = CamActor()
light = Actor(ActorType.Light)
actors = [earth, 
          moon, 
          cam, light]

def setup_light(lx):
    lx.actor = opa_vtk.vtk.vtkLight()
    lx.actor.SetPositional(False)
    lx.actor.SetColor(opa_vtk.Color('white').rgb)
    
light.setup_f = setup_light

def pos_rot_func(poskey, rotkey):
    def f(actor, t, data=None, **kwargs):
      actor.set_pos_and_rot(data[poskey], data[rotkey])
    return f


        
def state_cb(data, tdesc):
    from_box=Z.opa_struct.g_one_box
    proj_moon = MatHelper.mat_apply_nd(cam.proj_mat, moon.get_pts_world().T, n=3, point=True).T
    proj_earth = MatHelper.mat_apply_nd(cam.proj_mat, earth.get_pts_world().T, n=3, point=True).T
    proj_moon = proj_moon[np.abs(proj_moon[:,2]) < 1]
    proj_earth = proj_earth[np.abs(proj_earth[:,2]) <= 1.01]
    
    

    target_box = Z.g_unit_box
    proj_moon = from_box.change_rect_space(target_box, proj_moon[:,:2])
    proj_earth = from_box.change_rect_space(target_box, proj_earth[:,:2])
    #pts = list(Z.shapely.geometry.MultiPoint(proj_moon).convex_hull.exterior.coords)
    #print(pts)

    moon_hull = None
    earth_hull = None
    view_box = Z.g_unit_box.shapely
    hulls = []
    if len(proj_moon) > 0:
        moon_hull = Z.geometry.MultiPoint(proj_moon).convex_hull
        moon_hull =moon_hull.intersection(view_box)
        if moon_hull.intersects(view_box): hulls.append(moon_hull)
    if len(proj_earth) > 0:
        earth_hull = Z.geometry.MultiPoint(proj_earth).convex_hull
        if earth_hull.intersects(view_box): hulls.append(earth_hull)
    
    tstr = str(datetime.datetime.utcfromtimestamp(tdesc.t))
    res =  A(label=f'{tdesc.idx} > {tstr}', overlay=[A(lines=hulls, color='r' )])
    res.want = moon_hull and earth_hull and not moon_hull.contains(earth_hull) and moon_hull.intersects(view_box)
    res.want = True
    return res

def func_cam(self, t, first=None, data=None, **kwargs):
    
    m = MatHelper.mat_apply_nd(MatHelper.mat_translate(data.sat_pos), MatHelper.mat_rot(data.sat_rot))
    setup_camera(self.cam, m, view_angle=10)
    return
    assert 0
    if not first: 
        self.set_pos(data.sat_pos)
        return
    
    
def func_light(actor, t, data=None, **kwargs):
    actor.actor.SetPosition(data.sun_pos)

    
    

    
    

earth.runt = pos_rot_func('earth_pos', 'earth_rot')
moon.runt = pos_rot_func('moon_pos', 'moon_rot')
cam.runt = pos_rot_func('sat_pos', 'sat_rot')
cam.runt = func_cam
light.runt = func_light


# In[7]:


ctx.nframes= 1
ctx.width=2000
ctx.height=2000
ren = Renderer(ctx.width, ctx.height, offscreen=ctx.offscreen, actors=actors, dataf=get_data_at_time, state_cb=state_cb)

tgtime = norm_date('2015-10-12 12:18:31')
nstart = tgtime + datetime.timedelta(seconds=0)
nend = tgtime + datetime.timedelta(seconds=0)
tl = np.linspace(nstart.timestamp(), nend.timestamp(), ctx.nframes, endpoint=False)
ren.process(tl, outfile=ctx.outfile, no_render=ctx.no_render)


# In[8]:


print(1)


# In[6]:


img= K.ImageData(cv2.imread('/home/benoit/data/moon/earth_moon_lro.jpg')[::-1,:,::-1])
oplt.plot(A(images=[img]))


# In[8]:


print

