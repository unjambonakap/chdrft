#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('gui', 'qt5')
from chdrft.config.env import g_env
g_env.set_qt5(1)
init_jupyter()
import georinex as gr
#import ppp_tools.gpstime
import chdrft.utils.misc as cmisc
from chdrft.display.vtk import vtk_main_obj
import chdrft.display.vtk as opa_vtk
import  scipy.constants as constants
import chdrft.utils.K as K
from chdrft.geo.satsim import TileGetter
import mercantile
import pymap3d
from geopy.geocoders import Nominatim
import cv2
import calcephpy
from astroquery.jplhorizons import Horizons

from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body, get_moon
import datetime
from astropy import units as U
import math
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS


# In[2]:


tg = TileGetter()


def ecef_to_latlng(x,y,z):
  lng = math.atan2(y,x)
  lat = math.atan2(z, np.linalg.norm((x,y)))
  return Z.rad2deg(np.array((lat, lng)))

def eph_to_xyz(eph, unit=U.AU):
  a = SkyCoord(ra=eph['RA'], dec=eph['DEC'], distance=eph['delta'])
  return a.cartesian.xyz.to(unit).value.T


# In[3]:


SkyCoord(ra=10.68458*U.degree, dec=41.26917*U.degree, distance=770*U.km).cartesian


# In[4]:


x = Horizons(id=calcephpy.NaifId.MOON,
                  location=f'@{calcephpy.NaifId.EARTH}',
                  epochs={'start':'2019-01-01',
                          'stop':'2019-01-25',
                          'step': '5h'},
                  id_type='majorbody')
    


# In[26]:


x.ephemerides()


# In[5]:


eph_to_xyz(dict(RA=1 * U.deg, DEC=0 * U.deg, delta=123 * U.km))


# In[ ]:


app.exit_jup()


# In[6]:


nav = gr.load('../../../dsp/gps/brdc3640.15n')
nav = gr.load('/home/benoit/Downloads/brdc1850.19n')


# In[ ]:


nav


# In[7]:


import datetime
import time
from astropy import constants as const

from astropy.coordinates.earth import OMEGA_EARTH
class Ephemerid:
  def __init__(self, s):
    self.s = s
    self.a = s['sqrtA'] ** 2
    
  @property
  def utc(self):
    time_utc= ppp_tools.gpstime.UTCFromGps(self.s['GPSWeek'], self.s['Toe'])
    time_utc = list(time_utc)
    time_utc[-1] = int(time_utc[-1])
    return datetime.datetime(*time_utc)

  
  def get_pos(self, t):
    
    dt = t -  self.utc
    dt_sec =dt.total_seconds()
    print('DT >>> ', dt)
    
    sv =self.s
    
    n0 = np.sqrt(const.GM_earth.value/self.a**3)  # computed mean motion
#    T = 2*pi / n0  # Satellite orbital period
    omega_e = OMEGA_EARTH.value

    n = n0 + sv['DeltaN']
    e = sv['Eccentricity']
# %% Kepler's eqn of eccentric anomaly
    Mk = sv['M0'] + n*dt_sec  # Mean Anomaly
    Ek = Mk + e * np.sin(Mk)  # Eccentric anomaly
# %% true anomaly
    nuK = np.arctan2(np.sqrt(1 - e**2) * np.sin(Ek),
                     np.cos(Ek) - e)
# %% latitude
    PhiK = nuK + sv['omega'] # argument of latitude
    duk = sv['Cuc'] * np.cos(2*PhiK) + sv['Cus']*np.sin(2*PhiK)  # argument of latitude correction
    uk = PhiK + duk  # corred argument of latitude
# %% inclination (same)
    dik = sv['Cic']*np.cos(2*PhiK) + sv['Cis']*np.sin(2*PhiK)  # inclination correction
    ik = sv['Io'] + sv['IDOT']*dt_sec + dik  # corrected inclination
# %% radial distance (same)
    drk = sv['Crc'] * np.cos(2*PhiK) + sv['Crs'] * np.sin(2*PhiK)  # radial correction
    rk = self.a * (1 - e * np.cos(Ek)) + drk  # corrected radial distance
# %% right ascension  (same)
    OmegaK = sv['Omega0'] + (sv['OmegaDot'] - omega_e)*dt_sec - omega_e*sv['Toe']
# %% transform
    Xk1 = rk * np.cos(uk)
    Yk1 = rk * np.sin(uk)

    X = Xk1 * np.cos(OmegaK) - Yk1 * np.sin(OmegaK) * np.cos(ik)

    Y = Xk1*np.sin(OmegaK) + Yk1 * np.cos(OmegaK) * np.cos(ik)

    Z = Yk1*np.sin(ik)
    return X,Y,Z


# In[3]:





# In[8]:


def analyse_sat(sat):
  tmp = nav.sel(sv=sat)
  df = tmp.to_dataframe().dropna()
  last = df.iloc[-1]
  a = Ephemerid(last)

  now = datetime.datetime(2015, 12, 30, 22)
  now = datetime.datetime.utcnow()
  xyz= a.get_pos(now)
  print(xyz)
  res = pymap3d.ecef2geodetic(*xyz)
  print(res)
  return res


# In[ ]:


for val in nav['sv'].values:
  print('\n\n>>>', val)
  analyse_sat(val)


# In[ ]:


gr.keplerian2ecef(tmp)


# In[9]:


body_a = calcephpy.NaifId.EARTH
from astroquery.jplhorizons import Horizons
obj = Horizons(id='2019 OK', location=f'@{calcephpy.NaifId.EARTH}',
epochs={'start':'2019-06-01', 'stop':'2019-08-01',
'step':'1d'})
eph = obj.ephemerides()


# In[ ]:


print(eph)


# In[ ]:



start = '1969-07-16 16:40'
end = '1969-07-28'
start = '2019-06-01'
end = '2019-08-01'
body_center = calcephpy.NaifId.MARS
body_a = calcephpy.NaifId.PHOBOS
body_b = calcephpy.NaifId.DEIMOS
body_b =-399110
body_a = calcephpy.NaifId.MOON

bodies = [calcephpy.NaifId.DEIMOS, calcephpy.NaifId.PHOBOS, -3, -41, -74, ]
body_center = calcephpy.NaifId.MARS
bodies = [calcephpy.NaifId.EARTH, calcephpy.NaifId.MOON, '2019 OK']
body_center = calcephpy.NaifId.SUN
bodies = [calcephpy.NaifId.MOON, '2019 OK']
body_center = calcephpy.NaifId.EARTH
step = '1d'
objs = []
for body in bodies:
  objs.append(Horizons(id=body,
                 location=f'@{body_center}',
                 epochs={'start':start,
                         'stop':end,
                         'step':step},
                 id_type='majorbody'))


# In[4]:





# In[ ]:


eph_to_xyz(obj.ephemerides(), U.km)


# In[ ]:


import chdrft.utils.K as K


# In[ ]:


get_ipython().run_line_magic('gui', 'qt4')


# In[ ]:


main = vtkMain()


for obj in objs:
  pts = eph_to_xyz(obj.ephemerides())
  actor = opa_vtk.create_line_actor(pts)
  main.ren.AddActor(actor)

sp = opa_vtk.SphereActor(1e-5, 10, (0,0,0), K.vispy_utils.Color('y').rgb)
main.ren.AddActor(sp)


# In[ ]:


main.run()


# In[ ]:


main.app.quit()


# In[ ]:


from sbpy.data import Orbit
from astropy.time import Time
epoch = Time('2018-05-14', scale='utc')
eph = Orbit.from_horizons('Ceres', epochs=epoch)


# In[ ]:


epoch2 = Time('2018-05-15', scale='utc')
eph2 = eph.oo_propagate(epoch2)


# In[ ]:



import pyoorb as oo
oo.pyoorb.oorb_init()
from sbpy.data import Orbit
from astropy.time import Time
epoch = Time.now().jd + 100
ceres = Orbit.from_horizons('Ceres')      # doctest: +REMOTE_DATA
future_ceres = ceres.oo_propagate(epoch)  # doctest: +SKIP
print(future_ceres)  # doctest: +SKIP


# In[ ]:


cart = eph2.oo_transform('CART')


# In[ ]:


cart['x']


# In[ ]:


for i in range(10):
  print(a[i].cartesian.x)


# In[ ]:


type(elems[0])


# In[ ]:


import reverse_geocoder as rg
coordinates = (51.5214588,-0.1729636),(9.936033, 76.259952),(37.38605,-122.08385)

results = rg.search(coordinates) # default mode = 2

print(results)


# In[ ]:


gl = Nominatim(user_agent='VGYV7gGlNoWapA==')
obj = gl.geocode('Paris')
tile = mercantile.tile(obj.longitude, obj.latitude, 12)


# In[ ]:


from chdrft.utils.cache import global_cache_list
global_cache_list


# In[ ]:


tx = tg.get_tile(*tile)
img = K.opa_vtk.read_img_from_buf(tx)


# In[ ]:


K.vispy_utils.render_for_meshes([cmisc.Attr(images=[img[::-1]])])
#K.plot_img(img[::-1])


# In[ ]:


open('/tmp/test.png', 'wb').write(tx)


# In[ ]:


img = cv2.imread('/tmp/test.png', 1)


# In[ ]:


print(img.shape)


# In[ ]:


#img_vtk = K.opa_vtk.numpy_to_vtk_image(np.ascontiguousarray(img[:,:,0]))
img_vtk = K.opa_vtk.numpy_to_vtk_image(np.ascontiguousarray(img.reshape((1,) + img.shape)))
tex = K.opa_vtk.reader2tex(img_vtk)
#tex = K.opa_vtk.jpeg2tex('/tmp/test.png')

actor= K.opa_vtk.TriangleActor(tex=tex).full_quad(Z.opa_struct.g_unit_box.poly(z_coord=1)).build()


# In[ ]:


main = vtkMain()
#$main.ren.AddActor(actor)
main.run()


# In[16]:



class Quad:

def __init__(self, box, depth):
  self.box = box
  self.depth = depth
  self._children = None
  
@property
def children(self):
  if self._children is None:
    self._children = []
    for qd in self.box.quadrants:
      self._children.append(Quad(qd, self.depth+1))
  return self._children

def __iter__(self):
  return iter(self.children)


@staticmethod
def Root():
  return Quad(Z.opa_struct.g_unit_box, 0)    
 


# In[17]:


class Consts:
  EARTH_ELLIPSOID = pymap3d.Ellipsoid('wgs84')
  MOON_ELLIPSOID = pymap3d.Ellipsoid('moon')
 
class TMSQuad:
  LMAX=85.05113
  MAX_DEPTH=20
  
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z
    self._children = None
    
  @property
  def children(self):
    if self._children is None:
      self._children = []
      if self.z +1 < TMSQuad.MAX_DEPTH:
        for i in range(4):
          self._children.append(TMSQuad(2*self.x+(i&1), 2*self.y+(i>>1), self.z+1))
    return self._children
  
  def __iter__(self):
    return iter(self.children)
  
  @property
  def box_latlng(self):
    bounds = mercantile.bounds(*self.xyz)
    box = Z.Box(low=(bounds.west, bounds.south), high=(bounds.east, bounds.north))
    return box
    
  @property
  def quad_ecef(self):
    p = self.box_latlng.poly()
    return Z.opa_struct.Quad(np.stack(pymap3d.geodetic2ecef(p[:,1], p[:,0], 0, ell=Consts.EARTH_ELLIPSOID), axis=-1) / 1e3) 
  
  @property
  def xyz(self):
    return self.x, self.y, self.z
  
  def tile(self, tg):
    return tg.get_tile(*self.xyz)
  
  @staticmethod
  def Root():
    return TMSQuad(0,0,0)    
  
  
def do_visit(obj, func):
  if func(obj):
    for x in obj:
      do_visit(x, func)


# In[18]:


class SimpleVisitor:
  def __init__(self, max_depth=2):
    self.max_depth = max_depth
    self.actors = []
    self.pts = []
    
  def __call__(self, obj):
    if obj.z < self.max_depth: return 1
    tx = obj.tile(tg)
    img = K.opa_vtk.read_img_from_buf(tx)
    actor = opa_vtk.TriangleActor(tex=K.opa_vtk.numpy2tex(img)).full_quad_yinv(obj.quad_ecef).build()
    self.actors.append(actor)
    self.pts.extend(obj.quad_ecef.pts)
    return 0
  
  def run_tms(self):
    root = TMSQuad.Root()
    do_visit(root, self)


# In[19]:


class CylindricalVisitor:
  def __init__(self, max_depth=2):
    self.max_depth = max_depth
    self.ta = opa_vtk.TriangleActor()
    self.real_box = Z.Box(low=(-np.pi, -np.pi/2), high= (np.pi, np.pi/2))
    self.pts = []
    
  def __call__(self, obj):
    if obj.depth < self.max_depth: return 1
    base_p = obj.box.poly()
    p = self.real_box.from_box_space(base_p)
    
    pmap = np.stack(pymap3d.geodetic2ecef(p[:,1], p[:,0], 0, ell=Consts.MOON_ELLIPSOID, deg=0), axis=-1) / 1e3
    self.pts.extend(pmap)
    self.ta.add_quad(pmap, base_p)
    return 0
  
  def run(self):
    root = Quad.Root()
    do_visit(root, self)
    
def create_moon_data():
  tex = K.opa_vtk.jpeg2tex('/home/benoit/Downloads/Moon_LRO_LOLA_global_LDEM_1024.jpg')
  width, height =  tex.GetInput().GetDimensions()[:2]
  cv = CylindricalVisitor(4)
  cv.run()
  actor = cv.ta.set_tex(tex).build()
  return cmisc.Attr(actor=actor, pts=cv.pts)
  


# In[20]:


def make_norm(a):
  return a / np.linalg.norm(a)
def make_orth_norm(a, b):
  a = np.array(a)
  b = make_norm(np.array(b))
  return make_norm(a - np.dot(a, b) * b)

def compute_angles(center, focal_point, up, pts):
  center, focal_point, up, pts = cmisc.to_numpy_args(center, focal_point, up, pts)
  pts =pts.reshape((-1, 3))
  
  dpt = pts - focal_point
  dfocal = focal_point - center
  dfocal_len = np.linalg.norm(dfocal)
  z = dfocal / dfocal_len
  
  y = make_orth_norm(up, z)
  x = make_norm(np.cross(y, z))
  pt_len = np.linalg.norm(dpt)
  dpc = pts - center
  dproj = np.dot(dpc, z)
  
  #xp = np.arcsin(np.dot(pts, x) / np.linalg.norm(pts-center, axis=1))
  #yp = np.arcsin(np.dot(dpt, y) / np.linalg.norm(pts-center, axis=1))
  
  xp = np.arctan2(np.dot(dpc, x), dproj)
  yp = np.arctan2(np.dot(dpc, y), dproj)
  
  res = np.stack((xp, yp), axis=-1)
  return res,y


def compute_cam_parameters(center, focal_point, up, pts, expand=0.05, aspect=None):
  angles,y = compute_angles(center, focal_point, up, pts)
  box = Z.Box.FromPoints(angles)
  print(box)
  box = box.center_on(np.zeros((2,)))
  print(box)
  box =box.expand(1 + expand)
  if aspect is not None:
    box = box.set_aspect(aspect)
  return cmisc.Attr(box=box, y=y)
  
#res =compute_cam_parameters((0, 0, 0), (0,0, 1), (0, 1, 0), [(1, 1, 1), (1,1,2)], expand=0.05, aspect=1.5)

print(np.min(pts_list, axis=0))
print(np.max(pts_list, axis=0))
print(focal_point)
params =compute_cam_parameters(mars_pos, focal_point, up, pts_list, expand=0.10, aspect=main.aspect)
print(params.box)
print(params.box.width)
print(params.box.height)
print(main.aspect)
print(params.box.aspect)


# In[ ]:


moon_data = create_moon_data()
main = vtkMain()
main.ren.AddActor(moon_data.actor)
main.run()
  


# In[ ]:



t = Time(datetime.datetime.utcnow()-datetime.timedelta(days=0))
with solar_system_ephemeris.set('builtin'):
  moon = get_body('moon', t, None) 
  mars = get_body('mars', t, None) 
  earth = get_body('earth', t, None) 
  moon = moon.transform_to('hcrs')
  mars = mars.transform_to('hcrs')
  earth = earth.transform_to('hcrs')
  moon_pos = moon.cartesian.xyz.to(u.km).value
  mars_pos = mars.cartesian.xyz.to(u.km).value
  earth_pos = earth.cartesian.xyz.to(u.km).value
  up = (0,0,1)
  
  main = vtkMain()
  
  moon_data = create_moon_data()
  moon_actor = moon_data.actor
  moon_actor.GetProperty().SetAmbient(1)
  moon_actor.GetProperty().SetDiffuse(0)
  moon_actor.SetPosition(*moon_pos)

  u = SimpleVisitor(4)
  u.run_tms()
  earth_assembly = K.opa_vtk.vtk.vtkAssembly()
  for actor in u.actors:
    earth_assembly.AddPart(actor)
    actor.GetProperty().SetAmbient(1)
    actor.GetProperty().SetDiffuse(0)
  earth_assembly.SetPosition(*earth_pos)
  
  
  pts_list = np.concatenate((np.array(u.pts) +earth_pos, np.array(moon_data.pts) + moon_pos))
  focal_point = np.mean(pts_list, axis=0)
  #pts_list = np.concatenate((np.array(moon_data.pts) + moon_pos, ))
  #focal_point = moon_pos
  #focal_point = earth_pos + (moon_pos - earth_pos)*0.59

  main.ren.AddActor(earth_assembly)
  main.ren.AddActor(moon_actor)
    
  
  params =compute_cam_parameters(mars_pos, focal_point, up, pts_list, expand=0.05, aspect=main.aspect)
  
  #main.cam.SetFocalPoint(*moon_pos)
  main.cam.SetClippingRange(1, 1e20)
  angle_y = Z.rad2deg(params.box.height)
  print('ANGLE > > ', angle_y, params.box.height)
  
  main.cam.SetPosition(*mars_pos)
  main.cam.SetFocalPoint(*focal_point)
  main.cam.SetViewAngle(angle_y)
  main.cam.SetViewUp(*params.y)

  
  main.run(reset=0)


# In[ ]:



start = '2019-07-24'
end = '2019-07-28'
bodies = [calcephpy.NaifId.MOON, '2019 OK']
starts = {}
starts[calcephpy.NaifId.MOON] = '2019-07-01'
body_center = calcephpy.NaifId.EARTH
step = '30m'
steps = {}
steps[calcephpy.NaifId.MOON] = '5h'
objs = []
for body in bodies:
  objs.append(Horizons(id=body,
                 location=f'@{body_center}',
                 epochs={'start':starts.get(body, start),
                         'stop':end,
                         'step':steps.get(body, step)},
                 id_type='majorbody'))
main = vtkMain()

for obj in objs:
  eph = obj.ephemerides()
  pts = eph_to_xyz(eph, U.km)
  dpts = np.diff(pts, axis=0)
  speed = np.linalg.norm(dpts, axis=1)
  dists = np.linalg.norm(pts, axis=1)
  imin=  np.argmin(dists)
  
  print(obj, eph['datetime_str'][imin], speed[imin], dists[imin-1:imin+2])
  print(min(speed), max(speed))
  actor = opa_vtk.create_line_actor(pts)
  main.ren.AddActor(actor)

assert 0
#  JPLHorizons instance "301"; location=@399, epochs={'start': '2019-07-01', 'stop': '2019-07-28', 'step': '5h'}, id_type=majorbody 2019-Jul-05 04:00 19468.309901514996 [363785.679 363745.979 363764.904]
#JPLHorizons instance "2019 OK"; location=@399, epochs={'start': '2019-07-24', 'stop': '2019-07-28', 'step': '1m'}, id_type=majorbody 2019-Jul-25 01:21 1472.0863310984244 [71370.228 71349.445 71358.746]
u = SimpleVisitor(4)
u.run_tms()
earth_assembly = K.opa_vtk.vtk.vtkAssembly()
for actor in u.actors:
  earth_assembly.AddPart(actor)
  actor.GetProperty().SetAmbient(1)
  actor.GetProperty().SetDiffuse(0)
earth_pos = (0,0,0)
earth_assembly.SetPosition(*earth_pos)
  
main.ren.AddActor(earth_assembly)
main.run()


# In[ ]:


print(eph)


# In[11]:


start = '2015-10-12'
end = '2015-10-13'
step = '1d'
steps = {}
starts={}
bodies = [calcephpy.NaifId.MOON, calcephpy.NaifId.EARTH]
body_center = 'LRO'
objs = []
for body in bodies:
  obj = Horizons(id=body,
                 location=f'@{body_center}',
                 epochs={'start':starts.get(body, start),
                         'stop':end,
                         'step':steps.get(body, step)},
                 id_type='majorbody')
  
  print(obj.ephemerides())
  print(obj.uri)
  objs.append(obj)
assert 0
main = vtkMain()

for obj in objs:
  eph = obj.ephemerides()
  pts = eph_to_xyz(eph, U.km)
  dpts = np.diff(pts, axis=0)
  speed = np.linalg.norm(dpts, axis=1)
  dists = np.linalg.norm(pts, axis=1)
  imin=  np.argmin(dists)
  
  print(obj, eph['datetime_str'][imin], speed[imin], dists[imin-1:imin+2])
  print(min(speed), max(speed))
  actor = opa_vtk.create_line_actor(pts)
  main.ren.AddActor(actor)

assert 0
#  JPLHorizons instance "301"; location=@399, epochs={'start': '2019-07-01', 'stop': '2019-07-28', 'step': '5h'}, id_type=majorbody 2019-Jul-05 04:00 19468.309901514996 [363785.679 363745.979 363764.904]
#JPLHorizons instance "2019 OK"; location=@399, epochs={'start': '2019-07-24', 'stop': '2019-07-28', 'step': '1m'}, id_type=majorbody 2019-Jul-25 01:21 1472.0863310984244 [71370.228 71349.445 71358.746]
u = SimpleVisitor(4)
u.run_tms()
earth_assembly = K.opa_vtk.vtk.vtkAssembly()
for actor in u.actors:
  earth_assembly.AddPart(actor)
  actor.GetProperty().SetAmbient(1)
  actor.GetProperty().SetDiffuse(0)
earth_pos = (0,0,0)
earth_assembly.SetPosition(*earth_pos)
  
main.ren.AddActor(earth_assembly)
main.run()


# In[14]:


print(obj.ephemerides().columns)


# In[ ]:


import time
import pymap3d
cam = main.ren.GetActiveCamera()
d = np.linalg.norm(cam.GetPosition())
def spherical_to_xyz(alpha, phi):
  return np.array((np.cos(alpha) * np.cos(phi), np.sin(alpha) * np.cos(phi), np.sin(phi)))
for t in np.linspace(0,1, 1000):
  pos = spherical_to_xyz(2*np.pi*3*t  / 2, 0*np.cos(2*np.pi*t) * np.pi /2 *0.4 ) * d
  cam.SetPosition(*pos)
  print(np.linalg.norm(pos))
  main.ren.ResetCameraClippingRange()
  main.ren_win.Render()
  time.sleep(0.01)


# In[ ]:


print(obj.ephemerides()['datetime_str'][0])


# In[ ]:


def mulnorm(m, v):
  v = list(v)+[1]
  v = m.MultiplyPoint(v)
  return np.array(v[:3])/v[-1]

z = make_norm(focal_point - mars_pos)
y = make_orth_norm((1,0,0), z)
m = main.cam.GetCompositeProjectionTransformMatrix(main.aspect, -1,1 )
print(Z.deg2rad(angle_y))
y = params.y
print(np.linalg.norm(np.cross(y,z)))
x = make_norm(np.cross(y,z))
for pt in pts_list:
  print(np.arcsin(np.dot(x, pt - mars_pos) / np.linalg.norm(pt - mars_pos)))
  print(mulnorm(m, pt))
  print()


# In[ ]:


app.exit_jup()

