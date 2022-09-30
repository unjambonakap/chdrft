#!/usr/bin/env python
# coding: utf-8

# In[2]:



from ipyleaflet import Map, Marker, Popup
import ipyleaflet
from ipywidgets import HTML

center = (52.204793, 360.121558)

m = Map(center=center, zoom=1)
m


# In[3]:



def create_marker(pos, text):
  popup = Popup(
      location=pos,
      child=HTML(text),
      close_button=False,
      auto_close=False,
      close_on_escape_key=False
  )
  m.add_layer(popup);


# In[4]:


gl = Nominatim(user_agent='VGYV7gGlNoWapA==')
o0 = gl.geocode('Hat Creek Radio Observatory')
o1 = gl.geocode('San Francisco')


# In[19]:


create_marker((o0.latitude, o0.longitude), 'HCRO')


# In[2]:


import pymap3d
p0 = pymap3d.geodetic2ecef(o0.latitude, o0.longitude, 0)
p1 = pymap3d.geodetic2ecef(o1.latitude, o1.longitude, 0)
print(p0, p1)
np.linalg.norm(np.array(p0)-p1)


# In[5]:


get_ipython().run_line_magic('gui', 'qt4')
from PyQt4 import QtGui, QtCore, QtSvg
from chdrft.config.env import g_env
g_env.set_qt5(0)
init_jupyter()
import georinex as gr
import ppp_tools.gpstime
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
tg = TileGetter()

import math
def ecef_to_latlng(x,y,z):
  lng = math.atan2(y,x)
  lat = math.atan2(z, np.linalg.norm((x,y)))
  return Z.rad2deg(np.array((lat, lng)))
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS
def eph_to_xyz(eph, unit=U.AU):
  a = SkyCoord(eph['RA'], eph['DEC'], distance=eph['delta'], frame='icrs')
  return a.cartesian.xyz.to(unit).value.T


# In[ ]:


app.exit_jup()


# In[5]:


import datetime
datetime.datetime(2010, 10, 8) - datetime.datetime(2010, 1, 1)


# In[4]:


nav = gr.load('../../../dsp/gps/brdc3640.15n')
nav = gr.load('/home/benoit/data/astro/gps/brdc2360.19n')
nav = gr.load('/home/benoit/data/astro/gps/brdc2810.10n')


# In[6]:


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

  def get_speed(self, t, seconds=1):
    dt = datetime.timedelta(seconds=seconds)
    return (self.get_pos(t+dt) - self.get_pos(t))/seconds
  
  def get_pos(self, t):
    
    dt = t -  self.utc
    dt_sec =dt.total_seconds()
    
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
    return np.array([X,Y,Z])


# In[64]:


def analyse_sat(sat, t ):
  if isinstance(sat, int):
    sat = 'G%02d'%sat
  tmp = nav.sel(sv=sat)
  df = tmp.to_dataframe().dropna()
  last = df.iloc[-1]
  a = Ephemerid(last)

  xyz= np.array(a.get_pos(t))
  res = np.array(pymap3d.ecef2geodetic(*xyz))
  return cmisc.Attr(xyz=xyz, geodetic=res, v=a.get_speed(t, seconds=1e-2))


# In[8]:


t = datetime.datetime(2010, 10, 8, 13, 23)
import pytz
pdt=pytz.timezone('US/Pacific')
t_pdt = pdt.localize(t)
t_utc =  t_pdt.astimezone(pytz.utc)
t_utc = t_utc.replace(tzinfo=None)


# In[9]:


analyse_sat(1, t_utc)


# In[67]:


def compute_doppler(v, freq):
  return -v * freq / constants.c
def compute_doppler2(pos, v, freq):
  pos_norm = Z.geo_utils.make_norm(pos)
  return -np.dot(pos_norm, v) * freq / constants.c


# In[24]:


obs_ecef = pymap3d.geodetic2ecef(o0.latitude, o0.longitude, 986)
prev = 0
for dmin in range(-10, 10):
  ux = analyse_sat(1, t_utc + datetime.timedelta(minutes=dmin))
  dt = 10
  ux2 = analyse_sat(1, t_utc + datetime.timedelta(minutes=dmin, seconds=dt))
  
  vv = np.dot(Z.geo_utils.make_norm(ux.xyz- obs_ecef), ux.v)
  acc = (ux2.v - ux.v)/dt
  vv = np.linalg.norm(ux.v)
  kGPS_CENTER_FREQ = 1575420000
  print()
  doppler = -vv * kGPS_CENTER_FREQ / constants.c
  
  pos_above = np.array(pymap3d.geodetic2ecef(o0.latitude, o0.longitude, ux.geodetic[2])) - obs_ecef
  dt = 10
  x = np.linalg.norm(pos_above)
  v_max = vv**2 / x
  
  
  d0 = compute_doppler(2*v_max, kGPS_CENTER_FREQ)
  print('aaa', d0 / dt, v_max)
  
  print(doppler, (doppler-prev) / 60)
  prev=doppler
  print(-np.linalg.norm(acc) * kGPS_CENTER_FREQ / constants.c)


# In[68]:


tb = []
for dhours in np.linspace(0, 100, 200):
  dt = 1
  ux0 = analyse_sat(1, t_utc + datetime.timedelta(hours=dhours))
  ux1 = analyse_sat(1, t_utc + datetime.timedelta(hours=dhours, seconds=dt))
  ux2 = analyse_sat(1, t_utc + datetime.timedelta(hours=dhours, seconds=2*dt))
  p0 = ux0.xyz - obs_ecef
  p1 = ux1.xyz - obs_ecef
  p2 = ux2.xyz - obs_ecef
  v0 = (p1-p0)/dt
  v1 = (p2-p1)/dt
  d0 = compute_doppler2(p0, v0, kGPS_CENTER_FREQ)
  d1 = compute_doppler2(p1, v1, kGPS_CENTER_FREQ)
  tb.append(abs((d0-d1)/dt))


# In[69]:


print(max(tb))


# In[46]:


import astropy.constants as consts
import astropy.units as u
m_earth = consts.M_earth
G = consts.G


# In[63]:


def get_orbit_duration(r): return 2*np.pi / (G * m_earth / r**3)**(0.5)
print(get_orbit_duration((20000+7000)*u.km).to(u.hour))


# In[42]:


u.m(123)


# In[69]:


for val in nav['sv'].values:
  pos = analyse_sat(val, t_utc)
  create_marker(pos[:2], val.geodetic)


# In[23]:


import folium
m = folium.Map(
    zoom_start=2,
  location=[0,0]
)


# In[21]:


x = folium.map.Marker(location=(100, 0), tooltip=folium.map.Tooltip('test', permanent=True)).add_to(m)


# In[ ]:





# In[ ]:


gr.keplerian2ecef(tmp)


# In[ ]:


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


# In[ ]:


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


# In[ ]:





# In[1]:


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

