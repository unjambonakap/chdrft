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

import geopandas

import skimage.transform
import numpy as np
from osgeo import gdal
import meshio
ctx = app.setup_jup('', parser_funcs=[render_params])
ctx.view_angle=  np.rad2deg(np.pi/8)


# In[2]:


moon_ell = pymap3d.Ellipsoid('moon')
moon_ell.semimajor_axis = moon_ell.semiminor_axis = 1737400 # not the default 1738000 !!
radius_sun = 696340 * U.km
u2m = 1000


# In[3]:


from pds.core.common import open_pds
from pds.core.parser import Parser

def stereo_XY2xyz(X, Y, r):
    X = X/2/r
    Y = Y/2/r
    
    a = X**2 + Y**2
    return np.array([2*X, 2*Y, -a+1])/(a+1) * r

def stereo_xyz2XY(x,y,z, r=1):
    x = x/r
    y = y/r
    z = z/r
    return np.array([x/(z+1), y/(z+1)])*r*2

def stereo_XYalt2ecef(X, Y, alt):
    lon, lat, _ = XY2ll(X,Y)
    return pymap3d.geodetic2ecef(lat, lon, alt, ell=moon_ell, deg=0)
    
def XY2ll(X,Y, xaxis_lon=np.pi/2):
    xyz=  stereo_XY2xyz(X,Y, r=moon_ell.semimajor_axis)
    lat, lon, alt= pymap3d.ecef2geodetic(*xyz, ell=moon_ell, deg=0)
    return lon+xaxis_lon, lat, alt

def ll2XY(lon, lat, xaxis_lon=np.pi/2):
    xyz = pymap3d.geodetic2ecef(lat, lon-xaxis_lon, ell=moon_ell, deg=0, alt=0)
    return stereo_xyz2XY(*xyz, r=moon_ell.semimajor_axis)


def compute_pix_box(e, hull):
    return (Z.Box.FromShapely(e.box.shapely.intersection(hull))* e.res.value).to_int_round()

class KDTreeInt:
    def __init__(self, box):
        self.box = box
        
    def children(self, box):
        xm = (box.xl+box.xh+1)//2
        ym = (box.yl+box.yh+1)//2
        res = []
        res.append(Z.Box(low=box.low, high=(xm, ym), is_int=1))
        res.append(Z.Box(low=(xm, box.yl), high=(box.xh, ym), is_int=1))
        res.append(Z.Box(low=(box.xl, ym), high=(xm, box.yh), is_int=1))
        res.append(Z.Box(low=(xm, ym), high=box.high, is_int=1))
        for x in res:
            if x.area > 0:
                yield x
        
    def visit(self, visitor, box=None):
        if box is None: box=self.box
        for x in self.children(box):
            if visitor(x):
                self.visit(visitor, x)
                
def get_data_for_entry(e, hull, dim):
        
    pts = []
    mapres=  e.res.value
    boundary_pix = Z.geo_ops.transform(lambda *x: np.array(x) * mapres, hull)
    def v1(box):
        print(box)
        if not box.shapely.intersects(boundary_pix): return False
        if boundary_pix.contains(box.shapely):

            pts.extend(box.get_grid_pos(stride=(1,1)))
        return box.area > 1
    
    
    
    full_box = (e.box * mapres).to_int_round()
    print(e.box, mapres, full_box)
    res = A(pix_box=compute_pix_box(e, hull), entry=e, full_box=full_box)
    #if 0:
    #    KDTreeInt(e.box * e.res.value).visit(v1)
    #    res.pix = [x-e.box.low for x in pts]
    #else:
    f = Z.os.path.join(data_dir, e.fname)
    res.img = None
    if Z.os.path.exists(f):
        res.img = read_gdal(f, dim=dim, scale_factor=0.5)
        res.vis_hull = backface_analysis(res)

    return res

def backface_analysis(e):
    points,xygrid = imgbox2grid(e.img, 100, regular_map)
    vecs = points - cam_moonspace
    dp = np.sum(vecs * points, axis=2)
    vis_faces = xygrid[:2,dp < 0].reshape((2,-1)).T
    vis_hull = Z.geometry.MultiPoint(vis_faces).convex_hull
    return vis_hull

def make_mesh_xyzgrid(xyz_grid):
    nx,ny,_=xyz_grid.shape
    pts = []
    ids = {}
    for ix in range(nx):
        for iy in range(ny):
            ids[(ix,iy)] = len(ids)
            pts.append(xyz_grid[ix,iy])
    faces = []
    for ix in range(nx-1):
        for iy in range(ny-1):
           a,b,c,d = ids[(ix,iy)], ids[(ix+1,iy)], ids[(ix+1,iy+1)], ids[(ix,iy+1)]
           faces.append([a,b,c])
           faces.append([a,c,d])
    m = meshio.Mesh(pts, [('triangle', faces)])
    return m

def img2mesh(img, fname, downscale, mapfunc):
    grid, _ = imgbox2grid(img , downscale, mapfunc)
    grid = grid / u2m
    t1 = make_mesh_xyzgrid(grid)
    t1.write(fname, binary=1)

def imgbox2grid(img, downscale, mapfunc):
    g= np.transpose(np.array(np.meshgrid(list(img.img_box.xr.range)[::downscale], list(img.img_box.yr.range)[::downscale], indexing='ij')), (1,2,0))
    xg,yg = np.transpose(img.img_box.change_rect_space(img.box, g), (2,0,1))
    nimg = skimage.transform.resize(img.img, xg.shape)
    grid = np.array([xg, yg, nimg])
    p = np.transpose(np.array(mapfunc(*grid)), [1,2,0])
    return p,grid

def read_gdal(fname, dim=None, scale_factor=1):
    ds = gdal.Open(fname)
    args = A()
    if dim is not None:
        args.buf_xsize = dim[0]
        args.buf_ysize = dim[1]
    img = np.array(ds.GetRasterBand(1).ReadAsArray(**args)).T
    ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()
    b=  Z.Box(low=(ulx, uly), size=(ds.RasterXSize * xres,ds.RasterYSize * yres))
    print(b)
    if b.xn < 0:
        b=  b.make_new(yr=b.yr, xr=(b.xh, b.xl))
        img = img[::-1]
    if b.yn < 0:
        b=  b.make_new(xr=b.xr, yr=(b.yh, b.yl))
        img = img[:,::-1]
        
    i = K.ImageData(img * scale_factor, box=b, yx=0)
    return i

class MeshBuilder:
    def __init__(self):
        self.poslist = []
        self.quadlist = []
        self.trlist = []
    def add_quad(self, pts):
        n = len(self.poslist)
        self.poslist.extend(pts)
        self.quadlist.append([n, n+1, n+3, n+2])
        
    def add_tri(self, pts):
        n = len(self.poslist)
        self.poslist.extend(pts)
        self.quadlist.append([n, n+1, n+2])
        
def parse_pds_val(x):
    val, unit= x.split(' ')
    val = float(val)
    return val * U.Unit(unit[1:-1])
    
def proc(parser, fname):
    labels = A.RecursiveImport(parser.parse(open(fname, 'r')))


    iproj =  labels.IMAGE_MAP_PROJECTION
    lat_max = parse_pds_val(iproj.MAXIMUM_LATITUDE)
    lat_min = parse_pds_val(iproj.MINIMUM_LATITUDE)
    lon_min = parse_pds_val(iproj.WESTERNMOST_LONGITUDE)
    lon_max = parse_pds_val(iproj.EASTERNMOST_LONGITUDE)
    map_res = parse_pds_val(iproj.MAP_RESOLUTION)

    latlon_box = Z.Box(xr=(lon_min.value, lon_max.value), yr=(lat_min.value, lat_max.value), is_int=1)
    if latlon_box.xl >= 180: latlon_box -= (360, 0)
    res = A(box=latlon_box, res=map_res, inter=latlon_box.shapely.intersection(hull),
            fname=labels.COMPRESSED_FILE.FILE_NAME[1:-1],
            labels=labels,
            ignore=1)
    if not latlon_box.shapely.intersects(hull): return res
    
    tb = []
    res.ignore =0 
    return res

def regular_map(x,y,z): return pymap3d.geodetic2ecef(lon=x, lat=y, alt=z, ell=moon_ell, deg=1)
def stereo_map(x,y,z): return stereo_XYalt2ecef(x,y,z)


# In[15]:


pymap3d.ecef2geodetic(-182.576508, 955.956482, 1435.179321, ell=moon_ell)


# In[24]:


a = K.ImageData(rx.subimg(Z.Box(center=(100.8126, 55.8577), dim=(0.1,0.1))))
a.plot()


# In[4]:


polar_mode = 0


# In[14]:


polar_mode = 1
f = '/home/benoit/Downloads/LRO_LOLA_DEM_NPolar875_10m.tif'
f = '/home/benoit/Downloads/LRO_LOLA_DEM_NPole75_30m.tif'
dim_polar = (15000, 15000)
i = read_gdal(f, dim=dim_polar, scale_factor=0.5)


# In[15]:


img2mesh(i, 'north.large.stl', 9, stereo_map)


# In[4]:


circles = []
for lat in np.arange(80, 90, step=1):
    xyl =ll2XY(np.linspace(0, 2*np.pi, 100), np.deg2rad(lat)).T
    circles.append(xyl)
oplt.plot(A(images=[i], lines=circles))


# In[4]:


from chdrft.sim.moon_sunrise import MoonSunrise, ActorType, norm_date, set_analysis_parameters
ctx.earth_depth=3
ctx.earth_tile_depth=2
ms = MoonSunrise(ctx)

set_analysis_parameters(ctx, 'lro_earthrise', large=False)
if 0: ctx.moon_model = 'pic'
ctx.moon_details = 8

dx = ms.get_data_at_time(ctx.t0)
moon = dx[ActorType.Moon]
sun = dx[ActorType.Light]

l = np.linalg.norm((moon.pos - sun.pos) / (radius_sun.to(U.km).value))
math.sin(1/l)

ms.build()
moon = ms.objs[ActorType.Moon]
cam = ms.objs[ActorType.Cam]


# In[5]:


ms.configure_cam(aspect=1, view_angle=10)
ms.configure_at(ctx.t0)
#ms.configure_cam(aspect=1, view_angle=90)


# In[6]:


tomoon = np.linalg.inv(moon._toworld)
cam_moonspace = MatHelper.mat_apply_nd(tomoon @ cam._toworld, [0,0,0], point=1) * u2m
print(cam_moonspace)
cam_geo =pymap3d.ecef2geodetic(*cam_moonspace, ell=moon_ell, deg=1)
cam_ll = cam_geo[:2][::-1]
cam_forward_pt = MatHelper.mat_apply_nd(tomoon, cam._toworld, [0,0,-1]) *u2m
front_pos = pymap3d.ecef2geodetic(*cam_forward_pt, ell=moon_ell, deg=1)
cam_dir_ll = (np.array(front_pos) -cam_geo)[:2][::-1]

moon2clip = cam.internal.local2clip @ np.linalg.inv(cam._toworld) @  moon._toworld
res = Z.opa_math.MatHelper.mat_apply_nd(moon2clip, np.array(moon.internal.points).T, point=1).T
vis_points = np.array(moon.internal.points)[np.max(np.abs(res), axis=1) <1].T*u2m
backp = np.array(pymap3d.ecef2geodetic(*vis_points, ell=moon_ell, deg=1)).T

backp_lonlat = backp[:,:2][:, ::-1]
backp_lonlat[:,0] += (backp_lonlat[:,0]<0) * 360
hull = Z.geometry.MultiPoint(backp_lonlat).convex_hull


# In[41]:


pts_XY = ll2XY(*np.deg2rad(backp_lonlat.T))
hull_XY = Z.geometry.MultiPoint(pts_XY.T).convex_hull
print(hull_XY.bounds)
oplt.plot(A(images=[i], lines=[hull_XY], points=[ll2XY(*np.deg2rad(cam_ll))], points_color='r'))


# In[7]:


parser = Parser()
 
data_dir ='/home/benoit/data/moon/lola/'
lst = list(cmisc.filter_glob_list(cmisc.list_files_rec(data_dir), '*512*.LBL'))
#http://imbrium.mit.edu/DATA/SLDEM2015/TILES/JP2/
tb=  []
for x in lst:
    tb.append(proc(parser, x))
for x in tb:
    if '60N_045_090' not in x.fname and '135.JP2' not in x.fname:
        x.ignore = True
    if not x.ignore: print(x.fname)
tdim = None
#tdim = (3000, 2000)
data = [x for x in tb if not x.ignore]
data = [get_data_for_entry(x, hull, tdim) for x in data]
data = [x for x in data if x.img is not None]


# In[12]:


import cv2
img = K.ImageData(cv2.imread('../Moon_LRO_LOLA_global_LDEM_1024.jpg')[::-1], box=Z.Box(yr=(-90, 90), xr=(-180, 180)))
hulls = [A(polyline=hull, color='g')]
for x in data: hulls.append(A(polyline=x.vis_hull, color='r'))
    
oplt.plot([A(images=[img], points=[cam_ll, cam_ll + make_norm(cam_dir_ll)*10], lines=[x.box for x in tb] + hulls, misc=[A(text=x.fname, pos=x.box.mid) for x in tb], points_color=('g', 'r'))])


# In[14]:


K.ImageData(rx.img).plot()


# In[ ]:





# In[8]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
from chdrft.display.render import ImageComposer
get_ipython().run_line_magic('aimport', 'chdrft.display.render')

import functools
buf_val = 1
vis_hull = functools.reduce(lambda a,b: a.union(b), [x.vis_hull.buffer(buf_val) for x in data])
fhull = hull.intersection(vis_hull)
ic = ImageComposer([x.img for x in data])
rx = ic.render_box_fixed_dim(Z.Box.FromShapely(fhull), yx=0)


# In[36]:


img2mesh(rx, 't1.large.stl', 3, regular_map)


# In[ ]:


import pymeshlab as ml
ms = ml.MeshSet()
ms.load_new_mesh('./t1.large.stl')
m = ms.current_mesh()
ms.simplification_quadric_edge_collapse_decimation(targetperc=0.1, preserveboundary=True)
ms.save_current_mesh('./test1.simpl.stl')


# In[13]:


mp = [[(1112, 535),
 (978, 642),
 (1524, 615),
(1288, 797),
],
[(550, 692),
 (371, 824.5),
 (1078, 814),
 (764, 1038),
]]
mp = np.array(mp)
from chdrft.dsp.image import read_tiff
H,_=cv2.estimateAffinePartial2D(mp[0], mp[1])
a = cv2.imread('../../../../writeup/lro/data/lrox_render_earthrise.png')
b = cv2.imread('../../../../writeup/lro/data/lro_earthrise_real.jpg')
res = cv2.warpAffine(a,H, b.shape[:2], cv2.INTER_LINEAR)
K.ImageData(res).plot()


# In[10]:



mp = [[(276,648),
      (544, 647),
      (446, 592),
       (193, 672),
      ],
[(1455,1375),
(3625, 1352),
 (2637, 784),
 (672, 1730),
 
]]
mp = np.array(mp)
from chdrft.dsp.image import read_tiff
import cv2
H,residuals=cv2.estimateAffinePartial2D(mp[0], mp[1])
a = cv2.imread('../../../../writeup/lro/data/earthmoon_square1.png')
b = cv2.imread('../../../../writeup/lro/data/lro2_render.png')
res = cv2.warpAffine(a,H, b.shape[:2][::-1], cv2.INTER_LINEAR)
K.ImageData(res).plot()


# In[ ]:





# In[11]:


K.ImageData(b).plot()


# In[127]:


H


# In[14]:


from chdrft.dsp.image import read_tiff, save_image
save_image('/tmp/res.png', res)


# In[123]:


for i, x in enumerate(list(cmisc.filter_glob_list(cmisc.list_files_rec('/home/benoit/programmation/vid_earthrise/'), '*.png'))):
    
    if i==0: continue
    a = cv2.imread(x)
    a = a.transpose((1,0,2))[::-1]
    cv2.imwrite(x, a)
    

