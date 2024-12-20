#!/usr/bin/env python
import chdrft.utils.Z as Z

from astropy import units as U
import math
from astropy.coordinates import SkyCoord

def ecef_to_latlng(x,y,z):
  lng = math.atan2(y,x)
  lat = math.atan2(z, np.linalg.norm((x,y)))
  return Z.rad2deg(np.array((lat, lng)))
def eph_to_xyz(eph, unit=U.AU):
  a = SkyCoord(eph['RA'], eph['DEC'], distance=eph['delta'], frame='icrs')
  return a.cartesian.xyz.to(unit).value.T
