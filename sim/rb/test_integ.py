#!/usr/bin/env python

import chdrft.config.env
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from chdrft.utils.types import *
from pydantic.v1 import Field
from chdrft.utils.path import FileFormatHelper
from chdrft.utils.fmt import Format
from chdrft.sim.rb.rb_gen import *
from chdrft.sim.rb.ctrl import *
from chdrft.sim.rb.scenes import *
import pygmo as pg

a = Vec3.Z().exp_rot()
a.get_axis_angle()


p1 = Transform.From(pos=Vec3.Vec([1,2,3]), rot=R.from_rotvec([0,0,np.pi/2]))
v_p1 = p1.tsf_rot.inv @ SpatialVector.Vector(np.array([1,0,0, 0,0,1], dtype=np.float64))
v_p0 = p1 @ v_p1

m1_p1 = p1.to_sv()
m1_p0 =p1 @ m1_p1

dt = 1e-1
m1_p0 += v_p0 * dt

nm1_p1 = m1_p1 + v_p1 * dt
nm2_p1 = p1.inv @ m1_p0

pa = p1 @ (v_p1*dt).m4_tsf
pa = pa.to_sv()

print(nm1_p1)
print(nm2_p1)
print(pa)






def delta_map(v : SpatialVector, dt: float, com: Vec3) -> Transform:
  v = v.around(com)
  dp =v.v  * dt
  dr = (v.w * dt).exp_rot()
  return Transform.From(pos=dp, rot=dr)



def spos2tsf(p: SpatialVector) -> Transform:
  return Transform.From(pos=p.v, rot=p.w.exp_rot())
