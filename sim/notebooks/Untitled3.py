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
import numpy as np
from osgeo import gdal
import meshio
ctx = app.setup_jup('', parser_funcs=[render_params])
ctx.view_angle=  np.rad2deg(np.pi/8)

