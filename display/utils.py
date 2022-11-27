#!/usr/bin/env python

from IPython.utils.frame import extract_module_locals
from asq.initiators import query as asq_query
from enum import Enum
from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE, USE_PYQT5
from rx import operators as ops
from scipy import signal
from scipy.stats.mstats import mquantiles
import cv2
import glog
import glog
import itertools
import math
import math, sys, os
import numpy as np
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import scipy.ndimage as ndimage
import sys
import tempfile

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.struct.base import Box, Range2D, g_unit_box
from chdrft.struct.base import Range1D, Range2D, Intervals, ListTools, Box
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.colors import ColorPool
from chdrft.utils.fmt import Format
from chdrft.utils.fmt import Format
from chdrft.utils.misc import Attributize
from chdrft.utils.misc import Attributize as A
from chdrft.utils.misc import to_list, Attributize, proc_path
from chdrft.utils.path import FileFormatHelper
from chdrft.utils.swig import swig
from chdrft.utils.types import *

import chdrft.display.base
import chdrft.dsp.dataop as DataOp
import chdrft.utils.misc as cmisc
import chdrft.utils.misc as cmisc

