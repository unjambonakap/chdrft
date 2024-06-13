#!/usr/bin/env python

from chdrft.config import env
import logging

#import chdrft.utils.lazy import do_lazy_import
#
#do_lazy_import(__file__)

import math
from asq.initiators import query as asq_query

from collections import defaultdict, deque
from contextlib import ExitStack
from enum import Enum
from math import nan
from math import pi
from matplotlib import cm
from pprint import pprint
from queue import Queue
try:
  from queue import SimpleQueue
except: pass
import base64
import binascii
import codecs
import csv
import ctypes
import curses.ascii
import glob
import glog
import gmpy2
import heapq
import io
import itertools
import math
import multiprocessing
import networkx as nx
import numpy as np
import os.path
import pandas as pd
import pickle
import random
import re
import requests
import shutil
import socket
import struct
import subprocess as sp
import tempfile
import threading
import time
import traceback as tb
import wrapt
import shapely.ops as geo_ops
import shapely.geometry as geometry
import functools
import typing





from chdrft.cmds import CmdsList
from chdrft.dsp.cv_utils import to_grayscale
from chdrft.main import app
from chdrft.struct.base import Intervals, get_primitive_ranges, Range1D, Range2D, Box, g_unit_box, g_one_box, GenBox
from chdrft.tools.xxd import xxd
from chdrft.utils.arg_gram import LazyConf
from chdrft.utils.cache import Cachable
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.colors import ColorPool, ColorMapper
from chdrft.utils.fmt import Format
from chdrft.utils.geo import Line, smallest_circle
from chdrft.utils.geo import to_shapely
from chdrft.utils.math import MatHelper, deg2rad, rad2deg
from chdrft.utils.misc import Attributize, failsafe
from chdrft.utils.opa_string import FuzzyMatcher, lcs, min_substr_diff
from chdrft.utils.path import FileFormatHelper
from chdrft.utils.swig import swig, swig_unsafe
import chdrft.conv.utils as conv_utils
import chdrft.crypto.common as ccrypto
import chdrft.display.utils as dsp_utils
import chdrft.graph.base as graph_base
import chdrft.math.sampling as opa_sampling
import chdrft.struct.base as opa_struct
import chdrft.utils.cache as opa_cache
import chdrft.utils.geo as geo_utils
import chdrft.utils.math as opa_math
import chdrft.utils.misc as cmisc




