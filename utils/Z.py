#!/usr/bin/env python


from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
from chdrft.struct.base import Intervals, get_primitive_ranges, Range1D
import glob
import re
import os.path
from chdrft.interactive.base import create_kernel
from collections import defaultdict, deque
from chdrft.tools.xxd import xxd
import binascii
import struct
import codecs
import base64
import gmpy2
from chdrft.tube.connection import Connection
from chdrft.tube.process import Process
from chdrft.tube.serial import SerialFromProcess, Serial
from chdrft.tube.file_like import FileLike, FileTube
import requests
import traceback as tb
import random
import time
from chdrft.utils.fmt import Format
from chdrft.utils.path import FileFormatHelper
from chdrft.utils.arg_gram import LazyConf
import tempfile
import shutil
import socket
import csv
import pickle
import numpy as np
from pprint import pprint
import pandas as pd
import io
from asq.initiators import query as asq_query
import curses.ascii
from contextlib import ExitStack
from chdrft.emu.elf import ElfUtils
import Crypto.Hash.MD5 as MD5
import itertools
import Crypto.Cipher.AES as AES
from Crypto.Util.Padding  import unpad, pad
import chdrft.crypto.common as ccrypto
from chdrft.gen.types import  g_types_helper
import subprocess as sp
from scipy import fftpack
from scipy import optimize
from scipy import signal
from scipy import cluster
from scipy import stats
from scipy.stats.mstats import mquantiles
from chdrft.utils.swig import swig, swig_unsafe
from chdrft.display.utils import DataOp, DataFile, DataSet, DynamicDataset
import chdrft.display.utils as dsp_utils
import matplotlib.pyplot as plt
from enum import Enum
import chdrft.conv.utils as conv_utils
import math
from math import pi
from math import nan
import networkx as nx
from chdrft.utils.colors import ColorPool
from chdrft.utils.opa_string import FuzzyMatcher, lcs, min_substr_diff
from scipy.cluster.vq import kmeans, whiten
import threading
from chdrft.utils.cache import Cachable
import ctypes
import wrapt
from chdrft.dbg.gdbdebugger import GdbDebugger, launch_gdb
import multiprocessing
from chdrft.emu.trace import Tracer, Display, WatchedMem, WatchedRegs
from queue import Queue

np.set_printoptions(edgeitems=30, linewidth=120,
    formatter=dict(float=lambda x: "%.03f" % x))
