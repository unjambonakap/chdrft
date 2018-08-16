#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, BitOps
import glog
import time
import struct

def decode_data(x):
   tmp = struct.unpack('>I', x)[0]
   sgn = tmp >> 31
   val = (tmp >> 18) & BitOps.mask(13)
   if sgn == 1:
     val = (1 << 13) - val
   fault = tmp >> 16 & 1
   val = val * BitOps.bit2sign(sgn)

   ref_junc = tmp >> 4 & BitOps.mask(11)
   if tmp >> 15 & 1: ref_junc = (1 << 11) - ref_junc;
   ref_junc /= 2**4
   return val / 4, ref_junc
