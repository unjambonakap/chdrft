#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.types import *
import re
import itertools
from enum import Enum
from chdrft.utils.path import FileFormatHelper

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--outfile')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def id2num(idx):
  if idx is None: return None
  num = idx + 1
  if idx >= 10:
    num += 2
  if idx >= 26:
    num += 2
  return num


def num2pos(num):
  if num is None: return None
  x = (num - 1) % 2
  y = (num - 1) // 2
  return (x, y)


class PinTypes(Enum):
  GPIO = 'gpio'
  ALIM = 'alim'
  LED = 'led'

class PortBase:

  def eq(self, other):
    return (self.typ, self.vecpos) == (other.typ, other.vecpos)
  def norm_arg(self, arg):
    if isinstance(arg, slice):
      vrange = list(range(len(self.vecpos)))
      return vrange[arg]
    if arg is not None and not is_num(arg): arg = np.array(arg, dtype=int)

    return arg

  def __init__(self, typ, vecpos=None):
    self.typ = typ
    self.vecpos = self.norm_arg(vecpos)

  def subport_impl(self, i):
    return PortBase(self.typ, self.vecpos[i])


  def subport(self, iv): return self.subport_impl(self.norm_arg(iv))
  def __getitem__(self, i): return self.subport(i)

  def is_vec(self): return is_list(self.vecpos)


  @cmisc.yield_wrapper
  def normalize(self):
    if self.is_vec():
      for i in range(len(self.vecpos)): yield self.subport(i)
    else: yield self

class Pin(PortBase):

  def __init__(self, group=None, num=None, pin=None, typ=None, idx=None, used=0):
    super().__init__(typ, idx)
    self.typ = typ
    self.group = group
    self.pin = pin
    if num is None:
      num = id2num(idx)
    self.num = num
    self.pos = num2pos(num)
    self.used = used
  def __str__(self):
    return f'{self.typ} {self.group} {self.pin} {self.num} {self.pos}'
  def __repr__(self):
    return str(self)


def build_pins():

  pins = []
  for gpio_str, pin_str in pin_list:
    m = re.match('GPIO_(?P<group>\d)\[(?P<idx>\d+)\]', gpio_str)
    pin = pin_str.split('_')[1]

    pins.append(Pin(int(m['group']), pin=pin, idx=int(m['idx']), typ=PinTypes.GPIO))
  for grp in (0, 1):
    pins.append(Pin(grp, pin='5V', num=11, typ=PinTypes.ALIM, used=1))
    pins.append(Pin(grp, pin='0V', num=12, typ=PinTypes.ALIM, used=1))
    pins.append(Pin(grp, pin='3.3V', num=29, typ=PinTypes.ALIM, used=1))
    pins.append(Pin(grp, pin='0V', num=30, typ=PinTypes.ALIM, used=1))

  for i, v in enumerate(led_pins):
    pin = v.split('_')[1]
    pins.append(Pin(typ=PinTypes.LED, pin=pin, num=i))

  return pins


class AssignmentRequest:

  def __init__(self, name, ids=None, group=None, x=None, typ=None, label=None):
    self.name = name
    self.label = label
    self.typ = typ
    self.is_vec = ids is not None
    if ids is None: ids = [0]
    else: ids = list(ids)

    self.group = group
    self.x = x
    self.ids = ids

  def get_name(self, i):
    if self.is_vec: return f'{self.name}[{i}]'
    return self.name


class PinDb:

  def __init__(self):
    self.pins = build_pins()
    self.pinq = cmisc.asq_gen(self.pins)

    self.mp = [dict(), dict()]
    for x in self.pins:
      if x.group is None: continue
      self.mp[x.group][x.pos] = x

  def get_str(self, pin, name):
    return f'''
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to {name}
set_location_assignment PIN_{pin.pin} -to {name}
'''

  def create_assignment(self, req):
    if req.group is not None: return self.create_assignment_gpio(req)
    avail = self.pinq().where(lambda x: x.typ == req.typ and x.used == 0).order_by(lambda x: x.num).to_list()
    res = []
    for i, id in enumerate(req.ids):
      res.append(cmisc.Attr(pin=avail[i], id=id, req=req, port=PortBase(req.label, i)))
    return res



  def create_assignment_gpio(self, req):
    res = []

    xr = req.x
    if xr is None: xr = 0
    for y in itertools.count():
      if not self.mp[req.group][(xr, y)].used:
        break

    for i, id in enumerate(req.ids):
      for ny in itertools.count(y):
        pin = self.mp[req.group][(xr, ny)]
        if not pin.used:
          break
      y = ny
      pin.used = 1
      res.append(cmisc.Attr(pin=pin, id=id, req=req, port=PortBase(req.label, i)))
      if req.x is None: xr ^=1

    return res

  def create_assignments(self, reqs):
    tb = []
    for req in reqs:
      tb.extend(self.create_assignment(req))

    groups = cmisc.Attr()
    for req in reqs:
      if req.label is None: continue
      groups[req.label] = PortBase(req.label, range(len(req.ids)))
    return cmisc.Attr(asst=tb , groups=groups, db=self)



  def build_str(self, assignments):
    res = []
    for x in assignments:
      fname= x.req.get_name(x.id)
      res.append(self.get_str(x.pin, fname))
    return '\n'.join(res)


# yapf: disable
pin_list = [
['GPIO_1[0]', 'PIN_Y15',],
['GPIO_1[1]', 'PIN_AG28',],
['GPIO_1[2]', 'PIN_AA15',],
['GPIO_1[3]', 'PIN_AH27',],
['GPIO_1[4]', 'PIN_AG26',],
['GPIO_1[5]', 'PIN_AH24',],
['GPIO_1[6]', 'PIN_AF23',],
['GPIO_1[7]', 'PIN_AE22',],
['GPIO_1[8]', 'PIN_AF21',],
['GPIO_1[9]', 'PIN_AG20',],
['GPIO_1[10]','PIN_AG19',],
['GPIO_1[11]','PIN_AF20',],
['GPIO_1[12]','PIN_AC23',],
['GPIO_1[13]','PIN_AG18',],
['GPIO_1[14]','PIN_AH26',],
['GPIO_1[15]','PIN_AA19',],
['GPIO_1[16]','PIN_AG24',],
['GPIO_1[17]','PIN_AF25',],
['GPIO_1[18]','PIN_AH23',],
['GPIO_1[19]','PIN_AG23',],
['GPIO_1[20]','PIN_AE19',],
['GPIO_1[21]','PIN_AF18',],
['GPIO_1[22]','PIN_AD19',],
['GPIO_1[23]','PIN_AE20',],
['GPIO_1[24]','PIN_AE24',],
['GPIO_1[25]','PIN_AD20',],
['GPIO_1[26]','PIN_AF22',],
['GPIO_1[27]','PIN_AH22',],
['GPIO_1[28]','PIN_AH19',],
['GPIO_1[29]','PIN_AH21',],
['GPIO_1[30]','PIN_AG21',],
['GPIO_1[31]','PIN_AH18',],
['GPIO_1[32]','PIN_AD23',],
['GPIO_1[33]','PIN_AE23',],
['GPIO_1[34]','PIN_AA18',],
['GPIO_1[35]','PIN_AC22',],

['GPIO_0[0]', 'PIN_V12', ],
['GPIO_0[1]', 'PIN_AF7', ],
['GPIO_0[2]', 'PIN_W12', ],
['GPIO_0[3]', 'PIN_AF8', ],
['GPIO_0[4]', 'PIN_Y8',  ],
['GPIO_0[5]', 'PIN_AB4', ],
['GPIO_0[6]', 'PIN_W8',  ],
['GPIO_0[7]', 'PIN_Y4',  ],
['GPIO_0[8]', 'PIN_Y5',  ],
['GPIO_0[9]', 'PIN_U11', ],
['GPIO_0[10]','PIN_T8',  ],
['GPIO_0[11]','PIN_T12', ],
['GPIO_0[12]','PIN_AH5', ],
['GPIO_0[13]','PIN_AH6', ],
['GPIO_0[14]','PIN_AH4', ],
['GPIO_0[15]','PIN_AG5', ],
['GPIO_0[16]','PIN_AH3', ],
['GPIO_0[17]','PIN_AH2', ],
['GPIO_0[18]','PIN_AF4', ],
['GPIO_0[19]','PIN_AG6', ],
['GPIO_0[20]','PIN_AF5', ],
['GPIO_0[21]','PIN_AE4', ],
['GPIO_0[22]','PIN_T13', ],
['GPIO_0[23]','PIN_T11', ],
['GPIO_0[24]','PIN_AE7', ],
['GPIO_0[25]','PIN_AF6', ],
['GPIO_0[26]','PIN_AF9', ],
['GPIO_0[27]','PIN_AE8', ],
['GPIO_0[28]','PIN_AD10',],
['GPIO_0[29]','PIN_AE9', ],
['GPIO_0[30]','PIN_AD11',],
['GPIO_0[31]','PIN_AF10',],
['GPIO_0[32]','PIN_AD12',],
['GPIO_0[33]','PIN_AE11',],
['GPIO_0[34]','PIN_AF11',],
['GPIO_0[35]','PIN_AE12',],
]
# yapf: enable
led_pins = list(map(str.strip, '''
PIN_W15
PIN_AA24
PIN_V16
PIN_V15
PIN_AF26
PIN_AE26
PIN_Y16
PIN_AA23
'''.strip().splitlines()))


def test(ctx):
  db = PinDb()
  reqs = [
      AssignmentRequest('toolbox_0_mic_conduit_export_scl', ids=range(8), group=0, x=0),
      AssignmentRequest('toolbox_0_mic_conduit_export_sda', ids=range(8), group=0, x=1),
      AssignmentRequest(kPortName, ids=range(8,32), group=1, label=kGpioLabel),
      AssignmentRequest(kPortName, ids=range(8), typ=PinTypes.LED, label=kLedLabel),
  ]
  res = db.create_assignments(reqs)
  print(db.build_str(res))
  if ctx.outfile: FileFormatHelper.Write(ctx.outfile, res)

  for i in res:
    print(i.name, i.pin.group, i.pin.pos, i.pin.num, i.pin.typ)

def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
