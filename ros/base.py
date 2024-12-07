#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import A
import glog
import numpy as np
from pydantic import Field, BaseModel
from chdrft.utils.path import FileFormatHelper
import subprocess as sp
import pydantic
import pprint
import rclpy
import rclpy.node, rclpy.task
from chdrft.utils.othreading import ThreadGroup
import time
import shutil

global flags, cache
flags = None
cache = None
from plotjuggler_msgs.msg import DataPoint
from rclpy_message_converter import message_converter
import pymavlink.dialects.v20.ardupilotmega as mavlink_msgs


class CustomRosMessage(cmisc.PatchedModel):
  ros_typename: str
  mavlink: bool
  typ: object

  @cmisc.cached_property
  def ros_typ(self):
    return message_converter.get_message(self.ros_fulltypename)

  @property
  def ros_fulltypename(self):
    return f'chdrft_ros_messages/msg/{self.ros_typename}'

  @classmethod
  def Make(cls, typ):

    name = typ.__name__
    mavlink = issubclass(typ, mavlink_msgs.MAVLink_message)

    if mavlink:
      name = f'Mavlink_{typ.msgname}'
      name = name.lower().title().replace('_', '')  # needs to be camelcase, no underscore

    return CustomRosMessage(
        typ=typ,
        mavlink=mavlink,
        ros_typename=name,
    )

  def py2ros(self, pyo):
    if self.mavlink:
      d = {k: getattr(pyo, k) for k in type(pyo).fieldnames}
    else:
      d = pyo.dict()
    return message_converter.convert_dictionary_to_ros_message(self.ros_typ, d)


class RosMessageHelper(cmisc.PatchedModel):
  py2ros: dict[type, CustomRosMessage] = cmisc.pyd_f(dict)

  def __post_init__(self):
    for x in message_db_list():
      self.register(x)

  def register(self, pyd_type):
    self.py2ros[pyd_type] = CustomRosMessage.Make(pyd_type)

  def get_ros_typename(self, name=None, type=None):
    if type is not None:
      name = type.__name__

    return f'chdrft_ros_messages/msg/{name}'

  def generate_mavlink_messages(self, typs: list[CustomRosMessage]) -> dict:
    msgs = {}
    for e in typs:
      content = []
      for name, type in zip(e.typ.fieldnames, e.typ.fieldtypes):
        assert type.endswith('_t')
        content.append(f'{type[:-2]} {name}')

      msgs[e.ros_typename] = content
    return msgs

  def generate_messages(self, typs: list[CustomRosMessage]) -> dict:
    typ2ref, defs = pydantic.json_schema.models_json_schema(
        [(x.typ, 'serialization') for x in typs]
    )

    defs = defs['$defs']

    context = defs
    msgs = {}

    def gen_msg(entry, name=None):

      entries = []
      if '$ref' in entry:
        target = cmisc.os.path.basename(entry['$ref'])
        return gen_msg(defs[target], name=target)
      typ = entry['type']
      mp = {'number': 'float64', 'integer': 'int32'}
      if name is None:
        return mp[typ]

      if name not in msgs:
        msgs[name] = None

        if 'enum' in entry:

          if 1:
            # enum handling is poor
            del msgs[name]
            return mp[typ]

          for v in entry['enum']:
            entries.append(f'int8 VAL_{v}={v}')
          entries.append(f'int8 data')

        else:
          for prop, v in entry['properties'].items():
            typ = gen_msg(v)
            entries.append(f'{typ} {prop}')

        msgs[name] = entries

      return name

    for (typ, ser), entry in typ2ref.items():
      gen_msg(entry, typ.__name__)

    return msgs

  def build_ros_package(self):

    msgs = {}
    msgs |= self.generate_messages([v for v in self.py2ros.values() if not v.mavlink])
    msgs |= self.generate_mavlink_messages([v for v in self.py2ros.values() if v.mavlink])

    chdrft_ros_proj_dir = '/home/benoit/repos/ws/chdrft_ros_messages'
    auto_dir = f'{chdrft_ros_proj_dir}/chdrft/auto'
    shutil.rmtree(auto_dir)
    cmisc.makedirs(auto_dir)
    for fname, desc in msgs.items():
      f = f'{auto_dir}/{fname}.msg'
      content = '\n'.join(desc)
      glog.info(f'Output message >> {fname} >> {content}')

      FileFormatHelper.Write(f, content, mode='txt')

    self._build_package()

  def _build_package(self):
    sp.check_call(
        '''
pushd ~/repos/ws;
source ~/programmation/env/env_px4.sh
colcon build --cmake-args="-DCMAKE_EXPORT_COMPILE_COMMANDS=1" --merge-install;
''',
        shell=True
    )


class RospyContext(cmisc.PatchedModel):
  pubs: list = cmisc.pyd_f(list)
  node: rclpy.node.Node = None
  cbs: A = cmisc.pyd_f(A)
  tg: ThreadGroup = cmisc.pyd_f(ThreadGroup)
  subid: str = None
  started: bool = False
  created_delayed: list = cmisc.pyd_f(list)

  def __post_init__(self):
    pass

  def create_publisher(self, name, pub_type, raw=False):
    if not self.started:
      self.created_delayed.append(lambda: self.create_publisher(name, pub_type))
      return lambda x: self.cbs[name](x)

    pub_name = name
    if self.subid is not None:
      pub_name = f'{self.subid}/{name}'

    if raw:
      ros_type = pub_type
    else:
      desc = self.mh.py2ros[pub_type]
      ros_type = desc.pub_type
    pub = self.node.create_publisher(ros_type, pub_name, 10)
    self.pubs.append(pub)

    def push_func(data):
      pub.publish(data if raw else desc.py2ros(data))

    self.cbs[name] = push_func
    return push_func

  @cmisc.cached_property
  def mh(self) -> RosMessageHelper:
    return RosMessageHelper()

  @cmisc.contextlib.contextmanager
  def enter(self):
    rclpy.init()
    self.node = rclpy.node.Node('chdrft', namespace='chdrft')
    self.started = True
    for x in self.created_delayed:
      x()
    try:
      done_future = rclpy.task.Future()
      self.tg.done_cbs.append(lambda: done_future.set_result(None))
      self.tg.add(func=lambda ev: rclpy.spin_until_future_complete(self.node, done_future))

      with self.tg.enter():
        yield self
    finally:
      self.node.destroy_node()
      rclpy.shutdown()


#%%


def test_publisher(ctx):
  from chdrft.projects.uav_py.lib.siyi import SiyiJoyData
  with RospyContext().enter() as rctx:
    cb = rctx.create_publisher('pub1', SiyiJoyData)
    cb2 = rctx.create_publisher('pub2', mavlink_msgs.MAVLink_radio_status_message)
    i = 0
    while True:
      i += 0.1
      time.sleep(0.1)
      cb(SiyiJoyData(ctrl=np.cos(i)))
      cb2(
          mavlink_msgs.MAVLink_radio_status_message(
              rssi=10, fixed=10, noise=3, txbuf=2, remrssi=4, remnoise=10, rxerrors=100
          )
      )


def message_db_list():
  from chdrft.projects.uav_py.lib.siyi import SiyiJoyData

  return [SiyiJoyData, mavlink_msgs.MAVLink_radio_status_message]


def ros_register_messages(ctx):
  RosMessageHelper().build_ros_package()


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
