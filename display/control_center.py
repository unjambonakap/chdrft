#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import chdrft.utils.Z as Z
import numpy as np
from pydantic.v1 import Field, validator
from chdrft.external.http_vlc import HttpVLC
import mpv
import folium
import folium.plugins

import fastapi, fastapi.encoders
from asyncio import run
import uvicorn
import chdrft.utils.rx_helpers as rx_helpers
import threading
import contextlib
from chdrft.config.env import qt_imports
import io
import shapely
import time
import reactivex as rx
from chdrft.dsp.datafile import Dataset
from chdrft.display.ui import OpaPlot
import random
#<NMEA(GPGGA, time=22:03:28, lat=44.8988163333, NS=N, lon=6.6427143333, EW=E, quality=1, numSV=6, HDOP=2.37, alt=1319.2, altUnit=M, sep=47.3, sepUnit=M, diffAge=, diffStation=)>

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--infile')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def shapely_as_feature(f, **kwargs):
  if isinstance(f, dict): return f
  return dict(
      type='Feature', geometry=cmisc.json_flatten(f), properties=dict(objectid=str(hash(f)), **kwargs)
  )


class MPVPlayer(qt_imports.QWidget):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    import locale
    locale.setlocale(locale.LC_NUMERIC, 'C')
    self.setAttribute(qt_imports.QtCore.Qt.WA_DontCreateNativeAncestors)
    self.setAttribute(qt_imports.QtCore.Qt.WA_NativeWindow)
    self.player = mpv.MPV(
        wid=str(int(self.winId())),
        vo='x11',  # You may not need this
        #log_handler=print,
        input_default_bindings=True,
        input_vo_keyboard=True,
        osc=True,
    )

  def add_obs(self, prop: str, target: rx.Subject, as_kv=True):

    @self.player.property_observer(prop)
    def obs_func(_name, value):
      if as_kv: target.on_next({prop: value})
      else: target.on_next(value)


class Server(uvicorn.Server):

  def install_signal_handlers(self):
    pass

  def run_in_thread(self):
    self.thread = threading.Thread(target=self.run)
    self.thread.start()
    while not self.started:
      time.sleep(1e-3)

  def quit(self):
    self.should_exit = True
    self.thread.join()


class ObsEntry(cmisc.PatchedModel):
  name: str
  f: rx_helpers.WrapRX = None
  cb: cmisc.typing.Callable = None

  @validator('f')
  def _(cls, v):
    if v is None: return None
    return v.listen_value()

  @property
  def value(self):
    if self.cb: return self.cb()
    return self.f.value

  def dict(self, *args, **kwargs):
    return dict(name=self.name)


class ObsEntryDesc(cmisc.PatchedModel):
  obs: ObsEntry
  endpoint: str
  full_path: str
  push: bool


class RxServer(cmisc.PatchedModel):
  app: fastapi.FastAPI = None
  port: int = 9090
  endpoints: dict[str, ObsEntryDesc] = Field(default_factory=dict)
  server: Server = None

  def __init__(self):
    super().__init__()
    self.app = fastapi.FastAPI()
    from fastapi.middleware.cors import CORSMiddleware
    origins = [
        '*',
    ]
    self.app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @self.app.get('/listen')
    def listen(a):
      return cmisc.json_flatten(self.endpoints)

    config = uvicorn.Config(self.app, host='127.0.0.1', port=self.port, log_level='error')
    self.server = Server(config=config)

  def start(self):
    PortRegister.register(self.port, self.run())

  @contextlib.contextmanager
  def run_ctx(self):
    yield from self.run()

  def run(self):
    self.server.run_in_thread()
    yield
    self.server.quit()

  def add(self, obs: ObsEntry, key=None, push=False) -> ObsEntryDesc:
    if key is None: key = obs.name
    if key in self.endpoints:
      return self.add(obs, key+str(random.randint(0,9)))

    if push:

      endpoint = f'/push/{key}'

      @self.app.post(endpoint)
      def proc(r: fastapi.Request):
        obs.f.on_next(A.RecursiveImport(run(r.json())))
    else:
      endpoint = f'/obj/{key}'

      @self.app.get(endpoint)
      def proc():
        return cmisc.json_flatten(obs.value)

    res = ObsEntryDesc(
        obs=obs,
        endpoint=endpoint,
        full_path=f'http://{self.server.config.host}:{self.server.config.port}{endpoint}',
        push=push,
    )
    self.endpoints[key] = res
    return res


class Synchronizer(cmisc.PatchedModel):
  pl: MPVPlayer
  sx: RxServer
  at_t: rx_helpers.WrapRX
  opp: OpaPlot

  @classmethod
  def Make(cls, gw, sx, vid_name, data):
    pl = MPVPlayer()
    gw.add(pl)

    def proc(kv):
      t = kv['time-pos']
      if t is None:
        t = data.t[-1]
      idx = min(len(data.t) - 1, np.searchsorted(data.t, t))
      pos = data.lla[idx]
      return A(idx=idx, t=t, pos_geojson=shapely.geometry.Point(pos[:2]))

    event_source = rx_helpers.WrapRX(rx.Subject())
    pl.add_obs('time-pos', event_source)
    at_t = event_source.map(proc).listen_value()

    opp = OpaPlot(Dataset(x=data.t, y=data.pos[:, 2], name='z'), legend=1)
    gw.add(opp, label='data')

    pl.player.play(vid_name)
    return cls(
        pl=pl,
        sx=sx,
        at_t=at_t,
        opp=opp,
    )


class SingletonMeta(type):
  _instances = {}

  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
    return cls._instances[cls]


class PortRegister():
  services: dict[int, object] = dict()
  in_ctx: bool = False

  @classmethod
  def maybe_enter_ctx(cls):
    if cls.in_ctx: return
    cls.in_ctx = True

    @contextlib.contextmanager
    def proc():
      yield
      cls.unregister_all()

    app.global_context.enter_context(proc())

  @classmethod
  def unregister_all(cls):
    for port in list(cls.services.keys()):
      cls.unregister(port)

  @classmethod
  def unregister(cls, port):
    try:
      next(cls.services[port])
    except StopIteration as e:
      pass
    del cls.services[port]

  @classmethod
  def register(cls, port, obj):
    cls.maybe_enter_ctx()
    if port in cls.services:
      cls.unregister(port)
    next(obj)
    cls.services[port] = obj


def folium2widget(m):
  data = io.BytesIO()
  m.save(data, close_file=False)
  m.save('/tmp/test.html')

  q = qt_imports.QtWebEngineWidgets.QWebEngineView()
  q.setHtml(data.getvalue().decode())
  return q


class IconDesc(cmisc.PatchedModel):
  rotation: int = None
  name: str = None


def add_folium_rt(m, feed_endpoint, interval=1000):

  rt = folium.plugins.Realtime(
      feed_endpoint,
      get_feature_id=folium.JsCode('(f) => { return f.properties.objectid; }'),
      style = folium.JsCode( '(f) => { return f.properties.style; } '),
      interval=1000,
      remove_missing=True,
  ).add_to(m)
  print('laa')

  rt.functions['onEachFeature'] = folium.JsCode(
      f'''(geo, l) => {{ 
      if (geo.properties.icon) {{
        {rt.get_name()}._container.addLayer(l);
        l.setIcon(eval(geo.properties.icon.name)); 
        l._icon.children[0].style.transform = `rotate(${{geo.properties.icon.rotation}}deg)`;
      }}
  }};
  '''
  ).js_code


class IconAdder(cmisc.PatchedModel):
  reqs: list
  icons: dict = Field(default_factory=dict)

  def postproc(self, m):
    for req in self.reqs:
      icon = folium.plugins.BeautifyIcon(icon=req, text_color="#b3334f", icon_shape="triangle")

      folium.Marker(location=(0, 0), icon=icon, show=False).add_to(m)
      self.icons[req] = icon.get_name()



class FoliumHelper(cmisc.PatchedModel):
  m: folium.Map = None
  ica: IconAdder = None
  obs_endpoint: str = None



  def create_folium(self, lonlat, sat=True, zoom_start=18):
    m = folium.Map(location=(lonlat[1], lonlat[0]), zoom_start=zoom_start, max_zoom=22)
    if sat:
      folium.TileLayer(
          tiles=
          'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          attr='Esri',
          name='Esri Satellite',
          overlay=False,
          control=True,
        max_zoom=22,
      ).add_to(m)
    self.m = m

  def get_html(self) -> str:
    data = io.BytesIO()
    self.m.save(data, close_file=False)
    return data.getvalue().decode()

  def setup(self, rt_kwargs={}):
    if self.ica is not None:
      self.ica.postproc(self.m)


    if self.obs_endpoint:
      add_folium_rt(self.m, self.obs_endpoint, **rt_kwargs)

  def proc1(self, e, k) -> dict:
    if isinstance(e, dict):
      feature = e['feature']
      d = self.proc1(feature, k)
      prop = d['properties']
      if 'icon' in e:
        rot = e['rot']
        prop['objectid'] += f'_{rot}'
        prop['icon'] = IconDesc(rotation=rot, name=self.ica.icons[e['icon']])
      elif 'style' in e:
        prop['style'] = e['style']

    elif isinstance(e, shapely.geometry.base.BaseGeometry):
      d = shapely_as_feature(e)
      d['properties']['objectid'] += f'_{k}'

    return d

  def generate_widget(self):
    q = qt_imports.QtWebEngineWidgets.QWebEngineView()
    q.setHtml(self.get_html())
    return q

class FoliumRTHelper(cmisc.PatchedModel):
  fh: FoliumHelper
  data: A = Field(default_factory=A)
  obs: ObsEntryDesc = None
  seen_features: set = Field(default_factory=set)
  ev: rx_helpers.WrapRX = Field(
      default_factory=lambda: rx_helpers.WrapRX(rx.subject.BehaviorSubject(None))
  )

  def setup(self, sx: Synchronizer):
    self.obs = sx.add(ObsEntry(name='refresh_folium', cb=self.proc))
    assert self.fh.obs_endpoint is None
    self.fh.obs_endpoint = self.obs.full_path

  def proc(self):
    self.data['obj'] = self.ev.value
    features = []
    for k, v in self.data.items():
      if v is None: continue
      tb = v
      if not isinstance(v, list): tb = [v]
      for i, x in enumerate(tb):
        features.append(self.proc1(x, f'{k}_{i:03d}'))

    ids = {x['properties']['objectid'] for x in features}
    rem_features = self.seen_features - ids

    if False:
      # not working because don't know the state of the client
      features = cmisc.asq_query(features).where(
          lambda x: x['properties']['objectid'] not in self.seen_features
      ).to_list()
      for u in rem_features:
        features.append(dict(properties=dict(objectid=u)))

    self.seen_features |= ids
    return features


class VlcHelper(cmisc.PatchedModel):
  proc: object = None
  port: int = 12345
  password: str = '12345'

  @cmisc.cached_property
  def ctrl(self) -> HttpVLC:
    return HttpVLC(f'http://localhost:{self.port}', password=self.password)

  def start(self, vid):
    self.proc = Z.sp.Popen(
        [
            'vlc',
            f'--http-port={self.port}',
            f'--http-password={self.password}',
            '--extraintf',
            'http',
            vid,
        ]
    )
    time.sleep(0.1)


def test_plot(ctx):
  import chdrft.utils.K as K
  K.oplt.plot(np.array(np.arange(1, 10)), typ='graph')
  input()


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
