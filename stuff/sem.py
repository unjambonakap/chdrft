#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import chdrft.utils.K as K
from chdrft.tools.file_to_img import to_greyscale
from flask import Flask, request,jsonify, Response
import ctypes
import PIL
import cv2
import numpy as np
import json
import mocksem
from mocksem import g, deg2rad
import uuid
import ntpath
import os
import copy

global flags, cache
flags = None
cache = None
def read_tiff(fname):
  return cv2.imread(fname, 0)

g.test_tiff_fname = r'C:\T2.TIF'

code_scan_image = '''
x.SetFilterMode(ctypes.c_short(2), ctypes.c_short({0}))
a16, b16 = ctypes.c_short(), ctypes.c_short(0)
for i in range(10000):
  time.sleep(0.01)
  x.GetFilterMode(ctypes.byref(a16), ctypes.byref(b16))
  if a16.value == 3:
    width, height =  768, 484
    buf = (ctypes.c_char * (width * height))()
    x.GetImage(buf)
    sys.stdout.write(buf.raw)
    break
else:
  print('failed')
'''

wait_stage_done = '''
a16 = ctypes.c_short()
for i in range(1000):
  x.IsStageMoving(ctypes.byref(a16))
  if a16.value == 0:
    print('ok')
    break
else:
  print('failed')
'''

code_get_file = '''
sys.stdout.write(open(r"{}", 'rb').read())
'''

def hook_scan_image(img):
  print(type(img))
  a = np.array(bytearray(img))
  a = a.reshape((g.sem_height, g.sem_real_width))
  a = a[:, :g.sem_width]
  return bytes(np.ravel(a))

def get_image_tiff(auto):
  auto.get('write_tiff_image', 0<<13 | 1<<4, g.test_tiff_fname)
  return auto.file(g.test_tiff_fname)

def compute_m_per_px(auto):
  assert 0
def unimplemented_func(auto):
  assert 0

def to_position(a): return dict(X=a[0], Y=a[1])
def from_position(a): return a[0]['X'], a[0]['Y']

g_funcs_def = [
  cmisc.Attr(endpoint='mag', name='Magnification', args=['float']),
  cmisc.Attr(endpoint='pos', name='Position', args=['float']*3),
  cmisc.Attr(endpoint='pos_xy', name='PositionXY', args=['float']*2, typ=to_position, from_typ=from_position),
  cmisc.Attr(endpoint='rot', name='Rot', args=['float']),
  cmisc.Attr(endpoint='screen', name='ScreenMetrics', args=['float']*2),
  cmisc.Attr(endpoint='z', name='Z', args=['float']),
  cmisc.Attr(endpoint='tilt', name='Tilt', args=['float']),
  cmisc.Attr(endpoint='stage_pos', name='StagePosition', args=['float']*5),
  cmisc.Attr(endpoint='write_image', name='WriteImage', args=['str'], raw=1),
  cmisc.Attr(endpoint='write_tiff_image', name='WriteTIFFImage', args=['str', 'short'], raw=1),
  cmisc.Attr(endpoint='wait_stage_done', name=None, code=wait_stage_done),
  cmisc.Attr(endpoint='instrument_id', name='InstrumentID', args=['int32']),
  cmisc.Attr(endpoint='tension', name='HighTension', args=['float']),
  cmisc.Attr(endpoint='tension_onfoff', name='HTOnOff', args=['short']),
  cmisc.Attr(endpoint='scan_mode', name='ScanMode', args=['short']),
  cmisc.Attr(endpoint='scan_mode2', name='ScanMode2', args=['short']),
  cmisc.Attr(endpoint='lines_per_frame', name='NrOfLines', args=['short']),
  cmisc.Attr(endpoint='line_time', name='LineTime', args=['short']),
  cmisc.Attr(endpoint='scan_format', name='ScanFrameFormat', args=['short', 'short']),

  cmisc.Attr(endpoint='m_per_px', name=None, typ=to_position, from_typ=from_position, func=compute_m_per_px),
  cmisc.Attr(endpoint='rotation', name=None, func=unimplemented_func),
  cmisc.Attr(endpoint='flipx', name=None, func=unimplemented_func),
  cmisc.Attr(endpoint='flipy', name=None, func=unimplemented_func),
  cmisc.Attr(endpoint='file', name=None, code=code_get_file, raw_output=1),
  cmisc.Attr(endpoint='image', name=None, code=code_scan_image, raw_output=1, hook=hook_scan_image),
  cmisc.Attr(endpoint='image_tiff', name=None, raw_output=1, func=get_image_tiff),
]


funcs_map = cmisc.Attr({x.endpoint: x for x in g_funcs_def})



def jsonify_set(f):
  def g(*args):
    res = f(request.get_json(), *args)
    if res is None: return jsonify({})
    return res
  return g

def jsonify_get(f):
  def g(*args):
    res = f(*args)
    if isinstance(res, bytes): return res
    data = json.dumps(res)
    return Response(data, content_type='application/json; charset=utf-8')

  return g

def test(ctx):
  def read(a):
    return set([l.strip() for l in open(a, 'r').readlines()])
  b = read('./list_public')
  a = read('./list_private')
  print(len(a), len(b), len(b.intersection(a)))
  diff = a.difference(b)

  orig = read('./private.orig')
  res=  []
  for l in orig:
    cnt =0
    for j in diff:
      if l.find('WINAPI '+j)!=-1:
        res.append(l)
        cnt += 1
    assert cnt <= 1
  assert len(res) == len(diff)

  print('\n'.join(res))


def idem(args): return args


#    print(a)
#    x.GetMagnification(ctypes.byref(a))
#    x.GetPosition(ctypes.byref(a), ctypes.byref(b), ctypes.byref(c))
#    x.GetScreenMetrics(ctypes.byref(a), ctypes.byref(b))

class AutoFuncsRC:
  def __init__(self, funcs, rc, ctx):
    self.funcs = funcs
    self.rc = rc
    self.ctx = ctx
    self.mock = None
    if ctx.mock: self.mock = mocksem.MockSEM(ctx)

  def normalize_func(self, func):
    if isinstance(func, dict): return func

    for f in self.funcs:
      if f.name == func or f.endpoint == func:
        return f
    assert 0

  def get_vars(self, f, vals=None):
    args = []
    if not vals:
      vals = [''] * len(f.args)
    for arg, val in zip(f.args, vals):
      if arg == 'str': args.append(f'ctypes.create_unicode_buffer({repr(val)})')
      else: args.append(f'ctypes.c_{arg}({str(val)})')

    return f'[{",".join(args)}]'


  def get_name(self, f, is_get):
    if f.get('raw', 0): return f.name
    if is_get: return 'Get'+f.name
    return 'Set'+f.name

  def get(self, f, vals):
    if vals is not None: vals = cmisc.to_list(vals)
    f = self.normalize_func(f)

    if self.mock: u= self.mock.get(f, vals)
    else:
      if 'code' in f: u =  self.get_code_func(f, vals)
      elif 'func' in f: u = f['func'](self, *vals)
      else:
        code = f'''
TMP_VAR = {self.get_vars(f, vals=vals)}
byref = list(map(ctypes.byref, TMP_VAR))
x.{self.get_name(f, 1)}(*byref)
print(list([a.value for a in TMP_VAR]))
    '''
        print(code)
        u= eval(self.rc.send(code))
    u = f.get('typ', idem)(u)
    return u

  def set(self, f, vals):
    vals = cmisc.to_list(vals)
    f =self.normalize_func(f)

    vals = f.get('from_typ', idem)(vals)
    if self.mock: return self.mock.set(f.endpoint, vals)

    code = f'''
TMP_VAR = {self.get_vars(f, vals)}
x.{self.get_name(f, 0)}(*TMP_VAR)
'''
    return self.rc.send(code)


  def get_code_func(self, f, arg):
    f = self.normalize_func(f)
    arg = cmisc.to_list(arg)
    code = f.code.format(*arg)
    res =  self.rc.send(code)
    return self.post_processing(f, res)

  def post_processing(self, f, res):
    hook = f.get('hook', None)
    if hook is None: return res
    return hook(res)

class RemoteController:
  def __init__(self, conn):
    self.conn = conn


  def send(self, a):
    glog.debug(a)
    self.conn.send(Z.Format(a).to_bytes().v + b'\x00')
    sz_packed = self.conn.recv_fixed_size(4)
    sz, = Z.struct.unpack('<I', sz_packed)
    print('want ', sz)
    res = self.conn.recv_fixed_size(sz)
    return res



def exec_client_prog(ctx):
  with Z.Server(12345) as conn:
    content = open('./client_prog.py', 'r').read()
    rc = RemoteController(conn)
    res = rc.send(content).decode()
    print('Result >> ',res)




def imgbuf_to_pil(content, w=g.sem_width, h=g.sem_height):
  content = bytearray(content)
  arr = Z.np.array(content, dtype='uint8')
  print(len(arr))
  arr = arr.reshape((h,w))
  img =PIL.Image.fromarray(arr)
  return img


def save_image(ctx):
  with Z.Server(12345) as conn:
    rc = RemoteController(conn)
    client = AutoFuncsRC(g_funcs_def, rc, ctx)
    set_ctx(client, ctx)
    #client.set('mag', 10000)
    res = client.get_code_func('image', ctx.integrate_nframes)
    if res == b'failed': raise Exception('failed')

    open('./img', 'wb').write(res)
    img = imgbuf_to_pil(res)
    img.save('./test.png')
    img.show()

def save_image_tiff(ctx):
  with Z.Server(12345) as conn:
    rc = RemoteController(conn)
    client = AutoFuncsRC(g_funcs_def, rc, ctx)
    set_ctx(client, ctx)
    client.set('write_tiff_image', (g.test_tiff_fname, 1<<4))
    open(ctx.outtiff, 'wb').write(client.get_code_func('file', g.test_tiff_fname))


def image_get():
  x = open('./img.b64', 'r').read()
  content = Z.base64.b64decode(x)
  content = bytearray(content)
  res= dict(Width=g.sem_width, Height=g.sem_height, Content=x)
  return jsonify(res)





def rest_server(ctx):
  app = Flask(__name__)
  app.config['JSON_AS_ASCII'] = False
  if ctx.mock: conn = Z.ExitStack()
  else: conn = Z.Server(12345)

  with conn as conn:
    rc = RemoteController(conn)
    client = AutoFuncsRC(g_funcs_def, rc, ctx)
    def define_getset(endpoint, f):
      if f is None: f = endpoint
      @jsonify_set
      def do_set(x): return client.set(f, x)

      @jsonify_set
      @jsonify_get
      def do_get(*args): return client.get(f, *args)


      def getset():
        if request.method == 'POST':
          return do_set()
        else:
          return do_get()

      return cmisc.Attr(getset=getset, set=do_set, get=do_get)


    for func in g_funcs_def:
      handler = None

      if 'code' in func:
        handler = jsonify_set(jsonify_get(lambda arg: client.get(func, arg)))
      else:
        fx = define_getset(func.endpoint, func.name)
        if 'raw' in func: handler = fx.get
        else: handler = fx.getset

      app.route(f'/api/{func.endpoint}', endpoint=func.endpoint, methods=['GET', 'POST'])(handler)

    @jsonify_set
    @jsonify_get
    def test1(args):
      print('OOON ', args)
      return args

    @jsonify_get
    def image_info(*args): return (g.sem_width, g.sem_height)

    app.route(f'/api/test', endpoint='test', methods=['GET', 'POST'])(test1)
    app.route(f'/api/image_info', endpoint='image_info', methods=['GET'])(image_info)

    app.run(host='0.0.0.0', port=8080)



def test_tiling(ctx):
  with Z.Server(12345) as conn:
    rc = RemoteController(conn)
    client = AutoFuncsRC(g_funcs_def, rc)
    set_ctx(client, ctx)

    data = cmisc.Attr()
    data.mag = client.get('mag')
    data.screen = Z.np.array(client.get('screen'))


    client.set('pos_xy', (0.210, -0.290))
    data.pos = Z.np.array(client.get('pos')[:2])
    print(data.pos)
    data.crop_ratio = 0.1
    data.overlap_ratio = 0.05
    data.crop_ratio = 0.0
    data.overlap_ratio = 0.00

    img_dim = data.screen / data.mag
    data.tile_dim = img_dim * (1-data.crop_ratio)
    data.stride = data.tile_dim * (1-data.overlap_ratio)
    print(img_dim, data.stride)

    sz = 2
    data.tiles = []
    for i in range(sz):
      for j in range(sz):
        pos = data.pos + Z.np.array((i,j)) * data.stride

        print('start set pos')
        for k in range(10):
          client.set('pos_xy', pos.tolist())
        print('end set pos')
        tile = cmisc.Attr(want_pos=pos, real_pos=client.get('pos'), coord=(i,j))
        print(tile.real_pos, tile.want_pos.tolist())
        image = client.get_code_func('image', ctx.integrate_nframes)
        assert image != 'failed'
        tile.image = image
        data.tiles.append(tile)


    Z.pickle.dump(data, open('./dump.pickle', 'wb'))


def print_tiling(ctx):
  data = Z.pickle.load(open('./dump.pickle', 'rb'))

  sz = 2
  dims = sz * g.sem_height, sz * g.sem_width

  res = np.zeros(dims, dtype='uint8')

  for tile in data.tiles:
    print(tile.want_pos.tolist(), tile.real_pos)
    content = bytearray(tile.image)
    arr = Z.np.array(content, dtype='uint8')
    w, h = g.sem_width, g.sem_height
    arr = arr.reshape((h,w))
    pos = np.array([h * (sz-1-tile.coord[1]), w * tile.coord[0]])
    print(pos, w, h, sz, tile.coord)
    res[pos[0]:pos[0]+h,pos[1]:pos[1]+w] = arr;

  img =PIL.Image.fromarray(res)
  img.save('tiling.png')
  #img.show()


def set_ctx(client, ctx): 
  if ctx.tension: client.set('tension', ctx.tension)
  if ctx.mag: client.set('mag', ctx.mag)
  if ctx.line_time: client.set('line_time', ctx.line_time)
  if ctx.lines_per_frame: client.set('lines_per_frame', ctx.lines_per_frame)

def test_server(ctx):
  with Z.Server(12345) as conn:
    rc = RemoteController(conn)
    client = AutoFuncsRC(g_funcs_def, rc)
    print(client.get('instrument_id'))
    print(client.get('tension_onfoff'))
    print(client.get('scan_mode'))
    print(client.get('line_time'))
    print(client.get('lines_per_frame'))
    print(client.get('scan_format'))
    return
    if 0:
      res = rc.send(open('./client_prog.py', 'r').read())
      img = imgbuf_to_pil(res, w=768, h=484)
      img.show()
      return
    #print(client.set('write_image', r'C:\benoit\t1.img'))
    print(client.set('write_tiff_image', (g.test_tiff_fname, 1<<4)))
    open(ctx.outtiff, 'wb').write(client.get_code_func('file', g.test_tiff_fname))
    return
    print(client.set('write_tiff_image', (g.test_tiff_fname, 1<<4 | 1<<15 | 1<<14 | 1<<13)))



def test_json(ctx):
  data = json.load(open(ctx.filename, 'rb'))
  data = cmisc.Attr.RecursiveImport(data)

  rangex = Z.Range1D(0, n=0, is_int=1)
  rangey = Z.Range1D(0, n=0, is_int=1)

  for x in data:
    x.rx= rangex.make_new(x.Position.X, n=x.Width)
    x.ry= rangex.make_new(x.Position.Y, n=x.Height)
    rangex = rangex.union(x.rx)
    rangey = rangey.union(x.ry)

  img = np.zeros((rangey.n, rangex.n))
  for x in data:

    content = Z.base64.b64decode(x.Image.Pixels)
    arr = Z.np.array(bytearray(content), dtype='uint8')
    arr = arr.reshape((x.Height, x.Width))
    img[(x.ry - rangey.low).window,(x.rx - rangex.low).window] = arr
  cv2.imwrite('res.png', img)


  print(rangex)
  print(rangey)


class SEMProjectHelper:
  def __init__(self, name=None, outdir=None):
    self.data = {}
    self.name = name
    self.outdir = outdir
    self.scans=  []
    self.data['Scans'] = self.scans
    self.images_to_dump ={}
    def pointtojson(self, obj, data):
      assert len(obj) == 2, obj
      print(type(obj[0]), isinstance(obj[0], np.int64))
      return {'X': self.context.flatten(obj[0], reset=False), 'Y': self.context.flatten(obj[1], reset=False)}

    def int64tojson(_, obj ,data): return int(obj)
    def float64tojson(_, obj ,data): return float(obj)

    cmisc.define_json_handler(cmisc.Attr(tojson=pointtojson, type=np.ndarray, name='pointhandler'))
    cmisc.define_json_handler(cmisc.Attr(tojson=int64tojson, type=np.int64, name='int64handler'))
    cmisc.define_json_handler(cmisc.Attr(tojson=float64tojson, type=np.float64, name='int64handler'))

  def from_pos_obj(self, x):
    return np.array((x['X'], x['Y']))

  def generate_filename(self, img):
    return f'image-{img["Guid"]}.tif'

  def add_scan(self, info, imgs):
    scan = {}
    info['Guid'] = str(uuid.uuid4())
    info['NumberOfTiles'] = len(imgs)

    img_data = []
    for img in imgs:
      entry = {}
      entry['Position'] = img.pos
      entry['GridPosition'] = np.array(img.grid_pos)
      entry['OrigScanId'] = info['Guid']
      entry['Guid'] = str(uuid.uuid4())
      fname=  self.generate_filename(entry)
      entry['Filename'] = fname
      self.images_to_dump[fname] = img

      img_data.append(entry)


    scan['ScanInformation'] = info.to_dict()
    scan['Tiles'] = img_data

    print(scan)
    self.scans.append(scan)


  def write(self, outdir=None):
    if outdir is None: outdir = self.outdir

    output_fname = Z.os.path.join(outdir, self.name+'.json')
    img_outdir = Z.os.path.join(outdir, self.name)
    Z.failsafe(lambda: Z.os.makedirs(img_outdir))
    open(output_fname, 'w').write(cmisc.jsonpickle.dumps(self.data, make_refs=0))

    for k, v in self.images_to_dump.items():
      fname=  Z.os.path.join(img_outdir, k)
      print('writing ', fname)
      if 'img' in v: cv2.imwrite(fname, v['img'])
      else: Z.shutil.copy(v.fullpath, fname)





def create_sem_project(ctx):
  img = read_tiff(ctx.filename)
  img = mocksem.to_grayscale(img)
  width = g.sem_width
  height = g.sem_height

  cmisc.failsafe(lambda x: Z.os.makedirs(img_dir))
  r1 = Z.Box(low=(500, 800), dim=(width*3, height*5))
  r2 = Z.Box(low=(1000, 600), dim=(width*1, height*3))


  ph = SEMProjectHelper(name='sem_project')
  base =  np.array((3, 4))
  size_ppx=  2e-3 / img.shape[0] # 2mm
  stride = (width * (1-ctx.overlap), height * (1-ctx.overlap))
  img_dim = np.array([width, height])

  for r in (r1, r2):
    imgs = []
    for xx, yy in Z.itertools.product(np.arange(0, r.width, stride[0]), np.arange(0, r.height, stride[1])):
      x = int(xx)
      y = int(yy)
      ix = xx // stride[0]
      iy = yy // stride[1]
      subimg = img[r.yl+y:r.yl+y+height, r.xl+x:r.xl+x+width]
      rel_pos = np.array([x,y]) * size_ppx
      imgs.append(cmisc.Attr(img=subimg, pos=rel_pos, grid_pos=(ix, iy)))

    print(r.dim * size_ppx)
    ph.add_scan(cmisc.Attr(Position=(r.low * size_ppx + base),
                           Size=r.dim * size_ppx, TileSize=img_dim*size_ppx,
                           ImageDim=img_dim,
                           Magnification=10, m_per_px=size_ppx), imgs)

  ph.write(ctx.outdir)




def correl_processing(a, b, want_color=1, ROI=None):
  ch = Z.dsp_utils.CorrelHelper(a, b)
  ch.compute_full_correrl()
  ch.normalize_incomplete_correls()
  best = ch.compute_best(min_overlap=(0.05, 0.9), ROI=ROI)
  power_colored= ch.get_colored_powerlog() if want_color else None
  return cmisc.Attr(best=best, power_colored=power_colored)



def plot_image_correl(ctx):
  im0 = read_tiff(ctx.infiles[0]).astype(float)
  im1 = read_tiff(ctx.infiles[1]).astype(float)
  target_dim = np.array(im0.shape) / ctx.downscale
  im0 = cv2.resize(im0, tuple(target_dim[::-1].astype(int)))
  im1 = cv2.resize(im1, tuple(target_dim[::-1].astype(int)))


  cp = correl_processing(im0, im1)
  print('BETS IS AT ', cp.best)

  K.plot_img(cp.power_colored)


def downscale_img(img, downscale):
  target_scale = np.array(img.shape) / downscale
  return cv2.resize(img, tuple(target_scale[::-1].astype(int)))


def compute_flow(im0, im1, downscale=10, ROI=None, offset_hint=None, use_mse=False):

  if offset_hint is None:
    v0 = downscale_img(im0, downscale)
    v1 = downscale_img(im1, downscale)
    cp = correl_processing(v0, v1, ROI=ROI / downscale)
    print('GOOOT BEST ', cp.best)
    offset= cp.best.offset * downscale
  else:
    offset =offset_hint

  rework_range = 3*downscale
  rx = np.arange(-rework_range, rework_range) + offset[0]
  ry = np.arange(-rework_range, rework_range) + offset[1]

  im0, im1 = Z.dsp_utils.norm_for_correl(im0, im1)

  ch = Z.dsp_utils.CorrelHelper(im0, im1)
  best = ch.compute_best_with_offsets(Z.itertools.product(rx, ry), use_mse=use_mse)
  if 0:
    Z.plt.hist(np.ravel(ch.power), bins=256)
    Z.plt.show()
    K.plot_img(ch.get_colored_powerlog())
  return best





def to_rgba(a, alpha):
  a = np.stack((a,)*4, axis=-1)
  a[:,:,3] = alpha
  return a



def overlay_imgs(im0, im1, offset):
  G = K.GraphHelper()
  offset *= 1
  #im0 = shift_image(im0, -offset)
  #im1 = shift_image(im1, offset)
  ch = Z.dsp_utils.CorrelHelper(im0, im1)
  im0 = to_rgba(im0, 128)
  im1 = to_rgba(im1, 128)
  print(ch.compute_score_at_offset(offset))


  p0 = Z.Dataset2d(im0)
  p1 = Z.Dataset2d(im1, x0=Z.Sampler1D(offset[0]), x1=Z.Sampler1D(offset[1]))
  G.create_plot(images=[p0, p1])

  if 0:
    im0 = Z.Dataset2d(get_image(images[0].fname).astype(float))
    im1 = Z.Dataset2d(get_image(images[1].fname).astype(float))
    G.create_plot(images=[im0])
    G.create_plot(images=[im1])
  G.run()


def test_flow(ctx):
  im0 = read_tiff(ctx.infiles[0]).astype(float)
  im1 = read_tiff(ctx.infiles[1]).astype(float)
  if 1:
    best = compute_flow(im0, im1, ctx.downscale, ROI=ctx.ROI, offset_hint=ctx.offset_hint, use_mse=ctx.use_mse)
    offset= best.dp
    print(best)
  else:
    offset = (-16, 760)
  #offset = (-69, 0)
  overlay_imgs(im0, im1, np.array(offset))




def compute_poslist(ctx, mag, pos):
  mock = mocksem.MockSEM(ctx)
  mock.data.mag = mag
  mock.view_angle_y = deg2rad(30)
  mock.data.pos = pos
  rect = mock.get_chip_rect()
  return rect

def test_poslist(ctx):
  res = compute_poslist(ctx, ctx.mag, np.array(ctx.pos))
  print(res)


def get_mag_factor(ctx, dx, dpix, z, mag, screen_width):
  dx = abs(dx)
  dpix = abs(dpix)
  pl = compute_poslist(ctx, mag, np.array((0, 0, z)))
  screen_range = abs(pl[1][0] - pl[0][0])
  real_screen_range = dx / dpix * screen_width
  return screen_range / real_screen_range

def test_sem_params(ctx):
  dx = 14.9e-3
  z = 10.232e-3
  dpix = 339
  screen_width = g.sem_width
  mag = 2000
  f1 = get_mag_factor(ctx, dx, dpix, z, mag, screen_width)

  dx = 4.9e-3
  mag = 6500
  dpix =340
  screen_width = g.sem_width
  f2 = get_mag_factor(ctx, dx, dpix, z, mag, screen_width)
  print('FACTOR IS >>> ', f1, f2)

  pl = compute_poslist(ctx, f1 * 3500, np.array((-0.41275, 0.877419, 9.96667))*1e-3)
  pl2 = compute_poslist(ctx, f1 * 3500, np.array((-0.382532, 0.877419, 9.96667))*1e-3)

  print()
  print(pl)
  print(pl2)


def convert_to_sem_project(infile):
  mag = None
  images = []
  images_dir = os.path.dirname(infile)
  grid = dict()

  xvals = set()
  yvals = set()
  rex = Z.re.compile('S(?P<x>\d{3})_(?P<y>\d{3}).tif$', Z.re.I)
  for line in open(infile, 'r').readlines():
    cols = line.strip().split(' ')
    if cols[2] == 'Magnification:': mag = int(cols[3][:-1])
    if cols[2] == 'Position:': pos = np.array(eval(' '.join(cols[3:6]))) / 1e3 # mm to m
    elif cols[2] == '=>':
      fname = ntpath.basename(cols[3]).upper()
      fname_grid = rex.match(fname)
      obj = cmisc.Attr(pos=pos, fname=fname, x=int(fname_grid['x']),
                       y=int(fname_grid['y']), fullpath=os.path.join(images_dir, fname))
      print(obj.fullpath)
      if not os.path.exists(obj.fullpath): continue
      xvals.add(obj.x)
      yvals.add(obj.y)

      grid[(obj.x, obj.y)] = obj


  assert len(grid) > 0
  one_image = next(iter(grid.items()))[1]
  image_dim = np.array(read_tiff(one_image.fullpath).shape)
  xr = Z.Range1D.FromSet(xvals)
  yr = Z.Range1D.FromSet(yvals)

  return cmisc.Attr(images_dir=images_dir, image_dim=image_dim, grid=grid, logfile=infile, mag=mag, xr=xr, yr=yr,)

def test_retrieve_params(ctx):
  hint_dy = (-16, 760)
  hint_dx = (1247, -1)
  proj = convert_to_sem_project(ctx.infile)
  p01 = proj.grid[(5,1)]
  p00 = proj.grid[(5,0)]
  d01pos = p01.pos - p00.pos
  f1 = get_mag_factor(ctx, d01pos[1], hint_dy[1], p01.pos[2], proj.mag, proj.image_dim[1])
  m_per_px = np.linalg.norm(d01pos[:-1] / hint_dy)

  #pl1 = compute_poslist(ctx, f1 * proj.mag, p00.pos)
  #pl2 = compute_poslist(ctx, f1 * proj.mag, p01.pos)
  #print(pl1)
  #print(pl2)
  #print(p00.pos)
  #print(p01.pos)

  #return

  print(f1)
  oxy = proj.grid[(10,12)]
  ox1y = proj.grid[(11,12)]
  pxy = oxy.pos
  px1y = ox1y.pos


  dp_real = px1y - pxy
  dp_screen = dp_real / m_per_px
  dp_screen[1] *= -1
  print('guessing ', dp_screen, dp_real)
  ia =read_tiff(oxy.fullpath)
  ib =read_tiff(ox1y.fullpath)
  kernel = np.ones((5,5), np.float32) / 25
  offset = dp_screen[:2].astype(int)
  ia = cv2.filter2D(ia, -1, kernel)
  ib = cv2.filter2D(ib, -1, kernel)

  offset = (1248, -5)
  overlay_imgs(ia, ib, offset)
  res =compute_flow(ia, ib, offset_hint=offset, use_mse=ctx.use_mse)
  overlay_imgs(ia, ib, res.offset)
  print(res)



  return

  #hint_dx = 

  print(images[0].fname)
  print(images[1].fname)
  g.sem_height = image_dim[0]
  g.sem_width = image_dim[1]

  target_dim = image_dim / 10


  sx = None
  for i in range(10):
    im0 = get_image(images[i].fname).astype(float)
    im1 = get_image(images[i+1].fname).astype(float)
    im0 = cv2.resize(im0, tuple(target_dim[::-1].astype(int)))
    im1 = cv2.resize(im1, tuple(target_dim[::-1].astype(int)))


    res = Z.dsp_utils.compute_normed_correl2(im0, im1)
    if sx is None: sx = res
    else: sx += res
    print(type(res))

  print(np.min(sx))
  print(np.max(sx))
  power = 10 * np.log10(np.abs(sx))
  power_colored = Z.dsp_utils.g_fft_helper.map(power)


  G = K.GraphHelper()
  p1 = Z.Dataset2d(power_colored)
  G.create_plot(images=[p1])
  im0 = Z.Dataset2d(get_image(images[0].fname).astype(float))
  im1 = Z.Dataset2d(get_image(images[1].fname).astype(float))
  G.create_plot(images=[im0])
  G.create_plot(images=[im1])
  G.run()

  return
  rect =  compute_poslist(ctx, mag, images[0].pos)
  print(rect)
  mock.data.pos = images[1].pos
  rect =  compute_poslist(ctx, mag, images[1].pos)
  rect = mock.get_chip_rect()
  print(rect)


def compute_updated_pos(cur, prev, hint_vec, **kwargs):
  icur =read_tiff(cur.fullpath)
  iprev =read_tiff(prev.fullpath)

  res = compute_flow(iprev, icur, offset_hint=hint_vec, **kwargs)
  print(res, hint_vec)
  return prev.corrected_pos + res.offset


def add_corrected_position_to_images(proj, hint_dx, hint_dy):
  dp_tot_realspace = np.zeros((2,))
  dp_tot_screenspace = np.zeros((2,))
  for x,y in Z.Range2D(proj.xr, proj.yr):
    p = np.array([x,y])
    cur = proj.grid[(x,y)]

    poslist = []
    for dv, hint in (((1, 0), hint_dx), ((0, 1),hint_dy)):
      dv = np.array(dv)
      pp = p - dv
      if pp[0] not in proj.xr.range or pp[1] not in proj.yr.range: continue

      adj = proj.grid[tuple(pp.tolist())]
      dp_tot_realspace += cur.pos[:2] - adj.pos[:2]
      npos = compute_updated_pos(cur, adj, hint)
      dp_tot_screenspace += npos - adj.corrected_pos
      poslist.append(npos)

    if not poslist: poslist.append(np.array([0,0]))
    cur.corrected_pos = np.mean(poslist, axis=0)
  proj.guessed_size_ppx = dp_tot_realspace/  dp_tot_screenspace


def test_remap_project(ctx):
  hint_dy = (-16, 760)
  hint_dx = (1247, -1)
  proj = convert_to_sem_project(ctx.infile)
  add_corrected_position_to_images(proj, hint_dx, hint_dy)


def numpy_slice_from_point_and_dim(pt, dim):
  res = []
  for i in range(len(pt)):
    res.append(slice(pt[i], pt[i]+dim[i]))
  return tuple(res)



def write_temp_image(img):
  tfile = app.global_context.enter_context(Z.tempfile.NamedTemporaryFile(suffix='.tif'))
  cv2.imwrite(tfile.name, img)
  return tfile.name

def dump_remapped_project(proj, outdir, clip_borders, name='sem_project'):
  ph = SEMProjectHelper(name=name)
  size_ppx = np.max(proj.guessed_size_ppx)
  print(size_ppx)

  image_dim = proj.image_dim[::-1]
  clip_image_dim = np.round(image_dim * clip_borders).astype(int)
  center_img_off = (image_dim - clip_image_dim) // 2
  imgs = []
  for gridpos, dimg in proj.grid.items():
    img = read_tiff(dimg.fullpath)
    img = img[numpy_slice_from_point_and_dim(center_img_off, clip_image_dim)[::-1]]
    print(img.shape)

    imgs.append(cmisc.Attr(fullpath=write_temp_image(img), pos=dimg.corrected_pos * size_ppx, grid_pos = gridpos))

  im0 = next(iter(proj.grid.values()))


  scan_data = cmisc.Attr(
    Position=im0.pos[:2],
    ImageDim=clip_image_dim,
    TileSize=clip_image_dim*size_ppx,
    Magnification=10,
    m_per_px=size_ppx)

  ph.add_scan(scan_data, imgs)

  ph.write(outdir)


def remap_project(ctx):
  proj = convert_to_sem_project(ctx.infile)

  hint_dy = (-16, 760)
  hint_dx = (1247, -1)
  add_corrected_position_to_images(proj, hint_dx, hint_dy)

  dump_remapped_project(proj, ctx.outdir, ctx.clip_borders)


def read_sem_project(filepath, normalize=0):
  res =  cmisc.Attr.FromJson(filepath)
  if not normalize: return res
  helper = SEMProjectHelper()
  image_dir = os.path.splitext(filepath)[0]

  for scan in res.Scans:
    scan.ScanInformation.TileSize = helper.from_pos_obj(scan.ScanInformation.TileSize)
    scan.ScanInformation.ImageDim = helper.from_pos_obj(scan.ScanInformation.ImageDim)
    print(scan.ScanInformation)
    scan.ScanInformation.Position = helper.from_pos_obj(scan.ScanInformation.Position)

    for tile in scan.Tiles:
      tile.Position = helper.from_pos_obj(tile.Position)
      tile.GridPos = helper.from_pos_obj(tile.GridPos)
      tile.fname = os.path.join(image_dir, helper.generate_filename(tile))
      tile.rect= Z.Range2D.FromPosAndSize(tile.Position, scan.ScanInformation.TileSize)


  return res

def render_project(ctx):
  proj = read_sem_project(ctx.infile, normalize=1)

  helper = SEMProjectHelper()


  image_dim = None
  imgs_all = []
  rects = []
  for scan in proj.Scans:
    image_dim = scan.ScanInformation.ImageDim
    ts = scan.ScanInformation.TileSize
    for tile in scan.Tiles:

      r = tile.rect + scan.ScanInformation.Position
      if not r.intersect(ctx.ROI): continue
      rects.append(r)
      imgs_all.append(cmisc.Attr(fname=tile.fname, rect=r, clip_ratio=0.01))
  vb = Z.Range2D.Union(rects).box



  aspect= vb.aspect
  print('TARGET ASPECT >> ', aspect)
  height =  6000
  if ctx.outfile:
    target_dim = (int(aspect * height), height)
    mocksem.create_image_composition(imgs_all, target_dim, ctx.outfile)

  G = K.GraphHelper()
  imgs = []
  for img_data in imgs_all:
    img = read_tiff(img_data.fname)
    imgs.append(Z.Dataset2d(img, img_data.rect.xr, img_data.rect.yr))

  G.create_plot(images=imgs)
  if ctx.outfile:
    G.create_plot(images=[Z.Dataset2d(cv2.imread(ctx.outfile)[::-1,:,:])])
  G.run()




def rotate_project(proj, angle, helper):

  assert len(proj.Scans) == 1
  scan = proj.Scans[0]
  rects = []
  for tile in scan.Tiles:
    rects.append(tile.rect)
  viewbox = Z.Range2D.Union(rects).box
  world2screen_scale = Z.mat_scale(scan.ScanInformation.ImageDim / scan.ScanInformation.TileSize)
  screen2world_scale = Z.mat_scale( 1/ (scan.ScanInformation.ImageDim / scan.ScanInformation.TileSize))
  rot_center = viewbox.center
  tsf_mat = Z.mat_apply2d(
    Z.mat_translate(rot_center),
    Z.mat_rotz(angle),
    Z.mat_translate(-rot_center),
  )


  imgs=  []
  for tile in scan.Tiles:
    tbox = tile.rect.box.transform_by_mat(tsf_mat)
    inner_box = tbox.get_aabb(inner=1)
    screen2world = Z.mat_apply2d(Z.mat_translate(tile.Position), screen2world_scale)
    world2screen = Z.mat_apply2d( world2screen_scale, Z.mat_translate(-inner_box.low))

    affine_mat = Z.mat_apply2d(world2screen, tsf_mat, screen2world, affine=1)

    img = read_tiff(tile.fname)
    dsize=  np.round(inner_box.dim / scan.ScanInformation.m_per_px).astype(int)
    print(dsize, inner_box.dim)
    res = cv2.warpAffine(img, affine_mat, tuple(dsize.tolist()))
    imgs.append(cmisc.Attr(fullpath=write_temp_image(res), pos=inner_box.low, grid_pos=tile.GridPos))

  scan.ScanInformation.ImageDim = dsize
  scan.ScanInformation.TileSize = inner_box.dim
  helper.add_scan(scan.ScanInformation, imgs)
  helper.write()





def test_rotate(ctx):
  proj = read_sem_project(ctx.infile, normalize=1)
  helper = SEMProjectHelper(name=ctx.proj_name, outdir=ctx.outdir)
  nproj = rotate_project(proj, ctx.angle, helper)


def dump_proj_info(ctx):
  proj = read_sem_project(ctx.infile, normalize=1)
  print(f'Project {ctx.infile}')

  for scan in proj.Scans:
    rects = [tile.rect for tile in scan.Tiles]
    viewbox = (Z.Range2D.Union(rects) + scan.ScanInformation.Position).box
    print(f'Scan >>> ntiles={len(scan.Tiles)}')
    Z.pprint(scan.ScanInformation)
    print(viewbox)
    print()



def args(parser):
  clist = CmdsList().add(test)
  parser.add_argument('--outtiff', type=str, default='test.tiff')
  parser.add_argument('--tension', type=float)
  parser.add_argument('--mag', type=float)
  parser.add_argument('--pos', type=float, nargs='*')
  parser.add_argument('--filename')
  parser.add_argument('--outdir')
  parser.add_argument('--outfile')
  parser.add_argument('--proj-name', default='sem_project')
  parser.add_argument('--no-display', action='store_true')
  parser.add_argument('--infile')
  parser.add_argument('--angle', type=float)
  parser.add_argument('--infiles', nargs='*')
  parser.add_argument('--downscale', type=int, default=1)
  parser.add_argument('--line_time', type=int)
  parser.add_argument('--lines_per_frame', type=int)
  parser.add_argument('--integrate-nframes', type=int, default=0)
  parser.add_argument('--mock', action='store_true')
  parser.add_argument('--clip-borders', type=float, default=1.)
  parser.add_argument('--use-mse', action='store_true')
  parser.add_argument('--overlap', type=float, default=0.1)
  parser.add_argument('--offset-hint', type=float, nargs=2, default=None)
  parser.add_argument('--xr', type=Z.Range1D.FromString, default=Z.Range1D.All())
  parser.add_argument('--yr', type=Z.Range1D.FromString, default=Z.Range1D.All())
  mocksem.mocksem_args(parser)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)




'''
lines/frame: 3=968, 5=1936
line_time: 3=1.68, 4=3.36, 5=6.7,:13.4
'''


def main():
  ctx = Attributize()
  ctx.ROI = Z.Range2D(flags.xr, flags.yr)
  ActionHandler.Run(ctx)


app()
