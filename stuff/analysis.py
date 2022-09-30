#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import chdrft.utils.K as K
import cv2
import numpy as np
import math
from shapely.geometry import Polygon, LineString, Point
import shapely.geometry

global flags, cache
flags = None
cache = None

g_render_via_str = 'via'
g_render_track_str = 'track'
g_render_image_str = 'image'
g_render_grid_str = 'grid'
g_target_shape = (16,16)


def args(parser):

  clist = CmdsList().add(test)
  parser.add_argument('--indir')
  parser.add_argument('--infile')
  parser.add_argument('--outdir')

  parser.add_argument('--needle')
  parser.add_argument('--grid-dx', type=float)
  parser.add_argument('--box-step-factor', type=float, default=1)
  parser.add_argument('--grid-dy', type=float)
  parser.add_argument('--haystack')
  parser.add_argument('--tile-id', type=int, default=0)
  parser.add_argument('--tile-ids', nargs='*', type=int, default=[])

  parser.add_argument('--render', action='store_true')
  parser.add_argument('--render-what', type=str, nargs='*', default=[])
  parser.add_argument('--tracks', action='store_true')
  parser.add_argument('--image', action='store_true')
  parser.add_argument('--vias', action='store_true')
  parser.add_argument('--vert-lines', action='store_true')

  parser.add_argument('--outfile')
  parser.add_argument('--xrange', nargs='*', type=int)
  parser.add_argument('--tracknums', nargs='*', type=int)
  parser.add_argument('--yrange', nargs='*', type=int)
  parser.add_argument('--theta-range', nargs='*', type=float)
  parser.add_argument('--xn', type=int)
  parser.add_argument('--yn', type=int)
  parser.add_argument('--hough-step', type=int, default=100)
  parser.add_argument('--remap', action='store_true')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


ix = 0


def dump_data(g, x):
  global ix
  cur = ix
  ix += 1
  g.add_node(ix, label=f'{x["Type"]}, {x["Size"]/1000.}')
  lst = x.get('lst')
  if lst:
    for i, nx in enumerate(lst):
      if nx is None: continue
      if i >= 10: break
      child = dump_data(g, nx)
      g.add_edge(cur, child)

  if x['kv']:
    for i, (k, v) in enumerate(x['kv'].items()):
      if v is None: continue
      if i >= 10: break
      child = dump_data(g, v)
      g.add_edge(cur, child, label=k)
  return cur


def test(ctx):
  data = cmisc.json.load(open('./graph.json', 'r'))
  g = Z.nx.DiGraph()
  dump_data(g, data)
  Z.nx.drawing.nx_pydot.write_dot(g, 'graph.res')
  pass


def extract_grid(ctx):
  rex = Z.re.compile('S(?P<col>\d{3})_(?P<row>\d{3}).tif$', Z.re.IGNORECASE)
  xrange = Z.Range1D(*ctx.xrange, n=ctx.xn, is_int=1)
  yrange = Z.Range1D(*ctx.yrange, n=ctx.yn, is_int=1)
  cmisc.failsafe(lambda: Z.os.makedirs(ctx.outdir))

  print(ctx.indir)
  for fname in Z.glob.glob(Z.os.path.join(ctx.indir, '*')):
    bname = Z.os.path.basename(fname)
    m = rex.match(bname)
    if not m: continue

    x = int(m.group('col'))
    y = int(m.group('row'))

    if x in xrange and y in yrange:
      x -= xrange.low
      y -= yrange.low

      if ctx.remap: nfname = f'S{x:03d}_{y:03d}.tif'
      else: nfname = bname
      nfname = Z.os.path.join(ctx.outdir, nfname)
      cmisc.failsafe(lambda: Z.shutil.copy2(fname, nfname))
      print(nfname)

def plot_image_hist(img):
  data = np.ravel(img)
  Z.plt.hist(data, bins=256)
  Z.plt.show()

def plot_histogram(ctx):
  data = []
  if ctx.infile:
    img = cv2.imread(ctx.infile)
    data.extend(np.ravel(img))
  if ctx.indir:
    for i, fname in enumerate(Z.glob.glob(Z.os.path.join(ctx.indir, '*.tif'))):
      img = cv2.imread(fname)
      data.extend(np.ravel(img))
      if i == 4: break

  Z.plt.hist(data, bins=256)
  Z.plt.show()




def render_img_with_line(img, center, angle):
  print(center)
  v = np.array([img.shape[1], 0])
  matrot = Z.mat_rotz(angle)
  nv = Z.mat_apply2d(matrot, v, vec=1)
  p1 = center - nv
  p2 = center + nv
  print(p1, p2)

  G = K.GraphHelper()
  G.create_plot(plots=[Z.Dataset.FromPoints((p1, p2))], images=[Z.Dataset2d(img)])
  G.run()

def hough(ctx):
  img = Z.to_grayscale(cv2.imread(ctx.infile, 0))
  shape_xy= np.array(img.shape)[::-1]


  print(img.shape)
  theta = np.linspace(ctx.theta_range.low, ctx.theta_range.high, ctx.hough_step)
  if ctx.vert_lines:
    theta -= math.pi / 2
  #res, theta, d = Z.hough_line(img, theta=np.linspace(-np.pi, np.pi, 1000))
  #res, _, d = Z.hough_line(img, theta=theta)
  res = Z.skimage.transform.radon(img, theta=Z.rad2deg(theta), circle=False)
  print('RES SHAPE >> ', res.shape)

  n = len(res)
  res = res[n // 4 : 3 *n // 4,:]
  scores = []
  print(res.shape)
  Z.plt.plot(res)
  Z.plt.show()

  for i in range(len(theta)):
    v = res[:, i].tolist()
    v.sort()
    scores.append(np.mean(v[:int(len(v) * 0.4)]))
  print(scores)

  #qx = np.quantile(res, 0.2, axis=0)

  Z.plt.plot(theta, scores)
  bestpos = np.argmin(scores)
  best = theta[bestpos]
  Z.plt.show()

  render_img_with_line(img, shape_xy/2, -best+math.pi/2)
  print('FOUND BEST aat ', best, scores[bestpos])

  return 0






  res = np.log(res + 1)
  res_flat = np.ravel(res).reshape(-1, 1)

  kmeans = Z.sklearn.cluster.KMeans(n_clusters=2, random_state=0).fit(res_flat)
  Z.plt.hist(np.ravel(res), bins=3000)
  Z.plt.show()
  target_cluster = np.argmax(kmeans.cluster_centers_)

  print(kmeans.cluster_centers_)
  zeropts = np.argwhere(kmeans.predict(res_flat).reshape(res.shape) == target_cluster)
  d = Z.defaultdict(int)
  for e in zeropts:
    d[e[1]] += 1

  cnts = np.array(list(d.items()))
  mval = np.min(cnts[:, 1])

  sel = cnts[:, 1] < 1.01 * mval
  ang_ids = cnts[:, 0][sel]
  if 1:
    Z.plt.plot(*zip(*cnts))
    Z.plt.show()
  print(ang_ids)
  print(len(ang_ids))
  print(theta[int(min(ang_ids))])
  print(theta[int(max(ang_ids))])
  return

  plot_image(res)


def plot_image(img):
  G = K.GraphHelper()
  G.create_plot(images=[Z.Dataset2d(img)])
  G.run()

def align_project(ctx):
  angle = -(1.22869 - math.pi /2)
  img = cv2.imread(ctx.infile, 0)
  height, width = img.shape

  cx = math.cos(angle)
  sx = math.sin(angle)
  m = np.array([
      [cx, -sx, 0],
      [sx, cx, 0],
  ])
  img = cv2.warpAffine(img, m, (width, height))

  cv2.imwrite(ctx.outfile, img)


def read_md5_file(fname):
  res = {}
  for x in open(fname, 'r').readlines():
    x = x.strip().split()
    res[x[0]] = x[1]
  return res


def md5_inter(ctx):
  needle = read_md5_file(ctx.needle)
  haystack = read_md5_file(ctx.haystack)
  for k, v in needle.items():
    if k in haystack:
      print(f'{k} mapped in haystack to {haystack[k]}')


class IntervalTree:

  def __init__(self, poslist):
    poslist = list(sorted(set(poslist)))
    poslist.append(poslist[-1] + 1)
    self.mp = {}

    for i, v in enumerate(poslist):
      self.mp[v] = i

    self.root = IntervalNode.BuildFrom(poslist)

  def to_id(self, x):
    return self.mp[x]

  def add_score(self, r1, r2, v):
    assert r1 in self.mp
    assert r2 in self.mp
    if r1 > r2: r1, r2 = r2, r1

    if r1 == r2: self.root.add_score_one(r1, v)
    else:
      self.root.add_score(r1, r2, v / (r2 - r1))  # repartition lineique du score

  def get_score(self, r1, r2):
    return self.root.get_score(r1, r2)


class IntervalNode:

  def __init__(self, T=None, H=None):
    self.L = None
    self.R = None
    self.T = T
    self.H = H
    self.score = 0
    self.score_impulse = 0

  @staticmethod
  def BuildFrom(lst):
    if len(lst) == 1: return None
    res = IntervalNode(T=lst[0], H=lst[-1])
    if len(lst) == 2: return res

    mid = len(lst) // 2
    res.L = IntervalNode.BuildFrom(lst[:mid + 1])
    res.R = IntervalNode.BuildFrom(lst[mid:])
    return res

  def add_score_one(self, r1, v):
    if r1 < self.T or r1 >= self.H: return
    self.score_impulse += v
    if self.L:
      self.L.add_score_one(r1, v)
      self.R.add_score_one(r1, v)

  def add_score(self, r1, r2, v):
    r1 = max(r1, self.T)
    r2 = min(r2, self.H)
    if r1 >= r2: return
    self.score_impulse += v * (r2 - r1)
    if r1 == self.T and r2 == self.H:
      self.score += v
      return

    if self.L:
      self.L.add_score(r1, r2, v)
      self.R.add_score(r1, r2, v)

  def get_score(self, r1, r2):
    r1 = max(r1, self.T)
    r2 = min(r2, self.H)
    if r1 <= self.T and self.H <= r2:
      return self.score_impulse
    if r1 >= r2: return 0

    res = self.score * (r2 - r1)
    if self.L:
      res += self.L.get_score(r1, r2)
      res += self.R.get_score(r1, r2)
    elif r1 == self.T:
      res += self.score_impulse
    return res




def get_box(tb):
  tb = np.reshape(cmisc.flatten(tb), (-1, 2))
  minx = np.min(tb[:, 0])
  maxx = np.max(tb[:, 0])
  miny = np.min(tb[:, 1])
  maxy = np.max(tb[:, 1])
  return Z.Box(minx, miny, maxx, maxy)


def diff_to_poly_list(main, diff):
  rem = main - diff
  if isinstance(rem, Polygon):
    yield list(rem.exterior.coords)
    return

  for poly in rem.geoms:
    if isinstance(poly, shapely.geometry.LineString): continue
    yield poly.exterior.coords


def to_point(p):
  return np.array([p.X, p.Y])


class TileData:

  def __init__(self, ctx):
    data = cmisc.Attr.FromJson(ctx.infile)
    self.data = data
    self.ctx = ctx

  def get_tracks_from_tile(self, tile):
    tracks = list([list([to_point(p) for p in track]) for track in tile.tracks])
    for i, track in enumerate(tracks):
      track.append(track[0])  # closing the polyline
    if self.ctx.tracknums:
      tracks = list(track for i, track in enumerate(tracks) if i in self.ctx.tracknums)
    return list(
        cmisc.Attr(typ='track', polyline=np.array(track), trackid=i, box=get_box(track))
        for i, track in enumerate(tracks)
    )

  def set_tile(self, tile_id):
    self.tile_id = tile_id
    self.tile = self.data[tile_id]

    grid_xpos = cmisc.asq_query(
        self.tile.gridLines.Items
    ).where(lambda x: x.Orientation == 0).select(lambda x: x.Value).order_by().to_list()
    grid_ypos = cmisc.asq_query(
        self.tile.gridLines.Items
    ).where(lambda x: x.Orientation == 1).select(lambda x: x.Value).order_by().to_list()

    self.grid_xpos = grid_xpos
    self.grid_ypos = grid_ypos
    self.tracks = self.get_tracks_from_tile(self.tile)
    if 'image' in self.tile:
      img = self.tile.image.Image
      self.width = img.Width
      self.height = img.Height
      buf = Z.base64.b64decode(img.Pixels)
      self.img_data = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width))

    self.tile_box = get_box([x.polyline for x in self.tracks]).expand(1.3)

    self.grid_dx = np.mean(np.diff(grid_xpos))
    self.grid_dy = np.mean(np.diff(grid_ypos))
    self.analyse_gridx = np.arange(self.tile_box.xl, self.tile_box.xh, self.grid_dx / 10)
    self.analyse_gridy = np.arange(self.tile_box.yl, self.tile_box.yh, self.grid_dy / 10)
    self.vias = list(
        [
            make_via(to_point(x.Center), x.Diameter)
            for x in self.tile.get('vias', [])
        ]
    )

  def do1(self, trackdata):
    print()
    print('Processing', trackdata.trackid)
    track = trackdata.polyline
    n = len(track) - 1
    poly = Polygon(track[:n])
    box = get_box(track)

    perim = poly.length
    ratio = poly.area / perim
    center = poly.centroid
    xlist = set(x for x, y in track)
    ylist = set(y for x, y in track)
    itx = IntervalTree(xlist)
    ity = IntervalTree(ylist)

    sneg = 0
    spos = 0
    for i in range(n):
      p1 = track[i]
      p2 = track[i + 1]
      pdiff = p2 - p1
      if pdiff[1] > 0: spos += pdiff[1]
      else: sneg -= pdiff[1]
      itx.add_score(p1[0], p2[0], pdiff[1])
      ity.add_score(p1[1], p2[1], pdiff[0])

    ldx = self.grid_dx
    ldy = self.grid_dy
    dx = []
    dy = []
    print('FUU ', spos, sneg, ldx)

    for x in self.analyse_gridx:
      v = itx.get_score(x - ldx / 2, x) - itx.get_score(x, x + ldx / 2)
      dx.append(v)
    for y in self.analyse_gridy:
      v = ity.get_score(y - ldy / 2, y) - ity.get_score(y, y + ldy / 2)
      dy.append(v)
    res = cmisc.Attr(
        dx=dx, dy=dy, boxes=[], small=[], fail=[], rem=[], trackdata=trackdata, diff=[]
    )

    adx = np.abs(dx)
    ady = np.abs(dy)
    mx = max(adx) / perim
    my = max(ady) / perim
    if perim < ldx * 10:
      res.small = [track]
      return res  # track too small to compress

    if mx > 0.3:
      xcenter = self.analyse_gridx[np.argmax(adx)]
      xcenter = center.x
      xcenter2 = np.mean(self.analyse_gridx[np.argwhere(adx > np.max(adx) * 0.8)])
      diff = max(xcenter - box.xl, box.xh - xcenter)

      box_maxwidth = ldx * 1.3

      take_box = Z.Box(xcenter - ldx * 1.1 / 2, box.yl, xcenter + ldx * 1.1 / 2, box.yh)
      psum = np.zeros((2,))
      tot = 0
      for i in range(n):
        p1, p2 = track[i:i + 2]
        if p1 in take_box and p2 in take_box:
          s = np.linalg.norm(p2 - p1)
          psum += (p1 + p2) / 2 * s
          tot += s
        else:
          print('BAD ', p1, p2, take_box)
      psum /= tot
      real_xcenter = psum[0]
      real_xcenter = xcenter2
      maxbox = Z.Box(
          real_xcenter - box_maxwidth / 2, box.yl, real_xcenter + box_maxwidth / 2, box.yh
      )

      if box in maxbox:
        final_box = Z.Box(real_xcenter - ldx / 3, box.yl, real_xcenter + ldx / 3, box.yh)
        res.boxes.append(cmisc.Attr(final_box=final_box, polyline=final_box.poly(closed=True)))
      else:
        res.diff = list(diff_to_poly_list(poly, take_box.shapely))
        res.rem = [cmisc.Attr(polyline=take_box.poly(closed=True))]

    elif my > 0.3:
      ycenter = self.analyse_gridy[np.argmax(ady)]
      ycenter = center.y
      ycenter2 = np.mean(self.analyse_gridy[np.argwhere(ady > np.max(ady) * 0.8)])
      diff = max(ycenter - box.yl, box.yh - ycenter)

      box_maxheight = ldy * 1.3

      take_box = Z.Box(box.xl, ycenter - ldy * 1.1 / 2, box.xh, ycenter + ldy * 1.1 / 2)
      psum = np.zeros((2,))
      tot = 0
      for i in range(n):
        p1, p2 = track[i:i + 2]
        if p1 in take_box and p2 in take_box:
          s = np.linalg.norm(p2 - p1)
          psum += (p1 + p2) / 2 * s
          tot += s
        else:
          print('BAD ', p1, p2, take_box)
      psum /= tot
      real_ycenter = psum[1]
      real_ycenter = ycenter2

      maxbox = Z.Box(
          box.xl, real_ycenter - box_maxheight / 2, box.xh, real_ycenter + box_maxheight / 2
      )
      print('FIND ', ycenter2, ycenter, psum[1])
      print(box, maxbox)

      if box in maxbox:
        final_box = Z.Box(box.xl, real_ycenter - ldy / 3, box.xh, real_ycenter + ldy / 3)
        res.boxes.append(cmisc.Attr(final_box=final_box, polyline=final_box.poly(closed=True)))
      else:
        res.diff = list(diff_to_poly_list(poly, take_box.shapely))
        res.rem = [cmisc.Attr(polyline=take_box.poly(closed=True))]
    else:
      res.fail = [cmisc.Attr(polyline=track)]
    return res


def via_polyline(via, res=3):
  p = shapely.geometry.Point(via.pos)
  return cmisc.Attr(polyline=p.buffer(via.d / 2, resolution=res).exterior.coords, obj=via)


def render(ctx):
  td = TileData(ctx)
  td.set_tile(0)
  grid_lines_x = [[(xpos, td.tile_box.yl), (xpos, td.tile_box.yh)] for xpos in td.grid_xpos]
  grid_lines_y = [[(td.tile_box.xl, ypos), (td.tile_box.xh, ypos)] for ypos in td.grid_ypos]

  meshes = []

  if g_render_via_str in ctx.render_what:
    meshes.append(cmisc.Attr(lines=map(via_polyline, td.vias), color='red'))

  if g_render_track_str in ctx.render_what:
    tracks_mod = []
    tracks_rem = []
    tracks_fail = []
    tracks_small = []
    tracks_diff = []
    print(len(td.tracks))
    for track in td.tracks:

      res = td.do1(track)

      def add_trackdata(lst):
        for x in lst:
          if isinstance(x, cmisc.Attr):
            x.obj = track
          else:
            x = cmisc.Attr(polyline=x, obj=track)
          yield x

      tracks_mod.extend(add_trackdata(res.boxes))
      tracks_rem.extend(add_trackdata(res.rem))
      tracks_fail.extend(add_trackdata(res.fail))
      tracks_small.extend(add_trackdata(res.small))
      tracks_diff.extend(add_trackdata(res.diff))

    #meshes.append(cmisc.Attr(lines=td.tracks, color=K.vispy_utils.Color('green', alpha=0.3)))
    meshes.append(cmisc.Attr(lines=tracks_mod, color='b'))
    meshes.append(cmisc.Attr(lines=tracks_diff, color='purple'))
    meshes.append(cmisc.Attr(lines=tracks_small, color='gray'))
    meshes.append(cmisc.Attr(lines=tracks_rem, color='orange'))
    meshes.append(cmisc.Attr(lines=tracks_fail, color='red'))

  if g_render_grid_str in ctx.render_what:
    meshes.append(cmisc.Attr(lines=grid_lines_x, color='white'))

  if g_render_image_str in ctx.render_what and hasattr(td, 'img_data'):
    meshes.append(cmisc.Attr(images=[td.img_data], cmap='gray'))

  if 0:
    transform = K.vispy_utils.transforms.MatrixTransform()
    transform.translate(td.tile_box.get_grid(1))
    mesh_tracks_mod.transform = transform

  render_for_meshes(meshes)





def test_tracks(ctx):
  td = TileData(ctx)
  td.set_tile(ctx.tile_id)
  ddx = []
  ddy = []

  if 0:
    largest = max(td.tracks, key=lambda x: x.box.width)
    print(largest)
    print(largest.trackid)
    return

  for track in td.tracks:
    res = td.do1(track)
    ddx.append(res.dx)
    ddy.append(res.dy)
  ddx = np.array(ddx).T
  ddy = np.array(ddy).T

  df = Z.pd.DataFrame(ddx, index=td.analyse_gridx)
  print(df.columns)
  for col in df.columns:
    dfc = df[col]
    dfc.plot(label=col, legend=True)
  Z.plt.show()
  if 0:

    df = Z.pd.DataFrame(ddy, index=ta.grid_ypos)
    print(df.columns)
    for col in df.columns:
      df[col].plot()
    Z.plt.show()



def grid_sample(nw, nh):
  X, Y = np.meshgrid(np.linspace(0, 1, nw), np.linspace(0, 1, nh))
  X += np.random.uniform(-1/nw, 1/nw, (nh, nw)) * 0.5
  Y += np.random.uniform(-1/nh, 1/nh, (nh, nw)) * 0.5
  return np.stack((X.ravel(), Y.ravel()), axis=-1)



def gather_data(img, center, box_dim, active_box, visualize=False, **kwargs):
  box_dim = np.ceil(box_dim).astype(int)
  box = Z.Box(center=center, dim=box_dim, is_int=1)
  if box not in active_box: return None

  img_box = img[box.get_window()]
  learn_img = Z.skimage.transform.resize(img_box, g_target_shape)
  vis=None

  if visualize:
    features,  vis = Z.skimage.feature.hog(learn_img, cells_per_block=(2,2), visualise=True)
  else:
    features = Z.skimage.feature.hog(learn_img, cells_per_block=(2,2), visualise=False)
    learn_img=None

  assert np.any(np.isnan(features)) == False, f'{center} {box_dim} {features}'
  return cmisc.Attr(box=box, features=features, vis=vis, learn_img=learn_img, **kwargs)

def get_active_box(img):
  return Z.Box(low=(0,0), dim=img.shape[::-1]).expand(0.99)



def compute_via_stats(vias):
  d = []
  for via in vias:
    d.append(via.d)
  d_target = np.quantile(d, 0.75)
  return cmisc.Attr(d_target=d_target,)

def gather_via_training_data(vias, img_data):
  via_stats = compute_via_stats(vias)
  d_target = via_stats.d_target

  active_box = get_active_box(img_data)
  box_base_dim = np.array((d_target, d_target))

  via_points = list([cmisc.Attr(geo=Point(x.pos)) for x in vias])
  qtree = QuadTree(via_points, max_objs=10)


  boxes = []
  factor_range = (2, 3)
  factors_per_via = 5


  nw, nh = active_box.dim.astype(int) // 40
  nsel = 0.9
  ntarget = nw * nh * 3
  bad_items = []

  active_to_real_mat = active_box.mat_from()
  while len(bad_items) < ntarget:
    grid = grid_sample(nw, nh)
    sel_items = np.random.choice(len(grid), size=int(len(grid) * nsel), replace=False)
    for sel_id in sel_items:
      pt = grid[sel_id]
      pt = Z.mat_apply2d(active_to_real_mat, pt)
      factor =  np.random.uniform(factor_range[0], factor_range[1])
      if pt not in active_box: continue
      q = qtree.query(Point(pt))
      assert q is not None
      if q.dist < d_target: continue
      res = gather_data(img_data, pt, box_base_dim * factor, active_box, label=0)
      if res is not None: bad_items.append(res)


  good_items =[]

  for via in vias:
    for factor in np.random.uniform(factor_range[0], factor_range[1], size=factors_per_via):
      box_dim = box_base_dim * factor
      jitter_space = box_dim / 5
      center = via.pos + np.random.uniform(-jitter_space, jitter_space)
      res = gather_data(img_data, center, box_dim, active_box, label=1)
      if res is not None:
        good_items.append(res)

  res = cmisc.Attr(
    good=good_items,
    bad=bad_items,
    data=cmisc.Attr(d_target=d_target, active_box=active_box, factor_range=factor_range)
  )
  return res

def file_info(ctx):
  td = TileData(ctx)
  print(len(td.data))

def test_vias(ctx):
  td = TileData(ctx)


  items = []
  extraction_data=[]
  if not ctx.tile_ids: ctx.tile_ids = list(range(len(td.data)))
  for idx in ctx.tile_ids:
    print(f'Processing tile {idx}')
    td.set_tile(idx)
    rx = gather_via_training_data(td.vias, td.img_data)
    rx.data.tile_id = idx
    extraction_data.append(rx.data)
    items.extend(rx.good)
    items.extend(rx.bad)

  ser_data = cmisc.Attr(items=items, td=td, extraction_data=extraction_data)


  Z.pickle.dump(ser_data, open(ctx.outfile, 'wb'))



  if 0:
    plot_image_hist(td.img_data)

    meshes = []
    meshes.append(cmisc.Attr(lines=map(via_polyline, td.vias), color='red'))
    meshes.append(cmisc.Attr(lines=boxes, color='blue'))
    meshes.append(cmisc.Attr(images=[td.img_data], cmap='gray'))
    render_for_meshes(meshes)


def test_via_learn(ctx):
  data = Z.pickle.load(open(ctx.infile, 'rb'))
  td = data.td


  train_x = []
  train_y = []
  train = []
  for item in data['items']:
    train_x.append(item.features)
    train_y.append(item.label)

  train_x = Z.pd.DataFrame.from_records(train_x)
  train_y = Z.pd.Series(train_y)
  scaler = Z.sklearn.preprocessing.MinMaxScaler()
  train_x = scaler.fit_transform(train_x)
  #test_x = scaler.transform(test_x) #

  sub_val_data = Z.sklearn.model_selection.train_test_split(train_x, train_y, random_state=0, test_size=0.2)
  train_sub = sub_val_data[0], sub_val_data[2]
  train_sub_val = sub_val_data[1], sub_val_data[3]



  print(train_x)
  print(train_y)
  params = dict(learning_rate=(0.1,0.2,0.3, 0.5,0.7), n_estimators=(50, 100, 200, 300, 500))

  gb = Z.sklearn.ensemble.GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
  #gb.fit(*train_sub)
  tuning = Z.sklearn.model_selection.GridSearchCV(estimator=gb, cv=5, param_grid=params)
  tuning.fit(train_x, train_y)
  print(tuning)
  print(tuning.best_score_)

  print(tuning.best_params_)

  res = cmisc.Attr(estimator=tuning.best_estimator_, td=td, extraction_data=data.extraction_data)
  Z.pickle.dump(res, open(ctx.outfile, 'wb'))



def get_annotations(img, model, extraction_params):
  factor = extraction_params.factor_range[0]
  d_target = extraction_params.d_target
  box = np.array((1,1)) * factor * d_target
  active_box = extraction_params.active_box

  xdata = []
  cndlist = []

  for x, y in active_box.get_grid_pos(box*flags.box_step_factor):
    dx =  gather_data(img, (x,y), box, active_box)
    if dx is None: continue
    cndlist.append(dx)
    xdata.append(dx.features)

  pred = model.predict(xdata)
  for pos in np.argwhere(pred==1).ravel():
    yield cndlist[pos].box


def make_via(pos, d):
  return cmisc.Attr(pos=pos, d=d, typ='via')


def test_annotation(ctx):
  data = Z.pickle.load(open(ctx.infile, 'rb'))
  model = data.estimator
  td = data.td
  for extraction_params in data.extraction_data:
    td.set_tile(extraction_params.tile_id)

    annotations = get_annotations(td, model, extraction_params)
    print(annotations)
    break



def test_model(ctx):
  data = Z.pickle.load(open(ctx.infile, 'rb'))
  model = data.estimator
  td = data.td
  active_box = get_active_box(td.img_data)

  class RunCtx:
    def __init__(self):
      self.mouse_pos = None
      self.p1 = None
      self.p2 = None
      self.cur_objs = None

    def mouse_move(self, ev):
      self.mouse_pos = vctx.screen_to_world(ev.pos)

    def key_press(self, ev):
      if ev.key == '1':
        self.p1 = self.mouse_pos
      if ev.key == '2':
        self.p2 = self.mouse_pos

      if ev.key == '3':
        self.p1 = None
        self.p2 = None
        vctx.remove_objs(self.cur_objs)
        self.cur_objs = None

      if self.p1 is not None and self.p2 is not None:
        if self.cur_objs:
          vctx.remove_objs(self.cur_objs)
        box = Z.Box.FromPoints([self.p1, self.p2])
        print('QUERYING BOX ', box)
        dx = gather_data(td.img_data, box.center, box.dim, active_box)
        val, = model.predict([dx.features])
        print('Predict result >> ', val)
        col = 'rg'[val]
        self.cur_objs = vctx.plot_meshes(cmisc.Attr(lines=[box.poly(closed=1, z_coord=-1)], color=col))
        print(self.cur_objs)
        self.p1 = None
        self.p2 = None



  annotations = list(get_annotations(td.img_data, model, data.extraction_data[0]))

  meshes = []
  meshes.append(cmisc.Attr(lines=map(via_polyline, td.vias), color='blue'))
  meshes.append(cmisc.Attr(lines=annotations, color='orange'))

  if 0:
    rx = gather_via_training_data(td.vias, td.img_data)
    good = list([x.box.poly(closed=1) for x in rx.good])
    bad = list([x.box.poly(closed=1) for x in rx.bad])
    meshes.append(cmisc.Attr(lines=good[:100], color='green'))
    meshes.append(cmisc.Attr(lines=bad[:100], color='red'))


  meshes.append(cmisc.Attr(images=[td.img_data], cmap='gray'))


  vctx = K.vispy_utils.VispyCtx(display_status=1)
  rctx = RunCtx()
  vctx.canvas.events.key_press.connect(rctx.key_press)
  vctx.canvas.events.mouse_move.connect(rctx.mouse_move)
  render_for_meshes(meshes, click=False, vctx=vctx)


def test_hog(ctx):
  td = TileData(ctx)
  td.set_tile(0)

  via_stats = compute_via_stats(td.vias)
  dd = via_stats.d_target

  cx = K.vispy_utils.ImageComposer()
  active_box = get_active_box(td.img_data)
  pos=0
  for via in td.vias:
    dx = gather_data(td.img_data, via.pos, (dd*2, dd*2), active_box, visualize=True)
    if dx is None: continue
    vis2 = Z.skimage.exposure.rescale_intensity(dx.vis, in_range=(0,10)) * 100
    cx.add_img(cmisc.Attr(data=vis2, box=Z.Box(low=(0, 2*pos), dim=(1, 1))))
    cx.add_img(cmisc.Attr(data=dx.learn_img, box=Z.Box(low=(2, 2*pos), dim=(1, 1))))
    pos+=1

    break
  K.plot_img(cx.render())

def test_interval_tree(ctx):
  it = IntervalTree([0, 1, 3, 4])
  #it.add_score(0, 1, 0.5)
  it.add_score(0, 3, 2)
  print('======')
  print(it.get_score(0.5, 2))
  print('======')
  print(it.get_score(-1, 2))


def test_quadtree(ctx):
  lines = []
  lines.append(cmisc.Attr(geo=Z.Line((1, 1), (4, 1))))
  lines.append(cmisc.Attr(geo=Z.Line((3, 1.5), (5, 2.5))))
  lines.append(cmisc.Attr(geo=Z.Line((4.5, 0), (4.5, 3.5))))
  lines.append(cmisc.Attr(geo=Point(0.5, 0.5)))
  q = QuadTree(lines, max_objs=2)
  p1 = Point(10, 0.5)
  for line in lines:
    print(line.geo.distance(p1))
  print(q.query(p1).geo)
  print(q.query(Point((0, 0))).best.geo)
  l0 = lines[0].geo.shapely.envelope


def main():
  ctx = Attributize()
  if 'all' in flags.render_what:
    flags.render_what = [
        g_render_via_str, g_render_track_str, g_render_grid_str, g_render_image_str
    ]
  if flags.theta_range: ctx.theta_range = Z.Range1D(*flags.theta_range, is_int=0)
  ActionHandler.Run(ctx)


app()
