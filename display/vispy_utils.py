#!/usr/bin/env python
import sys

from chdrft.cmds import CmdsList
from chdrft.graphics.helper import stl_to_meshdata
from chdrft.graphics.loader import stl_parser
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, flatten
from chdrft.utils.swig import swig
from vispy.geometry.meshdata import MeshData
from vispy.scene.cameras import TurntableCamera, PanZoomCamera
from vispy.scene.visuals import Markers, Text
from vispy.scene.visuals import Mesh, Line, Isocurve, Image
from vispy.visuals import transforms
from vispy.visuals.collections import PointCollection
from vispy.color import Color
import numpy as np
import numpy.random
import pprint
import vispy, vispy.app, vispy.scene, vispy.visuals
from asq.initiators import query as q
import time
from vispy.color import get_colormap

mathgame = None
common = None
tc = None
fc = None
import glog

global flags, cache
flags = None
cache = None


def vec_pos_to_list(vecpos):
  return list([vecpos[i] for i in range(len(vecpos))])


def init(ctx):
  if not ctx.noopa:
    from opa.math.game.proto import common_pb2
    global mathgame, common, tc, fc
    mathgame = swig.opa_math_game_swig
    common = swig.opa_common_swig
    tc = mathgame.FaceCollection(True)
    fc = mathgame.FaceCollection(False)
    ctx.mathgame = mathgame
    ctx.common = common
    ctx.tc = tc
    ctx.fc = fc


def setup_args(parser):
  parser.add_argument('--stl_file', type=str)
  parser.add_argument('--infile', type=str)
  parser.add_argument('--outfile', type=str)
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--nogui', action='store_true')
  parser.add_argument('--improve', action='store_true')
  parser.add_argument('--noopa', action='store_true')
  parser.add_argument('--round_ts', type=float, default=4)


def args(parser):
  clist = CmdsList().add(test).add(test2).add(clean_mesh).add(test_atomize).add(plot).add(
      best_box).add(test_atomize_heavy).add(test_box)
  clist.add(test_intersection)
  clist.add(test_atomize_solution)
  clist.add(test_isocurve)
  clist.add(test_facegraph)
  clist.add(plot_meshlist)
  clist.add(meshlist_to_openscad)

  ActionHandler.Prepare(parser, clist.lst, init)
  setup_args(parser)


def get_polyline_from_edges(edges, pc, color):
  pts = []
  colors = []
  prev = None
  color = Color(color).rgba

  for edge in edges:
    if prev is not None:
      pts.append(prev)
      pts.append(pc[edge.from_])
      colors.append((0, 0, 0, 0))
      colors.append((0, 0, 0, 0))
    pts.append(pc[edge.from_])
    pts.append(pc[edge.to])
    colors.append(color)
    colors.append(color)
    prev = pc[edge.to]
  print(pts)
  return [Attributize(polyline=pts, colors=colors, width=2)]


def get_non_manifold_plot_data(graph_data):
  pc = graph_data.points
  faces = vec_pos_to_list(graph_data.faces)
  res = Attributize()
  g = graph_data.graph

  bad_edges = []
  for edge in g.edges:
    if g.get_edge_count(edge.to, edge.from_, False) != 1:
      bad_edges.append(edge)

  if len(bad_edges) == 0:
    return None

  color = 'r'
  res.lines = get_polyline_from_edges(bad_edges, pc, color)

  bad_point_ids = set(flatten([(edge.to, edge.from_) for edge in bad_edges]))
  res.points = [pc[x] for x in bad_point_ids]
  res.points_size = 20
  res.color = color
  return res


def get_mesh_data_from_graph(graph_data, want_lines=True, want_points=True, **kwargs):

  pc = graph_data.points
  faces = vec_pos_to_list(graph_data.faces)

  for face_vertices in faces:
    assert len(face_vertices) == 3
  faces = np.array(faces)
  print(faces)
  polylines = []

  if want_points:
    pc = list([pc[i] for i in range(len(pc))])
    pc = np.array(pc)

  if want_lines:
    cc = graph_data.graph.make_bidirectional().split_cc()

    polylines = []
    for x in cc:
      polyline = []
      walk = x.get_cover_walk_dumb()
      for y in walk:
        assert y >= 0 and y < len(pc)
      polyline.extend([pc[y] for y in walk])
      polylines.append(polyline)

  return Attributize(mesh=MeshData(vertices=pc, faces=faces), lines=polylines, points=pc, **kwargs)


def get_mesh_data(mesh, want_lines=True, want_points=True, **kwargs):

  mesh.compute_remap()
  pc = mesh.point_cloud()

  faces = []
  for fid in mesh.faces():
    ids = mesh.get_face_ids(fid)
    ids = list(ids)
    for c in ids:
      assert c >= 0 and c < len(pc)
    # if non triangular faces, assume they are convex. Poor man triangulation
    for i in range(len(ids) - 2):
      faces.append([ids[0], ids[i + 1], ids[i + 2]])
  faces = np.array(faces)
  polylines = []

  if want_points:
    pc = list([pc[i] for i in range(len(pc))])
    pc = np.array(pc)

  if want_lines:
    graph = mesh.compute_graph()
    cc = graph.split_cc()

    polylines = []
    if 0:
      for edge in mesh.edges():
        start = mesh.get_vertex_attr(mesh.start(edge)).pos
        end = mesh.get_vertex_attr(mesh.end(edge)).pos
        polylines.append(np.array([start, end]))
    else:
      print(len(cc))
      for x in cc:
        polyline = []
        walk = x.get_cover_walk_dumb()
        walk = list([x.inormv(y) for y in walk])
        for y in walk:
          assert y >= 0 and y < len(pc)
        print(walk)
        polyline.extend([pc[y] for y in walk])
        polylines.append(polyline)

  return Attributize(
      mesh=MeshData(
          vertices=pc,
          faces=faces,
          face_colors=np.random.rand(len(faces), 4),),
      lines=polylines,
      points=pc,
      **kwargs)


class VispyCtx:

  def __init__(self, display_status=False):
    vispy.app.use_app('pyqt5')
    #vispy.app.set_interactive()
    self.canvas = vispy.scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    self.view = self.canvas.central_widget.add_view()
    self.cur_objs = []
    if display_status:
      self.canvas.events.mouse_move.connect(self.on_mouse_move)
      self.status_text = Text(
          'where info goes', color='w', anchor_x='left', parent=self.view, pos=(20, 30))

  def plot_meshes(self, *mesh):
    lst = []
    for x in mesh:
      if x is None: continue
      lst.append(
          add_mesh_data_to_view(x, self.view, want_lines=True, want_points=False, want_faces=False))
    for x in mesh:
      if x is None: continue
      lst.append(add_mesh_data_to_view(x, self.view, want_lines=False))
    return lst

  def on_mouse_move(self, ev):
    tsf = self.view._scene.transform
    pos = tsf.imap(ev.pos)
    self.status_text.text = str(pos[:2])

  def run(self, cam=True, center=None):
    if cam:
      self.view.camera = TurntableCamera(fov=70, center=center)
    self.canvas.show()
    #self.view.camera.set_range()
    vispy.app.run()

  def remove_objs(self, objs):
    for obj in flatten([x.objs for x in objs]):
      obj.parent = None
    objs.clear()

  def free_objs(self):
    self.remove_objs(self.cur_objs)
    self.cur_objs = []


def add_mesh_data_to_view(mdata,
                          view,
                          want_faces=True,
                          want_points=True,
                          want_lines=True,
                          transform=transforms.MatrixTransform()):
  color = mdata.get('color', 'g')
  if isinstance(color, str) and color == 'random': color = np.random.random(3).tolist()
  else: color = Color(color).rgb.tolist()

  res = Attributize()
  res.lines = []
  res.objs = []
  if want_lines and 'lines' in mdata:
    for polyline in mdata.lines:
      if isinstance(polyline, Attributize):
        width = polyline.get('width', 1)
        method = 'gl'
        if width > 1: method = 'gl'
        l = Line(
            pos=np.array(np.array(polyline.polyline)),
            antialias=True,
            method=method,
            color=np.array(polyline.colors),
            width=width,)
      else:
        l = Line(pos=np.array(np.array(polyline)), antialias=False, method='gl', color=color)
      l.transform = transform
      view.add(l)
      res.lines.append(l)
      res.objs.append(l)

  if want_points and mdata.get('points', None) is not None:
    scatter = Markers()
    #mdata.points = np.array((
    #    (10.928140, -51.417831, -213.253723),
    #    (0.000000, -46.719570, -205.607208),
    #    (0.000000, -53.499737, -215.031876),
    #    (0.000000, -69.314308, -223.780746),
    #    (0.000000, -89.549263, -170.910568),))
    #mdata.points = np.array(((-12.138942,-55.812309,-217.007050),(10.928140,-51.417831,-213.253723),(-7.289741,-43.585541,-200.506531)))
    points_color = mdata.get('points_color', color)
    points_size = mdata.get('points_size', 10)
    points = np.array(mdata.points)

    #print('PLOTTING ', points, points_size)
    if len(points) == 0: return
    scatter.set_data(points, edge_color=None, face_color=points_color, size=points_size)
    scatter.transform = transform
    view.add(scatter)
    res.objs.append(scatter)
    res.points = scatter

  if want_faces and 'mesh' in mdata:
    mesh = Mesh(meshdata=mdata.mesh, color=color + [0.7])
    mesh.transform = transform
    view.add(mesh)
    res.mesh = mesh
    res.objs.append(mesh)
  return res


def create_mesh_visual_from_stl(stl_data, **data):
  return Mesh(meshdata=stl_to_meshdata(stl_data), **data)


def clean_mesh(ctx):
  in_fc = mathgame.FaceCollection(True)
  mesh = in_fc.load_stl(ctx.stl_file).to_mesh()

  tc = mathgame.FaceCollection(True)
  cclist = mesh.split_connected()
  for target in cclist:
    target = target.correct_orientation()
    tc.push(target.to_faces())
  print(ctx.outfile)
  tc.write_stl(ctx.outfile)


def plot_mesh_dirs(mesh, correct=False):
  cclist = mesh.split_connected()
  for target in cclist:
    if correct:
      target = target.correct_orientation()
    graph = target.to_faces().to_mesh_builder().get_graph()
    res = graph.graph.split_cc()

    target.compute_remap()
    fid = target.faces()[0]
    pts = []
    for c in target.get_face_ids(fid):
      pts.append(target.get_vertex_attr(c).pos)

    mdata = get_mesh_data(target)
    #mdata.points = np.array(pts)
    #mdata.points_color=['r', 'g', 'b']
    print('JAMBON >> ', target.is_correctly_oriented())
    yield mdata


def test(ctx):
  tc = mathgame.FaceCollection(True)
  to_plot = []
  points = [(2.036540, -3.363836, -0.534037)]
  box = mathgame.BoxSpec_3DPython()
  box.corner = (0.545343, -3.325997, -0.959748)
  box.set(0, (0.869472, -0.388906, 0.942803))
  box.set(1, (0.328980, -0.406752, -0.471176))
  box.set(2, (3.443821, 4.374201, -1.371605))
  print(box.in_(points[0]))
  return

  if 0:
    meshes = common_pb2.MeshList.FromString(open("/tmp/debug.out", "rb").read())
    meshes2 = list([Attributize(raw=x, stl=stl_parser(data=x.stl_content)) for x in meshes.mesh])

    print([x.raw.name for x in meshes2])
    deb_mesh = q(meshes2).where(lambda x: x.raw.name == 'DEBUG').to_list()
    assert len(deb_mesh) == 1
    deb_mesh = deb_mesh[0]
    content = deb_mesh.raw.stl_content
    print(type(content))
    print(len(content))
    print(content)
    d1 = tc.clear().load_stl_from_data(content).to_mesh_builder()
    print(type(d1))
    graph_data = d1.get_graph()
    res = graph_data.graph.split_cc()
    print(len(res))
    assert graph_data.graph.is_connected()

    lines = []
    walk = graph_data.graph.get_cover_walk()
    pts = graph_data.points
    pts = list([pts[i] for i in range(len(pts))])
    for y in walk:
      assert y >= 0 and y < len(pts)
    lines.append([pts[y] for y in walk])
    to_plot.append(Attributize(lines=lines, points=np.array(pts)))

  else:
    hp = mathgame.HyperPlaneSpec()
    hp.plane.v = 0
    hp.plane.dir = (0, 0, 1)
    print(hp.plane.v)
    print(hp.plane.dir)
    tc.clear()

    curmesh = tc.load_stl(ctx.stl_file).triangulate().to_mesh()
    print(curmesh.str().decode())
    pc = curmesh.point_cloud()
    cclist = curmesh.split_connected()

    if 1:
      for target in cclist:
        graph = target.to_faces().to_mesh_builder().get_graph()
        res = graph.graph.split_cc()
        print(len(res))
        assert graph.graph.is_connected()
        res = mathgame.mesh_cut_plan(target, hp)
        for x in res:
          to_plot.append(get_mesh_data(x))
          break
    else:
      to_plot.extend(plot_mesh_dirs(curmesh, correct=False))

    curbox = mathgame.compute_aabb_box_norot(pc).to_box()
    bbox = mathgame.compute_best_box(pc)
    tc2 = mathgame.FaceCollection(True)
    #tc2 = mathgame.FaceCollection()
    print(type(curbox))
    print(type(bbox))
    box_data = get_mesh_data(tc2.add_box(curbox).to_mesh())
    bbox_data = get_mesh_data(tc2.clear().add_box(bbox).to_mesh())
    to_plot.append(get_mesh_data(tc.clear().add_plane(hp.plane, (0, 0, 0), 1000).to_mesh()))

  #for i in range(len(pc)):
  #  print(pc[i])
  #print(lst)
  ctx = VispyCtx()

  ctx.plot_meshes(*to_plot)
  #res2 = add_mesh_data_to_view(box_data, view)
  #res3 = add_mesh_data_to_view(bbox_data, view)
  #res_plane = add_mesh_data_to_view(plane_mesh, view)

  #stl = stl_parser(filename=ctx.stl_file)

  #mesh.transform.translate((30, 20, 600))

  ctx.run()


def plot(ctx):
  fc = mathgame.FaceCollection(True)
  print('FUU ', ctx)
  mb = fc.clear().load_stl(ctx.stl_file).to_mesh_builder()

  graph_data = mb.get_graph()
  print(graph_data.graph.edges[0])
  print(type(graph_data.graph.edges[0]))
  print(graph_data.graph.edges[0].from_)
  data = [get_mesh_data_from_graph(graph_data, want_lines=True, want_points=True)]
  data.append(get_non_manifold_plot_data(graph_data))
  #  data[0].points = ((-0.197085,-12.849998,0.827523),(-0.245660,-4.668231,0.813985),
  #(-0.094095,-4.788983,0.845746),
  #(-0.190819,-4.842065,0.828393),
  #(-0.317571,-12.849995,0.789017),
  #      )
  print('FUUU HEHERE')

  ctx = VispyCtx()
  ctx.plot_meshes(*data)
  ctx.run()


def test2(ctx):
  fc = mathgame.FaceCollection(True)
  mathgame.add_icosahedron(fc)
  mesh = fc.to_mesh()
  mesh = fc.clear().load_stl(ctx.stl_file).triangulate().to_mesh()
  mesh.whiten()
  print(mesh.str())
  data = list(plot_mesh_dirs(mesh))

  ctx = VispyCtx()
  ctx.plot_meshes(*data)
  print(data[0].points)
  ctx.run()
  #t =swig.opa_math_common_swig.bignum()
  #print(t)
  #tmp = swig.opa_common_swig
  pass


def best_box(ctx):
  mesh = tc.load_stl(ctx.stl_file).triangulate().to_mesh()
  box = mathgame.compute_best_box(mesh.point_cloud())
  ctx = VispyCtx()

  box_mesh = fc.clear().add_box(box).to_mesh()
  data = []
  data.append(get_mesh_data(mesh))
  data.append(get_mesh_data(box_mesh))
  data[-1].color = 'r'
  ctx.plot_meshes(*data)
  ctx.run()


def get_assignment_data(assignment):
  data = []
  for cluster_data in assignment.cluster_data:
    mx = fc.clear().add_box(cluster_data.box).to_mesh()
    data.append(get_mesh_data(mx))
    data[-1].color = 'r'
  return data


def load_stl(fil):
  mesh = tc.clear().load_stl(fil).triangulate().to_mesh()
  mesh.whiten()
  return mesh


def test_atomize_heavy(ctx):

  mesh = load_stl(cxt.stl_file)
  cclist = mesh.split_connected()
  mesh0 = cclist[0]
  for meshx in cclist:
    if len(mesh0.vertices()) < len(meshx.vertices()):
      mesh0 = meshx
  atomizer = mathgame.KAtomizer(mesh0, 7)

  x = atomizer.initial_assignment()

  search_params = mathgame.KAtomizer_SearchParams()
  ts = mathgame.TimeStoppingCriteria(int(20e6))
  #ts = mathgame.RoundStoppingCriteria(1)
  search_params.stop_criteria = ts
  search_params.T0 = 0.5
  search_params.state = 0
  atomizer.improve_assignment(x, search_params)

  data = list(plot_mesh_dirs(mesh0))
  data.extend(get_assignment_data(x))
  ctx = VispyCtx()
  ctx.plot_meshes(*data)
  for cl in x.cluster_data:
    print(cl.faces)
  ctx.run()


def test_atomize_solution(ctx):
  mesh = tc.clear().load_stl(ctx.stl_file).triangulate().to_mesh()
  cclist = mesh.split_connected()
  mesh0 = cclist[0]
  for meshx in cclist:
    if len(mesh0.vertices()) < len(meshx.vertices()):
      mesh0 = meshx
  atomizer = mathgame.KAtomizer(mesh0, 7)

  repartition = [
      (89, 110, 78, 89, 497, 512, 527, 510, 525, 110, 89, 94, 93, 121, 105, 111, 538, 93, 127, 106,
       111, 108, 111, 111, 123, 124, 111, 109, 108, 138, 127, 124, 138, 109, 96, 106, 106, 122, 92,
       106, 106, 80, 105, 108, 80),
      (414, 434, 415, 433, 432, 447, 461, 475, 476, 462, 435, 448, 460, 446, 413, 430, 429, 474,
       459, 490, 504, 473, 489, 472, 491, 505, 458, 450, 444, 479, 465, 396, 398, 394, 378, 395,
       426, 451, 445, 503, 519, 521, 506, 520, 412, 418, 438, 431, 363, 377, 393, 487, 488, 502,
       440, 408, 464, 478, 492, 419, 428, 427, 437, 390, 410, 358, 364, 379, 484, 469, 409, 400,
       457, 443, 517, 518, 540, 496, 362, 406, 542, 381, 530, 399, 482, 425, 405, 346, 404, 529,
       347, 386, 424, 385, 368, 369, 449, 332, 313, 330, 421, 351, 515, 335, 511, 316, 370, 352,
       452, 387, 300, 317, 397, 454, 350, 383, 331, 337, 436, 485, 336, 384, 367, 416, 439, 420,
       423, 403, 402, 422, 536, 467, 533, 522, 539, 441, 453, 477, 455, 470, 456, 463, 545, 535,
       526, 543, 493, 532, 508, 524, 534, 507, 523, 417, 544, 501, 471, 442, 298, 281, 486, 516,
       500, 341, 342, 499, 481, 466, 480, 531, 541, 468, 483, 528, 513, 498, 514, 494, 509, 495,
       537, 53, 318, 339),
      (328, 329, 345, 360, 344, 361, 312, 375, 376, 297, 311, 296, 343, 327, 278, 279, 259, 322,
       340, 307, 290, 321, 271, 222, 348, 299, 314, 333, 260, 280, 243, 282, 407, 401, 389, 372,
       382, 388, 366, 380, 349, 356, 353, 359, 334, 315, 365, 371, 355, 354, 357, 411, 391, 338,
       373, 392, 374, 242, 262, 241, 301),
      (163, 151, 180, 133, 145, 162, 128, 115, 98, 114, 169, 199, 179, 198, 181, 188, 207, 163, 263,
       118, 218, 201, 284, 208, 200, 209, 228, 238, 220, 219, 239, 226, 186, 213, 67, 264, 167, 285,
       227, 146, 223, 149, 148, 168, 187, 131, 146, 235, 289, 254, 135, 160, 270, 324, 320, 193,
       214, 240, 257, 325, 203, 163, 306, 308, 230, 229, 250, 210, 326, 268, 310, 287, 206, 304,
       189, 202, 249, 183, 258, 291, 295, 292, 294, 272, 288, 309, 166, 305, 174, 176, 256, 194,
       269, 234, 323, 215, 273, 277, 276, 293, 236, 253, 244, 168, 255, 217, 275, 196, 303, 237,
       274, 286, 221, 224, 231, 245, 246, 265, 184, 168, 204, 225, 267, 266, 248, 251, 252, 233,
       247, 232, 116, 144, 164, 170, 83, 97, 81, 116, 136, 178, 302, 82, 159, 161, 154, 117, 216,
       82, 99, 171, 113, 185, 173, 197, 144, 125, 153, 173, 205, 185, 141, 131, 157, 156, 112, 112,
       112, 130, 129, 164, 129, 144, 164, 197, 144, 261, 319, 283, 154, 175, 140, 175, 165, 192,
       132, 170, 150, 211, 100, 118, 212, 117, 191, 177, 182, 190, 118, 130, 158, 195, 182, 182,
       147, 152, 132, 144, 172, 150, 143, 155, 147, 100, 134, 139, 142, 139, 126, 142, 126, 143,
       180),
      (61, 13, 14, 7, 7, 24, 21, 13, 61, 21, 45, 22, 14, 34, 22, 7, 60, 11, 45, 24, 32, 12, 2, 15,
       11, 2, 33, 46, 47, 6, 23, 46, 23, 13, 15, 46, 44, 13, 2, 44, 14, 32, 22),
      (71, 25, 25, 35, 36, 26, 35, 51, 69, 37, 25, 51, 51, 79, 42, 48, 71, 26, 71, 86, 30, 26, 36,
       62, 49, 36, 62, 95, 75, 88, 75, 95, 64, 66, 62, 102, 76, 70, 74, 64, 77, 64, 63, 52, 74, 90,
       50, 50, 120, 64, 120, 120, 104, 104, 104, 65, 103, 76, 91, 70, 107, 87, 65, 102, 107, 104,
       103, 65, 42, 95, 103, 17, 90, 62, 29, 29, 28, 71, 71, 39, 53, 30, 120, 87, 57, 28, 39, 28,
       28, 39, 39, 56, 56, 54, 56, 54, 54, 56, 55, 40, 30, 55, 19, 30, 30, 19, 20, 30, 57, 20, 57,
       73, 57, 19, 29, 42, 41, 29, 58, 101, 137, 101, 137, 119, 101, 101, 85, 85, 31, 58, 43, 72,
       31, 43, 58, 72, 43, 72, 58, 43, 31, 59, 31, 43, 88, 84, 84, 84, 84),
      (3, 8, 5, 3, 1, 4, 4, 8, 19, 134, 53, 53, 134, 27, 27, 38, 53, 27, 68, 83, 83, 68, 27, 83, 4,
       0, 0, 0, 5, 4, 134, 16, 10, 10, 10, 18, 16, 17, 9, 16, 9, 9, 9, 18, 18, 16, 18, 17, 17, 16,
       38),
  ]
  x = atomizer.create_assignment_from_faces(repartition)

  data = list(plot_mesh_dirs(mesh0))
  data.extend(get_assignment_data(x))
  ctx = VispyCtx()
  ctx.plot_meshes(*data)
  ctx.run()


class AtomizerHelper:

  def __init__(self, atomizer, ctx):
    self.data = None
    self.init_debug_round = 0
    self.init_debug_round_objs = []
    self.ctx = ctx
    self.atomizer = atomizer
    self.atomizer.debug = True
    self.init_objs = []
    self.init_cm = get_colormap('viridis')
    self.debug_cm = get_colormap('autumn')
    self.default_cm = get_colormap('winter')

  def initial_assignment(self):
    self.init_debug_round = 0
    self.data = self.atomizer.initial_assignment()

  def plot_initial_assignment(self):
    self.ctx.remove_objs(self.init_objs)
    self.init_objs = self.plot_assignment(self.data.cluster_data, cm=self.init_cm)

  def plot_assignment(self, cluster_data, cm=None):
    if cm is None: cm = self.default_cm
    res = []
    for i, cl in enumerate(cluster_data):
      if cl.box.almost_empty(): continue
      mx = tc.clear().add_box(cl.box).to_mesh()
      res.append(get_mesh_data(mx, color=cm[i / (len(cluster_data) - 1)]))
    return self.ctx.plot_meshes(*res)

  @property
  def has_debug(self):
    return self.data is not None and len(self.data.debug_data.init_round_debug) > 0

  def inc_init_debug_round(self, dv):
    if not self.has_debug: return
    self.init_debug_round += dv
    self.init_debug_round %= len(self.data.debug_data.init_round_debug)

  def plot_init_debug_round(self):
    if not self.has_debug: return
    self.ctx.remove_objs(self.init_debug_round_objs)
    round_data = self.data.debug_data.init_round_debug[self.init_debug_round]
    self.init_debug_round_objs = self.plot_assignment(round_data.cluster_data, cm=self.debug_cm)

  def improve_asst(self, asst, round_ts):
    search_params = mathgame.KAtomizer_SearchParams()
    ts = mathgame.TimeStoppingCriteria(int(round_ts * 1e6))
    #ts = mathgame.RoundStoppingCriteria(1)
    search_params.stop_criteria = ts
    search_params.T0 = 1
    search_params.state = 2
    search_params.force_state = True

    self.atomizer.improve_assignment(asst, search_params)
    return asst


def test_atomize(ctx):
  mesh = load_stl(ctx.stl_file)
  cclist = mesh.split_connected()
  mesh0 = cclist[0]
  for meshx in cclist:
    if len(mesh0.vertices()) < len(meshx.vertices()):
      mesh0 = meshx

  atomizer = mathgame.KAtomizer(mesh0, 5)

  if ctx.nogui:
    h = AtomizerHelper(atomizer, None)
    h.initial_assignment()
    if ctx.improve: h.improve_asst(h.data, ctx.round_ts)
    return
  vctx = VispyCtx()
  h = AtomizerHelper(atomizer, vctx)

  @vctx.canvas.events.key_press.connect
  def on_key_press(ev):

    if ev.key == 'Right' or ev.key == 'Left':
      direction = 1
      if ev.key == 'Left': direction = -1
      h.inc_init_debug_round(direction)
      h.plot_init_debug_round()

    if ev.key == 'Space':
      h.initial_assignment()
      h.plot_initial_assignment()
    if ev.key == 'a':
      vctx.remove_objs(h.init_objs)
    if ev.key == '2':
      h.improve_asst(h.data, ctx.round_ts)
      h.plot_initial_assignment()

  data = list(plot_mesh_dirs(mesh0))
  print(data[0].points)
  vctx.cur_objs = vctx.plot_meshes(*data)
  vctx.run()
  return vctx


def test_intersection(ctx):
  ctx = VispyCtx()
  text = Text('where info goes', color='w', anchor_x='left', parent=ctx.view, pos=(20, 30))

  def run1():
    ctx.free_objs()
    for i in range(10):
      b1 = mathgame.BoxSpec_3DPython.Rand()
      b2 = mathgame.BoxSpec_3DPython.Rand()
      bl2 = mathgame.get_in_box_space(b1, b2)
      bl1 = mathgame.get_in_box_space(b1, b1)
      print(b1.almost_empty(), b1.area())
      print(b2.almost_empty(), b2.area())
      print(mathgame.BoxAASpec.FromBoxSpace(b1).str())
      print(b1.str())
      print(b2.str())
      print(bl1.str())
      print(bl2.str())

      are_intersecting = mathgame.are_boxes_intersecting(b1, b2)
      intersection = mathgame.get_boxes_intersection_area_grid(b1, b2, 5)
      text.text = '%s' % (dict(
          are_intersecting=are_intersecting,
          intersection=intersection,
          r1=intersection / b1.area(),
          r2=intersection / b2.area()))
      data = []
      data.append(get_mesh_data(fc.clear().add_box(b1).to_mesh(), color='r'))
      data.append(get_mesh_data(fc.clear().add_box(b2).to_mesh(), color='g'))
      data.append(get_mesh_data(fc.clear().add_box(bl2).to_mesh(), color='y'))
      data.append(get_mesh_data(fc.clear().add_box(bl1).to_mesh(), color='gray'))

    ctx.cur_objs = ctx.plot_meshes(*data)

  run1()

  @ctx.canvas.events.key_press.connect
  def on_key_press(ev):
    if ev.text == ' ':
      run1()

  ctx.run()


def f(d):
  x, y = d
  return 2 * x**2 + 2*x*y -2*x -1
  return x**2 + 2*x*y -2*x+ 3
  return (5000 - 0.005 * (x * x + y * y + x * y) + 12.5 *
          (x + y)) * np.exp(-abs(0.000001 * (x * x + y * y) - 0.0015 * (x + y) + 0.7))


def test_isocurve(ctx):
  ctx = VispyCtx(display_status=True)

  scale = 0.1
  cx = np.arange(-100, 100, scale)
  data = np.meshgrid(cx, cx)
  ff = f(data)
  levels = [0, 10]

  image = Image(ff, parent=ctx.view.scene)
  # move image behind curves
  image.transform = transforms.STTransform(scale=(scale, scale), translate=(0, 0, 0.5))
  color_lev = ['r', 'black']
  curve = Isocurve(ff, levels=levels, color_lev=color_lev, parent=ctx.view.scene)
  curve.transform = transforms.STTransform(scale=(scale, scale))

  # Set 2D camera
  ctx.view.camera = PanZoomCamera(aspect=1)
  # the camera will scale to the contents in the scene
  ctx.view.camera.set_range()
  ctx.run(cam=False)
  return ctx


def test_box(ctx):
  #Box=(corner=(-16.142605,-39.040726,44.207771), vx=(-5.391224,0.212364,-0.703033), vy=(-12.023680,17.890350,97.608040), vz=(1.840761,29.550863,-5.189526), volume=16348.84340034796),pt=(-26.325520,8.400475,136.626297),res.dist2(pt)=1751.31,points=((-26.979507,-18.498589,91.406258),(-26.325520,8.400475,136.626297),(-15.585112,-9.439316,38.850906),(-26.979511,-18.498600,91.406250),(-15.585115,-9.439322,38.850910),(-21.837727,-30.566788,90.440765)),

  points = ((-26.979507, -18.498589, 91.406258), (-26.325520, 8.400475, 136.626297),
            (-15.585112, -9.439316, 38.850906), (-26.979511, -18.498600, 91.406250),
            (-15.585115, -9.439322, 38.850910), (-21.837727, -30.566788, 90.440765))
  #points = mathgame.vec_pos(points)
  res = mathgame.compute_best_box(points)
  for pt in points:
    print(res.dist2(pt))
  print(res.in_tb(points))
  print(res.str())


def test_facegraph(ctx):
  box = mathgame.BoxSpec_3DPython()
  box.corner = (0, 0, 0)
  box.set(0, (1, 0, 0))
  box.set(1, (0, 2, 0))
  box.set(2, (0, 0, 3))

  mesh = fc.clear().add_box(box).to_mesh()
  fg = mesh.compute_face_graph2()
  print(fg.edge_costs)
  for face in mesh.list_faces():
    print(mesh.get_face_center(face))

  ctx = VispyCtx(display_status=True)
  ctx.plot_meshes(get_mesh_data(mesh, color='g'))
  ctx.run()


def read_meshlist(filename, set_attrs):
  meshlist = common_pb2.MeshList.FromString(open(filename, "rb").read())
  print(meshlist)

  meshes = []
  for mesh in meshlist.mesh:
    tags = mesh.name.split('-')
    points = []
    faces = []
    lines = []
    for face in mesh.face:
      pts = list([[v.x, v.y, v.z] for v in face.vertex])
      pos = len(points)
      n = len(pts)
      points.extend(pts)
      faces.append(list(range(pos, pos + n)))
      for i in range(len(pts)):
        lines.append((pts[i], pts[(i + 1) % n]))
    m = Attributize(
        mesh=MeshData(vertices=np.array(points), faces=np.array(faces)),
        lines=lines,
        points=points,
        color=mesh.col.desc,
        faces=faces)
    m.tags = tags
    if set_attrs(m):
      meshes.append(m)
  return meshes


def barvinok_set_attrs(m):
  m.color = 'rg' ['0' in m.tags]
  m.shift = [0, 0, 0]
  if 'split' in m.tags: m.shift = [10, 10, 10]
  return True


def plot_meshlist(ctx):
  meshes = read_meshlist(ctx.infile, barvinok_set_attrs)

  meshes.append(Attributize(points=np.array([[0, 0, 0]]), color='white', points_size=20))
  meshes.append(
      Attributize(lines=np.array([[[0, 0, 0], [20, 0, -30]]]), color='white', points_size=20))

  ctx = VispyCtx(display_status=True)
  ctx.plot_meshes(*meshes)
  ctx.run()


def meshlist_to_openscad(ctx):
  meshes = read_meshlist(ctx.infile, barvinok_set_attrs)
  res = []
  for mesh in meshes:
    col = mesh.get('color', 'yellow')

    res.append(f'''translate({mesh.shift}) color({Color(col).rgb.tolist()}) polyhedron (
    points = {mesh.points},
    faces = {mesh.faces});
    ''')
    for pt in mesh.points:
      res.append(f'''translate({mesh.shift}) translate({pt})
      sphere(0.1);
      ''')

  return '\n'.join(res)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
