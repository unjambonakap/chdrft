#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import gdspy
import numpy as np
import io
from chdrft.interactive.scripter import get_shelllike_interface
import re
from chdrft.struct.geo import QuadTree
import cv2

global flags, cache
flags = None
cache = None
from enum import IntFlag


class LayerType:
  VIA = 1 << 0
  METAL = 1 << 1
  POLY = 1 << 2
  UP = 1 << 3
  DOWN = 1 << 4


def args(parser):
  clist = CmdsList().add(test)
  parser.add_argument('--infile')
  parser.add_argument('--layermap-files', nargs='*', type=str)
  parser.add_argument('--outfile')
  parser.add_argument('--cell')
  parser.add_argument('--outdir')
  parser.add_argument('--via-up', action='store_true')
  parser.add_argument('--via-down', action='store_true')
  parser.add_argument('--overlap', type=float, default=0.2)
  parser.add_argument('--via-size', type=int, default=10)
  parser.add_argument('--image-dim', nargs=2, type=int, default=(2000, 1000))
  parser.add_argument('--xr-scandim', type=Z.Range1D.FromString, default=Z.Range1D(1, 2))
  parser.add_argument('--yr-scandim', type=Z.Range1D.FromString, default=Z.Range1D(1, 2))
  parser.add_argument('--scan-overlap', type=float, default=0.1)
  parser.add_argument('--upscale-range', type=Z.Range1D.FromString, default=Z.Range1D(1, 3))

  ActionHandler.Prepare(parser, clist.lst, global_action=1)


g_label_type = 'label'
g_poly_type = 'poly'
g_cellref_type = 'cellref'


class GDSFile:

  def __init__(self, fname):
    self.x = gdspy.GdsLibrary()
    self.x.read_gds(fname)
    self.cell_to_data = {}
    self.gidgen = cmisc.IdGen()
    self.polyid = 0
    for k, v in self.x.cell_dict.items():
      self.get_cell_info(v)
    print('GOT CELLS ', self.cell_to_data.keys())

  def next_polyid(self):
    res = self.polyid
    self.polyid += 1
    return res

  def get_cell(self, cellname):
    data = self.cell_to_data[cellname]
    res = cmisc.Attr(
        layers=Z.defaultdict(lambda: cmisc.Attr(polys=[])),
        cellrefs=[],
        labels=[],
    )
    for entry in data:
      if entry.type == g_poly_type:
        res.layers[entry.layer].polys.append(entry)
      elif entry.type == g_label_type:
        res.labels.append(entry)
      elif entry.type == g_cellref_type:
        res.cellrefs.append(entry)
      else:
        assert 0

    return res

  def get_cell_info(self, cell, pos=np.array([0., 0.]), mirror_x=False, rot180=False):

    if not cell.name in self.cell_to_data:
      res = []
      for e in cell.references:
        rd = self.get_cell_info(
            e.ref_cell,
            e.origin,
            mirror_x=e.x_reflection,
            rot180=e.rotation is not None,
        )
        res.extend(rd.entries)
        res.append(
            cmisc.Attributize(
                cell_id=rd.cell_id,
                pos=e.origin,
                box=rd.box,
                cell_name=e.ref_cell.name,
                mirror_x=e.x_reflection,
                rot180=e.rotation is not None,
                type=g_cellref_type,
            )
        )

      curpoly = []
      for e in cell.polygons:
        assert len(e.polygons) == 1
        pts = e.polygons[0]
        assert len(pts) == 4
        entry = cmisc.Attributize(
            pos=pts,
            layer=e.layers[0],
            idx=self.next_polyid(),
            type=g_poly_type,
            cell_name=cell.name,
            box=Z.Box().union_points(pts),
        )
        res.append(entry)
        curpoly.append(entry)

      if len(cell.labels):
        qtree = QuadTree(curpoly)

        for label in cell.labels:
          qx = qtree.query(label.position, filter=lambda x: x.layer == label.layer)
          assert qx.dist < 1e-9
          res.append(
              cmisc.Attributize(
                  pos=label.position,
                  layer=label.layer,
                  text=label.text,
                  poly_idx=qx.best.idx,
                  poly_box=qx.best.box,
                  type=g_label_type,
                  cell_name=cell.name
              )
          )

      self.cell_to_data[cell.name] = res

    curid = self.gidgen()
    seen_gid = {}
    seen_gid[None] = curid
    nres = []
    finalid = 0
    box = Z.Box()
    poly_rmp_idx = {}
    for e in self.cell_to_data[cell.name]:
      ne = e._do_clone()
      ne.pos = np.array(e.pos)
      if rot180:
        ne.pos = -ne.pos
      if mirror_x:
        ne.pos = ne.pos * np.array([1, -1])

      ne.pos += pos
      if ne.type == g_poly_type:
        ne.idx = self.next_polyid()
        poly_rmp_idx[e.idx] = ne
        ne.box = Z.Box().union_points(ne.pos)
        box = box.union(ne.box)

      if ne.type == g_label_type:
        assert ne.poly_idx in poly_rmp_idx
        poly = poly_rmp_idx[ne.poly_idx]
        ne.poly_idx = poly.idx
        ne.poly_box = poly.box

      finalid += 1

      cid = ne.get('cell_id', None)
      if cid not in seen_gid:
        seen_gid[cid] = self.gidgen()
      ne.cell_id = seen_gid[cid]

      ne.cell_name = ne.cell_name
      nres.append(ne)
    return cmisc.Attr(box=box, entries=nres, cell_id=curid)


def dump_cell_info(cinfo):
  f = io.StringIO()

  print(len(cinfo), file=f)
  for e in cinfo:
    pos = np.around(e.pos * 10000).astype(np.int32)
    if e.type == g_poly_type:
      box = Z.Box().union_points(pos)
      minv = np.amin(pos, axis=0)
      maxv = np.amax(pos, axis=0)

    print(
        e.idx, [0, 1][e.type == g_poly_type],
        e.layer,
        e.cell_name + '_' + str(e.cell_id),
        end='',
        file=f
    )
    if e.type == g_poly_type:
      for i in range(4):
        print('', pos[i, 0], pos[i, 1], end='', file=f)
      print(file=f)
    else:
      print('', pos[0], pos[1], e.text, file=f)
  return f.getvalue()


def test(ctx):
  x = GDSFile(ctx.infile)
  print(set(a.layer for a in x.cell_to_data[ctx.cell]))
  #print(dump_cell_info(x.cell_to_data[ctx.cell]))


class KLayoutDB:

  def __init__(self, fname):
    groups = {}
    self.groups = groups
    cur = None
    tb = []
    groups[cur] = tb
    for line in open(fname, 'r').readlines():
      line = line.strip()
      if line.startswith('<< '):
        m = re.match('<< (?P<name>\w+) >>', line)
        cur = m['name']
        if cur == 'end':
          tb = None
        else:
          tb = [line]
          groups[cur] = tb
        continue

      wx = line.split(' ')[0]
      if cur is not None and wx not in ('rlabel', 'port', 'rect'):
        continue
      tb.append(line)

  def rebuild_for(self, layername):
    return '\n'.join(self.groups[None] + self.groups[layername] + ['<< end >>'])

  @property
  def layers(self):
    return list(filter(None, self.groups.keys()))


def test1(ctx):
  db = KLayoutDB(ctx.infile)

  layer_map = {}
  for layer in db.layers:
    if layer == 'labels':
      continue
    with Z.tempfile.TemporaryDirectory() as tempdir:
      tempdir = Z.tempfile.mkdtemp()
      Z.shutil.copy('./.magicrc', tempdir)
      ix = get_shelllike_interface('/bin/bash')
      kname = 'base'
      magfile = f'{kname}.mag'
      gdsfile = f'{kname}.gds'
      with open(f'{tempdir}/{magfile}', 'w') as f:
        f.write(db.rebuild_for(layer))

      ix(f'cd {tempdir}')
      ix(f'''magic -dnull -noconsole {kname} <<'END'
  gds write {gdsfile}
END
true''')
      x = GDSFile(f'{tempdir}/{gdsfile}')
      lst = list(set(a.layer for a in x.cell_to_data[kname]))
      print(tempdir, layer, lst)
      layer_map[layer] = lst

  Z.pprint(layer_map)

  Z.FileFormatHelper.Write(ctx.outfile, layer_map)


def groupby_to_dict(x):
  res = cmisc.Attr()
  for e in x:
    res[e.key] = e.to_list()
  return res


def export_layer(layer):
  nlayer = layer._do_clone()
  nlayer.qtree = None

  def repl_by_num(obj, field):
    if obj[field]:
      print(obj)
      cur = obj[field]
      if isinstance(cur, list):

        if len(cur) > 0 and isinstance(cur[0], int):
          obj[field] = [x for x in cur]
        else:
          obj[field] = [x.num for x in cur]
      elif isinstance(cur, int):
        obj[field] = cur
      else:
        obj[field] = cur.num

  if 'metal' in nlayer:
    repl_by_num(nlayer.metal, 'up')
    repl_by_num(nlayer.metal, 'down')

  if 'via' in nlayer:
    repl_by_num(nlayer.via, 'ups')
    repl_by_num(nlayer.via, 'downs')
  return nlayer


def get_layermap(fnames):
  tmpdata = {}
  for fname in fnames:
    layer_map = Z.FileFormatHelper.Read(fname)
    tmpdata.update(layer_map)

  allx = []

  by_num = {}
  others = set()
  Z.pprint(tmpdata)
  for name, nums in tmpdata.items():
    if name.endswith('contact'):
      allx.append(
          cmisc.Attr(
              name=name,
              nums=nums,
              type='via',
              metal=cmisc.Attr(
                  up=None,
                  down=None,
              ),
              layertype=LayerType.VIA
          )
      )

    elif name.startswith('metal'):
      metalnum = int(name[5:])
      assert len(nums) == 1
      entry = cmisc.Attr(
          name=name,
          num=nums[0],
          nums=nums,
          metalnum=metalnum,
          via=cmisc.Attr(ups=[], downs=[]),
          layertype=LayerType.METAL,
          type='metal'
      )
      by_num[entry.num] = entry
      allx.append(entry)
    else:
      others.update(nums)

  other_entry = cmisc.Attr(
      num=-1,
      name='other',
      layertype=LayerType.POLY,
      nums=others,
      metalnum=0,
      via=cmisc.Attr(ups=[], downs=[]),
      type='other',
  )
  for num in others:
    by_num[num] = other_entry

  data = groupby_to_dict(Z.asq_query(allx).group_by(lambda x: x.type))
  nums_novia = set(cmisc.flatten([x.nums for x in data.metal] + list(others)))

  for x in data.via:
    rem = list(set(x.nums) - nums_novia)
    assert len(rem) == 1
    x.num = rem[0]
    metals = {}
    for num in x.nums:
      if num == x.num:
        continue
      metal = by_num[num]
      metals[metal.metalnum] = metal
    metals = list(metals.values())

    if len(metals) == 1:
      assert metals[0].type == 'other'
      metals.append(other_entry)

    assert len(metals) == 2, x
    metals.sort(key=lambda x: x.metalnum)
    metals[0].via.ups.append(x)
    metals[1].via.downs.append(x)
    x.metal.down = metals[0]
    x.metal.up = metals[1]
    x.metalnum = x.metal.down.metalnum
    x.nums = rem
    by_num[x.num] = x

  for metal in data.metal:
    metal.via.ups = cmisc.make_uniq(metal.via.ups, key=lambda x: x.num)
    metal.via.downs = cmisc.make_uniq(metal.via.downs, key=lambda x: x.num)

  data.layers = [other_entry] + data.via + data.metal
  data.metal = {x.metalnum: x for x in data.metal}
  data.other = other_entry
  data.by_num = by_num
  data.name2num = tmpdata

  return data


class GDSRenderer:

  def __init__(self, layermap, cell, ctx):
    self.layermap = layermap
    self.cell = cell
    self.ctx = ctx
    self.render_info = cmisc.Attr(layers=[])

    for layer in self.layermap.layers:
      self.setup_layer(layer)
    self.params = None

  def setup_layer(self, layer):
    objs = []
    area_num_list = []
    for num in layer.nums:
      area_tot = 0
      curlayer = self.cell.layers[num]
      for poly in curlayer.polys:
        objs.append(cmisc.Attr(num=num, box=poly.box, idx=poly.idx))
        area_tot += poly.box.area

      area_num_list.append((area_tot, num))

    if layer.type == 'other':
      num_to_level = {}
      area_num_list.sort(reverse=1)
      maxv = 192
      for i in range(len(area_num_list)):
        num = area_num_list[i][1]
        num_to_level[num] = maxv * (i + 1) // len(area_num_list)

    elif layer.type == 'via':
      num_to_level = Z.defaultdict(lambda: 255)
    elif layer.type == 'metal':
      num_to_level = Z.defaultdict(lambda: 128)
    else:
      assert 0

    for obj in objs:
      obj.level = num_to_level[obj.num]

    layer.qtree = QuadTree(objs)

  def render_layer(self, layer_desc):
    assert self.ctx.outdir is not None
    useful_box = Z.Box()
    for layer in layer_desc.group:
      useful_box = useful_box.union(layer.qtree.box)
    target_box = useful_box.expand_l(self.params.image_dim_u)

    scan_count = self.ctx.ROI_scandim.uniform()
    scan_dim = np.ceil(target_box.dim * (1 + self.ctx.scan_overlap) / scan_count)
    stride = scan_dim / (1 + self.ctx.scan_overlap)
    jitter = stride * 0.1
    scan_dim_jittered = scan_dim + 2 * jitter
    ids, scan_poslist = target_box.get_grid_pos_with_id(stride=scan_dim)
    scan_poslist += scan_dim / 2 + np.random.uniform(
        low=-jitter, high=jitter, size=(len(scan_poslist), 2)
    )
    print(
        'RENDERING LAYER ', scan_count, scan_dim, len(scan_poslist), ids, scan_poslist, target_box
    )

    for idx, pos in zip(ids, scan_poslist):
      scan_box = Z.Box(center=pos, dim=scan_dim_jittered)
      scan_name = f'scan_{idx[0]}_{idx[1]}_{layer_desc.name}'
      upscale = self.ctx.upscale_range.uniform()
      sd = self.render_scan(layer_desc, scan_box, scan_name, upscale)
      sd.useful_box = useful_box

  def render_scan(self, layer_desc, target_box, scan_name, upscale):
    u2px = self.params.u2px * upscale
    stride_u = self.params.stride_u / upscale
    image_dim_u = self.params.image_dim_u / upscale

    layer_data = cmisc.Attr(
        name=scan_name,
        outdir=f'layer_{scan_name}',
        layertype=layer_desc.layertype,
        metalnum=layer_desc.get('metalnum', None),
        u2px = u2px,
        upscale=upscale,
        tiles=[],
        layers=[export_layer(layer) for layer in layer_desc.group],
    )
    self.render_info.layers.append(layer_data)

    layer_outdir = Z.os.path.join(self.ctx.outdir, layer_data.outdir)
    cmisc.makedirs(layer_outdir)

    layer_data.strict_box = target_box
    layer_data.render_box = target_box

    print('rendering layer ', scan_name, target_box, upscale)
    for gridpos, pos in zip(*target_box.get_grid_pos_with_id(stride_u)):
      img = np.zeros(self.params.image_dim[::-1], dtype=np.uint8)
      box = Z.Box(low=pos, dim=image_dim_u)
      for layer in layer_desc.group:
        self.render_layer_box(layer, box, img)

      fname = f'S{gridpos[0]:03d}_{gridpos[1]:03d}.tif'
      layer_data.tiles.append(cmisc.Attr(pos=pos, gridpos=gridpos, fname=fname, box=box))
      cv2.imwrite(Z.os.path.join(layer_outdir, fname), img)
    return layer_data

  def render_layer_box(self, layer, box, img):
    objs = layer.qtree.query_box(box, make_uniq=0)
    objs = cmisc.make_uniq(objs, lambda x: x.idx)
    image_box = Z.Box.FromImage(img)
    objs.sort(key=lambda x: x.level)
    for obj in objs:
      p1 = box.change_rect_space(image_box, obj.box.low)
      p2 = box.change_rect_space(image_box, obj.box.high)
      color = (obj.level, 0, 0)
      cv2.rectangle(
          img,
          tuple(p1.round().astype(int)),
          tuple(p2.round().astype(int)),
          color=color,
          thickness=-1
      )

    return img

  def get_render_info(self):
    self.render_info.params = self.params
    self.render_info.ctx = self.ctx
    self.render_info.cellrefs = self.cell.cellrefs
    self.render_info.labels = self.cell.labels
    return self.render_info


def render_gds(ctx):

  layermap = get_layermap(ctx.layermap_files)
  cmisc.failsafe(lambda: Z.shutil.rmtree(ctx.outdir))

  gds = GDSFile(ctx.infile)
  cell = gds.get_cell(ctx.cell)
  renderer = GDSRenderer(layermap, cell, ctx)

  render_params = ctx.render_params
  render_params.image_box = Z.Box(low=(0, 0), dim=render_params.image_dim)

  for layer in layermap.via:
    if layer.qtree.root is None:
      continue
    via_dims = np.mean([x.box.dim for x in layer.qtree.objs], axis=0)
    assert abs(via_dims[0] - via_dims[1]) < 1e-4
    render_params.via_dim = via_dims[0]
    render_params.u2px = render_params.via_size / via_dims[0]
    render_params.image_dim_u = render_params.image_dim / render_params.u2px
    render_params.stride_u = render_params.image_dim_u * (1 - render_params.overlap)
    break

  Z.pprint(render_params)
  renderer.params = render_params

  render_groups = []
  seen = set()
  for layer in list(layermap.metal.values()) + [layermap.other]:
    if layer.qtree.root is None:
      continue
    group = []
    vias = []
    name = 'metal%d' % layer.metalnum
    layertype = layer.layertype
    via_up = (ctx.via_up or layer.metalnum == 0)
    if via_up:
      vias = layer.via.ups
      name += '_via_up'
      layertype |= LayerType.VIA | LayerType.UP

    if ctx.via_down and not via_up:
      vias = layer.via.downs
      name += '_via_down'
      layertype |= LayerType.VIA | LayerType.DOWN

    group.append(layer)
    if vias:
      group.extend(filter(lambda x: x.qtree.root, vias))  # render vias last
      seen.update(x.num for x in vias)

    render_groups.append(
        cmisc.Attr(name=name, layertype=layertype, metalnum=layer.metalnum, group=group)
    )

  for via in layermap.via:
    if via.qtree.root is None:
      continue
    if via.num not in seen:
      render_groups.append(
          cmisc.Attr(
              name='via_%d_%d' % (via.metalnum, via.metalnum + 1),
              layertype=via.layertype,
              group=[via],
          )
      )

  for render_group in render_groups:
    renderer.render_layer(render_group)

  render_info = renderer.get_render_info()
  Z.FileFormatHelper.Write(f'{ctx.outdir}/data.pickle', render_info)


def test12(ctx):
  render_params = ctx.render_params
  via_dims = np.array((123., 323.), dtype=np.float64)
  render_params.u2px = render_params.via_size / via_dims[0]
  render_params.image_dim_u = render_params.image_dim / render_params.u2px
  render_params.stride_u = render_params.image_dim_u * (1 - render_params.overlap)
  res = cmisc.json_dumps(render_params)
  print(type(render_params.image_dim))
  print(res)


def main():
  ctx = Attributize()
  ctx.ROI_scandim = Z.Range2D(flags.xr_scandim, flags.yr_scandim, is_int=1)
  ctx.render_params = cmisc.Attr(
      via_size=flags.via_size,
      overlap=flags.overlap,
      image_dim=np.array(flags.image_dim),
  )
  ActionHandler.Run(ctx)


app()
