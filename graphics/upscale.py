#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import chdrft.utils.Z as Z
import chdrft.utils.K as K
import numpy as np
import cv2
from chdrft.graphics.upscale_colab import *

global flags, cache
flags = None
cache = None

kEps = 1e-9

class PartitionFunc(cmisc.PatchedModel):

  def weight(self, pt: np.ndarray) -> float:
    d_int, d_ext = self.get_dist(pt)
    return self.get(d_int, d_ext)

  def weight_np(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d_int, d_ext = self.get_dist_np(x, y)
    return self.get_np(d_int, d_ext)

  def get_dist_np(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pass

  def f(self, v: np.ndarray) -> np.ndarray:
    return np.where(v < kEps, 0, np.exp(-1 / v))

  def get_np(self, d_int: np.ndarray, d_ext: np.ndarray) -> np.ndarray:
    fi = self.f(-d_int)
    fe = self.f(d_ext)
    res = fe / (fe + fi)
    assert np.all(res >= 0)
    return res

  def get(self, d_int: float, d_ext: float) -> float:
    return self.get_np(d_int, d_ext)


class SquarePartitionFunc(PartitionFunc):
  center: np.ndarray
  core_dist: float
  fade_dist: float

  def get_dist_np(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = np.maximum(np.abs(self.center[0] - x), np.abs(self.center[1] - y))
    return self.core_dist - dist, self.fade_dist - dist

  def get_dist(self, pt: np.ndarray) -> tuple[float, float]:
    dist = np.max(np.abs(self.center - pt))
    return self.core_dist - dist, self.fade_dist - dist


class PartitionOfUnityEntry(cmisc.PatchedModel):
  f: Z.typing.Callable[[np.ndarray], object] = None
  f_np: Z.typing.Callable[[Z.Box], object] = None
  part: PartitionFunc


class PartitionOfUnityBase(cmisc.PatchedModel):
  entries: list[PartitionOfUnityEntry]

  def eval(self, pt: np.ndarray):
    vl = []
    wsum = 0
    for e in self.entries:
      w = e.part.weight(pt)
      if w > 0:
        vl.append(e.f(pt) * w)
        wsum += w

    assert vl, pt
    return sum(vl) / wsum

  def eval_np(self, box: Z.Box, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    vl = []
    wsum = np.zeros_like(x, dtype=np.float64)
    vl = 0

    for e in self.entries:
      w = e.part.weight_np(x, y)
      vl = e.f_np(box) * w + vl
      wsum += w

    assert np.all(wsum >= 0), np.argwhere(wsum < 0)
    return vl / wsum


class ImageQueryOracle(cmisc.PatchedModel):

  def query(self, query: Query) -> QueryResult:
    pass


class ImageQueryOracleCache(ImageQueryOracle):
  context: CacheContext

  def query(self, query: Query) -> QueryResult:
    kstr = cmisc.json_dumps((query.action, query.box))
    img = self.context.data.get(kstr, {}).get(kImgKey, None)
    if img is None:
      img = self.process_one(query)
      self.context.data[kstr] = {kQueryKey: query, kImgKey: img}
    return img

  def process_one(self, query: Query) -> QueryResult:
    box = Z.Box(low=query.box[0], high=query.box[1], is_int=1) * query.action.upsample
    return QueryResult(img_data=None, box=box.as_tuple)


class ImagePartition(cmisc.PatchedModel):
  oracle: ImageQueryOracle = None
  tile_size: int
  core_dist: int
  fade_dist: int
  overlap_factor: float
  base_img: K.ImageData
  action: Action

  @property
  def padding(self) -> int:
    return self.tile_size

  @cmisc.cached_property
  def img(self) -> K.ImageData:
    data = np.pad(
        self.base_img.img, [(self.padding, self.padding), (self.padding, self.padding), (0, 0)],
        mode='reflect'
    )
    return K.ImageData(img=data)

  def make_partition(self) -> PartitionOfUnityBase:

    box: Z.Range2D = self.img.box
    stride = int(self.tile_size * (1 - self.overlap_factor))
    assert stride <= 2 * self.core_dist
    centers = list(
        Z.itertools.product(
            *[
                range(
                    self.padding + int(self.core_dist * 0.8), dim - self.padding -
                    int(self.core_dist * 0.8) + stride, stride
                ) for dim in box.dim
            ]
        )
    )
    boxes = [Z.Range2D(center=p, dim=(self.tile_size, self.tile_size)) for p in centers]

    entries = [self._make_entry(box) for box in boxes]
    entries = list(filter(None, entries))
    return PartitionOfUnityBase(entries=entries)

  def _make_entry(self, box: Z.Box) -> PartitionOfUnityEntry:
    subimg = self.img.subimg(box)
    q = Query(img_data=K.ImageData(img=subimg).encode(), box=box.as_tuple, action=self.action)
    res = self.oracle.query(q)
    resimg = K.ImageData.Make(res.img_data).img if res.img_data is not None else None
    if resimg is not None: print(resimg.dtype, np.max(resimg))
    rbox = Z.Box(low=res.box[0], high=res.box[1], is_int=1)

    def f(pos: np.ndarray) -> np.ndarray:
      coord = (pos - res.box[0])[::-1]
      tmp = resimg[tuple(coord)]
      return tmp

    def f_np(b: Z.Box) -> np.ndarray:
      return cv2.warpAffine(
          resimg, np.array([[1, 0, rbox.xl - b.xl], [0, 1, rbox.yl - b.yl]], dtype=np.float32),
          b.dim
      )

    return PartitionOfUnityEntry(
        part=SquarePartitionFunc(
            core_dist=self.core_dist * self.action.upsample,
            fade_dist=self.fade_dist * self.action.upsample,
            center=rbox.center
        ),
        f=f,
        f_np=f_np,
    )

  def process(self) -> K.ImageData:

    assert self.action.upsample is not None
    target_box = Z.Box(dim=self.base_img.box.dim, low=(0, 0), is_int=True)
    target_box = (target_box + self.padding) * self.action.upsample
    target_img = K.ImageData(img=target_box.make_zero_image(nchannels=self.img.channels, dtype=np.float32))
    part = self.make_partition()

    y, x, _ = np.meshgrid(target_box.yr.range, target_box.xr.range, [0], indexing='ij')
    return K.ImageData(part.eval_np(target_box, x, y))

    for xy in target_box:
      xy = np.array(xy)
      pv = part.eval(xy)
      target_img.set_pix(xy - target_box.low, pv)

    return target_img


def test(ctx):
  img = K.ImageData.Make(ctx.image)
  #img = K.ImageData(img=cv2.resize(img.img, (800, 600)))
  print(img.box, img.img.shape)
  action = Action(upsample=4)
  core_dist = 400 // 2
  fade_dist = 490 // 2
  tile_size = 512
  overlap = 0.6

  ip = ImagePartition(
      action=action,
      base_img=img,
      oracle=None,
      tile_size=tile_size,
      overlap_factor=overlap,
      core_dist=core_dist,
      fade_dist=fade_dist,
  )

  if ctx.do_creation:
    with CacheContext(ctx.context_path) as cctx:
      cctx.clear()
      ip.oracle = ImageQueryOracleCache(context=cctx)
      parts = ip.make_partition()

      res_img = ip.img.clone().upscale(action.upsample)
      res_img.box = res_img.img_box
      lines = []
      for e in parts.entries:
        box = Z.Box(center=e.part.center, dim=e.part.core_dist * action.upsample)
        box = Z.Box(center=e.part.center, dim=tile_size * action.upsample)
        lines.append(box)

      if 0:
        K.oplt.plot(A(images=[res_img], lines=lines))
        input()
        return

  if ctx.do_fill:
    CacheFillerDumb.Run(ctx.context_path)

  if ctx.do_merge:
    with CacheContext(ctx.context_path, ro=True) as cctx:
      ip.oracle = ImageQueryOracleCache(context=cctx)
      res = ip.process()
      res.save(ctx.out_img)
      print(K.stats.describe(res.img.flatten()))


def extract(ctx):
  with CacheContext(ctx.context_path, ro=True) as cctx:
    with CacheContext(ctx.target_context_path) as tctx:
      tctx.data.clear()
      for k, v in list(cctx.data.items())[:3]:
        x = K.ImageData.Make(v[kImgKey].img_data).img
        y = K.ImageData.Make(v[kQueryKey].img_data).img
        print()
        print(K.stats.describe(x.flatten()))
        print(K.stats.describe(y.flatten()))
        tctx.data[k] = v

def render(ctx):
  from chdrft.display.render import ImageGrid
  with CacheContext(ctx.context_path, ro=True) as cctx:
    vals = list(cctx.data.values())
    images = []
    for v in vals[:10]:
      x = K.ImageData.Make(v[kImgKey].img_data).img
      y = K.ImageData.Make(v[kQueryKey].img_data).img
      print()
      print(K.stats.describe(x.flatten()))
      print(K.stats.describe(y.flatten()))
      images.extend([x,y])
    ig = ImageGrid(nc=2, images=images)

    K.oplt.plot(A(images=ig.get_images()))
    input()


def args(parser):
  clist = CmdsList()
  parser.add_argument('--image')
  parser.add_argument('--context-path')
  parser.add_argument('--target-context-path')
  parser.add_argument('--out-img')
  parser.add_argument('--do-fill', action='store_true')
  parser.add_argument('--do-creation', action='store_true')
  parser.add_argument('--do-merge', action='store_true')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
