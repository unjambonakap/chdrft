import contextlib
import pickle
import dataclasses
import numpy as np
import cv2

kImgKey = 'KEY_IMG'
kQueryKey = 'KEY_QUERY'


@dataclasses.dataclass
class Action:
  upsample: int = None


@dataclasses.dataclass
class Query:
  box: tuple
  img_data: bytes
  action: Action


@dataclasses.dataclass
class QueryResult:
  img_data: np.ndarray
  box: tuple


class CacheContext(contextlib.ExitStack):

  def __init__(self, path, ro=False):
    super().__init__()
    self.path = path
    self.ro = ro

  def clear(self):
    self.data = {}

  def __enter__(self):
    super().__enter__()
    try:
      self.data = pickle.load(open(self.path, 'rb'))
    except:
      self.data = {}

    self.push(self.exit)
    return self

  def exit(self, *args):
    if self.ro: return
    with open(self.path, 'wb') as f:
      pickle.dump(self.data, f)


@dataclasses.dataclass
class CacheFillerBase:
  context: CacheContext

  def process_one(self, query: Query) -> QueryResult:
    pass

  def process(self):
    for k, v in list(self.context.data.items()):
      v[kImgKey] = None

    ql = list(self.context.data.values())
    print('Orig size', len(ql))
    ql = [v for v in ql if v.get(kImgKey, None) is None]
    print('Processing size', len(ql))
    rl = self.process_batch([v[kQueryKey] for v in ql])

    for v, r in zip(ql, rl):
      v[kImgKey] = r

  def process_batch(self, queries: list[Query]) -> list[QueryResult]:
    return [self.process_one(x) for x in queries]

  @classmethod
  def Run(cls, context_path: str, *args, **kwargs):

    with CacheContext(context_path) as cctx:
      filler = cls(context=cctx, *args, **kwargs)
      filler.process()


def encode_img(img):
  return cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1]


def make_pipeline():
  from diffusers import StableDiffusionUpscalePipeline
  import torch
  # load model and scheduler
  model_id = "stabilityai/stable-diffusion-x4-upscaler"
  pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
  pipeline = pipeline.to("cuda")

  pipeline.enable_attention_slicing()

  return pipeline


def clean_ram():
  import time
  import gc
  gc.collect()
  import torch
  torch.cuda.empty_cache()
  time.sleep(10)


@dataclasses.dataclass
class CacheFillerSD(CacheFillerBase):
  pipeline: object

  def process_one(self, query: Query) -> QueryResult:
    pipeline = make_pipeline()
    try:
      from PIL import Image
      assert query.action.upsample == 4
      us = query.action.upsample
      (a, b), (c, d) = query.box
      box = ((a * us, b * us), (c * us, d * us))
      img = cv2.imdecode(query.img_data, cv2.IMREAD_UNCHANGED)
      res = np.array(pipeline(prompt='dont care', image=Image.fromarray(img[:,:,::-1])).images[0])[:, :, ::-1]
      print(res.shape, res.dtype, np.max(res), np.max(img))
      #res = (res*255).astype(np.uint8)
  
      return QueryResult(img_data=encode_img(res), box=box)
    finally:
      del pipeline
      clean_ram()

  # def process_batch(self, queries: list[Query]) -> list[QueryResult]:
  #   images = [cv2.imdecode(query.img_data, cv2.IMREAD_UNCHANGED)[:, :, ::-1] for query in queries]
  #   rl = self.pipeline(prompt=['dont care'] * len(images), image=images).images
  #   for query, r in zip(queries, rl):
  #     assert query.action.upsample == 4
  #     us = query.action.upsample
  #     (a, b), (c, d) = query.box
  #     box = ((a * us, b * us), (c * us, d * us))
  #     res = np.array(r)[:, :, ::-1]
  #     yield QueryResult(img_data=encode_img(res), box=box)


@dataclasses.dataclass
class CacheFillerDumb(CacheFillerBase):

  def process_one(self, query: Query) -> QueryResult:
    us = query.action.upsample
    (a, b), (c, d) = query.box
    box = ((a * us, b * us), (c * us, d * us))
    img = cv2.imdecode(query.img_data, cv2.IMREAD_UNCHANGED)
    res = cv2.resize(img, tuple(np.array(img.shape[:2]) * us)[::-1])
    return QueryResult(img_data=encode_img(res), box=box)
