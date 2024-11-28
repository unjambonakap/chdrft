#!/usr/bin/env python

import chdrft.utils.misc as cmisc
import seaborn as sns
import numpy as np

kelly_colors_hex_c1 = [
    0xFFB300,  # Vivid Yellow
    0xFF6800,  # Vivid Orange
    0xA6BDD7,  # Very Light Blue
    0xC10020,  # Vivid Red
    0xCEA262,  # Grayish Yellow
    0x817066,  # Medium Gray
    0x803E75,  # Strong Purple
]

kelly_colors_hex_c2 = [
    # The following don't work well for people with defective color vision
    0x007D34,  # Vivid Green
    0xF6768E,  # Strong Purplish Pink
    0x00538A,  # Strong Blue
    0xFF7A5C,  # Strong Yellowish Pink
    0x53377A,  # Strong Violet
    0xFF8E00,  # Vivid Orange Yellow
    0xB32851,  # Strong Purplish Red
    0xF4C800,  # Vivid Greenish Yellow
    0x7F180D,  # Strong Reddish Brown
    0x93AA00,  # Vivid Yellowish Green
    0x593315,  # Deep Yellowish Brown
    0xF13A13,  # Vivid Reddish Orange
    0x232C16,  # Dark Olive Green
]

kelly_colors = dict(
    vivid_yellow=(255, 179, 0),
    strong_purple=(128, 62, 117),
    vivid_orange=(255, 104, 0),
    very_light_blue=(166, 189, 215),
    vivid_red=(193, 0, 32),
    grayish_yellow=(206, 162, 98),
    medium_gray=(129, 112, 102),

    # these aren't good for people with defective color vision:
    vivid_green=(0, 125, 52),
    strong_purplish_pink=(246, 118, 142),
    strong_blue=(0, 83, 138),
    strong_yellowish_pink=(255, 122, 92),
    strong_violet=(83, 55, 122),
    vivid_orange_yellow=(255, 142, 0),
    strong_purplish_red=(179, 40, 81),
    vivid_greenish_yellow=(244, 200, 0),
    strong_reddish_brown=(127, 24, 13),
    vivid_yellowish_green=(147, 170, 0),
    deep_yellowish_brown=(89, 51, 21),
    vivid_reddish_orange=(241, 58, 19),
    dark_olive_green=(35, 44, 22)
)


class ColorConv:

  @staticmethod
  def to_rgb(v, as_float=False):
    if isinstance(v, int):
      v = ((v >> 16) % 256, (v >> 8) % 256, (v >> 0) % 256)
    v = np.array(v)
    if as_float:
      if np.issubdtype(v.dtype, np.integer):
        v = v / 255
    return v

  @staticmethod
  def to_hex(v):
    if isinstance(v, int):
      return v
    return (v[0] << 16) | (v[1] << 8) | v[2]


class BackupColors:

  def __init__(self):
    self.pos = 0

  def get(self, remove=True):
    col = kelly_colors[self.pos]
    if remove:
      self.pos = (self.pos + 1) % len(kelly_colors)
    return col


class ColorPool:

  def __init__(self, want_backup_colors=True, looping=False):
    self.colors = list(kelly_colors_hex_c1) + list(kelly_colors_hex_c2)
    self.backup_colors = None
    self.looping = looping
    self.pos = 0
    if want_backup_colors:
      self.backup_colors = BackupColors()

  def release(self, c):
    c = ColorConv.to_hex(c)
    self.colors.append(c)

  def __call__(self, remove=True):
    return self.get(remove=remove)

  def get(self, remove=True):
    if self.colors:
      if self.looping:
        pos = self.pos
        if remove:
          self.pos += 1
        return self.colors[pos % len(self.colors)]
      else:
        res = self.colors[0]
        if remove:
          del self.colors[0]
        return res

    assert self.backup_colors is not None
    return self.backup_colors.get(remove)

  def get_rgb(self, **kwargs):
    return ColorConv.to_rgb(self.get(**kwargs))


class ColorMapper:

  def __init__(self, vals):
    self.rmp = cmisc.Remap(vals)
    self.cmap = sns.color_palette(n_colors=self.rmp.n)

  def get(self, x):
    return self.cmap[self.rmp.get(x)]


def get_inf_cpool(name) -> cmisc.InfGenerator:
  from vispy.color import get_colormap
  return cmisc.InfGenerator(get_colormap('viridis'))
