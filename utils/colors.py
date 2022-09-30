#!/usr/bin/env python 


import chdrft.utils.misc  as cmisc
import seaborn as sns
import numpy as np
kelly_colors_hex_c1 = [
    0xFFB300,  # Vivid Yellow
    0x803E75,  # Strong Purple
    0xFF6800,  # Vivid Orange
    0xA6BDD7,  # Very Light Blue
    0xC10020,  # Vivid Red
    0xCEA262,  # Grayish Yellow
    0x817066,  # Medium Gray
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

kelly_colors = dict(vivid_yellow=(255, 179, 0),
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
                    dark_olive_green=(35, 44, 22))


class ColorConv:

  @staticmethod
  def to_rgb(v):
    if isinstance(v, int):
      v= ((v >> 16) % 256, (v >> 8) % 256, (v >> 0) % 256)
    return np.array(v)

  @staticmethod
  def to_hex(v):
    if isinstance(v, int):
      return v
    return (v[0] << 16) | (v[1] << 8) | v[2]


class BackupColors:

  def __init__(self):
    self.pos = 0

  def get(self):
    col = kelly_colors[self.pos]
    self.pos = (self.pos + 1) % len(kelly_colors)
    return col


class ColorPool:

  def __init__(self, want_backup_colors=True):
    self.p1 = list(kelly_colors_hex_c1)
    self.p2 = list(kelly_colors_hex_c2)

    self.backup_colors = None
    if want_backup_colors:
      self.backup_colors = BackupColors()

  def get_from(self, p):
    c = p[0]
    del p[0]
    return c

  def release(self, c):
    c = ColorConv.to_hex(c)
    if c in kelly_colors_hex_c1 and c not in self.p1:
      self.p1.append(c)
    elif c in kelly_colors_hex_c2 and c not in self.p2:
      self.p2.append(c)
  def __call__(self): return self.get()

  def get(self):
    if len(self.p1) > 0:
      return self.get_from(self.p1)
    if len(self.p2) > 0:
      return self.get_from(self.p2)
    assert self.backup_colors is not None
    self.backup_colors.get()

  def get_rgb(self):
    return ColorConv.to_rgb(self.get())


class ColorMapper:
  def __init__(self, vals):
    self.rmp = cmisc.Remap(vals)
    self.cmap = sns.color_palette(n_colors=self.rmp.n)

  def get(self, x):
    return self.cmap[self.rmp.get(x)]


def get_inf_cpool(name) -> cmisc.InfGenerator:
    from vispy.color import Color, get_colormap
    return cmisc.InfGenerator(get_colormap('viridis'))

