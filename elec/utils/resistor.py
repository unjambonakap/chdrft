#!/usr/bin/env python

from chdrft.utils.misc import Attributize, to_list
from chdrft.cmds import Cmds
from enum import Enum


class Resistor:
  Colors = Enum(
      'Colors',
      'black brown red orange yellow green blue purple grey white silver gold')
  Multipliers = Attributize(other=lambda x: 10 ** x.value,
                            elem={Colors.white: 0.01,
                                  Colors.black: 1})

  Tolerance = {
      Colors.silver: 10,
      Colors.gold: 5,
      Colors.brown: 1,
      Colors.red: 2,
      Colors.green: 0.5,
      Colors.blue: 0.25,
      Colors.purple: 0.1
  }

  @staticmethod
  def get_tol(tol):
    return Resistor.Tolerance[tol]

  @staticmethod
  def band_val(col):
    return col.value - 1

  @staticmethod
  def get_multiplier(col):
    assert not col in (Resistor.Colors.white, Resistor.Colors.grey)
    return Resistor.Multipliers[col]

  @staticmethod
  def valstr(val):
    if val < 1e3:
      return val
    if val < 1e6:
      return '%.3fK' % (val / 1000)
    if val < 1e9:
      return '%.3fM' % (val / 1e6)
    return '%.3fG' % (val / 1e9)

  @staticmethod
  def get(tb):
    tb = to_list(tb)
    assert len(tb) in (4, 5, 6)
    tb = [Resistor.Colors[x] for x in tb]

    temp = None
    if len(tb) == 6:
      tb, temp = tb[:-1], tb[-1]
    mul = Resistor.get_multiplier(tb[-2])
    tol = Resistor.get_tol(tb[-1])
    val = 0
    for i in range(len(tb) - 2):
      val = val * 10 + Resistor.band_val(tb[i])
    val *= mul
    return Attributize(temp=temp,
                       mul=mul,
                       tol=tol,
                       val=val,
                       valstr=Resistor.valstr(val))


