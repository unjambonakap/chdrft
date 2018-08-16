#!/usr/bin/env python

from chdrft.cmds import Cmds
from chdrft.parser import grammar
from chdrft.conv.utils import to_csv

global flags
flags = None

fcf_grammar = grammar.build_grammar('''
content = ignore_lines+ (fir_content / iir_content)
ignore_lines = to_eol_w / comment
comment = "%" to_eol

fir_content = "Numerator:" to_eol_w fir_coeff+ to_eol_w+
fir_coeff = whitespaces float to_eol_w

iir_content = sos_matrix to_eol_w* scale_values to_eol_w+
sos_matrix = "SOS Matrix:" to_eol_w iir_coeff_line+ to_eol_w*
iir_coeff_line = (whitespaces iir_coeff)+ to_eol_w
iir_coeff = float
scale_values = "Scale Values:" to_eol_w scale_coeff+
scale_coeff = whitespaces float to_eol_w
''')


class FcfVisitor(grammar.BaseVisitor):
  grammar = fcf_grammar

  def __init__(self):
    super().__init__()


def to_c_array(tb):
  s = '{'
  for x in tb:
    s += str(x) + ','
  s = s[:-1] + '}'
  return s


def read_fcf(filename):
  with open(filename, 'r') as f:
    res = fcf_grammar.parse(f.read())
    return FcfVisitor().visit(res)


def fcf_to_csv(in_file):
  fx = read_fcf(in_file)
  if 'fir_content' in fx:
    return '\n'.join([str(x) for x in fx.fir_content])
  else:
    res = []
    for coeff_list, scale in zip(fx.iir_content.sos_matrix,
                                 fx.iir_content.scale_values):
      res.append(coeff_list + [scale])
    return to_csv(res)


def fcf_to_array(in_file):
  fx = read_fcf(in_file)
  if 'fir_content' in fx:
    return to_c_array(fx.fir_content)
  else:
    res = []
    for coeff_list, scale in zip(fx.iir_content.sos_matrix,
                                 fx.iir_content.scale_values):
      res.append(to_c_array([to_c_array(coeff_list), scale]))
    return to_c_array(res)

