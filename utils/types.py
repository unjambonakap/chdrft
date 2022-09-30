#!/usr/bin/env python
import numpy as np

def is_list(x):
  return isinstance(x, (list, tuple, np.ndarray))

def is_num(x):
  return isinstance(x, (int, float, np.int32, np.int64, np.float32, np.float64))

