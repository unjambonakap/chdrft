#!/usr/bin/env python

from chdrft.utils.misc import cwdpath
from chdrft.cmds import Cmds
import os
import math


def to_greyscale(src, dest):
    print(math)
    sz = os.stat(src).st_size
    w = int(math.ceil(math.sqrt(sz)))
    h = (sz + w - 1) // w

    with open(src, 'rb') as infile, open(dest, 'wb') as outfile:
        outfile.write("""P5
{} {}
255
""".format(w,h).encode())
        outfile.write(infile.read(sz))
        rem = h * w - sz
        outfile.write(b'\x00' * rem)

