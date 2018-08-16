#!/usr/bin/env python


def dump_rgb32(filename, data, w, h):
  data=bytearray(data)
  for i in range(h):
    for j in range(w):
      data[i*w*4+j*4+3]=0xff
  with open(filename, 'wb') as f:
    f.write('''P7
WIDTH %d
HEIGHT %d
DEPTH 4
MAXVAL 255
TUPLTYPE RGB_ALPHA
ENDHDR
'''%(w,h))
    f.write(data)


