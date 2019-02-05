#!/usr/bin/env python

import struct
import sys
import curses.ascii
import argparse
import io
from chdrft.utils.misc import Attributize, struct_helper
from chdrft.utils.fmt import Format

kDefaults = Attributize(
    reverse=0,
    num_cols=4,
    head=-1,
    skip_head=0,
    word_size=4,
    endian='little',
    offset=0,
)


class xxd:

  @staticmethod
  def args(parser):
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--num_cols', type=int, default=kDefaults.num_cols)
    parser.add_argument('--head', type=int, default=kDefaults.head)
    parser.add_argument('--skip_head', type=int, default=kDefaults.skip_head)
    parser.add_argument('--word_size', type=int, default=kDefaults.word_size)
    parser.add_argument('--endian', type=str, choices=['little', 'big'], default=kDefaults.endian)
    parser.add_argument('--offset', type=int, default=kDefaults.offset)

  @staticmethod
  def FromBuf(buf, args={}):
    input = io.BytesIO(buf)
    output = io.StringIO()
    args = dict(args)
    for k, v in kDefaults.items():
      if k not in args: args[k] = v
    x = xxd(Attributize(args), input, output)
    x.xxd()
    return output.getvalue()

  def __init__(self, args, input, output):
    self.args = args
    self.input = input
    self.output = output

  def is_print(self, c):
    return curses.ascii.isgraph(c) or c == b' '

  def format_xxd_line(self, data, off):
    res = '{:08x}: '.format(off)
    data = Format(data).modpad(self.word_size, 0).v

    for v in struct_helper.get(data, size=self.word_size, nelem=-1, little_endian=self.le):
      res += ('{:0%dx} ' % (self.word_size * 2)).format(v)

    res += ' '
    for i in data:
      if self.is_print(i):
        res += chr(i)
      else:
        res += '.'

    return res

  def xxd(self):
    self.num_cols = self.args.num_cols
    self.word_size = self.args.word_size
    self.endian = self.args.endian

    bytes_per_line = self.num_cols * self.word_size
    self.le = self.endian == 'little'

    off = self.args.offset
    eof = False
    skip_headc = self.args.skip_head
    headc = self.args.head

    while not eof:
      rem = bytes_per_line
      data = b''

      while rem > 0:
        tmp = self.input.read(rem)
        if isinstance(tmp, str):
          tmp = tmp.encode('ascii')
        if tmp == b'':
          eof = True
          break
        data += tmp
        rem -= len(tmp)
      if len(data) == 0: break

      if skip_headc == 0:
        res = self.format_xxd_line(data, off)
        res += '\n'
        off += len(data)
        self.output.write(res)
      else:
        skip_headc -= 1

      if headc == 0: break
      headc -= 1

  def xxdr(self, i, o):
    assert 0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  xxd.args(parser)
  args = parser.parse_args()
  x = xxd(args, sys.stdin.buffer, sys.stdout)

  if args.reverse:
    x.xxdr()
  else:
    x.xxd()
