#!/usr/bin/env python

import struct
import sys
import curses.ascii
import argparse
import io
from chdrft.utils.misc import Attributize


class xxd:

  @staticmethod
  def args(parser):
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--num_cols', type=int, default=4)
    parser.add_argument('--head', type=int, default=-1)
    parser.add_argument('--skip_head', type=int, default=0)
    parser.add_argument('--word_size', type=int, default=4)
    parser.add_argument('--endian', type=str, choices=['little', 'big'], default='little')
    parser.add_argument('--offset', type=int, default=0)

  @staticmethod
  def FromBuf(buf, args):
    input = io.BytesIO(buf)
    output = io.StringIO()
    x = xxd(args, input, output)
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

    for i in range((len(data) + self.word_size - 1) // self.word_size):
      cur = data[i * self.word_size:(i + 1) * self.word_size]
      cur += b'\x00' * (self.word_size - len(cur))
      val = struct.unpack(self.fx, cur)[0]
      res += ('{:0%dx} ' % (self.word_size * 2)).format(val)

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
    self.fx = ''

    if self.endian == 'little':
      self.fx += '<'
    elif self.endian == 'big':
      self.fx += '>'
    else:
      assert False, 'Bad endianness'

    if self.word_size == 4:
      self.fx += 'I'
    elif self.word_size == 8:
      self.fx += 'Q'
    else:
      assert False, 'bad word size: %d' % self.word_size

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


      if skip_headc == 0:
        res = self.format_xxd_line(data, off)
        res += '\n'
        off += len(data)
        self.output.write(res)
      else: skip_headc -= 1

      if headc == 0: break
      headc -= 1

  def xxdr(self, i, o):
    assert 0



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  self.args = self.parser.parse_args()
  x = xxd(parser, sys.stdin.buffer, sys.stdout)

  if args.reverse:
    x.xxdr()
  else:
    x.xxd()
