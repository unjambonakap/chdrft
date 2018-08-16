#!/usr/bin/env python

from chdrft.main import app
from chdrft.utils.misc import Arch, misc_backend
from chdrft.emu.structures import StructureReader, BufferMemReader
from chdrft.emu.code_db import code
import ctypes
import jsonpickle

global flags, cache
flags = None
cache = None

def args(parser):
  pass


def test1():
  buf=b'\xc7\x0b@\x00\x00\x00\x00\x00\x04\x00\x00\x04\x00\x00\x00\x00\x80\x16\xa3\xf7\xff\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

  reader=StructureReader(None)
  res=reader.parse_one(code.g_asm.typs.sigaction, buf)
  print(res)
  a=jsonpickle.encode(res)
  print(a)
  tmp=jsonpickle.decode(a)
  print(tmp)
  print(tmp.typ)
  return
  b1=res.sa_mask.raw
  b2=ctypes.c_char_p(b1)

  libc=ctypes.CDLL('libc.so.6')
  print(code.g_code.cats)
  for i in range(1, 29):
    print(code.g_code.cats.sig.SIG[i])
    res=libc.sigismember(b2, ctypes.c_int32(i))
    print(res)
  #ctypes.c_char_p(

def main():
  test1()

app()
