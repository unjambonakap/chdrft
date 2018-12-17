#!/usr/bin/env python

from chdrft.emu.base import Memory
from chdrft.emu.binary import Arch
import itertools

class Pwn:
  kMainArenaToChunkOffset = 0xb00 - 0xaa0

  def __init__(self, read, write):
    self.write = write
    self.read = read
    self.mem = Memory(reader=read, writer=write, arch=Arch.x86_64)

  def find_top(self, chunk):
    while True:
      next = chunk.next_mem
      if not next.prev_inuse:
        cur = chunk
        while cur.size != 0:
          cur = cur.prev_free
        return cur


  def get_main_arena_addr(self, chunk):
    top = self.find_top(chunk)
    arena_addr = top.addr - Pwn.kMainArenaToChunkOffset
    return arena_addr



class Chunk:

  def __init__(self, mem, addr):
    self.mem = mem
    self.addr = addr

  @property
  def size(self):
    return self.mem.read_u64(self.addr - 8)

  @property
  def next_free(self):
    return self.new(self.mem.read_u64(self.addr))
  @property
  def prev_free(self):
    return self.new(self.mem.read_u64(self.addr+8))

  @property
  def next_mem(self):
    return self.new(self.addr + (self.size&~1))

  @property
  def prev_inuse(self):
    return self.size&1

  def new(self, addr):
    return Chunk(self.mem, addr)


  def get_top(self):
    c = self
    while True:
      n = c.prev_free
      if n.addr == 0: return c
      c = n



def find_stack_top_from_environ(mem, environ_stack_addr):
  cur = environ_stack_addr - 8 -8
  for argc in itertools.count():
    if mem.read_ptr(cur) == argc:
      break
    cur -= 8
  return cur

