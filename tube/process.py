#!/usr/bin/env python

from . import tube
import subprocess as sp
import fcntl
import select
import os


class Process(tube.Tube):

  def __init__(self, args, want_stdout=True, **popen_args):
    super(Process, self).__init__()
    self.proc_args = args
    self.popen_args = popen_args
    self.proc = None
    self.want_stdout = want_stdout

  def _enter(self):
    cur_stdout = sp.PIPE
    if self.want_stdout is None:
      cur_stdout = open(os.devnull, 'wb')
    elif not self.want_stdout:
      cur_stdout = None

    self.proc = sp.Popen(self.proc_args, stdin=sp.PIPE, stdout=cur_stdout, **self.popen_args)
    if self.want_stdout:
      fd = self.proc.stdout.fileno()
      fl = fcntl.fcntl(fd, fcntl.F_GETFL)
      fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

  def _send(self, data):
    if self.proc.stdin.closed:
      raise EOFError

    written = self.proc.stdin.write(data)
    self.proc.stdin.flush()
    return written
  def close_stdin(self):
    self.proc.stdin.close()

  def _recv(self, n, timeout):
    if self.proc.stdout.closed:
      raise EOFError
    if not self._recv_ready(self.proc.stdout, timeout):
      return b''

    res = self.proc.stdout.read(n)
    if len(res) == 0:
      self.proc.stdout.close()
      raise EOFError
    return res

  def _shutdown(self):
    self.proc.terminate()
    if self.proc.poll():
      self.proc.kill()
