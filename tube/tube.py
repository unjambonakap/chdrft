#!/usr/bin/env python

import chdrft.utils as utils
import chdrft.utils.misc as misc
from chdrft.utils.misc import PatternMatcher
import traceback as tb
import select
import time
import glog
from chdrft.utils.fmt import Format
from contextlib import ExitStack
import sys
import threading
import curses.ascii

from datetime import datetime, timedelta


class Tube(ExitStack):

  def __init__(self, logfile=None):
    super().__init__()
    self.running = False
    self.buf = b''
    self.recv_size = 4096
    self.log = 0
    self.logfile=logfile
    self.eof = 0

  def _enter(self):
    raise "virtual_pur"

  def _send(self, data):
    raise "virtual_pur"

  def _recv(self, n, timeout):
    raise "virtual_pur"

  def recv_base(self, n, timeout):
    res =  self._recv(n, timeout)
    if self.logfile: self.logfile_obj.write(res)
    return res

  def _shutdown(self):
    raise "virtual_pur"

  def normalize_timeout(self, timeout):
    if isinstance(timeout, int):
      return misc.Timeout.from_sec(timeout)
    elif isinstance(timeout, float):
      return misc.Timeout.from_sec(timeout)
    return timeout

  def read(self, n):
    return self.recv(n)

  def recv_silent(self, n, timeout=None):
    try:
      return self.recv(n, timeout)
    except misc.TimeoutException as e:
      return None

  def _raise_timeout(self):
    raise misc.TimeoutException()

  def recv(self, n, timeout=None):
    data = b''
    timeout = self.normalize_timeout(timeout)

    while True:
      take = self.buf[:n]
      self.buf = self.buf[n:]
      n -= len(take)
      data += take
      if n == 0:
        return data

      if len(data) > 0:
        return data

      res = self.recv_base(self.recv_size, timeout=timeout)
      if res is None:
        self.eof = 1
        raise EOFError
      glog.debug('recv: %d >> %s', len(res), res)

      self.buf += res

  def recv_fixed_size(self, size, timeout=None):
    return self.recv_until(lambda x: -1 if len(x) < size else size, timeout)

  def readline(self):
    return self.recv_until(b'\n').rstrip()

  def recv_until(self, matcher, timeout=None):
    timeout = self.normalize_timeout(timeout)
    matcher = PatternMatcher.Normalize(matcher)
    cur = self.buf
    while True:
      pos = matcher(cur)
      if pos is not None and pos != -1:
        self.buf = cur[pos:]
        return cur[:pos]

      try:
        tmp = self.recv_base(self.recv_size, timeout)
        self.buf = cur
        glog.debug('recv_until: %d >> got=%s, cur=%s', len(tmp), tmp, cur)
        if len(tmp) == 0:
          return
      except:
        self.buf = cur
        raise

      if tmp is None:
        self.buf = cur
        self.eof = 1
        raise EOFError
      cur += tmp

  def trash(self, timeout=None):
    timeout = self.normalize_timeout(timeout)
    res = bytearray()
    res += self.buf
    self.buf = b''
    if timeout is None:
      timeout = misc.Timeout.from_sec(0)
    while True:
      try:
        tmp = self.recv_base(self.recv_size, timeout)
        glog.debug('Trashing %s', tmp)
      except EOFError:
        self.eof = 1
        raise
      except:
        break

      if tmp is None:
        self.eof = 1
        raise EOFError
      res += tmp
    return res

  def get_to_end(self, timeout=None):
    timeout = self.normalize_timeout(timeout)
    res = bytearray()
    res += self.buf
    self.buf = b''
    self.xres = res
    if timeout is None:
      timeout = misc.Timeout.from_sec(0)
    while True:
      try:
        tmp = None
        tmp = self.recv_base(self.recv_size, timeout)
      except EOFError:
        self.eof = 1
        pass
      except misc.TimeoutException:
        pass
      except:
        raise

      if tmp is None:
        return res
      res += tmp
      self.xres = res
    return res

  def recv_timeout(self, timeout):
    try:
      res = self.get_to_end(timeout=timeout)
    except misc.TimeoutException as e:
      res = self.xres
    return res



  def send_padded(self, data, npad):
    data=Format(data).tobytes().pad(npad, ord(' ')).v
    glog.debug('Sending data %s', data)
    return self._send(data)

  def send(self, data):
    data=Format.ToBytes(data)
    glog.debug('Sending data %s', data)
    return self._send(data)

  def send_and_expect(self, data, matcher, timeout=None):
    self.send(data)
    return self.recv_until(matcher, timeout)

  def start(self):
    self._enter()
    self.running = True

  def __enter__(self):
    super().__enter__()
    if self.logfile:
      self.logfile_obj = open(self.logfile, 'wb')
      self.enter_context(self.logfile_obj)
    self.start()
    return self

  def __exit__(self, typ, value, tb):
    glog.info('Exit tube')
    self.shutdown()
    super().__exit__(typ, value, tb)

  def shutdown(self):
    if self.running == False:
      return
    self._shutdown()
    self.running = False
    self.buf = b''

  def _recv_ready(self, fd, timeout):
    if timeout is None:
      return select.select([fd], [], []) == ([fd], [], [])
    else:
      res =  select.select([fd], [], [], timeout.get_sec()) == \
                ([fd], [], [])
      if not res:
        self._raise_timeout()


  def interactive(self, use_input=True, full_print=0, outfile=None, stop_msg=b'KAPPAQUIT'):

    print('Starting interactive mode >> ')
    done = threading.Event()
    def recv_func():
      while not done.is_set() and not self.eof:
        dx = self.recv_timeout(timeout=0.1)
        if outfile:
          outfile.write(dx)
        if dx:
          if full_print:
            print(dx)
          else:
            repl = bytearray()
            for x in dx:
              if curses.ascii.isascii(x): repl.append(x)
              else: repl.append(ord('.'))
            print(repl.decode(), end='')
            sys.stdin.flush()

    recv_thread = threading.Thread(target=recv_func)
    recv_thread.start()

    sent = bytearray()
    try:
      while True:
        if use_input:
          d = input('>> ') + '\n'
        else:
          d = sys.stdin.read(1)
        if outfile: outfile.flush()
        sent.extend(d.encode())
        if sent.find(stop_msg) != -1: break
        self.send(d)
    except Exception as e:
      tb.print_exc(e)
      pass
    done.set()
    recv_thread.join()



class TubeWrapper(Tube):
  def __init__(self, send, recv, enter=None, shutdown=None):
    super().__init__()
    self.send_func = send
    self.recv_func = recv
    self.enter_func = enter
    self.shutdown_func = shutdown

  def _enter(self):
    if self.enter_func:
      self.enter_func()

  def _send(self, data):
    return self.send_func(data)

  def _recv(self, n, timeout):
    return self.recv_func(n, timeout)

  def _shutdown(self):
    if self.shutdown_func:
      return self.shutdown_func()

