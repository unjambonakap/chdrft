from . import sock
import socket
import ssl
import os


class Connection(sock.Sock):

  def __init__(self, host, port=None, udp=False, udp_bind_port=0, **kwargs):
    if port is None:
      self.conn = host
    else:
      self.conn = (host, port)
    self.udp = udp
    self.udp_bind_port = udp_bind_port
    self.sock_type  = get_sock_type(self.conn)

    super().__init__(**kwargs)

  def _connect(self):
    s = socket.socket(self.sock_type, socket.SOCK_DGRAM if self.udp else socket.SOCK_STREAM)
    if not self.udp or self.sock_type == socket.AF_UNIX:
      s.connect(self.conn)

    if self.udp:
      s.bind(('0.0.0.0', self.udp_bind_port))
      print('SNDBUF >>> ', s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
    else:
      if self.sock_type == socket.AF_INET:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    s.setblocking(0)
    self.set_fileobj(s)
    super()._connect()

  def _do_write(self, data):
    if self.udp:
      return self.fileobj.sendto(data, self.conn)
    else:
      return super()._do_write(data)

  def _do_read(self, n):
    if self.udp:
      buf,addr = self.fileobj.recvfrom(n)
      return buf
    else:
      return super()._do_read(n)
    return  self.fileobj.recv(n)


def get_sock_type(host):
  if isinstance(host, str):
    return socket.AF_UNIX
  else:
    return socket.AF_INET

class Server(sock.Sock):

  def __init__(self, port):
    if isinstance(port, str):
      self.conn = port
    else:
      self.conn = ('', port)
    super().__init__()


  def _connect(self):

    typ = get_sock_type(self.conn)
    s = socket.socket(typ, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(self.conn)
    if typ == socket.AF_UNIX:
      self.callback(lambda: os.remove(self.conn))
    s.listen(1)
    ns, addr = s.accept()
    self.addr = addr
    if typ != socket.AF_UNIX:
      ns.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    ns.setblocking(0)
    self.set_fileobj(ns)
    super()._connect()

class SSLConnection(sock.Sock):

  def __init__(self, host, port, cert, **kwargs):
    self.conn = (host, port)
    self.cert = cert
    super().__init__(**kwargs)

  def _connect(self):


    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ssl_sock = ssl.wrap_socket(s, certfile=self.cert, keyfile=None, password=None)
    ssl_sock.connect(self.conn)
    ssl_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    ssl_sock.setblocking(0)
    self.set_fileobj(ssl_sock)
    super()._connect()

