from . import sock
import socket
import ssl


class Connection(sock.Sock):

  def __init__(self, host, port, **kwargs):
    self.conn = (host, port)
    super().__init__(**kwargs)

  def _connect(self):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(self.conn)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setblocking(0)
    self.set_fileobj(s)
    super()._connect()

class Server(sock.Sock):

  def __init__(self, port):
    self.conn = ('', port)
    super().__init__()

  def _connect(self):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(self.conn)
    s.listen(1)
    ns, addr = s.accept()
    self.addr = addr
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

