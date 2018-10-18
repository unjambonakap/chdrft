from . import sock
import socket


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
