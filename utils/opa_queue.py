from queue import Queue, Empty


class QueueWrapper:

  def __init__(self, recv_func):
    self.q = Queue()
    self.th = None
    self.recv_func = recv_func
    self.running = False
    self.err = None
    self.stop = False
    self.pending = bytearray()

  def recv(self, n, timeout=0):
    res = bytearray()

    assert n > 0

    timer = Timer()
    timer.start()
    try:
      while True:
        take = min(len(self.pending), n)
        res.extend(self.pending[0:take])
        self.pending = self.pending[take:]

        if len(res) == n:
          break
        if timeout is None:
          wait = None
        else:
          wait = timeout - timer.get().tot_sec()
          if wait < 0:
            wait = 0
        self.pending = self.q.get(timeout=wait)
    except Empty:
      pass

    return res

  def start(self):
    self.th = threading.Thread(target=self._run)
    self.th.setDaemon(True)
    self.running = True
    self.stop = False
    self.th.start()

  def _run(self):
    while not self.stop:
      try:
        r = self.recv_func()
        self.q.put(r)
      except Exception:
        self.err = tb.format_exc()
        break
    self.running = False

  def shutdown(self):
    self.stop = True
    self.th.join()
