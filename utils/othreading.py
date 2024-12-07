import threading
import contextlib
import chdrft.utils.misc as cmisc
import pydantic
import asyncio
import glog
import time


class ThreadGroup(cmisc.PatchedModel):
  group: list[threading.Thread] = pydantic.Field(default_factory=list)
  ev: threading.Event = pydantic.Field(default_factory=threading.Event)
  disable_async: bool = False
  asyncs: list = pydantic.Field(default_factory=list)
  context: cmisc.ExitStackWithPush = pydantic.Field(default_factory=cmisc.ExitStackWithPush)
  done_cbs: list = cmisc.pyd_f(list)
  started: bool = False

  def run_async(self):
    el = asyncio.new_event_loop()
    el.run_until_complete(self.async_runner())

  def __post_init__(self):
    if not self.disable_async:
      self.add(threading.Thread(target=self.run_async))

  @contextlib.contextmanager
  def enter(self):
    with self.context:
      self.start()
      try:
        yield self
      except Exception as e:
        glog.exception(e)
      finally:
        for x in self.done_cbs:
          x()
        self.stop()
        self.join()

  def add_timer(self, get_freq, action, force_async=0):

    is_async = False
    if force_async or asyncio.iscoroutinefunction(action):

      is_async = True
      action = cmisc.logged_failsafe_async(action)
    else:
      action = cmisc.logged_failsafe(action)

    async def timer_func(ev: threading.Event):
      await asyncio.sleep(1 / get_freq())

      while not ev.is_set():
        wait = 1 / get_freq()
        start = time.monotonic()

        if is_async:
          await action()
        else:
          action()
        ellapsed = time.monotonic() - start
        wait = max(wait / 2, wait - ellapsed)
        await asyncio.sleep(wait)

    self.asyncs.append(timer_func)

  def add_rich_monitor(self, get_freq, cb):
    import rich.live, rich.json
    live = rich.live.Live(refresh_per_second=10)
    self.context.pushs.append(live)
    self.add_timer(get_freq, lambda: live.update(rich.json.JSON(cb())))

  async def async_runner(self):
    glog.info('Async runner go')
    waitables = []

    while not self.ev.is_set():
      while self.asyncs:
        obj = cmisc.logged_failsafe_async(self.asyncs.pop())
        waitables.append(asyncio.create_task(obj(self.ev)))
      await asyncio.sleep(1e-3)

    await asyncio.gather(*waitables)

    glog.info('Async runner go')

  def add(self, th: threading.Thread = None, func=None, force_async=False):
    if func is not None:

      if force_async or asyncio.iscoroutinefunction(func):
        self.asyncs.append(func)
        return

      assert th is None
      func = cmisc.logged_failsafe(func)
      th = threading.Thread(target=func, kwargs=dict(ev=self.ev))

    if self.started:
      th.start()
    self.group.append(th)

  def should_stop(self) -> bool:
    return self.ev.is_set()

  def stop(self):
    self.ev.set()

  def start(self):
    self.started = True
    [x.start() for x in self.group]

  def join(self):
    [x.join() for x in self.group]
