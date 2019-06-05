#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z

import tornado.web
import tornado.httpserver
from tornado import ioloop, gen
import asyncio

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)
  parser.add_argument('--port', type=int, default=8080)

class VarsHandler(tornado.web.RequestHandler):
  def get(self):
    vx = self.settings['vx']
    self.write(vx)



def make_debug_app(vx):
  return tornado.web.Application(
      [
          (r"/debug/vars", VarsHandler),
      ], vx=vx, debug=0,
  )

def run_server(app, port, threaded=False):
  #http_server = tornado.httpserver.HTTPServer(app)

  def go():
    asyncio.set_event_loop(asyncio.new_event_loop())
    #http_server.listen(port)
    app.listen(port)
    tornado.ioloop.IOLoop.instance().start()


  if threaded:
    Z.threading.Thread(target=go).start()
  else:
    go()
  # call ret.stop() to stop
  return app


def test(ctx):
  res = {}
  res['a'] = []
  res['b'] = 'abcdef'
  x = run_server(make_debug_app(res), ctx.port, threaded=1)
  i =0
  try:
    while 1:
      i += 1
      res['a'].append(i)
      Z.time.sleep(0.5)
  except KeyboardInterrupt:
    print('Finish')
    pass

  x.stop()



def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
