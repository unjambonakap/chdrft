#!/usr/bin/env python

import time
import server_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Master(server_pb2.EarlyAdopterMasterServer):

  def LaunchJob(self, request, context):
    return server_pb2.RetVal(code=0)


def serve():
  server = msg_pb2.early_adopter_create_Master_server(
      Master(), 50051, None, None)
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop()

if __name__ == '__main__':
  serve()
