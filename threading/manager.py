#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
from plumbum import SshMachine
import os
import subprocess as sp
import logging

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1, init=init)
  parser.add_argument('--dependencies-conf', type=Z.FileFormatHelper.Read)
  parser.add_argument('--workers-conf', type=Z.FileFormatHelper.Read)
  parser.add_argument('--user', type=str, default='benoit')
  parser.add_argument('--key', type=cmisc.cwdpath)
  parser.add_argument('--deploy-dir', type=cmisc.cwdpath, default='/tmp/cloudy_deploy')
  parser.add_argument('--dry-run', action='store_true')
  parser.add_argument('--no-deploy', action='store_true')
  parser.add_argument('--from-python', action='store_true')


def do_foreach(action, lst):
  res = cmisc.chdrft_executor.map(lambda x: action(x), lst)
  res = list(res)


def deploy_file(x, src, dest, args):
  if flags.no_deploy: return
  glog.info(f'Deploy file src={src}, dest={dest}, host={x.host}')
  if flags.dry_run: return
  x.machine['mkdir']('-p', os.path.split(dest)[0])
  x.machine.upload(src, dest)


def start_binary(x, args, pidfile, *params):
  glog.info(f'Start binary host={x.host}, pidfile={pidfile}, params={params}')
  if flags.dry_run: return
  res = x.machine[args.utils_remote](args.deploy_dir, "run", pidfile, args.logfile, *params)
  print(res)


def kill_pidfile(x, args, pidfile):
  glog.info(f'Kill Pidfile host={x.host}, pidfile={pidfile}')
  if flags.dry_run: return
  x.machine[args.utils_remote](args.deploy_dir, "kill", pidfile)


def kill(args):
  glog.info(f'Kill args={args}')
  if flags.dry_run: return
  do_foreach(lambda x: kill_pidfile(x, args, args.pidfile), args.workers)
  #kill_pidfile(args, args.server, args.pidfile_server)

def wait(args):
  glog.info(f'wait args={args}')
  if flags.dry_run: return
  for x in args.workers:
    if x.server:
      x.machine[args.utils_remote](args.deploy_dir, "wait", args.pidfile)


def get_remote_path(args, fil):
  return os.path.join(args.deploy_dir, os.path.basename(fil))

def make_dep(**kw):
  return cmisc.Attributize(default=False, **kw)

def get_deps(dep):
  content = sp.check_output(['ldd', dep.file])
  for match in Z.re.finditer('(\S+) => (\S+)', content.decode()):
    b = match.group(2)
    yield make_dep(file=b)
  cx = sp.check_output(['readelf', '--program-header', dep.file])
  interp = Z.re.search('Requesting program interpreter: (\S+)\]', cx.decode())
  if interp is not None:
    ii = make_dep(file=interp.group(1))
    dep.interp = ii
    yield ii


def process_tgt_file(filename):
  for fil in open(filename, 'r').readlines():
    fil = fil.strip()
    cur = make_dep(file=fil)
    yield cur
    if fil.endswith('.so'):
      for dep in get_deps(cur):
        yield dep


def run(args):
  cmdline = []
  filelist = set()
  dependencies = []
  dependencies.append(make_dep(file=args.util_script))

  binary = None
  for dependency in args.dependencies:
    if dependency.tgt_file:
      dependencies.extend(process_tgt_file(dependency.tgt_file))
    else:
      dependencies.append(dependency)
      if dependency.deps:
        dependencies.extend(get_deps(dependency))
      if dependency.binary:
        binary = dependency

  assert binary is not None

  with Z.tempfile.TemporaryDirectory() as dpath:
    for dependency in dependencies:
      tmpfile = os.path.join(dpath, os.path.basename(dependency.file))
      Z.shutil.copy2(dependency.file, tmpfile)
      dependency.dstfile = os.path.join(args.deploy_dir, os.path.basename(dependency.file))

    content = None
    if binary.interp:
      content = f'''#!/bin/bash
  export LD_LIBRARY_PATH={args.deploy_dir}
  export PYTHONPATH=$PYTHONPATH:{args.deploy_dir}
  time {binary.interp.dstfile} {binary.dstfile} $@
  '''
    elif args.from_python:
      content = f'''#!/bin/bash
  source /home/benoit/.bashrc
  export LD_LIBRARY_PATH={args.deploy_dir}
  export PYTHONPATH=$PYTHONPATH:{args.deploy_dir}
  workon env3.7
  time {binary.dstfile} $@
  '''
    if content is None:
      start_cmd = [binary.dstfilei]
    else:

      glog.info(f'Runner content : {content}')
      runfilename = os.path.join(dpath, 'run.sh')
      with open(runfilename, 'w') as runfile:
        runfile.write(content)
      Z.os.chmod(runfilename, 0o755)
      start_cmd = [os.path.join(args.deploy_dir, 'run.sh')]

    print(binary)

    rsync_cmd = [
        'rsync',
        '-avzr',
    ]
    if args.key:
      rsync_cmd.extend(['-e' f'ssh -i {args.key}'])

    rsync_cmd.append(f'{dpath}/')
    for worker in args.workers:
      cmd = rsync_cmd + [f'{args.user}@{worker.host}:{args.deploy_dir}']
      glog.info(f'Syncing >> {cmd}')
      if not flags.dry_run: sp.check_output(cmd)

  server = None
  for worker in args.workers:
    if worker.server:
      server = worker

  for dependency in dependencies:
    if dependency.flag:
      cmdline.extend([dependency.flag, dependency.dstfile])

  def do_start(x):
    rem_args = [
        '--cloudy_action', ['client', 'both'][x.server], '--cloudy_server', Z.socket.gethostbyname(server.host),
        '--cloudy_nthread',
        str(x.nthread)
    ] + cmdline
    if args.from_python:
      rem_args = ['--', ' '.join(rem_args)]
    rem_args = args.other_args + rem_args
    start_binary(x, args, args.pidfile, start_cmd + rem_args)

  do_foreach(do_start, args.workers)


def test(ctx):
  print(ctx.workers)
  print(ctx.dependencies)


def read_conf(conf):
  conf = cmisc.Attributize.RecursiveImport(conf, default=False)
  return conf


def init(ctx):

  ctx.dependencies = read_conf(ctx.dependencies_conf)
  ctx.workers = read_conf(ctx.workers_conf)

  ctx.util_script = cmisc.path_from_script(__file__, './utils.sh')
  ctx.utils_remote = get_remote_path(ctx, ctx.util_script)
  ctx.pidfile = os.path.join(ctx.deploy_dir, 'client.pid')
  ctx.logfile = os.path.join(ctx.deploy_dir, 'log.log')
  for worker in ctx.workers:
    machine = SshMachine(worker.host, user=ctx.user, keyfile=ctx.key)
    worker.machine = app.global_context.enter_context(machine)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
