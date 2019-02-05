#!/usr/bin/env python

import argparse
from chdrft.utils.misc import path_from_script, cwdpath, proc_path, chdrft_executor
from chdrft.utils.misc import Target, Attributize
from chdrft.utils.string import find_closest
import socket
from plumbum import SshMachine
import os
import subprocess as sp
import logging

executor = chdrft_executor


def do_foreach(action, lst):
    res = executor.map(lambda x: action(x), lst)
    res = list(res)


class Host:

    def __init__(self, ip, arch, nthread):
        self.ip = None
        self.arch = arch
        self.nthread = nthread


def get_targets(conf_file):
    header, *lines = open(conf_file, 'r').readlines()
    attributes=['host', 'nthread', 'target']

    mp={k:find_closest(v, attributes) for k,v in enumerate(header.rstrip().split(','))}
    for x in attributes:
        assert x in mp.values()

    lst=[]
    for line in lines:
        if line.startswith('#'):
            continue
        line=line.rstrip()
        v=Attributize({mp[i]:x for i,x in enumerate(line.split(','))})
        v.target=Target.fromstr(v.target)
        lst.append(v)
    return lst


def deploy_file(src, dest, host, args):
    print(host, args.user, args.key)
    with SshMachine(host, user=args.user, keyfile=args.key) as machine:
        machine['mkdir']('-p', os.path.split(dest)[0])
        machine.upload(src, dest)


def start_binary(args, host, pidfile, *params):
    with SshMachine(host, user=args.user, keyfile=args.key) as machine:
        res = machine[args.utils_remote]("run", pidfile, args.binary_remote,
                                         *params)
        print(res)


def run(args):
    args.binary_remote = get_remote_path(args, args.binary)
    do_foreach(
        lambda x: deploy_file(args.binary, args.binary_remote, x.host, args),
        args.targets)
    do_foreach(lambda x: start_binary(args, x.host, args.pidfile_client,
                                      [args.log_client, 'client', '--server',
                                       args.server, '--nthread', str(x.nthread)]),
               args.targets)
    start_binary(args, args.server, args.pidfile_server,
                 [args.log_server, 'server', '--server', args.server])


def kill_pidfile(args, host, pidfile):
    with SshMachine(host, user=args.user, keyfile=args.key) as machine:
        machine[args.utils_remote]("kill", pidfile)


def kill(args):
    do_foreach(lambda x: kill_pidfile(args, x.host, args.pidfile_client),
               args.targets)
    kill_pidfile(args, args.server, args.pidfile_server)


def get_remote_path(args, fil):
    return os.path.join(args.deploy_dir, os.path.basename(fil))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf',
                        type=cwdpath,
                        default=path_from_script(__file__, './hosts.conf'))
    parser.add_argument('--user', type=str, default='spark')
    parser.add_argument('--key',
                        type=cwdpath,
                        default=proc_path('~/.ssh/id_rsa_spark'))
    parser.add_argument('--deploy-dir', type=cwdpath, default='/tmp/deploy')
    parser.add_argument('--server',
                        type=str,
                        default=socket.gethostname())

    p2 = parser.add_subparsers()

    run_parser = p2.add_parser('run')
    run_parser.add_argument('--binary', type=cwdpath, required=True)
    run_parser.set_defaults(func=run)

    kill_parser = p2.add_parser('kill')
    kill_parser.set_defaults(func=kill)

    args = parser.parse_args()
    args.targets= get_targets(args.conf)

    util_script = path_from_script(__file__, './utils.sh')
    args.utils_remote = get_remote_path(args, util_script)
    args.pidfile_client = os.path.join(args.deploy_dir, 'client.pid')
    args.pidfile_server = os.path.join(args.deploy_dir, 'server.pid')
    args.log_client = os.path.join(args.deploy_dir, 'client.log')
    args.log_server = os.path.join(args.deploy_dir, 'server.log')

    do_foreach(lambda x: deploy_file(util_script, args.utils_remote, x.host,
                                     args), args.targets)

    args.func(args)


if __name__ == '__main__':
    #logging.basicConfig(level = logging.DEBUG)
    main()
