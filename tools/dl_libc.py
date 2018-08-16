#!/usr/bin/env python

import requests
import re
import subprocess as sp
import tempfile
import os
import shutil
import argparse
from urllib.parse import urljoin
import concurrent.futures
import traceback as tb
from chdrft.utils.libc import LibcDatabase




def go(db):
    urls = []
    urls.append(
        ('http://security.ubuntu.com/ubuntu/pool/main/e/eglibc/',
            'href="(libc6-i386_2.*_amd64.deb)"'))
    urls.append(
        ('http://ftp.fr.debian.org/debian/pool/main/e/eglibc/',
            'href="(libc6-i386_2.*_amd64.deb)"'))
    urls.append(
        ('http://ftp.fr.debian.org/debian/pool/main/g/glibc/',
            'href="(libc6-i386_2.*_amd64.deb)"'))
    urls.append(
        ('http://security.ubuntu.com/ubuntu/pool/main/g/glibc/',
            'href="(libc6-i386_2.*_amd64.deb)"'))

    #db.get_i386_from_ubuntu_x64_build('https://launchpad.net/ubuntu/+source/glibc/2.8+20081027-0ubuntu3')
    #return
    #db.get_from_ubuntu_publishing_history('glibc')
    #db.get_from_ubuntu_publishing_history('eglibc')
    #return


    for x in urls:
        db.add_from_url(x[0], x[1])

    #low_ver = (2, 3, 0)
    low_ver = (2, 7, 0)
    high_ver = (3, 1, 0)
    db.get_from_debian_snapshot(low_ver, high_ver)
    #db.get_from_ubuntu_builds()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--libc_dir', type=str, required=True)
    args = parser.parse_args()
    num_workers = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        with LibcDatabase(args.libc_dir, executor) as db:
            #db.add_libc('/usr/lib/libc.so.6')
            #db.add_libc('/usr/lib32/libc.so.6')
            go(db)
            executor.shutdown()

