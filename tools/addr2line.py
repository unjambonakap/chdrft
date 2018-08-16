#!/usr/bin/env python
import argparse
from chdrft.utils.misc import cwdpath
import re
import subprocess as sp

FLAGS = None


def addr2line(fil, addr):
    return sp.check_output(['addr2line', '-C', '-f', '-e', fil, addr]).decode()


def go():
    content = open(FLAGS.file, 'r').read()
    tag_begin = 'OPA_STACKTRACE_BEGIN'
    tag_end = 'OPA_STACKTRACE_END'
    tb = re.split(tag_begin, content)[1:]

    res = []
    for x in tb:
        lines = x.split(tag_end)[0]
        for line in lines.splitlines():
            u = line.lstrip().rstrip()

            m1 = re.match('(\S+)\(\)\s+\[(\w+)\]', u)
            if m1:
                res.append(addr2line(m1.group(1), m1.group(2)))
                continue
            res.append(u)

    return res


def main():
    global FLAGS

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=cwdpath, required=True)

    FLAGS = parser.parse_args()
    res = go()
    for x in res:
        x=x.rstrip()
        print('#>> ' + x)


if __name__ == '__main__':
    main()
