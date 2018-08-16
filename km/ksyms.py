#!/usr/bin/env python

import subprocess as sp


class KSyms:

    def __init__(self):
        self.load()

    def load(self):
        pass

    def get_sym_addr(self, sym):
        cmd = r'cat /proc/kallsyms | grep -E "\s{}($|\s)"'.format(sym)
        res = sp.check_output(cmd, shell=True).decode()
        res=res.split(' ')[0]
        return int(res, 16)
