#!/usr/bin/python

import multiprocessing as mp
import subprocess as sp
import time
import argparse


def go():

    try:
        pythonCmd="""'python
import sol
import traceback
try:
    sol.go();
except Exception as e:
    traceback.print_exc()
finally:
    gdb.execute("q")
    pass
' """
        cmd=['gdb', '-ex', pythonCmd, './aerosol_can']
        cmd=' '.join(cmd)
        cmd+=' </dev/null'

        p=sp.Popen(cmd, shell=True)
        p.wait()

        print "DONE"
    except Exception as e:
        print "Problem >> ",  e

if __name__=='__main__':
    go()
