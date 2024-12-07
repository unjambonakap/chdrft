
from ipykernel.ipkernel import IPythonKernel
from ipykernel.embed import embed_kernel
from IPython.terminal import ipapp
import glog

import logging

a = ipapp.TerminalIPythonApp.instance()
a.initialize()
a.start()

print('aaa')
a.start()
