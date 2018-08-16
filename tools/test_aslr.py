#!/usr/bin/env python

from chdrft.dbg.gdbdebugger import GdbDebugger, launch_gdb
from chdrft.utils import proc, misc

import sys
import pprint as pp
import jsonpickle


def go(argv):
    x=GdbDebugger()
    x.set_aslr(True)
    x.add_entry_bpt()

    num_tests=100
    want='/usr/lib32/libc-2.20.so'
    for i in range(num_tests):
        x.run()
        pid=x.get_pid()
        helper=proc.ProcHelper(pid)
        regions=helper.get_memory_regions()
        #jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
        #print(jsonpickle.encode(regions))

        lst=[v for v in regions.regions if v.file==want and v.perms.has('x')]
        elem=next(iter(lst))
        libc_addr=x.get_sym_address('__libc_start_main')
        seg_start=elem.start_addr
        libc_addr&=~0xfff
        seg_start&=~0xfff
        print('diff >> ', elem.start_addr-libc_addr)




if __name__=='__main__':
    launch_gdb('chdrft.tools.test_aslr', 'go', sys.argv[1], args=sys.argv[2:])


