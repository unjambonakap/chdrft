#!/usr/bin/env python
from . import elf
import elftools
import traceback as tb


def libc_match(libc_file, pattern):
    cur=None
    try:
        x=elf.ElfUtils(filename=libc_file)
        cur={k:x.get_dyn_symbol(k) for k in pattern.keys()}
    except elftools.common.exceptions.ELFError:
        print('got failure for ', libc_file)
        tb.print_exc()
        return False
    except KeyError:
        return False

    k0=next(iter(cur))
    remap=pattern[k0]-cur[k0]
    for k in cur.keys():
        print(k, hex(cur[k]+remap), hex(pattern[k]))
        if cur[k]+remap!=pattern[k]:
            return False
    return True




