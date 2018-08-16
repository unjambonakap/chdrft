#!/usr/bin/env python

from chdrft.utils.misc import path_here, to_list, Attributize, Arch, opa_print
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import glog
from chdrft.emu.syscall import g_sys_db

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst)


def gen_mapping(data):
  pattern = '''
  #include <map>
  #include <string>

  namespace opa {
  namespace code_gen {
  class SyscallMapper {
  public:
    SyscallMapper(){
      %s
    }
    std::map<int,std::string> mp;
  };
  SyscallMapper s_mapper;
  }
  }
  '''


  elems = []
  entries = list(data.by_num.items())
  entries.sort()
  for _,entry in entries:
    name, num = entry.syscall_name, entry.syscall_num
    elems.append('mp[%s] = "%s";'%(num, name))
  res = pattern % '\n'.join(elems)
  print(res)




def test(ctx):
  gen_mapping(g_sys_db.data[Arch.x86])


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)

app()
