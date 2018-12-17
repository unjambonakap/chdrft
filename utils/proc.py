import os
import re
from asq.initiators import query
import chdrft.utils.misc as cmisc


class MemoryRegionType:
  UNKNOWN = -1
  STACK = 0
  HEAP = 1
  NONE = 2


class MemoryRegionPermission:
  P = 16
  S = 8
  R = 4
  W = 2
  X = 1
  perm_mapping = {'p': P, 's': S, 'r': R, 'w': W, 'x': X}

  @staticmethod
  def get_val(char):
    char = char.lower()
    if not char in MemoryRegionPermission.perm_mapping:
      return 0
    return MemoryRegionPermission.perm_mapping[char]

  def __init__(self, desc=None):
    self.perm = 0
    if desc is not None:
      for x in desc:
        self.perm = self.perm | self.get_val(x)

  def __repr__(self):
    res = ''
    for k in 'psrwx':
      if self.perm & self.perm_mapping[k] != 0:
        res += k
      else:
        res += '-'
    return res

  def has(self, test):
    for i in test:
      if self.perm & self.perm_mapping[i] == 0:
        return False
    return True

  def __getstate__(self):
    return self.__repr__()


class MemoryRegion:
  splitter = re.compile(' +')

  def __init__(self):
    self.tid = None
    self.raw = None
    self.start_addr = None
    self.end_addr = None
    self.dev = None
    self.perms = MemoryRegionPermission()
    self.offset = None
    self.inode = None
    self.file = None
    self.typ = None
    self.size = 0

  def build(self, line):
    region_pos = 0
    perm_pos = 1
    offset_pos = 2
    dev_pos = 3
    inode_pos = 4
    file_pos = 5
    end_pos = 6

    tb = MemoryRegion.splitter.split(line, maxsplit=end_pos - 1)
    while len(tb) < end_pos:
      tb.append('')

    self.start_addr, self.end_addr = tuple(int(x, 16) for x in tb[region_pos].split('-'))
    self.raw = line
    self.perms = MemoryRegionPermission(tb[perm_pos])
    self.offset = int(tb[offset_pos], 16)
    self.dev = tb[dev_pos]
    self.inode = int(tb[inode_pos], 16)
    self.file = tb[file_pos]
    self.size = self.end_addr - self.start_addr

    if self.file.startswith('['):
      self.typ = self.file[1:-1]
    else:
      self.typ = 'file'

    if self.typ.startswith('stack'):
      tmp = self.typ.split(':')
      if len(tmp) > 1:
        assert len(tmp) == 2
        tmp = int(tmp[1])

      self.typ = 'stack'
      self.tid = tmp

  def __repr__(self):
    tb = filter(lambda x: not x.startswith('__') and not callable(getattr(self, x)) and x != 'raw',
                self.__dict__)
    return ', '.join(['%s:%s' % (a, str(getattr(self, a))) for a in tb])

  def __getstate__(self):
    state = self.__dict__.copy()
    state.pop('raw')
    return state


class MemoryRegions:

  def __init__(self):
    self.regions = None

  def __getstate__(self):
    return self.regions

  def get_elf_entry(self, pattern):
    pattern = cmisc.PatternMatcher.smart(pattern)
    return query(self.regions).where(lambda u: pattern(u.file)).order_by(
        lambda u: u.start_addr).to_list()[0]

  def build_from_maps(self, maps):
    tb = maps.rstrip().lstrip().split('\n')
    self.regions = []
    for x in tb:
      tmp = MemoryRegion()
      tmp.build(x)
      self.regions.append(tmp)

  def __str__(self):
    return self.regions.__str__()

  def get_containing_region(self, addr):
    # sorted, but idc
    return list(filter(lambda x: x.start_addr <= addr and addr < x.end_addr, self.regions))


class ProcHelper:
  maps_file = '/proc/%d/maps'
  exe_file = '/proc/%d/exe'
  status_file = '/proc/%d/status'
  mem_file = '/proc/%d/mem'

  def __init__(self, pid=None):
    self.pid = pid
    self.mem_file = ProcHelper.mem_file % pid

  def get_map_content(self):
    with open(self.maps_file % self.pid, 'r') as f:
      return f.read()


  def get_memory_regions(self):
    maps=self.get_map_content()
    res = MemoryRegions()
    res.build_from_maps(maps)
    return res

  def get_exe_path(self):
    return os.readlink(self.exe_file % self.pid)

  def get_exe_path(self):
    return os.readlink(self.exe_file % self.pid)

  def get_ppid(self):
    with open(self.status_file % self.pid, 'r') as f:
      status = f.read()
      m = re.search('PPid:\W+([0-9]+)', status)
      return int(m.group(1))


class ProcTree:

  def __init__(self):
    self.nodes = None

  def _create_or_get(self, pid):
    if not pid in self.nodes:
      self.nodes[pid] = {'pid': pid, 'children': []}
    return self.nodes[pid]

  def refresh(self):
    helper = ProcHelper()
    self.nodes = {}
    for d in os.listdir('/proc'):
      if not re.match('[0-9]+', d):
        continue
      pid = int(d)
      helper.pid = pid
      try:
        ppid = helper.get_ppid()
        self._create_or_get(ppid)['children'].append(self._create_or_get(pid))
      except:
        pass
