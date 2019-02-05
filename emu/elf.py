import io
import elftools.elf.elffile as EF
from elftools.elf.sections import SymbolTableSection, NullSection
from elftools.elf.dynamic import DynamicSection
from elftools.elf.relocation import RelocationSection
from elftools.elf.constants import P_FLAGS

from chdrft.utils.misc import Dict, Attributize, opa_print, to_list, align
import chdrft.utils.misc as cmisc
from chdrft.emu.structures import Structure
from chdrft.struct.base import Range1D

MEM_FLAGS = P_FLAGS


class ElfSection(Attributize):

  def __init__(self, raw):
    super().__init__(raw, affect_elem=False)
    self.raw = raw
    self.data = None


class ElfUtils:

  def __del__(self):
    self.file.close()

  def __init__(self, filename=None, iobase=None, data=None, offset=0, load_sym=True, core=None):

    self.filename = filename
    self.core = core
    self.raw = None
    if iobase is not None:
      self.file = iobase

    elif data is not None:
      self.file = io.BytesIO(data)
      self.raw = data

    elif filename is not None:
      self.file = open(filename, 'rb')

    else:
      assert False

    if self.raw is None:
      self.raw = self.file.read()

    self.exec_segment = None
    self.symbols = {}
    self.dyn_symbols = {}
    self.relocs = {}
    self.offset = offset
    self.notes = []

    self.elf = EF.ELFFile(self.file)
    elf = self.elf

    if core is None:
      self.core = elf.header.e_type == 'ET_CORE'
    self.arch_str = self.elf.get_machine_arch()
    from chdrft.emu.binary import guess_arch
    self.arch = guess_arch(self.arch_str)

    dyn_section = self.get_section('.dynsym')
    if dyn_section and load_sym:
      for sym in dyn_section.raw.iter_symbols():
        #print(sym.name, hex(sym['st_value']))
        self.dyn_symbols[self.sanitize_sym(sym.name)] = sym['st_value']

    for rel in ('.rela.plt', '.rel.plt'):
      rel_section = self.get_section(rel)
      if not rel_section or not isinstance(rel_section.raw, RelocationSection):
        continue
      sym_tab = elf.get_section(rel_section['sh_link'])

      if not isinstance(sym_tab, NullSection):
        for rel in rel_section.raw.iter_relocations():
          sym = sym_tab.get_symbol(rel['r_info_sym'])
          off = rel['r_offset']
          self.relocs[self.sanitize_sym(sym.name)] = off

    sym_section = self.get_section('.symtab')
    if sym_section and load_sym:
      for sym in sym_section.iter_symbols():
        self.symbols[self.sanitize_sym(sym.name)] = sym['st_value']
    else:
      pass

    for seg in elf.iter_segments():
      if seg['p_type'] == 'PT_LOAD' and (seg['p_flags'] & P_FLAGS.PF_X) != 0:
        self.exec_segment = seg

    self.init_dynamic_tags()
    self.init_plt()

    self.setup_notes()

  def setup_notes(self):
    for seg in self.elf.iter_segments():
      notes = Attributize(seg)
      if notes.p_type != 'PT_NOTE':
        continue
      for note_ in notes.iter_notes():
        note = Attributize(note_)
        self.notes.append(note)
        typ = note.n_type

        st = note.n_offset
        st += self.elf.structs.Elf_Nhdr.sizeof()
        st += align(note.n_namesz, 4)
        st = st - notes.p_offset
        nd = st + note.n_descsz
        orig = self.get_seg_content(seg)
        note.data = orig[st:nd]

        from chdrft.emu.code_db import code
        if self.core:

          #print(code.g_code.cats.elf.NT_GNU_.values())
          if typ in code.g_code.cats.elf.NT_GNU_.values():  # thanks elfutils!
            typ = code.g_code.consts[typ]

          if isinstance(typ, int):
            typ = code.g_code.cats.elf.NT_[typ]
          else:
            if isinstance(typ, int):
              typ = code.g_code.cats.elf.NT_GNU_[typ]
        note.n_type = typ

        if note.n_type == 'NT_PRSTATUS':
          sx = Structure(code.g_code.typs.prstatus_t)
          sx.backend.buf.write(0, note.data)
          note.status = sx
        elif note.n_type == 'NT_SIGINFO':
          print('laaa', len(note.data))

  def init_dynamic_tags(self):
    self.dynamic_tags = {}
    self.dynamic_section = None

    self.dynamic_section = self.get_section('.dynamic')

    if self.dynamic_section and isinstance(self.dynamic_section.raw, DynamicSection):
      for x in self.dynamic_section.iter_tags():
        tag = x.entry.d_tag
        val = x.entry.d_val
        if not tag in self.dynamic_tags:
          self.dynamic_tags[tag] = []
        self.dynamic_tags[tag].append(val)

  def init_plt(self):
    plt_section = self.get_section('.plt')
    self.plt = {}


    if not plt_section:
      return
    reloc_map = {v: k for k, v in self.relocs.items()}

    plt_addr = plt_section.sh_addr

    if self.arch.typ == cmisc.Arch.x86_64:
      tb_ins = self.arch.mc.get_ins(plt_section.data, plt_addr)

      for ins in tb_ins:
        self.arch.mc.set_reg('rip', ins.address)
        addr = self.arch.mc.get_jmp_indirect(ins)
        if addr is None:
          continue
        if addr in reloc_map:
          sym = reloc_map[addr]
          self.plt[sym] = ins.address

  def sanitize_sym(self, sym):
    if isinstance(sym, str):
      sym = sym.encode('ascii')
    return sym

  def get_entry_address(self):
    return self.elf.header['e_entry'] + self.offset

  def get_dynamic_tag(self, tag):
    return self.dynamic_tags[tag]

  def get_got_plt_off(self):
    got_plt = self.get_section('.got.plt')
    return got_plt['p_offset']

  def get_symbol(self, sym):
    sym = self.sanitize_sym(sym)
    return self.symbols[sym] + self.offset

  def get_reloc(self, sym):
    sym = self.sanitize_sym(sym)
    return self.relocs[sym] + self.offset

  def get_plt(self, sym):
    sym = self.sanitize_sym(sym)
    if not sym in self.plt:
      return None
    return self.plt[sym] + self.offset

  def get_dyn_symbol(self, sym):
    sym = self.sanitize_sym(sym)
    return self.dyn_symbols[sym] + self.offset

  def get_section(self, section):
    assert isinstance(section, str)
    section = section
    #section = self.sanitize_sym(section)
    x = self.elf.get_section_by_name(section)

    if not x:
      return None
    # can get screwed with section offset tinkering, but whatev
    res = ElfSection(x)
    if x:
      off = x['sh_offset']
      sz = x['sh_size']
      res.data = self.raw[off:off + sz]
    return res

  def get_seg_content(self, seg):
    off = seg['p_offset']
    sz = seg['p_filesz']
    return self.raw[off:off + sz]

  def find_gadget(self, gadget):
    ops = self.arch.mc.get_disassembly(gadget)

    off = self.exec_segment['p_offset']
    fd = self.raw.find(ops, off, off + self.exec_segment['p_filesz'])
    if fd == -1:
      raise Exception('gadget not found')
    return self.offset + fd - off + self.exec_segment['p_vaddr']


  def find_section(self, addr):
    for x in self.elf.iter_sections():
      diff = addr - x['sh_addr']
      if 0 <= diff < x['sh_size']:
        return x
    return None

  # Map addr -> file offset
  def get_pos(self, addr):
    addr -= self.offset
    section = self.find_section(addr)
    if section is not None: return addr - section['sh_addr'] + section['sh_offset']


  def get_range(self, *args, **kwargs):
    r = Range1D(*args, **kwargs)
    pos = self.get_pos(r.low)
    assert pos is not None
    return self.raw[pos:pos + r.length()]

  def find_gadgets(self, gadget):
    ops = self.arch.mc.get_disassembly(gadget)

    off = self.exec_segment['p_offset']
    orig_off = off
    end = off + self.exec_segment['p_filesz']
    res = []
    while True:
      fd = self.raw.find(ops, off, end)
      if fd == -1:
        break
      res.append(self.offset + fd - orig_off + self.exec_segment['p_vaddr'])
      off = fd + 1
    if len(res) == 0:
      raise Exception('gadget not found')
    return res

  def get_one_ins(self, addr):
    content = self.get_range(addr, n=20)
    return self.arch.mc.get_one_ins(content, addr=addr)


def elf_main():
  ef = ElfUtils('/home/benoit/programmation/hack/rev/delroth/core_id1', core=True)
  pass


if __name__ == '__main__':
  elf_main()
