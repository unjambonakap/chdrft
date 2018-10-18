#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize, cwdpath, BitOps, Ops
from chdrft.utils.path import FileFormatHelper
from chdrft.utils.fmt import Format

import elftools.elf.elffile
import elftools.elf.enums
import elftools.elf.sections
import elftools.elf.segments
import elftools.common
import glog
import io
import subprocess as sp

global flags, cache
flags = None
cache = None


def remove_if(lst, func):
  return [x for x in lst if not func(x)]

def construct_make_default(st):
  return st.parse(b'\x00'*st.sizeof())


def find_first_if(lst, func):
  for x in lst:
    if func(x): return x
  return None


class ModElf:

  def __init__(self, filename):
    self.elf = elftools.elf.elffile.ELFFile(open(filename, 'rb'))
    self.sections = []
    self.progs = []

    for i, rsection in enumerate(self.elf.iter_sections()):
      self.add_section(i, rsection)

    for section in self.sections:
      section.linked = self.find_section_id(section.raw.header.sh_link)

    for i, prog in enumerate(self.elf.iter_segments()):
      phdr = prog.header
      prog_sections = []
      x = Attributize(
          id=i,
          raw=prog,
          )

      if phdr.p_vaddr!=0:
        prog_sections = list([
            section for section in self.sections
            if phdr.p_vaddr <= section.raw.header.sh_addr < phdr.p_vaddr + phdr.p_memsz
        ])

        if len(prog_sections)>0:
          mins = min(prog_sections, key=lambda x:x.raw.header.sh_offset)
          maxs = max(prog_sections, key=lambda x:x.raw.header.sh_offset)
          x.before = mins.raw.header.sh_offset - phdr.p_offset
          x.after = phdr.p_offset + phdr.p_memsz - (maxs.raw.header.sh_offset + maxs.raw.header.sh_size)
          #assert x.before == 0, 'Bad>> hdr=%s, before=%s'%(str(phdr), x.before)

      x.sections=prog_sections
      self.progs.append(x)

    self.shstrtab_section = self.find_section_id(self.elf['e_shstrndx'])
    assert self.shstrtab_section is not None

  def find_section_id(self, cid):
    return find_first_if(self.sections, lambda x: x.id == cid)

  def find_section_by_name(self, name):
    return find_first_if(self.sections, lambda x: x.raw.name == name)

  def get_str_section(self, names):
    return b'\x00'.join(names)

  def add_section_name_pos(self, name, name_section=None):
    if name_section is None: name_section = self.shstrtab_section
    pos = len(name_section.data)
    name_section.data += name + b'\x00'
    return pos

  def add_section(self, i, rsection):
    if rsection.stream is None: data=bytearray()
    else:data=bytearray(rsection.data())

    section = Attributize(
        id=i,
        deleted=False,
        data=data,
        raw=rsection,
        default=None,
        modified=False,
        linked=None,
        new=i is None,
        new_syms=[],)
    self.sections.append(section)
    return section

  def create_new_symtab_section(self, strtab_section):
    strtab_name = b'.symtab'
    header = construct_make_default(self.elf.structs.Elf_Shdr)
    header.sh_name = self.add_section_name_pos(strtab_name)
    header.sh_type = 'SHT_SYMTAB'
    header.sh_addralign = 8
    header.sh_entsize = self.elf.structs.Elf_Sym.sizeof()

    rsection = elftools.elf.sections.Section(header, strtab_name, None)
    res = self.add_section(None, rsection)
    res.linked = strtab_section
    return res

  def create_new_strtab_section(self):
    print('KAPPA LA')
    strtab_name = b'.strtab'
    header = construct_make_default(self.elf.structs.Elf_Shdr)
    header.sh_name = self.add_section_name_pos(strtab_name)
    header.sh_type = 'SHT_STRTAB'# elftools.elf.enums.SHT_STRTAB
    header.sh_addralign = 1
    header.sh_entsize = 0

    rsection = elftools.elf.sections.Section(header, strtab_name, None)
    return self.add_section(None, rsection)

  def add_sym0(self, symtab_section):
    sym = construct_make_default(self.elf.structs.Elf_Sym)
    sym.st_name = self.add_section_name_pos(b'', name_section=symtab_section.linked)
    sym.st_value = 0
    sym.st_size = 0
    sym.st_info.bind = 'STB_LOCAL'
    sym.st_info.type = 'STT_NOTYPE'
    sym.st_other.visibility = 'STV_DEFAULT'
    sym.st_shndx = 0
    res = Attributize(raw=sym)
    symtab_section.new_syms.append(res)
    return res

  def add_sym(self, addr, size, name, section, symtab_section):
    name = Format(name).tobytes().v
    strtab_section = symtab_section.linked
    sym = construct_make_default(self.elf.structs.Elf_Sym)
    sym.st_name = self.add_section_name_pos(name, name_section=symtab_section.linked)
    sym.st_value = addr
    sym.st_size = size
    sym.st_info.bind = 'STB_GLOBAL'
    sym.st_info.type = 'STT_FUNC'
    sym.st_other.visibility = 'STV_DEFAULT'
    sym.st_shndx = section.id
    res = Attributize(raw=sym, section=section)
    symtab_section.new_syms.append(res)
    return res

  def finalize_symtab(self, symtab):
    assert self.elf.structs.Elf_Sym.sizeof() == symtab.raw.header.sh_entsize
    for sym in symtab.new_syms:
      print(type(sym.raw), sym.raw)
      symtab.data += self.elf.structs.Elf_Sym.build(sym.raw)

  #Num:    Value          Size Type    Bind   Vis      Ndx Name
  #58: 00000000004004a6    11 FUNC    GLOBAL DEFAULT   11 main

  def remove_section_num(self, cid):
    torem = self.find_section_id(cid)
    if torem is None: return
    torem.deleted = True
    self.sections.remove(torem)

  def write_phdr(self, ehdr, res):
    for prog in self.progs:
      phdr = prog.raw.header

      res.seek(ehdr.e_phoff + prog.new_id * ehdr.e_phentsize)
      print('WRITING ', phdr)
      res.write(self.elf.structs.Elf_Phdr.build(phdr))

  def write_sections_content(self, res):
    endp = 0
    for section in self.sections:
      shdr = section.raw.header
      res.seek(shdr.sh_offset)
      res.write(section.data)
      endp = max(endp, shdr.sh_offset + len(section.data))
    return endp

  def write_sections(self, ehdr, res):
    for section in self.sections:
      shdr = section.raw.header
      if shdr.sh_type == 'SHT_SYMTAB': shdr.sh_info += len(section.new_syms)
      print(shdr)

      res.seek(ehdr.e_shoff + section.new_id * ehdr.e_shentsize)
      print(shdr)
      res.write(self.elf.structs.Elf_Shdr.build(shdr))
      res.seek(shdr.sh_offset)
      res.write(section.data)


  def set_section_at(self, section, pos):
    if section.done: return pos
    section.done = True
    shdr = section.raw.header
    pos = BitOps.align(pos, shdr.sh_addralign)

    shdr.sh_size = len(section.data)
    shdr.sh_offset = pos

    section.new_offset = pos
    pos += len(section.data)
    return pos

  def prepare_sections(self, maxp):
    for section in self.sections:
      section.done = False

    seen=False
    #for prog in self.progs:
    #  prog.fixed = False
    #  print('LAA ', prog.raw.header)
    #  if prog.raw.header.p_type != 'PT_LOAD': continue
    #  if prog.raw.header.p_offset != 0: continue
    #  print('FIXING ', len(prog.sections))
    #  prog.fixed = True

    #  assert not seen
    #  seen=True
    #  end_fixed = 0
    #  for section in prog.sections:
    #    #if section.raw.name in '.gnu.hash .dynstr .interp .text .plt .rela.plt .init .fini .rodata .rela.dyn .eh_frame .eh_frame_hdr .interpr .note.ABI-tag .note.gnu.build-id .gnu.version .gnu.version_r .dynsym'.split(' '):
    #    if section.new:
    #      section.done = True
    #      end_fixed = max(end_fixed, section.raw.header.sh_offset + section.raw.header.sh_size)
    #  pos = end_fixed
    #  print('>>> ENDP >> ', hex(pos))

    #  for section in prog.sections:
    #    pos = self.set_section_at(section, pos)

    #for prog in self.progs:
    #  pos = max(pos, prog.raw.header.p_offset)
    #  for section in prog.sections:
    #    pos = self.set_section_at(section, pos)
    #    section.raw.header.sh_addr = prog.raw.header.p_vaddr + section.raw.header.sh_offset - prog.raw.header.p_offset

    # left over sections
    for section in self.sections:
      if section.raw.name == '.shstrtab': section.new = True
      if not section.new: maxp = max(maxp, section.raw.header.sh_offset + section.raw.header.sh_size)
    for section in self.sections:
      if not section.new: continue
      section.done = False
      maxp = self.set_section_at(section, maxp)


    for section in self.sections:
      shdr = section.raw.header
      if section.linked is not None:
        shdr.sh_link = section.linked.new_id
      #pos = self.set_section_at(section, pos)
    return

    for prog in self.progs:
      if prog.fixed:
        filesz = max([section.raw.header.sh_offset + section.raw.header.sh_size
          for section in prog.sections ])
        print('FIXED SIZE >> ', hex(filesz))
        prog.raw.header.p_filesz = filesz
        prog.raw.header.p_memsz = filesz

      elif len(prog.sections)>0:
        minp = min([section.raw.header.sh_offset
          for section in prog.sections ])
        maxp = max([section.raw.header.sh_offset + section.raw.header.sh_size
          for section in prog.sections ])
        minp -= prog.before
        prog.raw.header.p_offset = minp
        prog.raw.header.p_filesz = maxp - minp
        prog.raw.header.p_memsz = maxp - minp + prog.after



  def build(self):
    for i, section in enumerate(self.sections):
      section.new_id = i
    for i, prog in enumerate(self.progs):
      prog.new_id = i

    res = io.BytesIO(bytearray())
    self.elf.header.e_shnum = len(self.sections)
    align_v = 1
    ehdr = self.elf.header
    ehdr.e_shnum = len(self.sections)
    ehdr.e_phnum = len(self.progs)
    ehdr.e_shstrndx = self.shstrtab_section.new_id
    ehdr_size = self.elf.structs.Elf_Ehdr.sizeof()
    shdr_size = ehdr.e_shentsize * ehdr.e_shnum
    phdr_size = ehdr.e_phentsize * ehdr.e_phnum

    self.write_phdr(ehdr, res)
    pos = BitOps.align(ehdr.e_phoff + phdr_size, align_v)


    self.prepare_sections(pos)

    pos = BitOps.align(pos, align_v)
    pos = self.write_sections_content(res)

    pos = BitOps.align(pos, align_v)
    ehdr.e_shoff  = pos

    #for prog in self.progs:
    #  if prog.raw.header.p_type != 'PT_PHDR': continue
    #  prog.raw.header.p_offset = ehdr.e_phoff
    #  prog.raw.header.p_filesz = phdr_size
    #  prog.raw.header.p_memsz = phdr_size


    self.write_sections(ehdr, res)

    res.seek(0)
    res.write(self.elf.structs.Elf_Ehdr.build(ehdr))
    print(ehdr_size, shdr_size, phdr_size, ehdr.e_shentsize)


    return res.getvalue()

def args(parser):
  clist = CmdsList().add(test).add(test2).add(add_syms)
  clist.add(func=lambda ctx: extract_binary_syms(ctx.infile), name='extract_binary_syms')
  parser.add_argument('--infile', type=cwdpath)
  parser.add_argument('--outfile', type=cwdpath)
  parser.add_argument('--symbols', type=FileFormatHelper.Read)
  ActionHandler.Prepare(parser, clist.lst)


def add_syms(ctx):
  mod = ModElf(ctx.infile)
  strtab_section=  mod.create_new_strtab_section()
  symtab_section=  mod.create_new_symtab_section(strtab_section)

  text_section = mod.find_section_by_name('.text')
  taddr = text_section.raw['sh_addr']
  tsize = text_section.raw['sh_size']

  mod.add_sym0(symtab_section)
  for sym in ctx.symbols:
    sym = Attributize(sym)
    if sym.type != 'fcn': continue
    if not (taddr <= sym.offset < taddr + tsize): continue
    mod.add_sym(sym.offset, sym.size, sym.name, text_section, symtab_section)
  mod.finalize_symtab(symtab_section)
  res = mod.build()
  save_elf(ctx.outfile, res)

def extract_binary_syms(filename):
  import r2pipe
  p = r2pipe.open(filename)
  p.cmd('aac')
  res = p.cmdj('aflj')
  return res

def test2(ctx):
  mod = ModElf(ctx.infile)
  strtab_section=  mod.create_new_strtab_section()
  symtab_section=  mod.create_new_symtab_section(strtab_section)

  text_section = mod.find_section_by_name('.text')
  mod.add_sym0(symtab_section)
  mod.add_sym(0x0804840b, 10, b'test_sym', text_section, symtab_section)
  mod.finalize_symtab(symtab_section)
  res = mod.build()
  save_elf(ctx.outfile, res)



def save_elf(name, content):
  open(name, 'wb').write(content)
  sp.check_call('chmod +x %s'%name, shell=True)

def test(ctx):
  mod = ModElf(ctx.infile)
  res = mod.build()
  save_elf(ctx.outfile, res)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
