from chdrft.emu.structures import StructBuilder, SimpleStructExtractor, CategoryStructExtractor, CodeStructExtractor
class Code:
  def __init__(self):
    self._g_code = None
    self._g_asm = None

  @property
  def g_code(self):
    if self._g_code is None:
      g_code = StructBuilder()
      g_code.add_extractor(SimpleStructExtractor('/usr/include/sys/procfs.h', 'prstatus_t'))
      g_code.add_extractor(SimpleStructExtractor('/usr/include/sys/user.h', 'user_regs_struct'))
      g_code.add_extractor(CategoryStructExtractor('elf', '/usr/include/elf.h',
                                                  'EI_ ET_ EM_ ELF NT_GNU_ NT_VERSION NT_ PT_ PF_ DT_'))
      g_code.add_extractor(CategoryStructExtractor('sig', '/usr/include/signal.h', 'SIG_ SA_ sigaction SIG'))
      g_code.build()
      g_code.typs.prstatus_t.fields.pr_reg.typ = g_code.typs.user_regs_struct
      self._g_code = g_code
    return self._g_code

  @property
  def g_asm(self):

    if self._g_asm is None:
      g_asm = StructBuilder()
      #holy fuck, too deep
      g_asm.add_extractor(CategoryStructExtractor('sig_asm', '/usr/include/asm/signal.h', 'SIG_ SA_ sigaction sigset SIG, SS_'))
      g_asm.add_extractor(CategoryStructExtractor('siginfo_asm', '/usr/include/asm/siginfo.h', 'siginfo SI_'))
      g_asm.add_extractor(CategoryStructExtractor('sigcontext', '/usr/include/asm/sigcontext.h', 'sigcontext FP_'))
      g_asm.add_extractor(CategoryStructExtractor('ucontext_asm', '/usr/include/asm/ucontext.h', 'ucontext UC_'))

      rt_sigframe_code='''
      #define SS_ONSTACK 1
      #define SS_DISABLE 2 //copy

      struct rt_sigframe {
        char *pretcode;
        struct ucontext uc;
        struct siginfo info;
        /* fp state follows here */
      };
      '''
      g_asm.add_extractor(CodeStructExtractor(rt_sigframe_code, 'rt_sigframe'))

      g_asm.build()
      self._g_asm = g_asm
    return self._g_asm
code=Code()

if __name__ == '__main__':
  print(code.g_code.typs.user_regs_struct.fields)
  print(code.g_code.typs.prstatus_t.fields.pr_reg)
  print(code.g_code.typs.sigaction.fields.__sigaction_handler.typ.fields)
  print(code.g_code.consts.SIGTRAP)
  print(code.g_asm.typs)
  print(code.g_asm.typs.siginfo_t)
  print(code.g_asm.typs.rt_sigframe)

  # size checked with BUILD_BUG_ON kernel sizeof
  print(code.g_asm.typs.rt_sigframe.size)
  print(code.g_asm.typs.ucontext.size)
  print(code.g_asm.typs.siginfo.size)
  print(code.g_asm.typs.sigset_t.size)
  print(code.g_asm.consts.SS_DISABLE)
