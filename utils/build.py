from chdrft.utils.opa_string import find_closest

class Target:
    X86 = 'X86'
    X86_64 = 'X86_64'
    ARMv7 = 'ARMv7'

    TARGETS = [X86, X86_64, ARMv7]

    @staticmethod
    def fromstr(x):
        return find_closest(x, Target.TARGETS)
