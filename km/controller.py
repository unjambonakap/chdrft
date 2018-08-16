import pyopa
import ctypes


class Controller(pyopa.ModuleController):

    def __init__(self):
        super().__init__()
        res = self.initialize()
        assert pyopa.opa_is_success(res)
        provider = pyopa.KernSymProvider()
        provider.initialize()
        self.setup_all_kernsyms(provider.this)

    def peek2(self, addr, n):
        buf = ctypes.create_string_buffer(n)
        addr = ctypes.c_uint64(addr).value
        err, read = self.peek(ctypes.addressof(buf), addr, n)
        assert pyopa.opa_is_success(err)
        return ctypes.string_at(ctypes.addressof(buf), read)

    def poke2(self, dest_addr, src):
        buf = ctypes.c_buffer(src)
        dest_addr = ctypes.c_uint64(dest_addr).value
        err, written = self.poke(dest_addr, ctypes.addressof(buf), len(src))
        assert pyopa.opa_is_success(err)
        return written

    def get_sysnum(self, syscall):
        return getattr(pyopa, '__NR_{}'.format(syscall))

    def __del__(self):
        self.finalize()
