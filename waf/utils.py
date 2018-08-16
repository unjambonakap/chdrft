from waflib.Task import Task
from waflib.Node import Node
import chdrft.waf.clang_compilation_database as wafClang
import os
import sys
from chdrft.utils.misc import Dict, failsafe, change_extension, normalize_path

compile_cmd = [
    "/usr/bin/gcc", "-nostdinc", "-isystem",
    "/usr/lib/gcc/x86_64-unknown-linux-gnu/4.9.2/include",
    "-I{kernel_dir}/arch/x86/include", "-I{kernel_dir}/arch/x86/include/generated", "-I{kernel_dir}/include",
    "-I{kernel_dir}/arch/x86/include/uapi", "-I{kernel_dir}/arch/x86/include/generated/uapi",
    "-I{kernel_dir}/include/uapi", "-I{kernel_dir}/include/generated/uapi", "-include",
    "{kernel_dir}/include/linux/kconfig.h", "-D__KERNEL__", "-Wall", "-Wundef",
    "-Wstrict-prototypes", "-Wno-trigraphs", "-fno-strict-aliasing",
    "-fno-common", "-Werror-implicit-function-declaration",
    "-Wno-format-security", "-m64", "-mno-80387",
    "-mno-fp-ret-in-387", "-mtune=generic", "-mno-red-zone",
    "-mcmodel=kernel", "-funit-at-a-time", "-maccumulate-outgoing-args",
    "-DCONFIG_AS_CFI=1", "-DCONFIG_AS_CFI_SIGNAL_FRAME=1",
    "-DCONFIG_AS_CFI_SECTIONS=1", "-DCONFIG_AS_FXSAVEQ=1",
    "-DCONFIG_AS_CRC32=1", "-DCONFIG_AS_AVX=1", "-DCONFIG_AS_AVX2=1",
    "-pipe", "-Wno-sign-compare", "-fno-asynchronous-unwind-tables",
    "-mno-sse", "-mno-mmx", "-mno-sse2", "-mno-3dnow", "-mno-avx",
    "-fno-delete-null-pointer-checks", "-O2",
    "--param=allow-store-data-races=0", "-Wframe-larger-than=2048",
    "-fno-stack-protector", "-Wno-unused-but-set-variable",
    "-fno-omit-frame-pointer", "-fno-optimize-sibling-calls",
    "-fno-var-tracking-assignments", "-Wdeclaration-after-statement",
    "-Wno-pointer-sign", "-fno-strict-overflow", "-fconserve-stack",
    "-Werror=implicit-int", "-Werror=strict-prototypes",
    "-Werror=date-time", "{flags}", "-DCC_HAVE_ASM_GOTO", "-DMODULE",
    "-DKBUILD_STR(s)=#s", "-DKBUILD_BASENAME=KBUILD_STR({basename})",
    "-DKBUILD_MODNAME=KBUILD_STR({module_name})", "{file}"]

clang_cmd = ' '.join(compile_cmd)


def configure_wafclang(ctx):
    ctx.load('clang_compilation_database',
             tooldir=os.path.dirname(wafClang.__file__))


class SrcManagement:

    def __init__(self, ctx):
        self.includes = {None: []}
        self.flags = {None: []}
        self.path = ctx.path
        self.ctx = ctx
        self.bld = None
        self.objs = []
        self.lib_deps = []
        self.deps_include = None

    def get_flags(self, f=None):
        res = self.flags_desc[None]
        if f is not None and f in self.flags_desc:
            res += self.flags_desc[f]
        return res

    def add_elem(self, typ, val, f=None):
        if not isinstance(f, list):
            f = [f]
        if isinstance(val, str):
            val = val.split(' ')

        for v in f:
            if not v in typ:
                typ[v] = []
            typ[v].extend(val)

    def add_lib_dep(self, lst):
        if isinstance(lst, str):
            lst = lst.split(' ')
        self.lib_deps.extend(lst)

    def add_includes(self, includes, f=None):
        self.add_elem(self.includes, includes, f)

    def add_flags(self, flags, f=None):
        self.add_elem(self.flags, flags, f)

    def add_objs(self, objs):
        self.objs += [self.path.make_node(x) for x in objs]

    def sanitize_obj_input(self, x):
        if isinstance(x, str):
            return x
        elif isinstance(x, Node):
            return x.name
        return x

    def build_include_flags(self, include_list):
        return ['-I{}'.format(os.path.join(self.path.get_src().abspath(), x))
                for x in include_list]

    def get_include_flags(self, f=None):
        f = self.sanitize_obj_input(f)
        if not f in self.includes:
            return []
        return self.build_include_flags(self.includes[f])

    def get_flags(self, f=None):
        f = self.sanitize_obj_input(f)
        if not f in self.flags:
            return []
        return self.flags[f]

    def get_full_flags(self, f=None):
        res = []
        res += self.get_flags(f)
        res += self.get_include_flags(f)
        if f is None:
            res += self.deps_include
        return res

    def setup(self, bld):
        self.bld = bld

        self.deps_include = []
        for dep in self.lib_deps:
            task = self.bld.get_tgen_by_name(dep)
            exported_includes = getattr(task, 'export_includes', [])
            normalized_list = [os.path.join(task.path.get_src().abspath(), x)
                               for x in exported_includes]
            self.deps_include.extend(normalized_list)
        self.deps_include = self.build_include_flags(self.deps_include)


def make_lkm(self):
    g = self.generator

    kernel_dir = g.kernel_dir
    module_name = g.module_name

    source = self.inputs[0]
    out_dir = g.path.get_bld().abspath()

    cmd = 'make -C {kernel_dir} M={module_dir}'.format(
        kernel_dir=kernel_dir,
        module_dir=out_dir)
    ret = g.bld.exec_command(cmd)

    return ret


def setup_kbuild(self):
    g = self.generator

    srcs = g.srcs
    module_name = g.module_name
    kernel_dir = g.kernel_dir

    out_dir = g.path.get_bld().abspath()
    src_dir = g.path.get_src().abspath()

    for obj in srcs.objs:
        base = obj.name
        try:
            src_file = os.path.join(src_dir, base)
            dest_file = os.path.join(out_dir, base)
            dest_dir = os.path.split(dest_file)[0]

            failsafe(lambda: os.makedirs(dest_dir))
            os.symlink(src_file, dest_file)
        except FileExistsError:
            pass

    srcs.setup(g.bld)

    module_name = g.module_name
    target = self.outputs[0]
    obj_list = [change_extension(x.name, 'o') for x in srcs.objs]
    obj_list = ' '.join(obj_list)
    fmt = """\
obj-m += {module}.o
{module}-y := {objs}
ccflags-y := {flags}
""".format(module=module_name, objs=obj_list, flags=' '.join(srcs.get_full_flags()))

    for obj in srcs.objs:
        tmp = srcs.get_full_flags(obj)
        if len(tmp) > 0:
            flags = ' '.join(tmp)
            fmt += 'CFLAGS_{} = {}\n'.format(
                change_extension(obj.name, 'o'), flags)

    target.write(fmt)

    for obj in srcs.objs:
        add_clangdb_elem(g, obj,
                         srcs.get_full_flags(None) + srcs.get_full_flags(obj),
                         module_name, kernel_dir)


def add_clangdb_elem(taskgen, obj, flags, module_name, kernel_dir):
    f = os.path.basename(obj.abspath())
    base = os.path.splitext(f)[0]
    cur_cmd = clang_cmd.format(
        flags=' '.join(flags),
        basename=base,
        kernel_dir=kernel_dir,
        file=obj.abspath(),
        module_name=module_name)

    task_data = Dict(
        {'last_cmd': cur_cmd.split(' '), 'inputs': [obj], 'cwd': kernel_dir})
    wafClang.addTask(taskgen.bld, task_data)


def add_kernel_module(ctx, module_name, kernel_dir, srcs, install_dir):
    kbuild_node = ctx.path.get_bld().make_node('Kbuild')
    module_node = ctx.path.get_bld().make_node('{}.ko'.format(module_name))
    src_dir = ctx.path.abspath()

    kernel_dir = os.path.join(src_dir, kernel_dir)

    ctx(rule=setup_kbuild,
        name='kbuild',
        always=True,
        target=kbuild_node,
        module_name=module_name,
        kernel_dir=kernel_dir,
        srcs=srcs
        )

    ctx(rule=make_lkm,
        always=True,
        name='lkm',
        source=[kbuild_node]+srcs.objs,
        target=module_node,
        module_name=module_name,
        kernel_dir=kernel_dir)

    ctx.install_files(install_dir, [module_node])
