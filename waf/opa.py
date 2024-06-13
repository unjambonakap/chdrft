from waflib.Tools.ccroot import link_task, stlink_task
from waflib.Tools import c_preproc
from waflib.TaskGen import extension, feature
import waflib.TaskGen as TaskGen
import waflib.Task as Task
from waflib import Errors, Options
from waflib.Build import BuildContext, InstallContext
import waflib.Context as Context
import waflib.Tools.ccroot as ccroot
import waflib.Configure as Configure
import os
import re
import traceback as tb
from chdrft.utils.misc import to_list, concat_flat, flatten, DictWithDefault, Attributize
from chdrft.utils.build import Target
from chdrft.utils.opa_string import FuzzyMatcher
import chdrft.waf.clang_compilation_database as wafClang
import traceback
import glob
from asq.initiators import query
import sys
from chdrft.main import app
import chdrft.utils.path as opa_paths
import waflib.Tools.cxx
import subprocess as sp

pindir = opa_paths.PinDir('')
pin_link_args = [
    '-Wl,--hash-style=sysv',
    '-L{pindir}/{pinarch}/runtime/pincrt',
    '-L{pindir}/{pinarch}/lib',
    '-L{pindir}/{pinarch}/lib-ext',
    '-L{pindir}/extras/xed-{pinarch}/lib',
]
pin_shared_args_begin = [
    '{pindir}/{pinarch}/runtime/pincrt/crtbeginS.o',
    '-Wl,-Bsymbolic',
    '-Wl,--version-script={pindir}/source/include/pin/pintool.ver',
    '-fabi-version=2',
]
pin_shared_args_end = [
    '-lpin',
    '-lxed',
    '{pindir}/{pinarch}/runtime/pincrt/crtendS.o',
    '-lpin3dwarf',
    '-ldl-dynamic',
    '-nostdlib',
    '-lstlport-dynamic',
    '-lm-dynamic',
    '-lc-dynamic',
    '-lunwind-dynamic',
]


class pincxxshlib(waflib.Tools.cxx.cxxprogram):
  "Link object files into a c++ shared library"
  inst_to = '${LIBDIR}'
  run_str = '${LINK_CXX} -shared ${LINKFLAGS} ${OPA_PIN_LINK_ARGS} ${OPA_PIN_SHARED_ARGS_BEGIN} ${CCLNK_SRC_F}${SRC} ${OPA_PIN_SHARED_ARGS_END} ${CCLNK_TGT_F}${TGT[0].abspath()} ${RPATH_ST:RPATH} ${FRAMEWORKPATH_ST:FRAMEWORKPATH} ${FRAMEWORK_ST:FRAMEWORK} ${ARCH_ST:ARCH} ${STLIB_MARKER} ${STLIBPATH_ST:STLIBPATH} ${STLIB_ST:STLIB} ${SHLIB_MARKER} ${LIBPATH_ST:LIBPATH} ${LIB_ST:LIB} ${LDFLAGS}'


try:
  import chdrft.gen.opa_clang as opa_clang
except:
  pass
from waflib import Logs

opa_swig_gen_feature = 'opa_swig_gen'
opa_link_feature = 'opa_link'
opa_strip_feature = 'opa_strip'
opa_proto_feature = 'opa_proto'


def list_files_rec(base, *paths):
  lst = []
  for path_entry in paths:
    for root, dirs, files in os.walk(os.path.join(base, path_entry)):
      for f in files:
        lst.append(os.path.join(root, f))
  return query(lst)


class protoc(Task.Task):
  # protoc expects the input proto file to be an absolute path.

  run_str = '${PROTOC} ${PROTOC_FLAGS} ${PROTOC_ST:INCPATHS} ${SRC[0].abspath()}'
  color = 'BLUE'
  ext_out = ['.h', 'pb.cc', '.py']

  def scan(self):
    """
        Scan .proto deps
        """
    node = self.inputs[0]

    nodes = []
    names = []
    seen = []

    if not node: return (nodes, names)

    search_paths = []
    if hasattr(self.generator, 'includes'):
      search_paths = [self.generator.path.find_node(x) for x in self.generator.includes]

    def parse_node(node):
      if node in seen:
        return
      seen.append(node)
      code = node.read().splitlines()
      for line in code:
        m = re.search(r'^import\s+"(.*)";.*(//)?.*', line)
        if m:
          dep = m.groups()[0]
          for incpath in search_paths:
            found = incpath.find_resource(dep)
            if found:
              nodes.append(found)
              parse_node(found)
            else:
              names.append(dep)

    parse_node(node)
    return (nodes, names)


@extension('.proto')
def process_protoc(self, node):
  proto_path = self.path.find_dir(self.proto_base).get_bld().abspath()
  #dest_dir = node.parent.get_bld().abspath()
  dest_dir = proto_path
  src_dir = node.parent.get_src().abspath()

  plugin = None
  flags = []
  flags += ['--proto_path=%s' % proto_path]
  out_var = None

  if 'cxx' in self.features:
    cpp_node = node.change_ext('.pb.cc')
    hpp_node = node.change_ext('.pb.h')
    self.create_task('protoc', node, [cpp_node, hpp_node], includes=self.includes)
    self.source.append(cpp_node)
    out_var = 'cpp_out'
    plugin = WafProtoc.GRPC_CPP

  elif 'py' in self.features:
    py_node = node.change_ext('_pb2.py')
    print('INSTALL from >> ', self.install_from)
    tsk = self.create_task(
        'protoc',
        node, [py_node],
        install_from=self.install_from,
        install_path=self.install_path,
        includes=self.includes)
    self.source.append(py_node)
    out_var = 'python_out'
    plugin = WafProtoc.GRPC_PYTHON

  else:
    print(node.abspath())
    assert False
  flags += ['--%s=%s' % (out_var, dest_dir)]

  if 'rpc' in self.features:
    flags += ['--grpc_out=%s' % dest_dir]
    flags += ['--plugin=protoc-gen-grpc=%s' % self.env[plugin][0]]
  self.env.PROTOC_FLAGS = flags
  print('PROT O FALGS ', flags)

  use = getattr(self, 'use', '')
  if not 'PROTOBUF' in use:
    self.use = self.to_list(use) + ['PROTOBUF']


@TaskGen.feature(opa_proto_feature)
@TaskGen.before_method('process_source')
def opa_proto_set_includes(self):
  ccroot.process_use(self)


@TaskGen.feature(opa_proto_feature)
@TaskGen.after_method('propagate_uselib_vars', 'process_source')
def apply_incpaths(self):
  ccroot.apply_incpaths(self)


class opa_asm(Task.Task):

  def scanner(task):
    (nodes, names) = c_preproc.scan(task)
    nodes2 = list(nodes)
    for x in nodes:

      try:
        content = open(x.get_src().abspath(), 'r').read()
        for m in re.finditer('OPA_RESOURCE_DECL\(\w+,\s*"([^"]+)"\)', content):
          fil = m.group(1)
          for inc in task.generator.includes_nodes:
            res = inc.find_node(fil)
            if res:
              nodes2.append(res)
              break

      except:
        tb.print_exc()

    return (nodes2, names)

  color = 'BLUE'
  run_str = '${AS} ${ASFLAGS} ${ASMPATH_ST:INCPATHS} ${DEFINES_ST:DEFINES} ${AS_SRC_F}${SRC} ${AS_TGT_F}${TGT}'
  scan = scanner


@extension('.s', '.S', '.asm', '.ASM', '.spp', '.SPP')
def asm_hook(self, node):
  """
    Bind the asm extension to the asm task

    :param node: input file
    :type node: :py:class:`waflib.Node.Node`
    """
  return self.create_compiled_task('opa_asm', node)


class asmstlib(stlink_task):
  "Link object files into a c static library"
  pass  # do not remove


def configure_asm(conf):
  conf.find_program(['gas', 'gcc'], var='AS')
  conf.env.AS_TGT_F = ['-c', '-o']
  conf.env.ASLNK_TGT_F = ['-o']
  conf.find_ar()
  conf.env['ASMPATH_ST'] = '-I%s'


class WafBasePkg:

  def do_register(self, ctx):
    if 'registered' in self.__dict__:
      return
    self.registered = True
    self.register(ctx)

  def register(self, ctx):
    assert 0


class WafPkg(WafBasePkg):

  def __init__(self, libname, name=None):
    if name is None:
      name = libname
    self.name = 'Waf_%s' % name
    self.libname = libname

  def register(self, ctx):
    print('REGISTEring >> ', self.libname, self.name)
    ctx.check_cfg(
        package=self.libname, args='--cflags --libs', uselib_store=self.name, mandatory=True)


class WafCeres(WafPkg):

  def __init__(self):
    super().__init__('ceres')


class WafGPMFParser(WafPkg):

  def __init__(self):
    super().__init__('gpmf-parser')

class WafNLohmannJson(WafPkg):

  def __init__(self):
    super().__init__('nlohmann_json')

class WafLemon(WafPkg):

  def __init__(self):
    super().__init__('lemon')


class WafYamlCPP(WafPkg):

  def __init__(self):
    super().__init__('yaml-cpp')


class WafAsmjit(WafPkg):

  def __init__(self):
    super().__init__('asmjit')


class Wafpng(WafPkg):

  def __init__(self):
    super().__init__('libpng')

class WafGDAL(WafPkg):

  def __init__(self):
    super().__init__('gdal')

class WafQI(WafPkg):

  def __init__(self):
    super().__init__('qi')


class WafGlew(WafPkg):

  def __init__(self):
    super().__init__('glew')


class WafQgis(WafPkg):

  def __init__(self):
    super().__init__('qgis')


class WafGLU(WafPkg):

  def __init__(self):
    super().__init__('glu')

class WafEGL(WafPkg):

  def __init__(self):
    super().__init__('egl')

class WafGL(WafPkg):

  def __init__(self):
    super().__init__('gl')

class WafGLFW(WafPkg):

  def __init__(self):
    super().__init__('glfw3')


class WafOpenSSL(WafPkg):

  def __init__(self):
    super().__init__('openssl')


class WafOpenCV(WafPkg):

  def __init__(self):
    super().__init__('opencv4')


class WafZeroMQ(WafPkg):

  def __init__(self):
    super().__init__('libzmq')


class WafGflags(WafPkg):

  def __init__(self):
    super().__init__('gflags')


class WafGlog(WafPkg):

  def __init__(self):
    super().__init__('libglog')

class WafGTest(WafPkg):

  def __init__(self):
    super().__init__('gtest')


class WafProtobuf(WafPkg):

  def __init__(self):
    super().__init__('protobuf')


class WafGSL(WafPkg):

  def __init__(self):
    super().__init__('gsl')


class WafGMP(WafBasePkg):
  name = 'WAF_GMP'

  def register(self, ctx):
    ctx.check_cxx(lib='gmp', uselib_store=self.name, mandatory=True)


class WafMPFR(WafBasePkg):
  name = 'WAF_MPFR'

  def register(self, ctx):
    ctx.check_cxx(lib='mpfr', uselib_store=self.name, mandatory=True)


class WafPThread(WafBasePkg):
  name = 'WAF_PTHREAD'

  def register(self, ctx):
    ctx.check_cxx(lib='pthread', uselib_store=self.name)


class WafFPLLL(WafBasePkg):
  name = 'WAF_FPLLL'

  def register(self, ctx):
    ctx.check_cxx(lib='fplll', uselib_store=self.name)


class WafPLLL(WafBasePkg):
  name = 'WAF_PLLL'

  def register(self, ctx):
    ctx.check_cxx(lib='plll', uselib_store=self.name)

class WafGlm(WafBasePkg):
  name = 'WAF_GLM'

  def register(self, ctx):
    ctx.check_cxx(lib='glm', uselib_store=self.name)



class WafPython(WafBasePkg):
  name = 'WAF_PYTHON'

  def register(self, ctx):
    ctx.load('python')
    ctx.check_python_version((3, 4, 0))
    ctx.check_python_headers()


class WafPythonEmbed(WafPkg):

  def __init__(self):
    super().__init__('python3-embed')

class WafCGAL(WafPkg):

  def __init__(self):
    super().__init__('cgal')



class WafSwig(WafBasePkg):

  def register(self, ctx):
    ctx.load('swig')
    assert ctx.check_swig_version() > (3, 0, 0)


class WafProtoc(WafBasePkg):
  GRPC_PYTHON = 'GRPC_PYTHON'
  GRPC_CPP = 'GRPC_CPP'
  name = ''

  def register(self, ctx):
    ctx.find_program('protoc', var='PROTOC')
    #ctx.find_program('grpc_python_plugin', var=WafProtoc.GRPC_PYTHON)
    #ctx.find_program('grpc_cpp_plugin', var=WafProtoc.GRPC_CPP)
    ctx.env.PROTOC_ST = '-I%s'



class WafPkgList:

  def __init__(self, *lst):
    self.tb = []
    for x in lst:
      self.tb.append(WafPkg(x))

  def register(self, ctx):
    for x in self.tb:
      x.register(ctx)

class WafAbsl(WafPkgList):

  def __init__(self):
    super().__init__('absl_any', 'absl_hash')


class WafGRC(WafPkgList):

  def __init__(self):
    super().__init__('gnuradio-analog', 'gnuradio-filter', 'gnuradio-trellis', 'gnuradio-uhd')


class WafUhd(WafPkg):

  def __init__(self):
    super().__init__('uhd')


class WafDwf(WafPkg):
  """Digilent waveform"""

  def __init__(self):
    super().__init__('dwf')

class WafGZ(WafPkgList):

  def __init__(self):
    super().__init__('gz-physics6', 'gz-common5', 'gz-math7', 'gz-transport12', 'gz-msgs9', 'gz-plugin2', 'gz-sim7', 'gz-gui7')

class WafEigen(WafPkg):
  """Eigen"""

  def __init__(self):
    super().__init__('eigen3')


class WafPackages:
  GZ = WafGZ()
  OpenSSL = WafOpenSSL()
  OpenCV = WafOpenCV()
  ZeroMQ = WafZeroMQ()
  GMP = WafGMP()
  MPFR = WafMPFR()
  Swig = WafSwig()
  PThread = WafPThread()
  Protobuf = WafProtobuf()
  Protoc = WafProtoc()
  GSL = WafGSL()
  PLLL = WafPLLL()
  FPLLL = WafFPLLL()
  GLFW = WafGLFW()
  Gflags = WafGflags()
  Glog = WafGlog()
  GLU = WafGLU()
  EGL = WafEGL()
  GL = WafGL()
  Glm = WafGlm()
  Glew = WafGlew()
  GDAL = WafGDAL()
  #QI = WafQI()
  png = Wafpng()
  Asmjit = WafAsmjit()
  Absl = WafAbsl()
  GRC = WafGRC()
  Uhd = WafUhd()
  YamlCPP = WafYamlCPP()
  Lemon = WafLemon()
  Python = WafPython()
  PythonEmbed = WafPythonEmbed()
  CGAL = WafCGAL()
  Qgis = WafQgis()
  Eigen = WafEigen()
  GTest = WafGTest()
  #dwf = WafDwf()
  GPMFParser = WafGPMFParser()
  Ceres = WafCeres()
  Json = WafNLohmannJson()


class WafLibs:
  Crypto_N = '@crypto'
  CryptoLa_N = '@cryptola'
  CryptoStream_N = '@cryptostream'
  Threading_N = '@threading'
  MathCO_N = '@math_co'
  MathGame_N = '@math_game'
  MathGameProto_N = '@math_game_proto'
  MathGameSwig_N = '@math_game_swig'
  Engine_N = '@engine'
  PyEngine_N = '@pyengine'
  Syscalls_N = '@syscalls'
  RapidJson_N = '@rapidjson'
  UhdLib_N = '@uhd_lib'
  UhdLibSwig_N = '@uhd_lib_swig'
  UhdLibProtoPython_N = '@uhd_lib_proto_python'

  MathCommon_N = '@math_common'
  GoogleCommon_N = '@google_common'
  MathAdv_N = '@math_adv'
  DSP_N = '@DSP'
  OR_N = '@OR'
  ORSwig_N = '@OR_swig'
  Algo_N = '@Algo'
  Common_N = '@Common'
  CommonBase_N = '@CommonBase'
  CommonBaseHdr_N = '@CommonBaseHdr'
  CommonBaseStatic_N = '@CommonBaseStatic'
  SwigCommon_N = '@SwigCommon'
  SwigMathCommon_N = '@SwigMathCommon'
  SwigAlgo_N = '@SwigAlgo'
  SwigCommon2_N = '@SwigCommon2'
  SwigCommon2H_N = '@SwigCommon2H'
  lodepng_N = '@lodepng'
  PLLL_N = '@PLLL_N'

  def __init__(self):
    self.v = {}

  def get(self, name):
    res = self.v[name]
    return res

  def set(self, name, val):
    self.v[name] = val


class WafBuildType:
  DEBUG = 'debug'
  RELEASE = 'release'
  ALL = [DEBUG, RELEASE]


def IsCPPFile(filename):
  return re.search('\.(c|cc|cpp)$', filename) is not None


def IsHeaderFile(filename):
  return re.search('\.(h|hpp)$', filename) is not None


class ClOpaWaf:
  packages = WafPackages()
  libs = WafLibs()
  build_type = WafBuildType()
  PYTHON_INSTALL_DIR = '${PREFIX}/lib'

  def __init__(self):
    self.conf_configure = False
    self.conf_options = False
    self.pin_mode = False
    self.registered = {}

  def get_swig(self, path):
    return list_files_rec(path, './swig').where(lambda x: x.endswith('.i')).to_list()

  def get_files(self, path, include_root=False):
    if include_root:
      srcs = list_files_rec(path, './').where(IsCPPFile).to_list()
      headers = list_files_rec(path, './').where(IsHeaderFile).to_list()
    else:
      srcs = list_files_rec(path, './src', './lib').where(IsCPPFile).to_list()
      headers = list_files_rec(path, './inc', './include').where(IsHeaderFile).to_list()
    return srcs, headers

  def get_asm(self, path):
    return list_files_rec(path, './src').where(lambda x: x.endswith('.S')).to_list()

  def get_proto(self, path, proto_base='proto/msg'):
    return os.path.join(path, proto_base), list_files_rec(
        path, proto_base).where(lambda x: x.endswith('.proto')).to_list()

  def options(self, ctx):
    if self.conf_options:
      return

    ctx.load('compiler_cxx')
    ctx.load('compiler_c')
    ctx.load('cs')
    ctx.load('waf_unit_test')
    ctx.load('python')
    ctx.load('swig')

    ctx.add_option('--arch', default=Target.X86_64, choices=Target.TARGETS)
    ctx.add_option('--build', choices=WafBuildType.ALL, default=WafBuildType.RELEASE)
    ctx.add_option('--target_pin', action='store_true')
    self.conf_options = True

  def configure(self, ctx):
    if self.conf_configure:
      return

    ctx.load('compiler_cxx')
    ctx.load('compiler_c')
    ctx.load('cs')
    ctx.load('waf_unit_test')
    configure_asm(ctx)

    self.conf_configure = True
    if ctx.options.build == WafBuildType.RELEASE:
      ctx.env.append_value('CXXFLAGS', ['-O3'])
      ctx.env.append_value('CFLAGS', ['-O3'])
    else:
      ctx.env.append_value('CXXFLAGS', ['-O0', '-g'])
      ctx.env.append_value('CFLAGS', ['-O0', '-g'])

    if ctx.options.arch == Target.X86_64:
      ctx.env.append_value('SWIGFLAGS', '-D__x86_64 -D__x86_64__ -Wno-error=stringop-overflow')

    ctx.env.pincxxshlib_PATTERN = 'lib%s.so'
    if ctx.options.target_pin:
      self.pin_mode = True
      args = [
          '-DBIGARRAY_MULTIPLIER=1',
          '-Werror',
          '-Wno-unknown-pragmas',
          '-D__PIN__=1',
          '-DPIN_CRT=1',
          '-fno-stack-protector',
          '-fno-exceptions',
          '-funwind-tables',
          '-fasynchronous-unwind-tables',
          '-fno-rtti',
          '-DTARGET_LINUX',
          '-fabi-version=2',
          '-fomit-frame-pointer',
          '-fno-strict-aliasing',
          '-std=c++17',
          '-nostdinc++',
      ]

      if ctx.options.arch == Target.X86:
        arch_mods = ['ia32', 'x86']
        args.extend([
            '-DTARGET_IA32',
            '-DHOST_IA32',
        ])
      else:
        arch_mods = ['intel64', 'x86_64']
        args.extend([
            '-DTARGET_IA32E',
            '-DHOST_IA32E',
            '-fPIC',
        ])
      dirs = [
          'source/include/pin',
          'source/include/pin/gen',
          'extras/components/include',
          'extras/xed-{}/include/xed'.format(arch_mods[0]),
      ]

      sdirs = [
          'extras/stlport/include',
          'extras/libstdc++/include',
          'extras/crt/include',
          'extras/crt/include/arch-{}'.format(arch_mods[1]),
          'extras/crt/include/kernel/uapi',
          'extras/crt/include/kernel/uapi/asm-x86',
      ]

      global pin_shared_args_end
      global pin_shared_args_begin
      global pin_link_args
      kvs = (('OPA_PIN_SHARED_ARGS_BEGIN', pin_shared_args_begin),
             ('OPA_PIN_SHARED_ARGS_END', pin_shared_args_end),
             ('OPA_PIN_LINK_ARGS', pin_link_args),)
      for k, vs in kvs:
        for v in vs:
          ctx.env.append_value(k, v.format(pindir=pindir, pinarch=arch_mods[0]))

      dirs = [opa_paths.PinDir(x) for x in dirs]
      sdirs = [opa_paths.PinDir(x) for x in sdirs]
      print(args)
      ctx.env.append_value('CFLAGS', args)
      ctx.env.append_value('CXXFLAGS', args)

      iargs = []
      iargs += ['-I' + x for x in dirs]
      iargs += flatten([['-isystem ' + x] for x in sdirs])
      ctx.env.append_value('CFLAGS', iargs)
      ctx.env.append_value('CXXFLAGS', iargs)

#'-I../../../source/include/pin', '-I../../../source/include/pin/gen', '-isystem', '/home/benoit/programmation/tools/pin-3.2-81205-gcc-linux/extras/stlport/include', '-isystem', '/home/benoit/programmation/tools/pin-3.2-81205-gcc-linux/extras/libstdc++/include', '-isystem', '/home/benoit/programmation/tools/pin-3.2-81205-gcc-linux/extras/crt/include', '-isystem', '/home/benoit/programmation/tools/pin-3.2-81205-gcc-linux/extras/crt/include/arch-x86', '-isystem', '/home/benoit/programmation/tools/pin-3.2-81205-gcc-linux/extras/crt/include/kernel/uapi', '-isystem', '/home/benoit/programmation/tools/pin-3.2-81205-gcc-linux/extras/crt/include/kernel/uapi/asm-x86', '-I../../../extras/components/include', '-I../../../extras/xed-ia32/include/xed', '-I../../../source/tools/InstLib'

    else:
      ctx.env.append_value('CFLAGS', ['-fPIC', '-Wreturn-type',])
      ctx.env.append_value('CXXFLAGS',
                           ['-fPIC', '-std=c++2b', '-Wreturn-type', '-Wno-stringop-overflow', '-Wno-unknown-warning-option', '-Wno-attributes',
                             ])

      ctx.env.append_value('CFLAGS', ['-Wno-error'])
      ctx.env.append_value('CXXFLAGS', ['-Wno-error'])

      ctx.env.append_value('ASFLAGS', ['-D__ASSEMBLY__'])
      ctx.env.append_value('LINKFLAGS', ['-ldl', '-lm'])

    if ctx.options.arch == Target.X86:
      ctx.env.append_value('CXXFLAGS', ['-m32'])
      ctx.env.append_value('ASFLAGS', ['-m32'])
      ctx.env.append_value('LINKFLAGS', ['-m32'])
      ctx.env.append_value('CFLAGS', ['-m32'])
    ctx.env.PYTHON_INSTALL_DIR = '${PREFIX}/python/opa'

  def build(self, ctx):
    pass

  def register(self, x, ctx):
    x.do_register(ctx)

opa_waf = ClOpaWaf()


@Configure.conf
def find_ld(ctx):
  ctx.find_program('ld', var='LD')
  ctx.find_program('strip', var='STRIP')


def configure(ctx):
  ctx.find_ld()


class opa_link_tsk(Task.Task):
  run_str = "${LD} ${tsk.link_flags} -o ${TGT[0].abspath()} -T ${tsk.link_script.abspath()} ${tsk.objs}"


class opa_strip_tsk(Task.Task):
  run_str = '${STRIP} ${STRIP_FLAGS} ${SRC}'


@TaskGen.feature(opa_link_feature, opa_swig_gen_feature)
@TaskGen.after_method('apply_link', 'process_source')
def opa_use_getter(self):
  ccroot.process_use(self)
  names = self.to_list(getattr(self, 'use', []))
  self.opa_dep_tgen = [self.bld.get_tgen_by_name(x) for x in names]


@TaskGen.feature(opa_swig_gen_feature)
@TaskGen.after(opa_use_getter.__name__)
def opa_export_includes(self):
  if not hasattr(self, 'includes'):
    self.includes = []

  for x in self.opa_dep_tgen:
    self.includes.extend(x.to_incnodes(x.export_includes))


@TaskGen.feature(opa_link_feature)
@TaskGen.after(opa_use_getter.__name__)
def opa_link(self):
  obj_list = []
  for tgen in self.opa_dep_tgen:
    for task in tgen.tasks:
      for output in task.outputs:
        if output.abspath().endswith('.o'):
          obj_list.append(output)

  link_node = self.path.get_src().find_resource(self.link_script)
  target = self.path.get_bld().make_node('{}.a'.format(self.target))

  self.link_task = self.create_task(
      opa_link_tsk.__name__,
      src=obj_list + [link_node],
      tgt=target,
      link_script=link_node,
      link_flags=self.link_flags,
      objs=[x.abspath() for x in obj_list])


@TaskGen.feature(opa_strip_feature)
@TaskGen.after('apply_link')
@TaskGen.after(opa_link.__name__)
def tgen_strip(self):

  node = self.link_task.outputs[0]
  tsk = self.create_task(opa_strip_tsk.__name__, node)
  strip_flags = self.to_list(getattr(self, 'stripflags', []))
  keep_syms = self.to_list(getattr(self, 'strip_keep_syms', []))
  for x in keep_syms:
    strip_flags.extend(['-K', x])

  keep_syms = self.to_list(getattr(self, 'strip_sections', []))
  for x in keep_syms:
    strip_flags.extend(['-R', x])
  tsk.env.STRIP_FLAGS = strip_flags

  inst = getattr(self, 'install_path', None)
  if inst:
    self.bld.install_files(inst, node)


class opa_swig_gen_tsk(Task.Task):
  vars = ['file_content', 'modifier']

  #after = ['swig']

  def runnable_status(self):
    for t in self.run_after:
      Logs.warn('KAPPA run after >> {}'.format(repr(t)))
      if not t.hasrun:
        return Task.ASK_LATER
    return Task.RUN_ME

  def run(self):
    file_content = self.env.file_content
    modifier = self.env.modifier

    target = self.outputs[0]

    index = opa_clang.OpaIndex.create_index(
        args=self.env.flags, file_content=file_content, cpp_mode=True)
    code = index.get_code(modifier)

    with open(target.abspath(), 'wb') as f:
      f.write(code.encode())


@TaskGen.feature(opa_swig_gen_feature)
@TaskGen.after(opa_export_includes.__name__)
def opa_swig_gen(self):
  tsk = self.create_task(opa_swig_gen_tsk.__name__, tgt=self.target)
  tsk.env.file_content = self.file_content
  tsk.env.modifier = self.modifier
  tsk.env.flags = ['-I{}'.format(x.abspath()) for x in self.includes]


class ExtraConf:

  def __init__(self, builder, basename, typ):
    self.typ = typ
    self.basename = basename
    self.builder = builder
    self.swig_flags = None
    self.proto_base = None
    self.clang = True
    self.binary = False
    self.post = None
    self.install = False
    self.install_path = None
    self.install_from = None
    self.use_global = True
    self.clear()
    self.deps = DictWithDefault(lambda: [])
    self.deps[BuilderType.LIB] = [BuilderType.PROTO, BuilderType.ASM]
    self.deps[BuilderType.TEST] = [BuilderType.LIB]
    self.deps[BuilderType.SAMPLE] = [BuilderType.LIB]
    self.deps[BuilderType.SWIG_H] = [BuilderType.LIB]
    self.deps[BuilderType.SWIG] = [BuilderType.LIB, BuilderType.SWIG_H]
    self.build_done = False
    for x in list(self.deps.keys()):
      new = set(self.deps[x])
      for y in self.deps[x]:
        new.update(self.deps[y])
      self.deps[x] = new

  def clear(self):
    self.includes = []
    self.libs = []
    self.exports = []
    self.sources = []
    self.features = []
    self.headers = []
    #not working
    self.deps_list = []

  def setup(self):
    self.libs = self.builder.proc_libs(self.libs)

  def update(self, register_name=None, append=True, **kwargs):

    for k, v in kwargs.items():
      assert hasattr(self, k)
      cur = getattr(self, k)

      if isinstance(cur, list):
        if append: cur += to_list(v)
        else:
          cur.clear()
          cur.extend(to_list(v))
      else:
        setattr(self, k, v)

    if register_name:
      print('registering name', register_name)
      self.builder.add_name(register_name, self.basename)
      if register_name in self.deps_list:
        self.deps_list.remove(register_name)

    return self

  def norm_path(self, source):
    if os.path.isabs(source):
      return os.path.relpath(source, self.builder.path)
    return source

  def guess_feature(self):

    features = self.features
    has_h = False
    for x in self.sources:
      if x.endswith('.cpp'):
        features.append('cxx')
      if x.endswith('.cc'):
        features.append('cxx')
      if x.endswith('.c'):
        features.append('c')
      if x.endswith('.S'):
        features.append('asm')
      if x.endswith('.py'):
        features.append('py')
      if x.endswith('.h'):
        has_h = True
    if len(features) == 0 and has_h:
      features.append('includes c')
    features = list(set(features))

    if self.binary:
      features.append('cprogram')
    else:
      if 'asm' in features:
        features.append('cxxshlib')
      else:
        if not 'cxxstlib' in features:
          features.append('cxxshlib')

    if 'cxxshlib' in features and not  'cxx' in features:
      features.append('cxx')
    tsf_features = []
    for feature in features:
      if feature == 'cxxshlib' and opa_waf.pin_mode:
        feature = 'pincxxshlib'
      tsf_features.append(feature)

    return tsf_features

  def build(self):
    if self.build_done: return
    self.build_done = 1

    if self.basename == None:
      assert len(self.sources) == 1
      self.basename = os.path.splitext(os.path.basename(self.sources[0]))[0]

    self.name = self.builder.get_target_name(self.basename)
    tb = Attributize({})
    print('GOT DEPS >> ', self.deps)
    if self.use_global:
      for x in self.deps[self.typ]:
        for j in self.builder.build_desc[x]:
          print(type(j), j.basename, self.name)
          j.build()
          print(j.name)
          self.libs.append(j.name)

    self.sources = flatten(self.sources, True)
    self.features = flatten(self.features, True)
    self.features = self.guess_feature()
    self.sources = [self.norm_path(x) for x in self.sources]
    self.deps_list = [self.builder.normalize_name(x) for x in self.deps_list]
    self.libs = [self.builder.normalize_name(x) for x in self.libs]
    mapper = 'includes export_includes:exports target:name source:sources swig_flags proto_base use:libs features install_path install_from after:deps_list'
    for x in mapper.split(' '):
      y = x.split(':')
      if len(y) == 1:
        y.append(y[0])
      attr = getattr(self, y[1])
      if attr is None:
        continue
      tb[y[0]] = attr

    if self.install and not self.install_path:
      print('INSTALL >>> ', self.features, self.sources)
      if 'py' in self.features or 'pyembed' in self.features:
        tb.install_path = opa_waf.PYTHON_INSTALL_DIR
        print('INSTALL TO ', opa_waf.PYTHON_INSTALL_DIR)
      else:
        tb.install_path = '${PREFIX}/lib'

      tb['install_path'] = tb.install_path
    if not 'install_from' in tb:
      tb['install_from'] = self.builder.ctx.path.get_bld()
    else:
      before = tb['install_from']
      tb['install_from'] = self.builder.ctx.path.get_bld().make_node(before)
      assert tb['install_from'] is not None, 'before=%s, curpath=%s %s %s' % (
          before, self.builder.ctx.path.abspath(), self.sources, self.features)

    tmp = tb._elem
    print('GOGO ', tmp)
    x = self.builder.ctx(**tmp)
    x.opa_data = self

    if 0 and self.install:
      if 'pyembed' in self.features:
        assert len(self.sources) == 1

        modulename = os.path.splitext(os.path.basename(self.sources[0]))[0]
        filename = os.path.join(os.path.dirname(self.sources[0]), modulename + '.py')
        from waflib import Build
        self.builder.ctx.post_mode = Build.POST_LAZY
        self.builder.ctx.add_group()
        generated_py = self.builder.ctx.path.find_or_declare(filename)
        print('GOT FILE >> ', generated_py, filename, self.sources,
              self.builder.ctx.path.get_bld().abspath())
        self.builder.ctx(
            feature='py',
            source=generated_py,
            name='kappa',
            install_path=tb.install_path,
            install_from=self.builder.ctx.path.get_bld())

    if self.post:
      self.post(x)
    if self.clang:
      wafClang.addHeaders(x, self.headers, self.sources)


class TgenGetter:

  def __init__(self, ctx):
    self.ctx = ctx
    self.get_tgen_orig = self.ctx.get_tgen_by_name
    self.ctx.get_tgen_by_name = self
    self.matcher = FuzzyMatcher()

  def add_name(self, name):
    self.matcher.add_name(name)

  def ctx_cache(self):
    return self.ctx.task_gen_cache_names

  def rebuild(self):
    cache = self.ctx_cache()
    if not cache:
      self.matcher.reset()
      for g in self.ctx.groups:
        for tg in g:
          try:
            cache[tg.name] = tg
            self.add_name(tg.name)
          except AttributeError:
            # raised if not a task generator, which should be uncommon
            pass

  def __call__(self, name, exact=True):
    if exact:
      return self.get_tgen_orig(name)
    self.rebuild()
    ans = self.matcher.find(name)
    if ans == None:
      print('TRY It ', self.ctx_cache())
      raise Errors.WafError('Could not find a task generator for the name %r (debug=%s)' %
                            (name, self.matcher.debug))
    return self.ctx_cache()[ans[1]]


class BuilderType:
  PROTO = 0
  ASM = 1
  LIB = 2
  SWIG = 3
  TEST = 4
  SAMPLE = 5
  PROTO_PYTHON = 6
  LIB_PYTHON = 7
  SWIG_H = 8


def override_get_targets(self):
  """
        Return the task generator corresponding to the 'targets' list, used by :py:meth:`waflib.Build.BuildContext.get_build_iterator`::

                $ waf --targets=myprogram,myshlib
        """
  to_post = []
  min_grp = 0
  for name in self.targets.split(','):
    tg = self.get_tgen_by_name(name, exact=False)
    if not tg:
      raise Errors.WafError('target %r does not exist' % name)

    m = self.get_group_idx(tg)
    if m > min_grp:
      min_grp = m
      to_post = [tg]
    elif m == min_grp:
      to_post.append(tg)
  return (min_grp, to_post)


class BuildContext2(BuildContext):

  def execute(self):
    with app.global_context:
      super().execute()

  get_targets = override_get_targets


class InstallContext2(InstallContext):

  get_targets = override_get_targets
  def do_install(self, src, tgt, **kw):
    super().do_install(src, tgt, **kw)
    self.installs.append(str(tgt))

  def execute(self):
    self.installs = []
    with app.global_context:

      app.global_context.callback(lambda *args: print('CLEANUP MOFO'))
      super().execute()
      with open(os.path.join(Options.options.out, 'installs.tgt'), 'w') as f:
        f.write('\n'.join(self.installs))


class RunnerContext(BuildContext2):
  cmd = 'run'

  def __init__(self, **kw):
    super().__init__(**kw)
    self.target = Options.options.targets
    if not self.target:
      raise Errors.WafError('no target specified')

  def execute(self):
    if len(self.stack_path) == 0:
      self.getter = TgenGetter(self)

    self.restore()
    if not self.all_envs:
      self.load_envs()

    self.recurse([self.run_dir])

    if len(self.stack_path) == 0:
      x = self.get_tgen_by_name(self.target)
      if 'py' in x.opa_data.features:
        assert len(x.source) == 1
        self.run_py(x.source[1], x.use)
      else:
        if query(x.opa_data.features).where(lambda y: y.find('program')).count() > 0:
          self.run_program(x)

  def run_py(self, pyfile, use):
    x.post()

  @staticmethod
  def options(ctx):
    ctx.add_option('--target', type=str)


class WafBuilder:
  PATH_SEPARATOR = '/'
  TARGET_SEPARATOR = '_'

  def __init__(self, var_globals):
    self.libs = opa_waf.libs
    self.packages = opa_waf.packages
    self._packages = []
    self.children = None
    self.typ = BuilderType
    self.ctx = None
    self._libs = []
    self._builders = []
    self._configure_list = []
    self.var_globals = var_globals
    self.a = opa_waf

    self.target_path = None

    data = self.var_globals['__waf_data']
    self.path = os.path.dirname(data['path'])
    self.stack_path = list(data['stack_path'])

    self.build_desc = DictWithDefault(lambda: [])

  def get(self, typ, pos=0):
    assert typ in self.build_desc, typ
    return self.build_desc[typ][pos]

  def list_dir(self, subdir):
    path = os.path.join(self.path, subdir)
    if not os.path.exists(path):
      return []
    return [os.path.join(subdir, x) for x in os.listdir(path)]

  def get_children(self):
    if self.children == None:
      self.children = []
      for x in self.list_dir('.'):
        t1 = os.path.join(self.path, x)
        t2 = os.path.join(t1, 'wscript')
        if os.path.isdir(t1) and os.path.exists(t2):
          self.children.append(x)

    return self.children

  def get_target_path(self):
    if not self.target_path:
      n = len(self.stack_path) - 1
      tb = []
      cur = self.path
      for x in range(n):
        cur, tmp = os.path.split(cur)
        tb.append(tmp)
      tb.reverse()
      self.target_path = self.PATH_SEPARATOR.join(tb)
    return self.target_path

  def recurse(self):
    children = self.get_children()
    self.ctx.recurse(children)

  def has_child(self, path):
    return os.path.exists(os.path.join(self.path, path))

  def setup(self):
    for x in self.build_desc.values():
      for y in x:
        y.setup()

    lst = 'options build build2 configure run'.split(' ')
    for x in lst:
      val = WafBuilder.__dict__[x]

      def dispatcher2(val):
        return lambda ctx: self.dispatch(ctx, val)

      self.var_globals[x] = dispatcher2(val)

  def dispatch(self, ctx, func):
    self.ctx = ctx
    self.recurse()
    func(self)

  def get_target_name(self, name):
    path = '%s%s%s' % (self.get_target_path(), self.TARGET_SEPARATOR, name)
    res = path.replace('/', self.TARGET_SEPARATOR)
    return res

  def is_full_target(self, target):
    return bool(target.find('#'))

  def options(self):
    RunnerContext.options(self.ctx)
    opa_waf.options(self.ctx)

  def configure(self):
    opa_waf.configure(self.ctx)
    if not self.ctx.options.target_pin:
      self._packages.append(self.packages.Python)
    self._packages.append(self.packages.Swig)

    if self.ctx.options.arch == Target.X86_64:
      for x in self._packages:
        opa_waf.register(x, self.ctx)
    else:
      opa_waf.register(self.packages.Protoc, self.ctx)
      opa_waf.register(self.packages.Protobuf, self.ctx)
      opa_waf.register(self.packages.Gflags, self.ctx)

    for x in self._configure_list:
      x(self)

  def build(self):
    if len(self.stack_path) == 0:
      self.getter = TgenGetter(self.ctx)

    self.auto_builder()
    for x in self._builders:
      x(self)

  def build2(self):
    pass

  def run(self):
    pass

  def normalize_name(self, name):
    if name.startswith('@'):
      name = self.libs.get(name)
    elif not self.is_full_target(name):
      name = self.get_target_name(name)
    return name

  def add_builders(self, builders):
    self._builders.extend(to_list(builders))

  def auto(self):
    self.auto_asm()
    self.auto_proto()
    self.auto_lib()
    self.auto_lib_python()
    self.auto_proto_python()
    self.auto_swig_h()
    self.auto_swig()

    self.auto_sample()
    self.auto_test()

  def register_packages(self, *packages):
    self._packages.extend(flatten(packages))

  def proc_libs(self, *libs):
    packages = []
    use_libs = []
    libs = flatten(libs)

    for x in to_list(libs):
      if isinstance(x, WafBasePkg):
        packages.append(x)
        use_libs.append(x.name)
      elif isinstance(x, WafPkgList):
        packages.extend(x.tb)
        use_libs.extend([a.name for a in x.tb])
      else:
        use_libs.append(x)
    self.register_packages(packages)
    return use_libs

  def register_libs(self, *libs):
    self._libs.extend(self.proc_libs(*libs))

  def add_name(self, name, subtarget):
    self.libs.set(name, self.get_target_name(subtarget))

  def add_configure(self, configures):
    configures = to_list(configures)
    self._configure_list.extend(configures)

  def auto_builder(self):
    for x in list(self.build_desc.values()):
      for y in to_list(x):
        self.do_build(y, self._libs)

  def get_libs(self, *add):
    lst = list(self._libs)
    lst.extend(flatten(add))
    return lst

  def do_build(self, desc, libs):
    if desc.use_global:
      desc.update(libs=libs)
    return desc.build()

  def create_conf(self, typ, extra_qual=''):
    build_typ_map = {}
    build_typ_map[BuilderType.ASM] = 'asm'
    build_typ_map[BuilderType.LIB] = 'lib'
    build_typ_map[BuilderType.TEST] = 'test'
    build_typ_map[BuilderType.SAMPLE] = 'sample'
    build_typ_map[BuilderType.PROTO] = 'proto_cpp'
    build_typ_map[BuilderType.PROTO_PYTHON] = 'proto_py'
    build_typ_map[BuilderType.SWIG] = 'swig'
    build_typ_map[BuilderType.SWIG_H] = 'header_swig'

    if not typ in self.build_desc:
      self.build_desc[typ] = []

    qual = build_typ_map[typ]
    if len(extra_qual) > 0:
      qual = '%s_%s' % (qual, extra_qual)
    conf = ExtraConf(self, qual, typ)
    self.build_desc[typ].append(conf)

    return conf

  def auto_lib(self, base_dir='.', include_root=False, filter_func=None, **kwargs):
    srcs, headers = opa_waf.get_files(os.path.join(self.path, base_dir), include_root)
    if filter_func:
      srcs = query(srcs).where(filter_func).to_list()
      headers = query(srcs).where(filter_func).to_list()

    ipath = ['./include-internal']
    eipath = ['./inc', './include']

    ipath = [os.path.join(base_dir, x) for x in ipath]
    eipath = [os.path.join(base_dir, x) for x in eipath]
    if include_root:
      eipath.append(base_dir)
      ipath.append(base_dir)

    num_inc_dir = query(eipath).where(lambda x: self.has_child(x)).count()
    found = False

    if len(srcs) + len(headers) == 0 and num_inc_dir == 0:
      return

    return self.create_conf(BuilderType.LIB, **kwargs).update(
        includes=ipath + eipath, exports=eipath, sources=srcs, headers=headers)

  def auto_lib_python(self, base_dir='.'):
    pass
    #srcs = opa_waf.get_python(os.path.join(self.path, base_dir))

    #num_inc_dir = query(eipath).where(lambda x: self.has_child(x)).count()
    #found = False

    #if len(srcs) + len(headers) == 0 and num_inc_dir == 0:
    #  return

    #return self.create_conf(BuilderType.LIB).update(includes=ipath + eipath,
    #                                                exports=eipath,
    #                                                sources=srcs,
    #                                                headers=headers)

  def auto_test(self):
    srcs = './test/test.cpp'
    if not self.has_child(srcs):
      return []

    return self.create_conf(self.typ.TEST).update(libs=self.packages.GTest, sources=srcs, binary=True)

  def auto_proto_base(self, proto_base, features, typ):
    base, proto_files = opa_waf.get_proto(self.path, proto_base)
    if len(proto_files) == 0:
      return []
    export_includes = proto_base
    return self.create_conf(typ).update(
        libs=self.packages.Protobuf,
        includes=export_includes,
        exports=export_includes,
        sources=proto_files,
        features=[features, opa_proto_feature],
        install_from=proto_base,
        proto_base=proto_base)

  def auto_proto_python(self, proto_base='./proto/msg'):
    return self.auto_proto_base(proto_base, 'py', self.typ.PROTO_PYTHON)

  def auto_proto(self, proto_base='./proto/msg'):
    return self.auto_proto_base(proto_base, 'cxx', self.typ.PROTO)

  def auto_sample(self):
    dirlist = ['./samples', './sample']
    allowed_extensions = ['.cpp', '.cc', '.c', '.py']
    for dircnd in dirlist:
      for x in self.list_dir(dircnd):
        for e in allowed_extensions:
          if x.endswith(e):
            break
        else:
          continue
        self.create_conf(self.typ.SAMPLE, os.path.basename(x)).update(sources=x, binary=True)

  def auto_asm(self, base_dir='.'):
    base_dir = os.path.join(self.path, base_dir)
    srcs = opa_waf.get_asm(base_dir)
    if len(srcs) == 0:
      return []

    return self.create_conf(self.typ.ASM).update(
        sources=srcs, includes=['./inc', './src'], features='cxx cxxshlib')

  def auto_swig(self):
    srcs = opa_waf.get_swig(self.path)
    print('FIND >> ', self.path, srcs)
    if len(srcs) == 0:
      return []

    def set_pattern(x):
      x.env.cxxshlib_PATTERN = '_%s.so'

    return self.create_conf(self.typ.SWIG).update(
        sources=srcs,
        libs=[self.libs.SwigCommon_N],
        includes=['./inc', './src', './swig', sp.check_output('python -c "import numpy; print(numpy.get_include())"', shell=True).decode().strip()],
        exports=['./swig'],
        features='cxx cxxshlib pyembed',
        swig_flags='-c++ -python -I/usr/include',
        post=set_pattern)

  def auto_swig_h(self):
    return self.create_conf(BuilderType.SWIG_H).update(
        includes=['./swig'],
        libs=[self.libs.SwigCommon_N],
        exports=['./swig'],
        sources=[],
        headers=[])
