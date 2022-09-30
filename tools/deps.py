#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import subprocess as sp
from contextlib import ExitStack
import collections
import os

global flags, cache
flags = None
cache = None

g_cmake_file = 'CMakeLists.txt'

def deps_args(parser):
  parser.add_argument('--prefix', type=cmisc.cwdpath, default=cmisc.proc_path('~/opt'))
  parser.add_argument(
      '--build', type=cmisc.cwdpath, default=cmisc.path_from_script(__file__, './build')
  )

  parser.add_argument('--uninstall', action='store_true')
  parser.add_argument('--arch32', action='store_true')
  parser.add_argument('--pkgbuild', action='store_true')
  parser.add_argument('--verbose', action='store_true')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--outdir', type=cmisc.cwdpath_abs)


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)
  deps_args(parser)


BASE_BUILD_DIR = 'build'


class Dep(ExitStack):

  def __init__(self, ctx, name):
    super().__init__()
    self.ctx = ctx
    self.name = name

  def __enter__(self):
    super().__enter__()
    self.pkgbuild = None
    if self.ctx.pkgbuild:
      self.pkgbuild = cmisc.Attr(url=None, prepare=[], build=[], package=[])
      self.callback(self.create_pkgbuild)
    return self

  def norm(self, path):
    if path == None:
      path = self.ctx.build
    else:
      path = os.path.join(self.ctx.build, path)
    return path

  def do_cmd(self, cmd, path=None, install=0, prepare=0):
    path = self.norm(path)
    if self.ctx.verbose:
      print('exec cmd=<%s> cwd=<%s>' % (cmd, path))
    if self.ctx.pkgbuild:
      if prepare: path = os.path.join(self.pkgbuild.basedir, path)
      cmd = f'pushd {path}; {cmd}'
      if cmd.endswith('\n'): cmd += 'popd'
      else: cmd += ';popd'
      if install: self.pkgbuild.package.append(cmd)
      elif prepare: self.pkgbuild.prepare.append(cmd)
      else: self.pkgbuild.build.append(cmd)
    else:
      return sp.check_call(cmd, cwd=path, shell=True)

  def python_setup_package(self, path):
    self.do_cmd('python setup.py install', path)

  def make_package(self, path, options='', post_opt='', only_install=False, has_install=1, nproc='$(nproc)', **kwargs):
    if self.ctx.uninstall:
      self.do_cmd('%s make uninstall' % options, path)
    else:
      if not only_install:
        self.do_cmd(f'{options} make -j{nproc} {post_opt}', path)
      if has_install:
        self.do_cmd(
            f'{options} make install DESTDIR={self.ctx.pkgdir} PREFIX={self.ctx.pkgdir} {post_opt}',
            path,
            install=1
        )

  def patch(self, fname, content, path, override=0):
    self.do_cmd(f'''
cat <<'EOF' {['>>', '>'][override]} {fname}
{content}
EOF
''', prepare=1)

  def cmake_package(self, path, options='', shared_lib=0, include_files=[],
                    add_pkgconfig=[], create_install_targets=[],
                    create_shared_lib_from=[],
                    **kwargs):
    build_dir = BASE_BUILD_DIR
    build_path = os.path.join(path, build_dir)
    self.do_cmd('mkdir -p %s' % build_dir, path)

    dummy_cpp_fname='dummy.cpp'
    self.patch(dummy_cpp_fname, '', path, override=1)

    for include in include_files:
      patch = f'''
INSTALL( DIRECTORY {include.directory} DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}}/{include.target})'''
      self.patch(g_cmake_file, patch, path)
    for install_target in create_install_targets:
      patch = cmisc.template_replace_safe('''install(TARGETS @{install_target} 
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})''', _opa_delimiter='@', install_target=install_target)
      self.patch(g_cmake_file, patch, path)

    for libname in cmisc.to_list(add_pkgconfig):
      self.cmake_pkgconfig(path, libname)


    opts2 = ''
    if shared_lib:
      opts2 += ' -DBUILD_SHARED_LIBS:BOOL=ON'

    if self.ctx.arch32:
      opts2 += ' -DCMAKE_C_FLAGS=-m32 -DCMAKE_CXX_FLAGS=-m32'

    if not self.ctx.uninstall:
      typ = ['Release', 'Debug'][self.ctx.debug]
      self.do_cmd(
          f'cmake -DCMAKE_BUILD_TYPE={typ} -DCMAKE_INSTALL_PREFIX={self.ctx.prefix} {options} {opts2} ..',
          build_path
      )

    self.make_package(build_path, **kwargs)

  def cmake_pkgconfig(self, path, libname):
    pkgconfig_content = '''
prefix=@CMAKE_INSTALL_PREFIX@
exec_prefix=@CMAKE_INSTALL_PREFIX@
libdir=${exec_prefix}/@CMAKE_INSTALL_LIBDIR@
includedir=${prefix}/@CMAKE_INSTALL_INCLUDEDIR@

Name: @PROJECT_NAME@
Description: @PROJECT_DESCRIPTION@
Version: @PROJECT_VERSION@

Requires:
Libs: -L${libdir} -l%s
Cflags: -I${includedir}
'''%libname
    pkgconfig_fname = f'{libname}.pc.in'
    self.patch(pkgconfig_fname, pkgconfig_content, path, override=1)


    patch_cmakefile = cmisc.template_replace_safe('''
include(GNUInstallDirs)
configure_file(@{pkgconfig_fname} @{libname}.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/@{libname}.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
''', _opa_delimiter='@', libname=libname, pkgconfig_fname=pkgconfig_fname)
    self.patch(g_cmake_file, patch_cmakefile, path)

  def configure_package(self, path, options, configure_rel='.', **kwargs):

    opts2 = ''
    if self.ctx.arch32:
      opts2 += 'CFLAGS=-m32 CXXFLAGS=-m32'

    self.do_cmd(f'{configure_rel}/configure --prefix={self.ctx.prefix} {options} {opts2}', path)
    self.make_package(path, **kwargs)

  def pkg_package(self, name, version, libname):
    if self.ctx.uninstall:
      return

    content = """
  prefix=%s
  exec_prefix=${prefix}
  libdir=${exec_prefix}/lib
  includedir=${prefix}/include

  Name: %s
  Description: %s autogen
  Version: %s
  Cflags: -I${includedir}
  """ % (self.ctx.prefix, name, name, version)
    if libname:
      content += 'Libs: -L${libdir} -l%s\n' % libname

    pkg_file = os.path.join(self.ctx.prefix, 'lib/pkgconfig/%s.pc' % name)
    if self.ctx.pkgbuild:
      self.pkgbuild.package.append(f'''
cat <<EOF > {pkg_file}
{content}
EOF
''')
    else:
      with open(pkg_file, 'w') as f:
        f.write(content)

  def github_package(self, url, version=None, dest='', rec=0):

    if not self.ctx.uninstall:
      if self.ctx.pkgbuild:
        self.pkgbuild.url = url
        self.pkgbuild.urltype = 'git'
        self.pkgbuild.basedir = dest
      else:
        if not os.path.exists(dest):
          self.do_cmd('git clone %s %s' % (url, dest))

        self.do_cmd('git fetch', dest)

      if version is not None: self.do_cmd('git checkout %s' % version, dest)

      #if os.path.exists(os.path.join(dest, '.gitmodules')):
      #  self.do_cmd('git submodule update --init', dest)
      self.do_cmd('git submodule update --init', dest)
      self.do_cmd('git submodule update --recursive', dest)
    return dest

  def extract_archive(self, archive, path=None):
    path = self.norm(path)
    base, ext = os.path.splitext(archive)
    base = os.path.basename(base)
    print(ext)
    if ext in ('.tgz', '.tar.gz'):
      self.do_cmd('tar xf %s' % archive, path)
    else:
      assert 0
    return os.path.join(path, base)

  def wget_package(self, url, destfile):
    cmd = 'wget %s -O %s' % (url, destfile)
    if self.ctx.pkgbuild:
      self.pkgbuild.url = url
      self.pkgbuild.urltype = 'archive'
    else:
      self.do_cmd(cmd)

  def create_pkgbuild(self):

    fmt = '''
pkgbase=${opa_pkgname}
pkgname=('${opa_pkgname}')
pkgver=1
pkgrel=1
basedir=${opa_basedir}
pkgdesc='Lightweight multi-platform, multi-architecture assembler framework'
arch=('i686' 'x86_64')
license=('GPL2')
makedepends=('python' 'python2')
options=('staticlibs' '!emptydirs')
source=(${opa_source})
sha512sums=()

prepare() {
true;
  ${opa_preparecmd}
}

build() {
  ${opa_buildcmd}
}

package() {
  ${opa_packagecmd}
}

# vim: ts=2 sw=2 et:
'''
    if self.pkgbuild.urltype == 'archive':
      source = self.pkgbuild.url
    else:
      source = '${basedir}::git+%s'%self.pkgbuild.url
    res = cmisc.template_replace_safe(
        fmt,
        opa_pkgname=self.name,
        opa_basedir=self.pkgbuild.basedir,
        opa_preparecmd='\n'.join(self.pkgbuild.prepare),
        opa_buildcmd='\n'.join(self.pkgbuild.build),
        opa_packagecmd='\n'.join(self.pkgbuild.package),
        opa_source=source,
    )

    with open(self.get_res_pkgbuild_filename(), 'w') as f:
      f.write(res)

  def get_res_pkgbuild_filename(self):
    return os.path.join(self.ctx.outdir, f'{self.name}.PKGBUILD')


class DepDesc:

  def __init__(self, name, url=None, giturl=None,configure=0, cmake=0, make=0, install_map=[], 
               configure_opts='', configure_dir=None, pre_cmds=[],
               **kwargs):
    self.pre_cmds =pre_cmds
    self.configure_dir = configure_dir
    self.configure_opts = configure_opts
    self.name = name
    self.url = url
    self.cmake = cmake
    self.configure = configure
    self.giturl = giturl

    self.make = make
    self.install_map = install_map
    self.kwargs =kwargs

  def run(self, ctx):
    with Dep(ctx, self.name) as d:
      if self.giturl:
        srcdir = f'{self.name}-build'
        path = d.github_package(self.giturl, dest=srcdir)
      else:
        destfile = os.path.basename(self.url)
        d.wget_package(self.url, destfile)
        path = cmisc.splitext_full(destfile)
        d.pkgbuild.basedir = path

      for cmd in self.pre_cmds:
        d.do_cmd(cmd, path)

      if self.configure:
        configure_rel = '.'
        if self.configure_dir:
          path = os.path.join(path, self.configure_dir)
          configure_rel = '..'
        d.configure_package(path, self.configure_opts, configure_rel=configure_rel, **self.kwargs)
      for append in self.kwargs.get('appends', []):
        d.patch(append.fname, append.content, path)

      if self.cmake: d.cmake_package(path, has_install=not self.install_map, **self.kwargs)
      if self.make: d.make_package(path, has_install=not self.install_map, **self.kwargs)

      for srcfile, destfile in self.install_map:
        if destfile.endswith('/'):
          destdir = destfile
        else:
          destdir = os.path.dirname(destfile)

        d.do_cmd(f'mkdir -p $pkgdir/{ctx.prefix}/{destdir}', path, install=1)
        d.do_cmd(f'cp {srcfile} $pkgdir/{ctx.prefix}/{destfile}', path, install=1)
      return d.get_res_pkgbuild_filename()


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
