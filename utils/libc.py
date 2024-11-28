import requests
import re
import subprocess as sp
import tempfile
import os
import shutil
import json
from urllib.parse import urljoin
import traceback as tb


class LibcDatabase:
    DB_FILE = 'db.json'

    def __init__(self, libc_dir, executor=None):
        self.libc_dir = libc_dir
        self.executor = executor
        self.db_file = os.path.join(self.libc_dir, self.DB_FILE)

    def __enter__(self):
        try:
            os.mkdir(self.libc_dir)
        except OSError:
            pass

        if not os.path.exists(self.db_file):
            with open(self.db_file, 'w') as f:
                json.dump({}, f)

        with open(self.db_file, 'r') as f:
            self.db = json.load(f)
        return self

    def __exit__(self, typ, value, tb):
        shutil.copy2(self.db_file, self.db_file+'.bak')

        with open(self.db_file, 'w') as f:
            json.dump(self.db, f, sort_keys=True)

    def get_libc_from_deb(self, deb_file, dest_file):
        old_dir = os.getcwd()
        temp_dir = os.path.split(deb_file)[0]

        try:
            sp.check_call(['ar', 'x', deb_file], cwd=temp_dir)
            if os.path.isfile(os.path.join(temp_dir, 'data.tar.gz')):
                sp.check_call(['tar', 'xf', 'data.tar.gz'], cwd=temp_dir)
            elif os.path.isfile(os.path.join(temp_dir, 'data.tar.xz')):
                sp.check_call(['tar', 'xJf', 'data.tar.xz'], cwd=temp_dir)
            else:
                assert False, 'unsupported file: {0}'.format(
                    str(os.listdir(temp_dir)))

            ifile = None
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    if f == 'libc.so.6':
                        link = os.path.join(root, f)
                        ifile = os.path.join(root, os.readlink(link))
                        break
                if ifile:
                    break
        finally:
            shutil.move(ifile, os.path.join(self.libc_dir, dest_file))

    def dl_url(self, url, dest_file):
        with open(dest_file, 'wb') as f:
            r = requests.get(url, stream=True)
            assert r.ok

            for b in r.iter_content(4096):
                if not b:
                    break
                f.write(b)

    def get_libc(self, url):

        if url in self.db:
            print('skipping ', url)
            return
        print('on ', url)

        temp_dir = tempfile.mkdtemp(prefix='libc_')
        dest_file = tempfile.mkstemp(
            dir=self.libc_dir,
            prefix='libc_',
            suffix='.so')[1]

        old_path = os.getcwd()

        try:
            deb_file = 'package.deb'
            deb_file = os.path.join(temp_dir, deb_file)
            self.dl_url(url, deb_file)
            self.get_libc_from_deb(deb_file, dest_file)

            print('ok for url ', url)
            self.db[url] = {
                'file': dest_file,
                }

        except:
            print('FOR url={0}, tempdir={1}'.format(url, temp_dir))
            tb.print_exc()
        finally:
            shutil.rmtree(temp_dir)

    def add_from_url(self, url, pattern):
        r = requests.get(url)
        text = r.text

        for m in re.finditer(pattern, text):
            self.executor.submit(self.get_libc, urljoin(url, m.group(1)))

    def get_i386_from_deb_snapshot(self, url, version):
        content = requests.get(url).text
        pattern = 'href="(.*libc6_{0}_i386\.deb)"'.format(re.escape(version))
        for x in re.finditer(pattern, content):
            deb_file = x.group(1)
            deb_url = urljoin(url, deb_file)
            self.get_libc(deb_url)
            break

    def get_i386_from_deb_snapshot_build_amd64(self, url, version):
        content = requests.get(url).text
        pattern = 'href="(.*libc6-i386_{0}_amd64\.deb)"'.format(re.escape(version))
        for x in re.finditer(pattern, content):
            deb_file = x.group(1)
            deb_url = urljoin(url, deb_file)
            self.get_libc(deb_url)
            break

    def get_from_debian_snapshot(self, low_ver, high_ver):
        url = 'http://snapshot.debian.org/package/glibc/'
        content = requests.get(url).text

        pattern = 'href="([0-9.-]+/)"'
        for x in re.finditer(pattern, content):
            orig_version = x.group(1)
            version = tuple(
                int(i) for i in re.split('[.-]', orig_version.rstrip('/')))

            if not(low_ver <= version and version < high_ver):
                continue

            self.executor.submit(
                self.get_i386_from_deb_snapshot_build_amd64,
                urljoin(
                    url,
                    orig_version),
                orig_version.rstrip('/'))

    def get_i386_from_ubuntu(self, url):
        content = requests.get(url).text

        end_pattern = '>i386</a>'

        pattern = 'href="([^"]*)"{0}'.format(re.escape(end_pattern))
        tb = re.findall(pattern, content)
        if len(tb) != 1:
            print('NOT normal page for ', url)

        content = requests.get(urljoin(url, tb[0])).text

        pattern = 'href="(.*libc6_.*_i386\.deb)"'
        tb = re.findall(pattern, content)

        if len(tb) != 1:
            print(url)
            print('NOT normal (2) page for ', url)
            input('continue')
            return

        self.get_libc(tb[0])

    def get_x64_from_ubuntu_x64_build(self, url):
        content = requests.get(url).text

        end_pattern = '>amd64</a>'

        pattern = 'href="([^"]*)"{0}'.format(re.escape(end_pattern))
        tb = re.findall(pattern, content)
        if len(tb) != 1:
            print('NOT normal page for ', url)


        url=urljoin(url, tb[0])
        content = requests.get(url).text

        pattern = 'href="(.*libc6_.*_amd64\.deb)"'
        tb = re.findall(pattern, content)

        if len(tb) != 1:
            print('NOT normal (2) page for ', url)
            return

        self.get_libc(tb[0])


    def get_i386_from_ubuntu_x64_build(self, url):
        content = requests.get(url).text

        end_pattern = '>amd64</a>'

        pattern = 'href="([^"]*)"{0}'.format(re.escape(end_pattern))
        tb = re.findall(pattern, content)
        if len(tb) != 1:
            print('NOT normal page for ', url)

        content = requests.get(urljoin(url, tb[0])).text

        pattern = 'href="(.*libc6-i386_.*_amd64\.deb)"'
        tb = re.findall(pattern, content)

        if len(tb) != 1:
            print(url)
            print('NOT normal (2) page for ', url)
            return
        print('ok >> ', url, tb[0])

        self.get_libc(tb[0])

    def ubuntu_get_from_package_list(self, url, package, content):
        url_pattern = '/ubuntu/+source/{0}/2.'.format(package)
        pattern = 'href="({0}[^"]*)"'.format(re.escape(url_pattern))
        for x in re.finditer(pattern, content):
            self.executor.submit(self.get_i386_from_ubuntu_x64_build,
                                    urljoin(url, x.group(1)))
            #self.executor.submit(self.get_x64_from_ubuntu_x64_build,
            #                        urljoin(url, x.group(1)))

    def get_from_ubuntu_builds(self):

        data = []
        data.append(('precise', 'eglibc'))
        data.append(('utopic', 'glibc'))
        data.append(('trusty', 'eglibc'))
        url_format = 'https://launchpad.net/ubuntu/{0}/+source/{1}'

        for d in data:
            url = url_format.format(*d)
            content = requests.get(url).text
            self.ubuntu_get_from_package_list(url, d[1], content)


    def get_from_ubuntu_publishing_history(self, package):
        url='https://launchpad.net/ubuntu/+source/{0}/+publishinghistory'
        url=url.format(package)
        content=requests.get(url).text

        pattern='href="({0}[^"]*)" rel=next"'.format(url)
        m=re.search(pattern, content)
        if m:
            url2=m.group(1)
            self.executor.submit(self.get_from_ubuntu_publishing_history, url2)

        self.ubuntu_get_from_package_list(url, package, content)


    def get_with_rev(self, low_ver, high_ver):
        res=[]
        reg=re.compile('_([0-9.-]+)')
        for k,v in self.db.items():
            try:
                m=reg.search(k).group(1)
                ver=re.split('[.-]', m)
                ver=tuple([int(x) for x in ver])
                if low_ver<=ver and ver<high_ver:
                    res.append(v['file'])
            except:
                pass

        return res

    def add_libc(self, libc_path):
        libc_path=os.path.join(os.getcwd(), libc_path)
        dest_file = tempfile.mkstemp(
            dir=self.libc_dir,
            prefix='libc_',
            suffix='.so')[1]

        shutil.copy2(libc_path, dest_file)
        self.db['file://'+dest_file]={'file':dest_file, 'desc':'manually added', 'orig_file':libc_path}




