#!/usr/bin/env python

from bs4 import BeautifulSoup as BS
from chdrft.cmds import Cmds
from chdrft.utils.cache import *
from chdrft.utils.misc import OpaExecutor, cwdpath, failsafe, PatternMatcher
from urllib.parse import urljoin
import argparse
import os
import re
import requests
import shutil
import time

global flags
flags = None


class Scraper(Cachable):

  def __init__(self,
               base_url=None,
               rec=False,
               file_matcher=None,
               num_tasks=10,
               dry_run=False,
               only_subfiles=False,
               requests_args = {},
               output_dir=None, **kwargs):
    super().__init__(**kwargs)
    assert base_url is not None
    assert output_dir is not None
    assert file_matcher is not None
    self.base_url = base_url
    self.rec = rec
    self.file_matcher = file_matcher
    self.num_tasks = num_tasks
    self.output_dir = output_dir
    self.dry_run = dry_run
    self.requests_args = requests_args
    self.only_subfiles = only_subfiles

  def __fini__(self):
    pass

  def go(self):
    with OpaExecutor(self.num_tasks) as executor:
      self.executor = executor
      self.do(self.base_url)

  @Cachable.cached('wget')
  def retrieve_listing(self, url):
    doc = requests.get(url, **self.requests_args)
    return doc.text

  @Cachable.cached('data')
  def retrieve_data(self, data):
    url, dest = data
    dest = os.path.join(self.output_dir, dest)
    if dest.endswith('/'): dest=dest[:-1]+'_dir'
    if self.dry_run:
      print('Here would retrieve %s to %s' % (url,dest))
      return
    cur_dir = os.path.dirname(dest)
    failsafe(lambda: os.makedirs(cur_dir))

    print('Retrieve', url, dest)
    res = requests.get(url, stream=True, **self.requests_args)
    with open(dest, 'wb') as f:
      shutil.copyfileobj(res.raw, f)

  def do(self, url):
    res = self.retrieve_listing(url)
    bs = BS(res, 'html.parser')
    print('on page ', url)

    lst = []
    lst_rec = []
    relpath = './'+url[len(self.base_url):]

    for x in bs.find_all('a'):
      lnk = x['href']
      nxt = urljoin(url, lnk)
      is_subfile = nxt.startswith(url)
      if self.only_subfiles and not is_subfile: continue

      if self.rec and re.match('.*/', lnk):

        if not is_subfile:
          continue  # only consider childrens
        lst_rec.append(nxt)

      if self.file_matcher(nxt):
        lst.append([nxt, os.path.join(relpath, lnk)])
    for e in lst:
      self.executor.submit(self.retrieve_data, e)
    print('recurse on ', lst_rec)
    for e in lst_rec:
      self.executor.submit(self.do, e)

