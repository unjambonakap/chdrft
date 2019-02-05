#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def parse(ctx):

  n = -1
  cur = []

  final_df =Z.pd.DataFrame()
  letter_count = None
  for line in open('../data/ngrams', 'r').readlines():

    if line.find('-gram\t*/*') != -1:
      if n != -1:
        df = Z.pd.read_csv(Z.io.StringIO('\n'.join(cur)), sep='\t', index_col=0)
        ngram_col = line[:6]
        col_map = dict(
            col_all='*/*',
            col_minus1=f'*/-{n}:-1',
            col_1=f'*/1:{n}',
            word=f'{n}/1:{n}',
        )

        ndf = Z.pd.DataFrame()
        if letter_count is None:
          assert n == 1
          letter_count = df[col_map['col_all']].sum()
        for name, col in col_map.items():
          cx =df[col]
          tot = cx.sum()
          best = cx.nlargest(20)
          best_all = best / letter_count
          best = best /tot
          ndf[name] = best.values
          ndf['ngram_'+name] = best.index
          ndf['default_'+name] = best.values[-1] / 2
          ndf['p_global_'+name] = best_all.values
        ndf['n'] = n
        final_df = final_df.append(ndf)

      cur = []
      n = int(line[0])
    cur.append(line)

  final_df.to_csv(open('./ngrams.csv', 'w'), index=0)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
