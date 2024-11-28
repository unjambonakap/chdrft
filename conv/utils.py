import pandas as pd
import bs4
from chdrft.cmds import Cmds
from chdrft.utils.misc import json_dumps

def to_csv(tb_2d):
  return pd.DataFrame(tb_2d).to_csv(index=False, header=False)


def html_to_df(s):
  soup = bs4.BeautifulSoup(s, features='lxml')

  table = soup.find('table')
  #table = soup.find('table', attrs={'class':'lineItemsTable'})

  data = []
  headers = []
  def tx_to_list(tx):
    cols = [ele.text.strip() for ele in tx]
    return cols

  for row in table.find_all('tr'):
    headers.append(tx_to_list(row.find_all('th')))
    data.append(tx_to_list(row.find_all('td')))

  headers = list(filter(None, headers))
  data = list(filter(None, data))

  if headers:
    assert len(headers) == 1
    headers=headers[0]
  return pd.DataFrame.from_records(data, columns=headers)



def read_bson(filename):
  import bson
  return json_dumps(bson.loads(open(filename, 'rb').read()))

@Cmds.make
def mesh_reencode(infile:str, outfile:str):
  import pymesh
  binary = '.bin' in outfile
  pymesh.save_mesh(outfile, pymesh.load_mesh(infile), ascii=not binary)

