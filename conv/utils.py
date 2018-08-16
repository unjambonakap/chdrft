import pandas

def to_csv(tb_2d):
  return pandas.DataFrame(tb_2d).to_csv(index=False, header=False)
