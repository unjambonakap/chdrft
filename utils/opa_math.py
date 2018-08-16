
def rotlu32(v, l):
  if l<0:
    l=32+l

  v1=v<<l
  v2=v>>(32-l)
  v1&=2**32-1
  return v1|v2

def modu32(v):
  return v%(2**32)
