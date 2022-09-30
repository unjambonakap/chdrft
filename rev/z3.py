
class Z3Helper:

  def __init__(self):
    self.id = 0

  def get_bvv(self, v, sz):
    return z3.BitVecVal(v, sz)

  def get_bv(self, n, name=None):
    if name is None:
      name = f'x{self.id}'
      self.id += 1
    return z3.BitVec(name, n)

  def simplify(self, x):
    if z3.is_expr(x): return z3.simplify(x)
    return x

  def is_const(self, x):
    return not z3.is_expr(x) or z3.is_bv_value(x)

  def lshr(self, x, v, mask=None):
    x = z3.LShR(x, v)
    if mask is not None:
      x = x & mask
    return x

  def extract1(self, x, pos, sz):
    if isinstance(x, int): return x >> pos & (2**sz - 1)
    return z3.Extract(pos + sz - 1, pos, x)

  def extract(self, x, sz, n=None):
    if n is None: n = x.size()
    assert n % sz == 0
    tb = []
    for i in range(0, n, sz):
      tb.append(self.extract1(x, i, sz))
    return tb

  def sub_force(self, target, subs):
    res = z3.simplify(z3.substitute(target, subs))
    assert self.is_const(res), f'{target} xx {subs}'
    return res

  def solve(self, cond, subs, target):
    s = z3.Solver()
    s.add(cond)

    for a, b in subs:
      s.add(a == b)

    s.check()
    model = s.model()
    return model[target]

  def enumerate(self, s, var):
    res = []
    while s.check() == z3.sat:
      m = s.model()
      res.append(m[var].as_long())
      s.push()
      s.add(var != res[-1])
    for i in range(len(res)):
      s.pop()
    return res

  def make_num(x=0, sz_=None):
    if sz_ is None: sz_ = sz
    return z3.BitVecVal(x, sz_)

@cmisc.yield_wrapper
def z3_list_vars(expr):
  import z3

  def visitor2(e, seen):
    import z3
    if e in seen:
      return
    seen[e] = True
    yield e
    if z3.is_app(e):
      for ch in e.children():
        for e in visitor2(ch, seen):
          yield e
      return
    if z3.is_quantifier(e):
      for e in visitor2(e.body(), seen):
        yield e
      return

  seen = {}
  for e in visitor2(expr, seen):
    if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
      yield e
    else:
      pass

