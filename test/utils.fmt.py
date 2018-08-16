import unittest
import random
from chdrft.utils.fmt import Format

class FmtTest(unittest.TestCase):
  def test_lshift(self):
    random.seed(10)
    for i in range(2, 20):
      x=random.getrandbits(8*i)
      sx=bytearray(x.to_bytes(i, 'little'))

      shiftv=random.randrange(0,8)
      ns=Format(sx).shiftl(shiftv).v
      nx = x << shiftv
      nx %= 2**(8*i)
      rx = int.from_bytes(ns, 'little')

      print(nx==rx, shiftv, hex(x), hex(nx), hex(rx))
      self.assertEqual(nx, rx)

  def test_rshift(self):
    random.seed(10)
    for i in range(2, 20):
      x=random.getrandbits(8*i)
      sx=bytearray(x.to_bytes(i, 'little'))

      shiftv=random.randrange(0,8)
      ns=Format(sx).shiftr(shiftv).v
      nx = x >> shiftv
      rx = int.from_bytes(ns, 'little')

      print(nx==rx, shiftv, hex(x), hex(nx), hex(rx))
      self.assertEqual(nx, rx)
if __name__ == '__main__':
  unittest.main()
