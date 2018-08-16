from chdrft.utils.misc import DictWithDefault
from curses.ascii import isspace, isalpha, isprint
import re


class Scoring:

    def __init__(self):
        self.freq_en = {}
        for x in freq_en_raw.splitlines():
            a, b = x.split('=')
            self.freq_en[a] = float(b) / 100.

        self.freq_sorted = sorted(self.freq_en.values())
        self.freq_sorted.reverse()

    def compute_score_en(self, s):
        x = DictWithDefault(0)
        nspace = 0
        nalpha = 0
        nbad = 0
        n = len(s)
        for c in s:
            if isspace(c):
                nspace += 1
            elif isalpha(c):
                nalpha += 1
                x[c | 32] += 1
            else:
                nbad += 1

        tb = sorted(x.values())
        tb.reverse()
        score = 0
        for i in range(len(tb)):
            score += (self.freq_sorted[i] - tb[i] / nalpha) ** 2
        score += (nspace / n - self.freq_sorted[0]) ** 2
        score += (300 * nbad / n) ** 2
        return score

    def compute_score(self, s):
        return self.compute_score_en(s)


def xor(a, b):
    if isinstance(b, int):
        return [a[i] ^ b for i in range(len(a))]
    assert len(a) <= len(b)
    return [a[i] ^ b[i] for i in range(len(a))]


class XorpadSolver:

    def __init__(self, data, pad=None):
        self.data = [bytearray(x) for x in data]
        self.n = len(data)
        self.maxl = max([len(x) for x in self.data])
        self.pad = bytearray(self.maxl)
        self.scoring = Scoring()
        self.left_margin = 6

        if pad is None:
            self.guess_best()
        else:
            self.pad = pad

    def guess_best(self):
        for i in range(self.maxl):
            sx = bytearray()
            for x in self.data:
                if len(x) <= i:
                    continue
                sx.append(x[i])

            tb = []
            for k in range(256):
                tmp = xor(sx, k)
                tb.append((self.scoring.compute_score(tmp), k))
            tb.sort()
            self.pad[i] = tb[0][1]

    def disp_cols_helper(self):
        n = 1
        while n < self.maxl:
            sx = ' ' * self.left_margin
            for j in range(self.maxl // n):
                sx += '{val:<{width}d}'.format(width=n, val=j % 10)
            print(sx)
            n *= 10

    def display(self):
        for i in range(len(self.data)):
            u = xor(self.data[i], self.pad)
            sanitized = bytearray()
            for c in u:
                if not isprint(c):
                    sanitized.append(ord('?'))
                else:
                    sanitized.append(c)
            print('{row:{width}d}{space:{space_width}}{content}'.format(
                width=self.left_margin - 2,
                space_width=2,
                space='',
                row=i,
                content=sanitized.decode()))
            if i % 10 == 9:
                self.disp_cols_helper()

    def solve(self):
        while True:
            try:
                self.display()
                tmp = input('row col newchar? ')
                tmp.rstrip()
                m = re.match('(?P<row>[0-9]+)\W(?P<col>[0-9]+)\W(?P<nc>.+)',
                             tmp)
                row = int(m.group('row'))
                col = int(m.group('col'))
                nc = m.group('nc')

                for i in range(len(nc)):
                    nv = ord(nc[i]) ^ self.data[row][col + i]
                    self.pad[col + i] = nv
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(e)
                pass
        pass


freq_en_raw = """E=12.02
T=9.10
A=8.12
O=7.68
I=7.31
N=6.95
S=6.28
R=6.02
H=5.92
D=4.32
L=3.98
U=2.88
C=2.71
M=2.61
F=2.30
Y=2.11
W=2.09
G=2.03
P=1.82
B=1.49
V=1.11
K=0.69
X=0.17
Q=0.11
J=0.10
Z=0.07"""


class Mt19937:

    tamp_inv = [270681289, 2, 2165448768, 263304, 16, 67670322, 64, 274877641,
                8392962, 2165317636, 33571848, 67405969, 201953586, 8196,
                807814344, 1074299152, 2148598304, 2165449284, 270943305,
                67666194, 2298684000, 270926024, 4196369, 76054832, 2165318212,
                308415681, 75534610, 2299600932, 302138440, 604014609,
                1275170866, 2148540932]
    MATRIX_A = 0x9908b0df
    N = 624
    M = 397
    MATRIX_A = 0x9908b0df
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7fffffff
    MAG01 = [0, MATRIX_A]

    def __init__(self):
        self.mt = [0] * 624
        self.index = 0

    def do_tamp(x):
        x ^= x >> 11
        x ^= (x << 7) & 0x9d2c5680
        x ^= (x << 15) & 0xefc60000
        x ^= x >> 18
        return x

    def do_itamp(x):
        res = 0
        for i in range(32):
            if x >> i & 1:
                res ^= Mt19937.tamp_inv[i]
        return res

    def refresh(self, i):
        y = (self.mt[i] & self.UPPER_MASK) | (
            self.mt[(i + 1) % self.N] & self.LOWER_MASK)
        self.mt[i] = self.mt[(i + self.M) % self.N] ^ y >> 1
        self.mt[i] ^= self.MAG01[y & 1]

    def next(self):

        if self.index >= 624:
            for i in range(self.N):
                self.refresh(i)

            self.index = 0

        y = self.mt[self.index]
        self.index += 1
        return Mt19937.do_tamp(y)

    def reset(self, tb, need_itamp=False):
        assert len(tb) == self.N

        if need_itamp:
            self.mt = [Mt19937.do_itamp(x) for x in tb]
        else:
            self.mt=list(tb)

        self.index = self.N

    def getstate(self):
        return (3, tuple(self.mt+[self.index]), None)

