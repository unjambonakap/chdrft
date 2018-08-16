def ror(v, n, sz):
    n %= sz
    v1 = (v >> n)
    v2 = (v << (sz - n)) % (2 ** sz)
    return v2 | v1


def rol(v, n, sz):
    n %= sz
    v1 = (v << n) % (2 ** sz)
    v2 = v >> (sz - n)
    return v2 | v1
