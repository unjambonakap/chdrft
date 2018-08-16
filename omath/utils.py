#!/usr/bin/env python

import gmpy2


def div_mod(a, b, mod):
    da = gmpy2.gcd(a, mod)
    db = gmpy2.gcd(b, mod)
    if da % db != 0:
        return None
    res = gmpy2.invert(b // db, mod) * (a // db) % mod
    assert res * b % mod == a
    return res

def solve_quadratic_eq(a,b,c):
    #solve a*x^2+bx+c=0 (integer coeff)

    d=b*b-4*a*c
    sols=[]
    if d==0:
        sols.append(-b//2//a)
    elif d>0:
        print('FUUU', b)
        print(gmpy2.is_square(d))
        dd=gmpy2.isqrt(d)
        sols.append((-b-dd)//(2*a))
        sols.append((-b+dd)//(2*a))
    return sols



