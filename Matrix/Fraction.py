import numpy as np
import math

def Irreducible(A):
    a, b = A
    if a != 0:
        A //= math.gcd(a, b)
    if b < 0:
        A *= -1
    return A

def Operations(A, B, s):
    a, b = A
    c, d = B
    if s == 'sum':
        m, n = a * d + b * c, b * d
    if s == 'minus':
        m, n = a * d - b * c, b * d
    if s == 'mul':
        m, n = a * c, b * d
    if s == 'div':
        m, n = a * d, b * c

    return Irreducible(np.array([m, n], dtype=int))

def FloatToFrac(A):
    m, n = A.shape
    new_A = np.ones((m, n, 2), dtype=int)

    for i in range(m):
        for j in range(n):
            pow, temp = 1, A[i, j]
            while temp != int(temp):
                temp *= 10
                pow *= 10
            new_A[i, j, :] = Irreducible(np.array([temp, pow], dtype=int))
    return new_A

def FracToFloat(A):
    m, n, _ = A.shape
    new_A = np.ones((m, n), dtype=float)

    for i in range(m):
        for j in range(n):
            new_A[i, j] = A[i, j, 0] / A[i, j, 1]
    return new_A
                