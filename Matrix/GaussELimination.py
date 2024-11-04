import numpy as np
from .Fraction import Operations

class GaussELimination():
    def interchangeRow(self, mat, r1, r2):
        mat[[r1, r2]] = mat[[r2, r1]]
        return mat
    
    def firstNonZero(self, row):
        i = 0
        while i < row.shape[0]:
            if row[i, 0] != 0:
                return i
            i += 1

        return -1

    def rowReduction(self, mat, r):
        m, n, _ = mat.shape
        if mat[r, r, 0] == 0:
            for i in range(r + 1, m):
                if mat[i, r, 0] != 0:
                    mat = self.interchangeRow(mat, r, i)
                       
        pivot = mat[r, r, :]
        if pivot[0] == 0:
            return mat

        for i in range(r + 1, m):
            num = Operations(mat[i, r, :], pivot, 'div')
            for j in range(r, n):
                mat[i, j, :] = Operations(mat[i, j, :], Operations(mat[r, j, :], num, 'mul'), 'minus')

        return mat

    def echelonForm(self, mat):
        for i in range(mat.shape[0] - 1):
            mat = self.rowReduction(mat, i)
        return mat
    
    def Rank(self, mat):
        echelon_mat = self.echelonForm(mat)
        
        rank = 0
        first = -1
        while rank < min(mat.shape[0], mat.shape[1]) and self.firstNonZero(echelon_mat[rank, :, :]) > first:
            first = self.firstNonZero(echelon_mat[rank, :, :])
            rank += 1
        
        return echelon_mat, rank 
    
def solveSLE(mat):
    new_mat = np.copy(mat)
    n = new_mat.shape[0]
    x = np.zeros((n, 1, 2), dtype=int)
    new_mat, rank = GaussELimination().Rank(new_mat)
    if rank != n:
        return x, "Infinite Solution"

    for i in range(n):
        x[i, 0, :] = [0, 1]

    for i in range(1, n + 1):
        t = new_mat[n - i, n, :]
        for j in range(n - 1, n - i, -1):
            t = Operations(t, Operations(new_mat[n - i, j, :], x[j, 0, :], 'mul'), 'minus')
        
        x[n - i, 0, :] = Operations(t, new_mat[n - i, n - i], 'div')

    return x, 'Unique Solution'

