import numpy as np
from Matrix.Fraction import Operations

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

        augmented_mat = False
        rank = 0
        first = -1
        while rank < mat.shape[0] and self.firstNonZero(echelon_mat[rank, :, :]) > first:
            first = self.firstNonZero(echelon_mat[rank, :, :])
            rank += 1
            if first == mat.shape[1] - 1:
                augmented_mat = True
                rank -= 1
                break
        
        return echelon_mat, rank, augmented_mat
    
def solveSLE(mat):
    new_mat = np.copy(mat)
    n = new_mat.shape[0]
    x = np.ones((n, 1, 2), dtype=int)
    new_mat, rank, augmented_mat = GaussELimination().Rank(new_mat)
    if rank != n:
        if augmented_mat:
            return x, 'No Solution'
        else:
            return x, 'Infinite Solution'

    for i in range(n):
        x[i, 0, :] = [0, 1]

    for i in range(1, n + 1):
        t = new_mat[n - i, n, :]
        for j in range(n - 1, n - i, -1):
            t = Operations(t, Operations(new_mat[n - i, j, :], x[j, 0, :], 'mul'), 'minus')
        
        x[n - i, 0, :] = Operations(t, new_mat[n - i, n - i], 'div')

    return x, 'Unique Solution'

