import numpy as np

class SolvingMethod():
    def splittingMat(self, A):
        b = np.zeros((A.shape[0], 1), dtype=float)
        for i in range(A.shape[0]):
            b[i, 0] = A[i, A.shape[1] - 1]

        A = A[:, :-1]
        D = np.zeros_like(A)
        L = np.zeros_like(A)
        U = np.zeros_like(A)
        
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i < j:
                    U[i, j] = A[i, j]
                elif i > j:
                    L[i, j] = A[i, j]
                else:
                    D[i, j] = A[i, j]
        
        return A, b, D, L, U
    
    def richardsonIteration(self, A, x, k):
        A, b, _, _, _ = self.splittingMat(A)
        w = 1 / 100

        for _ in range(k):
            x_new = w * (b - np.dot(A, x)) + x

            if np.allclose(x, x_new):
                break

            x = x_new
        
        return x

    def jacobiMethod(self, A, x, k):
        _, b, D, L, U = self.splittingMat(A)
        D = np.linalg.inv(D)
        for _ in range(k):
            x_new = np.dot(D, b - np.dot(L + U, x))

            if np.allclose(x, x_new):
                break

            x = x_new
        
        return x

    def gaussSeidelMethod(self, A, x, k):
        _, b, D, L, U = self.splittingMat(A)
        S = np.linalg.inv(D + L)

        for _ in range(k):
            x_new = np.dot(S, b - np.dot(U, x))

            if np.allclose(x, x_new):
                break

            x = x_new
        
        return x

    def SOR(self, A, x, k):
        _, b, D, L, U = self.splittingMat(A)
        w = 3 / 2
        S = np.linalg.inv(D + w * L)

        for _ in range(k):
            x_new = np.dot(S, w * b - np.dot(w * U + (w - 1) * D, x))

            if np.allclose(x, x_new):
                break

            x = x_new
        
        return x
    
    def CosineSimilarity(self, x0, x1):
        a, b, c = 0, 0, 0

        for i in range(x0.shape[0]):
            a += x0[i, 0] * x1[i, 0]
            b += x0[i, 0] ** 2
            c += x1[i, 0] ** 2
        
        if b == 0 or c == 0:
            return 0
        
        accuracy = a / ((b ** 0.5) * (c ** 0.5))
        if 0 < accuracy <= 100:
            return int(accuracy * (10 ** 8)) / (10 ** 6)
        return 0


