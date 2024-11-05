import numpy as np
from Matrix.GaussELimination import solveSLE
from Matrix.Fraction import FloatToFrac, FracToFloat

def Solution(elements):
    n = ((4 * len(elements) + 1) ** 0.5 - 1) // 2
    n = int(n)
    A = np.zeros((n, n + 1), dtype=float)
    for i in range(n):
        for j in range(n + 1):
            A[i, j] = float(elements[i * (n + 1) + j]) if elements[i * (n + 1) + j] != '' else 0
    
    new_A = FloatToFrac(A)
    solution, str = solveSLE(new_A)
    if str != 'Unique Solution':
        return [str]
    
    solution_str = ''
    for i in range(n):
        solution_str += f'x{i + 1} = {solution[i, 0, 0]}' + (f'/{solution[i, 0, 1]}' if solution[i, 0, 1] != 1 else '') + ('; ' if i != n - 1 else '')
    
    return [solution_str, FracToFloat(solution), A]
