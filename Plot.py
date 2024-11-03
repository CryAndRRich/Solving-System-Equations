import matplotlib.pyplot as plt
import numpy as np
import io
from Matrix.SolvingMethod import SolvingMethod

def Plot(method, start, end):
    iterations = np.arange(1, 1001, dtype=int)
    accuracy = np.arange(1, 1001, dtype=float)
    max_accuracy = 0
    sM = SolvingMethod()

    x = np.zeros(end.shape, dtype=float)
    for i in range(1, 1001):
        if method == 'Richardson Iteration':
            x = sM.richardsonIteration(start, x, 1)
        elif method == 'Jacobi Method':
            x = sM.jacobiMethod(start, x, 1)
        elif method == 'Gauss Seidel Method':
            x = sM.gaussSeidelMethod(start, x, 1)
        else:
            x = sM.SOR(start, x, 1)
            
        accuracy[i - 1] = (sM.CosineSimilarity(x, end))
        max_accuracy = max(max_accuracy, accuracy[i - 1])
    
    plt.figure()
    plt.scatter(iterations, accuracy, s=2)
    plt.title(f'Plot of {method}')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return [img, max_accuracy]


