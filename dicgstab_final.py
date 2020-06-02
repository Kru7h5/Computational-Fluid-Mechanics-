import numpy as np
from scipy.sparse import csc_matrix as cm
import time
import scipy.sparse.linalg as spla
from scipy.sparse import dia_matrix
from scipy.sparse import random
from scipy import stats

# Method 1: Generating a sparse matrix using scipy.sparse.random:
# Generating the sparse matrix with density 0.25. This solver took 0.278 seconds however the matrix generation took 294.8 seconds.
# This following code (lines 12-19) were taken from docs.scipy.org


class CustomRandomState(object):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2


rs = CustomRandomState()
rvs = stats.poisson(25, loc=10).rvs
a = random(10000, 10000, density=0.25, random_state=rs, data_rvs=rvs)
a = a.A
# Generating a vector for the b component
b = np.random.randint(23, size=(10000, 1))

# Method 2: Creating a random matrix and forcing the code to to make non diagonal elements to be zero. This allowed the iteration to be 0.1145 seconds, however took about 78 seconds for the matrix to be generated.

#A = np.random.randint(34, size=(50000, 50000))
"""for i in range(len(a)):
    for j in range(len(a)):
        if(i != j):
            if ((i - j) > 2) or ((j - i) > 2):
                a[i, j] = 0"""

print(a)
# Further increasing efficiency by converting the matrix a to a sparse matrix, where only the non zero terms are stored in memory.
a = cm(a)
# Starting the time calculations to see the time taken just for solving the matrix and not the matrix generation.
start = time.time()
# Due to non-convergence issues when dealing with randomly generated matrices, I decided to increase the tolerance and to control the amount of time taken for these iterations, I set the maximum number of iterations = 2000.
x = spla.bicgstab(a, b, tol=0.7, maxiter=2000)
print(x)

if (x[1] == 0):
    print("Solver successful")
else:
    print("Error detected")
# Printing time taken to double check which of the solvers was the fastest.
end = time.time()
print("Time for iteration: ", end - start)
