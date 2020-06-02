import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import time

# checking the time taken to execute the program.
start = time.time()
# Defining the first matrix and storing it as a csc_matrix. The system complies for a np.matrix initialization, however gives a SparseEfficiencyWarning. The matrix can be initialized as a CSR matrix.
A = csc_matrix([[300, -100, 0, 0, 0], [-100, 200, -100, 0, 0], [0, -100, 200, -100, 0], [0, 0, -100, 200, -100], [0, 0, 0, -100, 300]], dtype=int)
# Defining the second matrix b. Again these members could be defined as a csc_matrix.
b = np.matrix([[20000], [0], [0], [0], [100000]], dtype=int)
# Double checking the two matrices to make sure that they A is a n*n matric and b is a vector
print(b.shape)
print(A.shape)
print("A=", A)
print("B=", b)
# Solving for x by solving the linear system of equations A and b.
x = bicgstab(A, b)
print(x)
# Checking if the system of equations were able to converge. The resulting matrix x, returns an integer as well, where if the
# x[1] =0 : successful exit
# x[1] =>0 : convergence to tolerance not achieved, number of iterations
# x[1] =<0 : illegal input or breakdown
if (x[1] == 0):
    print("Solver successful")
else:
    print("Error detected")
# Printing time taken to double check which of the solvers was the fastest.
end = time.time()
print("Time for iteration: ", end - start)
