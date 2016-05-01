import numpy as np
import random as rd
from numpy import linalg as LA
import sys


def generate_problem(num_datapoints, num_parameters, sparsity, randomness, structure):
    A = np.zeros((num_datapoints, num_parameters))
    b = np.zeros((num_datapoints,1))

    if structure == 'uniform':
        f = np.vectorize(uniform_sparsity)
        A = f(A, sparsity, randomness)
        f = np.vectorize(randomness)
        b = f(b)
        return (A,b)

    
    return (A,b)


def problem_tofile(A,b, file_name, format = 'dense'):
    f = open(file_name, 'w')
    if format == 'dense': 
        f.write(str(A.shape[0]) + ' ' + str(A.shape[1]) + '\n')
        problem = np.concatenate((b,A), axis = 1)
        
        for i in range(0, A.shape[0]):
            problem[i,:].tofile(f, " ", "%f")
            f.write('\n')

    elif format == 'sparse':
        nnz = np.sum(abs(A) > 1e-10)
        f.write(str(A.shape[0]) + ' ' + str(A.shape[1]) + ' ' + str(nnz) +  '\n')
        for i in range(0, A.shape[0]):
            nnz_online = np.sum(abs(A[i,:]) > 1e-10)
            f.write('%f %d' % (b[i],nnz_online))
            for j in range(0, nnz_online):
                f.write(' ' + str(j) + ' ' + str(A[i,j]))
            f.write('\n')

    f.close()
    return


def result_tofile(A,b, file_name):
    f = open(file_name, 'w')

    (x, residuals, rank, s) = LA.lstsq(A,b)
    cond_num = max(s)/min(s)

    if residuals.shape[0] == 0:
        res = 0.0
    else:
        res = residuals[0]
        
    f.write(str(cond_num) + ' ' + str(res) + '\n')
    x.tofile(f, " ", "%f")

    f.close()
    return


def uniform_sparsity(entry, sparsity, randomness):
    if rd.uniform(0,1) > sparsity:
        return randomness(entry)
    return 0.0


def randu(entry):
    return rd.uniform(-1,1)
    
def main(argv):
    n = int(argv[0])
    d = int(argv[1])
    s = float(argv[2])
    format = argv[3]

    (X, y) = generate_problem(n, d, s, randu, 'uniform')
    
    filestem = "test_instance_n=%d_d=%d_s=%f" % (n, d, s)
    problem_tofile(X, y, filestem + ".prob", format)
    result_tofile(X, y, filestem + ".res")

    return
    
if __name__ == "__main__":
    main(sys.argv[1:])


