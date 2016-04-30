import numpy as np
import random as rd
from numpy import linalg as LA


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


def problem_tofile(A,b, file_name):
    f = open(file_name, 'w')
    f.write(str(A.shape[0]) + ' ' + str(A.shape[1]) + '\n')
    problem = np.concatenate((b,A), axis = 1)

    for i in range(0, A.shape[0]):
        problem[i,:].tofile(f, " ", "%f")
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
    
