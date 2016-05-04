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

    if structure == 'regular':
        degree = int(sparsity * num_parameters)
        f = np.vectorize(randomness)

        for i in range(num_datapoints):
            if (i + 1) % 1000 == 0:
                print 'Generated %d data points' % (i+1) 
  
            indices = np.random.permutation(num_parameters)[0:degree]           
            A[i,indices] = f(A[i,indices])

        b = f(b)
        return (A,b)

    if structure == 'super-regular':
        degree = int(sparsity * num_parameters)
        f = np.vectorize(randomness)
        for i in range(num_datapoints):
            if (i + 1) % 1000 == 0:
                print 'Generated %d data points' % (i+1) 
  
            indices = (np.arange(degree) + int(i / degree)*degree) % num_parameters
            A[i,indices] = f(A[i,indices])

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
            nnz_indices = A[i,:].nonzero()[0]
            # nnz_indices = filter(lambda j : np.abs(A[i,j]) > 1e-10, range(A.shape[1]))
            nnz_online = len(nnz_indices)

            f.write('%f %d ' % (b[i],nnz_online))
            f.write(make_sparse_string(nnz_indices, A[i,:]))

            f.write('\n')
            
            if (i + 1) % 1000 == 0:
                print 'Wrote to file %d data points' % (i + 1)

    f.close()
    return


def make_sparse_string(nnz_indices, row):
    def make_pair(j):
        return '%d %f' % (j, row[j])

    f = np.vectorize(make_pair)

    return ' '.join(f(nnz_indices))

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
    if rd.uniform(0,1) < sparsity:
        return randomness(entry)
    return 0.0


def randu(entry):
    return rd.uniform(-1,1)
    

def gen_huge_regular(num_datapoints, num_parameters, sparsity, structure='superregular'):
    degree = int(sparsity * num_parameters)
    nnz = degree * num_datapoints

    print str(degree)

    filestem = "test_instance_huge-regular_n=%d_d=%d_s=%f.prob" % (num_datapoints, num_parameters, sparsity)

    f = open(filestem, 'w')

    f.write(str(num_datapoints) + ' ' + str(num_parameters) + ' ' + str(nnz) +  '\n')

    index_permutation = np.random.permutation(num_datapoints)
    dimension_permutation = np.random.permutation(num_parameters)
    for i in index_permutation:

        f.write('%f %d ' % (rd.uniform(-1,1), degree))
    
        A = np.random.uniform(-1, 1, degree)
        
	if structure == 'superregular':
	     nnz_indices = dimension_permutation[(np.arange(degree) + int(i / degree)*degree) % num_parameters]
	else:
	     nnz_indices = np.random.randint(0, num_parameters, degree)

        f.write(make_sparse_stringII(nnz_indices, A))
        f.write('\n')
    
        if (i + 1) % 1000 == 0:
            print 'Wrote to file %d data points' % (i + 1)

    f.close()
    return 

def make_sparse_stringII(nnz_indices, row):
    def make_pair(i):
        return '%d %g' % (nnz_indices[i], row[i])

    f = np.vectorize(make_pair)

    return ' '.join(f(np.arange(len(nnz_indices))))

def main(argv):
    n = int(argv[0])
    d = int(argv[1])
    s = float(argv[2])
    structure = argv[3]
    format = argv[4]


    if structure == 'huge-regular':
        return gen_huge_regular(n, d, s)


    (X, y) = generate_problem(n, d, s, randu, structure)



    
    filestem = "test_instance_superregular_n=%d_d=%d_s=%f" % (n, d, s)
    problem_tofile(X, y, filestem + ".prob", format)

#    result_tofile(X, y, filestem + ".res")

    return
    
if __name__ == "__main__":
    main(sys.argv[1:])


