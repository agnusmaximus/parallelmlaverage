from __future__ import print_function
import sys
import random
import numpy as np

RANGE=1000

if len(sys.argv) != 5:
    print("Usage: ./generate_synthetic_data.py n_examples(n_rows) model_size(n_cols) sparsity_factor output_file")
    exit(0)

n_rows, n_cols, sparsity_factor = [float(x) for x in sys.argv[1:4]]
nnz_elements = int((n_rows * n_cols) * sparsity_factor)
out_fname = sys.argv[4]

A = np.zeros((n_rows, n_cols))
b = np.zeros((n_rows))

elements = {}

# Go through each row in A and make sure there is at least one nnz element per row
for i in range(int(n_rows)):
    rand_col = random.randint(0, n_cols-1)
    rand_value = random.uniform(-RANGE,RANGE)
    elements[i] = []
    elements[i].append((rand_col, rand_value))

for i in range(nnz_elements):
    rand_row = random.randint(0, n_rows-1)
    rand_col = random.randint(0, n_cols-1)
    rand_value = random.uniform(-RANGE,RANGE)
    elements[rand_row].append((rand_col, rand_value))

for i in range(int(n_rows)):
    b[i] = random.uniform(-RANGE,RANGE)

results = np.linalg.lstsq(A, b)
v = results[0]
resids = results[1]
#print("Residual: %f" % resids)

f_out = open(out_fname, "w")
for i, (row, datapoint) in enumerate(elements.items()):
    label = b[i]
    index_value_string = " ".join(["%d %f" % (x[0], x[1]) for x in datapoint])
    print("%d %s" % (label, index_value_string), file=f_out)

f_out.close()
