import problem_generator as pg
import sys

N = 100000
d = 10000
sparsity_levels = [0.002]
structure = "blockmodel"

print("Creating corpus to test blockmodel [N = %d, d = %d]..." % (N, d))

for s in sparsity_levels:
    for p_cross in [0.1]:	
        print("Creating corpus: structure = %s, sparsity = %g, p_cross = %g" % (structure, s, p_cross))
        pg.gen_huge_regular(N, d, s, structure, shuffle_data = True, p_cross = p_cross, outdir = "corpora_test")
