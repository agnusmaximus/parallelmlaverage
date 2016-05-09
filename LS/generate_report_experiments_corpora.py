N = 3000000

D_VALS = [10000]
S_VALS = [0.002]
P_CROSS_VALS = [0.0, 0.1, 0.25, 0.5, 1.0]

for d in D_VALS:
    for s in S_VALS:
        for p_cross in P_CROSS_VALS:
            pg.gen_huge_regular(N, d, s, "blockmodel", shuffle_data = True, p_cross = p_cross, outdir = "corpora-report")
