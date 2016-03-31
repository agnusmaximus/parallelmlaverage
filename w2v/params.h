#ifndef _PARAMS_
#define _PARAMS_

#define WORD_EMBEDDINGS_FILE "data/input_graph"
#define N_NODES 3510
#define N_DATAPOINTS 81147

#ifndef NTHREAD
#define NTHREAD 24
#endif

#ifndef N_EPOCHS
#define N_EPOCHS 100
#endif

#ifndef AVERAGING_FREQ
#define AVERAGING_FREQ 1
#endif

#ifndef K
#define K 10
#endif
#define K_TO_CACHELINE ((K / 8 + 1) * 8)

#ifndef START_GAMMA
#define START_GAMMA 1e-7
#endif

#ifndef PRINT_LOSS
#define PRINT_LOSS 1
#endif

double GAMMA = START_GAMMA;
double GAMMA_REDUCTION = 1;

#endif
