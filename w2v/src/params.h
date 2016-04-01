#ifndef _PARAMS_
#define _PARAMS_

#define WORD_EMBEDDINGS_FILE "data/input_graph"
#define N_NODES 16775
#define N_DATAPOINTS 622941

#ifndef CYC_BATCH_SIZE
#define CYC_BATCH_SIZE 4000
#endif

#ifndef NTHREAD
#define NTHREAD 24
#endif

#ifndef N_EPOCHS
#define N_EPOCHS 200
#endif

#ifndef AVERAGING_FREQ
#define AVERAGING_FREQ 1
#endif

#ifndef K
#define K 10
#endif
#define K_TO_CACHELINE ((K / 8 + 1) * 8)

#ifndef START_GAMMA
#define START_GAMMA 1e-6
#endif

#ifndef PRINT_LOSS
#define PRINT_LOSS 1
#endif

double GAMMA = START_GAMMA;
double GAMMA_REDUCTION = 1;

#endif
