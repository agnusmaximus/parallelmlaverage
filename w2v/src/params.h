#ifndef _PARAMS_
#define _PARAMS_

#ifndef LS_FILE
//#define WORD_EMBEDDINGS_FILE "data/25_graph"
#define LS_FILE "data/ls_data_file"
#define N_NODES 300
#define N_DATAPOINTS 20000
#endif

#define WORD_EMBEDDINGS_FILE "data/input_graph"
//#define WORD_EMBEDDINGS_FILE "data/input_graph_cache_partitioned"
//#define WORD_EMBEDDINGS_FILE "data/input_graph_cache_partitioned_rec"
//#define N_NODES 16774
//#define N_DATAPOINTS 622941

//#define WORD_EMBEDDINGS_FILE "data/full_graph"
//#define WORD_EMBEDDINGS_FILE "data/input_graph_cache_partitioned_full"
//#define N_NODES 213271
//#define N_DATAPOINTS 20207156

//#define WORD_EMBEDDINGS_FILE "data/75_graph"
//#define WORD_EMBEDDINGS_FILE "data/75_graph_cache_partitioned"
//#define N_NODES 183650
//#define N_DATAPOINTS 16373907

//#define WORD_EMBEDDINGS_FILE "data/50_graph"
//#define WORD_EMBEDDINGS_FILE "data/50_graph_cache_partitioned"
//#define N_NODES 142790
//#define N_DATAPOINTS 11982369

//#define WORD_EMBEDDINGS_FILE "data/10_graph"
//#define WORD_EMBEDDINGS_FILE "data/10_graph_cache_partitioned"
//#define N_NODES 60603
//#define N_DATAPOINTS 3357895

//#ifndef WORD_EMBEDDINGS_FILE
//#define WORD_EMBEDDINGS_FILE "data/25_graph"
//#define WORD_EMBEDDINGS_FILE "data/25_graph_cache_partitioned"
//#define N_NODES 98942
//#define N_DATAPOINTS 7005679
//#endif

#ifndef CYC_BATCH_SIZE
#define CYC_BATCH_SIZE 800
#endif

#ifndef NTHREAD
#define NTHREAD 1
#endif

#ifndef N_EPOCHS
#define N_EPOCHS 500
#endif

#ifndef AVERAGING_FREQ
#define AVERAGING_FREQ 1
#endif

#ifndef K
#define K 100
#endif
#define K_TO_CACHELINE ((K / 8 + 1) * 8)

#ifndef K_BLOCK
#define K_BLOCK 10
#endif

#ifndef START_GAMMA
#define START_GAMMA 1e-11
#endif

#ifndef PRINT_LOSS
#define PRINT_LOSS 1
#endif

double GAMMA = START_GAMMA;
double GAMMA_REDUCTION = 1;

#endif
