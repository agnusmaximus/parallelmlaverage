#include "util.h"
#include "params.h"
#include "cyclades.h"

long int cyc_word_embeddings_model_replication_per_node() {

    //Create a map from core/thread -> node
    int core_to_node[NTHREAD];
    int n_numa_nodes = 0;
    for (int i = 0; i < NTHREAD; i++) core_to_node[i] = -1;
    for (int i = 0; i < NTHREAD; i++) {
      core_to_node[i] = numa_node_of_cpu(i);
      n_numa_nodes = core_to_node[i];
    }
    n_numa_nodes++;

    double C = 0;
    double *C_sum_mult[NTHREAD];
    double *C_sum_mult2[NTHREAD];
    int *tree_thread[NTHREAD];
    double *model[n_numa_nodes];
    int thread_load_balance[NTHREAD];
    size_t cur_bytes_allocated[NTHREAD];
    int cur_datapoints_used[NTHREAD];
    
    //Initialization / read data block
    vector<DataPoint> points = get_word_embeddings_data(WORD_EMBEDDINGS_FILE);
    random_shuffle(points.begin(), points.end());    
    //allocate_memory_model(&model, N_NODES, K);
    for (int i = 0; i < n_numa_nodes; i++) {
      allocate_memory_model_on_node(&model[i], N_NODES, K, core_to_node[i]);
      initialize_model(model[i], N_NODES, K);
    }
    
    //Initialize variables for CC distribution
    for (int i = 0; i < NTHREAD; i++) 
      tree_thread[i] = (int *)numa_alloc_onnode(sizeof(int) * (CYC_BATCH_SIZE + N_NODES), core_to_node[i]);
    memset(thread_load_balance, 0, sizeof(int) * NTHREAD);
    memset(cur_bytes_allocated, 0, sizeof(size_t) * NTHREAD);
    memset(cur_datapoints_used, 0, sizeof(int) * NTHREAD);

    long int start_time = get_time();

    //Cyclades access pattern construction - connected components
    int n_batches = (int)ceil((points.size() / (double)CYC_BATCH_SIZE));
    map<int, vector<int> > CCs[n_batches];
#pragma omp parallel for
    for (int i = 0; i < n_batches; i++) {
      int start = i * CYC_BATCH_SIZE;
      int end = min((i+1)*CYC_BATCH_SIZE, (int)points.size());
      compute_CC_thread(CCs[i], points, start, end, omp_get_thread_num(), tree_thread[omp_get_thread_num()]);
    }

    //Cyclades access pattern construction - distribution of work
    vector<DataPoint *> datapoints_per_thread(NTHREAD);
    vector<vector<int > > access_length(NTHREAD);
    vector<vector<int > > batch_index_start(NTHREAD);
    vector<vector<int > > order(NTHREAD);
    for (int i = 0; i < NTHREAD; i++) {
      access_length[i].resize(n_batches);
      batch_index_start[i].resize(n_batches);
      order[i].resize(n_batches);
    }
    for (int i = 0; i < n_batches; i++) {
      distribute_ccs(CCs[i], datapoints_per_thread, access_length, 
		     batch_index_start, i, points, order,
		     thread_load_balance, cur_bytes_allocated, cur_datapoints_used,
		     core_to_node);
    }

    //Readjust memory for C_sum mults
    for (int i = 0; i < NTHREAD; i++) {
      C_sum_mult[i] = (double *)numa_alloc_onnode(sizeof(double) * order[i].size(), core_to_node[i]);
      C_sum_mult2[i] = (double *)numa_alloc_onnode(sizeof(double) * order[i].size(), core_to_node[i]);
    }
    
    //Divide to threads
    float copy_time = 0;
    volatile int thread_sync[NTHREAD];
    for (int i = 0; i < N_EPOCHS; i++) {

	if (PRINT_LOSS) {
	  cout << get_time() - start_time << " " << compute_loss(points, model[0], C, K) << endl;
	}

	memset((int *)thread_sync, 0, sizeof(int) * NTHREAD);

	//Cyclades
#pragma omp parallel for
	for (int j = 0; j < NTHREAD; j++) {
	  sgd_cyc(datapoints_per_thread[j], access_length[j], batch_index_start[j], order[j], 
		  K, model[core_to_node[j]], C_sum_mult, C_sum_mult2, C, j, thread_sync);
	}

	//Optimize C
	double C_A = 0, C_B = 0;
#pragma omp parallel for reduction(+:C_A,C_B)
	for (int t = 0; t < NTHREAD; t++) {
	    for (int d = 0; d < order[t].size(); d++) {
		C_A += C_sum_mult[t][d];
		C_B += C_sum_mult2[t][d];
	    }
	}
	C = C_A / C_B;

	GAMMA *= GAMMA_REDUCTION;
    }

    return get_time() - start_time;
}

