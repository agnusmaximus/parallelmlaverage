#include "util.h"
#include "params.h"


long int hog_word_embeddings_model_replication_per_core() {
  
    //Create a map from core/thread -> node
    int core_to_node[NTHREAD];
    for (int i = 0; i < NTHREAD; i++) core_to_node[i] = -1;
    int n_numa_nodes = 0;
    for (int i = 0; i < NTHREAD; i++) {
      core_to_node[i] = numa_node_of_cpu(i);
      n_numa_nodes = max(n_numa_nodes, core_to_node[i]);
    }
    n_numa_nodes++;

    NTHREADS = omp.get_thread_num();

    double *model[NTHREAD];

    //Initialization / read data block
    vector<DataPoint> points = get_word_embeddings_data(WORD_EMBEDDINGS_FILE);
    random_shuffle(points.begin(), points.end());
    for (int i = 0; i < NTHREAD; i++) {
        allocate_memory_on_node(points, &model[i], C_sum_mult, C_sum_mult2, N_NODES, K, NTHREAD, core_to_node[i]);
	initialize_model(model[i], N_NODES, K);
    }
    Long int start_time = get_time();

    //Hogwild access pattern construction
    vector<DataPoint *> datapoints_per_thread(NTHREAD);
    for (int i = 0; i < NTHREAD; i++) {
	int start = start_datapoint_for_thread(points, i, NTHREAD);
	int end = end_datapoint_for_thread(points, i, NTHREAD);
	datapoints_per_thread[i] = (DataPoint *)malloc(sizeof(DataPoint) * (end-start));
	for (int j = start; j < end; j++)
	    datapoints_per_thread[i][j-start] = points[j];
    }

    //Divide to threads
    float copy_time = 0;
    for (int i = 0; i < N_EPOCHS; i++) {

      //	if (PRINT_LOSS) {
      //    cout << get_time() - start_time << " " << compute_loss(points, model[0], C, K) << endl;
      //	}

	//Hogwild
#pragma omp parallel {
	// pin current thread to core indexed by thread id
	pin_to_core(omp_get_thread_num());

	
#pragma omp for 
	for (int t = 0; t < NTHREAD; t++) {
	    update_step(datapoints_per_thread[t], t, n_datapoints_for_thread(points, t, NTHREAD), K, model[t]);
	}

    }
    
    return get_time() - start_time;
}

  
