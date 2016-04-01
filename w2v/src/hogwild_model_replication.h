#include "util.h"
#include "params.h"

void average_n_models(int n, double **models, int n_coords, int vector_length) {
  //Optimize later
  #pragma omp parallel for
  for (int i = 0; i < n_coords; i++) {
    for (int j = 0; j < vector_length; j++) {
      double average = 0;
      for (int k = 0; k < n; k++) {
	average += models[k][i*vector_length+j];
      }
      average /= n;
      for (int k = 0; k < n; k++) {
	models[k][i*vector_length+j] = average;
      }
    }
  }
}

void average_two_models(double *model1, double *model2, int node1, int node2, int n_coords, int vector_length, int *core_to_node) {

  //Want cores to write to their own numa nodes, optimize later?
  #pragma omp parallel for
  for (int i = 0; i < n_coords; i++) {
    for (int j = 0; j < vector_length; j++) {
      double average = (model1[i*vector_length+j] + model2[i*vector_length+j])/2;
      model1[i*vector_length+j] = model2[i*vector_length+j] = average;
    }
  }
}

long int hog_word_embeddings_model_replication_per_core() {

    double C = 0;
    double *C_sum_mult[NTHREAD];
    double *C_sum_mult2[NTHREAD];
    double *model[NTHREAD];

    //Initialization / read data block
    vector<DataPoint> points = get_word_embeddings_data(WORD_EMBEDDINGS_FILE);
    random_shuffle(points.begin(), points.end());
    for (int i = 0; i < NTHREAD; i++) {
	allocate_memory(points, &model[i], C_sum_mult, C_sum_mult2, N_NODES, K, NTHREAD);
	initialize_model(model[i], N_NODES, K);
    }
    long int start_time = get_time();

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

	if (PRINT_LOSS) {
	    cout << get_time() - start_time << " " << compute_loss(points, model[0], C, K) << endl;
	}

	//Hogwild
#pragma omp parallel for
	for (int j = 0; j < NTHREAD; j++) {
	    sgd(datapoints_per_thread[j], j, n_datapoints_for_thread(points, j, NTHREAD), K, model[j], C, C_sum_mult, C_sum_mult2);
	}

	//Optimize C
	double C_A = 0, C_B = 0;
#pragma omp parallel for reduction(+:C_A,C_B)
	for (int t = 0; t < NTHREAD; t++) {
	    for (int d = 0; d < n_datapoints_for_thread(points, t, NTHREAD); d++) {
		C_A += C_sum_mult[t][d];
		C_B += C_sum_mult2[t][d];
	    }
	}
	C = C_A / C_B;

	GAMMA *= GAMMA_REDUCTION;

	//Model averaging
	if (i % AVERAGING_FREQ == 0) {
	  average_n_models(NTHREAD, model, N_NODES, K);
	}
    }

    return get_time() - start_time;
}

long int hog_word_embeddings_model_replication_per_node() {

    //Start numa
    if (numa_available() == -1) {
	cout << "NUMA NOT AVAILABLE" << endl;
	exit(0);
    }

    //Create a map from core/thread -> node
    int core_to_node[NTHREAD];
    for (int i = 0; i < NTHREAD; i++) core_to_node[i] = -1;
    int n_numa_nodes = 0;
    for (int i = 0; i < NTHREAD; i++) {
      core_to_node[i] = numa_node_of_cpu(i);
      n_numa_nodes = max(n_numa_nodes, core_to_node[i]);
    }
    n_numa_nodes++;

    double C = 0;
    double *C_sum_mult[NTHREAD];
    double *C_sum_mult2[NTHREAD];
    double *model[n_numa_nodes];

    //Initialization / read data block
    vector<DataPoint> points = get_word_embeddings_data(WORD_EMBEDDINGS_FILE);
    random_shuffle(points.begin(), points.end());
    for (int i = 0; i < n_numa_nodes; i++) {
	allocate_memory_on_node(points, &model[i], C_sum_mult, C_sum_mult2, N_NODES, K, NTHREAD, i);
	initialize_model(model[i], N_NODES, K);
    }
    long int start_time = get_time();

    //Hogwild access pattern construction
    vector<DataPoint *> datapoints_per_thread(NTHREAD);
    for (int i = 0; i < NTHREAD; i++) {
	int start = start_datapoint_for_thread(points, i, NTHREAD);
	int end = end_datapoint_for_thread(points, i, NTHREAD);
	datapoints_per_thread[i] = (DataPoint *)numa_alloc_onnode(sizeof(DataPoint) * (end-start), core_to_node[i]);
	for (int j = start; j < end; j++)
	    datapoints_per_thread[i][j-start] = points[j];
    }

    //Divide to threads
    float copy_time = 0;
    for (int i = 0; i < N_EPOCHS; i++) {

	if (PRINT_LOSS) {
	    cout << get_time() - start_time << " " << compute_loss(points, model[0], C, K) << endl;
	}

	//Hogwild
#pragma omp parallel for
	for (int j = 0; j < NTHREAD; j++) {
	    sgd(datapoints_per_thread[j], j, n_datapoints_for_thread(points, j, NTHREAD), K, model[core_to_node[j]], C, C_sum_mult, C_sum_mult2);
	}

	//Optimize C
	double C_A = 0, C_B = 0;
#pragma omp parallel for reduction(+:C_A,C_B)
	for (int t = 0; t < NTHREAD; t++) {
	    for (int d = 0; d < n_datapoints_for_thread(points, t, NTHREAD); d++) {
		C_A += C_sum_mult[t][d];
		C_B += C_sum_mult2[t][d];
	    }
	}
	C = C_A / C_B;

	//Model averaging
	if (i % AVERAGING_FREQ == 0 && n_numa_nodes >= 2) {
	  int node1 = rand() % n_numa_nodes, node2 = rand() % n_numa_nodes;
	  while (node1 == node2) node2 = rand() % n_numa_nodes;
	  average_two_models(model[node1], model[node2], node1, node2, N_NODES, K, core_to_node);
	}

	GAMMA *= GAMMA_REDUCTION;
    }

    return get_time() - start_time;
}
