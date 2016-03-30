#include "util.h"
#include "params.h"

static double volatile C = 0;
static double *C_sum_mult[NTHREAD];
static double *C_sum_mult2[NTHREAD];
static double **model;

void initialize_model() {
    for (int i = 0; i < N_NODES; i++) {
	for (int j = 0; j < K; j++) {
	    model[i][j] = rand() / (double)RAND_MAX;
	}
    }
}

void allocate_memory(vector<DataPoint> &points) {
    model = (double **)malloc(sizeof(double *) * N_NODES);
    for (int i = 0; i < N_NODES; i++) {
	model[i] = (double *)malloc(sizeof(double) * K_TO_CACHELINE);
    }
    for (int i = 0; i < NTHREAD; i++) {
	int n_points = n_datapoints_for_thread(points, i, NTHREAD);
	C_sum_mult[i] = (double *)malloc(sizeof(double) * n_points);
	C_sum_mult2[i] = (double *)malloc(sizeof(double) * n_points);
	memset(C_sum_mult[i], 0, sizeof(double) * n_points);
	memset(C_sum_mult2[i], 0, sizeof(double) * n_points);
    }
}

void hog_word_embeddings_shared() {

    //Initialization / read data block
    vector<DataPoint> points = get_word_embeddings_data(WORD_EMBEDDINGS_FILE);
    random_shuffle(points.begin(), points.end());
    allocate_memory(points);
    initialize_model();
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

	cout << compute_loss(points, model, C, K) << endl;

	//Hogwild
#pragma omp parallel for
	for (int j = 0; j < NTHREAD; j++) {
	    hogwild(datapoints_per_thread[j], j, n_datapoints_for_thread(points, j, NTHREAD), model, C, C_sum_mult, C_sum_mult2);
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
    }

    cout << get_time() - start_time << " ms" << endl;
}
