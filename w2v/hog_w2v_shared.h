#include "util.h"
#include "params.h"

long int hog_word_embeddings_shared() {

    double C = 0;
    double *C_sum_mult[NTHREAD];
    double *C_sum_mult2[NTHREAD];
    double **model;

    //Initialization / read data block
    vector<DataPoint> points = get_word_embeddings_data(WORD_EMBEDDINGS_FILE);
    random_shuffle(points.begin(), points.end());
    allocate_memory(points, &model, C_sum_mult, C_sum_mult2, N_DATAPOINTS, K, NTHREAD);
    initialize_model(model, N_DATAPOINTS, K);
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
	    cout << compute_loss(points, model, C, K) << endl;
	}

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

    return get_time() - start_time;
}
