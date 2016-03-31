#include "util.h"
#include "params.h"

long int full_word_embeddings() {

    double C = 0;
    double *C_sum_mult[NTHREAD];
    double *C_sum_mult2[NTHREAD];
    double *model;
    double *gd_space = (double *)malloc(sizeof(double) * N_NODES * K);

    //Initialization / read data block
    vector<DataPoint> points = get_word_embeddings_data(WORD_EMBEDDINGS_FILE);
    random_shuffle(points.begin(), points.end());
    allocate_memory(points, &model, C_sum_mult, C_sum_mult2, N_NODES, K, NTHREAD);
    initialize_model(model, N_NODES, K);
    long int start_time = get_time();

    //Divide to threads
    for (int i = 0; i < N_EPOCHS; i++) {

	if (PRINT_LOSS) {
	  cout << get_time() - start_time << " " << compute_loss(points, model, C, K) << endl;
	}

	full_gd(points, N_NODES, K, model, gd_space, &C);

	GAMMA *= GAMMA_REDUCTION;
    }

    return get_time() - start_time;
}

