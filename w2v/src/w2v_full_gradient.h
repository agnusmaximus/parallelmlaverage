#include "util.h"
#include "params.h"

void full_gd(vector<DataPoint> &points, int n_coords, int vector_length, double *model, double *gd, double *C) {
  memset(gd, 0, sizeof(double) * n_coords * vector_length);
  double C_A = 0, C_B = 0;
  
  for (int i = 0; i < points.size(); i++) {
    //Compute gradient
    DataPoint p = points[i];
    int x = get<0>(p), y = get<1>(p);
    double r = get<2>(p);
    
    //Get gradient multipliers
    double l2norm_sqr = 0;
    for (int j = 0; j < vector_length; j++) {
      l2norm_sqr += (model[x*vector_length+j] + model[y*vector_length+j]) * (model[x*vector_length+j] + model[y*vector_length+j]);
    }
    double mult = 2 * r * (log(r) - l2norm_sqr - *C);
    
    //vector_lengtheep track of sums for optimizing C
    C_A += r * (log(r) - l2norm_sqr);
    C_B += r;
    
    //Sum gradients
    for (int j = 0; j < vector_length; j++) {
      double gradient =  -1 * (mult * 2 * (model[x*vector_length+j] + model[y*vector_length+j]));
      gd[x*vector_length+j] += gradient;
      gd[y*vector_length+j] += gradient;
    }    
  }
  
  //Apply gradient
  for (int i = 0; i < n_coords; i++) {
    for (int j = 0; j < vector_length; j++) {
      model[i*vector_length+j] -= GAMMA * gd[i*vector_length+j];
    }
  }

  *C = C_A / C_B;
}

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

