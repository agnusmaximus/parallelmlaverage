void hogwild(DataPoint *datapoints, int thread, int n_points, int vector_length, double *local_model, double C, double **C_sum_mult, double **C_sum_mult2) {
    pin_to_core(thread);
    for (int i = 0; i < n_points; i++) {

	//Compute gradient
	DataPoint p = datapoints[i];
	int x = get<0>(p), y = get<1>(p);
	double r = get<2>(p);

	//Get gradient multipliers
	double l2norm_sqr = 0;
	for (int j = 0; j < vector_length; j++) {
	    l2norm_sqr += (local_model[x*vector_length+j] + local_model[y*vector_length+j]) * (local_model[x*vector_length+j] + local_model[y*vector_length+j]);
	}
	double mult = 2 * r * (log(r) - l2norm_sqr - C);

	//vector_lengtheep track of sums for optimizing C
	C_sum_mult[thread][i] = r * (log(r) - l2norm_sqr);
	C_sum_mult2[thread][i] = r;

	//Apply gradient update
	for (int j = 0; j < vector_length; j++) {
	    double gradient =  -1 * (mult * 2 * (local_model[x*vector_length+j] + local_model[y*vector_length+j]));
	    local_model[x*vector_length+j] -= GAMMA * gradient;
	    local_model[y*vector_length+j] -= GAMMA * gradient;
	}
    }
}

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
