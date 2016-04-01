void sgd(DataPoint *datapoints, int thread, int n_points, int vector_length, double *local_model, double C, double **C_sum_mult, double **C_sum_mult2) {
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

void sgd_cyc(DataPoint *datapoints, vector<int> &access_length, vector<int> &batch_index_start, vector<int> &order, 
	     int vector_length, double *local_model, double **C_sum_mult, double **C_sum_mult2, double C, int thread, 
	     volatile int *thread_sync) {
  pin_to_core(thread);
  for (int batch = 0; batch < access_length.size(); batch++) {
    
    //Synchronize threads
    thread_sync[thread] = batch;    
    int waiting_for_other_threads = 1;
    while (waiting_for_other_threads) {
      waiting_for_other_threads = 0;
      for (int i = 0; i < NTHREAD; i++) {
	if (thread_sync[i] < batch) {
	  waiting_for_other_threads = 1;
	  break;
	}
      }
    }
    
    int n_points = access_length[batch];
    for (int i = 0; i < n_points; i++) {
      
      //Compute gradient
      int index = batch_index_start[batch]+i;
      DataPoint p = datapoints[index];
      int update_order = order[index];
      int x = get<0>(p), y = get<1>(p);
      double r = get<2>(p);

      //Get gradient multipliers
      double l2norm_sqr = 0;
      for (int j = 0; j < vector_length; j++) {
	l2norm_sqr += (local_model[x*vector_length+j] + local_model[y*vector_length+j]) * (local_model[x*vector_length+j] + local_model[y*vector_length+j]);
      }
      double mult = 2 * r * (log(r) - l2norm_sqr - C);

      //vector_lengtheep track of sums for optimizing C
      C_sum_mult[thread][index] = r * (log(r) - l2norm_sqr);
      C_sum_mult2[thread][index] = r;

      //Apply gradient update
      for (int j = 0; j < vector_length; j++) {
	double gradient =  -1 * (mult * 2 * (local_model[x*vector_length+j] + local_model[y*vector_length+j]));
	local_model[x*vector_length+j] -= GAMMA * gradient;
	local_model[y*vector_length+j] -= GAMMA * gradient;
      }
    }
  }
}

void sgd_cyc_blocked(DataPoint *datapoints, vector<int> &access_length, vector<int> &batch_index_start, vector<int> &order, 
		     int vector_length, double *local_model, double **C_sum_mult, double **C_sum_mult2, double C, int thread, 
		     volatile int *thread_sync) {
  pin_to_core(thread);
  for (int batch = 0; batch < access_length.size(); batch++) {
    
    //Synchronize threads
    thread_sync[thread] = batch;    
    int waiting_for_other_threads = 1;
    while (waiting_for_other_threads) {
      waiting_for_other_threads = 0;
      for (int i = 0; i < NTHREAD; i++) {
	if (thread_sync[i] < batch) {
	  waiting_for_other_threads = 1;
	  break;
	}
      }
    }
    
    int n_points = access_length[batch];
    for (int i = 0; i < n_points; i++) {
      
      //Compute gradient
      int index = batch_index_start[batch]+i;
      DataPoint p = datapoints[index];
      int update_order = order[index];
      int x = get<0>(p), y = get<1>(p);
      double r = get<2>(p);

      for (int vector_block = 0; vector_block < vector_length; vector_block += K_BLOCK) {

	//Get gradient multipliers
	double l2norm_sqr = 0;
	for (int j = vector_block; j < vector_block+K_BLOCK; j++) {
	  if (j < vector_length)
	    l2norm_sqr += (local_model[x*vector_length+j] + local_model[y*vector_length+j]) * (local_model[x*vector_length+j] + local_model[y*vector_length+j]);
	}
	double mult = 2 * r * (log(r) - l2norm_sqr - C);
	
	//vector_lengtheep track of sums for optimizing C
	//C_sum_mult[thread][index] = r * (log(r) - l2norm_sqr);
	//C_sum_mult2[thread][index] = r;
	
	//Apply gradient update
	for (int j = vector_block; j < vector_block+K_BLOCK; j++) {
	  if (j < vector_length) {
	    double gradient =  -1 * (mult * 2 * (local_model[x*vector_length+j] + local_model[y*vector_length+j]));
	    local_model[x*vector_length+j] -= GAMMA * gradient;
	    local_model[y*vector_length+j] -= GAMMA * gradient;
	  }
	}
      }
    }
  }
}

void sgd_track_gds(DataPoint *datapoints, double *gd, int thread, int n_points, int vector_length, double *local_model, double C, double **C_sum_mult, double **C_sum_mult2) {
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
	    gd[x*vector_length+j] += gradient;
	    local_model[x*vector_length+j] -= GAMMA * gradient;
	    gd[y*vector_length+j] += gradient;
	    local_model[y*vector_length+j] -= GAMMA * gradient;
	}
    }
}
