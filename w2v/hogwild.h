void hogwild(DataPoint *datapoints, int thread, int n_points, double **local_model, double C, double **C_sum_mult, double **C_sum_mult2) {
    pin_to_core(thread);
    for (int i = 0; i < n_points; i++) {

	//Compute gradient
	DataPoint p = datapoints[i];
	int x = get<0>(p), y = get<1>(p);
	double r = get<2>(p);

	//Get gradient multipliers
	double l2norm_sqr = 0;
	for (int j = 0; j < K; j++) {
	    l2norm_sqr += (local_model[x][j] + local_model[y][j]) * (local_model[x][j] + local_model[y][j]);
	}
	double mult = 2 * r * (log(r) - l2norm_sqr - C);

	//Keep track of sums for optimizing C
	C_sum_mult[thread][i] = r * (log(r) - l2norm_sqr);
	C_sum_mult2[thread][i] = r;

	//Apply gradient update
	for (int j = 0; j < K; j++) {
	    double gradient =  -1 * (mult * 2 * (local_model[x][j] + local_model[y][j]));
	    local_model[x][j] -= GAMMA * gradient;
	    local_model[y][j] -= GAMMA * gradient;
	}
    }
}
