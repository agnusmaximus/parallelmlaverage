#include <iostream>
#include <cstdlib>
#include <string.h>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sstream>
#include <set>
#include <sys/time.h>
#include <thread>
#include <omp.h>
#include <numa.h>
#include "util.h"

#define WORD_EMBEDDINGS_FILE "input_graph"
#define N_NODES 3510
#define N_DATAPOINTS 81147

#ifndef NTHREAD
#define NTHREAD 16
#endif

#ifndef N_EPOCHS
#define N_EPOCHS 100
#endif

#ifndef K
#define K 10
#endif
#define K_TO_CACHELINE ((K / 8 + 1) * 8)

#ifndef START_GAMMA
#define START_GAMMA 1e-7
#endif

double volatile C = 0;
double GAMMA = START_GAMMA;
double GAMMA_REDUCTION = 1;

double *C_sum_mult[NTHREAD];
double *C_sum_mult2[NTHREAD];
double **model;

using namespace std;

double compute_loss(vector<DataPoint> points) {
    double loss = 0;
#pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < points.size(); i++) {
	int u = get<0>(points[i]), v = get<1>(points[i]);
	double w = get<2>(points[i]);
	double sub_loss = 0;
	for (int j = 0; j < K; j++) {
	    sub_loss += (model[u][j]+model[v][j]) *  (model[u][j]+model[v][j]);
	}
	loss += w * (log(w) - sub_loss - C) * (log(w) - sub_loss - C);
    }
    return loss / points.size();
}

void hogwild(DataPoint *datapoints, int thread, int n_points) {
    pin_to_core(thread);
    for (int i = 0; i < n_points; i++) {

	//Compute gradient
	DataPoint p = datapoints[i];
	int x = get<0>(p), y = get<1>(p);
	double r = get<2>(p);

	//Get gradient multipliers
	double l2norm_sqr = 0;
	for (int j = 0; j < K; j++) {
	    l2norm_sqr += (model[x][j] + model[y][j]) * (model[x][j] + model[y][j]);
	}
	double mult = 2 * r * (log(r) - l2norm_sqr - C);

	//Keep track of sums for optimizing C
	C_sum_mult[thread][i] = r * (log(r) - l2norm_sqr);
	C_sum_mult2[thread][i] = r;

	//Apply gradient update
	for (int j = 0; j < K; j++) {
	    double gradient =  -1 * (mult * 2 * (model[x][j] + model[y][j]));
	    model[x][j] -= GAMMA * gradient;
	    model[y][j] -= GAMMA * gradient;
	}
    }
}

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

void hog_word_embeddings() {

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

	cout << compute_loss(points) << endl;

	//Hogwild
#pragma omp parallel for
	for (int j = 0; j < NTHREAD; j++) {
	    hogwild(datapoints_per_thread[j], j, n_datapoints_for_thread(points, j, NTHREAD));
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

int main(void) {
    omp_set_num_threads(NTHREAD);
    hog_word_embeddings();
}
