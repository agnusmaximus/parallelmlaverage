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

#define WORD_EMBEDDINGS_FILE "input_graph"
#define N_NODES 106
#define N_DATAPOINTS 1096

#ifndef NTHREAD
#define NTHREAD 4
#endif

#ifndef N_EPOCHS
#define N_EPOCHS 100
#endif

#ifndef K
#define K 100
#endif
#define K_TO_CACHELINE ((K / 8 + 1) * 8)

#ifndef START_GAMMA
#define START_GAMMA 1e-8
#endif

using namespace std;

typedef tuple<int, int, double> DataPoint;

double volatile C = 0;
double GAMMA = START_GAMMA;
double GAMMA_REDUCTION = 1;

double *C_sum_mult[NTHREAD];
double *C_sum_mult2[NTHREAD];
double **model;

using namespace std;

void pin_to_core(size_t core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

long int get_time() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

vector<DataPoint> get_word_embeddings_data() {
    vector<DataPoint> datapoints;
    ifstream in(WORD_EMBEDDINGS_FILE);
    string s;
    if (!in) {
	cout << "Error reading file: " << WORD_EMBEDDINGS_FILE << endl;
	exit(0);
    }
    while (getline(in, s)) {
	stringstream linestream(s);
	int n1, n2;
	double occ;
	linestream >> n1 >> n2 >> occ;
	datapoints.push_back(DataPoint(n1, n2, occ));
    }
    in.close();
    return datapoints;
}

double compute_loss(vector<DataPoint> points) {
    double loss = 0;
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

int start_datapoint_for_thread(vector<DataPoint> &points, int thread) {
    int n_points_per_thread = points.size() / NTHREAD;
    return thread * n_points_per_thread;
}

int end_datapoint_for_thread(vector<DataPoint> &points, int thread) {
    int n_points_per_thread = points.size() / NTHREAD;
    int end = (thread+1) * n_points_per_thread;
    if (thread == NTHREAD) end = n_points_per_thread;
    return end;
}

int n_datapoints_for_thread(vector<DataPoint> &points, int thread) {
    return end_datapoint_for_thread(points, thread) -
	start_datapoint_for_thread(points, thread);
}

void hogwild(DataPoint *datapoints, int thread, int n_points) {

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
	int n_points = n_datapoints_for_thread(points, i);
	C_sum_mult[i] = (double *)malloc(sizeof(double) * n_points);
	C_sum_mult2[i] = (double *)malloc(sizeof(double) * n_points);
	memset(C_sum_mult[i], 0, sizeof(double) * n_points);
	memset(C_sum_mult2[i], 0, sizeof(double) * n_points);
    }
}

void hog_word_embeddings() {

    //Initialization / read data block
    vector<DataPoint> points = get_word_embeddings_data();
    random_shuffle(points.begin(), points.end());
    allocate_memory(points);
    initialize_model();
    long int start_time = get_time();

    //Hogwild access pattern construction
    vector<DataPoint *> datapoints_per_thread(NTHREAD);
    for (int i = 0; i < NTHREAD; i++) {
	int start = start_datapoint_for_thread(points, i);
	int end = end_datapoint_for_thread(points, i);
	datapoints_per_thread[i] = (DataPoint *)malloc(sizeof(DataPoint) * (end-start));
	for (int j = start; j < end; j++)
	    datapoints_per_thread[i][j-start] = points[j];
    }

    //Divide to threads
    float copy_time = 0;
    for (int i = 0; i < N_EPOCHS; i++) {

	cout << "LOSS: " << compute_loss(points) << endl;

	//Hogwild
	vector<thread> threads;
	for (int j = 0; j < NTHREAD; j++) {
	    threads.push_back(thread(hogwild, datapoints_per_thread[j], j, n_datapoints_for_thread(points, j)));
	}
	for (int j = 0; j < threads.size(); j++) {
	    threads[j].join();
	}

	//Optimize C
	double C_A = 0, C_B = 0;
	for (int t = 0; t < NTHREAD; t++) {
	for (int d = 0; d < n_datapoints_for_thread(points, t); d++) {
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
    hog_word_embeddings();
}
