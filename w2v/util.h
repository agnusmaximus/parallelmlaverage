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
//#include <numa.h>

using namespace std;

typedef tuple<int, int, double> DataPoint;

double compute_loss(vector<DataPoint> points, double **model, double C, int vector_length) {
    double loss = 0;
#pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < points.size(); i++) {
	int u = get<0>(points[i]), v = get<1>(points[i]);
	double w = get<2>(points[i]);
	double sub_loss = 0;
	for (int j = 0; j < vector_length; j++) {
	    sub_loss += (model[u][j]+model[v][j]) *  (model[u][j]+model[v][j]);
	}
	loss += w * (log(w) - sub_loss - C) * (log(w) - sub_loss - C);
    }
    return loss / points.size();
}

void pin_to_core(size_t core) {
    //cpu_set_t cpuset;
    //CPU_ZERO(&cpuset);
    //CPU_SET(core, &cpuset);
    //pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

long int get_time() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

vector<DataPoint> get_word_embeddings_data(string fname) {
    vector<DataPoint> datapoints;
    ifstream in(fname);
    string s;
    if (!in) {
	cout << "Error reading file: " << fname << endl;
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

int start_datapoint_for_thread(vector<DataPoint> &points, int thread, int n_total_threads) {
    int n_points_per_thread = points.size() / n_total_threads;
    return thread * n_points_per_thread;
}

int end_datapoint_for_thread(vector<DataPoint> &points, int thread, int n_total_threads) {
    int n_points_per_thread = points.size() / n_total_threads;
    int end = (thread+1) * n_points_per_thread;
    if (thread == n_total_threads) end = n_points_per_thread;
    return end;
}

int n_datapoints_for_thread(vector<DataPoint> &points, int thread, int n_total_threads) {
    return end_datapoint_for_thread(points, thread, n_total_threads) -
	start_datapoint_for_thread(points, thread, n_total_threads);
}