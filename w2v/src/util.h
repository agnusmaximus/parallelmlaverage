#ifndef _UTIL_
#define _UTIL_

#include <tuple>
#include <iostream>
#include <cstdlib>
#include <string.h>
#include <map>
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
#include <omp.h>
#include <numa.h>

using namespace std;

struct DataPointLS {
  vector<int> indices;
  vector<double> values;
  double label;
};
typedef struct DataPointLS DataPointLS;

typedef tuple<int, int, double> DataPoint;

double compute_loss_LS(vector<DataPointLS> points, double *model) {
  double loss_total = 0;
  for (int i = 0; i < points.size(); i++) {
    DataPointLS p = points[i];
    vector<int> indices = p.indices;
    vector<double> values = p.values;
    int n_values = indices.size();
    double label = p.label;
    double loss = 0;
    for (int j = 0; j < n_values; j++) {
      loss += model[indices[j]] * values[j];
    }
    loss_total += loss;
  }
  return loss_total / points.size();
}

double compute_loss(vector<DataPoint> points, double *model, double C, int vector_length) {
    double loss = 0;
#pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < points.size(); i++) {
	int u = get<0>(points[i]), v = get<1>(points[i]);
	double w = get<2>(points[i]);
	double sub_loss = 0;
	for (int j = 0; j < vector_length; j++) {
	    sub_loss += (model[u*vector_length+j]+model[v*vector_length+j]) *  (model[u*vector_length+j]+model[v*vector_length+j]);
	}
	loss += w * (log(w) - sub_loss - C) * (log(w) - sub_loss - C);
    }
    return loss / points.size();
}

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

vector<DataPointLS> get_LS_data(string fname) {
  //Format: label index value index2 value2 ... indexn valuen
  vector<DataPointLS> result;
  ifstream f_A(fname.c_str());
  string line;
  while (f_A >> line) {
    stringstream strstream(line);
    DataPointLS dp;
    double label, value;
    int index;
    strstream >> label;
    dp.label = label;
    while (strstream >> index >> value) {
      dp.indices.push_back(index);
      dp.values.push_back(value);
    }
    result.push_back(dp);
  }
  return result;
}

vector<DataPoint> get_word_embeddings_data(string fname) {
    vector<DataPoint> datapoints;
    //ifstream in(fname);
    ifstream in;
    in.open(fname.c_str());
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

template<class T>
int start_datapoint_for_thread(vector<T> &points, int thread, int n_total_threads) {
  //  return 0;
    int n_points_per_thread = points.size() / n_total_threads;
    return thread * n_points_per_thread;
}

template<class T>
int end_datapoint_for_thread(vector<T> &points, int thread, int n_total_threads) {
  //  return points.size();
    int n_points_per_thread = points.size() / n_total_threads;
    int end = (thread+1) * n_points_per_thread;
    if (thread == n_total_threads) end = n_points_per_thread;
    return end;
}

template<class T>
int n_datapoints_for_thread(vector<T> &points, int thread, int n_total_threads) {
    return end_datapoint_for_thread(points, thread, n_total_threads) -
	start_datapoint_for_thread(points, thread, n_total_threads);
}

void initialize_model(double *model, int n_coords, int vector_length) {
    srand(0);
    for (int i = 0; i < n_coords; i++) {
	for (int j = 0; j < vector_length; j++) {
	  model[i*vector_length+j] = rand() / (double)RAND_MAX;
	}
    }
}

void allocate_memory_model(double **model, int n_coords, int vector_length) {
  *model = (double *)malloc(sizeof(double) * n_coords * vector_length);
}

void allocate_memory_model_on_node(double **model, int n_coords, int vector_length, int node) {
  *model = (double *)numa_alloc_onnode(sizeof(double) * n_coords * vector_length, node);
}

void allocate_memory(vector<DataPoint> &points, double **model, double **C_sum_mult, double **C_sum_mult2, int n_coords, int vector_length, int nthread) {
    allocate_memory_model(model, n_coords, vector_length);
    for (int i = 0; i < nthread; i++) {
	int n_points = n_datapoints_for_thread(points, i, nthread);
	C_sum_mult[i] = (double *)malloc(sizeof(double) * n_points);
	C_sum_mult2[i] = (double *)malloc(sizeof(double) * n_points);
	memset(C_sum_mult[i], 0, sizeof(double) * n_points);
	memset(C_sum_mult2[i], 0, sizeof(double) * n_points);
    }
}

void allocate_memory_on_node(vector<DataPoint> &points, double **model, double **C_sum_mult, double **C_sum_mult2, int n_coords, int vector_length, int nthread, int node_to_alloc_on) {
    *model = (double *)numa_alloc_onnode(sizeof(double *) * n_coords * vector_length, node_to_alloc_on);
    for (int i = 0; i < nthread; i++) {
	int n_points = n_datapoints_for_thread(points, i, nthread);
	C_sum_mult[i] = (double *)numa_alloc_onnode(sizeof(double) * n_points, node_to_alloc_on);
	C_sum_mult2[i] = (double *)numa_alloc_onnode(sizeof(double) * n_points, node_to_alloc_on);
	memset(C_sum_mult[i], 0, sizeof(double) * n_points);
	memset(C_sum_mult2[i], 0, sizeof(double) * n_points);
    }
}

void random_shuffle_c_array(void *array, size_t n, size_t elem_size) {
  if (n > 1) {
      size_t i;
      char dummy[elem_size];
      for (i = 0; i < n - 1; i++) {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          //int t = array[j];
	  memcpy(dummy, (char *)array + elem_size * j, elem_size);
          //array[j] = array[i];
	  memcpy((char *)array + elem_size * j, (char *)array + elem_size * i, elem_size);
          //array[i] = t;
	  memcpy((char *)array + elem_size * i, dummy, elem_size);
        }
    }
}
#endif
