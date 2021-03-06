#ifndef __UTIL__
#define __UTIL__


#include <vector>
#include <set>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <numa.h>

#include "bipartite.h"

using namespace std;


void block_assignments_toFile(vector<int>& datapoint_blocks, int num_blocks, 
			      const char* file_name){
  FILE* f = fopen(file_name, "w");
  fprintf(f,"%d %d\n", datapoint_blocks.size(), num_blocks);

  for(int i = 0; i < datapoint_blocks.size(); i++){
    fprintf(f,"%d\n", datapoint_blocks[i]);
  }

  fclose(f);
  return;
}


double get_loss(double *model, vector<DataPoint> data){
  double loss = 0.0;
  double dot;
  for (int i = 0; i < data.size(); i++){
    dot = data[i].dot(model) - data[i].label();
    loss += dot * dot;
  }

  return loss;
}

long int get_time() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}


// Pin a a thread to a core. 
int pin_to_core(size_t core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  
  return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}


void print_model(double *model, int num_parameters){
  printf("Printing model\n");
  for(int i = 0; i < num_parameters; i++){
    printf("%f ", model[i]);
  }
  printf("\n");
  
  return;
}

double randu(){
  //  unsigned int seed = omp_get_thread_num();
  return ((double) rand()/(RAND_MAX));
}

double randu_r(unsigned int* seed) {
  return (((double) rand_r(seed)) / RAND_MAX);
}

int randi(int max){
  return ((int)(randu()*max));
}

int randi_r(int max, unsigned int *seed) {
  return ((int) (randu_r(seed) * max));
}

double * initialize_model(int num_parameters){
  double * model = (double *) malloc(sizeof(double)*num_parameters);
  //srand(0);

  for(int i = 0; i < num_parameters; i++){
    model[i] = 2*(randu() - 0.5);
  }

  return model;
}


void shuffle_array(int * array, int size) {
  for(int i = 0; i < size - 1; i++){
    int j = randi(size - i);
    int aux = array[i];
    array[i] = array[i + j];
    array[i + j] = aux;
  }

  return;
}

void shuffle_array_threadsafe(int * array, int size, unsigned int *seed) {
  for(int i = 0; i < size - 1; i++){
    int j = randi_r(size - i, seed);
    int aux = array[i];
    array[i] = array[i + j];
    array[i + j] = aux;
  }

  return;
}

void shuffle_indices(int *shuffled_indices, int num_datapoints, 
		     vector<int> &offsets, int start, int end){

  int num_offsets = offsets.size();
  int* permutation = (int *) malloc(sizeof(int)*num_offsets);

  for(int i = 0; i < num_offsets; i++){
    permutation[i] = i;
  }
  
  shuffle_array(permutation, num_offsets);
  
  for(int i = 0, idx = 0; i < num_offsets; i++){
    int last_idx = idx;
    int upper_bound = (permutation[i] < num_offsets -1) ? offsets[permutation[i] + 1] : num_datapoints; 
    for (int j = offsets[permutation[i]];  j < upper_bound; j++, idx++){
      shuffled_indices[idx] = j;
    }


    shuffle_array(shuffled_indices + last_idx, idx - last_idx);
  }

  free(permutation);
  return;
}

void shuffle_indices_threadsafe(int *shuffled_indices,
				int *shuffled_offsets, 
				int num_datapoints, 
		     		vector<int> &offsets, 
				int start, int end,
				unsigned int *seed){

  int num_offsets = offsets.size();
  int* permutation = shuffled_offsets;

  for(int i = 0; i < num_offsets; i++){
    permutation[i] = i;
  }
  
  shuffle_array(permutation, num_offsets);
  
  for(int i = 0, idx = 0; i < num_offsets; i++){
    int last_idx = idx;
    int upper_bound = (permutation[i] < num_offsets -1) ? offsets[permutation[i] + 1] : num_datapoints; 
    for (int j = offsets[permutation[i]];  j < upper_bound; j++, idx++){
      shuffled_indices[idx] = j;
    }

    shuffle_array_threadsafe(shuffled_indices + last_idx, idx - last_idx, seed);
  }

  return;
}

int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}


int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

double read_double( int argc, char **argv, const char *option, double default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atof( argv[iplace+1] );
    return default_value;
}


const char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}

vector<DataPoint> read_sparse_datapoints(const char *file_name) {
  string line;

  ifstream  data_file (file_name);
  getline(data_file, line);
  stringstream first_linestream(line);

  int num_datapoints, num_parameters, num_nonzeros;
  
  first_linestream >> num_datapoints >> num_parameters >> num_nonzeros;

  vector<DataPoint> data(num_datapoints);
  int *indices = (int *) malloc(sizeof(int) * num_nonzeros);
  double *x = (double *) malloc(sizeof(double)*num_nonzeros);  
  int nnz;
  double y;

  int max = -1;

  for(int i = 0, nnz_sofar = 0; i < num_datapoints; i++){
    int *indices_here = indices + nnz_sofar;
    double *x_here = x + nnz_sofar;

    getline(data_file, line);
    stringstream linestream(line);

    linestream  >> y;
    linestream >> nnz;
    
    if(nnz > max){
      max = nnz;
    }
    
    for(int j = 0; j < nnz; j++, nnz_sofar++) {
      linestream >> indices_here[j];
      linestream  >> x_here[j];
    }

    data[i].setTo(indices_here, nnz, x_here, num_parameters, y);
  } 
  
  printf("Maximum degree of data point is %d\n", max);

  return data;
}

vector<DataPoint> read_dense_datapoints(const char *file_name){

  string line;

  ifstream  data_file (file_name);
  getline(data_file, line);
  stringstream first_linestream(line);

  int num_datapoints, num_parameters;
  
  first_linestream >> num_datapoints >> num_parameters;

  vector<DataPoint> data(num_datapoints);
  double *x = (double *) malloc(sizeof(double)*num_parameters*num_datapoints);  
  double y;

  for(int i = 0; i < num_datapoints; i++){
    double *x_here = x + i * num_parameters;

    getline(data_file, line);
    stringstream linestream(line);

    linestream >> y;
    for(int j = 0; j < num_parameters; j++)
      linestream >> x_here[j];

    data[i].setTo(x_here, num_parameters, y);
  } 

  return data;
}

vector<DataPoint> read_datapoints(const char *file_name, int is_sparse) {
  if (is_sparse) return read_sparse_datapoints(file_name);
  
  return read_dense_datapoints(file_name);
}


BipartiteGraph parse_bipartiteGraph(vector<DataPoint>& data){
  int num_datapoints = data.size();
  int num_parameters = data[0].dimension();

  adj_list_t* left = new adj_list_t(num_datapoints);
  adj_list_t* right = new adj_list_t(num_parameters);

  for(int i = 0; i < num_datapoints; i++){

    int * p_first_idx = data[i].p_first_idx();
    
    left->at(i) = vector<int>(data[i].numnz());

    for(int j = 0; j < data[i].numnz(); j++){
      left->at(i)[j] = p_first_idx[j];
      
      right->at(p_first_idx[j]).push_back(i);
    }
  }

  return BipartiteGraph(*left, *right);
}



vector<DataPoint> block_order_data(vector<DataPoint> &data, vector<int> &datapoints_blocks, vector<int> &offsets){

  int num_datapoints = (int) data.size();
  int num_blocks = (int) offsets.size();
  vector<DataPoint> blocked_data(num_datapoints);
  vector<int> counters(num_blocks, 0);

  for(int i = 0; i < num_datapoints; i++){
    int current_block = datapoints_blocks[i];
    int new_position = counters[current_block] + offsets[current_block];
    blocked_data[new_position] = data[i];
    counters[current_block] ++;
  }

  return blocked_data;
}




vector<int> num_params_per_block(vector<DataPoint> &data, vector<int> &datapoints_blocks, int num_blocks){
  
  vector< set<int> > parameters_of_block(num_blocks);
  int num_datapoints = (int) data.size();
  
  for(int i = 0; i < num_datapoints; i++){
    int current_block = datapoints_blocks[i];
    int num_nz = data[i].numnz();
    int* current_indices = data[i].p_first_idx();
    for(int j = 0; j < num_nz; j++)
      parameters_of_block[current_block].insert(current_indices[j]);
  }

  vector<int> counts(num_blocks);
  for(int i = 0; i < num_blocks; i++){
    counts[i] = parameters_of_block[i].size();
  }
  
  return counts;
}



#endif

