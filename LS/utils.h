#ifndef __UTIL__
#define __UTIL__


#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

using namespace std;

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
void pin_to_core(size_t core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
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

int randi(int max){
  return ((int)(randu()*max));
}


double * initialize_model(int num_parameters){
  double * model = (double *) malloc(sizeof(double)*num_parameters);
  //srand(0);

  for(int i = 0; i < num_parameters; i++){
    model[i] = 2*(randu() - 0.5);
  }

  return model;
}


void shuffle_array(int * array, int size){
  for(int i = 0; i < size - 1; i++){
    int j = randi(size - i);
    int aux = array[i];
    array[i] = array[i + j];
    array[i + j] = aux;
  }

  return;
}


void shuffle_indices(int *shuffled_indices, int num_datapoints, vector<int> offsets, int start, int end){

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


char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}

vector<DataPoint> read_sparse_datapoints(char *file_name) {
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

vector<DataPoint> read_dense_datapoints(char *file_name){

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

vector<DataPoint> read_datapoints(char *file_name, int is_sparse) {
  if (is_sparse) return read_sparse_datapoints(file_name);
  
  return read_dense_datapoints(file_name);
}

#endif
