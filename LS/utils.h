#ifndef __UTIL__
#define __UTIL__


#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>


using namespace std;

double randu(){
  return ((double) rand()/(RAND_MAX));
}

int randi(int max){
  return ((int)(randu()*max));
}


double * initialize_model(int num_parameters){
  double * model = (double *) malloc(sizeof(double)*num_parameters);
  srand(0);

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



void shuffle_indices(int *shuffled_indices, vector<int> offsets, int start, int end){

  int num_offsets = offsets.size();
  int* permutation = (int *) malloc(sizeof(int)*num_offsets);

  for(int i = 0; i < num_offsets; i++){
    permutation[i] = i;
  }
  
  shuffle_array(permutation, num_offsets);
  
  for(int i = 0, idx = 0; i < num_offsets; i++){
    int last_idx = idx;
    int upper_bound = (permutation[i] < num_offsets -1) ? offsets[permutation[i] + 1] : end; 
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

double read_double( int argc, char **argv, const char *option, int default_value )
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



vector<DataPoint> read_datapoints(char *file_name){

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
    getline(data_file, line);
    stringstream linestream(line);

    linestream >> y;
    for(int j = 0; j < num_parameters; j++)
      linestream >> x[i * num_parameters + j];

    data[i].setTo(x + i * num_parameters, num_parameters, y);
    
  } 

  return data;
}

#endif
