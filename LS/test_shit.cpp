#include <iostream>
#include <vector>
#include <omp.h>
#include <numa.h>

#include "datapoint.h"
#include "hogwild_LS_one_node.h"
#include "utils.h"
#include "bipartite.h"
#include "graph_algorithms.h"


int main(int argc, char **argv){

  // Read HyperParameters
  int num_threads = read_int(argc, argv, "-threads", 1);
  int num_epochs = read_int(argc, argv, "-epochs", 10);
  int threshold = read_int(argc, argv, "-max-block", 100);
  double step_size = read_double(argc, argv, "-step", 0.01);
  int read_sparse = read_int(argc, argv, "-sparse", 1);

  // Read Data
  const char *file_name = read_string(argc, argv, "-data", "data.in");
  
  string problem_filename = string(file_name).append(".prob");
  string blocks_filename = string(file_name).append(".blocks");

  int num_datapoints = 13, num_blocks = 4;
  double indices[] = {0.1,0.2,0.3,0.4,1.1,1.2,1.3,2.1,2.2,3.1,3.2,3.3,3.4};
  int shuffled_indices[13];
  int offsets_aux[] = {0,4,7,9};
  vector<int> offsets(offsets_aux, offsets_aux + sizeof(offsets_aux)/sizeof(int)); 
  int start = 0, end = num_blocks;

  
  for(int i = 0; i < 10; i++){
    shuffle_indices(shuffled_indices, num_datapoints, offsets, start, end);
    for(int j = 0; j < num_datapoints; j++){
      printf("%f ", indices[shuffled_indices[j]]);
    }
    printf("\n");
  }
  
  return 0;
}
