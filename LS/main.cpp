#include <iostream>
#include <vector>
#include <omp.h>


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
  char *file_name = read_string(argc, argv, "-data", "data.in");

  long long int read_start_time = get_time();
  vector<DataPoint> data = read_datapoints(file_name, read_sparse);
  long long int read_end_time = get_time() - read_start_time;

  vector<int> offsets(2);
  offsets[0] = 0;
  offsets[1] = 2;

  // Cache friendly shuffle
  long long int parse_start_time = get_time();
  BipartiteGraph graph = parse_bipartiteGraph(data);
  long long int parse_end_time = get_time() - parse_start_time;

  printf("Graph parsed in %f seconds\n", parse_end_time / 1000.0);
  fflush(stdout);

  long long int construct_blocker_start_time = get_time();
  GraphBlocker blocker(data.size());
  long long int construct_blocker_end_time = get_time() - construct_blocker_start_time;

  printf("Blocker constructed in %f seconds\n", construct_blocker_end_time / 1000.0);
  fflush(stdout);

  long long int execute_blocker_start_time = get_time();
  blocker.execute(graph, GREEDY, threshold);
  long long int execute_blocker_end_time = get_time() - execute_blocker_start_time;

  printf("BFS executed in %f seconds\n", execute_blocker_end_time /1000.0);
  fflush(stdout);
  
  int num_blocks = (int)blocker.offsets.size();

  /* 
     printf("Number of blocks = %d\n", num_blocks);
     
     for(int i = 0; i < num_blocks; i++){
     printf("%d %d\n", i, blocker.offsets[i]);
     }*/
  
  /*  printf("Block assignment\n");
      
      for(int i = 0; i < data.size(); i++){
      printf("%d %d\n", i, blocker.datapoints_blocks[i]);
      }*/
  
  long long int block_order_start_time = get_time();
  vector<DataPoint> blocked_data = block_order_data(data, blocker.datapoints_blocks, blocker.offsets);
  long long int block_order_end_time = get_time() - block_order_start_time;

  printf("Block ordering executed in %f seconds\n", block_order_end_time /1000.0);
  fflush(stdout);

  vector<int> num_parameters_per_block = num_params_per_block(data, blocker.datapoints_blocks, num_blocks);

  printf("Num params per block\n");
  for(int i = 0; i < num_blocks; i ++){
    printf("%d %d\n", i, num_parameters_per_block[i]);
  }

 ////////////////
  
  // HogWild
  hogwild_LS_one_node(data, offsets, num_threads, num_epochs, step_size);

  hogwild_LS_one_node(blocked_data, blocker.offsets, num_threads, num_epochs, step_size);

  return 0;
}
