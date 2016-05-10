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
  string blocks_greedy = string(file_name).append(".greedy");
  string blocks_bfs = string(file_name).append(".bfs");
  
  long long int read_start_time = get_time();
  vector<DataPoint> data = read_datapoints(problem_filename.c_str(), read_sparse);
  long long int read_end_time = get_time() - read_start_time;
  
  printf("Data read from file in %f seconds\n", read_end_time / 1000.0);
  fflush(stdout);
  

  GraphBlocker blocker;
  if(fopen(blocks_greedy.c_str(), "r") != NULL){

    printf("Reading blocks from file\n");
    fflush(stdout);

    long long int construct_blocker_start_time = get_time();
    blocker = GraphBlocker(blocks_greedy.c_str());
    long long int construct_blocker_end_time = get_time() - construct_blocker_start_time;
    
    printf("Blocker constructed from file in %f seconds\n", construct_blocker_end_time / 1000.0);
    fflush(stdout);    
  } else{

    printf("Computing blocks\n");
    fflush(stdout);

    long long int construct_blocker_start_time = get_time();
    blocker = GraphBlocker(data.size());
    BipartiteGraph graph = parse_bipartiteGraph(data);
    blocker.execute(graph, GREEDY, threshold);
    long long int construct_blocker_end_time = get_time() - construct_blocker_start_time;
    
    printf("Blocker executed in %f seconds\n", construct_blocker_end_time / 1000.0);
    fflush(stdout); 
    
    block_assignments_toFile(blocker.datapoints_blocks, blocker.offsets.size(), blocks_greedy.c_str());
    
    printf("Block assignments written to file\n");
  }

    //////////////////////////////////////////////////////////////////////////////
    int num_blocks = (int)blocker.offsets.size();  
    
    printf("Num Blocks = %d\n", num_blocks);
    fflush(stdout);


    long long int block_order_start_time = get_time();
    vector<DataPoint> blocked_data = block_order_data(data, blocker.datapoints_blocks, blocker.offsets);
    long long int block_order_end_time = get_time() - block_order_start_time;
    
    printf("Block ordered data in %f seconds\n", block_order_end_time /1000.0);
    fflush(stdout);
  
    // HogWild
    vector<int> trivial_offsets(1);
    trivial_offsets[0] = 0;
    
    hogwild_LS_one_node(data, trivial_offsets, num_threads, num_epochs, step_size);
    hogwild_LS_one_node(blocked_data, blocker.offsets, num_threads, num_epochs, step_size);
    
    return 0;
}
