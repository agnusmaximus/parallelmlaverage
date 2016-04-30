#include <iostream>
#include <vector>
#include <omp.h>


#include "datapoint.h"
#include "hogwild_LS_one_node.h"
#include "utils.h"

int main(int argc, char **argv){

  // Read HyperParameters
  int num_threads = read_int(argc, argv, "-threads", 1);
  int num_epochs = read_int(argc, argv, "-epochs", 10);
  double step_size = read_double(argc, argv, "-step", 0.01);



  // Read Data
  char *file_name = read_string(argc, argv, "-data", "data.in");

  vector<DataPoint> data = read_datapoints(file_name);

  vector<int> offsets(2);
  offsets[0] = 0;
  offsets[1] = 2;

  // Cache friendly shuffle


  /////////////////


  long long int start_time = get_time();
  // HogWild
  hogwild_LS_one_node(data, offsets, num_threads, num_epochs, step_size);

  long long int hogwild_time = get_time() - start_time;


  return 0;
}
