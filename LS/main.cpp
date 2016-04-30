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
  int read_sparse = read_int(argc, argv, "-sparse", 1);

  // Read Data
  char *file_name = read_string(argc, argv, "-data", "data.in");

  vector<DataPoint> data = read_datapoints(file_name, read_sparse);

  vector<int> offsets(2);
  offsets[0] = 0;
  offsets[1] = 2;

  // Cache friendly shuffle


  /////////////////


  long long int start_time = get_time();
  // HogWild
  hogwild_LS_one_node(data, offsets, num_threads, num_epochs, step_size);

  long long int hogwild_time = get_time() - start_time;

  printf("HOGWILD TIMING: %d threads: %d datapoints, %d dimensions, %d epochs: %f seconds\n",
	 num_threads, data.size(), data[0].dimension(), num_epochs, hogwild_time / 1000.0);
  fflush(stdout);

  return 0;
}
