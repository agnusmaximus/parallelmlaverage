#ifndef __HOGWILD_LS_ONE_NODE__
#define __HOGWILD_LS_ONE_NODE__

#include <vector>
#include <omp.h>

#include "utils.h"
#include "datapoint.h"
#include "gradient_updates.h"


double* hogwild_LS_one_node(vector<DataPoint> &data, vector<int> &offsets, int num_threads, int num_epochs, double step_size) {

  int num_datapoints = data.size();
  int num_parameters = data[0].dimension();
   
  double *model = initialize_model(num_parameters); 
  long long int shuffle_start_time, shuffle_end_time;

  long long int start_time = get_time();
  long long int loss_time;  

  double loss = 0.0;

  //Hogwild
#pragma omp parallel num_threads(num_threads)
  {
    // pin current thread to core indexed by thread id
    int pin_retval = pin_to_core(omp_get_thread_num());
    if (pin_retval != 0) {
        printf("ERROR: pin_to_core failed with error %d.\n", pin_retval);
        fflush(stdout);
    }

    unsigned int thread_rseed = ((unsigned int) get_time()) + omp_get_thread_num();    

    int * shuffled_indices = (int*) malloc(sizeof(int)*num_datapoints);
    int * shuffled_offsets = (int*) malloc(sizeof(int)*offsets.size());
    for(int i = 0; i < num_datapoints; i++){
      shuffled_indices[i] = i;
    }
    
    #pragma omp master
    {
    shuffle_start_time = get_time();
    }
    shuffle_indices_threadsafe(shuffled_indices, shuffled_offsets, 
				num_datapoints, offsets, 
				0, offsets.size(),
				&thread_rseed);
    
    #pragma omp master
    {
    shuffle_end_time = get_time() - shuffle_start_time;
    }
    
    for (int i = 0; i < num_epochs; i++) { 
    
      /*
      #pragma omp barrier
      #pragma omp master
      {
	loss_time -= get_time();
	loss = 0.0;
      }

      #pragma omp barrier
      #pragma omp for reduction(+:loss)
      for (int i = 0; i < data.size(); i++){
          double dot = data[i].dot(model) - data[i].label();
          loss += dot * dot;
      }

      #pragma omp barrier
      #pragma omp master
      {	
	printf("LOSS AT EPOCH: %d %f\n", i, loss);
	fflush(stdout);
	loss_time += get_time();
      }*/
 
      for(int j = 0; j < num_datapoints; j++) {
	  update_step(model, data[shuffled_indices[j]], step_size);
      }
    }

    free(shuffled_indices);
  }


  long long int hogwild_time = get_time() - start_time;
  //hogwild_time -= loss_time;

  printf("HOGWILD TIMING: %d threads: %d datapoints, %d dimensions, %d epochs: %f seconds: %f shuffle time\n",
	 num_threads, data.size(), data[0].dimension(), num_epochs, hogwild_time / 1000.0, shuffle_end_time / 1000.0);
  fflush(stdout);

  
  return model;
}
  
#endif
