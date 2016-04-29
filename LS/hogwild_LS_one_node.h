#ifndef __HOGWILD_LS_ONE_NODE__
#define __HOGWILD_LS_ONE_NODE__

#include <vector>


#include "utils.h"
#include "datapoint.h"
#include "gradient_updates.h"


long int hogwild_LS_one_node(vector<DataPoint> &data, vector<int> offsets, int num_parameters, int num_threads, int num_epochs, double step_size) {
  
  double *model = initialize_model(num_parameters); 

  int num_datapoints = data.size();
 
  
  
  //Hogwild
  #pragma omp parallel
  {

    // pin current thread to core indexed by thread id
    pin_to_core(omp_get_thread_num());

    int * shuffled_indices = (int*) malloc(sizeof(int)*num_datapoints);
    
    for (int i = 0; i < num_epochs; i++) {
      
      #pragma omp barrier
      #pragma omp master
      {
	printf("Loss at epoch %d = %f\n", i, get_loss(model, data));
      }
      
      shuffle_indices(shuffled_indices, offsets, 0, offsets.size());
      
      #pragma omp for 
      for (int t = 0; t < num_threads; t++) {
	for(int j = 0; j < num_datapoints; j++)
	  update_step(model, data[shuffled_indices[j]], step_size);
      }     
    }
  }
  
  return model;
}


  
#endif
