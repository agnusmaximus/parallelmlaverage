void update_step(double *model, int num_parameters, DataPoint point, double step_size){

  double loss = point.dot(model) - point.label();

  point.addMultTo(-2*loss*step_size, model);
  
  return;
}
