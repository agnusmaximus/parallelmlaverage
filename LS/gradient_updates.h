void update_step(double *model, DataPoint point, double step_size){

  double loss = point.dot(model) - point.label();

  point.addMultTo(-loss*step_size, model);
  
  return;
}
