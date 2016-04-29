#ifndef __DATAPOINT__
#define __DATAPOINT__



class DataPoint{
  double *x;
  int dim;
  double y;
 public:
  DataPoint(double *, int , double);
  void addMultTo(double, double *);
  double label();
  double dot(double *);
  void setTo(double*, int, double);
};

DataPoint::DataPoint(double *x, int dim, double y){
  this->x = x;
  this->dim = dim;
  this->y = y;
}

void DataPoint::setTo(double *x, int dim, double y){
  this->x = x;
  this->dim = dim;
  this->y = y;

  return;
}



double DataPoint::label(){
  return y;
}

void DataPoint::addMultTo(double scale, double *dst){
  for(int i = 0; i < dim; i++){
    dst[i] += scale * x[i];
  }
  
  return;
}

double DataPoint::dot(double *vec){
  double res = 0.0;
  for(int i = 0; i < dim; i++){
    res += x[i] * vec[i];
  }

  return res;
}

#endif
