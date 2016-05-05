#ifndef __DATAPOINT__
#define __DATAPOINT__

class DataPoint{
  int *indices;
  int nnz;
  double *x;
  int dim;
  double y;

 public:
  DataPoint();
  DataPoint(double *, int , double);
  DataPoint(int*, int, double*, int, double);
  void addMultTo(double, double *);
  int dimension();
  int numnz();
  int* p_first_idx();
  int* p_last_idx();

  double label();
  double dot(double *);
  void setTo(double*, int, double);
  void setTo(int*, int, double*, int, double);
};


DataPoint::DataPoint(){
  this->indices=NULL;
  this->nnz = 0;
  this->x = NULL;
  this->dim = 0;
  this->y = 0.0;
}

DataPoint::DataPoint(double *x, int dim, double y){
  this->indices=NULL;
  this->nnz = 0;
  this->x = x;
  this->dim = dim;
  this->y = y;
}

DataPoint::DataPoint(int *indices, int nnz, double *x, int dim, double y) {
  this->indices=indices;
  this->nnz = nnz;
  this->x=x;
  this->dim=dim;
  this->y=y;
}

void DataPoint::setTo(double *x, int dim, double y){
  this->x = x;
  this->dim = dim;
  this->y = y;

  return;
}

void DataPoint::setTo(int *indices, int nnz, double *x, int dim, double y) {
  this->indices = indices;
  this->nnz = nnz;
  this->x = x;
  this->dim = dim;
  this->y = y;

  return;
}

int DataPoint::dimension(){
  return dim;
}

int DataPoint::numnz(){
  return nnz;
}

double DataPoint::label(){
  return y;
}


int* DataPoint::p_first_idx(){
  return indices;
}


int* DataPoint::p_last_idx(){
  return indices + nnz;
}

void DataPoint::addMultTo(double scale, double *dst){
  if (indices == NULL) {
    for(int i = 0; i < dim; i++){
      dst[i] += scale * x[i];
    }
  } else {
    for (int i = 0; i < nnz; i++) {
      dst[indices[i]] += scale * x[i];
    }
  }
  
  return;
}


double DataPoint::dot(double *vec){
  double res = 0.0;
  
  if (indices == NULL) {
    for(int i = 0; i < dim; i++){
      res += x[i] * vec[i];
    }
  } else {
    for (int i = 0; i < nnz; i++) {
      res += x[i] * vec[indices[i]];
    }
  }

  return res;
}

#endif
