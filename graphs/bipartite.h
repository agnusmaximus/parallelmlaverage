#ifndef __BIPARTITE__
#define __BIPARTITE__


#include <vector>


using namespace std;

typedef vector<vector<int>> adj_list_t;

enum Side{
  LEFT,
  RIGHT
};

class BipartiteGraph{
 private:
  adj_list_t left;
  adj_list_t right;

 public:
  BipartiteGraph();
  BipartiteGraph(adj_list_t& left, adj_list_t& right);
  
  vector<int>* neighbors(int node, Side side);  
  int num_nodes(Side side);
};


BipartiteGraph::BipartiteGraph(){
  left = adj_list_t();
  right = adj_list_t();
}

BipartiteGraph::BipartiteGraph(adj_list_t& left, adj_list_t& right){
  this->left = left;
  this->right = right;
}

int BipartiteGraph::num_nodes(Side side){
  if(side == LEFT)
    return left.size();
  else if(side == RIGHT)
    return right.size();
  
  return -1;
}

vector<int>* BipartiteGraph::neighbors(int node, Side side){
  if(side == LEFT)
    return &left[node];
  if(side == RIGHT)
    return &right[node];
  return NULL;
}


#endif
