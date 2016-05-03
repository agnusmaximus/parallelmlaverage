#ifndef __BLOCKER__
#define __BLOCKER__


enum Algorithm{
  SIMPLE_BFS
};


class GraphBlocker{
 public:
  vector<int> datapoints_blocks;
  vector<int> parameters_shuffle;
  GraphBlocker();
  GraphBlocker(int);
  void execute(BipartiteGraph&, Algorithm);
 private:
  void simple_bfs(BipartiteGraph&);
};


GraphBlocker::GraphBlocker(){
  datapoints_blocks = vector<int>();
  parameters_shuffle = vector<int>();
}

GraphBlocker::GraphBlocker(int num_left_nodes){
  datapoints_blocks = vector<int>(num_left_nodes);
  parameters_shuffle = vector<int>();
}


void simple_bfs(BipartiteGraph&, GraphBlocker*);


void GraphBlocker::execute(BipartiteGraph &graph, Algorithm alg){
  
  for(int i = 0; i < graph.num_nodes(LEFT); i++)
    datapoints_blocks[i] = -1;

  if(alg == SIMPLE_BFS){
    this->simple_bfs(graph);
  }
}


void GraphBlocker::simple_bfs(BipartiteGraph &graph){
  int num_left_nodes = graph.num_nodes(LEFT);
  int num_right_nodes = graph.num_nodes(RIGHT);
  

  for(int i = 0, current_block = 0; i < num_left_nodes; i ++){
    if(datapoints_blocks[i] != -1){
      continue;
    }
    
    vector<int>* neighbors = graph.neighbors(i, LEFT);
    
    for(int j = 0; j < neighbors->size(); j++){
      int current_rightNode = neighbors->at(j);
      vector<int>* neighbors_of_rightNodes = graph.neighbors(current_rightNode, RIGHT);

      for(int k = 0; k < neighbors_of_rightNodes->size(); k ++){
	int current_leftNode = neighbors_of_rightNodes->at(k);
	if(datapoints_blocks[current_leftNode] != -1)
	  continue;
	
	datapoints_blocks[current_leftNode] = current_block;
	
      }
    }


    current_block++;
  }
  
  return;
}




#endif
