#ifndef __BLOCKER__
#define __BLOCKER__


#include <queue>

using namespace std;

enum Algorithm{
  SIMPLE_BFS
};


class GraphBlocker{
 public:
  vector<int> datapoints_blocks;
  vector<int> parameters_shuffle;
  vector<int> offsets;

  GraphBlocker();
  GraphBlocker(int);
  void execute(BipartiteGraph&, Algorithm, int);
 private:
  void simple_bfs(BipartiteGraph&, int);
};


GraphBlocker::GraphBlocker(){
  datapoints_blocks = vector<int>();
  parameters_shuffle = vector<int>();
}

GraphBlocker::GraphBlocker(int num_left_nodes){
  datapoints_blocks = vector<int>(num_left_nodes);
  offsets = vector<int>(1,0);
  parameters_shuffle = vector<int>();
}


void simple_bfs(BipartiteGraph&, GraphBlocker*);


void GraphBlocker::execute(BipartiteGraph &graph, Algorithm alg, int threshold){
  
  offsets.resize(1,0);
  datapoints_blocks.resize(graph.num_nodes(LEFT), -1);

  for(int i = 0; i < graph.num_nodes(LEFT); i++)
    datapoints_blocks[i] = -1;

  
  if(alg == SIMPLE_BFS){
    this->simple_bfs(graph, threshold);
  }
}


void GraphBlocker::simple_bfs(BipartiteGraph &graph, int threshold){
  int num_left_nodes = graph.num_nodes(LEFT);
  int num_right_nodes = graph.num_nodes(RIGHT);
  int num_assigned = 0, current_block_size = 0, smallness_flag = 0, current_block = 0;

  
  for(int i = 0; i < num_left_nodes; i ++){
    if(datapoints_blocks[i] != -1){
      continue;
    }
    else if(smallness_flag == 1){
      datapoints_blocks[i] = current_block;
      num_assigned++;
      current_block_size++;
      continue;
    }


    queue<int> current_block_queue;
    int current_block_size = 1;
    
    current_block_queue.push(i);
    datapoints_blocks[i] = current_block;
    num_assigned++;

    while(!current_block_queue.empty() && 
	  current_block_size < threshold && 
	  num_assigned < num_left_nodes){

      int current_node = current_block_queue.front();
      current_block_queue.pop();
      
      vector<int>* neighbors_current_node = graph.neighbors(current_node, LEFT);
      
      for(int j = 0; j < neighbors_current_node->size() && current_block_size < threshold; j++){
	int current_rightNode = neighbors_current_node->at(j);
	vector<int>* neighbors_current_rightNodes = graph.neighbors(current_rightNode, RIGHT);

	for(int k = 0; k < neighbors_current_rightNodes->size() && current_block_size < threshold; k ++){
	  int current_leftNode = neighbors_current_rightNodes->at(k);
	  if(datapoints_blocks[current_leftNode] != -1)
	    continue;

	  datapoints_blocks[current_node] = current_block;
	  current_block_size++;
	  num_assigned++;

	  current_block_queue.push(current_leftNode);
	}
      }
    }
    
    if(smallness_flag == 0 && current_block_size >= threshold/2){
      offsets.push_back(current_block_size + offsets.back());
      current_block_size = 0;
      current_block++;
    }

    if(num_left_nodes - num_assigned <= threshold){
      smallness_flag = 1;
    }

  }
  
  if(smallness_flag == 0)
    offsets.pop_back();

  return;
}





#endif
