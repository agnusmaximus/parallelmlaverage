#ifndef __BLOCKER__
#define __BLOCKER__


#include <queue>
#include <vector>
#include <set>

using namespace std;

enum Algorithm{
  SIMPLE_BFS,
  GREEDY
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
  void greedy(BipartiteGraph&, int);
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


void simple_bfs(BipartiteGraph&, int);
void greedy(BipartiteGraph&, int);


void GraphBlocker::execute(BipartiteGraph &graph, Algorithm alg, int threshold){
  
  offsets.resize(1,0);
  datapoints_blocks.resize(graph.num_nodes(LEFT), -1);

  if(threshold > graph.num_nodes(LEFT)){
    threshold = graph.num_nodes(LEFT);
  }

  for(int i = 0; i < graph.num_nodes(LEFT); i++)
    datapoints_blocks[i] = -1;

  
  if(alg == SIMPLE_BFS){
    this->simple_bfs(graph, threshold);
  } else if(alg == GREEDY){
    this->greedy(graph, threshold);
  }
}


void print_step(int step){
  printf("Step %d\n", step);
  fflush(stdout);
  return;
}



void GraphBlocker::greedy(BipartiteGraph &graph, int threshold){
  int num_left_nodes = (int) graph.num_nodes(LEFT);
  int num_blocks = num_left_nodes/threshold;


  vector<set<int> > params_per_block(num_blocks);
  vector<int> blocks_sizes(num_blocks);



  for(int i = 0; i < num_left_nodes; i++){
    if(datapoints_blocks[i] != -1){
      continue;
    }


    vector<int>* neighbors_current_node = graph.neighbors(i, LEFT);
    int max = -1;
    int max_idx;
    int size_of_max_block = -1;

     for(int j = 0; j < num_blocks; j++){
       int count = 0;

       for(int k = 0; k < neighbors_current_node->size(); k++){

	 int current_rightNode = neighbors_current_node->at(k);
	 count += params_per_block[j].count(current_rightNode);
       }
       
       if(count > max || (count == max && blocks_sizes[j] < size_of_max_block)){
	 max = count;
	 max_idx = j;
	 size_of_max_block = blocks_sizes[j];
       }
     }

     datapoints_blocks[i] = max_idx;
     blocks_sizes[max_idx] ++;

     for(int k = 0; k < neighbors_current_node->size(); k++){
       int current_rightNode = neighbors_current_node->at(k);
       params_per_block[max_idx].insert(current_rightNode);
     }
     
  } 

  for(int i = 0; i < num_blocks - 1 ; i ++){
    offsets.push_back(offsets.back() + blocks_sizes[i]);
  }

  return;
}




void GraphBlocker::simple_bfs(BipartiteGraph &graph, int threshold){
  int num_left_nodes = graph.num_nodes(LEFT);
  int num_right_nodes = graph.num_nodes(RIGHT);
  int num_assigned = 0, current_block_size = 0, current_block = 0;

  bool smallness_flag = false;
  
  for(int i = 0; i < num_left_nodes; i ++){
    if(datapoints_blocks[i] != -1){
      continue;
    }
    else if(smallness_flag){
      datapoints_blocks[i] = current_block;
      num_assigned++;
      current_block_size++;
      continue;
    }

    queue<int> current_block_queue;
    
    current_block_queue.push(i);
    datapoints_blocks[i] = current_block;
    current_block_size++;
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

	  datapoints_blocks[current_leftNode] = current_block;
	  current_block_size++;
	  num_assigned++;

	  current_block_queue.push(current_leftNode);
	}
      }
    }
    
    if(!smallness_flag && current_block_size >= threshold/2){
      offsets.push_back(current_block_size + offsets.back());
      current_block_size = 0;
      current_block++;
    }

    if(num_left_nodes - num_assigned <= threshold){
      smallness_flag = true;
    }

  }
  
  if(!smallness_flag)
    offsets.pop_back();

  return;
}





#endif
