#include "util.h"
#include "params.h"

//For load balancing
struct Comp
{
  bool operator()(const pair<int, int>& s1, const pair<int, int>& s2) {
    return s1.second > s2.second;
  }
};

void distribute_ccs(map<int, vector<int> > &ccs, vector<DataPoint *> &access_pattern, vector<vector<int> > &access_length, 
		    vector<vector<int> > &batch_index_start, int batchnum, vector<DataPoint> &points, vector<vector<int> > &order,
		    int *thread_load_balance, size_t *cur_bytes_allocated, int *cur_datapoints_used,
		    int *core_to_node) {
  
  int chosen_threads[ccs.size()];
  int total_size_needed[NTHREAD];
  int count = 0;
  vector<pair<int, int> > balances;

  for (int i = 0; i < NTHREAD; i++) {
    total_size_needed[i] = 0;
    balances.push_back(pair<int, int>(i, 0));
  }

  //Greedily load balance datapoints
  for (map<int, vector<int> >::iterator it = ccs.begin(); it != ccs.end(); it++, count++) {
    pair<int, int> best_balance = balances.front();
    int chosen_thread = best_balance.first;
    chosen_threads[count] = chosen_thread;
    pop_heap(balances.begin(), balances.end(), Comp()); balances.pop_back();
    vector<int> cc = it->second;
    total_size_needed[chosen_thread] += cc.size();
    best_balance.second += cc.size();
    balances.push_back(best_balance); push_heap(balances.begin(), balances.end(), Comp());
  }

  for (int i = 0; i < NTHREAD; i++) {
    //cout << total_size_needed[i] << " ";
  }
  //cout << endl;

  //Allocate memory
  int index_count[NTHREAD];
  int max_load = 0;
  for (int i = 0; i < NTHREAD; i++) {
    int numa_node = core_to_node[i];

    batch_index_start[i][batchnum] = cur_datapoints_used[i];
    if (cur_bytes_allocated[i] == 0) {
      size_t new_size = (size_t)total_size_needed[i] * sizeof(DataPoint);
      new_size = ((new_size / numa_pagesize()) + 1) * numa_pagesize();
      access_pattern[i] = (DataPoint *)numa_alloc_onnode(new_size, numa_node);
      cur_bytes_allocated[i] = new_size;
    }
    else {
      if ((cur_datapoints_used[i] + total_size_needed[i])*sizeof(DataPoint) >=
	  cur_bytes_allocated[i]) {
	size_t old_size = (size_t)(cur_bytes_allocated[i]);
	size_t new_size = (size_t)((cur_datapoints_used[i] + total_size_needed[i]) * sizeof(DataPoint));
	new_size = ((new_size / numa_pagesize()) + 1) * numa_pagesize();	
	access_pattern[i] = (DataPoint *)numa_realloc(access_pattern[i], old_size, new_size);
	cur_bytes_allocated[i] = new_size;
      }
    }
    cur_datapoints_used[i] += total_size_needed[i];
    order[i].resize(cur_datapoints_used[i]);
    if (access_pattern[i] == NULL) {
      cout << "OOM" << endl;
      exit(0);
    }      
    access_length[i][batchnum] = total_size_needed[i];
    index_count[i] = 0;
    thread_load_balance[i] += total_size_needed[i];
  }

  //Copy memory over
  count = 0;  
  for (map<int, vector<int> >::iterator it = ccs.begin(); it != ccs.end(); it++, count++) {
    vector<int> cc = it->second;
    int chosen_thread = chosen_threads[count];
    for (int i = 0; i < cc.size(); i++) {
      access_pattern[chosen_thread][index_count[chosen_thread]+batch_index_start[chosen_thread][batchnum] + i] = points[cc[i]];
      order[chosen_thread][index_count[chosen_thread]+batch_index_start[chosen_thread][batchnum] + i] = cc[i]+1;
    }
    index_count[chosen_thread] += cc.size();
  }
}

int union_find(int a, int *p) {
  int root = a;
  while (p[a] != a) {
    a = p[a];
  }
  while (root != a) {    
    int root2 = p[root];
    p[root] = a;
    root = root2;
  }
  
  return a;
}

void compute_CC_thread(map<int, vector<int> > &CCs, vector<DataPoint> &points, int start, int end, int thread_id, int *tree) {
  pin_to_core(thread_id);

  for (long long int i = 0; i < end-start + N_NODES; i++) 
    tree[i] = i;

  for (int i = start; i < end; i++) {
    DataPoint p = points[i];
    int src = i-start;
    int e1 = get<0>(p) + end-start;
    int e2 = get<1>(p) + end-start;
    int c1 = union_find(src, tree);
    int c2 = union_find(e1, tree);
    int c3 = union_find(e2, tree);
    tree[c3] = c1;
    tree[c2] = c1;
  }
  for (int i = 0; i < end-start; i++) {
    int group = union_find(i, tree);
    CCs[group].push_back(i+start);    
  }
}

long int cyc_word_embeddings_shared() {

    double C = 0;
    double *C_sum_mult[NTHREAD];
    double *C_sum_mult2[NTHREAD];
    int *tree_thread[NTHREAD];
    double *model;
    int thread_load_balance[NTHREAD];
    size_t cur_bytes_allocated[NTHREAD];
    int cur_datapoints_used[NTHREAD];

    //Initialization / read data block
    vector<DataPoint> points = get_word_embeddings_data(WORD_EMBEDDINGS_FILE);
    random_shuffle(points.begin(), points.end());
    allocate_memory_model(&model, N_NODES, K);
    initialize_model(model, N_NODES, K);
    
    //Initialize variables for CC distribution
    for (int i = 0; i < NTHREAD; i++) 
      tree_thread[i] = (int *)malloc(sizeof(int) * (CYC_BATCH_SIZE + N_NODES));
    memset(thread_load_balance, 0, sizeof(int) * NTHREAD);
    memset(cur_bytes_allocated, 0, sizeof(size_t) * NTHREAD);
    memset(cur_datapoints_used, 0, sizeof(int) * NTHREAD);

    //Create a map from core/thread -> node
    int core_to_node[NTHREAD];
    for (int i = 0; i < NTHREAD; i++) core_to_node[i] = -1;
    for (int i = 0; i < NTHREAD; i++) {
      core_to_node[i] = numa_node_of_cpu(i);
    }

    long int start_time = get_time();

    //Cyclades access pattern construction - connected components
    int n_batches = (int)ceil((points.size() / (double)CYC_BATCH_SIZE));
    map<int, vector<int> > CCs[n_batches];
#pragma omp parallel for
    for (int i = 0; i < n_batches; i++) {
      int start = i * CYC_BATCH_SIZE;
      int end = min((i+1)*CYC_BATCH_SIZE, (int)points.size());
      compute_CC_thread(CCs[i], points, start, end, omp_get_thread_num(), tree_thread[omp_get_thread_num()]);
    }

    //Cyclades access pattern construction - distribution of work
    vector<DataPoint *> datapoints_per_thread(NTHREAD);
    vector<vector<int > > access_length(NTHREAD);
    vector<vector<int > > batch_index_start(NTHREAD);
    vector<vector<int > > order(NTHREAD);
    for (int i = 0; i < NTHREAD; i++) {
      access_length[i].resize(n_batches);
      batch_index_start[i].resize(n_batches);
      order[i].resize(n_batches);
    }
    for (int i = 0; i < n_batches; i++) {
      distribute_ccs(CCs[i], datapoints_per_thread, access_length, 
		     batch_index_start, i, points, order,
		     thread_load_balance, cur_bytes_allocated, cur_datapoints_used,
		     core_to_node);
    }

    //Readjust memory for C_sum mults
    for (int i = 0; i < NTHREAD; i++) {
      C_sum_mult[i] = (double *)malloc(sizeof(double) * order[i].size());
      C_sum_mult2[i] = (double *)malloc(sizeof(double) * order[i].size());
    }
    
    //Divide to threads
    float copy_time = 0;
    volatile int thread_sync[NTHREAD];
    for (int i = 0; i < N_EPOCHS; i++) {

	if (PRINT_LOSS) {
	  cout << get_time() - start_time << " " << compute_loss(points, model, C, K) << endl;
	}

	memset((int *)thread_sync, 0, sizeof(int) * NTHREAD);

	//Cyclades
#pragma omp parallel for
	for (int j = 0; j < NTHREAD; j++) {
	  sgd_cyc(datapoints_per_thread[j], access_length[j], batch_index_start[j], order[j], 
		  K, model, C_sum_mult, C_sum_mult2, C, j, thread_sync);
	}

	//Optimize C
	double C_A = 0, C_B = 0;
#pragma omp parallel for reduction(+:C_A,C_B)
	for (int t = 0; t < NTHREAD; t++) {
	    for (int d = 0; d < order[t].size(); d++) {
		C_A += C_sum_mult[t][d];
		C_B += C_sum_mult2[t][d];
	    }
	}
	C = C_A / C_B;

	GAMMA *= GAMMA_REDUCTION;
    }

    return get_time() - start_time;
}
