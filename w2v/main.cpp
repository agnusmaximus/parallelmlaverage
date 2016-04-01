#include "src/util.h"
#include "src/params.h"
#include "src/sgd.h"
#include "src/cyclades.h"
#include "src/cyclades_model_replication.h"
#include "src/hogwild.h"
#include "src/hogwild_model_replication.h"
#include "src/w2v_full_gradient.h"

#define FULL 0
#define HOG_SHARED 1
#define HOG_MOD_REP_PER_CORE 2
#define HOG_MOD_REP_PER_NODE_AVG 3
#define CYC_SHARED 4
#define CYC_MOD_REP_PER_NODE 5
#define HOG_MOD_REP_PER_NODE_ADD 6

#ifndef METHOD
#define METHOD HOG_SHARED
#endif

using namespace std;

int main(void) {
    omp_set_num_threads(NTHREAD);
    int t_elapsed;
    if (METHOD == FULL) 
      t_elapsed = full_word_embeddings();
    if (METHOD == HOG_SHARED)
	t_elapsed = hog_word_embeddings_shared();
    if (METHOD == HOG_MOD_REP_PER_NODE_AVG)
	t_elapsed = hog_word_embeddings_model_replication_per_node_avg();
    if (METHOD == HOG_MOD_REP_PER_NODE_ADD)
	t_elapsed = hog_word_embeddings_model_replication_per_node_add();
    if (METHOD == HOG_MOD_REP_PER_CORE)
	t_elapsed = hog_word_embeddings_model_replication_per_core();
    if (METHOD == CYC_SHARED)
        t_elapsed = cyc_word_embeddings_shared();
    if (METHOD == CYC_MOD_REP_PER_NODE)
      t_elapsed = cyc_word_embeddings_model_replication_per_node();
    //cout << t_elapsed << " ms " << endl;
}
