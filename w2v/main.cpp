#include "src/util.h"
#include "src/params.h"
#include "src/hogwild.h"
#include "src/hog_w2v_shared.h"
#include "src/hog_w2v_model_replication.h"
#include "src/w2v_full_gradient.h"

#define FULL 0
#define HOG_SHARED 1
#define HOG_MOD_REP_PER_CORE 2
#define HOG_MOD_REP_PER_NODE 3

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
    if (METHOD == HOG_MOD_REP_PER_NODE)
	t_elapsed = hog_word_embeddings_model_replication_per_node();
    if (METHOD == HOG_MOD_REP_PER_CORE)
	t_elapsed = hog_word_embeddings_model_replication_per_core();
    //cout << t_elapsed << " ms " << endl;
}
