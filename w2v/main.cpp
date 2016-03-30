#include <iostream>
#include <cstdlib>
#include <string.h>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sstream>
#include <set>
#include <sys/time.h>
#include <thread>
#include <omp.h>
//#include <numa.h>
#include "util.h"
#include "params.h"
#include "hogwild.h"
#include "hog_w2v_shared.h"
#include "hog_w2v_model_replication.h"

#define HOG_SHARED 1
#define HOG_MOD_REP 2

#ifndef METHOD
#define METHOD HOG_SHARED
#endif

using namespace std;

int main(void) {
    omp_set_num_threads(NTHREAD);
    int t_elapsed;
    if (METHOD == HOG_SHARED)
	t_elapsed = hog_word_embeddings_shared();
    if (METHOD == HOG_MOD_REP)
	t_elapsed = hog_word_embeddings_model_replication();
    cout << t_elapsed << " ms " << endl;
}
