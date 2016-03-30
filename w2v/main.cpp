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

using namespace std;

int main(void) {
    omp_set_num_threads(NTHREAD);
    hog_word_embeddings_shared();
}
