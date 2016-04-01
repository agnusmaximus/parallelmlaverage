LIBS=-lpthread -lgomp -fopenmp -lnuma
FLAGS=-O3 -std=c++11
CC=g++

hog_shared:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=1 -o w2v_hog_shared
	#./w2v_hog_shared

hog_mod_rep_core:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=2 -o w2v_hog_mod_rep_core
	#./w2v_hog_mod_rep_core

hog_mod_rep_node:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=3 -o w2v_hog_mod_rep_node
	#./w2v_hog_mod_rep_node

cyc_shared:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=4 -o w2v_cyc_shared
	#./w2v_cyc_shared

cyc_mod_rep_node:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=5 -o w2v_cyc_mod_rep_node
	#./w2v_cyc_mod_rep_node

full:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=0 -o w2v_full
	#./w2v_full
