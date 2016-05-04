LIBS=-lpthread -openmp -lnuma
FLAGS=-Ofast -std=c++11
CC=icc

all: hog_shared hog_mod_rep_core hog_mod_rep_node_avg cyc_shared cyc_mod_rep_node hog_mod_rep_node_add  hog_mod_rep_node_avg_all

ls_mod_rep_node_avg:	
	$(CC)  $(LIBS) $(FLAGS) main.cpp -DMETHOD=8 -o w2v_hog_mod_rep_node_avg

hog_shared:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=1 -o w2v_hog_shared
	#./w2v_hog_shared

hog_mod_rep_core:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=2 -o w2v_hog_mod_rep_core
	#./w2v_hog_mod_rep_core

hog_mod_rep_node_avg:
	$(CC)  $(LIBS) $(FLAGS) main.cpp -DMETHOD=3 -o w2v_hog_mod_rep_node_avg
	#./w2v_hog_mod_rep_node_avg

cyc_shared:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=4 -o w2v_cyc_shared
	#./w2v_cyc_shared

cyc_mod_rep_node:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=5 -o w2v_cyc_mod_rep_node
	#./w2v_cyc_mod_rep_node

hog_mod_rep_node_add:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=6 -o w2v_hog_mod_rep_node_add
	#./w2v_hog_mod_rep_nod_add

hog_mod_rep_node_avg_all:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=6 -o w2v_hog_mod_rep_node_avg_all
	#./w2v_hog_mod_rep_nod_avg_all

full:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=0 -o w2v_full
	#./w2v_full
