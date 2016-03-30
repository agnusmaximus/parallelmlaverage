LIBS=-lpthread -lgomp -fopenmp -lnuma
FLAGS=-O3 -std=c++11
CC=g++

hog_shared:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=1 -o w2v_hog_shared
	./w2v_hog_shared

hog_mod_rep:
	$(CC) $(FLAGS) main.cpp $(LIBS) -DMETHOD=2 -o w2v_hog_mod_rep
	./w2v_hog_mod_rep
