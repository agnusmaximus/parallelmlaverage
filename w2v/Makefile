LIBS=-lpthread -lnuma -lgomp -fopenmp
FLAGS=-O3 -std=c++11
CC=g++

cyc_word_embeddings_hog:
	rm -rf word_embeddings_hog
	$(CC) $(FLAGS) w2v.cpp $(LIBS) -o w2v_hog
	./w2v_hog
