LIBS=-lpthread -openmp -lnuma
FLAGS=-Ofast -std=c++11
CC= g++

all: hogwild_one_node

hogwild_one_node:
	$(CC) $(FLAGS) main.cpp $(LIBS) -o hogwild_one_node
