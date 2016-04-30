LIBS=-lpthread -fopenmp
FLAGS=-Ofast -std=c++11
CC= g++ -g -Wall

all: hogwild_LS_one_node

hogwild_LS_one_node:
	$(CC) $(FLAGS) main.cpp $(LIBS) -o hogwild_LS_one_node

clean: 
	rm -f *.o hogwild_LS_one_node
