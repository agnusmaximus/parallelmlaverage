LIBS=-lpthread -fopenmp -lnuma
FLAGS= -O3 -std=c++11
CC= g++ -g -Wall -I../graphs -I/usr/include -L/usr/lib/x86_64-linux-gnu/

all: hogwild_LS_one_node

hogwild_LS_one_node:
	$(CC) $(FLAGS) main.cpp $(LIBS) -o hogwild_LS_one_node

clean: 
	rm -f *.o hogwild_LS_one_node
