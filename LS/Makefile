LIBS=-lpthread -fopenmp -lnuma
FLAGS= -O3 -std=c++11
CC= g++ -I../graphs 

all: hogwild_LS_one_node test_shit

hogwild_LS_one_node:
	$(CC) $(FLAGS) main.cpp $(LIBS) -o hogwild_LS_one_node

test_shit:
	$(CC) $(FLAGS) test_shit.cpp $(LIBS) -o test_shit


clean: 
	rm -f *.o hogwild_LS_one_node test_shit
