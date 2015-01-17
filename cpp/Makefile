# better to do in the future: 
# clang++ -I Eigen -O3 -std=c++11 Mat.cpp -o prog.o
# cuts time by 3 for stacked lstms
CC = clang++
all:
	$(CC) -I Eigen -c -std=c++11 main.cpp
	$(CC) -I Eigen -c -std=c++11 utils.cpp
	$(CC) -I Eigen -c -std=c++11 Mat.cpp
	$(CC) -I Eigen -c -std=c++11 Layers.cpp
	$(CC) -o app.o main.o utils.o Mat.o Layers.o

optimized:
	$(CC) -I Eigen -O3 -c -std=c++11 main.cpp
	$(CC) -I Eigen -O3 -c -std=c++11 utils.cpp
	$(CC) -I Eigen -O3 -c -std=c++11 Mat.cpp
	$(CC) -I Eigen -O3 -c -std=c++11 Layers.cpp
	$(CC) -o app.o -O3 main.o utils.o Mat.o Layers.o

layers:
	$(CC) -I Eigen -c -std=c++11 Layers.cpp
	$(CC) -o app.o main.o utils.o Mat.o Layers.o

program:
	$(CC) -I Eigen -std=c++11 main.cpp -o main.o utils.o Mat.o Layers.o

stacked_lstm:
	$(CC) -I Eigen -O3 -std=c++11 stacked_lstm.cpp -o stacked_lstm.o utils.o Mat.o Layers.o