# better to do in the future: 
# clang++ -I Eigen -O3 -std=c++11 Mat.cpp -o prog.o
# cuts time by 3 for stacked lstms

# rewire the linked libs:
# otool -L ./stuff.o
# install_name_tool -change libiomp5.dylib /opt/intel/composer_xe_2015.1.108/compiler/lib/libiomp5.dylib ./stuff.o
# otool -L ./stuff.o

CC = clang++
INTEL_DIR=/opt/intel/
INTEL_COMP_DIR=$(INTEL_DIR)composer_xe_2015.1.108/
MKL_DIR=$(INTEL_COMP_DIR)mkl/
INC=-IEigen -I$(MKL_DIR)include -IEigen/src/Core/util/
LLOC=-L$(MKL_DIR)lib/ -L$(INTEL_COMP_DIR)compiler/lib/
OBJECTS=utils.o Mat.o Layers.o cnpy.o
all:
	$(CC) $(INC) -c -std=c++11 main.cpp
	$(CC) $(INC) -c -std=c++11 utils.cpp
	$(CC) $(INC) -c -std=c++11 Mat.cpp
	$(CC) $(INC) -c -std=c++11 Layers.cpp
	$(CC) $(INC) -c -std=c++11 cnpy.cpp
	$(CC) -lblas -o app.o main.o utils.o Mat.o Layers.o cnpy.o

optimized:
	$(CC) $(INC) -O3 -c -std=c++11 main.cpp
	$(CC) $(INC) -O3 -c -std=c++11 utils.cpp
	$(CC) $(INC) -O3 -c -std=c++11 Mat.cpp
	$(CC) $(INC) -O3 -c -std=c++11 Layers.cpp
	$(CC) $(INC) -O3 -c -std=c++11 cnpy.cpp
	$(CC) $(LLOC) -lblas -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -o app.o -O3 main.o $(OBJECTS) -lm

layers:
	$(CC) $(INC) -O3 -c -std=c++11 Layers.cpp

program:
	$(CC) $(INC) -std=c++11 main.cpp -o main.o $(OBJECTS)

stacked_lstm:
	$(CC) $(INC) $(LLOC) -O3 -lblas -std=c++11 stacked_lstm.cpp -o stacked_lstm.o $(OBJECTS) 

character_predict:
	$(CC) $(INC) -O3 -lblas -std=c++11 character_prediction.cpp -o character_prediction.o $(OBJECTS) 

sparkfun_prediction:
	$(CC) $(INC) -lz -O3 -lblas -std=c++11 sparkfun_prediction.cpp -o sparkfun_prediction.o $(OBJECTS) 

maps:
	$(CC) $(INC) -O3 -std=c++11 maps.cpp -o maps.o $(OBJECTS)