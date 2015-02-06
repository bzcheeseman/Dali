# better to do in the future: 
# clang++ -I Eigen -O3 -std=c++11 Mat.cpp -o prog.o
# cuts time by 3 for stacked lstms

CC = clang++
INTEL_DIR=/opt/intel/
INTEL_COMP_DIR=$(INTEL_DIR)composer_xe_2015.1.108/
MKL_DIR=$(INTEL_COMP_DIR)mkl/
INC=-IEigen -I$(MKL_DIR)include -IEigen/src/Core/util/
LLOC=-L$(MKL_DIR)lib/ -L$(INTEL_COMP_DIR)compiler/lib/
OBJECTS=utils.o Mat.o Layers.o cnpy.o OptionParser/OptionParser.o CrossEntropy.o Softmax.o StackedGatedModel.o
LIBS=-lblas -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -headerpad_max_install_names
FIX_DYLIBS=sudo sh fix_dylib.sh
all:
	$(CC) -c -std=c++11 OptionParser/OptionParser.cpp -o OptionParser/OptionParser.o
	$(CC) $(INC) -c -std=c++11 main.cpp
	$(CC) $(INC) -c -std=c++11 utils.cpp
	$(CC) $(INC) -c -std=c++11 Mat.cpp
	$(CC) $(INC) -c -std=c++11 Layers.cpp
	$(CC) $(INC) -c -std=c++11 cnpy.cpp
	$(CC) $(INC) -c -std=c++11 Softmax.cpp
	$(CC) $(INC) -c -std=c++11 CrossEntropy.cpp
	$(CC) $(INC) -c -std=c++11 StackedGatedModel.cpp
	$(CC) $(LLOC) -lblas -o app.o main.o $(OBJECTS) -lm
	$(FIX_DYLIBS)

optimized:
	$(CC) -O3 -c -std=c++11 OptionParser/OptionParser.cpp -o OptionParser/OptionParser.o
	$(CC) $(INC) -O3 -c -std=c++11 main.cpp
	$(CC) $(INC) -O3 -c -std=c++11 utils.cpp
	$(CC) $(INC) -O3 -c -std=c++11 Mat.cpp
	$(CC) $(INC) -O3 -c -std=c++11 Layers.cpp
	$(CC) $(INC) -O3 -c -std=c++11 cnpy.cpp
	$(CC) $(INC) -O3 -c -std=c++11 Softmax.cpp
	$(CC) $(INC) -O3 -c -std=c++11 CrossEntropy.cpp
	$(CC) $(INC) -O3 -c -std=c++11 StackedGatedModel.cpp
	$(CC) $(LLOC) $(LIBS) -o app.o -O3 main.o $(OBJECTS)
	$(FIX_DYLIBS)

softmax:
	$(CC) $(INC) -O3 -c -std=c++11 Softmax.cpp
	$(FIX_DYLIBS)

entropy:
	$(CC) $(INC) -O3 -c -std=c++11 CrossEntropy.cpp
	$(FIX_DYLIBS)

stacked_lstm:
	$(CC) $(INC) $(LLOC) $(LIBS) -O3 -std=c++11 stacked_lstm.cpp -o stacked_lstm.o $(OBJECTS) 
	$(FIX_DYLIBS)

utils:
	$(CC) $(INC) -O3 -c -std=c++11 utils.cpp
	$(FIX_DYLIBS)

character_prediction:
	$(CC) $(INC) $(LLOC) $(LIBS) -O3 -std=c++11 examples/character_prediction.cpp -o character_prediction.o $(OBJECTS)
	$(FIX_DYLIBS)

sparkfun_prediction:
	$(CC) $(INC) -lblas -lz -O3 -std=c++11 examples/sparkfun_prediction.cpp -o examples/sparkfun_prediction.o $(OBJECTS)
	$(FIX_DYLIBS)

test_lattice:
	$(CC) -O3 -c -std=c++11 OptionParser/OptionParser.cpp -o OptionParser/OptionParser.o
	$(CC) $(INC) -O3 -c -std=c++11 utils.cpp
	$(CC) -O3 -std=c++11 examples/test_lattice.cpp -o examples/test_lattice.o utils.o OptionParser/OptionParser.o

lattice_prediction:
	$(CC) $(INC) -lblas -lz -O3 -std=c++11 examples/lattice_prediction.cpp -o examples/lattice_prediction.o $(OBJECTS)
	$(FIX_DYLIBS)