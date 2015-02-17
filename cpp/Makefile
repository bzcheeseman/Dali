# better to do in the future:
# clang++ -I Eigen -O3 -std=c++11 Mat.cpp -o prog.o
# cuts time by 3 for stacked lstms

CC = clang++
INCLUDE_DIR=/usr/include/
INTEL_DIR=/opt/intel/
INTEL_COMP_DIR=$(INTEL_DIR)composer_xe_2015.1.108/
MKL_DIR=$(INTEL_COMP_DIR)mkl/
INC=-I$(INCLUDE_DIR)eigen3/Eigen -I$(MKL_DIR)include -I$(INCLUDE_DIR)eigen3/Eigen/src/Core/util/
LLOC=-L$(MKL_DIR)lib/ -L$(INTEL_COMP_DIR)compiler/lib/
OBJECTS=utils.o Mat.o Layers.o cnpy.o OptionParser/OptionParser.o CrossEntropy.o Softmax.o StackedModel.o StackedGatedModel.o examples/protobuff/corpus.pb.o SST.o
BLAS_AND_MKL_LIBS=$(shell [ -d $(INTEL_COMP_DIR) ] && echo -lblas -lmkl_intel_thread -lmkl_core -liomp5)
LIBS=$(BLAS_AND_MKL_LIBS) -lz -lprotobuf -lpthread -lm -headerpad_max_install_names
FIX_DYLIBS=$(shell [ -d $(INTEL_COMP_DIR) ] && echo sudo sh fix_dylib.sh)
INSTANTIATOR=python3 template_instantiator.py CrossEntropy.cpp.template
PROTO_DIR=examples/protobuff/
PROTO_COMPILE=protoc --cpp_out=. --python_out=.
all:
	echo $(shell pushd $(PROTO_DIR) && $(PROTO_COMPILE) corpus.proto && popd)
	$(CC) -c -std=c++11 OptionParser/OptionParser.cpp -o OptionParser/OptionParser.o
	$(CC) $(INC) -c -std=c++11 main.cpp
	$(CC) $(INC) -c -std=c++11 utils.cpp
	$(CC) $(INC) -c -std=c++11 SST.cpp
	$(CC) $(INC) -c -std=c++11 Mat.cpp
	$(CC) $(INC) -c -std=c++11 Layers.cpp
	$(CC) $(INC) -c -std=c++11 cnpy.cpp
	$(CC) $(INC) -c -std=c++11 Softmax.cpp
	$(INSTANTIATOR)
	$(CC) $(INC) -c -std=c++11 CrossEntropy.cpp
	$(CC) $(INC) -c -std=c++11 StackedModel.cpp
	$(CC) $(INC) -c -std=c++11 StackedGatedModel.cpp
	$(CC) -c -std=c++11 $(PROTO_DIR)corpus.pb.cc -o $(PROTO_DIR)corpus.pb.o
	$(CC) $(LLOC) $(LIBS) -o app.o main.o $(OBJECTS) -lm
	$(FIX_DYLIBS)

optimized:
	echo $(shell pushd $(PROTO_DIR) && $(PROTO_COMPILE) corpus.proto && popd)
	$(CC) -O3 -c -std=c++11 OptionParser/OptionParser.cpp -o OptionParser/OptionParser.o
	$(CC) $(INC) -O3 -c -std=c++11 main.cpp
	$(CC) $(INC) -O3 -c -std=c++11 utils.cpp
	$(CC) $(INC) -O3 -c -std=c++11 SST.cpp
	$(CC) $(INC) -O3 -c -std=c++11 Mat.cpp
	$(CC) $(INC) -O3 -c -std=c++11 Layers.cpp
	$(CC) $(INC) -O3 -c -std=c++11 cnpy.cpp
	$(CC) $(INC) -O3 -c -std=c++11 Softmax.cpp
	$(INSTANTIATOR)
	$(CC) $(INC) -O3 -c -std=c++11 CrossEntropy.cpp
	$(CC) $(INC) -O3 -c -std=c++11 StackedModel.cpp
	$(CC) $(INC) -O3 -c -std=c++11 StackedGatedModel.cpp
	$(CC) -O3 -c -std=c++11 $(PROTO_DIR)corpus.pb.cc -o $(PROTO_DIR)corpus.pb.o
	$(CC) $(LLOC) $(LIBS) -o app.o -O3 main.o $(OBJECTS)
	$(FIX_DYLIBS)

mat:
	$(CC) $(INC) -O3 -c -std=c++11 Mat.cpp
	$(FIX_DYLIBS)

stacked_gated:
	$(CC) $(INC) -O3 -c -std=c++11 StackedModel.cpp
	$(CC) $(INC) -O3 -c -std=c++11 StackedGatedModel.cpp
	$(FIX_DYLIBS)

softmax:
	$(CC) $(INC) -O3 -c -std=c++11 Softmax.cpp
	$(FIX_DYLIBS)

entropy:
	$(INSTANTIATOR)
	$(CC) $(INC) -O3 -c -std=c++11 CrossEntropy.cpp
	$(FIX_DYLIBS)

stacked_lstm:
	$(CC) $(INC) $(LLOC) $(LIBS) -O3 -std=c++11 stacked_lstm.cpp -o stacked_lstm.o $(OBJECTS)
	$(FIX_DYLIBS)

utils:
	$(CC) $(INC) -O3 -c -std=c++11 utils.cpp
	$(FIX_DYLIBS)

SST:
	$(CC) $(INC) -c -std=c++11 SST.cpp
	$(FIX_DYLIBS)

character_prediction: utils mat
	$(CC) $(INC) $(LLOC) $(LIBS) -O3 -std=c++11 examples/character_prediction.cpp -o examples/character_prediction.o $(OBJECTS)
	$(FIX_DYLIBS)

sparkfun_prediction:
	$(CC) $(INC) -lblas -lz -O3 -std=c++11 examples/sparkfun_prediction.cpp -o examples/sparkfun_prediction.o $(OBJECTS)
	$(FIX_DYLIBS)

language_sentiment:
	$(CC) $(INC) -lblas -lprotobuf -lz -O3 -std=c++11 examples/language_model_from_senti.cpp -o examples/language_model_from_senti.o $(OBJECTS)
	$(FIX_DYLIBS)

language_model:
	$(CC) $(INC) $(BLAS_AND_MKL_LIBS) -lprotobuf -lz -O3 -std=c++11 examples/language_model.cpp -o examples/language_model.o $(OBJECTS)
	$(FIX_DYLIBS)

test_lattice:
	$(CC) -O3 -c -std=c++11 OptionParser/OptionParser.cpp -o OptionParser/OptionParser.o
	$(CC) $(INC) -O3 -c -std=c++11 utils.cpp
	$(CC) -O3 -lz -std=c++11 examples/test_lattice.cpp -o examples/test_lattice.o utils.o OptionParser/OptionParser.o

lattice_prediction:
	$(CC) $(INC) -lblas -lz -O3 -std=c++11 examples/lattice_prediction.cpp -o examples/lattice_prediction.o $(OBJECTS)
	$(FIX_DYLIBS)

lattice_prediction_protobuff:
	$(CC) $(INC) -lblas -lprotobuf -lz -O3 -std=c++11 examples/lattice_prediction_from_protobuff.cpp -o examples/lattice_prediction_from_protobuff.o $(OBJECTS)
	$(FIX_DYLIBS)
