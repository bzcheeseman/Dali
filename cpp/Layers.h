#ifndef RECURRENT_LAYERS_H
#define RECURRENT_LAYERS_H

#include "Mat.h"

template<typename T>
class Layer {
    /*
    Linear output layer of the form with affine offset bias
    vector b (broadcasted):

        > y = A * x + b

    */
    
    void create_variables();
    public:
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        shared_mat W;
        shared_mat b;
        const int hidden_size;
        const int input_size;
        Layer (int, int);
        shared_mat activate(Graph<T>&, shared_mat);
};

template<typename T>
class RNN {
    /*
    Combine the input of a hidden vector and an input vector
    into a single matrix product sum:

        > y = A * [x, h] + b

    */
    void create_variables();
    public:
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        shared_mat Wx;
        shared_mat Wh;
        shared_mat b;
        const int hidden_size;
        const int input_size;
        const int output_size;
        RNN (int, int);
        RNN (int, int, int);
        shared_mat activate(Graph<T>&, shared_mat, shared_mat);
};

template<typename T>
class GatedInput {
    typedef RNN<T>                   layer_type;
    public:
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        layer_type in_gate;
        GatedInput (int, int);
        shared_mat activate(Graph<T>&, shared_mat, shared_mat);
};

template<typename T>
class LSTM {
    /*

    LSTM layer with forget, output, memory write, and input
    modulate gates, that can remember sequences for long
    periods of time.

    See `Mat`, `HiddenLayer`
    */
    typedef RNN<T>                        layer_type;
    public:
        typedef Mat<T>                               mat;
        typedef std::shared_ptr<mat>          shared_mat;
        // cell input modulation:
        layer_type input_layer;
        // cell forget gate:
        layer_type forget_layer;
        // cell output modulation
        layer_type output_layer;
        // cell write params
        layer_type cell_layer;
        const int hidden_size;
        const int input_size;
        LSTM (int, int);
        LSTM (int&, int&);
        static std::pair<std::vector<shared_mat>, std::vector<shared_mat>> initial_states(std::vector<int>&);
        std::pair<shared_mat, shared_mat> activate(
            Graph<T>&,
            shared_mat,
            shared_mat,
            shared_mat);
};

template<typename celltype>
std::vector<celltype> StackedCells(const int&, const std::vector<int>&);

#endif