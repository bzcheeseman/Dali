#ifndef RECURRENT_LAYERS_H
#define RECURRENT_LAYERS_H

#include "Mat.h"
#include "Graph.h"

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
        std::vector<shared_mat> parameters() const;
        Layer (int, int);
        /**
        Layer<T>::Layer
        ---------------

        Copy constructor with option to make a shallow
        or deep copy of the underlying parameters.

        If the copy is shallow then the parameters are shared
        but separate gradients `dw` are used for each of 
        thread Layer<T>.

        Shallow copies are useful for Hogwild and multithreaded
        training

        See `Mat<T>::shallow_copy`, `examples/character_prediction.cpp`,
        `Layer<T>::shallow_copy`

        Inputs
        ------

          Layer<T> l : layer from which to source parameters and dw
         bool copy_w : whether parameters for new layer should be copies
                       or shared
        bool copy_dw : whether gradients for new layer should be copies
                       shared (Note: sharing `dw` should be used with
                       caution and can lead to unpredictable behavior
                       during optimization).

        Outputs
        -------

        Layer<T> out : the copied layer with deep or shallow copy

        **/
        Layer (const Layer&, bool, bool);
        shared_mat activate(Graph<T>&, shared_mat) const;
        Layer<T> shallow_copy() const;
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
        std::vector<shared_mat> parameters() const;
        RNN (int, int);
        RNN (int, int, int);
        RNN (const RNN&, bool, bool);
        shared_mat activate(Graph<T>&, shared_mat, shared_mat) const;
        RNN<T> shallow_copy() const;
};

template<typename T>
class ShortcutRNN {
    /*
    Combine the input of a hidden vector, an input vector, and 
    a second input vector (a shortcut) into a single matrix
    product sum, and also take an input from another layer as
    a "shortcut", s:

        > y = A * [x, s, h] + b

    */
    void create_variables();
    public:
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        shared_mat Wx;
        shared_mat Wh;
        shared_mat Ws;
        shared_mat b;
        const int hidden_size;
        const int input_size;
        const int shortcut_size;
        const int output_size;
        std::vector<shared_mat> parameters() const;
        ShortcutRNN (int, int, int);
        ShortcutRNN (int, int, int, int);
        ShortcutRNN (const ShortcutRNN&, bool, bool);
        shared_mat activate(Graph<T>&, shared_mat, shared_mat, shared_mat) const;
        ShortcutRNN<T> shallow_copy() const;
};

template<typename T>
class GatedInput {
    typedef RNN<T>                   layer_type;
    public:
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        std::vector<shared_mat> parameters() const;
        layer_type in_gate;
        GatedInput (int, int);
        GatedInput (const GatedInput&, bool, bool);
        shared_mat activate(Graph<T>&, shared_mat, shared_mat) const;
        GatedInput<T> shallow_copy() const;
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
    void name_internal_layers();
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
        LSTM (const LSTM&, bool, bool);
        std::vector<shared_mat> parameters() const;
        static std::pair<std::vector<shared_mat>, std::vector<shared_mat>> initial_states(std::vector<int>&);
        std::pair<shared_mat, shared_mat> activate(
            Graph<T>&,
            shared_mat,
            shared_mat,
            shared_mat) const;
        LSTM<T> shallow_copy() const;
};

template<typename T>
class ShortcutLSTM {
    /*

    ShortcutLSTM layer with forget, output, memory write, and input
    modulate gates, that can remember sequences for long
    periods of time.

    Unlike a traditional LSTM this layer type takes 2 inputs along
    with its previous hidden state: an input from the layer below,
    and a "shortcut" input from the base layer (or elsewhere).

    See `Mat`, `HiddenLayer`
    */
    typedef ShortcutRNN<T>          layer_type;
    void name_internal_layers();
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
        const int shortcut_size;
        ShortcutLSTM (int, int, int);
        ShortcutLSTM (int&, int&, int&);
        ShortcutLSTM (const ShortcutLSTM&, bool, bool);
        std::vector<shared_mat> parameters() const;
        std::pair<shared_mat, shared_mat> activate(
            Graph<T>&,
            shared_mat,
            shared_mat,
            shared_mat,
            shared_mat) const;
        ShortcutLSTM<T> shallow_copy() const;
};

template<typename celltype>
std::vector<celltype> StackedCells(const int&, const std::vector<int>&);

template<typename T>
std::vector<ShortcutLSTM<T>> StackedCells(const int&, const int&, const std::vector<int>&);

template<typename celltype>
std::vector<celltype> StackedCells(const std::vector<celltype>&, bool, bool);

template<typename T>
std::pair<std::vector<std::shared_ptr<Mat<T>>>, std::vector<std::shared_ptr<Mat<T>>>> forward_LSTMs(Graph<T>&,
    std::shared_ptr<Mat<T>>,
    std::pair<std::vector<std::shared_ptr<Mat<T>>>, std::vector<std::shared_ptr<Mat<T>>>>&,
    std::vector<LSTM<T>>&);

template<typename T>
std::pair<std::vector<std::shared_ptr<Mat<T>>>, std::vector<std::shared_ptr<Mat<T>>>> forward_LSTMs(Graph<T>&,
    std::shared_ptr<Mat<T>>,
    std::pair<std::vector<std::shared_ptr<Mat<T>>>, std::vector<std::shared_ptr<Mat<T>>>>&,
    LSTM<T>&,
    std::vector<ShortcutLSTM<T>>&);

#endif