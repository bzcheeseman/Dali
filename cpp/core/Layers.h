#ifndef RECURRENT_LAYERS_H
#define RECURRENT_LAYERS_H

#include "Mat.h"
#include "Graph.h"
#include <initializer_list>
#include "core/Seq.h"

#define MAT Mat<T>
#define SHARED_MAT std::shared_ptr<Mat<T>>
#define GRAPH Graph<T>

template<typename T>
class AbstractLayer {
    public:
        virtual std::vector<SHARED_MAT> parameters() const = 0;
};

template<typename T>
class AbstractMultiInputLayer : public AbstractLayer<T> {
    public:
        SHARED_MAT b;
        virtual SHARED_MAT activate(GRAPH&, SHARED_MAT) const = 0;
        virtual SHARED_MAT activate(GRAPH&, const std::vector<SHARED_MAT>&) const;
        virtual SHARED_MAT activate(GRAPH&, SHARED_MAT, const std::vector<SHARED_MAT>&) const;
};

template<typename T>
class Layer : public AbstractMultiInputLayer<T> {
    /*
    Linear output layer of the form with affine offset bias
    vector b (broadcasted):

        > y = A * x + b

    */
    void create_variables();
    public:
        typedef T value_t;
        SHARED_MAT W;
        const int hidden_size;
        const int input_size;
        virtual std::vector<SHARED_MAT> parameters() const;
        Layer (int, int);

        Layer (const Layer&, bool, bool);
        SHARED_MAT activate(GRAPH&, SHARED_MAT) const;

        Layer<T> shallow_copy() const;
};

template<typename T>
class StackedInputLayer : public AbstractMultiInputLayer<T> {
    /*
    Linear output layer of the form with affine offset bias
    vector b (broadcasted), taking as inputs many different
    vectors of disparate sizes:

        > y = [A_1, ..., A_n]^T * [x_1, ..., x_n] + b

    */
    void create_variables();
    std::vector<std::shared_ptr<MAT>> zip_inputs_with_matrices_and_bias(const std::vector<std::shared_ptr<MAT>>&) const;
    std::vector<std::shared_ptr<MAT>> zip_inputs_with_matrices_and_bias(std::shared_ptr<MAT>, const std::vector<std::shared_ptr<MAT>>&) const;
    public:
        typedef T value_t;
        std::vector<SHARED_MAT> matrices;
        const int hidden_size;
        const std::vector<int> input_sizes;
        virtual std::vector<SHARED_MAT> parameters() const;
        StackedInputLayer (std::initializer_list<int>, int);
        StackedInputLayer (std::vector<int>, int);

        StackedInputLayer (const StackedInputLayer&, bool, bool);


        SHARED_MAT activate(GRAPH&, const std::vector<SHARED_MAT>&) const;
        SHARED_MAT activate(GRAPH&, SHARED_MAT) const;

        SHARED_MAT activate(GRAPH&, SHARED_MAT, const std::vector<SHARED_MAT>&) const;

        StackedInputLayer<T> shallow_copy() const;
};

template<typename T>
class RNN : public AbstractLayer<T> {
    /*
    Combine the input of a hidden vector and an input vector
    into a single matrix product sum:

        > y = A * [x, h] + b

    */
    void create_variables();
    public:
        typedef T value_t;
        SHARED_MAT Wx;
        SHARED_MAT Wh;
        SHARED_MAT b;
        const int hidden_size;
        const int input_size;
        const int output_size;
        virtual std::vector<SHARED_MAT> parameters() const;
        /**
        By default the RNN constructor sets the output size
        equal to the hidden size, creating the recurrence
        relation.
        **/
        RNN (int input_size, int hidden_size);
        /**
        An RNN can also output a different size output that
        its inputs and hiddens, e.g. in the case of a gate.
        **/
        RNN (int input_size, int hidden_size, int output_size);

        RNN (const RNN&, bool, bool);
        SHARED_MAT activate(GRAPH& G, SHARED_MAT input_vector, SHARED_MAT prev_hidden) const;

        RNN<T> shallow_copy() const;
};

template<typename T>
class DelayedRNN : public AbstractLayer<T> {
    /*
        h' = A * [h,x] + b
        o  = B * [h,x] + d
    */
    public:
        RNN<T> hidden_rnn;
        RNN<T> output_rnn;
        virtual std::vector<SHARED_MAT> parameters() const;

        DelayedRNN (int input_size, int hidden_size, int output_size);
        DelayedRNN (const DelayedRNN&, bool, bool);

        SHARED_MAT initial_states() const;

        // output (next_hidden, output)
        std::pair<SHARED_MAT, SHARED_MAT> activate(GRAPH& G, SHARED_MAT input_vector, SHARED_MAT prev_hidden) const;
        DelayedRNN<T> shallow_copy() const;
};

template<typename T>
class ShortcutRNN : public AbstractLayer<T> {
    /*
    Combine the input of a hidden vector, an input vector, and
    a second input vector (a shortcut) into a single matrix
    product sum, and also take an input from another layer as
    a "shortcut", s:

        > y = A * [x, s, h] + b

    */
    void create_variables();
    public:
        typedef T value_t;
        SHARED_MAT Wx;
        SHARED_MAT Wh;
        SHARED_MAT Ws;
        SHARED_MAT b;
        const int hidden_size;
        const int input_size;
        const int shortcut_size;
        const int output_size;
        virtual std::vector<SHARED_MAT> parameters() const;
        ShortcutRNN (int, int, int);
        ShortcutRNN (int, int, int, int);

        ShortcutRNN (const ShortcutRNN&, bool, bool);
        SHARED_MAT activate(GRAPH&, SHARED_MAT, SHARED_MAT, SHARED_MAT) const;

        ShortcutRNN<T> shallow_copy() const;
};

template<typename T>
class GatedInput : public RNN<T> {
    public:
        GatedInput (int, int);
        GatedInput (const GatedInput&, bool, bool);
        GatedInput<T> shallow_copy() const;
};

template<typename T>
class LSTM : public AbstractLayer<T> {
    /*

    LSTM layer with forget, output, memory write, and input
    modulate gates, that can remember sequences for long
    periods of time.

    See `Mat`, `HiddenLayer`
    */
    typedef RNN<T>                        layer_type;
    void name_internal_layers();
    public:
        typedef T value_t;
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

        LSTM (const LSTM&, bool, bool);
        virtual std::vector<SHARED_MAT> parameters() const;
        static std::pair<std::vector<SHARED_MAT>, std::vector<SHARED_MAT>> initial_states(const std::vector<int>&);
        std::pair<SHARED_MAT, SHARED_MAT> activate(
            GRAPH&,
            SHARED_MAT,
            SHARED_MAT,
            SHARED_MAT) const;

        LSTM<T> shallow_copy() const;
};

template<typename T>
class ShortcutLSTM : public AbstractLayer<T> {
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
        typedef T value_t;
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
        ShortcutLSTM (const ShortcutLSTM&, bool, bool);
        virtual std::vector<SHARED_MAT> parameters() const;
        std::pair<SHARED_MAT, SHARED_MAT> activate(
            GRAPH&,
            SHARED_MAT,
            SHARED_MAT,
            SHARED_MAT,
            SHARED_MAT) const;
        ShortcutLSTM<T> shallow_copy() const;
};

template<typename T>
class AbstractStackedLSTM : public AbstractLayer<T> {
    public:
        typedef std::pair<std::vector<std::shared_ptr<MAT>>, std::vector<std::shared_ptr<MAT>>> state_t;

        const int input_size;
        const std::vector<int> hidden_sizes;

        AbstractStackedLSTM(const int& input_size, const std::vector<int>& hidden_sizes);
        AbstractStackedLSTM(const AbstractStackedLSTM<T>& model, bool copy_w, bool copy_dw);

        virtual state_t initial_states() const;

        virtual std::vector<SHARED_MAT> parameters() const = 0;

        virtual state_t activate(
            GRAPH& G,
            state_t previous_state,
            SHARED_MAT input_vector,
            T drop_prob = 0.0) const = 0;
        virtual state_t activate_sequence(
            GRAPH& G,
            state_t initial_state,
            const Seq<SHARED_MAT>& sequence,
            T drop_prob = 0.0) const;
};

template<typename T>
class StackedLSTM : public AbstractStackedLSTM<T> {
    public:
        typedef LSTM<T> lstm_t;
        typedef std::pair<std::vector<std::shared_ptr<MAT>>, std::vector<std::shared_ptr<MAT>>> state_t;

        std::vector<lstm_t> cells;
        virtual state_t activate(
            GRAPH& G,
            state_t previous_state,
            SHARED_MAT input_vector,
            T drop_prob = 0.0) const;
        virtual std::vector<SHARED_MAT> parameters() const;
        StackedLSTM(const int& input_size, const std::vector<int>& hidden_sizes);
        StackedLSTM(const StackedLSTM<T>& model, bool copy_w, bool copy_dw);
        StackedLSTM<T> shallow_copy() const;
};

template<typename T>
class StackedShortcutLSTM : public AbstractStackedLSTM<T> {
    public:
        typedef LSTM<T>                  lstm_t;
        typedef ShortcutLSTM<T> shortcut_lstm_t;
        typedef std::pair<std::vector<std::shared_ptr<MAT>>, std::vector<std::shared_ptr<MAT>>> state_t;

        std::vector<shortcut_lstm_t> cells;
        lstm_t base_cell;


        virtual state_t activate(
            GRAPH& G,
            state_t previous_state,
            SHARED_MAT input_vector,
            T drop_prob = 0.0) const;
        virtual std::vector<SHARED_MAT> parameters() const;
        StackedShortcutLSTM(const int& input_size, const std::vector<int>& hidden_sizes);
        StackedShortcutLSTM(const StackedShortcutLSTM<T>& model, bool copy_w, bool copy_dw);
        StackedShortcutLSTM<T> shallow_copy() const;
};

template<typename celltype>
std::vector<celltype> StackedCells(const int&, const std::vector<int>&);

/**
StackedCells specialization to StackedLSTM
------------------------------------------

Static method StackedCells helps construct several
LSTMs that will be piled up. In a ShortcutLSTM scenario
the input is provided to all layers not just the
bottommost layer, so a new construction parameter
is provided to this **special** LSTM, the "shorcut size",
e.g. the size of the second input vector coming from father
below (taking a shortcut upwards).

Inputs
------

const int& input_size : size of the input at the basest layer
const vector<int>& hidden_sizes : dimensions of hidden states
                                  at each stack level.

Outputs
-------

vector<ShortcutLSTM<T>> cells : constructed shortcutLSTMs

**/
template<typename celltype>
std::vector<celltype> StackedCells(const int&, const int&, const std::vector<int>&);

template<typename celltype>
std::vector<celltype> StackedCells(const std::vector<celltype>&, bool, bool);

template<typename T>
std::pair<std::vector<std::shared_ptr<MAT>>, std::vector<std::shared_ptr<MAT>>> forward_LSTMs(GRAPH&,
    std::shared_ptr<MAT>,
    std::pair<std::vector<std::shared_ptr<MAT>>, std::vector<std::shared_ptr<MAT>>>&,
    const std::vector<LSTM<T>>&,
    T drop_prob=0.0);

template<typename T>
std::pair<std::vector<std::shared_ptr<MAT>>, std::vector<std::shared_ptr<MAT>>> forward_LSTMs(GRAPH&,
    std::shared_ptr<MAT>,
    std::pair<std::vector<std::shared_ptr<MAT>>, std::vector<std::shared_ptr<MAT>>>&,
    const LSTM<T>&,
    const std::vector<ShortcutLSTM<T>>&,
    T drop_prob=0.0);

#endif
