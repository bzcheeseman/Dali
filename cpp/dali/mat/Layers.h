#ifndef CORE_LAYERS_H
#define CORE_LAYERS_H

#include <initializer_list>

#include "dali/mat/Mat.h"
#include "dali/mat/MatOps.h"
#include "dali/mat/Seq.h"

template<typename R>
class AbstractLayer {
    public:
        virtual std::vector<Mat<R>> parameters() const = 0;
};

template<typename R>
class AbstractMultiInputLayer : public AbstractLayer<R> {
    public:
        Mat<R> b;
        virtual Mat<R> activate(Mat<R>) const = 0;
        virtual Mat<R> activate(const std::vector<Mat<R>>&) const;
        virtual Mat<R> activate(Mat<R>, const std::vector<Mat<R>>&) const;
};

template<typename R>
class Layer : public AbstractMultiInputLayer<R> {
    /*
    Linear output layer of the form with affine offset bias
    vector b (broadcasted):

        > y = A * x + b

    */
    void create_variables();
    public:
        typedef R value_t;
        Mat<R> W;
        const int hidden_size;
        const int input_size;
        virtual std::vector<Mat<R>> parameters() const;
        Layer (int, int);

        Layer (const Layer&, bool, bool);
        Mat<R> activate(Mat<R>) const;

        Layer<R> shallow_copy() const;
};

template<typename R>
class StackedInputLayer : public AbstractMultiInputLayer<R> {
    /*
    Linear output layer of the form with affine offset bias
    vector b (broadcasted), taking as inputs many different
    vectors of disparate sizes:

        > y = [A_1, ..., A_n]^T * [x_1, ..., x_n] + b

    */
    void create_variables();
    std::vector<Mat<R>> zip_inputs_with_matrices_and_bias(const std::vector<Mat<R>>&) const;
    std::vector<Mat<R>> zip_inputs_with_matrices_and_bias(Mat<R>, const std::vector<Mat<R>>&) const;
    public:
        typedef R value_t;
        std::vector<Mat<R>> matrices;
        const int hidden_size;
        const std::vector<int> input_sizes;
        virtual std::vector<Mat<R>> parameters() const;
        StackedInputLayer (std::initializer_list<int>, int);
        StackedInputLayer (std::vector<int>, int);
        StackedInputLayer (int input_size, int output_size);
        StackedInputLayer (const StackedInputLayer&, bool, bool);

        Mat<R> activate(const std::vector<Mat<R>>&) const;
        Mat<R> activate(Mat<R>) const;
        Mat<R> activate(Mat<R>, const std::vector<Mat<R>>&) const;

        StackedInputLayer<R> shallow_copy() const;
};

template<typename R>
class SecondOrderCombinator : public AbstractLayer<R> {
    /* Implements the following expression

           y = W1 * i1 + W2 * i2 + b

       The purpose is to capture only a few relevant terms from:

           y = i1 * W * i2' + B

        without having to evaluate big matrix equation.
    */
    public:
        const int input1_size;
        const int input2_size;
        const int output_size;

        Mat<R> W1;
        Mat<R> W2;
        Mat<R> b;

        SecondOrderCombinator(int input1_size, int input2_size, int output_size);

        SecondOrderCombinator(const SecondOrderCombinator& m, bool copy_w, bool copy_dw);

        virtual std::vector<Mat<R>> parameters() const;

        Mat<R> activate(Mat<R> i1, Mat<R> i2) const;
};

template<typename R>
class RNN : public AbstractLayer<R> {
    /*
    Combine the input of a hidden vector and an input vector
    into a single matrix product sum:

        > y = A * [x, h] + b

    */
    void create_variables();
    public:
        typedef R value_t;
        Mat<R> Wx;
        Mat<R> Wh;
        Mat<R> b;
        const int hidden_size;
        const int input_size;
        const int output_size;
        virtual std::vector<Mat<R>> parameters() const;
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
        Mat<R> activate(Mat<R> input_vector, Mat<R> prev_hidden) const;

        RNN<R> shallow_copy() const;
};

template<typename R>
class DelayedRNN : public AbstractLayer<R> {
    /*
        h' = A * [h,x] + b
        o  = B * [h,x] + d
    */
    public:
        RNN<R> hidden_rnn;
        RNN<R> output_rnn;
        virtual std::vector<Mat<R>> parameters() const;

        DelayedRNN (int input_size, int hidden_size, int output_size);
        DelayedRNN (const DelayedRNN&, bool, bool);

        Mat<R> initial_states() const;

        // output (next_hidden, output)
        std::tuple<Mat<R>,Mat<R>> activate(Mat<R> input_vector, Mat<R> prev_hidden) const;
        DelayedRNN<R> shallow_copy() const;
};

template<typename R>
class GatedInput : public RNN<R> {
    public:
        GatedInput (int, int);
        GatedInput (const GatedInput&, bool, bool);
        GatedInput<R> shallow_copy() const;
};

template<typename R>
class LSTM : public AbstractLayer<R> {
    /*

    LSTM layer with forget, output, memory write, and input
    modulate gates, that can remember sequences for long
    periods of time.

    See `Mat`, `HiddenLayer`
    */
    typedef StackedInputLayer<R> layer_type;
    void name_internal_layers();

    public:
        struct State {
            Mat<R> memory;
            Mat<R> hidden;
            State(Mat<R> _memory, Mat<R> _hidden);
            static std::vector<Mat<R>> hiddens (const std::vector<State>&);
            static std::vector<Mat<R>> memories (const std::vector<State>&);
            operator std::tuple<Mat<R> &, Mat<R> &>();
        };

        Mat<R> Wci;
        Mat<R> Wcf;
        Mat<R> Wco;

        typedef R value_t;
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
        const bool shortcut;
        const bool memory_feeds_gates;
        LSTM (int _input_size, int _hidden_size, bool _memory_feeds_gates = false);
        LSTM (int _input_size, int shortcut_size, int _hidden_size, bool _memory_feeds_gates = false);
        LSTM (const LSTM&, bool, bool);
        virtual std::vector<Mat<R>> parameters() const;
        static std::vector<State> initial_states(const std::vector<int>&);
        State activate(
            Mat<R> input_vector,
            State prev_state) const;
        State activate(
            Mat<R> input_vector,
            Mat<R> shortcut_vector,
            State prev_state) const;

        LSTM<R> shallow_copy() const;
        State initial_states() const;

        virtual State activate_sequence(
            State initial_state,
            const Seq<Mat<R>>& sequence) const;
    private:
        State _activate(const std::vector<Mat<R>>&, const State&) const;
};

template<typename R>
class AbstractStackedLSTM : public AbstractLayer<R> {
    public:
        typedef std::vector < typename LSTM<R>::State > state_t;

        const int input_size;
        const std::vector<int> hidden_sizes;

        AbstractStackedLSTM(const int& input_size, const std::vector<int>& hidden_sizes);
        AbstractStackedLSTM(const AbstractStackedLSTM<R>& model, bool copy_w, bool copy_dw);

        virtual state_t initial_states() const;

        virtual std::vector<Mat<R>> parameters() const = 0;

        virtual state_t activate(
            state_t previous_state,
            Mat<R> input_vector,
            R drop_prob = 0.0) const = 0;
        virtual state_t activate_sequence(
            state_t initial_state,
            const Seq<Mat<R>>& sequence,
            R drop_prob = 0.0) const;
};

template<typename R>
class StackedLSTM : public AbstractStackedLSTM<R> {
    public:
        typedef LSTM<R> lstm_t;
        typedef std::vector< typename LSTM<R>::State > state_t;
        const bool shortcut;
        const bool memory_feeds_gates;

        std::vector<lstm_t> cells;
        virtual state_t activate(
            state_t previous_state,
            Mat<R> input_vector,
            R drop_prob = 0.0) const;
        virtual std::vector<Mat<R>> parameters() const;
        StackedLSTM(
            const int& input_size,
            const std::vector<int>& hidden_sizes,
            bool _shortcut,
            bool _memory_feeds_gates);
        StackedLSTM(const StackedLSTM<R>& model, bool copy_w, bool copy_dw);
        StackedLSTM<R> shallow_copy() const;
};

template<typename celltype>
std::vector<celltype> StackedCells(
    const int&,
    const std::vector<int>&,
    bool shortcut,
    bool memory_feeds_gates);

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

vector<ShortcutLSTM<R>> cells : constructed shortcutLSTMs

**/
template<typename celltype>
std::vector<celltype> StackedCells(const int&, const int&, const std::vector<int>&);

template<typename celltype>
std::vector<celltype> StackedCells(const std::vector<celltype>&, bool, bool);

template<typename R>
std::vector< typename LSTM<R>::State > forward_LSTMs(
    Mat<R>,
    std::vector< typename LSTM<R>::State >&,
    const std::vector<LSTM<R>>&,
    R drop_prob=0.0);

#endif
