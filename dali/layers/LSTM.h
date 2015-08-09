#ifndef CORE_LSTM_H
#define CORE_LSTM_H

#include "dali/layers/Layers.h"

template<typename R>
struct LSTMState {
    Mat<R> memory;
    Mat<R> hidden;
    LSTMState() = default;
    LSTMState(Mat<R> _memory, Mat<R> _hidden);
    static std::vector<Mat<R>> hiddens (const std::vector<LSTMState<R>>&);
    static std::vector<Mat<R>> memories (const std::vector<LSTMState<R>>&);
    operator std::tuple<Mat<R> &, Mat<R> &>();
};

template<typename R>
class LSTM : public AbstractLayer<R> {
    /*
    LSTM layer with forget, output, memory write, and input
    modulate gates, that can remember sequences for long
    periods of time.

    See `Layers.h`
    */
    typedef StackedInputLayer<R> layer_type;

    public:
        void name_internal_layers();

        typedef LSTMState<R> activation_t;

        // each child's memory to write controller for memory:
        std::vector<Mat<R>> Wcells_to_inputs;
        // each child's memory to forget gate for memory:
        std::vector<Mat<R>> Wcells_to_forgets;
        // memory to output gate
        Mat<R> Wco;

        typedef R value_t;
        // cell input modulation:
        layer_type input_layer;
        // cell forget gate:
        std::vector<layer_type> forget_layers;
        // cell output modulation
        layer_type output_layer;
        // cell write params
        layer_type cell_layer;

        int hidden_size;
        std::vector<int> input_sizes;
        int num_children;

        bool memory_feeds_gates;

        // In Alex Graves' slides / comments online you do not
        // backpropagate through memory cells at the gatestype
        // this is a boolean, so you can retrieve the true
        // gradient by setting this to true:
        bool backprop_through_gates = false;

        LSTM() = default;

        // This is a regular vanilla, but awesome LSTM constructor.
        LSTM (int _input_size, int _hidden_size, bool _memory_feeds_gates = false);

        // This constructors purpose is to create a tree LSTM.
        LSTM (int _input_size, int _hidden_size, int num_children, bool _memory_feeds_gates = false);

        // This constructor is generally intended to support shortcut LSTM. It also
        // happens to be the most general constructor available.
        LSTM (std::vector<int> _input_sizes, int _hidden_size, int num_children, bool _memory_feeds_gates = false);

        LSTM (const LSTM&, bool, bool);

        virtual std::vector<Mat<R>> parameters() const;

        activation_t activate(
            Mat<R> input_vector,
            activation_t previous_state) const;

        activation_t activate(
            Mat<R> input_vector,
            std::vector<activation_t> previous_children_states) const;

        activation_t activate(const std::vector<Mat<R>>&, const std::vector<activation_t>&) const;

        activation_t activate_shortcut(
            Mat<R> input_vector,
            Mat<R> shortcut_vector,
            activation_t prev_state) const;

        LSTM<R> shallow_copy() const;

        activation_t initial_states() const;

        virtual activation_t activate_sequence(
            activation_t initial_state,
            const std::vector<Mat<R>>& sequence) const;
};

template<typename R>
class AbstractStackedLSTM : public AbstractLayer<R> {
    public:
        typedef std::vector < typename LSTM<R>::activation_t > state_t;

        std::vector<int> input_sizes;
        std::vector<int> hidden_sizes;
        AbstractStackedLSTM();
        AbstractStackedLSTM(const int& input_size, const std::vector<int>& hidden_sizes);
        AbstractStackedLSTM(const std::vector<int>& input_sizes, const std::vector<int>& hidden_sizes);
        AbstractStackedLSTM(const AbstractStackedLSTM<R>& model, bool copy_w, bool copy_dw);

        virtual state_t initial_states() const = 0;

        virtual std::vector<Mat<R>> parameters() const = 0;

        virtual state_t activate(
            state_t previous_state,
            Mat<R> input_vector,
            R drop_prob = 0.0) const = 0;
        virtual state_t activate_sequence(
            state_t initial_state,
            const std::vector<Mat<R>>& sequence,
            R drop_prob = 0.0) const;
};

template<typename R>
class StackedLSTM : public AbstractStackedLSTM<R> {
    public:
        typedef LSTM<R> lstm_t;
        typedef std::vector< typename LSTM<R>::activation_t > state_t;
        bool shortcut;
        bool memory_feeds_gates;

        virtual state_t initial_states() const;

        std::vector<lstm_t> cells;
        virtual state_t activate(
            state_t previous_state,
            Mat<R> input_vector,
            R drop_prob = 0.0) const;
        virtual state_t activate(
            state_t previous_state,
            const std::vector<Mat<R>>& inputs,
            R drop_prob = 0.0) const;
        virtual std::vector<Mat<R>> parameters() const;
        StackedLSTM();
        StackedLSTM(
            const int& input_size,
            const std::vector<int>& hidden_sizes,
            bool _shortcut,
            bool _memory_feeds_gates);
        StackedLSTM(
            const std::vector<int>& input_sizes,
            const std::vector<int>& hidden_sizes,
            bool _shortcut,
            bool _memory_feeds_gates);
        StackedLSTM(const StackedLSTM<R>& model, bool copy_w, bool copy_dw);
        StackedLSTM<R> shallow_copy() const;
};

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
std::vector<celltype> StackedCells(
    const int& input_size,
    const std::vector<int>& hidden_sizes,
    bool shortcut,
    bool memory_feeds_gates);

template<typename celltype>
std::vector<celltype> StackedCells(
    const std::vector<int>& input_sizes,
    const std::vector<int>& hidden_sizes,
    bool shortcut,
    bool memory_feeds_gates);

template<typename celltype>
std::vector<celltype> StackedCells(const std::vector<celltype>&, bool, bool);

template<typename R>
std::vector< typename LSTM<R>::activation_t > forward_LSTMs(
    Mat<R> input,
    std::vector< typename LSTM<R>::activation_t >&,
    const std::vector<LSTM<R>>&,
    R drop_prob=0.0);

template<typename R>
std::vector< typename LSTM<R>::activation_t > forward_LSTMs(
    const std::vector<Mat<R>>& inputs,
    std::vector< typename LSTM<R>::activation_t >&,
    const std::vector<LSTM<R>>&,
    R drop_prob=0.0);

template<typename R>
std::vector< typename LSTM<R>::activation_t > shortcut_forward_LSTMs(
    Mat<R> input,
    std::vector< typename LSTM<R>::activation_t >&,
    const std::vector<LSTM<R>>&,
    R drop_prob=0.0);

template<typename R>
std::vector< typename LSTM<R>::activation_t > shortcut_forward_LSTMs(
    const std::vector<Mat<R>>& inputs,
    std::vector< typename LSTM<R>::activation_t >&,
    const std::vector<LSTM<R>>&,
    R drop_prob=0.0);

#endif
