#ifndef CORE_LSTM_H
#define CORE_LSTM_H

#include "dali/mat/Layers.h"

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
        int hidden_size;
        int input_size;
        bool shortcut;
        bool memory_feeds_gates;

        // In Alex Graves' slides / comments online you do not
        // backpropagate through memory cells at the gates
        // this is a boolean, so you can retrieve the true
        // gradient by setting this to true:
        bool backprop_through_gates = false;
        LSTM (int _input_size, int _hidden_size, bool _memory_feeds_gates = false);
        LSTM (int _input_size, int shortcut_size, int _hidden_size, bool _memory_feeds_gates = false);
        LSTM (const LSTM&, bool, bool);
        virtual std::vector<Mat<R>> parameters() const;
        static std::vector<State> initial_states(const std::vector<int>&);
        State activate(
            Mat<R> input_vector,
            State prev_state) const;
        State activate_shortcut(
            Mat<R> input_vector,
            Mat<R> shortcut_vector,
            State prev_state) const;

        LSTM<R> shallow_copy() const;
        State initial_states() const;

        virtual State activate_sequence(
            State initial_state,
            const std::vector<Mat<R>>& sequence) const;
    private:
        State _activate(const std::vector<Mat<R>>&, const State&) const;
};

template<typename R>
class AbstractStackedLSTM : public AbstractLayer<R> {
    public:
        typedef std::vector < typename LSTM<R>::State > state_t;

        int input_size;
        std::vector<int> hidden_sizes;
        AbstractStackedLSTM();
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
            const std::vector<Mat<R>>& sequence,
            R drop_prob = 0.0) const;
};

template<typename R>
class StackedLSTM : public AbstractStackedLSTM<R> {
    public:
        typedef LSTM<R> lstm_t;
        typedef std::vector< typename LSTM<R>::State > state_t;
        bool shortcut;
        bool memory_feeds_gates;

        std::vector<lstm_t> cells;
        virtual state_t activate(
            state_t previous_state,
            Mat<R> input_vector,
            R drop_prob = 0.0) const;
        virtual std::vector<Mat<R>> parameters() const;
        StackedLSTM();
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
