#include "LSTM.h"
using std::vector;

template<typename R>
LSTM<R>::LSTM (int _input_size, int _hidden_size, bool _memory_feeds_gates) :
        shortcut(false),
        memory_feeds_gates(_memory_feeds_gates),
        hidden_size(_hidden_size),
        input_size(_input_size),
        input_layer({_input_size, _hidden_size}, _hidden_size),
        forget_layer({_input_size, _hidden_size}, _hidden_size),
        output_layer({_input_size, _hidden_size}, _hidden_size),
        cell_layer({_input_size, _hidden_size}, _hidden_size) {

    if (memory_feeds_gates) {
        Wci = Mat<R>(hidden_size, 1, weights<R>::uniform(2. / sqrt(hidden_size)));
        Wco = Mat<R>(hidden_size, 1, weights<R>::uniform(2. / sqrt(hidden_size)));
        Wcf = Mat<R>(hidden_size, 1, weights<R>::uniform(2. / sqrt(hidden_size)));
    }
    // Note: Ilya Sutskever recommends initializing with
    // forget gate at high value
    // http://yyue.blogspot.fr/2015/01/a-brief-overview-of-deep-learning.html
    // forget_layer.b.w().array() += 2;
    name_internal_layers();
}

template<typename R>
LSTM<R>::LSTM (int _input_size, int shortcut_size, int _hidden_size, bool _memory_feeds_gates) :
        shortcut(true),
        memory_feeds_gates(_memory_feeds_gates),
        hidden_size(_hidden_size),
        input_size(_input_size),
        input_layer( {_input_size, _hidden_size, shortcut_size}, _hidden_size),
        forget_layer({_input_size, _hidden_size, shortcut_size}, _hidden_size),
        output_layer({_input_size, _hidden_size, shortcut_size}, _hidden_size),
        cell_layer(  {_input_size, _hidden_size, shortcut_size}, _hidden_size) {

    if (memory_feeds_gates) {
        Wci = Mat<R>(hidden_size, 1, weights<R>::uniform(2. / sqrt(hidden_size)));
        Wco = Mat<R>(hidden_size, 1, weights<R>::uniform(2. / sqrt(hidden_size)));
        Wcf = Mat<R>(hidden_size, 1, weights<R>::uniform(2. / sqrt(hidden_size)));
    }
    // Note: Ilya Sutskever recommends initializing with
    // forget gate at high value
    // http://yyue.blogspot.fr/2015/01/a-brief-overview-of-deep-learning.html
    // forget_layer.b.w().array() += 2;
    name_internal_layers();
}

template<typename R>
LSTM<R>::LSTM (const LSTM<R>& lstm, bool copy_w, bool copy_dw) :
        shortcut(lstm.shortcut),
        Wci(lstm.Wci, copy_w, copy_dw),
        Wcf(lstm.Wcf, copy_w, copy_dw),
        Wco(lstm.Wco, copy_w, copy_dw),
        memory_feeds_gates(lstm.memory_feeds_gates),
        hidden_size(lstm.hidden_size),
        input_size(lstm.input_size),
        input_layer(lstm.input_layer, copy_w, copy_dw),
        forget_layer(lstm.forget_layer, copy_w, copy_dw),
        output_layer(lstm.output_layer, copy_w, copy_dw),
        cell_layer(lstm.cell_layer, copy_w, copy_dw) {
    name_internal_layers();
}

template<typename R>
LSTM<R> LSTM<R>::shallow_copy() const {
    return LSTM<R>(*this, false, true);
}


template<typename R>
void LSTM<R>::name_internal_layers() {
}

template<typename R>
LSTM<R>::State::State(Mat<R> _memory, Mat<R> _hidden) : memory(_memory), hidden(_hidden) {}

template<typename R>
LSTM<R>::State::operator std::tuple<Mat<R> &, Mat<R> &>() {
    return std::tuple<Mat<R>&, Mat<R>&>(memory, hidden);
}

template<typename R>
vector<Mat<R>> LSTM<R>::State::hiddens( const vector< typename LSTM<R>::State>& states) {
    vector<Mat<R>> hiddens;
    std::transform(
        states.begin(),
        states.end(),
        std::back_inserter(hiddens), [](const State& s) {
            return s.hidden;
        }
    );
    return hiddens;
}

template<typename R>
vector<Mat<R>> LSTM<R>::State::memories( const vector<typename LSTM<R>::State>& states) {
    vector<Mat<R>> memories;
    std::transform(
        states.begin(),
        states.end(),
        std::back_inserter(memories), [](const State& s) {
            return s.memory;
        }
    );
    return memories;
}

template<typename R>
typename LSTM<R>::State LSTM<R>::_activate(
    const std::vector<Mat<R>>& gate_input,
    const State& initial_state) const {

    Mat<R> input_gate, forget_gate, output_gate;

    assert(initial_state.memory.dims(0) == hidden_size);
    assert(initial_state.hidden.dims(0) == hidden_size);
    assert(gate_input[0].dims(0) == input_size);
    if (shortcut) {
        assert2(gate_input.size() == 3,
            utils::MS() << "Expected 3 inputs, but got " << gate_input.size() << " instead.");
        assert2(gate_input[2].dims(0) == input_layer.input_sizes()[2],
            utils::MS() << "Gate inputs don't match: gate expected a shortcut input of size "
                 << input_layer.input_sizes()[2]
                 << " but got " << gate_input[2].dims(0) << " instead.");
    } else {
        assert(gate_input.size() == 2);
    }

    if (memory_feeds_gates) {
        auto constant_memory = MatOps<R>::consider_constant_if(initial_state.memory, !backprop_through_gates);
        // if the memory feeds the gates (Alex Graves 2013) then
        // a diagonal matrix (Wci and Wcf) connect memory to input
        // and forget gates

        // input gate:
        input_gate  = (
            input_layer.activate(gate_input) + (constant_memory * Wci)
        ).sigmoid();
        // forget gate
        forget_gate = (
            forget_layer.activate(gate_input) + (constant_memory * Wcf)
        ).sigmoid();
    } else {
        // (Zaremba 2014 style)

        // input gate:
        input_gate  = input_layer.activate(gate_input).sigmoid();
        // forget gate
        forget_gate = forget_layer.activate(gate_input).sigmoid();
    }
    // write operation on cells
    auto cell_write  = cell_layer.activate(gate_input).tanh();

    // compute new cell activation
    auto retain_cell = forget_gate * initial_state.memory; // what do we keep from cell
    auto write_cell  = input_gate  * cell_write; // what do we write to cell
    auto cell_d      = retain_cell + write_cell; // new cell contents

    if (memory_feeds_gates) {
        // output gate uses new memory (cell_d) to control its gate
        output_gate = (
            output_layer.activate(gate_input) + (MatOps<R>::consider_constant_if(cell_d, !backprop_through_gates) * Wco)
        ).sigmoid();
    } else {
        // output gate
        output_gate = output_layer.activate(gate_input).sigmoid();
    }

    // compute hidden state as gated, saturated cell activations
    auto hidden_d = output_gate * cell_d.tanh();

    DEBUG_ASSERT_NOT_NAN(hidden_d.w());
    DEBUG_ASSERT_NOT_NAN(cell_d.w());

    return State(cell_d, hidden_d);
}

template<typename R>
typename LSTM<R>::State LSTM<R>::activate(
    Mat<R> input_vector,
    State  state) const {
    auto gate_input = std::vector<Mat<R>>({input_vector, state.hidden});
    return _activate(gate_input, state);
}

template<typename R>
typename LSTM<R>::State LSTM<R>::activate_shortcut(
    Mat<R> input_vector,
    Mat<R> shortcut_vector,
    State  state) const {
    utils::assert2(shortcut, "Error: LSTM without Shortcuts received shortcut_vector.");
    auto gate_input = std::vector<Mat<R>>({input_vector, state.hidden, shortcut_vector});
    return _activate(gate_input, state);
}

template<typename R>
typename LSTM<R>::State LSTM<R>::activate_sequence(
    State state,
    const vector<Mat<R>>& sequence) const {
    for (auto& input_vector : sequence)
        state = activate(
            input_vector,
            state
        );
    return state;
};

template<typename R>
std::vector<Mat<R>> LSTM<R>::parameters() const {
    std::vector<Mat<R>> parameters;

    if (memory_feeds_gates) {
        parameters.emplace_back( Wci);
        parameters.emplace_back( Wcf);
        parameters.emplace_back( Wco);
    }

    auto input_layer_params  = input_layer.parameters();
    auto forget_layer_params = forget_layer.parameters();
    auto output_layer_params = output_layer.parameters();
    auto cell_layer_params   = cell_layer.parameters();

    parameters.insert( parameters.end(), input_layer_params.begin(),  input_layer_params.end() );
    parameters.insert( parameters.end(), forget_layer_params.begin(), forget_layer_params.end() );
    parameters.insert( parameters.end(), output_layer_params.begin(), output_layer_params.end() );
    parameters.insert( parameters.end(), cell_layer_params.begin(),   cell_layer_params.end() );

    return parameters;
}

template<typename R>
typename LSTM<R>::State LSTM<R>::initial_states() const {
    return LSTM<R>::State(Mat<R>(hidden_size, 1), Mat<R>(hidden_size, 1));
}

template<typename R>
typename std::vector< typename LSTM<R>::State > LSTM<R>::initial_states(
        const std::vector<int>& hidden_sizes) {
    std::vector< typename LSTM<R>::State >  initial_state;
    initial_state.reserve(hidden_sizes.size());
    for (auto& size : hidden_sizes) {
        initial_state.emplace_back(Mat<R>(size, 1), Mat<R>(size, 1));
    }
    return initial_state;
}

// TODO: make this a static method of StackedLSTM
// since this class is only customer to stackedcells
template<typename celltype>
vector<celltype> StackedCells(
    const int& input_size,
    const vector<int>& hidden_sizes,
    bool shortcut,
    bool memory_feeds_gates) {
    vector<celltype> cells;
    cells.reserve(hidden_sizes.size());
    int prev_size = input_size;
    int i = 0;
    for (auto& hidden_size : hidden_sizes) {
        if (shortcut) {
            if (i == 0) {
                // first cell in a shorcut
                // stack cannot "steal" or
                // shorcut from anywhere else
                // so no shorcut is used
                cells.emplace_back(
                    prev_size,
                    hidden_size,
                    memory_feeds_gates);
            } else {
                // other cells in a shorcut
                // stack steal the input
                // from the lower stack
                // input_size
                cells.emplace_back(
                    prev_size,
                    input_size,
                    hidden_size,
                    memory_feeds_gates);
            }
        } else {
            cells.emplace_back(
                prev_size,
                hidden_size,
                memory_feeds_gates);
        }
        prev_size = hidden_size;
        i++;
    }
    return cells;
}

template<typename celltype>
vector<celltype> StackedCells(const vector<celltype>& source_cells,
                              bool copy_w,
                              bool copy_dw) {
    vector<celltype> cells;
    cells.reserve(source_cells.size());
    for (const auto& cell : source_cells)
        cells.emplace_back(cell, copy_w, copy_dw);
    return cells;
}

template<typename R>
std::vector< typename LSTM<R>::State > forward_LSTMs(
    Mat<R> base_input,
    std::vector< typename LSTM<R>::State >& previous_state,
    const vector<LSTM<R>>& cells,
    R drop_prob) {

    std::vector< typename LSTM<R>::State> out_state;
    out_state.reserve(cells.size());

    auto layer_input = base_input;
    auto state_iter = previous_state.begin();

    for (auto& layer : cells) {
        if (layer.shortcut) {
            out_state.emplace_back(
                layer.activate_shortcut(
                    MatOps<R>::dropout_normalized(layer_input, drop_prob),
                    MatOps<R>::dropout_normalized(base_input, drop_prob),
                    *state_iter
                )
            );
        } else {
            out_state.emplace_back(
                layer.activate(
                    MatOps<R>::dropout_normalized(layer_input, drop_prob),
                    *state_iter
                )
            );
        }
        ++state_iter;
        layer_input = out_state.back().hidden;
    }
    return out_state;
}

template class LSTM<float>;
template class LSTM<double>;

template<typename R>
AbstractStackedLSTM<R>::AbstractStackedLSTM() : input_size(0) {
}

template<typename R>
AbstractStackedLSTM<R>::AbstractStackedLSTM(const int& input_size, const std::vector<int>& hidden_sizes) :
        input_size(input_size),
        hidden_sizes(hidden_sizes) {
}

template<typename R>
AbstractStackedLSTM<R>::AbstractStackedLSTM(const AbstractStackedLSTM<R>& model, bool copy_w, bool copy_dw) :
    input_size(model.input_size),
    hidden_sizes(model.hidden_sizes) {
}

template<typename R>
std::vector<typename LSTM<R>::State> AbstractStackedLSTM<R>::initial_states() const {
    return LSTM<R>::initial_states(hidden_sizes);
}

template<typename R>
typename AbstractStackedLSTM<R>::state_t AbstractStackedLSTM<R>::activate_sequence(
    state_t initial_state,
    const vector<Mat<R>>& sequence,
    R drop_prob) const {
    for (auto& input_vector : sequence)
        initial_state = activate(initial_state, input_vector, drop_prob);
    return initial_state;
};

template<typename R>
StackedLSTM<R>::StackedLSTM(
    const int& input_size,
    const std::vector<int>& hidden_sizes,
    bool _shortcut,
    bool _memory_feeds_gates) : shortcut(_shortcut), memory_feeds_gates(_memory_feeds_gates),
        AbstractStackedLSTM<R>(input_size, hidden_sizes) {
    cells = StackedCells<lstm_t>(input_size, hidden_sizes, shortcut, memory_feeds_gates);
};

template<typename R>
StackedLSTM<R>::StackedLSTM() : AbstractStackedLSTM<R>(),
                                shortcut(false),
                                memory_feeds_gates(false) {
}

template<typename R>
StackedLSTM<R>::StackedLSTM(const StackedLSTM<R>& model, bool copy_w, bool copy_dw) :
         shortcut(model.shortcut), memory_feeds_gates(model.memory_feeds_gates),
        AbstractStackedLSTM<R>(model, copy_w, copy_dw) {
    cells = StackedCells<lstm_t>(model.cells, copy_w, copy_dw);
};

template<typename R>
StackedLSTM<R> StackedLSTM<R>::shallow_copy() const {
    return StackedLSTM<R>(*this, false, true);
}

template<typename R>
std::vector<Mat<R>> StackedLSTM<R>::parameters() const {
    vector<Mat<R>> parameters;
    for (auto& cell : cells) {
        auto cell_params = cell.parameters();
        parameters.insert(parameters.end(), cell_params.begin(), cell_params.end());
    }
    return parameters;
}

template<typename R>
typename StackedLSTM<R>::state_t StackedLSTM<R>::activate(
            state_t previous_state,
            Mat<R> input_vector,
            R drop_prob) const {
    return forward_LSTMs(input_vector, previous_state, cells, drop_prob);
};

template class StackedLSTM<float>;
template class StackedLSTM<double>;
