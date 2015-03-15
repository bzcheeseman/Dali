#include "Layers.h"

using std::make_shared;
using std::vector;
using std::make_tuple;
using std::get;
using std::vector;
using std::shared_ptr;

typedef std::pair<int,int> PII;

template<typename R>
Mat<R> AbstractMultiInputLayer<R>::activate(const std::vector<Mat<R>>& inputs) const {
    assert(inputs.size() > 0);
    return activate(inputs[0]);
};

template<typename R>
Mat<R> AbstractMultiInputLayer<R>::activate(Mat<R> first_input, const std::vector<Mat<R>>& inputs) const {
    if (inputs.size() > 0) {
        return activate(inputs.back());
    } else {
        return activate(first_input);
    }
};

template<typename R>
void Layer<R>::create_variables() {
    W = Mat<R>(hidden_size, input_size, weights<R>::uniform(1.0 / sqrt(input_size)));
    this->b = Mat<R>(hidden_size, 1, weights<R>::uniform(1.0 / sqrt(input_size)));
}

template<typename R>
Layer<R>::Layer (int _input_size, int _hidden_size) : hidden_size(_hidden_size), input_size(_input_size) {
    create_variables();
}

template<typename R>
Mat<R> Layer<R>::activate(Mat<R> input_vector) const {
    return MatOps<R>::mul_with_bias(W, input_vector, this->b);
}

template<typename R>
Layer<R>::Layer (const Layer<R>& layer, bool copy_w, bool copy_dw) : hidden_size(layer.hidden_size), input_size(layer.input_size) {
    W = Mat<R>(layer.W, copy_w, copy_dw);
    this->b = Mat<R>(layer.b, copy_w, copy_dw);
}

template<typename R>
Layer<R> Layer<R>::shallow_copy() const {
    return Layer<R>(*this, false, true);
}

template<typename R>
std::vector<Mat<R>> Layer<R>::parameters() const{
    return std::vector<Mat<R>>({W, this->b});
}

// StackedInputLayer:
template<typename R>
void StackedInputLayer<R>::create_variables() {
    int total_input_size = 0;
    for (auto& input_size : input_sizes) total_input_size += input_size;
    matrices.reserve(input_sizes.size());
    for (auto& input_size : input_sizes) {
        matrices.emplace_back(hidden_size, input_size,
                weights<R>::uniform(1. / sqrt(total_input_size)));
        DEBUG_ASSERT_MAT_NOT_NAN(matrices[matrices.size() -1])
    }
    this->b = Mat<R>(hidden_size, 1, weights<R>::uniform(1.0 / sqrt(total_input_size)));
}
template<typename R>
StackedInputLayer<R>::StackedInputLayer (vector<int> _input_sizes,
                                         int _hidden_size) :
        hidden_size(_hidden_size),
        input_sizes(_input_sizes) {
    create_variables();
}

template<typename R>
StackedInputLayer<R>::StackedInputLayer (int input_size,
                                         int output_size) :
        hidden_size(output_size),
        input_sizes({input_size}) {
    create_variables();
}

template<typename R>
StackedInputLayer<R>::StackedInputLayer (std::initializer_list<int> _input_sizes,
                                         int _hidden_size) :
        hidden_size(_hidden_size),
        input_sizes(_input_sizes) {
    create_variables();
}

template<typename R>
vector<Mat<R>> StackedInputLayer<R>::zip_inputs_with_matrices_and_bias(const vector<Mat<R>>& inputs) const {
    vector<Mat<R>> zipped;
    zipped.reserve(matrices.size() * 2 + 1);
    auto input_ptr = inputs.begin();
    auto mat_ptr = matrices.begin();
    while (mat_ptr != matrices.end()) {
        zipped.emplace_back(*mat_ptr++);
        zipped.emplace_back(*input_ptr++);
    }
    zipped.emplace_back(this->b);
    return zipped;
}

template<typename R>
vector<Mat<R>> StackedInputLayer<R>::zip_inputs_with_matrices_and_bias(
        Mat<R> input,
        const vector<Mat<R>>& inputs) const {
    vector<Mat<R>> zipped;
    zipped.reserve(matrices.size() * 2 + 1);
    auto input_ptr = inputs.begin();
    auto mat_ptr = matrices.begin();

    // We are provided separately with anoter input vector
    // that will go first in the zip, while the remainder will
    // be loaded in "zip" form with the vector of inputs
    zipped.emplace_back(*mat_ptr);
    zipped.emplace_back(input);

    mat_ptr++;

    DEBUG_ASSERT_MAT_NOT_NAN((*mat_ptr))
    DEBUG_ASSERT_MAT_NOT_NAN(input)

    while (mat_ptr != matrices.end()) {

        DEBUG_ASSERT_MAT_NOT_NAN((*mat_ptr))
        DEBUG_ASSERT_MAT_NOT_NAN((*input_ptr))

        zipped.emplace_back(*mat_ptr);
        zipped.emplace_back(*input_ptr);
        mat_ptr++;
        input_ptr++;
    }
    zipped.emplace_back(this->b);
    return zipped;
}

template<typename R>
Mat<R> StackedInputLayer<R>::activate(
    const vector<Mat<R>>& inputs) const {
    auto zipped = zip_inputs_with_matrices_and_bias(inputs);
    return MatOps<R>::mul_add_mul_with_bias(zipped);
}

template<typename R>
Mat<R> StackedInputLayer<R>::activate(
    Mat<R> input_vector) const {
    if (matrices.size() == 0) {
        return MatOps<R>::mul_with_bias(matrices[0], input_vector, this->b);
    } else {
        throw std::runtime_error("Error: Stacked Input Layer parametrized with more than 1 inputs only received 1 input vector.");
    }
}

template<typename R>
Mat<R> StackedInputLayer<R>::activate(
    Mat<R> input,
    const vector<Mat<R>>& inputs) const {
    DEBUG_ASSERT_MAT_NOT_NAN(input)
    auto zipped = zip_inputs_with_matrices_and_bias(input, inputs);

    auto out = MatOps<R>::mul_add_mul_with_bias(zipped);

    DEBUG_ASSERT_MAT_NOT_NAN(out)

    return out;
}

template<typename R>
StackedInputLayer<R>::StackedInputLayer (const StackedInputLayer<R>& layer, bool copy_w, bool copy_dw) : hidden_size(layer.hidden_size), input_sizes(layer.input_sizes) {
    matrices.reserve(layer.matrices.size());
    for (auto& matrix : layer.matrices)
        matrices.emplace_back(matrix, copy_w, copy_dw);
    this->b = Mat<R>(layer.b, copy_w, copy_dw);
}

template<typename R>
StackedInputLayer<R> StackedInputLayer<R>::shallow_copy() const {
    return StackedInputLayer<R>(*this, false, true);
}

template<typename R>
std::vector<Mat<R>> StackedInputLayer<R>::parameters() const{
    auto params = vector<Mat<R>>(matrices);
    params.emplace_back(this->b);
    return params;
}

template<typename R>
void RNN<R>::create_variables() {
    Wx = Mat<R>(output_size, input_size,  weights<R>::uniform(1. / sqrt(input_size)));

    Wh = Mat<R>(output_size, hidden_size, weights<R>::uniform(1. / sqrt(hidden_size)));
    b  = Mat<R>(output_size, 1, weights<R>::uniform(1. / sqrt(hidden_size)));
}

template<typename R>
std::vector<Mat<R>> RNN<R>::parameters() const {
    return std::vector<Mat<R>>({Wx, Wh, b});
}

/* DelayedRNN */

template<typename R>
DelayedRNN<R>::DelayedRNN(int input_size, int hidden_size, int output_size) :
        hidden_rnn(input_size, hidden_size),
        output_rnn(input_size, hidden_size, output_size) {
}

template<typename R>
DelayedRNN<R>::DelayedRNN (const DelayedRNN<R>& rnn, bool copy_w, bool copy_dw) :
        hidden_rnn(rnn.hidden_rnn, copy_w, copy_dw),
        output_rnn(rnn.output_rnn, copy_w, copy_dw) {
}

template<typename R>
std::vector<Mat<R>> DelayedRNN<R>::parameters() const {
    std::vector<Mat<R>> ret;
    for (auto& model: {hidden_rnn, output_rnn}) {
        auto params = model.parameters();
        ret.insert(ret.end(), params.begin(), params.end());
    }
    return ret;
}

template<typename R>
Mat<R> DelayedRNN<R>::initial_states() const {
    return Mat<R>(hidden_rnn.hidden_size, 1, weights<R>::uniform(1. / sqrt(hidden_rnn.hidden_size)));
}

template<typename R>
std::tuple<Mat<R>,Mat<R>> DelayedRNN<R>::activate(
        Mat<R> input_vector,
        Mat<R> prev_hidden) const {

    return std::make_tuple(
        hidden_rnn.activate(input_vector, prev_hidden),
        output_rnn.activate(input_vector, prev_hidden)
    );
}

template<typename R>
DelayedRNN<R> DelayedRNN<R>::shallow_copy() const {
    return DelayedRNN<R>(*this, false, true);
}

template class DelayedRNN<float>;
template class DelayedRNN<double>;

/* StackedInputLayer */
/* SecondOrderCombinator */

template<typename R>
SecondOrderCombinator<R>::SecondOrderCombinator(int input1_size, int input2_size, int output_size) :
        input1_size(input1_size), input2_size(input2_size), output_size(output_size) {
    W1 = Mat<R>(output_size, input1_size, weights<R>::uniform(1.0/sqrt(input1_size)));
    W2 = Mat<R>(output_size, input2_size, weights<R>::uniform(1.0/sqrt(input2_size)));
    b =  Mat<R>(output_size, 1,      weights<R>::uniform(1.0 / sqrt(input1_size)));
}
template<typename R>
SecondOrderCombinator<R>::SecondOrderCombinator(const SecondOrderCombinator& m,
                                                bool copy_w,
                                                bool copy_dw) :
        input1_size(m.input1_size),
        input2_size(m.input2_size),
        output_size(m.output_size),
        W1(m.W1, copy_w, copy_dw),
        W2(m.W2, copy_w, copy_dw),
        b(m.b, copy_w, copy_dw) {
}

template<typename R>
std::vector<Mat<R>> SecondOrderCombinator<R>::parameters() const {
    return { W1, W2, b };
}

template<typename R>
Mat<R> SecondOrderCombinator<R>::activate(Mat<R> i1, Mat<R> i2) const {
    // TODO(jonathan): should be replaced with mul_mul_mul_with_mul
    return W1.dot(i1) * W2.dot(i2) + b;
}

template class SecondOrderCombinator<float>;
template class SecondOrderCombinator<double>;

/* RNN */
template<typename R>
RNN<R>::RNN (int _input_size, int _hidden_size) :
        hidden_size(_hidden_size),
        input_size(_input_size),
        output_size(_hidden_size) {
    create_variables();
}

template<typename R>
RNN<R>::RNN (int _input_size, int _hidden_size, int _output_size) :\
        hidden_size(_hidden_size),
        input_size(_input_size),
        output_size(_output_size) {
    create_variables();
}

template<typename R>
RNN<R>::RNN (const RNN<R>& rnn, bool copy_w, bool copy_dw) :
        hidden_size(rnn.hidden_size),
        input_size(rnn.input_size),
        output_size(rnn.output_size) {
    Wx = Mat<R>(rnn.Wx, copy_w, copy_dw);
    Wh = Mat<R>(rnn.Wh, copy_w, copy_dw);
    b = Mat<R>(rnn.b, copy_w, copy_dw);
}

template<typename R>
RNN<R> RNN<R>::shallow_copy() const {
    return RNN<R>(*this, false, true);
}

template<typename R>
Mat<R> RNN<R>::activate(
    Mat<R> input_vector,
    Mat<R> prev_hidden) const {
    // takes 5% less time to run operations when grouping them (no big gains then)
    // 1.118s with explicit (& temporaries) vs 1.020s with grouped expression & backprop
    // return G.add(G.mul(Wx, input_vector), G.mul_with_bias(Wh, prev_hidden, b));
    DEBUG_ASSERT_MAT_NOT_NAN(Wx)
    DEBUG_ASSERT_MAT_NOT_NAN(input_vector)
    DEBUG_ASSERT_MAT_NOT_NAN(Wh)
    DEBUG_ASSERT_MAT_NOT_NAN(prev_hidden)
    DEBUG_ASSERT_MAT_NOT_NAN(b)
    return MatOps<R>::mul_add_mul_with_bias(Wx, input_vector, Wh, prev_hidden, b);
}

template<typename R>
GatedInput<R>::GatedInput (int _input_size, int _hidden_size) :
        RNN<R>(_input_size, _hidden_size, 1) {
}

template<typename R>
GatedInput<R>::GatedInput (const GatedInput<R>& gate,
                           bool copy_w,
                           bool copy_dw) :
        RNN<R>(gate, copy_w, copy_dw) {
}

template<typename R>
GatedInput<R> GatedInput<R>::shallow_copy() const {
    return GatedInput<R>(*this, false, true);
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
LSTM<R>::LSTM (int _input_size, int _hidden_size, bool _memory_feeds_gates) :
        shortcut(false),
        memory_feeds_gates(_memory_feeds_gates),
        hidden_size(_hidden_size),
        input_size(_input_size),
        input_layer(_input_size, _hidden_size),
        forget_layer(_input_size, _hidden_size),
        output_layer(_input_size, _hidden_size),
        cell_layer(_input_size, _hidden_size) {

    if (memory_feeds_gates) {
        Wci = Mat<R>(hidden_size, 1, weights<R>::uniform(1. / sqrt(hidden_size)));
        Wco = Mat<R>(hidden_size, 1, weights<R>::uniform(1. / sqrt(hidden_size)));
        Wcf = Mat<R>(hidden_size, 1, weights<R>::uniform(1. / sqrt(hidden_size)));
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
        input_layer( {_input_size, shortcut_size}, _hidden_size),
        forget_layer({_input_size, shortcut_size}, _hidden_size),
        output_layer({_input_size, shortcut_size}, _hidden_size),
        cell_layer(  {_input_size, shortcut_size}, _hidden_size) {

    if (memory_feeds_gates) {
        Wci = Mat<R>(hidden_size, 1, weights<R>::uniform(1. / sqrt(hidden_size)));
        Wco = Mat<R>(hidden_size, 1, weights<R>::uniform(1. / sqrt(hidden_size)));
        Wcf = Mat<R>(hidden_size, 1, weights<R>::uniform(1. / sqrt(hidden_size)));
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
typename LSTM<R>::State LSTM<R>::_activate(
    const std::vector<Mat<R>>& gate_input,
    const State& initial_state) const {

    Mat<R> input_gate, forget_gate, output_gate;

    if (memory_feeds_gates) {
        // if the memory feeds the gates (Alex Graves 2013) then
        // a diagonal matrix (Wci and Wcf) connect memory to input
        // and forget gates

        // input gate:
        input_gate  = (
            input_layer.activate(gate_input) + (initial_state.memory * Wci)
        ).sigmoid();
        // forget gate
        forget_gate = (
            forget_layer.activate(gate_input) + (initial_state.memory * Wcf)
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
            output_layer.activate(gate_input) + (cell_d * Wco)
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
    State  initial_state) const {
    auto gate_input = std::vector<Mat<R>>({input_vector, initial_state.hidden});
    return _activate(gate_input, initial_state);
}

template<typename R>
typename LSTM<R>::State LSTM<R>::activate(
    Mat<R> input_vector,
    Mat<R> shortcut_vector,
    State  initial_state) const {
    if (!shortcut)
        throw std::runtime_error("Error: LSTM without Shorcuts received shortcut_vector.");
    auto gate_input = std::vector<Mat<R>>({input_vector, shortcut_vector, initial_state.hidden});
    return _activate(gate_input, initial_state);
}

template<typename R>
typename LSTM<R>::State LSTM<R>::activate_sequence(
    State initial_state,
    const Seq<Mat<R>>& sequence) const {
    for (auto& input_vector : sequence)
        initial_state = activate(
            input_vector,
            initial_state
        );
    return initial_state;
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
    for (auto& size : hidden_sizes) {
        if (shortcut) {
            if (i == 0) {
                // first cell in a shorcut
                // stack cannot "steal" or
                // shorcut from anywhere else
                // so no shorcut is used
                cells.emplace_back(
                    prev_size,
                    size,
                    memory_feeds_gates);
            } else {
                // other cells in a shorcut
                // stack steal the input
                // from the lower stack
                // input_size
                cells.emplace_back(
                    prev_size,
                    input_size,
                    size,
                    memory_feeds_gates);
            }
        } else {
            cells.emplace_back(
                prev_size,
                size,
                memory_feeds_gates);
        }
        prev_size = size;
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
    Mat<R> input_vector,
    std::vector< typename LSTM<R>::State >& previous_state,
    const vector<LSTM<R>>& cells,
    R drop_prob) {

    std::vector< typename LSTM<R>::State> out_state;
    out_state.reserve(cells.size());

    auto layer_input = input_vector;
    auto state_iter = previous_state.begin();

    for (auto& layer : cells) {
        if (layer.shortcut) {
            out_state.emplace_back(
                layer.activate(
                    MatOps<R>::dropout_normalized(layer_input, drop_prob),
                    MatOps<R>::dropout_normalized(input_vector, drop_prob),
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

template class Layer<float>;
template class Layer<double>;

template class StackedInputLayer<float>;
template class StackedInputLayer<double>;

template class RNN<float>;
template class RNN<double>;

template class GatedInput<float>;
template class GatedInput<double>;

template class LSTM<float>;
template class LSTM<double>;

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
    const Seq<Mat<R>>& sequence,
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
