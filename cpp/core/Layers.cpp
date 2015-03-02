#include "Layers.h"

using std::make_shared;
using std::vector;

template<typename T>
void Layer<T>::create_variables() {
    T upper = 1. / sqrt(input_size);
    W = make_shared<mat>(hidden_size, input_size, -upper, upper);
    b = make_shared<mat>(hidden_size, 1);
}
template<typename T>
Layer<T>::Layer (int _input_size, int _hidden_size) : hidden_size(_hidden_size), input_size(_input_size) {
    create_variables();
}

template<typename T>
typename Layer<T>::shared_mat Layer<T>::activate(Graph<T>& G, typename Layer<T>::shared_mat input_vector) const {
    return G.mul_with_bias(W, input_vector, b);
}

template<typename T>
Layer<T>::Layer (const Layer<T>& layer, bool copy_w, bool copy_dw) : hidden_size(layer.hidden_size), input_size(layer.input_size) {
    W = make_shared<mat>(*layer.W, copy_w, copy_dw);
    b = make_shared<mat>(*layer.b, copy_w, copy_dw);
}

template<typename T>
Layer<T> Layer<T>::shallow_copy() const {
    return Layer<T>(*this, false, true);
}

template<typename T>
std::vector<typename Layer<T>::shared_mat> Layer<T>::parameters() const{
    return std::vector<typename Layer<T>::shared_mat>({W, b});
}

// StackedInputLayer:
template<typename T>
void StackedInputLayer<T>::create_variables() {
    int total_input_size = 0;
    for (auto& input_size : input_sizes) total_input_size += input_size;
    T upper = 1. / sqrt(total_input_size);
    matrices.reserve(input_sizes.size());
    for (auto& input_size : input_sizes) {
        matrices.emplace_back(make_shared<mat>(hidden_size, input_size, -upper, upper));
        DEBUG_ASSERT_MAT_NOT_NAN(matrices[matrices.size() -1]);
    }
    b = make_shared<mat>(hidden_size, 1);
}
template<typename T>
StackedInputLayer<T>::StackedInputLayer (vector<int> _input_sizes, int _hidden_size) : hidden_size(_hidden_size), input_sizes(_input_sizes) {
    create_variables();
}

template<typename T>
StackedInputLayer<T>::StackedInputLayer (std::initializer_list<int> _input_sizes, int _hidden_size) : hidden_size(_hidden_size), input_sizes(_input_sizes) {
    create_variables();
}

template<typename T>
vector<typename StackedInputLayer<T>::shared_mat> StackedInputLayer<T>::zip_inputs_with_matrices_and_bias(const vector<typename StackedInputLayer<T>::shared_mat>& inputs) const {
    vector<shared_mat> zipped;
    zipped.reserve(matrices.size() * 2 + 1);
    auto input_ptr = inputs.begin();
    auto mat_ptr = matrices.begin();
    while (mat_ptr != matrices.end()) {
        zipped.emplace_back(*mat_ptr++);
        zipped.emplace_back(*input_ptr++);
    }
    zipped.emplace_back(b);
    return zipped;
}

template<typename T>
vector<typename StackedInputLayer<T>::shared_mat> StackedInputLayer<T>::zip_inputs_with_matrices_and_bias(
    typename StackedInputLayer<T>::shared_mat input,
    const vector<typename StackedInputLayer<T>::shared_mat>& inputs) const {
    vector<shared_mat> zipped;
    zipped.reserve(matrices.size() * 2 + 1);
    auto input_ptr = inputs.begin();
    auto mat_ptr = matrices.begin();

    // We are provided separately with anoter input vector
    // that will go first in the zip, while the remainder will
    // be loaded in "zip" form with the vector of inputs
    zipped.emplace_back(*mat_ptr);
    zipped.emplace_back(input);

    mat_ptr++;

    DEBUG_ASSERT_MAT_NOT_NAN((*mat_ptr));
    DEBUG_ASSERT_MAT_NOT_NAN(input);

    while (mat_ptr != matrices.end()) {

        DEBUG_ASSERT_MAT_NOT_NAN((*mat_ptr));
        DEBUG_ASSERT_MAT_NOT_NAN((*input_ptr));

        zipped.emplace_back(*mat_ptr);
        zipped.emplace_back(*input_ptr);
        mat_ptr++;
        input_ptr++;
    }
    zipped.emplace_back(b);
    return zipped;
}

template<typename T>
typename StackedInputLayer<T>::shared_mat StackedInputLayer<T>::activate(
    Graph<T>& G,
    const vector<typename StackedInputLayer<T>::shared_mat>& inputs) const {
    auto zipped = zip_inputs_with_matrices_and_bias(inputs);
    return G.mul_add_mul_with_bias(zipped);
}

template<typename T>
typename StackedInputLayer<T>::shared_mat StackedInputLayer<T>::activate(
    Graph<T>& G,
    typename StackedInputLayer<T>::shared_mat input,
    const vector<typename StackedInputLayer<T>::shared_mat>& inputs) const {
    DEBUG_ASSERT_MAT_NOT_NAN(input);
    auto zipped = zip_inputs_with_matrices_and_bias(input, inputs);

    auto out = G.mul_add_mul_with_bias(zipped);

    DEBUG_ASSERT_MAT_NOT_NAN(out);

    return out;
}

template<typename T>
StackedInputLayer<T>::StackedInputLayer (const StackedInputLayer<T>& layer, bool copy_w, bool copy_dw) : hidden_size(layer.hidden_size), input_sizes(layer.input_sizes) {
    matrices.reserve(layer.matrices.size());
    for (auto& matrix : layer.matrices)
        matrices.emplace_back(make_shared<mat>(*matrix, copy_w, copy_dw));
    b = make_shared<mat>(*layer.b, copy_w, copy_dw);
}

template<typename T>
StackedInputLayer<T> StackedInputLayer<T>::shallow_copy() const {
    return StackedInputLayer<T>(*this, false, true);
}

template<typename T>
std::vector<typename StackedInputLayer<T>::shared_mat> StackedInputLayer<T>::parameters() const{
    auto params = vector<shared_mat>(matrices);
    params.emplace_back(b);
    return params;
}

template<typename T>
void RNN<T>::create_variables() {
    T upper = 1. / sqrt(input_size);
    Wx = make_shared<mat>(output_size, input_size,  -upper, upper);
    upper = 1. / sqrt(hidden_size);
    Wh = make_shared<mat>(output_size, hidden_size, -upper, upper);
    b  = make_shared<mat>(output_size, 1, -upper, upper);
}

template<typename T>
void ShortcutRNN<T>::create_variables() {
    T upper = 1. / sqrt(input_size);
    Wx = make_shared<mat>(output_size, input_size,  -upper, upper);
    upper = 1. / sqrt(shortcut_size);
    Ws = make_shared<mat>(output_size, shortcut_size,  -upper, upper);
    upper = 1. / sqrt(hidden_size);
    Wh = make_shared<mat>(output_size, hidden_size, -upper, upper);
    b  = make_shared<mat>(output_size, 1, -upper, upper);
}

template<typename T>
std::vector<typename RNN<T>::shared_mat> RNN<T>::parameters() const {
    return std::vector<typename RNN<T>::shared_mat>({Wx, Wh, b});
}

template<typename T>
std::vector<typename ShortcutRNN<T>::shared_mat> ShortcutRNN<T>::parameters() const {
    return std::vector<typename ShortcutRNN<T>::shared_mat>({Wx, Wh, Ws, b});
}

template<typename T>
RNN<T>::RNN (int _input_size, int _hidden_size) : hidden_size(_hidden_size), input_size(_input_size), output_size(_hidden_size) {
    create_variables();
}

template<typename T>
RNN<T>::RNN (int _input_size, int _hidden_size, int _output_size) : hidden_size(_hidden_size), input_size(_input_size), output_size(_output_size) {
    create_variables();
}

template<typename T>
ShortcutRNN<T>::ShortcutRNN (int _input_size, int _shortcut_size, int _hidden_size) : hidden_size(_hidden_size), input_size(_input_size), output_size(_hidden_size), shortcut_size(_shortcut_size) {
    create_variables();
}

template<typename T>
ShortcutRNN<T>::ShortcutRNN (int _input_size, int _shortcut_size, int _hidden_size,  int _output_size) : hidden_size(_hidden_size), input_size(_input_size), shortcut_size(_shortcut_size), output_size(_output_size) {
    create_variables();
}

template<typename T>
RNN<T>::RNN (const RNN<T>& rnn, bool copy_w, bool copy_dw) : hidden_size(rnn.hidden_size), input_size(rnn.input_size), output_size(rnn.output_size) {
    Wx = make_shared<mat>(*rnn.Wx, copy_w, copy_dw);
    Wh = make_shared<mat>(*rnn.Wh, copy_w, copy_dw);
    b = make_shared<mat>(*rnn.b, copy_w, copy_dw);
}

template<typename T>
ShortcutRNN<T>::ShortcutRNN (const ShortcutRNN<T>& rnn, bool copy_w, bool copy_dw) : hidden_size(rnn.hidden_size), input_size(rnn.input_size), output_size(rnn.output_size), shortcut_size(rnn.shortcut_size) {
    Wx = make_shared<mat>(*rnn.Wx, copy_w, copy_dw);
    Wh = make_shared<mat>(*rnn.Wh, copy_w, copy_dw);
    Ws = make_shared<mat>(*rnn.Ws, copy_w, copy_dw);
    b = make_shared<mat>(*rnn.b, copy_w, copy_dw);
}

template<typename T>
RNN<T> RNN<T>::shallow_copy() const {
    return RNN<T>(*this, false, true);
}

template<typename T>
ShortcutRNN<T> ShortcutRNN<T>::shallow_copy() const {
    return ShortcutRNN<T>(*this, false, true);
}

template<typename T>
typename RNN<T>::shared_mat RNN<T>::activate(
    Graph<T>& G,
    typename RNN<T>::shared_mat input_vector,
    typename RNN<T>::shared_mat prev_hidden) const {
    // takes 5% less time to run operations when grouping them (no big gains then)
    // 1.118s with explicit (& temporaries) vs 1.020s with grouped expression & backprop
    // return G.add(G.mul(Wx, input_vector), G.mul_with_bias(Wh, prev_hidden, b));
    DEBUG_ASSERT_NOT_NAN(Wx->w);
    DEBUG_ASSERT_NOT_NAN(input_vector->w);
    DEBUG_ASSERT_NOT_NAN(Wh->w);
    DEBUG_ASSERT_NOT_NAN(prev_hidden->w);
    DEBUG_ASSERT_NOT_NAN(b->w);
    return G.mul_add_mul_with_bias(Wx, input_vector, Wh, prev_hidden, b);
}

template<typename T>
typename ShortcutRNN<T>::shared_mat ShortcutRNN<T>::activate(
    Graph<T>& G,
    typename ShortcutRNN<T>::shared_mat input_vector,
    typename ShortcutRNN<T>::shared_mat shortcut_vector,
    typename ShortcutRNN<T>::shared_mat prev_hidden) const {
    // takes 5% less time to run operations when grouping them (no big gains then)
    // 1.118s with explicit (& temporaries) vs 1.020s with grouped expression & backprop
    // return G.add(G.mul(Wx, input_vector), G.mul_with_bias(Wh, prev_hidden, b));
    DEBUG_ASSERT_NOT_NAN(Wx->w);
    DEBUG_ASSERT_NOT_NAN(input_vector->w);
    DEBUG_ASSERT_NOT_NAN(Wh->w);
    DEBUG_ASSERT_NOT_NAN(prev_hidden->w);
    DEBUG_ASSERT_NOT_NAN(b->w);
    return G.add( G.mul(Ws, shortcut_vector), G.mul_add_mul_with_bias(Wx, input_vector, Wh, prev_hidden, b));
}

template<typename T>
GatedInput<T>::GatedInput (int _input_size, int _hidden_size) : in_gate(_input_size, _hidden_size, 1) {
    in_gate.b->set_name("Gated Input bias");
    in_gate.Wx->set_name("Gated Input Wx");
    in_gate.Wh->set_name("Gated Input Wx");
}

template<typename T>
GatedInput<T>::GatedInput (const GatedInput<T>& gate, bool copy_w, bool copy_dw) : in_gate(gate.in_gate, copy_w, copy_dw) {
    in_gate.b->set_name("Gated Input bias");
    in_gate.Wx->set_name("Gated Input Wx");
    in_gate.Wh->set_name("Gated Input Wx");
}

template<typename T>
GatedInput<T> GatedInput<T>::shallow_copy() const {
    return GatedInput<T>(*this, false, true);
}

template<typename T>
std::vector<typename GatedInput<T>::shared_mat> GatedInput<T>::parameters () const {
    return in_gate.parameters();
}

template<typename T>
typename GatedInput<T>::shared_mat GatedInput<T>::activate(Graph<T>& G, typename GatedInput<T>::shared_mat input_vector, typename GatedInput<T>::shared_mat prev_hidden) const {
    auto unsigmoided_gate = in_gate.activate(G, input_vector, prev_hidden);
    return G.sigmoid( unsigmoided_gate );
}

template<typename T>
void LSTM<T>::name_internal_layers() {
    forget_layer.b->set_name("LSTM Forget bias");
    forget_layer.Wx->set_name("LSTM Forget Wx");
    forget_layer.Wh->set_name("LSTM Forget Wh");

    input_layer.b->set_name("LSTM Input bias");
    input_layer.Wx->set_name("LSTM Input Wx");
    input_layer.Wh->set_name("LSTM Input Wh");

    output_layer.b->set_name("LSTM Output bias");
    output_layer.Wx->set_name("LSTM Output Wx");
    output_layer.Wh->set_name("LSTM Output Wh");

    cell_layer.b->set_name("LSTM Cell bias");
    cell_layer.Wx->set_name("LSTM Cell Wx");
    cell_layer.Wh->set_name("LSTM Cell Wh");
}
template<typename T>
void ShortcutLSTM<T>::name_internal_layers() {
    forget_layer.b->set_name("Shortcut Forget bias");
    forget_layer.Wx->set_name("Shortcut Forget Wx");
    forget_layer.Wh->set_name("Shortcut Forget Wh");
    forget_layer.Ws->set_name("Shortcut Forget Ws");

    input_layer.b->set_name("Shortcut Input bias");
    input_layer.Wx->set_name("Shortcut Input Wx");
    input_layer.Wh->set_name("Shortcut Input Wh");
    input_layer.Ws->set_name("Shortcut Input Ws");

    output_layer.b->set_name("Shortcut Output bias");
    output_layer.Wx->set_name("Shortcut Output Wx");
    output_layer.Wh->set_name("Shortcut Output Wh");
    output_layer.Ws->set_name("Shortcut Output Ws");

    cell_layer.b->set_name("Shortcut Cell bias");
    cell_layer.Wx->set_name("Shortcut Cell Wx");
    cell_layer.Wh->set_name("Shortcut Cell Wh");
    cell_layer.Ws->set_name("Shortcut Cell Ws");
}

template<typename T>
LSTM<T>::LSTM (int _input_size, int _hidden_size) :
    hidden_size(_hidden_size),
    input_size(_input_size),
    input_layer(_input_size, _hidden_size),
    forget_layer(_input_size, _hidden_size),
    output_layer(_input_size, _hidden_size),
    cell_layer(_input_size, _hidden_size) {
    // Note: Ilya Sutskever recommends initializing with
    // forget gate at high value
    // http://yyue.blogspot.fr/2015/01/a-brief-overview-of-deep-learning.html
    // forget_layer.b->w.array() += 2;
    name_internal_layers();
}

template<typename T>
ShortcutLSTM<T>::ShortcutLSTM (int _input_size, int _shortcut_size, int _hidden_size) :
    hidden_size(_hidden_size),
    input_size(_input_size),
    shortcut_size(_shortcut_size),
    input_layer(_input_size, _shortcut_size, _hidden_size),
    forget_layer(_input_size, _shortcut_size, _hidden_size),
    output_layer(_input_size, _shortcut_size, _hidden_size),
    cell_layer(_input_size, _shortcut_size, _hidden_size) {
    // forget_layer.b->w.array() += 2;
    name_internal_layers();
}

template<typename T>
LSTM<T>::LSTM (const LSTM<T>& lstm, bool copy_w, bool copy_dw) :
    hidden_size(lstm.hidden_size),
    input_size(lstm.input_size),
    input_layer(lstm.input_layer, copy_w, copy_dw),
    forget_layer(lstm.forget_layer, copy_w, copy_dw),
    output_layer(lstm.output_layer, copy_w, copy_dw),
    cell_layer(lstm.cell_layer, copy_w, copy_dw)
    {
    name_internal_layers();
}

template<typename T>
ShortcutLSTM<T>::ShortcutLSTM (const ShortcutLSTM<T>& lstm, bool copy_w, bool copy_dw) :
    hidden_size(lstm.hidden_size),
    input_size(lstm.input_size),
    shortcut_size(lstm.shortcut_size),
    input_layer(lstm.input_layer, copy_w, copy_dw),
    forget_layer(lstm.forget_layer, copy_w, copy_dw),
    output_layer(lstm.output_layer, copy_w, copy_dw),
    cell_layer(lstm.cell_layer, copy_w, copy_dw)
    {
    name_internal_layers();
}

template<typename T>
LSTM<T> LSTM<T>::shallow_copy() const {
    return LSTM<T>(*this, false, true);
}
template<typename T>
ShortcutLSTM<T> ShortcutLSTM<T>::shallow_copy() const {
    return ShortcutLSTM<T>(*this, false, true);
}

template<typename T>
std::pair<typename LSTM<T>::shared_mat, typename LSTM<T>::shared_mat> LSTM<T>::activate (
    Graph<T>& G,
    typename LSTM<T>::shared_mat input_vector,
    typename LSTM<T>::shared_mat cell_prev,
    typename LSTM<T>::shared_mat hidden_prev) const {

    // input gate:
    auto input_gate  = G.sigmoid(input_layer.activate(G, input_vector, hidden_prev));
    // forget gate
    auto forget_gate = G.sigmoid(forget_layer.activate(G, input_vector, hidden_prev));
    // output gate
    auto output_gate = G.sigmoid(output_layer.activate(G, input_vector, hidden_prev));
    // write operation on cells
    auto cell_write  = G.tanh(cell_layer.activate(G, input_vector, hidden_prev));

    // compute new cell activation
    auto retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
    auto write_cell  = G.eltmul(input_gate, cell_write); // what do we write to cell
    auto cell_d      = G.add(retain_cell, write_cell); // new cell contents

    // compute hidden state as gated, saturated cell activations

    auto hidden_d    = G.eltmul(output_gate, G.tanh(cell_d));

    DEBUG_ASSERT_NOT_NAN(hidden_d->w);
    DEBUG_ASSERT_NOT_NAN(cell_d->w);

    return std::pair<shared_mat,shared_mat>(cell_d, hidden_d);
}

template<typename T>
std::pair<typename ShortcutLSTM<T>::shared_mat, typename ShortcutLSTM<T>::shared_mat> ShortcutLSTM<T>::activate (
    Graph<T>& G,
    typename ShortcutLSTM<T>::shared_mat input_vector,
    typename ShortcutLSTM<T>::shared_mat shortcut_vector,
    typename ShortcutLSTM<T>::shared_mat cell_prev,
    typename ShortcutLSTM<T>::shared_mat hidden_prev) const {

    // input gate:
    auto input_gate  = G.sigmoid(input_layer.activate(G, input_vector, shortcut_vector, hidden_prev));
    // forget gate
    auto forget_gate = G.sigmoid(forget_layer.activate(G, input_vector, shortcut_vector, hidden_prev));
    // output gate
    auto output_gate = G.sigmoid(output_layer.activate(G, input_vector, shortcut_vector, hidden_prev));
    // write operation on cells
    auto cell_write  = G.tanh(cell_layer.activate(G, input_vector, shortcut_vector, hidden_prev));

    // compute new cell activation
    auto retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
    auto write_cell  = G.eltmul(input_gate, cell_write); // what do we write to cell
    auto cell_d      = G.add(retain_cell, write_cell); // new cell contents

    // compute hidden state as gated, saturated cell activations

    auto hidden_d    = G.eltmul(output_gate, G.tanh(cell_d));

    DEBUG_ASSERT_NOT_NAN(hidden_d->w);
    DEBUG_ASSERT_NOT_NAN(cell_d->w);

    return std::pair<shared_mat,shared_mat>(cell_d, hidden_d);
}

template<typename T>
std::vector<typename LSTM<T>::shared_mat> LSTM<T>::parameters() const {
    std::vector<shared_mat> parameters;

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

template<typename T>
std::vector<typename ShortcutLSTM<T>::shared_mat> ShortcutLSTM<T>::parameters() const {
    std::vector<shared_mat> parameters;

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

template<typename T>
std::pair< std::vector<typename LSTM<T>::shared_mat >, std::vector<typename LSTM<T>::shared_mat > > LSTM<T>::initial_states(const std::vector<int>& hidden_sizes) {
    std::pair< std::vector<typename LSTM<T>::shared_mat >, std::vector<typename LSTM<T>::shared_mat > > initial_state;
    initial_state.first.reserve(hidden_sizes.size());
    initial_state.second.reserve(hidden_sizes.size());
    for (auto& size : hidden_sizes) {
        initial_state.first.emplace_back(std::make_shared<typename LSTM<T>::mat>(size, 1));
        initial_state.second.emplace_back(std::make_shared<typename LSTM<T>::mat>(size, 1));
    }
    return initial_state;
}

using std::pair;
using std::vector;
using std::shared_ptr;

template<typename celltype>
vector<celltype> StackedCells(const int& input_size, const vector<int>& hidden_sizes) {
    vector<celltype> cells;
    cells.reserve(hidden_sizes.size());
    int prev_size = input_size;
    for (auto& size : hidden_sizes) {
        cells.emplace_back(prev_size, size);
        prev_size = size;
    }
    return cells;
}

template <typename celltype>
vector<celltype> StackedCells(const int& input_size, const int& shortcut_size, const vector<int>& hidden_sizes) {
    vector<celltype> cells;
    cells.reserve(hidden_sizes.size());
    int prev_size = input_size;
    for (auto& size : hidden_sizes) {
        cells.emplace_back(prev_size, size);
        prev_size = size;
    }
    return cells;
}

template<> vector<ShortcutLSTM<float>> StackedCells<ShortcutLSTM<float>>(const int& input_size, const int& shortcut_size, const vector<int>& hidden_sizes) {
    vector<ShortcutLSTM<float>> cells;
    cells.reserve(hidden_sizes.size());
    int prev_size = input_size;
    for (auto& size : hidden_sizes) {
        cells.emplace_back(prev_size, shortcut_size, size);
        prev_size = size;
    }
    return cells;
}

template<> vector<ShortcutLSTM<double>> StackedCells<ShortcutLSTM<double>>(const int& input_size, const int& shortcut_size, const vector<int>& hidden_sizes) {
    vector<ShortcutLSTM<double>> cells;
    cells.reserve(hidden_sizes.size());
    int prev_size = input_size;
    for (auto& size : hidden_sizes) {
        cells.emplace_back(prev_size, shortcut_size, size);
        prev_size = size;
    }
    return cells;
}

template<typename celltype>
vector<celltype> StackedCells(const vector<celltype>& source_cells, bool copy_w, bool copy_dw) {
    vector<celltype> cells;
    cells.reserve(source_cells.size());
    for (const auto& cell : source_cells)
        cells.emplace_back(cell, copy_w, copy_dw);
    return cells;
}

template<typename T>
pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>> forward_LSTMs(Graph<T>& G,
    shared_ptr<Mat<T>> input_vector,
    pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>>& previous_state,
    const vector<LSTM<T>>& cells,
    T drop_prob) {

    auto previous_state_cells = previous_state.first;
    auto previous_state_hiddens = previous_state.second;

    auto cell_iter = previous_state_cells.begin();
    auto hidden_iter = previous_state_hiddens.begin();

    pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>> out_state;
    out_state.first.reserve(cells.size());
    out_state.second.reserve(cells.size());

    auto layer_input = input_vector;

    for (auto& layer : cells) {

        auto layer_out = layer.activate(G,
                                        G.dropout_normalized(layer_input, drop_prob),
                                        *cell_iter,
                                        *hidden_iter);

        out_state.first.push_back(layer_out.first);
        out_state.second.push_back(layer_out.second);

        ++cell_iter;
        ++hidden_iter;

        layer_input = layer_out.second;
    }

    return out_state;
}


template<typename T>
pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>> forward_LSTMs(Graph<T>& G,
    shared_ptr<Mat<T>> input_vector,
    pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>>& previous_state,
    const LSTM<T>& base_cell,
    const vector<ShortcutLSTM<T>>& cells,
    T drop_prob) {

    auto previous_state_cells = previous_state.first;
    auto previous_state_hiddens = previous_state.second;

    auto cell_iter = previous_state_cells.begin();
    auto hidden_iter = previous_state_hiddens.begin();

    pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>> out_state;
    out_state.first.reserve(cells.size() + 1);
    out_state.second.reserve(cells.size() + 1);

    auto layer_input = input_vector;

    auto layer_out = base_cell.activate(G, layer_input, *cell_iter, *hidden_iter);
    out_state.first.push_back(layer_out.first);
    out_state.second.push_back(layer_out.second);

    ++cell_iter;
    ++hidden_iter;

    layer_input = layer_out.second;

    for (auto& layer : cells) {

        // The next cell up gets both the base input (input_vector)
        // and the cell below's input activation (layer_input)
        // => façon Alex Graves
        layer_out = layer.activate(G,
                                   G.dropout_normalized(layer_input, drop_prob),
                                   G.dropout_normalized(input_vector, drop_prob),
                                   *cell_iter,
                                   *hidden_iter);

        out_state.first.push_back(layer_out.first);
        out_state.second.push_back(layer_out.second);

        ++cell_iter;
        ++hidden_iter;

        layer_input = layer_out.second;
    }

    return out_state;
}

template class Layer<float>;
template class Layer<double>;

template class StackedInputLayer<float>;
template class StackedInputLayer<double>;

template class RNN<float>;
template class RNN<double>;

template class ShortcutRNN<float>;
template class ShortcutRNN<double>;

template std::vector<RNN<float>> StackedCells <RNN<float>>(const int&, const std::vector<int>&);
template std::vector<RNN<double>> StackedCells <RNN<double>>(const int&, const std::vector<int>&);

template class GatedInput<float>;
template class GatedInput<double>;

template std::vector<GatedInput<float>> StackedCells <GatedInput<float>>(const int&, const std::vector<int>&);
template std::vector<GatedInput<double>> StackedCells <GatedInput<double>>(const int&, const std::vector<int>&);

template class LSTM<float>;
template class LSTM<double>;

template class ShortcutLSTM<float>;
template class ShortcutLSTM<double>;

template std::vector<LSTM<float>> StackedCells <LSTM<float>>(const int&, const std::vector<int>&);
template std::vector<LSTM<double>> StackedCells <LSTM<double>>(const int&, const std::vector<int>&);

template std::vector<LSTM<float>> StackedCells <LSTM<float>>(const std::vector<LSTM<float>>&, bool, bool);
template std::vector<LSTM<double>> StackedCells <LSTM<double>>(const std::vector<LSTM<double>>&, bool, bool);

template std::vector<LSTM<float>> StackedCells (const int&, const int&, const std::vector<int>&);
template std::vector<LSTM<double>> StackedCells (const int&, const int&, const std::vector<int>&);

template std::vector<ShortcutLSTM<float>> StackedCells (const int&, const int&, const std::vector<int>&);
template std::vector<ShortcutLSTM<double>> StackedCells (const int&, const int&, const std::vector<int>&);

template std::vector<ShortcutLSTM<float>> StackedCells <ShortcutLSTM<float>>(const std::vector<ShortcutLSTM<float>>&, bool, bool);
template std::vector<ShortcutLSTM<double>> StackedCells <ShortcutLSTM<double>>(const std::vector<ShortcutLSTM<double>>&, bool, bool);

template pair<vector<shared_ptr<Mat<double>>>, vector<shared_ptr<Mat<double>>>> forward_LSTMs(Graph<double>&,
    shared_ptr<Mat<double>>,
    pair<vector<shared_ptr<Mat<double>>>, vector<shared_ptr<Mat<double>>>>&,
    const vector<LSTM<double>>&,
    double);

template pair<vector<shared_ptr<Mat<float>>>, vector<shared_ptr<Mat<float>>>> forward_LSTMs(Graph<float>&,
    shared_ptr<Mat<float>>,
    pair<vector<shared_ptr<Mat<float>>>, vector<shared_ptr<Mat<float>>>>&,
    const vector<LSTM<float>>&,
    float);

template pair<vector<shared_ptr<Mat<double>>>, vector<shared_ptr<Mat<double>>>> forward_LSTMs(Graph<double>&,
    shared_ptr<Mat<double>>,
    pair<vector<shared_ptr<Mat<double>>>, vector<shared_ptr<Mat<double>>>>&,
    const LSTM<double>&,
    const vector<ShortcutLSTM<double>>&,
    double);

template pair<vector<shared_ptr<Mat<float>>>, vector<shared_ptr<Mat<float>>>> forward_LSTMs(Graph<float>&,
    shared_ptr<Mat<float>>,
    pair<vector<shared_ptr<Mat<float>>>, vector<shared_ptr<Mat<float>>>>&,
    const LSTM<float>&,
    const vector<ShortcutLSTM<float>>&,
    float);
