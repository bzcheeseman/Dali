#include "Layers.h"

template<typename T>
void Layer<T>::create_variables() {
    using std::make_shared;
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
std::vector<typename Layer<T>::shared_mat> Layer<T>::parameters() const{
    return std::vector<typename Layer<T>::shared_mat>({W, b});
}

template<typename T>
void RNN<T>::create_variables() {
    using std::make_shared;
    T upper = 1. / sqrt(input_size);
    Wx = make_shared<mat>(output_size, input_size,  -upper, upper);
    upper = 1. / sqrt(hidden_size);
    Wh = make_shared<mat>(output_size, hidden_size, -upper, upper);
    b  = make_shared<mat>(output_size, 1);
}

template<typename T>
std::vector<typename RNN<T>::shared_mat> RNN<T>::parameters() const {
    return std::vector<typename RNN<T>::shared_mat>({Wx, Wh, b});
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
typename RNN<T>::shared_mat RNN<T>::activate(
    Graph<T>& G,
    typename RNN<T>::shared_mat input_vector,
    typename RNN<T>::shared_mat prev_hidden) const {
    // takes 5% less time to run operations when grouping them (no big gains then)
    // 1.118s with explicit (& temporaries) vs 1.020s with grouped expression & backprop
    // return G.add(G.mul(Wx, input_vector), G.mul_with_bias(Wh, prev_hidden, b));
    return G.mul_add_mul_with_bias(Wx, input_vector, Wh, prev_hidden, b);
}

template<typename T>
GatedInput<T>::GatedInput (int _input_size, int _hidden_size) : in_gate(_input_size, _hidden_size, 1) {
    in_gate.b->set_name("Gated Input bias");
    in_gate.Wx->set_name("Gated Input Wx");
    in_gate.Wh->set_name("Gated Input Wx");
}


template<typename T>
std::vector<typename GatedInput<T>::shared_mat> GatedInput<T>::parameters () const {
    return in_gate.parameters();
}

template<typename T>
typename GatedInput<T>::shared_mat GatedInput<T>::activate(Graph<T>& G, typename GatedInput<T>::shared_mat input_vector, typename GatedInput<T>::shared_mat prev_hidden) const {
    auto unsigmoided_gate = in_gate.activate(G, input_vector, prev_hidden);
    unsigmoided_gate->set_name("unsigmoided_gate");
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
LSTM<T>::LSTM (int _input_size, int _hidden_size) :
    hidden_size(_hidden_size),
    input_size(_input_size),
    input_layer(_input_size, _hidden_size),
    forget_layer(_input_size, _hidden_size),
    output_layer(_input_size, _hidden_size),
    cell_layer(_input_size, _hidden_size) {
    // Ilya Sutskever recommends initializing with
    // forget gate at high value
    // http://yyue.blogspot.fr/2015/01/a-brief-overview-of-deep-learning.html
    forget_layer.b->w(0) = 100;
    name_internal_layers();
}

template<typename T>
LSTM<T>::LSTM (int& _input_size, int& _hidden_size) :
    hidden_size(_hidden_size),
    input_size(_input_size),
    input_layer(_input_size, _hidden_size),
    forget_layer(_input_size, _hidden_size),
    output_layer(_input_size, _hidden_size),
    cell_layer(_input_size, _hidden_size) {
    // Ilya Sutskever recommends initializing with
    // forget gate at high value
    // http://yyue.blogspot.fr/2015/01/a-brief-overview-of-deep-learning.html
    forget_layer.b->w(0) = 100;
    name_internal_layers();
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
    return std::pair<shared_mat,shared_mat>(cell_d, hidden_d);
}

template<typename T>
std::vector<typename LSTM<T>::shared_mat> LSTM<T>::parameters() const {
    std::vector<typename LSTM<T>::shared_mat> parameters;

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
std::pair< std::vector<typename LSTM<T>::shared_mat >, std::vector<typename LSTM<T>::shared_mat > > LSTM<T>::initial_states(std::vector<int>& hidden_sizes) {
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

template<typename T>
pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>> forward_LSTMs(Graph<T>& G,
    shared_ptr<Mat<T>> input_vector,
    pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>>& previous_state,
    vector<LSTM<T>>& cells) {

    auto previous_state_cells = previous_state.first;
    auto previous_state_hiddens = previous_state.second;

    auto cell_iter = previous_state_cells.begin();
    auto hidden_iter = previous_state_hiddens.begin();

    pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>> out_state;
    out_state.first.reserve(cells.size());
    out_state.second.reserve(cells.size());

    auto layer_input = input_vector;

    for (auto& layer : cells) {

        auto layer_out = layer.activate(G, layer_input, *cell_iter, *hidden_iter);

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

template class RNN<float>;
template class RNN<double>;

template std::vector<RNN<float>> StackedCells <RNN<float>>(const int&, const std::vector<int>&);
template std::vector<RNN<double>> StackedCells <RNN<double>>(const int&, const std::vector<int>&);

template class GatedInput<float>;
template class GatedInput<double>;

template std::vector<GatedInput<float>> StackedCells <GatedInput<float>>(const int&, const std::vector<int>&);
template std::vector<GatedInput<double>> StackedCells <GatedInput<double>>(const int&, const std::vector<int>&);

template class LSTM<float>;
template class LSTM<double>;

template std::vector<LSTM<float>> StackedCells <LSTM<float>>(const int&, const std::vector<int>&);
template std::vector<LSTM<double>> StackedCells <LSTM<double>>(const int&, const std::vector<int>&);

template pair<vector<shared_ptr<Mat<double>>>, vector<shared_ptr<Mat<double>>>> forward_LSTMs(Graph<double>&,
    shared_ptr<Mat<double>>,
    pair<vector<shared_ptr<Mat<double>>>, vector<shared_ptr<Mat<double>>>>&,
    vector<LSTM<double>>&);

template pair<vector<shared_ptr<Mat<float>>>, vector<shared_ptr<Mat<float>>>> forward_LSTMs(Graph<float>&,
    shared_ptr<Mat<float>>,
    pair<vector<shared_ptr<Mat<float>>>, vector<shared_ptr<Mat<float>>>>&,
    vector<LSTM<float>>&);
