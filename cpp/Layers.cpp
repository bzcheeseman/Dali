#include "Layers.h"

template<typename T>
void Layer<T>::create_variables() {
    using std::make_shared;
    T std = 0.08;
    W = make_shared<mat>(hidden_size, input_size, std);
    b = make_shared<mat>(hidden_size, 1);
}
template<typename T>
Layer<T>::Layer (int _input_size, int _hidden_size) : hidden_size(_hidden_size), input_size(_input_size) {
    create_variables();
}

template<typename T>
typename Layer<T>::shared_mat Layer<T>::activate(Graph<T>& G, typename Layer<T>::shared_mat input_vector) {
    return G.mul_with_bias(W, input_vector, b);
}

template<typename T>
void RNN<T>::create_variables() {
    using std::make_shared;
    T std = 0.08;
    Wx = make_shared<mat>(output_size, input_size,  std);
    Wh = make_shared<mat>(output_size, hidden_size, std);
    b  = make_shared<mat>(output_size, 1);
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
typename RNN<T>::shared_mat RNN<T>::activate(Graph<T>& G, typename RNN<T>::shared_mat input_vector, typename RNN<T>::shared_mat prev_hidden) {
    // takes 5% less time to run operations when grouping them (no big gains then)
    // 1.118s with explicit (& temporaries) vs 1.020s with grouped expression & backprop
    // return G.add(G.mul(Wx, input_vector), G.mul_with_bias(Wh, prev_hidden, b));
    return G.mul_add_mul_with_bias(Wx, input_vector, Wh, prev_hidden, b);
}

template<typename T>
GatedInput<T>::GatedInput (int _input_size, int _hidden_size) : in_gate(_input_size, _hidden_size, 1) {}


template<typename T>
typename GatedInput<T>::shared_mat GatedInput<T>::activate(Graph<T>& G, typename GatedInput<T>::shared_mat input_vector, typename GatedInput<T>::shared_mat prev_hidden) {
    return G.sigmoid(in_gate.activate(G, input_vector, prev_hidden) );
}

template<typename T>
LSTM<T>::LSTM (int _input_size, int _hidden_size) :
    hidden_size(_hidden_size),
    input_size(_input_size),
    input_layer(_input_size, _hidden_size),
    forget_layer(_input_size, _hidden_size),
    output_layer(_input_size, _hidden_size),
    cell_layer(_input_size, _hidden_size) {}

template<typename T>
LSTM<T>::LSTM (int& _input_size, int& _hidden_size) :
    hidden_size(_hidden_size),
    input_size(_input_size),
    input_layer(_input_size, _hidden_size),
    forget_layer(_input_size, _hidden_size),
    output_layer(_input_size, _hidden_size),
    cell_layer(_input_size, _hidden_size) {}

template<typename T>
std::pair<typename LSTM<T>::shared_mat, typename LSTM<T>::shared_mat> LSTM<T>::activate (
    Graph<T>& G,
    typename LSTM<T>::shared_mat input_vector,
    typename LSTM<T>::shared_mat cell_prev,
    typename LSTM<T>::shared_mat hidden_prev) {

    // input gate:
    auto input_gate  = G.sigmoid(input_layer.activate(G, input_vector, hidden_prev));
    // forget gate
    auto forget_gate = G.sigmoid(forget_layer.activate(G, input_vector, hidden_prev));
    // output gate
    auto output_gate = G.sigmoid(output_layer.activate(G, input_vector, hidden_prev));
    // write operation on cells
    auto cell_write  = G.tanh(output_layer.activate(G, input_vector, hidden_prev));

    // compute new cell activation
    auto retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
    auto write_cell  = G.eltmul(input_gate, cell_write); // what do we write to cell
    auto cell_d      = G.add(retain_cell, write_cell); // new cell contents

    // compute hidden state as gated, saturated cell activations
    auto hidden_d    = G.eltmul(output_gate, G.tanh(cell_d));
    return std::pair<shared_mat,shared_mat>(cell_d, hidden_d);
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


template<typename celltype>
std::vector<celltype> StackedCells(const int& input_size, const std::vector<int>& hidden_sizes) {
    std::vector<celltype> cells;
    cells.reserve(hidden_sizes.size());
    int prev_size = input_size;
    for (auto& size : hidden_sizes) {
        cells.emplace_back(prev_size, size);
        prev_size = size;
    }
    return cells;
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