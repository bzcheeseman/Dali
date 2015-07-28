#include "dali/layers/GRU.h"

using std::vector;

template<typename R>
GRU<R>::GRU() : GRU<R>(0,0) {}

template<typename R>
GRU<R>::GRU(int _input_size, int _hidden_size) :
    input_size(_input_size),
    hidden_size(_hidden_size),
    reset_layer({_input_size, _hidden_size}, _hidden_size),
    memory_interpolation_layer({_input_size, _hidden_size}, _hidden_size),
    memory_to_memory_layer({_input_size, _hidden_size}, _hidden_size) {}

template<typename R>
GRU<R>::GRU(const GRU<R>& other, bool copy_w, bool copy_dw) :
    input_size(other.input_size),
    hidden_size(other.hidden_size),
    reset_layer(other.reset_layer, copy_w, copy_dw),
    memory_interpolation_layer(other.memory_interpolation_layer, copy_w, copy_dw),
    memory_to_memory_layer(other.memory_to_memory_layer, copy_w, copy_dw) {}

template<typename R>
GRU<R> GRU<R>::shallow_copy() const {
    return GRU<R>(*this, false, true);
}

template<typename R>
Mat<R> GRU<R>::activate(
        Mat<R> input_vector,
        Mat<R> previous_state) const {

    auto reset_gate = reset_layer.activate({input_vector, previous_state}).sigmoid();

    // the new state dampened by resetting
    auto reset_state = reset_gate * previous_state;

    // the new hidden state:
    auto candidate_new_state = memory_to_memory_layer.activate({input_vector, reset_state}).tanh();

    // how much to update the new hidden state:
    auto update_gate = memory_interpolation_layer.activate({input_vector, previous_state}).sigmoid();

    // the new state interploated between candidate and old:
    auto new_state = (
        previous_state      * (1.0 - update_gate) +
        candidate_new_state * update_gate
    );
    return new_state;
}

template<typename R>
Mat<R> GRU<R>::activate_sequence(const vector<Mat<R>>& input_sequence) const {
    return activate_sequence(input_sequence, initial_states());
}

template<typename R>
Mat<R> GRU<R>::activate_sequence(const vector<Mat<R>>& input_sequence, Mat<R> state) const {
    for (auto& input: input_sequence) {
        state = activate(input, state);
    }
    return state;
}

template<typename R>
std::vector<Mat<R>> GRU<R>::parameters() const {
    auto params = reset_layer.parameters();
    auto memory_interpolation_layer_params = memory_interpolation_layer.parameters();
    params.insert(
        params.end(),
        memory_interpolation_layer_params.begin(),
        memory_interpolation_layer_params.end()
    );
    auto memory_to_memory_layer_params = memory_to_memory_layer.parameters();
    params.insert(
        params.end(),
        memory_to_memory_layer_params.begin(),
        memory_to_memory_layer_params.end()
    );
    return params;
}

template<typename R>
Mat<R> GRU<R>::initial_states() const {
    return Mat<R>(hidden_size, 1);
}

template class GRU<float>;
template class GRU<double>;
