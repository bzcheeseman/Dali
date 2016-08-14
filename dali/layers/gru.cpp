#include "gru.h"

#include "dali/tensor/op.h"

using std::vector;

GRU::GRU() : GRU(0,0) {}

GRU::GRU(int _input_size,
         int _hidden_size,
         DType dtype,
         memory::Device device) :
    AbstractLayer(dtype, device),
    input_size(_input_size),
    hidden_size(_hidden_size),
    reset_layer({_input_size, _hidden_size}, _hidden_size, dtype, device),
    memory_interpolation_layer({_input_size, _hidden_size}, _hidden_size, dtype, device),
    memory_to_memory_layer({_input_size, _hidden_size}, _hidden_size, dtype, device) {}

GRU::GRU(const GRU& other, bool copy_w, bool copy_dw) :
    input_size(other.input_size),
    hidden_size(other.hidden_size),
    reset_layer(other.reset_layer, copy_w, copy_dw),
    memory_interpolation_layer(other.memory_interpolation_layer, copy_w, copy_dw),
    memory_to_memory_layer(other.memory_to_memory_layer, copy_w, copy_dw),
    AbstractLayer(other.dtype, other.device) {}

GRU GRU::shallow_copy() const {
    return GRU(*this, false, true);
}

Tensor GRU::activate(
        Tensor input_vector,
        Tensor previous_state) const {

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

Tensor GRU::activate_sequence(const vector<Tensor>& input_sequence) const {
    return activate_sequence(initial_states(), input_sequence);
}

Tensor GRU::activate_sequence(Tensor state, const vector<Tensor>& input_sequence) const {
    for (auto& input: input_sequence) {
        state = activate(input, state);
    }
    return state;
}

std::vector<Tensor> GRU::parameters() const {
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

Tensor GRU::initial_states() const {
    return Tensor::zeros({hidden_size}, dtype, device)[Broadcast()];
}
