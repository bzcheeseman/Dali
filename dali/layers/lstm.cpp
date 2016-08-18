#include "dali/layers/lstm.h"

#include "dali/tensor/op.h"
#include "dali/utils/concatenate.h"

using std::vector;
using utils::assert2;
using std::string;
using std::make_shared;

///////////////////////////// LSTM STATE /////////////////////////////////////////////

LSTMState::LSTMState(Tensor _memory, Tensor _hidden) : memory(_memory), hidden(_hidden) {}

LSTMState::operator std::tuple<Tensor &, Tensor &>() {
    return std::tuple<Tensor&, Tensor&>(memory, hidden);
}

vector<Tensor> LSTMState::hiddens( const vector<LSTMState>& states) {
    vector<Tensor> hiddens;
    std::transform(
        states.begin(),
        states.end(),
        std::back_inserter(hiddens), [](const LSTMState& s) {
            return s.hidden;
        }
    );
    return hiddens;
}

vector<Tensor> LSTMState::memories( const vector<LSTMState>& states) {
    vector<Tensor> memories;
    std::transform(
        states.begin(),
        states.end(),
        std::back_inserter(memories), [](const LSTMState& s) {
            return s.memory;
        }
    );
    return memories;
}


///////////////////////////// LSTM /////////////////////////////////////////////////////

LSTM::LSTM(int _input_size,
           int _hidden_size,
           bool _memory_feeds_gates,
           DType dtype,
           memory::Device device) :
    LSTM(vector<int>({_input_size}), _hidden_size, 1,
         _memory_feeds_gates, dtype, device) {
}

LSTM::LSTM(int _input_size,
           int _hidden_size,
           int _num_children,
           bool _memory_feeds_gates,
           DType dtype,
           memory::Device device) :
    LSTM(vector<int> {_input_size}, _hidden_size, _num_children,
         _memory_feeds_gates, dtype, device) {
}

LSTM::LSTM (vector<int> _input_sizes,
            int _hidden_size,
            int _num_children,
            bool _memory_feeds_gates,
            DType dtype,
            memory::Device device) :
        memory_feeds_gates(_memory_feeds_gates),
        input_sizes(_input_sizes),
        hidden_size(_hidden_size),
        num_children(_num_children),
        AbstractLayer(dtype, device) {

    auto gate_input_sizes = utils::concatenate({
        input_sizes,
        vector<int>(num_children, hidden_size) // num_children * [hidden_size]
    });

    input_layer = StackedInputLayer(gate_input_sizes, hidden_size, dtype, device);
    for (int cidx=0; cidx < num_children; ++cidx) {
        forget_layers.emplace_back(gate_input_sizes, hidden_size, dtype, device);
    }
    output_layer = StackedInputLayer(gate_input_sizes, hidden_size, dtype, device);
    cell_layer   = StackedInputLayer(gate_input_sizes, hidden_size, dtype, device);

    if (memory_feeds_gates) {
        Wco = Tensor::uniform(
            1. / sqrt(hidden_size),
            {hidden_size},
            dtype,
            device
        )[Broadcast()];
        for (int cidx=0; cidx < num_children; ++cidx) {
            Wcells_to_forgets.emplace_back(
                Tensor::uniform(1. / sqrt(hidden_size), {hidden_size}, dtype, device)[Broadcast()]
            );
            Wcells_to_inputs.emplace_back(
                Tensor::uniform(1. / sqrt(hidden_size), {hidden_size}, dtype, device)[Broadcast()]
            );
        }
    }
    // Note: Ilya Sutskever recommends initializing with
    // forget gate at high value
    // http://yyue.blogspot.fr/2015/01/a-brief-overview-of-deep-learning.html
    // forget_layer.b.w().array() += 2;
}

LSTM::LSTM (const LSTM& other, bool copy_w, bool copy_dw) :
        memory_feeds_gates(other.memory_feeds_gates),
        input_sizes(other.input_sizes),
        hidden_size(other.hidden_size),
        num_children(other.num_children),
        AbstractLayer(other.dtype, other.device) {

    input_layer = StackedInputLayer(other.input_layer, copy_w, copy_dw);
    for (int cidx=0; cidx < num_children; ++cidx) {
        forget_layers.emplace_back(other.forget_layers[cidx], copy_w, copy_dw);
    }
    output_layer = StackedInputLayer(other.output_layer, copy_w, copy_dw);
    cell_layer = StackedInputLayer(other.cell_layer, copy_w, copy_dw);

    if (memory_feeds_gates) {
        Wco = Tensor(other.Wco, copy_w, copy_dw);
        for (int cidx=0; cidx < num_children; ++cidx) {
            Wcells_to_forgets.emplace_back(other.Wcells_to_forgets[cidx], copy_w, copy_dw);
            Wcells_to_inputs.emplace_back(other.Wcells_to_inputs[cidx],   copy_w, copy_dw);
        }
    }
}

LSTM LSTM::shallow_copy() const {
    return LSTM(*this, false, true);
}

void LSTM::name_internal_layers() {
    int i = 0;
    for (auto& cell_to_input : Wcells_to_inputs) {
        cell_to_input.name = make_shared<string>("Wcells_to_inputs[" + std::to_string(i++) + "]");
    }
    i = 0;
    for (auto& cell_to_forget : Wcells_to_forgets) {
        cell_to_forget.name = make_shared<string>("Wcells_to_forgets[" + std::to_string(i++) + "]");
    }
    Wco.name = make_shared<string>("Wco");
    i = 0;
    for (auto& mat : input_layer.tensors) {
        mat.name =  make_shared<string>("input_layer.tensors[" + std::to_string(i++) + "]");
    }
    input_layer.b.name = make_shared<string>("input_layer.b");

    i = 0;
    for (auto& mat : cell_layer.tensors) {
        mat.name =  make_shared<string>("cell_layer.tensors[" + std::to_string(i++) + "]");
    }
    cell_layer.b.name = make_shared<string>("cell_layer.b");

    i = 0;
    for (auto& mat : output_layer.tensors) {
        mat.name =  make_shared<string>("output_layer.tensors[" + std::to_string(i++) + "]");
    }
    output_layer.b.name = make_shared<string>("output_layer.b");

    int l = 0;
    for (auto& forget_layer : forget_layers) {
        i = 0;
        for (auto& mat : forget_layer.tensors) {
            mat.name =  make_shared<string>("forget_layers[" + std::to_string(l) + "].tensors[" + std::to_string(i++) + "]");
        }
        forget_layer.b.name = make_shared<string>("forget_layers[" + std::to_string(l) + "].b");
        l++;
    }
}

LSTM::activation_t LSTM::activate(
        const vector<Tensor>& inputs,
        const vector<activation_t>& states) const {
    Tensor input_gate, output_gate;
    vector<Tensor> forget_gates;

    for (auto& state: states) {
        ASSERT2(state.memory.ndim() == 2,
            utils::MS() << "LSTM: State memory should have ndim = 2 (got "
                        << state.memory.ndim() << ").");
        ASSERT2(state.hidden.ndim() == 2,
            utils::MS() << "LSTM: State memory should have ndim = 2 (got "
                        << state.memory.ndim() << ").");
        ASSERT2(state.memory.shape()[1] == hidden_size,
            utils::MS() << "LSTM: State memory should have hidden size "
                        << hidden_size << " not " << state.memory.shape()[1]);
        ASSERT2(state.hidden.shape()[1] == hidden_size,
            utils::MS() << "LSTM: State hidden should have hidden size "
                        << hidden_size << " not " << state.memory.shape()[1]);
    }
    ASSERT2(input_sizes.size() == inputs.size(),
        utils::MS() << "LSTM: Got " << inputs.size() << " inputs but expected " << input_sizes.size() << " instead."
    );
    for (int iidx = 0; iidx < input_sizes.size(); ++iidx) {
        ASSERT2(inputs[iidx].ndim() == 2,
            utils::MS() << "LSTM: " << iidx
                        << "-th input to LSTM should have ndim = 2 (got "
                        << inputs[iidx].ndim() << ").");
        ASSERT2(inputs[iidx].ndim() == 2 && inputs[iidx].shape()[1] == input_sizes[iidx],
                utils::MS() << "LSTM: " << iidx << "-th input to LSTM should have size "
                            << input_sizes[iidx] << " not " << inputs[iidx].shape()[1]);
    }
    auto gate_input = utils::concatenate({inputs, activation_t::hiddens(states)});

    if (memory_feeds_gates) {
        input_gate  = input_layer.activate(gate_input);
        // if the memory feeds the gates (Alex Graves 2013) then
        // a diagonal matrices (Wci and Wcf) connect memory to input
        // and forget gates
        for (int cidx = 0; cidx < num_children; ++cidx) {
            auto constant_memory = tensor_ops::consider_constant_if(states[cidx].memory, !backprop_through_gates);
            input_gate           = input_gate + constant_memory * Wcells_to_inputs[cidx];
            forget_gates.emplace_back(
                (
                    forget_layers[cidx].activate(gate_input) + constant_memory * Wcells_to_forgets[cidx]
                ).sigmoid()
            );
        }
        input_gate  = input_gate.sigmoid();
    } else {
        // (Zaremba 2014 style)

        // input gate:
        input_gate  = input_layer.activate(gate_input).sigmoid();
        // forget gate
        for (int cidx = 0; cidx < num_children; ++cidx) {
            forget_gates.emplace_back(forget_layers[cidx].activate(gate_input).sigmoid());
        }
    }

    // write operation on cells
    auto cell_write  = cell_layer.activate(gate_input).tanh();

    // compute new cell activation
    vector<Tensor> memory_contributions;
    for (int cidx = 0; cidx < num_children; ++cidx) {
        memory_contributions.emplace_back(forget_gates[cidx] * states[cidx].memory);
    }

    auto retain_cell = tensor_ops::add(memory_contributions);
    auto write_cell  = input_gate  * cell_write; // what do we write to cell
    auto cell_d      = retain_cell + write_cell; // new cell contents

    if (memory_feeds_gates) {
        // output gate uses new memory (cell_d) to control its gate
        output_gate = (
            output_layer.activate(gate_input) + (tensor_ops::consider_constant_if(cell_d, !backprop_through_gates) * Wco)
        ).sigmoid();
    } else {
        // output gate
        output_gate = output_layer.activate(gate_input).sigmoid();
    }

    // compute hidden state as gated, saturated cell activations
    auto hidden_d = output_gate * cell_d.tanh();

    return activation_t(cell_d, hidden_d);
}

LSTM::activation_t LSTM::activate(
        Tensor input_vector,
        activation_t  state) const {
    return activate(
        vector<Tensor>({input_vector}),
        vector<activation_t>({state})
    );
}

LSTM::activation_t LSTM::activate(
        Tensor input_vector,
        vector<activation_t> previous_children_states) const {
    return activate(
        vector<Tensor>({input_vector}),
        previous_children_states
    );
}

LSTM::activation_t LSTM::activate_shortcut(
        Tensor input_vector,
        Tensor shortcut_vector,
        activation_t  state) const {
    return activate(
        vector<Tensor>({input_vector, shortcut_vector}),
        vector<activation_t>({state})
    );
}

LSTM::activation_t LSTM::activate_sequence(
        activation_t state,
        const vector<Tensor>& sequence) const {
    for (auto& input_vector : sequence)
        state = activate(
            input_vector,
            vector<activation_t>({state})
        );
    return state;
};

std::vector<Tensor> LSTM::parameters() const {
    std::vector<Tensor> parameters;

    if (memory_feeds_gates) {
        for (int cidx = 0; cidx < num_children; ++cidx) {
            parameters.emplace_back(Wcells_to_forgets[cidx]);
            parameters.emplace_back(Wcells_to_inputs[cidx]);
        }
        parameters.emplace_back(Wco);
    }

    auto input_layer_params  = input_layer.parameters();
    parameters.insert( parameters.end(), input_layer_params.begin(),  input_layer_params.end() );

    for (int cidx = 0; cidx < num_children; ++cidx) {
        auto forget_layer_params = forget_layers[cidx].parameters();
        parameters.insert( parameters.end(), forget_layer_params.begin(), forget_layer_params.end() );
    }

    auto output_layer_params = output_layer.parameters();
    parameters.insert( parameters.end(), output_layer_params.begin(), output_layer_params.end() );

    auto cell_layer_params   = cell_layer.parameters();
    parameters.insert( parameters.end(), cell_layer_params.begin(),   cell_layer_params.end() );

    return parameters;
}

LSTM::activation_t LSTM::initial_states() const {
    return activation_t(
        Tensor::zeros({hidden_size}, dtype, device)[Broadcast()],
        Tensor::zeros({hidden_size}, dtype, device)[Broadcast()]
    );
}
