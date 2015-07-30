#include "dali/layers/LSTM.h"

using std::vector;
using utils::assert2;
using std::string;
using std::make_shared;

///////////////////////////// LSTM STATE /////////////////////////////////////////////

template<typename R>
LSTMState<R>::LSTMState(Mat<R> _memory, Mat<R> _hidden) : memory(_memory), hidden(_hidden) {}

template<typename R>
LSTMState<R>::operator std::tuple<Mat<R> &, Mat<R> &>() {
    return std::tuple<Mat<R>&, Mat<R>&>(memory, hidden);
}

template<typename R>
vector<Mat<R>> LSTMState<R>::hiddens( const vector<LSTMState<R>>& states) {
    vector<Mat<R>> hiddens;
    std::transform(
        states.begin(),
        states.end(),
        std::back_inserter(hiddens), [](const LSTMState<R>& s) {
            return s.hidden;
        }
    );
    return hiddens;
}

template<typename R>
vector<Mat<R>> LSTMState<R>::memories( const vector<LSTMState<R>>& states) {
    vector<Mat<R>> memories;
    std::transform(
        states.begin(),
        states.end(),
        std::back_inserter(memories), [](const LSTMState<R>& s) {
            return s.memory;
        }
    );
    return memories;
}

template class LSTMState<float>;
template class LSTMState<double>;

///////////////////////////// LSTM /////////////////////////////////////////////////////

template<typename R>
LSTM<R>::LSTM(int _input_size, int _hidden_size, bool _memory_feeds_gates) :
        LSTM<R>(vector<int>({_input_size}), _hidden_size, 1,_memory_feeds_gates) {
}

template<typename R>
LSTM<R>::LSTM(int _input_size, int _hidden_size, int _num_children, bool _memory_feeds_gates) :
        LSTM<R>(vector<int> {_input_size}, _hidden_size, _num_children,_memory_feeds_gates) {
}

template<typename R>
LSTM<R>::LSTM (vector<int> _input_sizes, int _hidden_size, int _num_children, bool _memory_feeds_gates) :
        memory_feeds_gates(_memory_feeds_gates),
        input_sizes(_input_sizes),
        hidden_size(_hidden_size),
        num_children(_num_children) {

    auto gate_input_sizes = utils::concatenate({
        input_sizes,
        vector<int>(num_children, hidden_size) // num_children * [hidden_size]
    });

    input_layer = StackedInputLayer<R>(gate_input_sizes, hidden_size);
    for (int cidx=0; cidx < num_children; ++cidx) {
        forget_layers.emplace_back(gate_input_sizes, hidden_size);
    }
    output_layer = StackedInputLayer<R>(gate_input_sizes, hidden_size);
    cell_layer   = StackedInputLayer<R>(gate_input_sizes, hidden_size);

    if (memory_feeds_gates) {

        Wco = Mat<R>(hidden_size, 1, weights<R>::uniform(2. / sqrt(hidden_size)));
        for (int cidx=0; cidx < num_children; ++cidx) {
            Wcells_to_forgets.emplace_back(1, hidden_size, weights<R>::uniform(2. / sqrt(hidden_size)));
            Wcells_to_inputs.emplace_back(1, hidden_size,  weights<R>::uniform(2. / sqrt(hidden_size)));
        }
    }
    // Note: Ilya Sutskever recommends initializing with
    // forget gate at high value
    // http://yyue.blogspot.fr/2015/01/a-brief-overview-of-deep-learning.html
    // forget_layer.b.w().array() += 2;
}

template<typename R>
LSTM<R>::LSTM (const LSTM<R>& other, bool copy_w, bool copy_dw) :
        memory_feeds_gates(other.memory_feeds_gates),
        input_sizes(other.input_sizes),
        hidden_size(other.hidden_size),
        num_children(other.num_children) {

    input_layer = StackedInputLayer<R>(other.input_layer, copy_w, copy_dw);
    for (int cidx=0; cidx < num_children; ++cidx) {
        forget_layers.emplace_back(other.forget_layers[cidx], copy_w, copy_dw);
    }
    output_layer = StackedInputLayer<R>(other.output_layer, copy_w, copy_dw);
    cell_layer = StackedInputLayer<R>(other.cell_layer, copy_w, copy_dw);

    if (memory_feeds_gates) {
        Wco = Mat<R>(other.Wco, copy_w, copy_dw);
        for (int cidx=0; cidx < num_children; ++cidx) {
            Wcells_to_forgets.emplace_back(other.Wcells_to_forgets[cidx], copy_w, copy_dw);
            Wcells_to_inputs.emplace_back(other.Wcells_to_inputs[cidx],   copy_w, copy_dw);
        }
    }
}

template<typename R>
LSTM<R> LSTM<R>::shallow_copy() const {
    return LSTM<R>(*this, false, true);
}

template<typename R>
void LSTM<R>::name_internal_layers() {
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
    for (auto& mat : input_layer.matrices) {
        mat.name =  make_shared<string>("input_layer.matrices[" + std::to_string(i++) + "]");
    }
    input_layer.b.name = make_shared<string>("input_layer.b");

    i = 0;
    for (auto& mat : cell_layer.matrices) {
        mat.name =  make_shared<string>("cell_layer.matrices[" + std::to_string(i++) + "]");
    }
    cell_layer.b.name = make_shared<string>("cell_layer.b");

    i = 0;
    for (auto& mat : output_layer.matrices) {
        mat.name =  make_shared<string>("output_layer.matrices[" + std::to_string(i++) + "]");
    }
    output_layer.b.name = make_shared<string>("output_layer.b");

    int l = 0;
    for (auto& forget_layer : forget_layers) {
        i = 0;
        for (auto& mat : forget_layer.matrices) {
            mat.name =  make_shared<string>("forget_layers[" + std::to_string(l) + "].matrices[" + std::to_string(i++) + "]");
        }
        forget_layer.b.name = make_shared<string>("forget_layers[" + std::to_string(l) + "].b");
        l++;
    }
}

template<typename R>
typename LSTM<R>::activation_t LSTM<R>::activate(
        const vector<Mat<R>>& inputs,
        const vector<activation_t>& states) const {

    Mat<R> input_gate, output_gate;
    vector<Mat<R>> forget_gates;

    for (auto& state: states) {
        assert2(state.memory.dims(1) == hidden_size,
            utils::MS() << "LSTM: State memory should have hidden size "
                        << hidden_size << " not " << state.memory.dims(1));
        assert2(state.hidden.dims(1) == hidden_size,
            utils::MS() << "LSTM: State hidden should have hidden size "
                        << hidden_size << " not " << state.memory.dims(1));
    }
    assert2(input_sizes.size() == inputs.size(),
        utils::MS() << "LSTM: Got " << inputs.size() << " inputs but expected " << input_sizes.size() << " instead."
    );
    for (int iidx = 0; iidx < input_sizes.size(); ++iidx) {
        assert2(inputs[iidx].dims(1) == input_sizes[iidx],
                utils::MS() << "LSTM: " << iidx << "-th input to LSTM should have size "
                            << input_sizes[iidx] << " not " << inputs[iidx].dims(1));
    }
    auto gate_input = utils::concatenate({inputs, activation_t::hiddens(states)});

    if (memory_feeds_gates) {
        input_gate  = input_layer.activate(gate_input);
        // if the memory feeds the gates (Alex Graves 2013) then
        // a diagonal matrices (Wci and Wcf) connect memory to input
        // and forget gates
        for (int cidx = 0; cidx < num_children; ++cidx) {
            auto constant_memory = MatOps<R>::consider_constant_if(states[cidx].memory, !backprop_through_gates);
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
    vector<Mat<R>> memory_contributions;
    for (int cidx = 0; cidx < num_children; ++cidx) {
        memory_contributions.emplace_back(forget_gates[cidx] * states[cidx].memory);
    }
    auto retain_cell = MatOps<R>::add(memory_contributions);
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

    return activation_t(cell_d, hidden_d);
}

template<typename R>
typename LSTM<R>::activation_t LSTM<R>::activate(
        Mat<R> input_vector,
        activation_t  state) const {
    return activate(
        vector<Mat<R>>({input_vector}),
        vector<activation_t>({state})
    );
}

template<typename R>
typename LSTM<R>::activation_t LSTM<R>::activate(
        Mat<R> input_vector,
        vector<activation_t> previous_children_states) const {
    return activate(
        vector<Mat<R>>({input_vector}),
        previous_children_states
    );
}

template<typename R>
typename LSTM<R>::activation_t LSTM<R>::activate_shortcut(
        Mat<R> input_vector,
        Mat<R> shortcut_vector,
        activation_t  state) const {
    return activate(
        vector<Mat<R>>({input_vector, shortcut_vector}),
        vector<activation_t>({state})
    );
}

template<typename R>
typename LSTM<R>::activation_t LSTM<R>::activate_sequence(
        activation_t state,
        const vector<Mat<R>>& sequence) const {
    for (auto& input_vector : sequence)
        state = activate(
            input_vector,
            vector<activation_t>({state})
        );
    return state;
};

template<typename R>
std::vector<Mat<R>> LSTM<R>::parameters() const {
    std::vector<Mat<R>> parameters;

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

template<typename R>
typename std::vector<typename LSTM<R>::activation_t> LSTM<R>::initial_states(
        const std::vector<int>& hidden_sizes) {
    std::vector<activation_t>  initial_state;
    initial_state.reserve(hidden_sizes.size());
    for (auto& size : hidden_sizes) {
        initial_state.emplace_back(Mat<R>(1, size), Mat<R>(1, size));
    }
    return initial_state;
}

template<typename R>
typename LSTM<R>::activation_t LSTM<R>::initial_states() const {
    return activation_t(Mat<R>(1, hidden_size), Mat<R>(1, hidden_size));
}

template class LSTM<float>;
template class LSTM<double>;

