#include "dali/layers/lstm.h"

#include "dali/tensor/op.h"
#include "dali/utils/concatenate.h"
#include "dali/utils/fmap.h"


using std::vector;
using utils::assert2;


vector<Tensor> dropout_many_tensors(const std::vector<Tensor>& tensors, double dropprob) {
    return utils::fmap(tensors, [&](const Tensor& t) -> Tensor {
        return tensor_ops::dropout(t, dropprob);
    });
}

/** Abstract Stacked LSTM **/

std::vector<LSTM::activation_t> StackedLSTM::initial_states() const {
    state_t init_states;
    init_states.reserve(cells.size());
    for (const auto& cell : cells) {
        init_states.emplace_back(cell.initial_states());
    }
    return init_states;
}

std::vector<int> StackedLSTM::hidden_sizes() const {
    std::vector<int> sizes;
    for (const auto& cell : cells) {
        sizes.emplace_back(cell.hidden_size);
    }
    return sizes;
}

std::vector<int> StackedLSTM::input_sizes() const {
    if (cells.empty()) {
        return {};
    } else {
        return cells[0].input_sizes;
    }
}

typename AbstractStackedLSTM::state_t AbstractStackedLSTM::activate_sequence(
    state_t initial_state,
    const vector<Tensor>& sequence,
    const double drop_prob) const {
    for (auto& input_vector : sequence)
        initial_state = activate(input_vector, initial_state, drop_prob);
    return initial_state;
};

/** Stacked LSTM **/

StackedLSTM::StackedLSTM(
    const int& input_size,
    const std::vector<int>& hidden_sizes,
    bool _shortcut,
    bool memory_feeds_gates,
    DType dtype,
    memory::Device device) : shortcut(_shortcut) {
    cells = stacked_cells<lstm_t>(input_size, hidden_sizes, shortcut, memory_feeds_gates, dtype, device);
};

StackedLSTM::StackedLSTM(
    const std::vector<int>& input_sizes,
    const std::vector<int>& hidden_sizes,
    bool _shortcut,
    bool memory_feeds_gates,
    DType dtype,
    memory::Device device) : shortcut(_shortcut) {
    cells = stacked_cells<lstm_t>(input_sizes, hidden_sizes, shortcut, memory_feeds_gates, dtype, device);
};

StackedLSTM::StackedLSTM() : shortcut(false) {
}

StackedLSTM::StackedLSTM(const StackedLSTM& model, bool copy_w, bool copy_dw) :
         shortcut(model.shortcut) {
    cells = stacked_cells<lstm_t>(model.cells, copy_w, copy_dw);
};

StackedLSTM StackedLSTM::shallow_copy() const {
    return StackedLSTM(*this, false, true);
}

std::vector<Tensor> StackedLSTM::parameters() const {
    vector<Tensor> parameters;
    for (auto& cell : cells) {
        auto cell_params = cell.parameters();
        parameters.insert(parameters.end(), cell_params.begin(), cell_params.end());
    }
    return parameters;
}

typename StackedLSTM::state_t StackedLSTM::activate(
            Tensor input_vector,
            state_t previous_state,
            const double drop_prob) const {
    if (shortcut) {
        return shortcut_forward_LSTMs(input_vector, previous_state, cells, drop_prob);
    } else {
        return forward_LSTMs(input_vector, previous_state, cells, drop_prob);
    }
};

typename StackedLSTM::state_t StackedLSTM::activate(
            const std::vector<Tensor>& inputs,
            state_t previous_state,
            const double drop_prob) const {

    if (shortcut) {
        return shortcut_forward_LSTMs(inputs, previous_state, cells, drop_prob);
    } else {
        return forward_LSTMs(inputs, previous_state, cells, drop_prob);
    }
};

template<typename celltype>
vector<celltype> stacked_cells(
        const int& input_size,
        const vector<int>& hidden_sizes,
        bool shortcut,
        bool memory_feeds_gates,
        DType dtype,
        memory::Device device) {
    return stacked_cells<celltype>(
        std::vector<int>({input_size}),
        hidden_sizes,
        shortcut,
        memory_feeds_gates,
        dtype,
        device
    );
}

// TODO: make this a static method of StackedLSTM
// since this class is only customer to stackedcells
template<typename celltype>
vector<celltype> stacked_cells(
        const vector<int>& input_sizes,
        const vector<int>& hidden_sizes,
        bool shortcut,
        bool memory_feeds_gates,
        DType dtype,
        memory::Device device) {
    vector<celltype> cells;
    cells.reserve(hidden_sizes.size());
    int prev_size;
    int i = 0;
    for (auto& hidden_size : hidden_sizes) {
        if (shortcut) {
            if (i == 0) {
                // first cell in a shorcut
                // stack cannot "steal" or
                // shorcut from anywhere else
                // so no shorcut is used
                cells.emplace_back(
                    input_sizes,
                    hidden_size,
                    1,
                    memory_feeds_gates,
                    dtype,
                    device);
            } else {
                // other cells in a shorcut
                // stack steal the input
                // from the lower stack
                // input_size
                cells.emplace_back(
                    utils::concatenate({
                        std::vector<int>({prev_size}),
                        input_sizes
                    }),
                    hidden_size,
                    1,
                    memory_feeds_gates,
                    dtype,
                    device);
            }
        } else {
            if (i == 0) {
                cells.emplace_back(
                    input_sizes,
                    hidden_size,
                    1,
                    memory_feeds_gates,
                    dtype,
                    device);
            } else {
                cells.emplace_back(
                    prev_size,
                    hidden_size,
                    1,
                    memory_feeds_gates,
                    dtype,
                    device);
            }
        }
        prev_size = hidden_size;
        i++;
    }
    return cells;
}

template<typename celltype>
vector<celltype> stacked_cells(const vector<celltype>& source_cells,
                               bool copy_w,
                               bool copy_dw) {
    vector<celltype> cells;
    cells.reserve(source_cells.size());
    for (const auto& cell : source_cells)
        cells.emplace_back(cell, copy_w, copy_dw);
    return cells;
}

std::vector<LSTM::activation_t > forward_LSTMs(
        Tensor base_input,
        std::vector<LSTM::activation_t >& previous_state,
        const vector<LSTM>& cells,
        const double drop_prob) {

    ASSERT2(cells.size() == previous_state.size(),
        utils::MS() << "Activating LSTM stack of size " << cells.size()
        << " with different number of states " << previous_state.size());

    std::vector<LSTM::activation_t> out_state;
    out_state.reserve(cells.size());

    auto layer_input = base_input;
    auto state_iter = previous_state.begin();
    for (auto& layer : cells) {
        out_state.emplace_back(
            layer.activate(
                tensor_ops::dropout(layer_input, drop_prob),
                *state_iter
            )
        );
        ++state_iter;
        layer_input = out_state.back().hidden;
    }
    return out_state;
}

std::vector<LSTM::activation_t > shortcut_forward_LSTMs(
        Tensor base_input,
        std::vector<LSTM::activation_t >& previous_state,
        const vector<LSTM>& cells,
        const double drop_prob) {

    ASSERT2(cells.size() == previous_state.size(),
        utils::MS() << "Activating LSTM stack of size " << cells.size()
        << " with different number of states " << previous_state.size());

    std::vector<LSTM::activation_t> out_state;
    out_state.reserve(cells.size());

    auto layer_input = base_input;
    auto state_iter = previous_state.begin();
    int level = 0;

    for (auto& layer : cells) {
        if (level == 0) {
            out_state.emplace_back(
                layer.activate(
                    tensor_ops::dropout(layer_input, drop_prob),
                    *state_iter
                )
            );
        } else {
            out_state.emplace_back(
                layer.activate_shortcut(
                    tensor_ops::dropout(layer_input, drop_prob),
                    tensor_ops::dropout(base_input, drop_prob),
                    *state_iter
                )
            );
        }
        ++state_iter;
        ++level;
        layer_input = out_state.back().hidden;
    }
    return out_state;
}

std::vector<LSTM::activation_t > forward_LSTMs(
        const std::vector<Tensor>& inputs,
        std::vector<LSTM::activation_t >& previous_state,
        const std::vector<LSTM>& cells,
        const double drop_prob) {

    ASSERT2(cells.size() == previous_state.size(),
        utils::MS() << "Activating LSTM stack of size " << cells.size()
        << " with different number of states " << previous_state.size());

    std::vector<LSTM::activation_t> out_state;
    out_state.reserve(cells.size());

    Tensor layer_input;
    auto state_iter = previous_state.begin();

    int layer_idx = 0;
    for (auto& layer : cells) {
        if (layer_idx == 0) {
            out_state.emplace_back(
                layer.activate(
                    dropout_many_tensors(inputs, drop_prob),
                    {*state_iter}
                )
            );
        } else {
            out_state.emplace_back(
                layer.activate(
                    tensor_ops::dropout(out_state.back().hidden, drop_prob),
                    *state_iter
                )
            );
        }
        ++state_iter;
        layer_idx++;
    }
    return out_state;
}

std::vector<LSTM::activation_t > shortcut_forward_LSTMs(
        const std::vector<Tensor>& inputs,
        std::vector<LSTM::activation_t >& previous_state,
        const std::vector<LSTM>& cells,
        const double drop_prob) {

    ASSERT2(cells.size() == previous_state.size(),
        utils::MS() << "Activating LSTM stack of size " << cells.size()
        << " with different number of states " << previous_state.size());

    std::vector<LSTM::activation_t> out_state;
    out_state.reserve(cells.size());

    auto state_iter = previous_state.begin();
    int level = 0;

    // each layer above the first receive the base inputs +
    // hidden from layer below:
    vector<Tensor> layer_inputs(1);
    layer_inputs.insert(layer_inputs.end(), inputs.begin(), inputs.end());

    for (auto& layer : cells) {
        if (level == 0) {
            out_state.emplace_back(
                layer.activate(
                    dropout_many_tensors(inputs, drop_prob),
                    {*state_iter}
                )
            );
        } else {
            layer_inputs[0] = out_state.back().hidden;
            out_state.emplace_back(
                layer.activate(
                    dropout_many_tensors(layer_inputs, drop_prob),
                    {*state_iter}
                )
            );
        }
        ++state_iter;
        ++level;
    }
    return out_state;
}
