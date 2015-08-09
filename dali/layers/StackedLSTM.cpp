#include "dali/layers/LSTM.h"

using std::vector;
using utils::assert2;

/** Abstract Stacked LSTM **/

template<typename R>
std::vector<typename LSTM<R>::activation_t> StackedLSTM<R>::initial_states() const {
    state_t init_states;
    init_states.reserve(cells.size());
    for (const auto& cell : cells) {
        init_states.emplace_back(cell.initial_states());
    }
    return init_states;
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

/** Stacked LSTM **/

template<typename R>
StackedLSTM<R>::StackedLSTM(
    const int& input_size,
    const std::vector<int>& hidden_sizes,
    bool _shortcut,
    bool _memory_feeds_gates) : shortcut(_shortcut), memory_feeds_gates(_memory_feeds_gates) {
    cells = StackedCells<lstm_t>(input_size, hidden_sizes, shortcut, memory_feeds_gates);
};

template<typename R>
StackedLSTM<R>::StackedLSTM(
    const std::vector<int>& input_sizes,
    const std::vector<int>& hidden_sizes,
    bool _shortcut,
    bool _memory_feeds_gates) : shortcut(_shortcut), memory_feeds_gates(_memory_feeds_gates) {
    cells = StackedCells<lstm_t>(input_sizes, hidden_sizes, shortcut, memory_feeds_gates);
};

template<typename R>
StackedLSTM<R>::StackedLSTM() : shortcut(false),
                                memory_feeds_gates(false) {
}

template<typename R>
StackedLSTM<R>::StackedLSTM(const StackedLSTM<R>& model, bool copy_w, bool copy_dw) :
         shortcut(model.shortcut), memory_feeds_gates(model.memory_feeds_gates) {
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
    if (shortcut) {
        return shortcut_forward_LSTMs(input_vector, previous_state, cells, drop_prob);
    } else {
        return forward_LSTMs(input_vector, previous_state, cells, drop_prob);
    }
};

template<typename R>
typename StackedLSTM<R>::state_t StackedLSTM<R>::activate(
            state_t previous_state,
            const std::vector<Mat<R>>& inputs,
            R drop_prob) const {

    if (shortcut) {
        return shortcut_forward_LSTMs(inputs, previous_state, cells, drop_prob);
    } else {
        return forward_LSTMs(inputs, previous_state, cells, drop_prob);
    }
};

template<typename celltype>
vector<celltype> StackedCells(
        const int& input_size,
        const vector<int>& hidden_sizes,
        bool shortcut,
        bool memory_feeds_gates) {
    return StackedCells<celltype>(
        std::vector<int>({input_size}),
        hidden_sizes,
        shortcut,
        memory_feeds_gates
    );
}

// TODO: make this a static method of StackedLSTM
// since this class is only customer to stackedcells
template<typename celltype>
vector<celltype> StackedCells(
    const vector<int>& input_sizes,
    const vector<int>& hidden_sizes,
    bool shortcut,
    bool memory_feeds_gates) {
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
                    memory_feeds_gates);
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
                    memory_feeds_gates);
            }
        } else {
            if (i == 0) {
                cells.emplace_back(
                    input_sizes,
                    hidden_size,
                    1,
                    memory_feeds_gates);
            } else {
                cells.emplace_back(
                    prev_size,
                    hidden_size,
                    1,
                    memory_feeds_gates);
            }
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
std::vector< typename LSTM<R>::activation_t > forward_LSTMs(
        Mat<R> base_input,
        std::vector< typename LSTM<R>::activation_t >& previous_state,
        const vector<LSTM<R>>& cells,
        R drop_prob) {

    std::vector< typename LSTM<R>::activation_t> out_state;
    out_state.reserve(cells.size());

    auto layer_input = base_input;
    auto state_iter = previous_state.begin();
    for (auto& layer : cells) {
        out_state.emplace_back(
            layer.activate(
                MatOps<R>::dropout_normalized(layer_input, drop_prob),
                *state_iter
            )
        );
        ++state_iter;
        layer_input = out_state.back().hidden;
    }
    return out_state;
}

template<typename R>
std::vector< typename LSTM<R>::activation_t > shortcut_forward_LSTMs(
        Mat<R> base_input,
        std::vector< typename LSTM<R>::activation_t >& previous_state,
        const vector<LSTM<R>>& cells,
        R drop_prob) {

    std::vector< typename LSTM<R>::activation_t> out_state;
    out_state.reserve(cells.size());

    auto layer_input = base_input;
    auto state_iter = previous_state.begin();
    int level = 0;

    for (auto& layer : cells) {
        if (level == 0) {
            out_state.emplace_back(
                layer.activate(
                    MatOps<R>::dropout_normalized(layer_input, drop_prob),
                    *state_iter
                )
            );
        } else {
            out_state.emplace_back(
                layer.activate_shortcut(
                    MatOps<R>::dropout_normalized(layer_input, drop_prob),
                    MatOps<R>::dropout_normalized(base_input, drop_prob),
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

template<typename R>
std::vector< typename LSTM<R>::activation_t > forward_LSTMs(
    const std::vector<Mat<R>>& inputs,
    std::vector< typename LSTM<R>::activation_t >& previous_state,
    const std::vector<LSTM<R>>& cells,
    R drop_prob) {

    std::vector< typename LSTM<R>::activation_t> out_state;
    out_state.reserve(cells.size());

    Mat<R> layer_input;
    auto state_iter = previous_state.begin();

    int layer_idx = 0;
    for (auto& layer : cells) {
        if (layer_idx == 0) {
            out_state.emplace_back(
                layer.activate(
                    MatOps<R>::dropout_normalized(inputs, drop_prob),
                    {*state_iter}
                )
            );
        } else {
            out_state.emplace_back(
                layer.activate(
                    MatOps<R>::dropout_normalized(out_state.back().hidden, drop_prob),
                    *state_iter
                )
            );
        }
        ++state_iter;
        layer_idx++;
    }
    return out_state;
}

template<typename R>
std::vector< typename LSTM<R>::activation_t > shortcut_forward_LSTMs(
    const std::vector<Mat<R>>& inputs,
    std::vector< typename LSTM<R>::activation_t >& previous_state,
    const std::vector<LSTM<R>>& cells,
    R drop_prob) {

    std::vector< typename LSTM<R>::activation_t> out_state;
    out_state.reserve(cells.size());

    auto state_iter = previous_state.begin();
    int level = 0;

    // each layer above the first receive the base inputs +
    // hidden from layer below:
    vector<Mat<R>> layer_inputs(1);
    layer_inputs.insert(layer_inputs.end(), inputs.begin(), inputs.end());

    for (auto& layer : cells) {
        if (level == 0) {
            out_state.emplace_back(
                layer.activate(
                    MatOps<R>::dropout_normalized(inputs, drop_prob),
                    {*state_iter}
                )
            );
        } else {
            layer_inputs[0] = out_state.back().hidden;
            out_state.emplace_back(
                layer.activate(
                    MatOps<R>::dropout_normalized(layer_inputs, drop_prob),
                    {*state_iter}
                )
            );
        }
        ++state_iter;
        ++level;
    }
    return out_state;
}

template class StackedLSTM<float>;
template class StackedLSTM<double>;

