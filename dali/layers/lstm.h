#ifndef DALI_LAYERS_LSTM_H
#define DALI_LAYERS_LSTM_H

#include "dali/layers/layers.h"
#include "dali/array/memory/device.h"
#include "dali/array/dtype.h"
#include "dali/runtime_config.h"
#include "dali/tensor/tensor.h"

struct LSTMState {
    Tensor memory;
    Tensor hidden;
    LSTMState() = default;
    LSTMState(Tensor _memory, Tensor _hidden);
    static std::vector<Tensor> hiddens (const std::vector<LSTMState>&);
    static std::vector<Tensor> memories (const std::vector<LSTMState>&);
    operator std::tuple<Tensor &, Tensor &>();
};

class LSTM : public AbstractLayer {
    /*
    LSTM layer with forget, output, memory write, and input
    modulate gates, that can remember sequences for long
    periods of time.

    See `Layers.h`
    */
    typedef StackedInputLayer layer_type;

    public:
        void name_internal_layers();

        typedef LSTMState activation_t;

        // each child's memory to write controller for memory:
        std::vector<Tensor> Wcells_to_inputs;
        // each child's memory to forget gate for memory:
        std::vector<Tensor> Wcells_to_forgets;
        // memory to output gate
        Tensor Wco;

        // cell input modulation:
        layer_type input_layer;
        // cell forget gate:
        std::vector<layer_type> forget_layers;
        // cell output modulation
        layer_type output_layer;
        // cell write params
        layer_type cell_layer;

        int hidden_size;
        std::vector<int> input_sizes;
        int num_children;

        bool memory_feeds_gates;

        // In Alex Graves' slides / comments online you do not
        // backpropagate through memory cells at the gatestype
        // this is a boolean, so you can retrieve the true
        // gradient by setting this to true:
        bool backprop_through_gates = false;

        LSTM() = default;

        // This is a regular vanilla, but awesome LSTM constructor.
        LSTM (int _input_size,
              int _hidden_size,
              bool _memory_feeds_gates = false,
              DType dtype=DTYPE_FLOAT,
              memory::Device device=memory::default_preferred_device);

        // This constructors purpose is to create a tree LSTM.
        LSTM (int _input_size,
              int _hidden_size,
              int num_children,
              bool _memory_feeds_gates = false,
              DType dtype=DTYPE_FLOAT,
              memory::Device device=memory::default_preferred_device);

        // This constructor is generally intended to support shortcut LSTM. It also
        // happens to be the most general constructor available.
        LSTM (std::vector<int> _input_sizes,
              int _hidden_size,
              int num_children,
              bool _memory_feeds_gates = false,
              DType dtype=DTYPE_FLOAT,
              memory::Device device=memory::default_preferred_device);

        LSTM (const LSTM&, bool, bool);

        virtual std::vector<Tensor> parameters() const;

        activation_t activate(
            Tensor input_vector,
            activation_t previous_state) const;

        activation_t activate(
            Tensor input_vector,
            std::vector<activation_t> previous_children_states) const;

        activation_t activate(const std::vector<Tensor>&, const std::vector<activation_t>&) const;

        activation_t activate_shortcut(
            Tensor input_vector,
            Tensor shortcut_vector,
            activation_t prev_state) const;

        LSTM shallow_copy() const;

        activation_t initial_states() const;

        virtual activation_t activate_sequence(
            activation_t initial_state,
            const std::vector<Tensor>& sequence) const;
};

class AbstractStackedLSTM : public AbstractLayer {
    public:
        typedef std::vector <LSTM::activation_t > state_t;
        AbstractStackedLSTM() = default;
        virtual state_t initial_states() const = 0;
        virtual std::vector<Tensor> parameters() const = 0;
        virtual state_t activate(
            state_t previous_state,
            Tensor input_vector,
            const double drop_prob = 0.0) const = 0;
        virtual state_t activate_sequence(
            state_t initial_state,
            const std::vector<Tensor>& sequence,
            const double drop_prob = 0.0) const;
};

class StackedLSTM : public AbstractStackedLSTM {
    public:
        typedef LSTM lstm_t;
        typedef std::vector<LSTM::activation_t > state_t;
        bool shortcut;
        bool memory_feeds_gates;

        virtual state_t initial_states() const;

        std::vector<lstm_t> cells;
        virtual state_t activate(
            state_t previous_state,
            Tensor input_vector,
            const double drop_prob = 0.0) const;
        virtual state_t activate(
            state_t previous_state,
            const std::vector<Tensor>& inputs,
            const double drop_prob = 0.0) const;
        virtual std::vector<Tensor> parameters() const;
        StackedLSTM();
        StackedLSTM(
            const int& input_size,
            const std::vector<int>& hidden_sizes,
            bool _shortcut,
            bool _memory_feeds_gates,
            DType dtype=DTYPE_FLOAT,
            memory::Device device=memory::default_preferred_device);
        StackedLSTM(
            const std::vector<int>& input_sizes,
            const std::vector<int>& hidden_sizes,
            bool _shortcut,
            bool _memory_feeds_gates,
            DType dtype=DTYPE_FLOAT,
            memory::Device device=memory::default_preferred_device);

        StackedLSTM(const StackedLSTM& model, bool copy_w, bool copy_dw);
        StackedLSTM shallow_copy() const;
};

/**
StackedCells specialization to StackedLSTM
------------------------------------------

Static method StackedCells helps construct several
LSTMs that will be piled up. In a ShortcutLSTM scenario
the input is provided to all layers not just the
bottommost layer, so a new construction parameter
is provided to this **special** LSTM, the "shorcut size",
e.g. the size of the second input vector coming from farther
below (taking a shortcut or skip connection upwards).

Inputs
------

const int& input_size : size of the input at the basest layer
const vector<int>& hidden_sizes : dimensions of hidden states
                                  at each stack level.

Outputs
-------

vector<ShortcutLSTM> cells : constructed shortcutLSTMs

**/
template<typename celltype>
std::vector<celltype> stacked_cells(
    const int& input_size,
    const std::vector<int>& hidden_sizes,
    bool shortcut,
    bool memory_feeds_gates,
    DType dtype=DTYPE_FLOAT,
    memory::Device device=memory::default_preferred_device);

template<typename celltype>
std::vector<celltype> stacked_cells(
    const std::vector<int>& input_sizes,
    const std::vector<int>& hidden_sizes,
    bool shortcut,
    bool memory_feeds_gates,
    DType dtype=DTYPE_FLOAT,
    memory::Device device=memory::default_preferred_device);

template<typename celltype>
std::vector<celltype> stacked_cells(const std::vector<celltype>&, bool, bool);

std::vector<LSTM::activation_t > forward_LSTMs(
    Tensor input,
    std::vector<LSTM::activation_t >&,
    const std::vector<LSTM>&,
    const double drop_prob=0.0);

std::vector<LSTM::activation_t > forward_LSTMs(
    const std::vector<Tensor>& inputs,
    std::vector<LSTM::activation_t >&,
    const std::vector<LSTM>&,
    const double drop_prob=0.0);

std::vector<LSTM::activation_t > shortcut_forward_LSTMs(
    Tensor input,
    std::vector<LSTM::activation_t >&,
    const std::vector<LSTM>&,
    const double drop_prob=0.0);

std::vector<LSTM::activation_t > shortcut_forward_LSTMs(
    const std::vector<Tensor>& inputs,
    std::vector<LSTM::activation_t >&,
    const std::vector<LSTM>&,
    const double drop_prob=0.0);

#endif  // DALI_LAYERS_LSTM_H
