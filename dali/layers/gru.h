#ifndef DALI_LAYERS_GRU_H
#define DALI_LAYERS_GRU_H

#include <vector>
#include "dali/tensor/tensor.h"
#include "dali/layers/layers.h"

class GRU : public AbstractLayer {
    typedef StackedInputLayer layer_type;

    public:
        int input_size;
        int hidden_size;
        StackedInputLayer reset_layer;
        StackedInputLayer memory_interpolation_layer;
        StackedInputLayer memory_to_memory_layer;
        typedef Tensor activation_t;

        GRU();

        GRU(int _input_size,
            int _hidden_size,
            DType dtype=DTYPE_FLOAT,
            memory::Device device=memory::default_preferred_device);

        GRU(const GRU& other, bool copy_w, bool copy_dw);

        GRU shallow_copy() const;

        Tensor activate(Tensor input_vector, Tensor previous_state) const;

        Tensor activate_sequence(const std::vector<Tensor>& input_sequence) const;

        Tensor activate_sequence(Tensor initial_state, const std::vector<Tensor>& input_sequence) const;

        std::vector<Tensor> parameters() const;

        Tensor initial_states() const;
};

#endif  // DALI_LAYERS_GRU_H
