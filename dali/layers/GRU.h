#ifndef CORE_GRU_H
#define CORE_GRU_H

#include <vector>
#include "dali/tensor/Mat.h"
#include "dali/layers/Layers.h"

template<typename R>
class GRU {
    typedef StackedInputLayer<R> layer_type;
    public:
        int input_size;
        int hidden_size;
        layer_type reset_layer;
        layer_type memory_interpolation_layer;
        layer_type memory_to_memory_layer;
        typedef Mat<R> activation_t;

        GRU();

        GRU(int _input_size, int _hidden_size);

        GRU(const GRU<R>& other, bool copy_w, bool copy_dw);

        GRU<R> shallow_copy() const;

        Mat<R> activate(Mat<R> input_vector, Mat<R> previous_state) const;

        Mat<R> activate_sequence(const std::vector<Mat<R>>& input_sequence) const;

        Mat<R> activate_sequence(const std::vector<Mat<R>>& input_sequence, Mat<R> initial_state) const;

        std::vector<Mat<R>> parameters() const;

        Mat<R> initial_states() const;
};

#endif
