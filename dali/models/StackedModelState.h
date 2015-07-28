#ifndef DALI_MODELS_STACKED_MODEL_STATE_H
#define DALI_MODELS_STACKED_MODEL_STATE_H

#include <vector>
#include "dali/tensor/Mat.h"
#include "dali/layers/LSTM.h"

template<typename R>
struct StackedModelState {
    typedef std::vector<typename LSTM<R>::State> state_type;

    state_type lstm_state;
    Mat<R> prediction;
    Mat<R> memory;
    StackedModelState() = default;
    StackedModelState(const state_type&, Mat<R> prediction, Mat<R> memory);
    operator std::tuple<state_type&, Mat<R>&, Mat<R>&>();
};

#endif
