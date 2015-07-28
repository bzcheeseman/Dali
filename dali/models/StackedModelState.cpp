#include "StackedModelState.h"

template<typename R>
StackedModelState<R>::StackedModelState(
		const state_type& _lstm_state,
		Mat<R> _prediction,
		Mat<R> _memory) :
    prediction(_prediction), lstm_state(_lstm_state), memory(_memory) {}

template<typename R>
StackedModelState<R>::operator std::tuple<state_type&, Mat<R>&, Mat<R>&>() {
    return std::make_tuple(
        std::ref(lstm_state),
        std::ref(prediction),
        std::ref(memory)
    );
}

template struct StackedModelState<float>;
template struct StackedModelState<double>;
