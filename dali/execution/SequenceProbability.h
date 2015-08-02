#ifndef SEQUENCE_PROBABILITY_MAT_H
#define SEQUENCE_PROBABILITY_MAT_H

#include <functional>
#include <vector>

#include "dali/data_processing/Batch.h"
#include "dali/tensor/Tape.h"
#include "dali/tensor/Mat.h"
namespace sequence_probability {

    template<typename R, typename state_t>
    Mat<R> sequence_score(
            Batch<R> batch,
            state_t state,
            std::function<Mat<R>(state_t)> decode,
            std::function<state_t(Mat<int>, state_t)> observe,
            int offset = 0) {
        int num_examples = batch.data.dims(1);
        int max_length   = batch.data.dims(0);

        if (max_length - offset == 0) {
            return Mat<R>(num_examples, 1);
        }

        graph::NoBackprop nb;

        Mat<R> result(num_examples, 1);

        // state
        for (int t = 0; t < max_length - offset; ++t) {
            state = observe(batch.data[t], state);
            // NUM_CLASSES X NUM EXAMPLES
            auto scores = decode(state);

            for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
                auto target_for_ex = batch.target.w(t + offset, example_idx);
                if (t < batch.example_length(example_idx))
                    result.w(example_idx, 0) += scores.w(target_for_ex, example_idx);
            }
        }

        return result;
    }

    // #define LOG1P(X) (((X) + 1).log())
    // #define SURPRISE(X) -(LOG1P(-(1 - (X).array()).sqrt()) - LOG1P((1 - (X).array()).sqrt()))

    // template<typename model_t, typename K, typename C>
    // Mat<typename model_t::value_t> sequence_surprises(
    //     const model_t& model,
    //     K& data,
    //     C& codelens) {
    //     typedef std::vector<uint> seq_type;
    //     typedef std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type > open_list_t;
    //     typedef Mat<typename model_t::value_t> mat;

    //     graph::NoBackprop nb;

    //     auto initial_state      = model.initial_states();
    //     auto out_state_and_prob = model.activate(initial_state, data.col(0));
    //     auto log_prob           = mat(data.rows(), data.cols()-1);

    //     int n = data.cols();

    //     auto data_pluck = Indexing::Index::arange(0, data.rows());

    //     MAT(log_prob).col(0) = SURPRISE( MAT(std::get<1>(out_state_and_prob)(data.col(1), data_pluck)).row(0).transpose() );

    //     for (int i = 1; i < n-1; i++) {
    //         out_state_and_prob = model.activate(
    //             std::get<0>(out_state_and_prob),
    //             data.col(i)
    //         );
    //         auto plucked_activations = std::get<1>(out_state_and_prob)(
    //             data.col(i+1),
    //             data_pluck
    //         );
    //         MAT(log_prob).col(i).array() += SURPRISE(MAT(plucked_activations).row(0).transpose().array()) + MAT(log_prob).col(i-1).array();
    //     }

    //     // inelegant manner of dealing with scalar corrections on codelens:
    //     auto relevant_log_prob = log_prob(
    //         data_pluck,
    //         (codelens.array() - (uint)1).matrix()
    //     );

    //     return relevant_log_prob;
    // }
}

#endif
