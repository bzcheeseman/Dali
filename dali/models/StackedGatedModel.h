#ifndef CORE_STACKED_GATED_MAT_H
#define CORE_STACKED_GATED_MAT_H

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>

#include "dali/models/StackedModel.h"
#include "dali/data_processing/Batch.h"
#include "dali/core.h"
#include "dali/utils.h"

DECLARE_double(memory_penalty);

/**
StackedGatedModel
-----------------

A Model for making sequence predictions using stacked LSTM cells,
that constructs an embedding matrix as a convenience
and also passes the inputs through a gate for pre-filtering.

The input is gated using a sigmoid linear regression that takes
as input the last hidden cell's activation and the input to the network.

The network uses an embedding layer, and can reconstruct a sequence.

The objective function is built using masked cross entropy (only certain
input channels collect error over small intervals), and L1 loss on the
total memory used (the input gate's total activation).

**/


template<typename Z>
class StackedGatedModel : public StackedModel<Z> {
    typedef LSTM<Z>                                             lstm;
    typedef Layer<Z>                                    classifier_t;
    typedef GatedInput<Z>                                     gate_t;
    typedef std::map<std::string, std::vector<std::string>> config_t;

    public:
        typedef Mat<Z> mat;
        typedef std::vector< typename LSTM<Z>::activation_t > state_type;

        typedef StackedModelState<Z> State;

        typedef Z value_t;

        const gate_t gate;
        Z memory_penalty;
        virtual std::vector<mat> parameters() const;
        /**
        Configuration
        -------------

        Return a map with keys corresponding to hyperparameters for
        the model and where values are vectors of strings containing
        the assignments to each hyperparameter for the loaded model.

        Useful for saving the model to file and reloading it later.

        Outputs
        -------

        std::map<std::string, std::vector< std::string >> config : configuration map

        **/
        virtual config_t configuration() const;
        static StackedGatedModel<Z> load(std::string);
        static StackedGatedModel<Z> build_from_CLI(std::string load_location,
                                                   int vocab_size,
                                                   int output_size,
                                                   bool verbose);
        StackedGatedModel() = default;
        StackedGatedModel(
            int vocabulary_size,
            int input_size,
            int hidden_size,
            int stack_size,
            int output_size,
            bool use_shortcut,
            bool memory_feeds_gates,
            Z _memory_penalty);
        StackedGatedModel(
            int vocabulary_size,
            int input_size,
            int output_size,
            const std::vector<int>& hiddens_sizes,
            bool use_shortcut,
            bool memory_feeds_gates,
            Z _memory_penalty);
        StackedGatedModel(const config_t&);
        StackedGatedModel(const StackedGatedModel<Z>&, bool, bool);

        struct MaskedActivation {
            Mat<Z> prediction_error;
            Mat<Z> memory_error;
            MaskedActivation(Mat<Z> prediction_error, Mat<Z> memory_error);
            operator std::tuple<Mat<Z>&, Mat<Z>&>();
        };

        MaskedActivation masked_predict_cost(Mat<int> data,
                                   Mat<int> target_data,
                                   Mat<Z> prediction_mask,
                                   Z drop_prob = 0.0,
                                   int temporal_offset = 0,
                                   uint softmax_offset = 0) const;

        MaskedActivation masked_predict_cost(const Batch<Z>& data,
                                   Z drop_prob = 0.0,
                                   int temporal_offset = 0,
                                   uint softmax_offset = 0) const;

        virtual std::vector<int> reconstruct(Indexing::Index, int, int symbol_offset = 0) const;
        state_type get_final_activation(Indexing::Index, Z drop_prob=0.0) const;

        State activate(state_type&, const uint&) const;
        State activate(state_type&, const Indexing::Index) const;

        virtual std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(
            Indexing::Index,
            utils::OntologyBranch::shared_branch,
            int) const;

        StackedGatedModel<Z> shallow_copy() const;

};

#endif
