#ifndef STACKED_MAT_H
#define STACKED_MAT_H

#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <initializer_list>

#include "CrossEntropy.h"
#include "Layers.h"
#include "Mat.h"
#include "Softmax.h"
#include "utils.h"
#include "core/RecurrentEmbeddingModel.h"

/**
StackedModel
-----------------

A Model for making sequence predictions using stacked LSTM cells.

The network uses an embedding layer, and can reconstruct a sequence.

The objective function is built using masked cross entropy (only certain
input channels collect error over small intervals).

**/

DECLARE_int32(stack_size);
DECLARE_int32(input_size);
DECLARE_int32(hidden);
DECLARE_double(decay_rate);
DECLARE_double(rho);


template<typename T>
class StackedModel : public RecurrentEmbeddingModel<T>  {
    typedef LSTM<T>                    lstm;
    typedef Layer<T>           classifier_t;
    typedef Mat<T>                      mat;
    typedef std::shared_ptr<mat> shared_mat;
    typedef Graph<T>                graph_t;
    typedef std::map<std::string, std::vector<std::string>> config_t;

    inline void name_parameters();

    public:

        typedef std::pair<std::vector<shared_mat>, std::vector<shared_mat>> state_type;
        typedef std::pair<state_type, shared_mat > activation_t;
        typedef T value_t;
        const bool use_shortcut;

        std::shared_ptr<AbstractStackedLSTM<T>> stacked_lstm;

        typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
        typedef std::shared_ptr< index_mat > shared_index_mat;

        const classifier_t decoder;
        virtual std::vector<shared_mat> parameters() const;
        /**
        Load
        ----

        Load a saved copy of this model from a directory containing the
        configuration file named "config.md", and from ".npy" saves of
        the model parameters in the same directory.

        Inputs
        ------

        std::string dirname : directory where the model is currently saved

        Outputs
        -------

        StackedModel<T> model : the saved model

        **/
        /**
        StackedModel Constructor from configuration map
        ----------------------------------------------------

        Construct a model from a map of configuration parameters.
        Useful for reinitializing a model that was saved to a file
        using the `utils::file_to_map` function to obtain a map of
        configurations.

        **/
        static StackedModel<T> load(std::string);

        static StackedModel<T> build_from_CLI(std::string load_location,
                                              int vocab_size,
                                              int output_size,
                                              bool verbose = true);

        StackedModel(int, int, int, int, int, bool use_shortcut = false);
        StackedModel(int, int, int, std::vector<int>&, bool use_shortcut = false);
        /**StackedModel Constructor from configuration map
        ----------------------------------------------------

        Construct a model from a map of configuration parameters.
        Useful for reinitializing a model that was saved to a file
        using the `utils::file_to_map` function to obtain a map of
        configurations.

        Inputs
        ------

        std::map<std::string, std::vector<std::string>& config : model hyperparameters

        **/
        virtual config_t configuration() const;
        StackedModel(const config_t&);
        /**
        StackedModel<T>::StackedModel
        -----------------------------

        Copy constructor with option to make a shallow
        or deep copy of the underlying parameters.

        If the copy is shallow then the parameters are shared
        but separate gradients `dw` are used for each of
        thread StackedModel<T>.

        Shallow copies are useful for Hogwild and multithreaded
        training

        See `Mat<T>::shallow_copy`, `examples/character_prediction.cpp`,
        `StackedModel<T>::shallow_copy`

        Inputs
        ------

              StackedModel<T> l : StackedModel from which to source parameters and dw
                    bool copy_w : whether parameters for new StackedModel should be copies
                                  or shared
                   bool copy_dw : whether gradients for new StackedModel should be copies
                                  shared (Note: sharing `dw` should be used with
                                  caution and can lead to unpredictable behavior
                                  during optimization).

        Outputs
        -------

        StackedModel<T> out : the copied StackedModel with deep or shallow copy of parameters

        **/
        StackedModel(const StackedModel<T>&, bool, bool);
        T masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, shared_eigen_index_vector, shared_eigen_index_vector, uint offset=0, T drop_prob = 0.0);
        T masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, uint, shared_eigen_index_vector, uint offset=0, T drop_prob = 0.0);

        virtual std::vector<int> reconstruct(Indexing::Index, int, int symbol_offset = 0) const;

        state_type get_final_activation(graph_t&, Indexing::Index, T drop_prob=0.0) const;
        /**
        Activate
        --------

        Run Stacked Model by 1 timestep by observing
        the element from embedding with index `index`
        and report the activation, cell, and hidden
        states

        Inputs
        ------

        Graph<T>& G : computation graph
        std::pair<std::vector<std::shared_ptr<Mat<T>>>, std::vector<std::shared_ptr<Mat<T>>>>& : previous state
        uint index : embedding observation

        Outputs
        -------

        std::pair<std::pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>>, shared_ptr<Mat<T>> > out :
            pair of LSTM hidden and cell states, and probabilities from the decoder.

        **/
        activation_t activate(graph_t&, state_type&, const uint& ) const;
        activation_t activate(graph_t&, state_type&, const eigen_index_block ) const;

        virtual std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(Indexing::Index, utils::OntologyBranch::shared_branch, int) const;

        /**
        Shallow Copy
        ------------

        Perform a shallow copy of a StackedModel<T> that has
        the same parameters but separate gradients `dw`
        for each of its parameters.

        Shallow copies are useful for Hogwild and multithreaded
        training

        See `StackedModel<T>::shallow_copy`, `examples/character_prediction.cpp`.

        Outputs
        -------

        StackedModel<T> out : the copied layer with sharing parameters,
                                   but with separate gradients `dw`

        **/
        StackedModel<T> shallow_copy() const;
};

#endif
