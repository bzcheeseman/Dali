#ifndef RECURRENT_EMBEDDING_MODEL_MAT_H
#define RECURRENT_EMBEDDING_MODEL_MAT_H

#include "core/Mat.h"
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include "core/utils.h"
#include "core/Index.h"
#include "core/Layers.h"

template<typename T>
class RecurrentEmbeddingModel {
    public:
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        typedef std::map<std::string, std::vector<std::string>> config_t;

        int vocabulary_size;
        const int output_size;
        const int stack_size;
        const int input_size;
        std::vector<int> hidden_sizes;

        shared_mat embedding;
        virtual std::vector<int> reconstruct(Indexing::Index, int, int symbol_offset = 0) const = 0;
        virtual std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(Indexing::Index, utils::OntologyBranch::shared_branch, int) const = 0;

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
        /**
        Save Configuration
        ------------------
        Save model configuration as a text file with key value pairs.
        Values are vectors of string, and keys are known by the model.

        Input
        -----

        std::string fname : where to save the configuration

        **/

        typedef std::pair<std::vector<shared_mat>, std::vector<shared_mat>> state_type;
        virtual state_type initial_states() const;
        virtual void save_configuration(std::string fname) const;
        std::string reconstruct_string(Indexing::Index, const utils::Vocab&, int, int symbol_offset = 0) const;
        std::string reconstruct_lattice_string(Indexing::Index, utils::OntologyBranch::shared_branch, int) const;
        RecurrentEmbeddingModel(int _vocabulary_size, int _input_size, int _hidden_size, int _stack_size, int _output_size);
        RecurrentEmbeddingModel(int _vocabulary_size, int _input_size, const std::vector<int>& _hidden_sizes, int _output_size);
        RecurrentEmbeddingModel(const config_t&);
};

#endif
