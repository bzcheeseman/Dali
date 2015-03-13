#ifndef SST_MAT_H
#define SST_MAT_H

#include <memory>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "dali/utils.h"

/**
Stanford Sentiment Treebank
---------------------------

This namespace contains a dataset wrapper,
the AnnotatedParseTree, and its assocatiated
loading / printing methods. Useful for sentiment
and parse tree based corpuses.

See `examples/language_model_from_senti.cpp` for
usage.

**/
namespace SST {
    class AnnotatedParseTree {
        public:
            typedef std::shared_ptr<AnnotatedParseTree> shared_tree;
            std::vector<shared_tree> children;
            std::vector<shared_tree> general_children;
            std::weak_ptr<AnnotatedParseTree> parent;
            uint label;
            uint depth;
            uint udepth;
            bool has_parent;
            std::string sentence;
            void add_general_child(shared_tree);
            AnnotatedParseTree(uint);
            AnnotatedParseTree(uint, shared_tree);
            void add_words_to_vector(std::vector<std::string>&) const;
            /**
            To Labeled Pair
            ---------------

            Convert a tree into all its possible
            subtrees, with their associated labels.

            Outputs
            -------

            std::pair<std::vector<std::string>, uint> out :
                tokenized sequence of words, paired with a label
                (a uint number).

            **/
            std::pair<std::vector<std::string>, uint> to_labeled_pair() const;
    };
    void replace_char_by_char(std::vector<char>&, const char&, const char&);
    /**
    Create Tree From String
    -----------------------

    Take a character vector and build up
    the parse tree representation using
    a stack based technique to detect
    the level of nesting.

    Inputs
    ------

    const std::string& str : input text contains the tree.

    Outputs
    -------

    AnnotatedParseTree::shared_tree tree : shared pointer to tree
    **/
    AnnotatedParseTree::shared_tree create_tree_from_string(const std::string&);
    /**
    Stream to Sentiment Treebank
    ----------------------------

    Allow loading from a file stream given from
    either a gzip or regular fstream.

    Inputs
    ------

    T& stream : the stream source

    std::vector<AnnotatedParseTree::shared_tree>& dest :
        where to place the loaded trees.
    **/
    template<typename T>
    void stream_to_sentiment_treebank(T&, std::vector<AnnotatedParseTree::shared_tree>&);
    std::vector<AnnotatedParseTree::shared_tree> load(const std::string&);

    class Databatch {
        public:
            typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
            typedef std::shared_ptr<index_mat> shared_index_mat;
            typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> eigen_index_vector;
            typedef std::shared_ptr<eigen_index_vector> shared_eigen_index_vector;

            shared_index_mat data;
            shared_eigen_index_vector targets;
            shared_eigen_index_vector codelens;
            int total_codes;
            Databatch(int n, int d);

            void insert_example_indices_into_matrix(
                utils::Vocab& word_vocab,
                std::pair<std::vector<std::string>, uint>& example,
                size_t& row);
            static Databatch convert_sentences_to_indices(
                utils::tokenized_uint_labeled_dataset& examples,
                utils::Vocab& word_vocab,
                size_t num_elements,
                std::vector<size_t>::iterator indices,
                std::vector<size_t>::iterator lengths_sorted);
            static std::vector<Databatch> create_labeled_dataset(
                utils::tokenized_uint_labeled_dataset& examples,
                utils::Vocab& word_vocab,
                size_t minibatch_size);
            static std::vector<Databatch> create_labeled_dataset(
                std::vector<SST::AnnotatedParseTree::shared_tree>& examples,
                utils::Vocab& word_vocab,
                size_t minibatch_size);
    };

    extern const std::vector<std::string> label_names;

    utils::Vocab get_word_vocab(std::vector<SST::AnnotatedParseTree::shared_tree>& trees, int min_occurence);
}

std::ostream &operator <<(std::ostream &, const SST::AnnotatedParseTree&);

#endif
