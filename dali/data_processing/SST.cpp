#include "SST.h"

using std::string;
using std::vector;
using std::stringstream;
using std::make_shared;
using utils::tokenized_uint_labeled_dataset;
using utils::Vocab;
using std::min;

const string START = "**START**";

namespace SST {
    const std::vector<std::string> label_names = {"--", "-", "=", "+", "++"};

    AnnotatedParseTree::AnnotatedParseTree(uint _depth) : depth(_depth), udepth(1), has_parent(false) {}
    AnnotatedParseTree::AnnotatedParseTree(uint _depth, shared_tree _parent) : depth(_depth), udepth(1), has_parent(true), parent(_parent) {}

    void replace_char_by_char(std::vector<char>& list, const char& target, const char& replacement) {
        for (auto& c : list)
            if (c == target) c = replacement;
    }

    void AnnotatedParseTree::add_general_child(AnnotatedParseTree::shared_tree child) {
        general_children.emplace_back(child);
    }

    void AnnotatedParseTree::add_words_to_vector(vector<string>& list) const {
        if (children.size() == 0) {
            list.emplace_back(sentence);
        } else {
            for (auto& child : children)
                child->add_words_to_vector(list);
        }
    }

    std::pair<vector<string>, uint> AnnotatedParseTree::to_labeled_pair() const {
        std::pair<vector<string>, uint> pair;
        pair.second = label;
        add_words_to_vector(pair.first);
        return pair;
    }

    AnnotatedParseTree::shared_tree create_tree_from_string(const string& line) {
        int depth = 0;
        bool awaiting_num = false;
        std::vector<char> current_word;
        AnnotatedParseTree::shared_tree root = nullptr;
        auto current_node = root;
        stringstream ss(line);
        char ch;
        const char left_parenthesis  = '(';
        const char right_parenthesis = ')';
        const char space             = ' ';

        while (ss) {
            ch = ss.get();
            if (awaiting_num) {
                current_node->label = (uint)((int) (ch - '0'));
                awaiting_num = false;
            } else {
                if (ch == left_parenthesis) {
                    if (++depth > 1) {
                        // replace current head node by this node:
                        current_node->children.emplace_back(make_shared<AnnotatedParseTree>(depth, current_node));
                        current_node = current_node->children.back();
                        root->add_general_child(current_node);
                    } else {
                        root = make_shared<AnnotatedParseTree>(depth);
                        current_node = root;
                    }
                    awaiting_num = true;
                } else if (ch == right_parenthesis) {
                    // assign current word:
                    if (current_word.size() > 0) {
                        replace_char_by_char(current_word, '\xa0', space);
                        current_node->sentence = string(current_word.begin(), current_word.end());
                        current_node->udepth   = 1;
                        // erase current word
                        current_word.clear();
                    }
                    // go up a level:
                    depth--;
                    if (current_node->has_parent) {
                        uint& current_node_udepth = current_node->udepth;
                        current_node = current_node->parent.lock();
                        current_node->udepth = std::max(current_node_udepth+1, current_node->udepth);
                    } else {
                        current_node = nullptr;
                    }
                } else if (ch == space) {
                    // ignore spacing
                    continue;
                } else {
                    // add to current read word
                    current_word.emplace_back(ch);
                }
            }
        }
        if (depth != 0)
            throw std::invalid_argument("ParseError: Not an equal amount of closing and opening parentheses");
        return root;
    }

    template<typename T>
    void stream_to_sentiment_treebank(T& fp, vector<AnnotatedParseTree::shared_tree>& trees) {
        string line;
        while (std::getline(fp, line))
            trees.emplace_back( create_tree_from_string(line) );
    }

    vector<AnnotatedParseTree::shared_tree> load(const string& fname) {
        vector<AnnotatedParseTree::shared_tree> trees;
        if (utils::file_exists(fname)) {
            if (utils::is_gzip(fname)) {
                igzstream fpgz(fname.c_str(), std::ios::in | std::ios::binary);
                stream_to_sentiment_treebank(fpgz, trees);
            } else {
                std::fstream fp(fname, std::ios::in | std::ios::binary);
                stream_to_sentiment_treebank(fp, trees);
            }
        } else {
            stringstream error_msg;
            error_msg << "FileNotFound: No file found at \"" << fname << "\"";
            throw std::runtime_error(error_msg.str());
        }
        return trees;
    }

    /**
    Databatch
    ---------

    Datastructure handling the storage of training
    data, length of each example in a minibatch,
    and total number of prediction instances
    within a single minibatch.

    **/
    Databatch::Databatch(int n, int d) {
        data        = make_shared<index_mat>(n, d);
        targets     = make_shared<eigen_index_vector>(n);
        codelens    = make_shared<eigen_index_vector>(n);
        total_codes = 0;
        data->fill(0);
    }

    void Databatch::insert_example_indices_into_matrix(
        Vocab& word_vocab,
        std::pair<std::vector<std::string>, uint>& example,
        size_t& row) {
        auto description_length = example.first.size();
        (*data)(row, 0) = word_vocab.word2index[START];
        for (size_t j = 0; j < description_length; j++)
            (*data)(row, j + 1) = word_vocab.word2index.find(example.first[j]) != word_vocab.word2index.end() ? word_vocab.word2index[example.first[j]] : word_vocab.unknown_word;
        (*data)(row, description_length + 1) = word_vocab.word2index[utils::end_symbol];
        (*codelens)(row) = description_length + 1;
        total_codes += description_length + 1;
        (*targets)(row) = example.second;
    }

    Databatch Databatch::convert_sentences_to_indices(
        tokenized_uint_labeled_dataset& examples,
        Vocab& word_vocab,
        size_t num_elements,
        vector<size_t>::iterator indices,
        vector<size_t>::iterator lengths_sorted) {

        auto indices_begin = indices;
        Databatch databatch(num_elements, *std::max_element(lengths_sorted, lengths_sorted + num_elements));
        for (size_t k = 0; k < num_elements; k++)
            databatch.insert_example_indices_into_matrix(
                word_vocab,
                examples[*(indices++)],
                k);
        return databatch;
    }

    vector<Databatch> Databatch::create_labeled_dataset(
        vector<SST::AnnotatedParseTree::shared_tree>& trees,
        Vocab& word_vocab,
        size_t minibatch_size) {

        utils::tokenized_uint_labeled_dataset dataset;
        for (auto& tree : trees) {
            dataset.emplace_back(tree->to_labeled_pair());
            for (auto& child : tree->general_children) {
                if (((int)child->label) > 4)
                    utils::exit_with_message("Error: One of the trees's children has a label other than 0-4");
                dataset.emplace_back(child->to_labeled_pair());
            }
        }
        return create_labeled_dataset(dataset, word_vocab, minibatch_size);
    }

    vector<Databatch> Databatch::create_labeled_dataset(
        tokenized_uint_labeled_dataset& examples,
        Vocab& word_vocab,
        size_t minibatch_size) {

        vector<Databatch> dataset;
        vector<size_t> lengths = vector<size_t>(examples.size());
        for (size_t i = 0; i != lengths.size(); ++i) lengths[i] = examples[i].first.size() + 2;
        vector<size_t> lengths_sorted(lengths);

        auto shortest = utils::argsort(lengths);
        std::sort(lengths_sorted.begin(), lengths_sorted.end());
        size_t piece_size = minibatch_size;
        size_t so_far = 0;

        auto shortest_ptr = lengths_sorted.begin();
        auto end_ptr = lengths_sorted.end();
        auto indices_ptr = shortest.begin();

        while (shortest_ptr != end_ptr) {
            dataset.emplace_back( Databatch::convert_sentences_to_indices(
                examples,
                word_vocab,
                min(piece_size, lengths.size() - so_far),
                indices_ptr,
                shortest_ptr) );
            shortest_ptr += min(piece_size,          lengths.size() - so_far);
            indices_ptr  += min(piece_size,          lengths.size() - so_far);
            so_far        = min(so_far + piece_size, lengths.size());
        }
        return dataset;
    }

    /**
    get word vocab
    --------------

    Collect a mapping from words to unique indices
    from a collection of Annnotate Parse Trees
    from the Stanford Sentiment Treebank, and only
    keep words ocurring more than some threshold
    number of times `min_occurence`

    Inputs
    ------

    std::vector<SST::AnnotatedParseTree::shared_tree>& trees : Stanford Sentiment Treebank trees
                                           int min_occurence : cutoff appearance of words to include
                                                               in vocabulary.


    Outputs
    -------

    Vocab vocab : the vocabulary extracted from the trees with the
                  addition of a special "**START**" word.

    **/
    Vocab get_word_vocab(vector<SST::AnnotatedParseTree::shared_tree>& trees, int min_occurence) {
        tokenized_uint_labeled_dataset examples;
        for (auto& tree : trees)
            examples.emplace_back(tree->to_labeled_pair());
        auto index2word  = utils::get_vocabulary(examples, min_occurence);
        Vocab vocab(index2word);
        vocab.word2index[START] = vocab.index2word.size();
        vocab.index2word.emplace_back(START);
        return vocab;
    }
}

std::ostream &operator <<(std::ostream &os, const SST::AnnotatedParseTree&v) {
    if (v.children.size() > 0) {
       os << "(" << v.label << " ";
       for (auto& child : v.children)
           os << *child;
       return os << ")";
    } else {
        return os << "(" << v.label << " " << v.sentence << ") ";
    }
}