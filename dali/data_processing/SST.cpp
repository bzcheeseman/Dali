#include "dali/data_processing/SST.h"
#include "dali/tensor/Index.h"

using std::string;
using std::vector;
using std::stringstream;
using std::make_shared;
using utils::tokenized_uint_labeled_dataset;
using utils::Vocab;
using std::min;
using json11::Json;
using std::atomic;

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

    treebank_minibatch_dataset convert_trees_to_indexed_minibatches(
        const Vocab& word_vocab,
        const std::vector<AnnotatedParseTree::shared_tree>& trees,
        int minibatch_size) {
        treebank_minibatch_dataset dataset;

        auto to_index_pair = [&word_vocab](std::pair<std::vector<std::string>, uint>&& pair, bool&& is_root) {
            return std::tuple<std::vector<uint>, uint, bool>(
                word_vocab.encode(pair.first),
                pair.second,
                is_root);
        };

        if (dataset.size() == 0)
            dataset.emplace_back(0);

        for (auto& tree : trees) {

            // create new minibatch
            if (dataset[dataset.size()-1].size() == minibatch_size) {
                dataset.emplace_back(0);
                dataset.back().reserve(minibatch_size);
            }

            // add root
            dataset[dataset.size()-1].emplace_back(
                to_index_pair(
                    tree->to_labeled_pair(),
                    true
                )
            );

            // add children:
            for (auto& child : tree->general_children) {
                if (dataset[dataset.size()-1].size() == minibatch_size) {
                    dataset.emplace_back(0);
                    dataset.back().reserve(minibatch_size);
                }
                dataset[dataset.size()-1].emplace_back(
                    to_index_pair(
                        child->to_labeled_pair(),
                        false
                    )
                );
            }
        }
        return dataset;
    }
    /**
    SentimentBatch
    ---------

    Datastructure handling the storage of training
    data, length of each example in a minibatch,
    and total number of prediction instances
    within a single minibatch.
    **/
    template<typename R>
    SentimentBatch<R>::SentimentBatch(int max_example_length, int num_examples) {
        this->data        = Mat<int>(max_example_length, num_examples);
        this->target      = Mat<int>(num_examples, 1);
        this->mask        = Mat<R>(max_example_length, num_examples);
        this->code_lengths.clear();
        this->code_lengths.resize(num_examples);
        this->total_codes = 0;
    }

    template<typename R>
    void SentimentBatch<R>::add_example(
            const Vocab& vocab,
            const std::pair<std::vector<std::string>, uint>& example,
            size_t example_idx,
            bool add_start_symbol) {
        int offset = 0;
        if (add_start_symbol) {
            this->insert_example({START}, vocab, example_idx, offset);
            offset += 1;
        }
        this->insert_example(example.first, vocab, example_idx, offset);

        this->code_lengths[example_idx] = example.first.size();

        this->total_codes += example.first.size();
        // add label for this example
        this->target.w(example_idx) = example.second;

        // ensure model collects error for this label position using non-zero mask.
        this->mask.w(example.first.size() - 1, example_idx) = (R)1.0;
    }

    template<typename R>
    SentimentBatch<R> SentimentBatch<R>::from_examples(
            data_ptr data_begin,
            data_ptr data_end,
            const Vocab& vocab,
            bool add_start_symbol) {
        int num_examples = data_end - data_begin;
        size_t max_length = (*data_begin)->first.size();
        for (auto it = data_begin; it != data_end; ++it) {
            max_length = std::max(max_length, (*it)->first.size());
        }
        if (add_start_symbol) max_length++;

        SentimentBatch<R> batch(max_length, num_examples);
        for (size_t example_idx = 0; example_idx < num_examples; example_idx++) {
            batch.add_example(vocab, **(data_begin + example_idx), example_idx, add_start_symbol);
        }
        return batch;
    }

    template<typename R>
    int SentimentBatch<R>::target_for_example(size_t example_idx) const {
        return this->target.w(example_idx);
    }


    template<typename R>
    vector<SentimentBatch<R>> SentimentBatch<R>::create_dataset(
            const utils::tokenized_uint_labeled_dataset& examples,
            const utils::Vocab& vocab,
            size_t minibatch_size,
            bool add_start_symbol) {

        typedef std::pair<vector<string>, uint> example_t;

        vector<SentimentBatch<R>> dataset;
        vector<const example_t*> sorted_examples;
        for (auto& example: examples) {
            sorted_examples.emplace_back(&example);
        }
        // sort by length to make batches more packed.
        std::sort(sorted_examples.begin(), sorted_examples.end(),
                [](const example_t* a, const example_t* b) {
            return a->first.size() < b->first.size();
        });

        for (int i = 0; i < sorted_examples.size(); i += minibatch_size) {
            auto batch_begin = sorted_examples.begin() + i;
            auto batch_end   = batch_begin + min(
                    minibatch_size,
                    sorted_examples.size() - i
                );

            auto res = SentimentBatch<R>::from_examples(
                batch_begin,
                batch_end,
                vocab,
                add_start_symbol
            );

            dataset.emplace_back(res);
        }
        return dataset;
    }

    template<typename R>
    vector<SentimentBatch<R>> SentimentBatch<R>::create_dataset(
            const vector<SST::AnnotatedParseTree::shared_tree>& trees,
            const Vocab& vocab,
            size_t minibatch_size,
            bool add_start_symbol) {

        utils::tokenized_uint_labeled_dataset dataset;
        for (auto& tree : trees) {
            dataset.emplace_back(tree->to_labeled_pair());
            for (auto& child : tree->general_children) {
                if (((int)child->label) > 4)
                    utils::exit_with_message("Error: One of the trees's children has a label other than 0-4");
                dataset.emplace_back(child->to_labeled_pair());
            }
        }
        return create_dataset(dataset, vocab, minibatch_size, add_start_symbol);
    }

    template class SentimentBatch<float>;
    template class SentimentBatch<double>;


    template<typename R>
    Json json_classification(const vector<string>& sentence, const Mat<R>& probs) {
        // store sentence memory & tokens:
        auto sentence_viz = visualizable::Sentence<R>(sentence);

        // store sentence as input + distribution as output:
        Json::object json_example = {
            { "type", "classifier_example"},
            { "input", sentence_viz.to_json()},
            { "output",  utils::json_finite_distribution(probs, SST::label_names) },
        };

        return json_example;
    }

    template Json json_classification<float>(const vector<string>& sentence, const Mat<float>& probs);
    template Json json_classification<double>(const vector<string>& sentence, const Mat<double>& probs);

    template<typename R>
    Json json_classification(const vector<string>& sentence, const Mat<R>& probs, const Mat<R>& word_weights) {

        // store sentence memory & tokens:
        auto sentence_viz = visualizable::Sentence<R>(sentence);
        sentence_viz.set_weights(word_weights);

        // store sentence as input + distribution as output:
        Json::object json_example = {
            { "type", "classifier_example"},
            { "input", sentence_viz.to_json()},
            { "output",  utils::json_finite_distribution(probs, SST::label_names) },
        };

        return json_example;
    }

    template Json json_classification<float>(const vector<string>& sentence, const Mat<float>& probs, const Mat<float>& word_weights);
    template Json json_classification<double>(const vector<string>& sentence, const Mat<double>& probs, const Mat<double>& word_weights);

    Vocab get_vocabulary(vector<SST::AnnotatedParseTree::shared_tree>& trees, int min_occurence) {
        tokenized_uint_labeled_dataset examples;
        for (auto& tree : trees)
            examples.emplace_back(tree->to_labeled_pair());
        auto index2word  = utils::get_vocabulary(examples, min_occurence);
        Vocab vocab(index2word);
        vocab.word2index[START] = vocab.size();
        vocab.index2word.emplace_back(START);
        return vocab;
    }

    std::tuple<double,double> average_recall(
        vector<vector<std::tuple<vector<uint>, uint, bool>>>& dataset,
        std::function<int(vector<uint>&)> predict,
        int num_threads) {
        ReportProgress<double> journalist("Average recall", dataset.size());
        atomic<int> seen_minibatches(0);
        atomic<int> correct(0);
        atomic<int> correct_root(0);
        atomic<int> total_root(0);
        atomic<int> total(0);
        ThreadPool pool(num_threads);

        for (int batch_id = 0; batch_id < dataset.size(); ++batch_id) {
            pool.run([batch_id, &predict, &dataset, &correct, &total, &correct_root, &total_root, &journalist, &seen_minibatches]() {
                auto& minibatch = dataset[batch_id];
                for (auto& example : minibatch) {
                    int prediction = predict(std::get<0>(example));
                    if (prediction == std::get<1>(example)) {
                        correct += 1;
                        if (std::get<2>(example)) {
                            correct_root +=1;
                        }
                    }
                    total += 1;
                    if (std::get<2>(example)) {
                        total_root +=1;
                    }
                }
                seen_minibatches += 1;
                journalist.tick(seen_minibatches, 100.0 * ((double) correct / (double) total));
            });
        }
        pool.wait_until_idle();
        journalist.done();
        return std::tuple<double, double>(100.0 * ((double) correct / (double) total), 100.0 * (double) correct_root  / (double) total_root);
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
