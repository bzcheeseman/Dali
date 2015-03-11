#include <algorithm>
#include <atomic>
#include <Eigen/Eigen>
#include <fstream>
#include <iterator>
#include <thread>
#include <chrono>

#include "core/gzstream.h"
#include "core/NlpUtils.h"
#include "core/SST.h"
#include "core/StackedModel.h"
#include "core/StackedGatedModel.h"
#include "core/utils.h"
#include "core/Reporting.h"
#include "core/ThreadPool.h"
#include "core/BeamSearch.h"
#include "core/SequenceProbability.h"
#include "core/Solver.h"


// #define USE_GATES
#ifdef USE_GATES
    #define MODEL_USED StackedGatedModel
#else
    #define MODEL_USED StackedModel
#endif

using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::istringstream;
using std::stringstream;
using std::string;
using std::min;
using std::thread;
using std::ref;
using utils::Vocab;
using utils::from_string;
using utils::OntologyBranch;
using utils::tokenized_uint_labeled_dataset;
using std::atomic;
using std::chrono::seconds;
using std::array;


typedef float REAL_t;
typedef Mat<REAL_t> mat;
typedef float price_t;
typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
typedef Eigen::Matrix<REAL_t, Eigen::Dynamic, 1> float_vector;
typedef std::pair<vector<string>, uint> labeled_pair;

const string START = "**START**";
const vector<string> label_names = {"--", "-", "=", "+", "++"};

ThreadPool* pool;

DEFINE_int32(minibatch, 100, "What size should be used for the minibatches ?");
DEFINE_double(cutoff, 91.0,   "KL Divergence error where stopping is acceptable");
DEFINE_int32(patience, 5,    "How many unimproving epochs to wait through before witnessing progress ?");
DEFINE_int32(epoch_batches, 3, "How many minibatches should each label's model do before doing cross-validation?");
DEFINE_int32(num_reconstructions,  1,    "How many sentences to demo after each epoch.");
DEFINE_double(dropout, 0.3, "How much dropout noise to add to the problem ?");
DEFINE_int32(memory_rampup, 30, "Over how many epochs should the memory grow ?");

DEFINE_bool(use_surprise, true, "Whether to compare choices using log likelihood, or surprise (entropy is the integral of surprise around a circle with prob 0 and 1 on either sides)");

/**
Databatch
---------

Datastructure handling the storage of training
data, length of each example in a minibatch,
and total number of prediction instances
within a single minibatch.

**/
class Databatch {
    typedef shared_ptr<index_mat> shared_index_mat;
    public:
        shared_index_mat data;
        shared_eigen_index_vector targets;
        shared_eigen_index_vector codelens;
        int total_codes;
        Databatch(int n, int d) {
            data        = make_shared<index_mat>(n, d);
            targets     = make_shared<eigen_index_vector>(n);
            codelens    = make_shared<eigen_index_vector>(n);
            total_codes = 0;
            data->fill(0);
        };
};

void insert_example_indices_into_matrix(
    Vocab& word_vocab,
    Databatch& databatch,
    labeled_pair& example,
    size_t& row) {
    auto description_length = example.first.size();
    (*databatch.data)(row, 0) = word_vocab.word2index[START];
    for (size_t j = 0; j < description_length; j++)
        (*databatch.data)(row, j + 1) = word_vocab.word2index.find(example.first[j]) != word_vocab.word2index.end() ? word_vocab.word2index[example.first[j]] : word_vocab.unknown_word;
    (*databatch.data)(row, description_length + 1) = word_vocab.word2index[utils::end_symbol];
    (*databatch.codelens)(row) = description_length + 1;
    databatch.total_codes += description_length + 1;
    (*databatch.targets)(row) = example.second;
}

Databatch convert_sentences_to_indices(
    tokenized_uint_labeled_dataset& examples,
    Vocab& word_vocab,
    size_t num_elements,
    vector<size_t>::iterator indices,
    vector<size_t>::iterator lengths_sorted) {

    auto indices_begin = indices;
    Databatch databatch(num_elements, *std::max_element(lengths_sorted, lengths_sorted + num_elements));
    for (size_t k = 0; k < num_elements; k++)
        insert_example_indices_into_matrix(
            word_vocab,
            databatch,
            examples[*(indices++)],
            k);
    return databatch;
}

vector<Databatch> create_labeled_dataset(
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
        dataset.emplace_back( convert_sentences_to_indices(
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

template<typename model_t>
void reconstruct_random_beams(
    vector<model_t>& models,
    const vector<vector<Databatch>>& datasets,
    const Vocab& word_vocab,
    const int& init_size,
    const int& k,
    const int& max_length) {

    int random_example_index;
    const Databatch* random_batch;
    while (true) {
        const vector<Databatch>& dataset = datasets[utils::randint(0, datasets.size() - 1)];
        random_batch = &dataset[utils::randint(0, dataset.size() - 1)];
        random_example_index = utils::randint(0, random_batch->data->rows() - 1);
        if ((*random_batch->codelens)(random_example_index) > init_size) {
            break;
        }
    }
    std::cout << "Reconstructions: \"";
    for (int j = 1; j < init_size; j++)
        std::cout << word_vocab.index2word[(*random_batch->data)(random_example_index, j)] << " ";
    std::cout << "\"" << std::endl;
    size_t name_num = 0;
    for (auto& model : models) {
        std::cout << "(" << label_names[name_num++] << ") ";
        auto beams = beam_search::beam_search(model,
            random_batch->data->row(random_example_index).head(init_size),
            max_length,
            0,  // offset symbols that are predicted
                // before being refed (no = 0)
            k,
            word_vocab.word2index.at(utils::end_symbol), // when to stop the sequence
            word_vocab.unknown_word
        );
        for (const auto& beam : beams) {
            std::cout << "=> (" << std::setprecision( 5 ) << std::get<1>(beam) << ") ";
            for (const auto& word : std::get<0>(beam)) {
                if (word != word_vocab.word2index.at(utils::end_symbol))
                    std::cout << word_vocab.index2word.at(word) << " ";
            }
            std::cout << std::endl;
        }
    }
}

class ConfusionMatrix {
    public:
        vector<vector<atomic<int>>> grid;
        vector<atomic<int>> totals;
        const vector<string>& names;
        ConfusionMatrix(int classes, const vector<string>& _names) : names(_names), totals(classes) {
            for (int i = 0; i < classes;++i) {
                grid.emplace_back(classes);
            }
        }
        void classified_a_when_b(int a, int b) {
            // update the misclassification:
            grid[b][a] += 1;
            // update the stakes:
            totals[b]  += 1;
        };
        void report() const {
            std::cout << "\nConfusion Matrix\n\t";
            for (auto & name : names) {
                std::cout << name << "\t";
            }
            std::cout << "\n";
            auto names_ptr = names.begin();
            auto totals_ptr = totals.begin();
            for (auto& category : grid) {
                std::cout << *names_ptr << "\t";
                for (auto & el : category) {
                    std::cout << std::fixed
                              << std::setw(4)
                              << std::setprecision(2)
                              << std::setfill(' ')
                              << ((*totals_ptr) > 0 ? (100.0 * ((double) el / (double)(*totals_ptr))) : 0.0)
                              << "%\t";
                }
                std::cout << "\n";
                names_ptr++;
                totals_ptr++;
            }
        }
};

template<typename model_t>
REAL_t average_error(const vector<model_t>& models, const vector<vector<Databatch>>& validation_sets) {
    atomic<int> correct(0);
    int set_size = 10;
    ReportProgress<double> journalist("Average error", validation_sets.size() * set_size);
    atomic<int> total(0);
    atomic<int> seen_minibatches(0);
    auto confusion = ConfusionMatrix(5, label_names);

    for (int validation_set_num = 0; validation_set_num < validation_sets.size(); validation_set_num++) {

        auto random_batch_order = utils::random_arange(validation_sets[validation_set_num].size());
        if (random_batch_order.size() > set_size)
            random_batch_order.resize(set_size);

        for (auto& minibatch_num : random_batch_order) {
            pool->run([&confusion, &journalist, &models, &correct,&seen_minibatches,  &total, &validation_sets, validation_set_num, minibatch_num] {
                auto& valid_set = validation_sets[validation_set_num][minibatch_num];
                vector<mat> probs;
                for (int k = 0; k < models.size();k++) {
                    probs.emplace_back(FLAGS_use_surprise ?
                        sequence_probability::sequence_surprises(
                            models[k],
                            *valid_set.data,
                            *valid_set.codelens) :
                        sequence_probability::sequence_probabilities(
                            models[k],
                            *valid_set.data,
                            *valid_set.codelens));
                }

                for (int row_num = 0; row_num < valid_set.data->rows(); ++row_num) {
                    int best_model = -1;
                    double best_prob = (FLAGS_use_surprise ? 1.0 : -1.0) * std::numeric_limits<REAL_t>::infinity();
                    if (FLAGS_use_surprise) {
                        for (int k = 0; k < models.size();k++) {
                            auto prob = probs[k].w()(row_num);
                            if (prob < best_prob) {
                                best_prob = prob;
                                best_model = k;
                            }
                        }
                    } else {
                        for (int k = 0; k < models.size();k++) {
                            auto prob = probs[k].w()(row_num);
                            if (prob > best_prob) {
                                best_prob = prob;
                                best_model = k;
                            }
                        }
                    }
                    confusion.classified_a_when_b(best_model, (*valid_set.targets)(row_num));
                    if (best_model == (*valid_set.targets)(row_num)) {
                        correct++;
                    }
                }
                seen_minibatches++;
                total+= valid_set.codelens->rows();
                journalist.tick(seen_minibatches, (REAL_t) 100.0 * correct / (REAL_t) total);
            });
        }
    }
    pool->wait_until_idle();
    confusion.report();

    return ((REAL_t) 100.0 * correct / (REAL_t) total);
};


int main( int argc, char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Sentiment Analysis as Competition amongst Language Models\n"
        "---------------------------------------------------------\n"
        "\n"
        "We present a dual formulation of the word sequence classification\n"
        "task: we treat each label’s examples as originating from different\n"
        "languages and we train language models for each label; at test\n"
        "time we compare the likelihood of a sequence under each label’s\n"
        "language model to find the most likely assignment.\n"
        "\n"
        " @author Jonathan Raiman\n"
        " @date February 13th 2015"
    );


    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    auto epochs              = FLAGS_epochs;
    auto sentiment_treebank  = SST::load(FLAGS_train);

    auto word_vocab          = get_word_vocab(sentiment_treebank, FLAGS_min_occurence);
    auto vocab_size          = word_vocab.index2word.size();

    // Load Dataset of Trees:
    std::cout << "Unique Treees Loaded : " << sentiment_treebank.size() << std::endl
              << "        Example tree : " << *sentiment_treebank[sentiment_treebank.size()-1] << std::endl
              << "     Vocabulary size : " << vocab_size << std::endl;

    // Put trees into matrices:
    const int NUM_SENTIMENTS = 5;
    vector<vector<Databatch>> datasets(NUM_SENTIMENTS);
    vector<vector<Databatch>> validation_sets(NUM_SENTIMENTS);

    {
        vector<tokenized_uint_labeled_dataset> tree_types(NUM_SENTIMENTS);
        vector<tokenized_uint_labeled_dataset> validation_tree_types(NUM_SENTIMENTS);

        for (auto& tree : sentiment_treebank) {
            if (((int) tree->label) > 4)
                utils::exit_with_message("Error: One of the trees has a label other than 0-4");
            tree_types[tree->label].emplace_back(tree->to_labeled_pair());
            for (auto& child : tree->general_children) {
                if (((int)child->label) > 4)
                    utils::exit_with_message("Error: One of the trees's children has a label other than 0-4");
                tree_types[(int) child->label].emplace_back(child->to_labeled_pair());
            }
        }
        auto validation_treebank = SST::load(FLAGS_validation);
        for (auto& tree : validation_treebank) {
            if (((int) tree->label) > 4)
                utils::exit_with_message("Error: One of the trees has a label other than 0-4");
            validation_tree_types[tree->label].emplace_back(tree->to_labeled_pair());
            for (auto& child : tree->general_children) {
                if (((int)child->label) > 4)
                    utils::exit_with_message("Error: One of the trees's children has a label other than 0-4");
                validation_tree_types[(int) child->label].emplace_back(child->to_labeled_pair());
            }
        }
        int i = 0;
        for (auto& tree_type : tree_types)
            std::cout << "Label type " << i++ << " has " << tree_type.size() << " different examples" << std::endl;
        i = 0;

        for (auto& tree_type : validation_tree_types) {
            std::cout << "Label type " << i++ << " has " << tree_type.size() << " validation examples" << std::endl;
        }

        i = 0;
        for (auto& tree_type : tree_types)
            datasets[i++] = create_labeled_dataset(tree_type, word_vocab, FLAGS_minibatch);
        i = 0;
        for (auto& tree_type : validation_tree_types)
            validation_sets[i++] = create_labeled_dataset(tree_type, word_vocab, FLAGS_minibatch);
    }

    std::cout     << "    Max training epochs = " << FLAGS_epochs << std::endl;
    std::cout     << "    Training cutoff     = " << FLAGS_cutoff << std::endl;
    std::cout     << "Minibatches/label/x-val = " << FLAGS_epoch_batches << std::endl;
    #ifdef USE_GATES
        std::cout << "      using gated model = true" << std::endl;
    #else
        std::cout << "      using gated model = false" << std::endl;
    #endif
    std::cout     << "     Use Shortcut LSTMs = " << (FLAGS_shortcut ? "true" : "false") << std::endl;
    std::cout     << " Comparing models using = " << (FLAGS_use_surprise ? "surprise" : "log likelihood") << std::endl;

    pool = new ThreadPool(FLAGS_j);

    int patience = 0;
    // with a rampup model we start with zero memory penalty and gradually increase the memory
    // L1 penalty until it reaches the desired level.
    // this allows early exploration, but only later forces sparsity on the model

    std::vector<MODEL_USED<REAL_t>> models;
    vector<vector<MODEL_USED<REAL_t>>> thread_models;
    vector<Solver::Adam<REAL_t>> solvers;


    for (int sentiment = 0; sentiment < NUM_SENTIMENTS; sentiment++) {

        if (!FLAGS_load.empty()) {
            std::cout << "Loading model : \"" << FLAGS_load << sentiment << "\"" << std::endl;
            models.emplace_back(MODEL_USED<REAL_t>::load(FLAGS_load + std::to_string(sentiment)));
        } else {
            models.emplace_back(
                word_vocab.index2word.size(),
                FLAGS_input_size,
                FLAGS_hidden,
                FLAGS_stack_size < 1 ? 1 : FLAGS_stack_size,
                word_vocab.index2word.size(),
                FLAGS_shortcut
            );
        }
        thread_models.emplace_back();
        for (int thread_no = 0; thread_no < FLAGS_j; ++thread_no) {
            thread_models[sentiment].push_back(models[sentiment].shallow_copy());
        }
        auto params = models[sentiment].parameters();
        //solvers.emplace_back(params, FLAGS_rho);
        solvers.emplace_back(params, 0.1, 0.001, 1e-9, 5.0);
    }
    int epoch = 0;
    REAL_t accuracy = 0.0;
    REAL_t new_accuracy;
    Throttled t;
    while (accuracy < FLAGS_cutoff && patience < FLAGS_patience) {
        stringstream ss;
        ss << "Epoch " << ++epoch;
        atomic<int> batches_processed(0);

        ReportProgress<double> journalist(ss.str(), NUM_SENTIMENTS * FLAGS_epoch_batches);

        for (int sentiment = 0; sentiment < NUM_SENTIMENTS; sentiment++) {
            for (int batch_id = 0; batch_id < FLAGS_epoch_batches; ++batch_id) {
                pool->run([&thread_models, &journalist, &solvers, &datasets, sentiment, &epoch, &accuracy, &batches_processed]() {
                    auto& thread_model = thread_models[sentiment][ThreadPool::get_thread_number()];
                    auto& solver = solvers[sentiment];

                    auto thread_parameters = thread_model.parameters();
                    auto& minibatch = datasets[sentiment][utils::randint(0, datasets[sentiment].size()-1)];

                    #ifdef USE_GATES
                        thread_model.memory_penalty = (FLAGS_memory_penalty / minibatch.data->cols()) * std::min((REAL_t)1.0, ((REAL_t) (epoch*epoch) / ((REAL_t) FLAGS_memory_rampup * FLAGS_memory_rampup)));
                    #endif

                    thread_model.masked_predict_cost(
                        minibatch.data, // the sequence to draw from
                        minibatch.data, // what to predict (the words offset by 1)
                        1,
                        minibatch.codelens,
                        0,
                        (REAL_t) FLAGS_dropout
                    );
                    graph::backward(); // backpropagate
                    solver.step(thread_parameters); // One step of gradient descent

                    journalist.tick(++batches_processed, accuracy);
                });
            }
        }

        while(true) {
            journalist.pause();
            reconstruct_random_beams(models, datasets, word_vocab,
                utils::randint(2, 6), // how many elements to use as a primer for beam
                FLAGS_num_reconstructions, // how many beams
                20 // max size of a sequence
            );
            journalist.resume();
            // TODO(jonathan): reconstructions go here..
            if (pool->wait_until_idle(seconds(20)))
                break;
        }

        journalist.done();
        new_accuracy = average_error(models, validation_sets);

        if (new_accuracy < accuracy) {
            patience +=1;
        } else {
            patience = 0;
        }
        accuracy = new_accuracy;

        t.maybe_run(seconds(600), [&models]() {
            int i = 0;
            for (auto& model : models) {
                model.save(FLAGS_save + std::to_string(i));
                i++;
            }
        });
    }

    return 0;
}
