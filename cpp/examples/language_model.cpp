#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <Eigen/Eigen>
#include <fstream>
#include <gflags/gflags.h>
#include <gflags/gflags_completions.h>
#include <iterator>
#include <mutex>
#include <thread>

#include "core/gzstream.h"
#include "core/NlpUtils.h"
#include "core/SST.h"
#include "core/StackedModel.h"
#include "core/ThreadPool.h"
#include "core/utils.h"
#include "core/Reporting.h"
#include "core/BeamSearch.h"
#include "core/Solver.h"

DEFINE_int32(minibatch,            100,  "What size should be used for the minibatches ?");
DEFINE_bool(sparse,                true, "Use sparse embedding");
DEFINE_double(cutoff,              2.0,  "KL Divergence error where stopping is acceptable");
DEFINE_int32(patience,             5,    "How many unimproving epochs to wait through before witnessing progress ?");
DEFINE_int32(num_reconstructions,  5,    "How many sentences to demo after each epoch.");

using std::ifstream;
using std::istringstream;
using std::make_shared;
using std::min;
using std::ref;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::thread;
using std::vector;
using utils::OntologyBranch;
using utils::tokenized_uint_labeled_dataset;
using utils::Vocab;
using utils::Timer;
using std::chrono::seconds;

typedef float REAL_t;
typedef Graph<REAL_t> graph_t;
typedef Mat<REAL_t> mat;
typedef float price_t;
typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
typedef Eigen::Matrix<REAL_t, Eigen::Dynamic, 1> float_vector;
typedef std::pair<vector<string>, uint> labeled_pair;


const string START = "**START**";

ThreadPool* pool;

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
        shared_eigen_index_vector codelens;
        shared_ptr<vector<uint>> row_keys;
        int total_codes;
        Databatch(int n, int d) {
            data        = make_shared<index_mat>(n, d);
            codelens    = make_shared<eigen_index_vector>(n);
            row_keys    = make_shared<vector<uint>>();
            total_codes = 0;
            data->fill(0);
        };
};

void insert_example_indices_into_matrix(
    Vocab& word_vocab,
    Databatch& databatch,
    const vector<string>& example,
    size_t& row) {
    auto description_length = example.size();
    (*databatch.data)(row, 0) = word_vocab.word2index[START];
    for (size_t j = 0; j < description_length; j++) {
        (*databatch.data)(row, j + 1) = word_vocab.word2index.find(example[j]) != word_vocab.word2index.end() ? word_vocab.word2index[example[j]] : word_vocab.unknown_word;
        utils::add_to_set(*databatch.row_keys, (*databatch.data)(row, j + 1));
    }
    (*databatch.data)(row, description_length + 1) = word_vocab.word2index[utils::end_symbol];
    (*databatch.codelens)(row) = description_length + 1;
    databatch.total_codes += description_length + 1;
}

Databatch convert_sentences_to_indices(
    const vector<vector<string>>& examples,
    Vocab& word_vocab,
    size_t num_elements,
    vector<size_t>::iterator indices,
    vector<size_t>::iterator lengths_sorted) {

    auto indices_begin = indices;
    Databatch databatch(num_elements, *std::max_element(lengths_sorted, lengths_sorted + num_elements));
    utils::add_to_set(*databatch.row_keys, word_vocab.word2index[utils::end_symbol]);
    utils::add_to_set(*databatch.row_keys, word_vocab.word2index[START]);
    for (size_t k = 0; k < num_elements; k++)
        insert_example_indices_into_matrix(
            word_vocab,
            databatch,
            examples[*(indices++)],
            k);
    return databatch;
}

vector<Databatch> create_dataset(
    const vector<vector<string>>& examples,
    Vocab& word_vocab,
    size_t minibatch_size) {

    vector<Databatch> dataset;
    vector<size_t> lengths = vector<size_t>(examples.size());
    for (size_t i = 0; i != lengths.size(); ++i) lengths[i] = examples[i].size() + 2;
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
Vocab get_word_vocab(const vector<vector<string>>& examples, int min_occurence) {
    auto index2word  = utils::get_vocabulary(examples, min_occurence);
    Vocab vocab(index2word);
    vocab.word2index[START] = vocab.index2word.size();
    vocab.index2word.emplace_back(START);
    return vocab;
}

/**
Reconstruct Random Beams
------------------------

TODO: rewrite this

Condition on the special "**START**" word, generate
a sentence from the language model and output it
to standard out.

Inputs
------
               StackedModel<T>& model : language model to query
           const Databatch& minibatch : where to pull the sample from
                         const int& i : what row to use from the databatch
              const Vocab& word_vocab : the word vocabulary with a lookup table
                                        mapping unique words to an index and
                                        vice-versa.


**/
template<typename model_t>
void reconstruct_random_beams(
    model_t& model,
    const vector<Databatch>& dataset,
    const Vocab& word_vocab,
    const int& init_size,
    const int& k,
    const int& max_length) {

    int random_example_index;
    const Databatch* random_batch;
    while (true) {
        random_batch = &dataset[utils::randint(0, dataset.size() - 1)];
        random_example_index = utils::randint(0, random_batch->data->rows() - 1);
        if ((*random_batch->codelens)(random_example_index) > init_size) {
            break;
        }
    }
    auto beams = beam_search::beam_search(model,
        random_batch->data->row(random_example_index).head(init_size),
        max_length,
        0,  // offset symbols that are predicted
            // before being refed (no = 0)
        k,
        word_vocab.word2index.at(utils::end_symbol), // when to stop the sequence
        word_vocab.unknown_word
    );
    std::cout << "Reconstructions: \"";
    for (int j = 1; j < init_size; j++)
        std::cout << word_vocab.index2word[(*random_batch->data)(random_example_index, j)] << " ";
    std::cout << "\"" << std::endl;
    for (const auto& beam : beams) {
        std::cout << "=> (" << std::setprecision( 5 ) << std::get<1>(beam) << ") ";
        for (const auto& word : std::get<0>(beam)) {
            if (word != word_vocab.word2index.at(utils::end_symbol))
                std::cout << word_vocab.index2word.at(word) << " ";
        }
        std::cout << std::endl;
    }
}

template<typename model_t>
REAL_t average_error(model_t& model, const vector<Databatch>& dataset) {
        auto G = graph_t(false); // create a new graph for each loop)
    Timer t("average_error");

    int full_code_size = 0;
    vector<double> costs(FLAGS_j);
    for (size_t i = 0; i < dataset.size();i++)
        full_code_size += dataset[i].total_codes;
    for (size_t batch_id = 0; batch_id < dataset.size(); ++batch_id) {
        pool->run([&costs, &dataset, &model, &G, batch_id]() {
            costs[ThreadPool::get_thread_number()] += model.masked_predict_cost(
                G,
                dataset[batch_id].data, // the sequence to draw from
                dataset[batch_id].data, // what to predict (the words offset by 1)
                1,
                dataset[batch_id].codelens,
                0
            );
        });
    }
    pool->wait_until_idle();

    REAL_t cost = 0.0;
    for (auto& v : costs) cost += v;
    return cost / full_code_size;
}

/**
Training Loop
-------------

Go through a single epoch of training by updating
parameters once for each minibatch in the dataset.
Takes an optimizer, model, and dataset and performs
several steps of gradient descent.
Moreover every `report_frequency` epochs this will
output the current training error and perform
a sentence reconstruction from the current model.

See `reconstruct`

Inputs
------
               StackedModel<T>& model : language model to train
     const vector<Databatch>& dataset : sentences broken into minibatches to
                                        train model on.
              const Vocab& word_vocab : the word vocabulary with a lookup table
                                        mapping unique words to an index and
                                        vice-versa.
                            S& Solver : Solver handling updates to parameters using
                                        a specific regimen (SGD, Adadelta, AdaGrad, etc.)
                     const int& epoch : how many epochs of training has been done so far.

**/
template<typename model_t, typename S>
void training_loop(model_t& model,
    const vector<Databatch>& dataset,
    const Vocab& word_vocab,
    S& solver,
    const int& epoch) {

    double cost = 0.0;
    std::atomic<int> full_code_size(0);
    auto random_batch_order = utils::random_arange(dataset.size());

    vector<model_t> thread_models;
    for (int i = 0; i <FLAGS_j; ++i)
        thread_models.emplace_back(model, false, true);

    std::atomic<int> batches_processed(0);

    stringstream ss;
    ss << "Training epoch " << epoch;

    ReportProgress<double> journalist(ss.str(), random_batch_order.size());

    for (auto batch_id : random_batch_order) {
        pool->run([&model, &dataset, &solver, &full_code_size,
                   &cost, &thread_models, batch_id, &random_batch_order,
                   &batches_processed, &journalist]() {

            auto& thread_model = thread_models[ThreadPool::get_thread_number()];
            auto thread_parameters = thread_model.parameters();
            auto& minibatch = dataset[batch_id];

            auto G = graph_t(true);
            cost += thread_model.masked_predict_cost(
                G,
                minibatch.data, // the sequence to draw from
                minibatch.data, // what to predict (the words offset by 1)
                0,
                minibatch.codelens,
                0
            );
            thread_model.embedding->sparse_row_keys = minibatch.row_keys;
            full_code_size += minibatch.total_codes;

            G.backward(); // backpropagate
            solver.step(thread_parameters, FLAGS_rho);

            journalist.tick(++batches_processed, cost / full_code_size);
        });
    }
    while(!pool->idle()) {
        int time_between_reconstructions_s = 10;
        pool->wait_until_idle(seconds(time_between_reconstructions_s));

        // Tell the journalist the news can wait
        journalist.pause();
        reconstruct_random_beams(model, dataset, word_vocab,
            utils::randint(1, 6), // how many elements to use as a primer for beam
            FLAGS_num_reconstructions, // how many beams
            20 // max size of a sequence
        );
        journalist.resume();
    }
    // Thank you for your service
    journalist.done();
}

/**
Train Model
-----------

Train a single language model on a corpus of minibatches of sentences
predicting the next word given the current sequence. Stops training
when max number of epochs is reached, training stops decreasing error
for 5 epochs, or cost dips below the cutoff.

Inputs
------

const vector<Databatch>& dataset : sentences broken into minibatches to
                                   train model on.
         const Vocab& word_vocab : the word vocabulary with a lookup table
                                   mapping unique words to an index and
                                   vice-versa.

**/
template<typename model_t, class S>
void train_model(const vector<Databatch>& dataset,
        const vector<Databatch>& validation_set,
        const Vocab& word_vocab) {

    model_t model = model_t::build_from_CLI(FLAGS_load,
                                            word_vocab.index2word.size(),
                                            word_vocab.index2word.size(),
                                            true);

    auto parameters = model.parameters();
    S solver(parameters, FLAGS_rho, 1e-9, 5.0);

    int epoch = 0;
    auto cost = std::numeric_limits<REAL_t>::infinity();
    double new_cost = 0.0;
    int patience = 0;

    while (cost > FLAGS_cutoff && epoch < FLAGS_epochs && patience < FLAGS_patience) {
        new_cost = 0.0;
        training_loop(model, dataset, word_vocab, solver, epoch);
        new_cost = average_error(model, validation_set);
        if (new_cost >= cost) {
            patience += 1;
        } else {
            patience = 0;
        }
        cost = new_cost;
        std::cout << "epoch (" << epoch << ") KL error = "
                  << std::setprecision(3) << std::fixed
                  << std::setw(5) << std::setfill(' ') << new_cost
                  << " patience = " << patience << std::endl;
        maybe_save_model(model);
        epoch++;
    }
}

std::pair<Vocab, vector<Databatch>> load_dataset_and_vocabulary(const string& fname, int min_occurence, int minibatch_size) {
        auto text_corpus        = utils::load_tokenized_unlabeled_corpus(fname);
        std::pair<Vocab, vector<Databatch>> pair;
        pair.first = get_word_vocab(text_corpus, min_occurence);
        pair.second = create_dataset(text_corpus, pair.first, minibatch_size);
        return pair;
}

vector<Databatch> load_dataset_with_vocabulary(const string& fname, Vocab& vocab, int minibatch_size) {
        auto text_corpus        = utils::load_tokenized_unlabeled_corpus(fname);
        return create_dataset(text_corpus, vocab, minibatch_size);
}

int main( int argc, char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "RNN Language Model using Stacked LSTMs\n"
        "--------------------------------------\n"
        "\n"
        "Predict next word in sentence using Stacked LSTMs.\n"
        "\n"
        " @author Jonathan Raiman\n"
        " @date February 15th 2015"
    );

    GFLAGS_NAMESPACE::HandleCommandLineCompletions();
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    // TODO(jonathan): figure out if this affects performance and/or
    // thread safety
    Eigen::initParallel();


    Timer dl_timer("Dataset loading");
        auto dataset_vocab      = load_dataset_and_vocabulary(
            FLAGS_train,
            FLAGS_min_occurence,
            FLAGS_minibatch);

        auto validation_set     = load_dataset_with_vocabulary(
            FLAGS_validation,
            dataset_vocab.first,
            FLAGS_minibatch);
    dl_timer.stop();

    auto vocab_size = dataset_vocab.first.index2word.size();

    std::cout << "    Vocabulary size = " << vocab_size << " (occuring more than " << FLAGS_min_occurence << ")" << std::endl
              << "Max training epochs = " << FLAGS_epochs           << std::endl
              << "    Training cutoff = " << FLAGS_cutoff           << std::endl
              << "  Number of threads = " << FLAGS_j                << std::endl
              << "     minibatch size = " << FLAGS_minibatch        << std::endl
              << "       max_patience = " << FLAGS_patience         << std::endl;

    pool = new ThreadPool(FLAGS_j);

    train_model<StackedModel<REAL_t>, Solver::AdaDelta<REAL_t>>(
        dataset_vocab.second,
        validation_set,
        dataset_vocab.first);

    Timer::report();
    return 0;
}
