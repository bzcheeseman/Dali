#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <gflags/gflags.h>
#include <iterator>
#include <mutex>
#include <thread>

#include "dali/core.h"
#include "dali/utils.h"
#include "dali/utils/NlpUtils.h"
#include "dali/models/StackedModel.h"
#include "dali/visualizer/visualizer.h"

DEFINE_int32(minibatch,            100,  "What size should be used for the minibatches ?");
DEFINE_bool(sparse,                true, "Use sparse embedding");
DEFINE_double(cutoff,              2.0,  "KL Divergence error where stopping is acceptable");
DEFINE_int32(patience,             5,    "How many unimproving epochs to wait through before witnessing progress ?");
DEFINE_int32(num_reconstructions,  5,    "How many sentences to demo after each epoch.");
DEFINE_double(dropout,             0.3,  "How many Hintons to include in the neural network.");

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
using std::tuple;
using std::chrono::seconds;

typedef float REAL_t;
typedef Mat<REAL_t> mat;
typedef float price_t;
typedef std::pair<vector<string>, uint> labeled_pair;

const string START = "**START**";

ThreadPool* pool;

template<typename R>
struct Databatch {
    Mat<int> data;
    Mat<R> mask;
    vector<int> code_lengths;
    int total_codes;
    Databatch(int max_example_length, int num_examples) {
        data        = Mat<int>(max_example_length, num_examples);
        mask        = Mat<R>(max_example_length, num_examples);
        code_lengths.clear();
        code_lengths.resize(num_examples);
        total_codes = 0;
    };

    Databatch() = default;

    void add_example(
            Vocab& word_vocab,
            const vector<string>& example,
            size_t& example_idx) {
        auto description_length = example.size();
        data.w(0, example_idx) = word_vocab.word2index[START];
        auto encoded = word_vocab.encode(example, true);
        mask.w(0, example_idx) = 0.0;
        for (size_t j = 0; j < encoded.size(); j++) {
            data.w(j + 1, example_idx) = encoded[j];
            mask.w(j + 1, example_idx) = (R)1.0;
        }
        code_lengths[example_idx] = description_length + 1;
        total_codes += description_length + 1;
    }

    typedef vector<vector<string>*>::iterator data_ptr;

    static Databatch<R> from_examples(data_ptr data_begin, data_ptr data_end, Vocab& vocab) {
        int num_elements = data_end - data_begin;
        size_t max_length = (*data_begin)->size();
        for (auto it = data_begin; it != data_end; ++it) {
            max_length = std::max(max_length, (*it)->size() + 2);
        }

        Databatch<R> databatch(max_length, num_elements);
        for (size_t k = 0; k < num_elements; k++) {
            databatch.add_example(vocab, **(data_begin + k), k);
        }
        return databatch;
    }
};

template<typename R>
vector<Databatch<R>> create_dataset(
        vector<vector<string>>& examples,
        Vocab& word_vocab,
        size_t minibatch_size) {
    vector<Databatch<R>> dataset;
    vector<vector<string>*> sorted_examples;
    for (auto& example: examples) {
        sorted_examples.emplace_back(&example);
    }
    std::sort(sorted_examples.begin(), sorted_examples.end(), [](vector<string>* a, vector<string>* b) {
        return a->size() < b->size();
    });

    for (int i = 0; i < sorted_examples.size(); i += minibatch_size) {
        auto batch_begin = sorted_examples.begin() + i;
        auto batch_end   = batch_begin + min(minibatch_size, sorted_examples.size() - i);

        dataset.emplace_back(Databatch<R>::from_examples(
            batch_begin,
            batch_end,
            word_vocab
        ));
    }
    return dataset;
}

Vocab get_vocabulary(const vector<vector<string>>& examples, int min_occurence) {
    Vocab vocab(utils::get_vocabulary(examples, min_occurence));
    vocab.word2index[START] = vocab.size();
    vocab.index2word.emplace_back(START);
    return vocab;
}


template<typename model_t, typename R>
REAL_t average_error(model_t& model, const vector<Databatch<R>>& dataset) {
    graph::NoBackprop nb;
    Timer t("average_error");

    int full_code_size = 0;
    vector<double> costs(FLAGS_j, 0.0);
    for (size_t i = 0; i < dataset.size();i++)
        full_code_size += dataset[i].total_codes;

    for (size_t batch_id = 0; batch_id < dataset.size(); ++batch_id) {
        pool->run([&costs, &dataset, &model, batch_id]() {
            costs[ThreadPool::get_thread_number()] += model.masked_predict_cost(
                dataset[batch_id].data, // the sequence to draw from
                dataset[batch_id].data, // what to predict (the words offset by 1)
                dataset[batch_id].mask,
                0.0
            ).w(0);

        });
    }
    pool->wait_until_idle();

    return utils::vsum(costs) / full_code_size;
}

template<typename R>
std::tuple<Vocab, vector<Databatch<R>>> load_dataset_and_vocabulary(const string& fname, int min_occurence, int minibatch_size) {
        std::tuple<Vocab, vector<Databatch<R>>> pair;

        auto text_corpus  = utils::load_tokenized_unlabeled_corpus(fname);
        std::get<0>(pair) = get_vocabulary(text_corpus, min_occurence);
        std::get<1>(pair) = create_dataset<R>(text_corpus, std::get<0>(pair), minibatch_size);
        return pair;
}

template<typename R>
vector<Databatch<R>> load_dataset_with_vocabulary(const string& fname, Vocab& vocab, int minibatch_size) {
        auto text_corpus        = utils::load_tokenized_unlabeled_corpus(fname);
        return create_dataset<R>(text_corpus, vocab, minibatch_size);
}

typedef std::tuple<Mat<REAL_t>, typename StackedModel<REAL_t>::state_type> beam_search_state_t;
typedef vector<beam_search::BeamSearchResult<REAL_t, beam_search_state_t>> beam_search_results_t;

beam_search_results_t the_beam_search(
        const StackedModel<REAL_t>& model,
        const Vocab& word_vocab,
        Indexing::Index indices) {
    graph::NoBackprop nb;
    const uint beam_width = 5;
    const int max_len    = 20;

    auto state = model.initial_states();
    int last_index = 0;
    for (auto index : indices) {
        auto input_vector = model.embedding[(int)index];
        state = model.stacked_lstm.activate(
            state,
            input_vector
        );
        last_index = index;
    }

    // state comprises of last input embedding and lstm state
    beam_search_state_t initial_state = make_tuple(model.embedding[last_index], state);

    auto candidate_scores = [&model](beam_search_state_t state) {
        auto& input_vector = std::get<0>(state);
        auto& lstm_state   = std::get<1>(state);
        return MatOps<REAL_t>::softmax(model.decode(input_vector, lstm_state)).log();
    };

    auto make_choice = [&model](beam_search_state_t state, uint candidate) {
        auto input_vector = model.embedding[candidate];
        auto lstm_state = model.stacked_lstm.activate(
            std::get<1>(state),
            input_vector
        );
        return make_tuple(input_vector, lstm_state);
    };

    auto beams = beam_search::beam_search<REAL_t, beam_search_state_t>(
        initial_state,
        beam_width,
        candidate_scores,
        make_choice,
        word_vocab.word2index.at(utils::end_symbol),
        max_len);

    return beams;
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

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    utils::Vocab      word_vocab;
    vector<Databatch<REAL_t>> training;
    vector<Databatch<REAL_t>> validation;

    Timer dl_timer("Dataset loading");
    std::tie(word_vocab, training) = load_dataset_and_vocabulary<REAL_t>(
        FLAGS_train,
        FLAGS_min_occurence,
        FLAGS_minibatch);

    validation = load_dataset_with_vocabulary<REAL_t>(
        FLAGS_validation,
        word_vocab,
        FLAGS_minibatch);
    dl_timer.stop();

    std::cout << "    Vocabulary size = " << word_vocab.size() << " (occuring more than " << FLAGS_min_occurence << ")" << std::endl
              << "Max training epochs = " << FLAGS_epochs           << std::endl
              << "    Training cutoff = " << FLAGS_cutoff           << std::endl
              << "  Number of threads = " << FLAGS_j                << std::endl
              << "     minibatch size = " << FLAGS_minibatch        << std::endl
              << "       max_patience = " << FLAGS_patience         << std::endl;

    pool = new ThreadPool(FLAGS_j);
    shared_ptr<Visualizer> visualizer;

    if (!FLAGS_visualizer.empty()) {
        try {
            visualizer = make_shared<Visualizer>(FLAGS_visualizer, true);
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl; // could not connect to redis.
        }
    }

    auto model = StackedModel<REAL_t>::build_from_CLI(
        FLAGS_load,
        word_vocab.size(),
        word_vocab.size(),
        true);

    auto parameters = model.parameters();
    auto solver     = Solver::construct(FLAGS_solver, parameters, (REAL_t) FLAGS_learning_rate);

    // replicate model for each thread:
    vector<StackedModel<REAL_t>> thread_models;
    if (FLAGS_j == 1) {
        thread_models.emplace_back(model, false, false);
    } else {
        for (int i = 0; i < FLAGS_j; ++i)
            thread_models.emplace_back(model, false, true);
    }

    Throttled throttled;

    int epoch       = 0;
    auto cost       = std::numeric_limits<REAL_t>::infinity();
    double new_cost = 0.0;
    int patience    = 0;

    while (cost > FLAGS_cutoff && epoch < FLAGS_epochs && patience < FLAGS_patience) {

        std::atomic<int> full_code_size(0);
        auto random_batch_order = utils::random_arange(training.size());

        std::atomic<int> batches_processed(0);

        ReportProgress<double> journalist(utils::MS() << "Training epoch " << epoch, random_batch_order.size());

        for (auto batch_id : random_batch_order) {
            pool->run([&, solver, batch_id]() {
                auto& thread_model = thread_models[ThreadPool::get_thread_number()];
                auto thread_parameters = thread_model.parameters();
                auto& minibatch = training[batch_id];

                auto error = thread_model.masked_predict_cost(
                    minibatch.data, // the sequence to draw from
                    minibatch.data, // what to predict (the words offset by 1)
                    minibatch.mask,
                    FLAGS_dropout
                );
                error.grad();

                graph::backward(); // backpropagate
                solver->step(thread_parameters);
                journalist.tick(++batches_processed, error.w(0) / minibatch.total_codes);
                throttled.maybe_run(seconds(10), [&]() {
                    // Tell the journalist the news can wait
                    journalist.pause();
                    graph::NoBackprop nb;
                    auto& random_batch = training[utils::randint(0, training.size() - 1)];
                    auto random_example_index = utils::randint(0, random_batch.data.dims(1) - 1);
                    std::cout << random_batch.code_lengths[random_example_index] << std::endl;

                    int priming_size = utils::randint(1, std::min(6, random_batch.code_lengths[random_example_index]));

                    vector<uint> priming;
                    for (int i = 0; i < priming_size; ++i) {
                        priming.push_back(random_batch.data.w(i, random_example_index));
                    }

                    auto beams = the_beam_search(model, word_vocab, &priming);

                    vector<uint> priming_no_start(priming.begin() + 1, priming.end());

                    std::cout << "Reconstructions: " << std::endl;
                    for (auto& beam : beams) {
                        std::cout << "=> (" << std::setprecision( 5 ) << beam.score << ") ";
                        std::cout << utils::join(word_vocab.decode(&priming_no_start), " ") << " ";
                        std::cout << utils::bold;
                        std::cout << utils::join(word_vocab.decode(&beam.solution, true), " ") << std::endl;
                        std::cout << utils::reset_color << std::endl;
                    }

                    if (visualizer != nullptr) {
                        vector<vector<string>> sentences;
                        vector<REAL_t>         probs;
                        for (auto& beam : beams) {
                            sentences.emplace_back(word_vocab.decode(&beam.solution, true));
                            probs.emplace_back(beam.score);
                        }

                        auto input_sentence = make_shared<visualizable::Sentence<REAL_t>>(
                                word_vocab.decode(&priming_no_start));
                        auto sentences_viz = make_shared<visualizable::Sentences<REAL_t>>(sentences);
                        sentences_viz->set_weights(probs);

                        auto input_output_pair = visualizable::GridLayout();

                        input_output_pair.add_in_column(0, input_sentence);
                        input_output_pair.add_in_column(1, sentences_viz);

                        visualizer->feed(input_output_pair.to_json());
                    }

                    journalist.resume();

                    journalist.done();
                });

            });
        }

        pool->wait_until_idle();

        new_cost = average_error(model, validation);
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
        maybe_save_model(&model);

        Timer::report();


        ELOG(memory_bank<REAL_t>::num_cpu_allocations);
        ELOG(memory_bank<REAL_t>::total_cpu_memory);
        #ifdef DALI_USE_CUDA
            ELOG(memory_bank<REAL_t>::num_gpu_allocations);
            ELOG(memory_bank<REAL_t>::total_gpu_memory);
        #endif

        epoch++;
    }

    return 0;
}
