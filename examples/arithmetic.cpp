#include "dali/core.h"
#include "dali/utils.h"
#include "dali/utils/stacked_model_builder.h"
#include "dali/models/StackedModel.h"
#include "dali/data_processing/Arithmetic.h"
#include "dali/execution/BeamSearch.h"

using std::string;
using std::vector;
using std::pair;
using std::to_string;
using std::make_shared;
using std::chrono::seconds;

typedef double REAL_t;

DEFINE_double(reg,           0.0,  "What penalty to place on L2 norm of weights?");
DEFINE_int32(minibatch,      100,  "What size should be used for the minibatches ?");
DEFINE_bool(fast_dropout,    true, "Use fast dropout?");
DEFINE_int32(epochs,         2000, "How many training loops through the full dataset ?");
DEFINE_int32(j,                 1, "How many threads should be used ?");
DEFINE_int32(expression_length, 5, "How much suffering to impose on our friend?");
DEFINE_int32(num_examples,      1500, "How much suffering to impose on our friend?");
DEFINE_int32(max_number_in_expression, 100000, "Maximum number used in mathematical expressions.");

ThreadPool* pool;

typedef std::tuple<Mat<REAL_t>, typename StackedModel<REAL_t>::state_type> beam_search_state_t;
typedef vector<beam_search::BeamSearchResult<REAL_t, beam_search_state_t>> beam_search_results_t;

beam_search_results_t arithmetic_beam_search(
        const StackedModel<REAL_t>& model, Indexing::Index indices) {
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
        return MatOps<REAL_t>::softmax_rowwise(model.decode(input_vector, lstm_state)).log();
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
        arithmetic::vocabulary.word2index.at(utils::end_symbol),
        max_len);

    return beams;
}




int main (int argc,  char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Beam Search Training of arithmetic\n"
        "----------------------------------\n"
        "\n"
        " @author Jonathan Raiman\n"
        " @date April 6th 2015"
    );

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    auto examples = arithmetic::generate_numerical(FLAGS_num_examples, FLAGS_expression_length);
    pool = new ThreadPool(FLAGS_j);

    // train a silly system to output the numbers it needs
    auto model = StackedModel<REAL_t>(
         arithmetic::vocabulary.size(),
         FLAGS_input_size,
         FLAGS_hidden,
         FLAGS_stack_size,
         arithmetic::vocabulary.size(),
         false,
         false);

    auto params = model.parameters();
    auto solver = Solver::construct(FLAGS_solver, params, (REAL_t)FLAGS_learning_rate, (REAL_t)FLAGS_reg);

    int epoch = 0;
    auto end_symbol_idx = arithmetic::vocabulary.word2index[utils::end_symbol];

    std::cout << "     Vocabulary size : " << arithmetic::vocabulary.size() << std::endl
              << "      minibatch size : " << FLAGS_minibatch << std::endl
              << "   number of threads : " << FLAGS_j << std::endl
              << "          stack size : " << FLAGS_stack_size << std::endl
              << "        Dropout type : " << (FLAGS_fast_dropout ? "fast" : "default") << std::endl
              << " Max training epochs : " << FLAGS_epochs << std::endl
              << "           LSTM type : " << (model.memory_feeds_gates ? "Graves 2013" : "Zaremba 2014") << std::endl
              << "          Stack size : " << model.hidden_sizes.size() << std::endl
              << "         Hidden size : " << FLAGS_hidden << std::endl
              << "          Input size : " << FLAGS_input_size << std::endl
              << " # training examples : " << examples.size() << std::endl
              << "              Solver : " << FLAGS_solver << std::endl;

    Throttled throttled1;
    Throttled throttled2;

    while (epoch < FLAGS_epochs) {
        auto indices = utils::random_arange(examples.size());

        REAL_t minibatch_error = 0.0;
        // one minibatch
        for (auto indices_begin = indices.begin();
                indices_begin < indices.begin() + std::min((size_t)FLAGS_minibatch, examples.size());
                indices_begin++) {
            utils::Timer training_timer("train");
            // <training>
            auto& example = examples[*indices_begin];
            auto initial_state = model.initial_states();

            Mat<REAL_t> input_vector;
            for (auto& c : example.first) {
                input_vector = model.embedding[c];
                initial_state = model.stacked_lstm.activate(
                    initial_state,
                    input_vector
                );
            }
            auto error = MatOps<REAL_t>::softmax_cross_entropy_rowwise(
                model.decode(
                    input_vector,
                    initial_state
                ),
                example.second.front()
            );
            for (auto label_ptr = example.second.begin(); label_ptr < example.second.end() -1; label_ptr++) {
                input_vector = model.embedding[*label_ptr];
                initial_state = model.stacked_lstm.activate(
                    initial_state,
                    input_vector
                );
                error = error + MatOps<REAL_t>::softmax_cross_entropy_rowwise(
                    model.decode(
                        input_vector,
                        initial_state
                    ),
                    *(label_ptr+1)
                );
            }
            error.grad();
            graph::backward();
            training_timer.stop();
            minibatch_error += error.w(0);
            // </training>
            throttled1.maybe_run(seconds(2), [&]() {
                auto random_example_index = utils::randint(0, examples.size() -1);

                std::cout << arithmetic::vocabulary.decode(&examples[random_example_index].first) << std::endl;

                auto beams = arithmetic_beam_search(model, &examples[random_example_index].first);


                for (const auto& beam : beams) {
                    std::cout << "= (" << std::setprecision( 3 ) << beam.score << ") ";
                    for (const auto& word : beam.solution) {
                        if (word != arithmetic::vocabulary.word2index.at(utils::end_symbol))
                            std::cout << arithmetic::vocabulary.index2word.at(word);
                    }
                    std::cout << std::endl;
                }

            });
            throttled2.maybe_run(seconds(30), [&]() {
                auto predict = [&model](const vector<uint>& example) {
                    vector<uint> cpy(example);
                    return arithmetic_beam_search(model, &cpy)[0].solution;
                };
                auto correct = arithmetic::average_recall(examples, predict, FLAGS_j);
                std::cout << "epoch: " << epoch << " Percent correct = " << std::setprecision( 3 )  << 100.0 * correct << "%" << std::endl;
            });
        }
        solver->step(params); // One step of gradient descent
        epoch++;
    }
    utils::Timer::report();
}
