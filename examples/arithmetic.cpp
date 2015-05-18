#include "dali/core.h"
#include "dali/utils.h"
#include "dali/models/StackedModel.h"
#include "dali/data_processing/Arithmetic.h"

using std::string;
using std::vector;
using std::pair;
using std::to_string;
using std::make_shared;
using std::chrono::seconds;

typedef double REAL_t;

vector<string> SYMBOLS = {"+", "*", "-"};
static int NUM_SYMBOLS = SYMBOLS.size();

static int ADADELTA_TYPE = 0;
static int ADAGRAD_TYPE = 1;
static int SGD_TYPE = 2;
static int ADAM_TYPE = 3;
static int RMSPROP_TYPE = 4;

DEFINE_string(solver,        "adadelta", "What solver to use (adadelta, sgd, adam)");
DEFINE_double(reg,           0.0,        "What penalty to place on L2 norm of weights?");
DEFINE_double(learning_rate, 0.01,       "Learning rate for SGD and Adagrad.");
DEFINE_int32(minibatch,      100,        "What size should be used for the minibatches ?");
DEFINE_bool(fast_dropout,    true,       "Use fast dropout?");
DEFINE_int32(epochs,             2000,
        "How many training loops through the full dataset ?");
DEFINE_int32(j,                  1,
        "How many threads should be used ?");

DEFINE_int32(expression_length, 5, "How much suffering to impose on our friend?");
DEFINE_int32(num_examples,      1500, "How much suffering to impose on our friend?");
DEFINE_int32(max_number_in_expression, 100000, "Maximum number used in mathematical expressions.");

ThreadPool* pool;

template<typename M>
double num_correct(M& model, vector<pair<vector<uint>, vector<uint>>> examples, int beam_width, uint stop_symbol) {
    std::atomic<int> correct(examples.size());
    for (size_t i = 0; i < examples.size(); i++) {
        pool->run([&model, i, &examples, &stop_symbol, &beam_width, &correct]() {
            auto beams = beam_search::beam_search(model,
                examples[i].first,
                20,
                0,  // offset symbols that are predicted
                    // before being refed (no = 0)
                beam_width,
                stop_symbol // when to stop the sequence
            );
            if (std::get<0>(beams[0]).size() == examples[i].second.size()) {
                for (auto beam_ptr = std::get<0>(beams[0]).begin(), example_ptr = examples[i].second.begin();
                    beam_ptr < std::get<0>(beams[0]).end() && example_ptr < examples[i].second.end();
                    beam_ptr++, example_ptr++) {
                    if (*beam_ptr != *example_ptr) {
                        correct--;
                        break;
                    }
                }
            } else {
                correct--;
            }
        });
    }
    pool->wait_until_idle();
    return (double) correct / (double) examples.size();
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

    auto ex_bonus = arithmetic::generate(FLAGS_num_examples, FLAGS_expression_length, 0, FLAGS_max_number_in_expression);

    for (auto& ex : ex_bonus) {
        std::cout << utils::join(std::get<0>(ex)) << " => " << utils::join(std::get<1>(ex)) << std::endl;
    }

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

    // Rho value, eps value, and gradient clipping value:
    std::shared_ptr<Solver::AbstractSolver<REAL_t>> solver;
    int solver_type;
    auto params = model.parameters();
    if (FLAGS_solver == "adadelta") {
        solver = make_shared<Solver::AdaDelta<REAL_t>>(params, 0.95, 1e-9, 100.0, (REAL_t) FLAGS_reg);
        solver_type = ADADELTA_TYPE;
    } else if (FLAGS_solver == "adam") {
        solver = make_shared<Solver::Adam<REAL_t>>(params, 0.1, 0.001, 1e-9, 100.0, (REAL_t) FLAGS_reg);
        solver_type = ADAM_TYPE;
    } else if (FLAGS_solver == "sgd") {
        solver = make_shared<Solver::SGD<REAL_t>>(params, 100.0, (REAL_t) FLAGS_reg);
        solver_type = SGD_TYPE;
        dynamic_cast<Solver::SGD<REAL_t>*>(solver.get())->step_size = FLAGS_learning_rate;
    } else if (FLAGS_solver == "adagrad") {
        solver = make_shared<Solver::AdaGrad<REAL_t>>(params, 1e-9, 100.0, (REAL_t) FLAGS_reg);
        solver_type = ADAGRAD_TYPE;
        dynamic_cast<Solver::AdaGrad<REAL_t>*>(solver.get())->step_size = FLAGS_learning_rate;
    } else {
        utils::exit_with_message("Did not recognize this solver type.");
    }

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
        auto indices_begin = indices.begin();

        REAL_t minibatch_error = 0.0;

        // one minibatch
        for (auto indices_begin = indices.begin(); indices_begin < indices.begin() + std::min((size_t)FLAGS_minibatch, examples.size()); indices_begin++) {
            // <training>
            auto& example = examples[*indices_begin];
            auto initial_state = model.initial_states();
            Mat<REAL_t> input_vector;
            for (auto& c : example.first) {
                input_vector = model.embedding[c];
                initial_state = model.stacked_lstm->activate(
                    initial_state,
                    input_vector
                );
            }
            auto error = MatOps<REAL_t>::softmax_cross_entropy(
                model.decode(
                    input_vector,
                    initial_state
                ),
                example.second.front()
            );
            for (auto label_ptr = example.second.begin(); label_ptr < example.second.end() -1; label_ptr++) {
                input_vector = model.embedding[*label_ptr];
                initial_state = model.stacked_lstm->activate(
                    initial_state,
                    input_vector
                );
                error = error + MatOps<REAL_t>::softmax_cross_entropy(
                    model.decode(
                        input_vector,
                        initial_state
                    ),
                    *(label_ptr+1)
                );
            }
            error.grad();
            graph::backward();
            minibatch_error += error.w()(0);
            // </training>
            // <reporting>
            throttled1.maybe_run(seconds(2), [&]() {
                auto random_example_index = utils::randint(0, examples.size() -1);
                auto beams = beam_search::beam_search(model,
                    examples[random_example_index].first,
                    20,
                    0,  // offset symbols that are predicted before being refed (no = 0)
                    5,
                    arithmetic::vocabulary.word2index.at(utils::end_symbol) // when to stop the sequence
                );

                std::cout << arithmetic::vocabulary.decode(examples[random_example_index].first) << std::endl;

                for (const auto& beam : beams) {
                    std::cout << "= (" << std::setprecision( 3 ) << std::get<1>(beam) << ") ";
                    for (const auto& word : std::get<0>(beam)) {
                        if (word != arithmetic::vocabulary.word2index.at(utils::end_symbol))
                            std::cout << arithmetic::vocabulary.index2word.at(word);
                    }
                    std::cout << std::endl;
                }
            });
            throttled2.maybe_run(seconds(30), [&]() {
                std::cout << "epoch: " << epoch << " Percent correct = " << std::setprecision( 3 )  << 100.0 * num_correct(
                    model, examples, 5, arithmetic::vocabulary.word2index.at(utils::end_symbol)
                ) << "%" << std::endl;
            });
            // </reporting>
        }
        solver->step(params); // One step of gradient descent
        epoch++;
    }
}
