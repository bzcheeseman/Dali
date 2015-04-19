#include <algorithm>
#include <atomic>
#include <Eigen/Eigen>
#include <fstream>
#include <ostream>
#include <fstream>
#include <iterator>
#include <chrono>
#include <vector>
#include <memory>
#include <string>

#include "dali/core.h"
#include "dali/utils.h"
#include "dali/models/StackedModel.h"

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


vector<pair<vector<string>, vector<string>>> generate_examples(int num) {
    vector<pair<vector<string>, vector<string>>> examples;
    int i = 0;
    while (i < num) {
        vector<string> example;
        auto expr_length = utils::randint(1, std::max(1, FLAGS_expression_length));
        bool use_operator = false;
        for (int j = 0; j < expr_length; j++) {
            if (use_operator) {
                auto operation = SYMBOLS[utils::randint(0, NUM_SYMBOLS-1)];
                example.push_back(operation);
                use_operator = false;
            } else {
                auto value = to_string(utils::randint(0, 9));
                example.push_back(value);
                use_operator = true;
            }
        }
        if (!use_operator) {
            auto value = to_string(utils::randint(0, 9));
            example.push_back(value);
            use_operator = true;
        }

        int result = 0;

        {
            int product_so_far = 1;
            vector<string> multiplied;
            for (auto& character : example) {
                if (utils::in_vector(SYMBOLS, character)) {
                    if (character == "*") {
                        // do nothing
                    } else {
                        multiplied.push_back(to_string(product_so_far));
                        multiplied.push_back(character);
                        product_so_far = 1;
                    }
                } else {
                    product_so_far *= character[0] - '0';
                }
            }
            multiplied.push_back(to_string(product_so_far));

            string last_operator = "";
            for (auto& character: multiplied) {
                if (utils::in_vector(SYMBOLS, character)) {
                    last_operator = character;
                } else {
                    if (last_operator == "") {
                        result = std::stoi(character);
                    } else if (last_operator == "+") {
                        result += std::stoi(character);
                    } else if (last_operator == "-") {
                        result -= std::stoi(character);
                    } else {
                        assert(NULL == "Unknown operator.");
                    }
                }
            }
        }

        if (result > -50 && result < 50) {
            i++;

            auto res = to_string(result);

            vector<string> character_result;
            for (int j = 0; j < res.size(); j++) {
                character_result.emplace_back(res.begin()+j, res.begin()+j+1);
            }
            examples.emplace_back(
                example,
                character_result
            );
        }
    }
    return examples;
}

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

    auto examples = generate_examples(FLAGS_num_examples);
    pool = new ThreadPool(FLAGS_j);

    // display the examples:
    for (auto example_ptr = examples.begin(); example_ptr < examples.end() && example_ptr < examples.begin() + 20; example_ptr++) {
        for (auto& val : example_ptr->first) {
            std::cout << val << " ";
        }
        std::cout << "= ";
        for (auto& val : example_ptr->second) {
            std::cout << val;
        }
        std::cout << std::endl;
    }

    // define symbols:
    vector<string> symbols;
    for (int i = 0; i < 10; i++) {
        symbols.push_back(to_string(i));
    }
    symbols.insert(symbols.end(), SYMBOLS.begin(), SYMBOLS.end());
    symbols.push_back(utils::end_symbol);
    std::cout << symbols << std::endl;

    utils::Vocab vocab(symbols, false);

    // train a silly system to output the numbers it needs
    auto model = StackedModel<REAL_t>(
         vocab.index2word.size(),
         FLAGS_input_size,
         FLAGS_hidden,
         FLAGS_stack_size,
         vocab.index2word.size(),
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
    } else if (FLAGS_solver == "adagrad") {
        solver = make_shared<Solver::AdaGrad<REAL_t>>(params, 1e-9, 100.0, (REAL_t) FLAGS_reg);
        solver_type = ADAGRAD_TYPE;
        dynamic_cast<Solver::AdaGrad<REAL_t>*>(solver.get())->step_size = FLAGS_learning_rate;
    } else {
        utils::exit_with_message("Did not recognize this solver type.");
    }

    vector<pair<vector<uint>, vector<uint>>> numerical_examples(examples.size());
    {
        for (size_t i = 0; i < examples.size();i++) {
            numerical_examples[i].first  = vocab.encode(examples[i].first, true);
            numerical_examples[i].second = vocab.encode(examples[i].second, true);
        }
    }

    for (auto example_ptr = numerical_examples.begin(); example_ptr < numerical_examples.end() && example_ptr < numerical_examples.begin() + 20; example_ptr++) {
        for (auto& val : example_ptr->first) {
            std::cout << val << " ";
        }
        for (auto& val : example_ptr->second) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    int epoch = 0;
    auto end_symbol_idx = vocab.word2index[utils::end_symbol];

    std::cout << "     Vocabulary size : " << symbols.size() << std::endl
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

        auto indices = utils::random_arange(numerical_examples.size());
        auto indices_begin = indices.begin();

        REAL_t minibatch_error = 0.0;

        // one minibatch
        for (auto indices_begin = indices.begin(); indices_begin < indices.begin() + std::min((size_t)FLAGS_minibatch, numerical_examples.size()); indices_begin++) {
            // <training>
            auto& example = numerical_examples[*indices_begin];
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
                model.decoder->activate(
                    input_vector,
                    LSTM<REAL_t>::State::hiddens(initial_state)
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
                    model.decoder->activate(
                        input_vector,
                        LSTM<REAL_t>::State::hiddens(initial_state)
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
                    numerical_examples[random_example_index].first,
                    20,
                    0,  // offset symbols that are predicted before being refed (no = 0)
                    5,
                    vocab.word2index.at(utils::end_symbol) // when to stop the sequence
                );
                for (auto& val : examples[random_example_index].first) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
                for (const auto& beam : beams) {
                    std::cout << "= (" << std::setprecision( 3 ) << std::get<1>(beam) << ") ";
                    for (const auto& word : std::get<0>(beam)) {
                        if (word != vocab.word2index.at(utils::end_symbol))
                            std::cout << vocab.index2word.at(word);
                    }
                    std::cout << std::endl;
                }
            });
            throttled2.maybe_run(seconds(30), [&]() {
                std::cout << "epoch: " << epoch << " Percent correct = " << std::setprecision( 3 )  << 100.0 * num_correct(
                    model, numerical_examples, 5, vocab.word2index.at(utils::end_symbol)
                ) << "%" << std::endl;
            });
            // </reporting>
        }
        solver->step(params); // One step of gradient descent
        epoch++;
    }
}
