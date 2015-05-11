#include "dali/core.h"
#include "dali/utils.h"
#include "dali/data_processing/Arithmetic.h"

using std::string;
using std::vector;
using std::pair;
using std::to_string;
using std::make_shared;
using std::chrono::seconds;
using utils::MS;
using std::tuple;
using std::min;
using utils::assert2;

typedef float REAL_t;

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
DEFINE_int32(epochs,         2000,       "How many training loops through the full dataset ?");
DEFINE_int32(j,                 1,       "How many threads should be used ?");
DEFINE_int32(expression_length, 5,       "How much suffering to impose on our friend?");
DEFINE_int32(num_examples,      1500,    "How much suffering to impose on our friend?");
DEFINE_bool(memory_feeds_gates, true,    "LSTM's memory cell also control gate outputs");
DEFINE_int32(input_size,        50,      "Size of the word vectors");
DEFINE_int32(hidden,            100,     "How many Cells and Hidden Units should each LSTM have ?");

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

template<typename T>
class LeafModule {
    typedef typename LSTM<T>::State lstm_state_t;

    public:
        int input_size;
        int hidden_size;

        Layer<T> c_layer;
        Layer<T> o_layer;

        LeafModule<T>(int input_size, int hidden_size) :
                input_size(input_size),
                hidden_size(hidden_size),
                c_layer(input_size, hidden_size),
                o_layer(input_size, hidden_size) {
        }

        LeafModule<T>(const LeafModule<T>& other, bool copy_w, bool copy_dw) :
                input_size(other.input_size),
                hidden_size(other.hidden_size),
                c_layer(other.c_layer, copy_w, copy_dw),
                o_layer(other.o_layer, copy_w, copy_dw) {
        }

        LeafModule<T> shallow_copy() const {
            return LeafModule<T>(*this, false, true);
        }

        lstm_state_t activate(Mat<T> embedding) const {
            auto c = c_layer.activate(embedding);
            auto o = o_layer.activate(embedding).sigmoid();
            auto h = c.tanh() * o;
            return lstm_state_t(c,h);
        }

        vector<Mat<T>> parameters() const {
            vector<Mat<T>> res;
            auto c_layer_params = c_layer.parameters();
            auto o_layer_params = o_layer.parameters();
            res.insert(res.end(), c_layer_params.begin(), c_layer_params.end());
            res.insert(res.end(), o_layer_params.begin(), o_layer_params.end());
            return res;
        }
};

template<typename T>
struct BeamNode {
    typedef typename LSTM<T>::State lstm_state_t;

    Mat<T> log_probability;
    lstm_state_t state;

    BeamNode() : state(Mat<T>(), Mat<T>()) {}

    BeamNode(Mat<T> log_probability, lstm_state_t state) :
            log_probability(log_probability),
            state(state) {
    }

    BeamNode(lstm_state_t state) :
            log_probability(1,1),
            state(state) {
        // log probability by default initializes to 0.0 (p = 1.0).
        log_probability.constant = true;
    }
};

template<typename T>
class BeamLSTM : public LSTM<T> {
    typedef BeamNode<T> Node;
    public:
        int output_size;
        Layer<T> decoder;

        BeamLSTM(int input_size, int hidden_size, int output_size, bool memory_feeds_gates=false) :
                LSTM<T>(input_size, hidden_size, memory_feeds_gates),
                output_size(output_size),
                decoder(hidden_size, output_size) {
        }

        BeamLSTM(const BeamLSTM& other, bool copy_w, bool copy_dw) :
                LSTM<T>(other, copy_w, copy_dw),
                output_size(other.output_size),
                decoder(other.decoder, copy_w, copy_dw) {
        }

        BeamLSTM<T> shallow_copy() const {
            return BeamLSTM<T>(*this, false, true);
        }

        vector<Mat<T>> parameters() const {
            auto params = LSTM<T>::parameters();
            auto decoder_params = decoder.parameters();
            params.insert(params.end(), decoder_params.begin(), decoder_params.end());
            return params;
        }

        vector<Node> activate(Mat<T> input, const vector<Node>& states) const {
            int beam_width = states.size();

            vector<Node> new_state;
            new_state.reserve(states.size());

            std::transform(states.begin(), states.end(), std::back_inserter(new_state),
                    [this, &input](const Node& prev_node) {
                return Node(
                    prev_node.log_probability,
                    LSTM<T>::activate(input, prev_node.state)
                );
            });

            return new_state;
        }

        Mat<T> decode(typename LSTM<T>::State state) const {
            return MatOps<T>::softmax(decoder.activate(state.hidden));
        }

        Mat<T> decode(const vector<Node>& states) const {
            vector<Mat<T>> distributions;
            vector<Mat<T>> scores;

            for (auto& node: states) {
                distributions.emplace_back(MatOps<T>::softmax(decoder.activate(node.state.hidden)));
                scores.emplace_back(node.log_probability);
            }

            // softmax achieves dual purpose here:
            // item 1. normalizes the probability distributions
            // item 2. exponentiates to remove logs.
            auto probabilites = MatOps<T>::softmax(scores);

            auto weighted_distributions = MatOps<T>::eltmul_broadcast_rowwise(distributions, probabilites);

            return MatOps<T>::add(weighted_distributions);
        }
};

template<typename T>
class BeamTree {
    typedef BeamNode<T> Node;
    public:
        typedef typename LSTM<T>::State lstm_state_t;

        int input_size;
        int hidden_size;

        LeafModule<T> leaf_module;
        LSTM<T> composer;
        Layer<T> prob_decoder;

        BeamTree(int input_size, int hidden_size, bool memory_feeds_gates = false) :
                input_size(input_size),
                hidden_size(hidden_size),
                leaf_module(input_size, hidden_size),
                composer(vector<int>(), hidden_size, 2, memory_feeds_gates),
                prob_decoder(hidden_size, 1) {
        }

        BeamTree(const BeamTree<T>& other, bool copy_w, bool copy_dw) :
                input_size(other.input_size),
                hidden_size(other.hidden_size),
                leaf_module(other.leaf_module, copy_w, copy_dw),
                composer(other.composer, copy_w, copy_dw),
                prob_decoder(other.prob_decoder, copy_w, copy_dw) {
        }

        BeamTree<T> shallow_copy() const {
            return BeamTree<T>(*this, false, true);
        }

        vector<Node> convert_to_leaves(vector<Mat<T>> input) const {
            vector<Node> leaves;
            for (auto& embedding : input) {
                leaves.emplace_back(leaf_module.activate(embedding));
            }
            return leaves;
        }

        // The returned node is incomplete.
        lstm_state_t join_states(Node a, Node b) const {
            return composer.activate(
                vector<Mat<T>>(),
                {a.state, b.state}
            );
        }

        vector<vector<Node>> cangen(vector<Node> states, int beam_width) const {

            assert2(states.size() >= 2, "Must at least have 2 states to join for candidate generation.");
            int num_candidates = min((size_t)beam_width, states.size() - 1);

            vector<Node> possible_joins;
            vector<Mat<T>> scores;
            for (size_t sidx = 0; sidx + 1 < states.size(); ++sidx) {
                possible_joins.emplace_back(
                    Mat<T>(),
                    join_states(states[sidx], states[sidx + 1])
                );
                scores.emplace_back(prob_decoder.activate(possible_joins.back().state.hidden));
            }
            auto normalized_scores = MatOps<T>::softmax(scores);
            for (size_t sidx = 0; sidx + 1 < states.size(); ++sidx) {
                possible_joins[sidx].log_probability =
                        normalized_scores[sidx].log() +
                        states[sidx].log_probability +
                        states[sidx + 1].log_probability;
            }

            // initialize original index locations
            vector<size_t> idx(possible_joins.size());
            for (size_t i = 0; i < idx.size(); ++i)
                idx[i] = i;

            // sort indexes based on comparing values in v
            sort(idx.begin(), idx.end(), [&possible_joins](size_t i1, size_t i2) {
                return possible_joins[i1].log_probability.w()(0) > possible_joins[i2].log_probability.w()(0);
            });
            vector<vector<Node>> results;

            for (size_t cidx = 0; cidx < num_candidates; ++cidx) {
                vector<Node> result;
                size_t join_idx = idx[cidx];
                for (size_t sidx = 0; sidx < join_idx; ++sidx)
                    result.emplace_back(states[sidx]);
                result.emplace_back(possible_joins[join_idx]);
                for (size_t sidx = join_idx + 2; sidx < states.size(); ++sidx) {
                    result.emplace_back(states[sidx]);
                }
                assert(result.size() == states.size() - 1);
                results.emplace_back(result);
            }

            return results;
        }

        T candidate_log_probability(vector<Node> candidate) const {
            T result = 0.0;
            for (auto& node: candidate) {
                result += node.log_probability.w()(0);
            }
            return result;
        }

        vector<Node> best_trees(vector<Mat<T>> input, int beam_width) const {
            auto leaves = convert_to_leaves(input);
            vector<vector<Node>> candidates = { leaves };
            while (candidates[0].size() > 1) {
                vector<vector<Node>> new_candidates;
                for (auto& candidate: candidates) {
                    for (auto& new_candidate: cangen(candidate, beam_width)) {
                        new_candidates.emplace_back(new_candidate);
                    }
                }
                sort(new_candidates.begin(), new_candidates.end(),
                        [this](const vector<Node>& c1, const vector<Node>& c2) {
                    return candidate_log_probability(c1) > candidate_log_probability(c2);
                });
                candidates = vector<vector<Node>>(
                    new_candidates.begin(),
                    new_candidates.begin() + min((size_t)beam_width, new_candidates.size())
                );
                for (size_t cidx = 0; cidx + 1 < candidates.size(); ++cidx) {
                    assert2(candidates[cidx].size() == candidates[cidx + 1].size(),
                            "Generated candidates of different sizes.");
                }

            }
            vector<Node> result;
            for (auto& candidate: candidates) {
                result.emplace_back(candidate[0]);
            }
            return result;
        }

        vector<Mat<T>> parameters() const {
            vector<Mat<T>> params = leaf_module.parameters();

            auto composer_params = composer.parameters();
            params.insert(params.end(), composer_params.begin(), composer_params.end());

            auto prob_decoder_params = prob_decoder.parameters();
            params.insert(params.end(), prob_decoder_params.begin(), prob_decoder_params.end());

            return params;
        }

};

template class LeafModule<float>;
template class LeafModule<double>;

template class BeamLSTM<float>;
template class BeamLSTM<double>;

template class BeamTree<float>;
template class BeamTree<double>;

typedef vector<uint> sequence_t;
typedef std::pair<sequence_t, sequence_t> example_t;

template<typename T>
struct PredictionNode {
    BeamNode<T> node;
    sequence_t prediction;
    PredictionNode() {};
    PredictionNode(BeamNode<T> node) : node(node) {}
    PredictionNode(BeamNode<T> node, const sequence_t& prediction) : node(node), prediction(prediction) {}

    PredictionNode<T> make_choice(uint choice, BeamNode<T> node) {
        PredictionNode<T> fork(node, this->prediction);
        // add new choice to fork:
        fork.prediction.emplace_back(choice);
        return fork;
    }
};

template<typename T>
class ArithmeticModel {
    typedef BeamNode<T> Node;

    public:
        Mat<T> embedding;
        BeamTree<T> tree;
        BeamLSTM<T> decoder_lstm;

        ArithmeticModel(int input_size,
                        int hidden_size,
                        int vocab_size,
                        bool memory_feeds_gates = false) :
                embedding(vocab_size, input_size),
                decoder_lstm(input_size, hidden_size, vocab_size, memory_feeds_gates),
                tree(input_size, hidden_size, memory_feeds_gates) {
        }

        ArithmeticModel(const ArithmeticModel<T>& other, bool copy_w, bool copy_dw) :
                embedding(other.embedding, copy_w, copy_dw),
                decoder_lstm(other.decoder_lstm, copy_w, copy_dw),
                tree(other.tree, copy_w, copy_dw) {
        }

        ArithmeticModel<T> shallow_copy() const {
            return ArithmeticModel<T>(*this, false, true);
        }

        vector<Mat<T>> parameters() const {
            auto params = decoder_lstm.parameters();
            params.emplace_back(embedding);
            auto tree_params = tree.parameters();
            params.insert(params.end(), tree_params.begin(), tree_params.end());
            return params;
        }

        Mat<T> error(const example_t& example, int beam_width) const {
            auto expression_embedding = convert_to_embeddings(example.first);
            auto state = tree.best_trees(expression_embedding, beam_width);
            auto& targets = example.second;

            Mat<T> error(1,1);
            for (int aidx = 0; aidx < targets.size(); ++aidx) {
                Mat<T> prediction = decoder_lstm.decode(state);
                error = error + MatOps<T>::cross_entropy(prediction, targets[aidx]);
                if (aidx + 1 < targets.size()) {
                    state = decoder_lstm.activate(embedding[targets[aidx]], state);
                }
            }
            return error;
       }

       vector<Mat<T>> convert_to_embeddings(const vector<uint>& expression) const {
            vector<Mat<T>> embeddings;
            embeddings.reserve(expression.size());
            std::transform(expression.begin(), expression.end(), std::back_inserter(embeddings),
                    [this](uint embedding_idx) {
                return embedding[embedding_idx];
            });
            return embeddings;
       }

       vector<PredictionNode<T>> predict(
                const vector<uint>& expression,
                int beam_width,
                int max_output_length,
                uint end_symbol,
                uint ignore_symbol = -1) const {
            auto embeddings = convert_to_embeddings(expression);

            /* BEGIN (TRUE) BEAM SEARCH */
            vector<Node> candidate_trees = tree.best_trees(embeddings, beam_width);

            vector<PredictionNode<T>> candidates;

            candidates.reserve(candidate_trees.size());

            std::transform(candidate_trees.begin(), candidate_trees.end(), std::back_inserter(candidates),
                    [](const Node& node) {
                return PredictionNode<T>(node);
            });

            for (int sidx = 0; sidx < max_output_length; ++sidx) {
                vector<PredictionNode<T>> new_candidates;

                for (auto& candidate : candidates) {
                    if (candidate.prediction.size() > 0 && candidate.prediction.back() == end_symbol) {
                        new_candidates.emplace_back(candidate);
                    } else {
                        auto next_symbol_distribution = MatOps<T>::softmax(
                            decoder_lstm.decode(candidate.node.state)
                        );
                        vector<uint> predicted_symbols;
                        for (size_t pidx = 0; pidx < next_symbol_distribution.dims(0); ++pidx)
                            predicted_symbols.push_back(pidx);

                        std::sort(predicted_symbols.begin(), predicted_symbols.end(),
                                [&next_symbol_distribution](uint a, uint b) {
                            return next_symbol_distribution[a].w()(0) > next_symbol_distribution[b].w()(0);
                        });
                        int n_generated_candidates = std::min((uint) beam_width, next_symbol_distribution.dims(0));
                        for (int ncidx = 0; ncidx < n_generated_candidates; ++ncidx) {
                            uint candide_idx = predicted_symbols[ncidx];
                            // For each generated symbol within the top part of the beam
                            // we show this "winning" symbol to the decoding LSTM
                            // and advance the internal state by 1. Also we keep track of the
                            // probability of this fork, and update the predictions list.
                            new_candidates.emplace_back(
                                candidate.make_choice(
                                    candide_idx,
                                    Node(
                                        candidate.node.log_probability + next_symbol_distribution[candide_idx].log(),
                                        decoder_lstm.LSTM<T>::activate(embedding[candide_idx], candidate.node.state)
                                    )
                                )
                            );
                        }

                    }
                }

                std::sort(new_candidates.begin(), new_candidates.end(),
                        [](const PredictionNode<T>& a, const PredictionNode<T> b) {
                    return a.node.log_probability.w()(0) > b.node.log_probability.w()(0);
                });

                new_candidates.resize(beam_width);
                candidates = new_candidates;

                bool all_predictions_stopped = true;
                for (auto& prediction_node: candidates) {
                    all_predictions_stopped = all_predictions_stopped &&
                            prediction_node.prediction.back() == end_symbol;
                }
                if (all_predictions_stopped)
                    break;
            }
            return candidates;
       }
};

template class ArithmeticModel<float>;
// template class ArithmeticModel<double>;

typedef ArithmeticModel<REAL_t> model_t;

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

    auto examples = arithmetic::generate(FLAGS_num_examples, FLAGS_expression_length);
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

    auto vocab = arithmetic::vocabulary();

    // train a silly system to output the numbers it needs
    auto model = model_t(
        FLAGS_input_size,
        FLAGS_hidden,
        vocab.index2word.size(),
        FLAGS_memory_feeds_gates);



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

    vector<pair<vector<uint>, vector<uint>>> numerical_examples(examples.size());
    {
        for (size_t i = 0; i < examples.size();i++) {
            numerical_examples[i].first  = vocab.encode(examples[i].first, true);
            numerical_examples[i].second = vocab.encode(examples[i].second, true);
        }
    }

    int epoch = 0;
    auto end_symbol_idx = vocab.word2index[utils::end_symbol];

    std::cout << "     Vocabulary size : " << vocab.index2word.size() << std::endl
              << "      minibatch size : " << FLAGS_minibatch << std::endl
              << "   number of threads : " << FLAGS_j << std::endl
              << "        Dropout type : " << (FLAGS_fast_dropout ? "fast" : "default") << std::endl
              << " Max training epochs : " << FLAGS_epochs << std::endl
              << "           LSTM type : " << (model.tree.composer.memory_feeds_gates ? "Graves 2013" : "Zaremba 2014") << std::endl
              << "         Hidden size : " << FLAGS_hidden << std::endl
              << "          Input size : " << FLAGS_input_size << std::endl
              << " # training examples : " << examples.size() << std::endl
              << "              Solver : " << FLAGS_solver << std::endl;
    /*
    Throttled throttled1;
    Throttled throttled2;
    */

    int BEAM_WIDTH = 10;

    while (epoch < FLAGS_epochs) {

        auto indices = utils::random_arange(numerical_examples.size());
        auto indices_begin = indices.begin();

        REAL_t minibatch_error = 0.0;

        // one minibatch
        for (auto indices_begin = indices.begin(); indices_begin < indices.begin() + std::min((size_t)FLAGS_minibatch, numerical_examples.size()); indices_begin++) {
            // <training>
            auto& example = numerical_examples[*indices_begin];

            auto error = model.error(example, BEAM_WIDTH);
            error.grad();
            graph::backward();
            minibatch_error += error.w()(0);
            // </training>
            // // <reporting>
            // throttled1.maybe_run(seconds(2), [&]() {
            //     auto random_example_index = utils::randint(0, examples.size() -1);
            //     auto beams = beam_search::beam_search(model,
            //         numerical_examples[random_example_index].first,
            //         20,
            //         0,  // offset symbols that are predicted before being refed (no = 0)
            //         5,
            //         vocab.word2index.at(utils::end_symbol) // when to stop the sequence
            //     );
            //     for (auto& val : examples[random_example_index].first) {
            //         std::cout << val << " ";
            //     }
            //     std::cout << std::endl;
            //     for (const auto& beam : beams) {
            //         std::cout << "= (" << std::setprecision( 3 ) << std::get<1>(beam) << ") ";
            //         for (const auto& word : std::get<0>(beam)) {
            //             if (word != vocab.word2index.at(utils::end_symbol))
            //                 std::cout << vocab.index2word.at(word);
            //         }
            //         std::cout << std::endl;
            //     }
            // });
            // throttled2.maybe_run(seconds(30), [&]() {
            //     std::cout << "epoch: " << epoch << " Percent correct = " << std::setprecision( 3 )  << 100.0 * num_correct(
            //         model, numerical_examples, 5, vocab.word2index.at(utils::end_symbol)
            //     ) << "%" << std::endl;
            // });
            // </reporting>
        }
        std::cout << "minibatch_error => " << minibatch_error << std::endl;
        solver->step(params); // One step of gradient descent
        epoch++;
    }
}
