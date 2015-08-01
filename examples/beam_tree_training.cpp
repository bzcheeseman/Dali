#include "dali/core.h"
#include "dali/utils.h"
#include "dali/data_processing/Arithmetic.h"
#include "dali/visualizer/visualizer.h"

using arithmetic::numeric_example_t;
using std::chrono::seconds;
using std::make_shared;
using std::min;
using std::pair;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::tuple;
using std::vector;
using utils::assert2;
using utils::MS;


typedef float REAL_t;

DEFINE_double(reg,                    0.0,     "What penalty to place on L2 norm of weights?");
DEFINE_int32(minibatch,               100,     "What size should be used for the minibatches ?");
DEFINE_int32(j,                       9,       "How many threads should be used ?");
DEFINE_int32(expression_length,       19,      "How much suffering to impose on our friend?");
DEFINE_int32(num_examples,            5000,    "How much suffering to impose on our friend?");
DEFINE_bool(memory_feeds_gates,       false,   "LSTM's memory cell also control gate outputs");
DEFINE_int32(input_size,              10,      "Size of the word vectors");
DEFINE_int32(graduation_time,         25000,    "How many epochs per difficulty class?");
DEFINE_int32(hidden,                  20,     "How many Cells and Hidden Units should each LSTM have?");
DEFINE_int32(beam_width,              10,      "Size of the training beam.");
DEFINE_bool(use_end_symbol,           false,   "Whether to use end symbol in expression when training.");
DEFINE_int32(visualizer_trees,        3,       "Max number of trees to show in visualizer per example.");
DEFINE_int32(max_number_in_expression, 100000, "Maximum number used in mathematical expressions.");

#define MAX_OUTPUT_LENGTH 10

shared_ptr<Visualizer> visualizer;

template<typename T>
class LeafModule {
    typedef typename LSTM<T>::activation_t lstm_state_t;

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

/**
Q: Should multiple trees pool their answers?
--------------------------------------------

It appears that beams reveal same answers through different
processes:

    3-4
    = (-1.6) -2

    = (-1.64) -1

    = (-2.81) 0

    = (-2.99) -1

    = (-3.1) -2
    = (-3.26) 3
    = (-3.28) 7
    = (-3.44) 2
    = (-3.44) 5
    = (-3.66) 6

Notice how -1 is not the predicted answer, but the cumulated
probability of both -1s being predicted does yield the correct
answer.
**/

template<typename T>
struct BeamNode {
    typedef typename LSTM<T>::activation_t lstm_state_t;

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

        Mat<T> decode(const vector<Node>& states) const {
            vector<Mat<T>> distributions;
            vector<Mat<T>> scores;

            for (auto& node: states) {
                distributions.emplace_back(
                    MatOps<T>::softmax_rowwise(
                        decoder.activate(node.state.hidden)
                    )
                );
                scores.emplace_back(node.log_probability);
            }

            // softmax achieves dual purpose here:
            // item 1. normalizes the probability distributions
            // item 2. exponentiates to remove logs.
            if (scores.size() == 1) {
                return distributions.front();
            } else {
                auto probabilites = MatOps<T>::softmax(scores);

                auto weighted_distributions = MatOps<T>::eltmul_broadcast_colwise(
                    distributions,
                    probabilites
                );

                return MatOps<T>::add(weighted_distributions);
            }
        }
};

template<typename T>
struct BeamTreeResult {
    BeamNode<T> node;
    vector<uint> derivation;

    BeamTreeResult(BeamNode<T> node, vector<uint> derivation) :
            node(node),
            derivation(derivation) {
    }

    static vector<BeamNode<T>> nodes(const vector<BeamTreeResult<T>>& results) {
        vector<BeamNode<T>> nodes;
        std::transform(results.begin(), results.end(), std::back_inserter(nodes),
                [](const BeamTreeResult<T>& result) {
            return result.node;
        });
        return nodes;
    }

    static vector<vector<uint>> derivations(const vector<BeamTreeResult<T>>& results) {
        vector<vector<uint>> nodes;
        std::transform(results.begin(), results.end(), std::back_inserter(nodes),
                [](const BeamTreeResult<T>& result) {
            return result.derivation;
        });
        return nodes;
    }
};

template class BeamTreeResult<float>;
template class BeamTreeResult<double>;

template<typename T>
class BeamTree {
    typedef BeamNode<T> Node;

    struct PartialTree {
        vector<Node> nodes;
        vector<uint> derivation;

        PartialTree(vector<BeamNode<T>> nodes,
                    vector<uint> derivation=vector<uint>()) :
                nodes(nodes),
                derivation(derivation) {
        }
    };


    public:
        typedef typename LSTM<T>::activation_t lstm_state_t;

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
        /**
        Given an ordered set of n nodes, find the best contiguous
        pairs to join to form n-1 nodes. Return the `beam_width`
        best set of nodes with the resulting join applied.

        Inputs
        ------

        vector<Node> states : nodes to join
        int      beam_width : number of joins to consider

        Outputs
        -------

        vector<Candidate> new states : new sets with joined nodes
        **/
        vector<PartialTree> cangen(PartialTree candidate, int beam_width) const {
            assert2(candidate.nodes.size() >= 2,
                    "Must at least have 2 states to join for candidate generation.");
            int num_candidates = min((size_t)beam_width, candidate.nodes.size() - 1);

            vector<Node> possible_joins;
            vector<Mat<T>> scores;
            for (size_t sidx = 0; sidx + 1 < candidate.nodes.size(); ++sidx) {
                possible_joins.emplace_back(
                    Mat<T>(),
                    join_states(candidate.nodes[sidx], candidate.nodes[sidx + 1])
                );
                scores.emplace_back(prob_decoder.activate(possible_joins.back().state.hidden));
            }
            auto normalized_scores = MatOps<T>::softmax(scores);
            for (size_t sidx = 0; sidx + 1 < candidate.nodes.size(); ++sidx) {
                possible_joins[sidx].log_probability =
                        normalized_scores[sidx].log() +
                        candidate.nodes[sidx].log_probability +
                        candidate.nodes[sidx + 1].log_probability;
            }

            // initialize original index locations
            vector<size_t> idx(possible_joins.size());
            for (size_t i = 0; i < idx.size(); ++i)
                idx[i] = i;

            // sort indexes based on comparing values in v
            sort(idx.begin(), idx.end(), [&possible_joins](size_t i1, size_t i2) {
                return possible_joins[i1].log_probability.w(0) > possible_joins[i2].log_probability.w(0);
            });
            vector<PartialTree> results;

            for (size_t cidx = 0; cidx < num_candidates; ++cidx) {
                vector<Node> result;
                size_t join_idx = idx[cidx];
                for (size_t sidx = 0; sidx < join_idx; ++sidx)
                    result.emplace_back(candidate.nodes[sidx]);
                result.emplace_back(possible_joins[join_idx]);
                for (size_t sidx = join_idx + 2; sidx < candidate.nodes.size(); ++sidx) {
                    result.emplace_back(candidate.nodes[sidx]);
                }
                assert(result.size() == candidate.nodes.size() - 1);
                auto new_derivation = candidate.derivation; // copy
                // here cidx encodes the decision we made to join nodes cidx and cidx + 1.
                new_derivation.push_back(cidx);
                results.emplace_back(PartialTree(result, new_derivation));
            }

            return results;
        }

        T candidate_log_probability(PartialTree candidate) const {
            T result = 0.0;
            for (auto& node: candidate.nodes) {
                result += node.log_probability.w(0);
            }
            return result;
        }

        vector<BeamTreeResult<T>> best_trees(vector<Mat<T>> input, int beam_width) const {
            auto leaves = convert_to_leaves(input);
            vector<PartialTree> candidates = { PartialTree(leaves) };
            while (candidates[0].nodes.size() > 1) {
                vector<PartialTree> new_candidates;
                for (auto& candidate: candidates) {
                    for (auto& new_candidate: cangen(candidate, beam_width)) {
                        new_candidates.emplace_back(new_candidate);
                    }
                }
                sort(new_candidates.begin(), new_candidates.end(),
                        [this](const PartialTree& c1, const PartialTree& c2) {
                    return candidate_log_probability(c1) > candidate_log_probability(c2);
                });
                candidates = vector<PartialTree>(
                    new_candidates.begin(),
                    new_candidates.begin() + min((size_t)beam_width, new_candidates.size())
                );
                for (size_t cidx = 0; cidx + 1 < candidates.size(); ++cidx) {
                    assert2(candidates[cidx].nodes.size() == candidates[cidx + 1].nodes.size(),
                            "Generated candidates of different sizes.");
                }
            }
            vector<BeamTreeResult<T>> results;
            for (auto& tree: candidates) {
                results.emplace_back(tree.nodes[0], tree.derivation);
            }
            return results;
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

template<typename T>
struct PredictionNode {
    vector<BeamNode<T>> nodes;
    sequence_t prediction;
    vector<vector<uint>> derivations;

    PredictionNode() {};
    PredictionNode(vector<BeamNode<T>> nodes, vector<vector<uint>> derivations) :
            nodes(nodes),
            derivations(derivations) {
    }
    PredictionNode(vector<BeamNode<T>> nodes, vector<vector<uint>> derivations, const sequence_t& prediction) :
            nodes(nodes),
            derivations(derivations),
            prediction(prediction) {
    }
    PredictionNode<T> make_choice(uint choice, vector<BeamNode<T>> nodes) const {
        PredictionNode<T> fork(nodes, this->derivations, this->prediction);
        // add new choice to fork:
        fork.prediction.emplace_back(choice);
        return fork;
    }

    Mat<T> get_probability() const {
        vector<Mat<T>> probabilities;
        std::transform(nodes.begin(), nodes.end(), std::back_inserter(probabilities),
                [](const BeamNode<T>& node) {
            return node.log_probability.exp();
        });
        return MatOps<T>::add(probabilities);
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

        Mat<T> error(const arithmetic::numeric_example_t& example, int beam_width) const {
            auto expression_embedding = convert_to_embeddings(example.first);
            auto candidates = BeamTreeResult<T>::nodes(tree.best_trees(expression_embedding, beam_width));
            auto& targets = example.second;

            Mat<T> error(1,1);
            error.constant = true;
            for (int aidx = 0; aidx < targets.size(); ++aidx) {
                // TODO: each error should act individually (ErrorTotal = Sum { error[idx] * prob[idx] } )
                Mat<T> prediction = decoder_lstm.decode(candidates);
                error = error + MatOps<T>::cross_entropy_rowwise(prediction, targets[aidx]);

                if (aidx + 1 < targets.size()) {
                    candidates = decoder_lstm.activate(embedding[targets[aidx]], candidates);
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
            auto candidate_trees = tree.best_trees(embeddings, beam_width);

            vector<PredictionNode<T>> candidates;
            candidates.emplace_back(
                BeamTreeResult<T>::nodes(candidate_trees),
                BeamTreeResult<T>::derivations(candidate_trees)
            );


            for (int sidx = 0; sidx < max_output_length; ++sidx) {
                vector<PredictionNode<T>> new_candidates;

                for (auto& candidate : candidates) {
                    if (candidate.prediction.size() > 0 && candidate.prediction.back() == end_symbol) {
                        new_candidates.emplace_back(candidate);
                    } else {
                        auto next_symbol_distribution = decoder_lstm.decode(candidate.nodes);

                        vector<uint> predicted_symbols;
                        for (size_t pidx = 0; pidx < next_symbol_distribution.dims(1); ++pidx)
                            predicted_symbols.push_back(pidx);

                        std::sort(predicted_symbols.begin(), predicted_symbols.end(),
                                [&next_symbol_distribution](const uint& a, const uint& b) {
                            return next_symbol_distribution.w(a) > next_symbol_distribution.w(b);
                        });
                        int n_generated_candidates = std::min(
                            (uint) beam_width,
                            next_symbol_distribution.dims(1)
                        );
                        for (int ncidx = 0; ncidx < n_generated_candidates; ++ncidx) {
                            uint candidate_idx = predicted_symbols[ncidx];
                            // For each generated symbol within the top part of the beam
                            // we show this "winning" symbol to the decoding LSTM
                            // and advance the internal state by 1. Also we keep track of the
                            // probability of this fork, and update the predictions list.
                            auto new_nodes = decoder_lstm.activate(embedding[candidate_idx], candidate.nodes);
                            for (auto& node: new_nodes) {
                                node.log_probability = node.log_probability + next_symbol_distribution(NULL, candidate_idx).log();
                            }
                            new_candidates.emplace_back(
                                candidate.make_choice(
                                    candidate_idx,
                                    new_nodes
                                )
                            );
                        }
                    }
                }

                std::sort(new_candidates.begin(), new_candidates.end(),
                        [](const PredictionNode<T>& a, const PredictionNode<T> b) {
                    return a.get_probability().w(0) > b.get_probability().w(0);
                });
                if (new_candidates.size() > beam_width)
                    new_candidates.resize(beam_width);
                candidates = new_candidates;

                bool all_predictions_stopped = true;
                for (auto& prediction_node: candidates) {
                    all_predictions_stopped = all_predictions_stopped &&
                            prediction_node.prediction.back() == end_symbol;
                    // cut out early if we noticed that some did not stop
                    if (!all_predictions_stopped)
                        break;
                }
                if (all_predictions_stopped)
                    break;
            }
            return candidates;
       }
};

template class ArithmeticModel<float>;
template class ArithmeticModel<double>;

typedef ArithmeticModel<REAL_t> model_t;

shared_ptr<visualizable::Tree> visualize_derivation(vector<uint> derivation, vector<string> words) {
    using visualizable::Tree;

    vector<shared_ptr<Tree>> result;
    std::transform(words.begin(), words.end(), std::back_inserter(result),
            [](const string& a) {
        return make_shared<Tree>(a);
    });
    for (auto merge_idx : derivation) {
        vector<shared_ptr<Tree>> new_result;
        for(size_t ridx = 0; ridx < merge_idx; ++ridx) {
            new_result.push_back(result[ridx]);
        }
        new_result.push_back(make_shared<Tree>(std::initializer_list<shared_ptr<Tree>>{
            result[merge_idx],
            result[merge_idx + 1]
        }));
        for(size_t ridx = merge_idx + 2; ridx < result.size(); ++ridx) {
            new_result.push_back(result[ridx]);
        }
        result = new_result;
    }
    assert2(result.size() == 1, "Szymon messed up.");

    return result[0];
}

void training_loop(std::shared_ptr<Solver::AbstractSolver<REAL_t>> solver,
        model_t& model,
        std::function<vector<uint>(vector<uint>&)> pred_fun,
        vector<numeric_example_t>& train,
        vector<numeric_example_t>& validate) {
    auto& vocab = arithmetic::vocabulary;

    auto params = model.parameters();

    int epoch = 0;
    int difficulty_waiting = 0;
    auto end_symbol_idx = vocab.word2index[utils::end_symbol];

    int beam_width = FLAGS_beam_width;

    if (beam_width < 1)
        utils::exit_with_message(MS() << "Beam width must be strictly positive (got " << beam_width << ")");

    Throttled throttled_examples;
    Throttled throttled_validation;

    bool target_accuracy_reached = false;

    while (!target_accuracy_reached && epoch++ < FLAGS_graduation_time) {

        auto indices = utils::random_arange(train.size());
        auto indices_begin = indices.begin();

        REAL_t minibatch_error = 0.0;

        // one minibatch
        for (auto indices_begin = indices.begin();
                indices_begin < indices.begin() + std::min((size_t)FLAGS_minibatch, train.size());
                indices_begin++) {
            // <training>
            auto& example = train[*indices_begin];

            auto error = model.error(example, beam_width);
            error.grad();
            graph::backward();
            minibatch_error += error.w(0);
            // </training>
            // // <reporting>
            throttled_examples.maybe_run(seconds(10), [&]() {
                graph::NoBackprop nb;
                auto random_example_index = utils::randint(0, validate.size() -1);
                auto& expression = validate[random_example_index].first;
                auto predictions = model.predict(expression,
                                                 beam_width,
                                                 MAX_OUTPUT_LENGTH,
                                                 vocab.word2index.at(utils::end_symbol));

                auto expression_string = arithmetic::vocabulary.decode(&expression);
                if (expression_string.back() == utils::end_symbol)
                    expression_string.resize(expression_string.size() - 1);
                std::cout << utils::join(expression_string) << std::endl;


                vector<string> prediction_string;
                vector<double> prediction_probability;

                for (auto& prediction : predictions) {
                    if (validate[random_example_index].second == prediction.prediction) {
                        std::cout << utils::green;
                    }
                    prediction_probability.push_back(prediction.get_probability().w(0));
                    std::cout << "= (" << std::setprecision( 3 ) << prediction.get_probability().log().w(0) << ") ";
                    auto digits = vocab.decode(&prediction.prediction);
                    if (digits.back() == utils::end_symbol)
                        digits.pop_back();
                    auto joined_digits = utils::join(digits);
                    prediction_string.push_back(joined_digits);
                    std::cout << joined_digits << utils::reset_color << std::endl;
                }
                auto vgrid = make_shared<visualizable::GridLayout>();

                assert2(predictions[0].derivations.size() == predictions[0].nodes.size(),
                        "Szymon messed up.");
                for (int didx = 0;
                        didx < min((size_t)FLAGS_visualizer_trees, predictions[0].derivations.size());
                        ++didx) {
                    auto visualization = visualize_derivation(
                            predictions[0].derivations[didx],
                            vocab.decode(&expression)
                    );
                    auto tree_prob = predictions[0].nodes[didx].log_probability.exp().w(0,0);
                    vgrid->add_in_column(0, make_shared<visualizable::Probability<double>>(tree_prob));
                    vgrid->add_in_column(0, visualization);
                }
                vgrid->add_in_column(1, make_shared<visualizable::Sentence<double>>(expression_string));
                vgrid->add_in_column(1, make_shared<visualizable::FiniteDistribution<double>>(
                    prediction_probability,
                    prediction_string
                ));

                if (visualizer)
                    visualizer->feed(vgrid->to_json());

            });
            double current_accuracy = -1;
            throttled_validation.maybe_run(seconds(30), [&]() {
                current_accuracy = arithmetic::average_recall(validate, pred_fun, FLAGS_j);
                std::cout << "epoch: " << epoch << ", accuracy = " << std::setprecision( 3 )
                          << 100.0 * current_accuracy << "%" << std::endl;
            });
            if (current_accuracy != -1 && current_accuracy > 0.9) {
                std::cout << "Current accuracy is now " << current_accuracy << std::endl;
                target_accuracy_reached = true;
                break;
            }
            // </reporting>
        }
        solver->step(params); // One step of gradient descent
        epoch++;
    }
}

void increase_dataset_difficulty(vector<numeric_example_t>& dataset,
                                 int new_difficulty,
                                 int target_size) {
    random_shuffle(dataset.begin(), dataset.end());
    dataset.erase(dataset.begin() + dataset.size() / 2, dataset.end());
    auto new_examples =
            arithmetic::generate_numerical(target_size - dataset.size(),
                                           new_difficulty,
                                           0,
                                           FLAGS_max_number_in_expression,
                                           FLAGS_use_end_symbol);
    for (auto& new_example : new_examples) {
        dataset.emplace_back(new_example);
    }
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

    auto model = model_t(
        FLAGS_input_size,
        FLAGS_hidden,
        arithmetic::vocabulary.size(),
        FLAGS_memory_feeds_gates);

    auto pred_fun = [&model](vector<uint>& example) {
        graph::NoBackprop nb;
        auto predictions = model.predict(example,
                                         FLAGS_beam_width,
                                         MAX_OUTPUT_LENGTH,
                                         arithmetic::vocabulary.word2index.at(utils::end_symbol));
        return predictions[0].prediction;
    };

    auto params = model.parameters();
    auto solver = Solver::construct(FLAGS_solver, params, (REAL_t) FLAGS_learning_rate, (REAL_t) FLAGS_reg);

    if (!FLAGS_visualizer.empty())
        visualizer = make_shared<Visualizer>(FLAGS_visualizer);

    std::cout << "     Vocabulary size : " << arithmetic::vocabulary.size() << std::endl
              << "      minibatch size : " << FLAGS_minibatch << std::endl
              << "   number of threads : " << FLAGS_j << std::endl
              << "           LSTM type : " << (model.tree.composer.memory_feeds_gates ? "Graves 2013" : "Zaremba 2014") << std::endl
              << "         Hidden size : " << FLAGS_hidden << std::endl
              << "          Input size : " << FLAGS_input_size << std::endl
              << " examples/difficulty : " << FLAGS_num_examples << std::endl
              << "              Solver : " << FLAGS_solver << std::endl;

    vector<numeric_example_t> train, validate, test;

    for (int difficulty = 1; difficulty < FLAGS_expression_length; difficulty += 2) {
        increase_dataset_difficulty(train,    difficulty, FLAGS_num_examples);
        increase_dataset_difficulty(validate, difficulty, FLAGS_num_examples / 10);
        increase_dataset_difficulty(test,     difficulty, FLAGS_num_examples / 10);

        std::cout << "Increasing difficulty to " << difficulty << "." << std::endl;
        training_loop(solver, model, pred_fun, train, validate);
        std::cout << "Test accuracy on difficulty 1 up to " << difficulty << " is "
                  << 100.0 * arithmetic::average_recall(test, pred_fun, FLAGS_j) << "%" << std::endl;
        for (int old_difficulty = 1; old_difficulty <=difficulty; old_difficulty += 2) {
            auto old_examples = arithmetic::generate_numerical(FLAGS_num_examples / 10, old_difficulty, FLAGS_use_end_symbol);
                    std::cout << "Test accuracy on difficulty " << old_difficulty << " when trained on difficulty up to "
                              << difficulty << " is "
                              << 100.0 * arithmetic::average_recall(old_examples, pred_fun, FLAGS_j) << "%" << std::endl;
        }
    }
}
