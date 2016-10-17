#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/layers/conv.h"
#include "dali/layers/gru.h"
#include "dali/layers/layers.h"
#include "dali/layers/lstm.h"
#include "dali/tensor/op.h"
#include "dali/tensor/tensor.h"
#include "dali/test_utils.h"
#include "dali/utils/random.h"

using std::vector;
using std::chrono::milliseconds;

typedef MemorySafeTest LayerTests;

TEST(LayerTests, layer) {
    int num_examples = 7;
    int hidden_size  = 10;
    int input_size   = 5;

    EXPERIMENT_REPEAT {
        auto X  = Tensor::uniform(-20,20, {num_examples, input_size}, DTYPE_DOUBLE);
        auto mylayer = Layer(input_size, hidden_size, DTYPE_DOUBLE);
        auto params = mylayer.parameters();
        params.emplace_back(X);
        auto functor = [&mylayer](vector<Tensor> Xs)-> Tensor {
            return mylayer.activate(Xs.back());
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-3));
    }
}

TEST(LayerTests, rnn) {
    int num_examples = 7;
    int hidden_size  = 10;
    int input_size   = 5;

    EXPERIMENT_REPEAT {
        auto rnn = RNN(input_size, hidden_size, DTYPE_DOUBLE);
        Tensor X   = Tensor::uniform(-20, 20, {num_examples, input_size}, DTYPE_DOUBLE);
        Tensor H   = Tensor::uniform(-20, 20, {hidden_size}, DTYPE_DOUBLE)[Broadcast()];
        auto params = rnn.parameters();
        params.emplace_back(X);
        params.emplace_back(H);
        auto functor = [&rnn, &X, &H](vector<Tensor> Xs)-> Tensor {
            return rnn.activate(X, H);
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-3));
    }
}


TEST(LayerTests, rnn_multipass) {
    int num_examples = 2;
    int hidden_size  = 5;
    int input_size   = 3;

    EXPERIMENT_REPEAT {
        auto rnn = RNN(input_size, hidden_size, DTYPE_DOUBLE);
        Tensor X   = Tensor::uniform(-20, 20, {num_examples, input_size}, DTYPE_DOUBLE);
        Tensor H   = Tensor::uniform(-20, 20, {hidden_size}, DTYPE_DOUBLE)[Broadcast()];
        auto params = rnn.parameters();
        params.emplace_back(X);
        params.emplace_back(H);
        auto functor = [&rnn, &X, &H](vector<Tensor> Xs)-> Tensor {
            auto prev_h = H;
            for (int i = 0; i < 10; ++i) {
                prev_h = rnn.activate(X, prev_h);
            }
            return prev_h;
        };
        ASSERT_TRUE(gradient_same(functor, params, 5e-3));
    }
}

TEST(LayerTests, rnn_multipass2) {
    int input_size = 3;
    int hidden_size = 5;
    int tsteps = 5;

    EXPERIMENT_REPEAT {
        auto rnn = RNN(input_size, hidden_size, DTYPE_DOUBLE);
        auto params = rnn.parameters();
        auto inputs = vector<Tensor>();
        for (int i = 0; i < tsteps; i++) {
            inputs.emplace_back(Tensor::uniform(-10.0, 10.0, {1, input_size}, DTYPE_DOUBLE));
        }
        auto functor = [&inputs, &rnn,&tsteps, &hidden_size, &input_size](vector<Tensor> Xs)-> Tensor {
            auto state = rnn.initial_states();
            for (int i = 0; i < tsteps; i++)
                state = rnn.activate(inputs[i], state);
            return (state -1.0) ^ 2;
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-3));
    }
}


TEST(LayerTests, BroadcastMultiply) {
    int large_size = 7;
    int out_size   = 10;

    // different input sizes passed to a stacked input layer
    vector<int> input_sizes   = {5,  2,  5,  1, 5};

    // broadcast the 1s into the larger dimension:
    vector<int> example_sizes = {large_size, 1, large_size, 1, large_size};

    EXPERIMENT_REPEAT {
        // build layer
        auto mylayer = StackedInputLayer(input_sizes, out_size, DTYPE_DOUBLE);

        // build inputs
        vector<Tensor> inputs;
        for (int i = 0; i < input_sizes.size(); i++) {
            if (example_sizes[i] == 1) {
                inputs.emplace_back(
                    Tensor::uniform(2.5, {input_sizes[i]}, DTYPE_DOUBLE)[Broadcast()]
                );
            } else {
                inputs.emplace_back(
                    Tensor::uniform(2.5, {example_sizes[i], input_sizes[i]}, DTYPE_DOUBLE)
                );
            }
        }

        // add the params
        auto params = mylayer.parameters();
        params.insert(params.end(), inputs.begin(), inputs.end());

        // project
        auto functor = [&mylayer, &inputs](vector<Tensor> Xs)-> Tensor {
            return mylayer.activate(inputs);
        };
        ASSERT_TRUE(gradient_same(functor, params, 0.0003));
    }
}

TEST(LayerTests, second_order_combinator) {

    int num_examples = 10;
    int hidden_size  = 10;
    int input_size_1 = 5;
    int input_size_2 = 8;

    EXPERIMENT_REPEAT {
        auto A  = Tensor::uniform(10.0, {num_examples, input_size_1}, DTYPE_DOUBLE);
        auto B  = Tensor::uniform(10.0, {num_examples, input_size_2}, DTYPE_DOUBLE);
        auto mylayer = SecondOrderCombinator(
            input_size_1,
            input_size_2,
            hidden_size,
            DTYPE_DOUBLE
        );
        auto params = mylayer.parameters();
        params.emplace_back(A);
        params.emplace_back(B);
        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return mylayer.activate(A, B);
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-3));
    }
}

TEST(LayerTests, stacked_layer_tanh_gradient) {
    int num_examples = 6;
    int hidden_size  = 6;
    int input_size_1 = 2;
    int input_size_2 = 3;
    int input_size_3 = 4;

    EXPERIMENT_REPEAT {
        auto A  = Tensor::uniform(10.0, {num_examples, input_size_1}, DTYPE_DOUBLE);
        auto B  = Tensor::uniform(10.0, {num_examples, input_size_2}, DTYPE_DOUBLE);
        auto C  = Tensor::uniform(10.0, {num_examples, input_size_3}, DTYPE_DOUBLE);
        auto mylayer = StackedInputLayer(
            {
                input_size_1,
                input_size_2,
                input_size_3
            },
            hidden_size,
            DTYPE_DOUBLE
        );
        auto params = mylayer.parameters();
        params.emplace_back(A);
        params.emplace_back(B);
        params.emplace_back(C);
        auto functor = [&mylayer, &A, &B, &C](vector<Tensor> Xs)-> Tensor {
            return mylayer.activate({A, B, C}).tanh();
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-3));
    }
}
//
TEST(LayerTests, LSTM_Zaremba_gradient) {
    utils::random::set_seed(1234);

    int num_examples           = 5;
    int hidden_size            = 2;
    int input_size             = 3;

    EXPERIMENT_REPEAT {
        auto X  = Tensor::uniform(10.0, {num_examples, input_size}, DTYPE_DOUBLE);
        auto mylayer = LSTM(input_size, hidden_size, false, DTYPE_DOUBLE);
        auto params = mylayer.parameters();
        params.emplace_back(X);
        auto initial_state = LSTMState(
            Tensor::uniform(0.05, {hidden_size}, DTYPE_DOUBLE)[Broadcast()],
            Tensor::uniform(0.05, {hidden_size}, DTYPE_DOUBLE)[Broadcast()]
        );
        auto functor = [&mylayer, &X, &initial_state](vector<Tensor> Xs)-> Tensor {
            auto myout_state = mylayer.activate(X, initial_state);
            return myout_state.hidden;
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-3));
    }
}

TEST(LayerTests, LSTM_Graves_gradient) {
    int num_examples           = 2;
    int hidden_size            = 5;
    int input_size             = 3;

    EXPERIMENT_REPEAT {
        auto X  = Tensor::uniform(10.0, {num_examples, input_size}, DTYPE_DOUBLE);
        auto mylayer = LSTM(input_size, hidden_size, true, DTYPE_DOUBLE);
        auto params = mylayer.parameters();
        params.emplace_back(X);

        // In an LSTM we do not back prop through the cell activations when using
        // it in a gate, but to test it here we set this to true
        mylayer.backprop_through_gates = true;

        auto initial_state = LSTMState(
            Tensor::uniform(0.05, {hidden_size}, DTYPE_DOUBLE)[Broadcast()],
            Tensor::uniform(0.05, {hidden_size}, DTYPE_DOUBLE)[Broadcast()]
        );
        auto functor = [&mylayer, &X, &initial_state](vector<Tensor> Xs)-> Tensor {
            auto myout_state = mylayer.activate(X, initial_state);
            return myout_state.hidden;
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-3));
    }
}

TEST(LayerTests, LSTM_Graves_shortcut_gradient) {
    int num_examples           = 10;
    int hidden_size            = 5;
    int input_size             = 3;
    int shortcut_size          = 2;

    EXPERIMENT_REPEAT {
        auto X   = Tensor::uniform(10.0, {num_examples,  input_size},    DTYPE_DOUBLE);
        auto X_s = Tensor::uniform(10.0, {num_examples,  shortcut_size}, DTYPE_DOUBLE);

        auto mylayer = LSTM({input_size, shortcut_size}, hidden_size, 1, true, DTYPE_DOUBLE);

        auto params = mylayer.parameters();
        params.emplace_back(X);
        params.emplace_back(X_s);

        // In an LSTM we do not back prop through the cell activations when using
        // it in a gate:
        mylayer.backprop_through_gates = true;

        auto initial_state = LSTMState(
            Tensor::uniform(0.05, {hidden_size}, DTYPE_DOUBLE)[Broadcast()],
            Tensor::uniform(0.05, {hidden_size}, DTYPE_DOUBLE)[Broadcast()]
        );
        auto functor = [&mylayer, &X, &X_s, &initial_state](vector<Tensor> Xs)-> Tensor {
            auto state = mylayer.activate_shortcut(X, X_s, initial_state);
            return state.hidden;
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-3));
    }
}

TEST(LayerTests, LSTM_Zaremba_shortcut_gradient) {
    int num_examples           = 6;
    int hidden_size            = 5;
    int input_size             = 3;
    int shortcut_size          = 2;

    EXPERIMENT_REPEAT {
        auto X   = Tensor::uniform(10.0, {num_examples, input_size},    DTYPE_DOUBLE);
        auto X_s = Tensor::uniform(10.0, {num_examples, shortcut_size}, DTYPE_DOUBLE);
        auto mylayer = LSTM({input_size, shortcut_size}, hidden_size, 1, false, DTYPE_DOUBLE);

        auto params = mylayer.parameters();
        params.emplace_back(X);
        params.emplace_back(X_s);

        auto initial_state = LSTMState(
            Tensor::uniform(0.05, {hidden_size}, DTYPE_DOUBLE)[Broadcast()],
            Tensor::uniform(0.05, {hidden_size}, DTYPE_DOUBLE)[Broadcast()]
        );
        auto functor = [&mylayer, &X, &X_s, &initial_state](vector<Tensor> Xs)-> Tensor {
            auto state = mylayer.activate_shortcut(X, X_s, initial_state);
            return state.hidden;
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-3));
    }
}

TEST(LayerTests, RNN_gradient_vs_Stacked_gradient) {
    int num_examples           = 10;
    int hidden_size            = 5;
    int input_size             = 3;

    EXPERIMENT_REPEAT {
        auto X  = Tensor::uniform(10.0, {num_examples, input_size}, DTYPE_DOUBLE);
        auto H  = Tensor::uniform(10.0, {num_examples, hidden_size},DTYPE_DOUBLE);

        auto X_s  = Tensor(X, true, true); // perform full copies
        auto H_s  = Tensor(H, true, true); // perform full copies

        auto rnn_layer     = RNN(input_size, hidden_size, DTYPE_DOUBLE);
        auto stacked_layer = StackedInputLayer({input_size, hidden_size}, hidden_size, DTYPE_DOUBLE);

        auto params = rnn_layer.parameters();
        auto stacked_params = stacked_layer.parameters();

        EXPECT_EQ(params.size(), stacked_params.size());

        for (int i = 0; i < params.size(); ++i) {
            EXPECT_EQ(params[i].shape(), stacked_params[i].shape());
            stacked_params[i].w.copy_from(params[i].w);
        }

        auto error = ((rnn_layer.activate(X, H).tanh() - 1) ^ 2).sum();
        error.grad();

        auto error2 = ((stacked_layer.activate({X_s, H_s}).tanh() - 1) ^ 2).sum();
        error2.grad();
        graph::backward();

        for (int i = 0; i < params.size(); ++i) {
            EXPECT_TRUE(Array::allclose(stacked_params[i].dw, params[i].dw, 1e-6));
        }
        EXPECT_TRUE(Array::allclose(X.dw, X_s.dw, 1e-6));
        EXPECT_TRUE(Array::allclose(H.dw, H_s.dw, 1e-6));
    }
}

TEST(LayerTests, shortcut_test) {
    int input_size = 10;
    int num_examples = 2;
    vector<int> hidden_sizes({40, 30});

    auto model = StackedLSTM(input_size, hidden_sizes, true, true, DTYPE_DOUBLE);
    auto X = {
        Tensor::uniform(10.0, {num_examples, input_size}, DTYPE_DOUBLE)
    };

    vector<LSTM::activation_t> initial_states;
    for (int i = 0; i < hidden_sizes.size(); ++i) {
        auto initial_state = LSTMState(
            Tensor::uniform(0.05, {hidden_sizes[i]}, DTYPE_DOUBLE)[Broadcast()],
            Tensor::uniform(0.05, {hidden_sizes[i]}, DTYPE_DOUBLE)[Broadcast()]
        );
        initial_states.emplace_back(initial_state);
    }

    auto out_states = model.activate_sequence(initial_states, X, 0.2);
}

using std::string;
using std::make_shared;

TEST(LayerTests, multi_input_lstm_test) {
    utils::random::set_seed(5000);

    int num_children = 3;
    int input_size = 4;
    int hidden_size = 2;
    int num_examples = 3;

    EXPERIMENT_REPEAT {
        auto input = Tensor::uniform(10.0, {num_examples, input_size}, DTYPE_DOUBLE);
        vector<LSTM::activation_t> states;
        for (int cidx = 0 ; cidx < num_children; ++cidx) {
            states.emplace_back(
                Tensor::uniform(10.0, {num_examples, hidden_size}, DTYPE_DOUBLE),
                Tensor::uniform(10.0, {num_examples, hidden_size}, DTYPE_DOUBLE)
            );
        }

        auto mylayer = LSTM(input_size, hidden_size, num_children, false, DTYPE_DOUBLE);
        mylayer.name_internal_layers();

        auto params  = mylayer.parameters();
        params.emplace_back(input);

        for(auto& state : states) {
            state.memory.name = make_shared<string>("state_memory");
            state.hidden.name = make_shared<string>("state_hidden");
            params.emplace_back(state.memory);
            params.emplace_back(state.hidden);
        }

        auto functor = [&mylayer, &input, &states](vector<Tensor> Xs)-> Tensor {
                auto state = mylayer.activate(input, states);
                return state.hidden;
        };
        ASSERT_TRUE(gradient_ratio_same(functor, params, 0.05, 0.0001));
    }
    utils::random::reseed();
}

TEST(LayerTests, activate_sequence) {
    vector<int> hidden_sizes = {7, 10};
    int input_size = 5;
    int num_out_states = hidden_sizes.size();
    vector<Tensor> sequence;
    for (int i = 0; i < 10; i++) {
        sequence.emplace_back(Tensor::zeros({input_size}, DTYPE_DOUBLE)[Broadcast()]);
    }
    auto model = StackedLSTM(input_size, hidden_sizes, false, false, DTYPE_DOUBLE);
    auto out_states = model.activate_sequence(model.initial_states(), sequence, 0.1);
    ASSERT_EQ(num_out_states, LSTMState::hiddens(out_states).size());
}

TEST(LayerTests, GRU) {
    int input_size = 3;
    int hidden_size = 5;
    int tsteps = 5;

    EXPERIMENT_REPEAT {
        auto gru = GRU(input_size, hidden_size, DTYPE_DOUBLE);
        auto params = gru.parameters();
        auto inputs = vector<Tensor>();
        for (int i = 0; i < tsteps; i++) {
            inputs.emplace_back(Tensor::uniform(-10.0, 10.0, {1, input_size}, DTYPE_DOUBLE));
        }
        auto functor = [&inputs, &gru,&tsteps, &hidden_size, &input_size](vector<Tensor> Xs)-> Tensor {
            auto state = gru.initial_states();
            for (int i = 0; i < tsteps; i++)
                state = gru.activate(inputs[i], state);
            return (state -1.0) ^ 2;
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-3));
    }
}


#ifdef DALI_USE_CUDA // TODO(jonathan): remove once working on CPU

TEST(LayerTests, conv) {
    int num_examples = 5;
    int out_channels = 4;
    int in_channels  = 3;
    int filter_h     = 2;
    int filter_w     = 2;
    int stride_h     = 2;
    int stride_w     = 2;


    EXPERIMENT_REPEAT {
        auto X  = Tensor::uniform(20.0, {5, in_channels, 8, 8}, DTYPE_DOUBLE);
        auto mylayer = ConvLayer(out_channels, in_channels,
                                 filter_h, filter_w,
                                 stride_h, stride_w,
                                 PADDING_T_SAME,
                                 "NCHW",
                                 DTYPE_DOUBLE);
        auto params = mylayer.parameters();
        params.emplace_back(X);
        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return mylayer.activate(X);
        };
        ASSERT_TRUE(gradient_same(functor, {params}, 1e-2));
    }
}

#endif
