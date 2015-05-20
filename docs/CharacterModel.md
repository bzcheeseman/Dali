# Usage Tutorial

Let's go through a character model forecasting problem and implement it using **Dali**.

#### Stacked LSTMs

Let's build a stacked LSTM network. We start by including the right header:

```cpp
#include "core/LSTM.h"
```

And let's populate our namespace with some goodies:

```cpp
using std::tuple;
using std::vector;
```

We also define some types we'll use to simplify notation:

```cpp
typedef Mat<float> mat;
typedef vector<mat> cell_outputs;
```

Let's build a set of stacked cells inside our main function: 3 layers of 100 hidden units with an original input of size 50 (Note how we template the static method `StackedCells<T>` to take a `LSTM<float>` -- this is how we create multiple layers of cells with the same type: `LSTM`, `RNN`, `ShortcutRNN`, `GatedInput`, etc..):

```cpp
auto input_size   = 50;
auto hidden_sizes = {100, 100, 100};

auto cells = StackedCells<LSTM<float>>(input_size, hidden_sizes);
```

We now collect the model parameters into one vector for optimization:

```cpp
vector<mat> parameters;

for (auto& layer : cells) {
    auto layer_params = layer.parameters();
    parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
}
```

For backpropagation we use a computation `Tape` (this object remembers backpropagation computations):

```cpp
graph::Tape
```

Let's create some random input, using a multivariate gaussian with a standard deviation of 2.0, and create a batch of 100 samples from this distribution:

```cpp
auto batch_size          = 100;
float std                = 2.0;

auto input_vector = mat(input_size, batch_size, std);
```

To run our network forward in time we'll need some initial states and a forward propagation function. Let's start with the propagation. This function takes the `Graph` to keep track of computations, an `input_vector`, and the *previous hidden and cell states* for the LSTM layers, and the `LSTM`s themselves:

```cpp
std::vector< typename LSTM<float>::State > forward_LSTMs(
    Mat<float> input,
    vector<LSTM<float>::State>& states,
    vector<LSTM<float>>& lstms);
```

Now that we can propagate our network forward let's run this forward in time, and start off with a blank cell activation and hidden activation for each LSTM, here we use the LSTM class's `initial_states` static method:

```cpp
auto state = LSTM<float>::initial_states(hidden_sizes);
```

And we can now run this network forward:

```cpp
auto timesteps = 20;

for (auto i = 0; i < timesteps; ++i)
    state = forward_LSTMs(input_vector, state, cells);
```

The input_vector isn't changing, but this is just an example. We could have instead used indices and plucked rows from an embedding matrix, or taken audio or video inputs.

### Character model extension:

Suppose we want to assign error using a prediction with a additional decoding layer that gets exponentially normalized via a *Softmax*:

```cpp
auto vocab_size = 300;
Layer<float> classifier(hidden_sizes[hidden_sizes.size() - 1], vocab_size);
```

Each character gets a vector in an embedding:

```cpp
auto embedding = mat(vocab_size, input_size, 0.08);
```

Then our forward function can be changed to:

```cpp
float cost_fun(
    vector<int>& hidden_sizes, // re-initialize the hidden cell states at each new sentence
    vector<LSTM<float>>& cells,  // the LSTMs
    mat embedding, // the embedding matrix
    classifier_t& classifier, // the classifier we just defined
    vector<int>& indices // the indices in a sentence whose perplexity we'd like to reduce
    ) {
    // construct hidden cell states:
    auto state = lstm::states(hidden_sizes);
    auto num_hidden_sizes = hidden_sizes.size();

    mat input_vector;
    mat logprobs;
    mat probs;

    float cost = 0.0;
    auto n = indices.size();

    for (int i = 0; i < n-1; ++i) {
        // pick this letter from the embedding
        input_vector  = embedding[indices[i]];
        // pass this letter to the LSTM for processing
        state = forward_LSTMs(input_vector, state, cells);
        // classifier takes as input the final hidden layer's activation:
        logprobs = classifier.activate(state[num_hidden_sizes-1].hidden);
        // compute the softmax probabilities
        probs         = logprobs.softmax_no_grad();
        // accumulate base 2 log prob and do smoothing
        cost         -= std::log(probs.w().(indices[i+1],0));
        // write gradients into log probabilities
        logprobs.dw()  = probs.w();
        logprobs.dw(indices[i+1], 0) -= 1;
    }
    return cost / (n-1);
}
```


#### Training

Now that we've computed the cost we need to tell the computation graph to backpropagate it to the parameters:

```cpp
graph::backward();
```

`graph::backward()` tallies all the gradients for each model parameter. We now use a solver to take a gradient step in the direction that reduces the error (here we use RMSprop with a decay rate of 0.95, an epsilon value for numerical stability of 1e-6, and we clip gradients larger than 5. element-wise):

```cpp
Solver::RMSProp<float> solver(0.95, 1e-6, 5.0);
```

Now at every minibatch we can call `step` on the solver to reduce the error (here using a learning rate of 0.01, and an L2 penalty of 1e-7):

```cpp
solver.step(parameters, 0.01, 1e-7);
```

This would look like the following. First we load the sentences using `fstream`:


```cpp
auto sentences = get_character_sequences(
    "data/paulgraham_text.txt",
    prepad,
    postpad,
    vocab_size);
```

Get a random number generator to uniformly sample from these sentences:

```cpp
static std::random_device rd;
static std::mt19937 seed(rd());
static std::uniform_int_distribution<> uniform(0, sentences.size() - 1);
```

Then we train by looping through the sentences:

```cpp
for (auto i = 0; i < epochs; ++i) {
    auto cost = cost_fun(
        hidden_sizes,            // to construct initial states
        cells,                   // LSTMs
        embedding,               // character embedding
        classifier,              // decoder for LSTM final hidden layer
        sentences[uniform(seed)] // the sequence to predict
    );
    graph::backward();           // backpropagate
    // progress by one step
    solver.step(parameters, 0.01, 0.0);
    if (i % report_frequency == 0)
        std::cout << "epoch (" << i << ") perplexity = " << cost << std::endl;
}
```

### Loading the file of characters:

To get the character sequence we can use this simple function and point it at the Paul Graham text:

```cpp
vector<vector<int>> get_character_sequences(const char* filename, int& prepad, int& postpad, int& vocab_size) {
    char ch;
    char linebreak = '\n';
    fstream file;
    file.open(filename);
    vector<vector<int>> lines;
    lines.emplace_back(2);
    vector<int>& line = lines[0];
    line->emplace_back(prepad);
    while(file) {
        ch = file.get();
        if (ch == linebreak) {
            line.emplace_back(postpad);
            lines.emplace_back(2);
            line = lines.back();
            line.emplace_back(prepad);
            continue;
        }
        if (ch == EOF) break;
        // make sure no character is higher than the vocab size:
        line->push_back(std::min(vocab_size-1, (int) ch));
    }
    return lines;
}
```
