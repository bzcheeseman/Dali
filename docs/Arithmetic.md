Basic Arithmetic
================

Let's teach our network about computation order and rules. In the end
we want to create a network that gets characters as input and outputs
characters describing the result:

```coffee
6 * 4 + 8 => 32
```

We'll proceed as follows:

1. Generate Examples
2. Construct Model
3. Create an objective function
4. Train
5. Output beam search predictions

### Generate Examples

To do this we will first need some good examples.
Let's define our basic operator:

```cpp
vector<string> SYMBOLS = {"+", "*", "-"};
```

Then let's dive into generating examples:

```cpp
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
}
```

To get the value of the examples we will loop through each generated sequence
twice and extract the final result:

```cpp
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
```

Now that we know the result we add this to our list of examples:

```cpp
vector<pair<vector<string>, vector<string>>> generate_examples(int num) {
        ...
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
```

We can now save a mapping from our characters and strings to indices in
and embedding matrix using a Vocabulary:

```cpp
#include "core/utils/core_utils.h"

vector<string> symbols;
for (int i = 0; i < 10; i++) {
    symbols.push_back(to_string(i));
}
symbols.insert(symbols.end(), SYMBOLS.begin(), SYMBOLS.end());
symbols.push_back(utils::end_symbol);
std::cout << symbols << std::endl;
utils::Vocab vocab(
    symbols, // what symbols to use
    false // whether a special unknown word should be added
);
```

Finally we convert our examples into indices using the vocabulary
as reference:

```cpp
auto char_examples = generate_examples(200);

vector<pair<vector<uint>, vector<uint>>> examples(char_examples.size());
{
    for (size_t i = 0; i < char_examples.size();i++) {
        examples[i].first  = vocab.encode(
            char_examples[i].first, // words to convert
            true                    // add end token
        );
        examples[i].second = vocab.encode(
            char_examples[i].second, // words to convert
            true                     // add end token
        );
    }
}
```

### Construct Model

Here are some Stacked LSTMs like those used in [Wojciech Zaremba, and Ilya Sutskever's "Learning to Execute"](http://arxiv.org/abs/1410.4615):

```cpp
#include "core/models/StackedModel.h"

auto model = StackedModel<REAL_t>(
     vocab.size(), // vocabulary size
     100, // embedding dimension
     100, // # of LSTM hidden cells
     3, // How many LSTM layers are connected up
     vocab.size(), // How many output dimensions in the decoder
     false, // Whether to use Alex Graves-style shortcuts from inputs to each layer +
            // from each hidden to the decoder
     false  // Whether to connect the memory units from the LSTM to the gates
            // as an additional signal
     );
```

Our model is now constructed, we can now query its parameters using:

```cpp
auto params = model.parameters();
```

Or use it for prediction:

```cpp
auto activation = model.activate(
    model.initial_states(),
    vocab.word2index["+"]
    );
```

### Create an objective function

The model we are constructed must report character by character the result of the computation. To do this we train the output sequence to match the correct sequence of characters and penalize the Kullback-Leibler divergence between the character distribution provided by the model and the true character expected.


#### 1. Feed the example to model

```cpp
void get_error(
        StackedModel<REAL_t>& model,
        pair<vector<uint>, vector<uint>>& example) {

    // initialize cell and hidden activations of LSTM to zeros:
    auto state = model.initial_states();

    // loop through input characters, while ignoring the
    // model's predictions, and keeping track of the
    // internal state of the model
    Mat<REAL_t> input_vector;
    for (auto& c : example.first) {
        // pick character's embedding:
        input_vector = model.embedding[c];
        // pass it to the model
        state = model.stacked_lstm->activate(
            initstateial_state,
            input_vector
        );
    }
```

#### 2. Make predictions:

The `state` of the LSTM is now supposed to contain
all the necessary information to output the correct
characters to complete the equation.

To train such a system we compute the cross entropy
between the LSTM decoder's output and the target
distribution. We pass all the hidden states of our
network as inputs to the decoder using
`LSTM<REAL_t>::State::hiddens(state)`:

```cpp
auto error = MatOps<REAL_t>::softmax_cross_entropy(
    model.decoder->activate(
        input_vector,
        LSTM<REAL_t>::State::hiddens(state)
    ),
    example.second.front()
);
```

For every remaining character in the target we can
now pass in as input the characters we **actually**
wanted to have predicted, and penalize the system's
output conditioned on those correct inputs (Note: we
loop only until the before last element since our
system must discover when to stop predicting on its
own by predicting a special **end** token; more on
this later).

```cpp
for (auto label_ptr = example.second.begin(); label_ptr < example.second.end() -1; label_ptr++) {
    // read the correct prediction in
    input_vector = model.embedding[*label_ptr];
    state = model.stacked_lstm->activate(
        state,
        input_vector
    );
    // sum the errors at each step
    error = error + MatOps<REAL_t>::softmax_cross_entropy(
        model.decoder->activate(
            input_vector,
            LSTM<REAL_t>::State::hiddens(state)
        ),
        *(label_ptr+1)
    );
}
```

#### 3. Backpropagation

We have now collected the error for this example. We can
now ask our network to correct its weights accordingly
using Paul Werbos' [backpropagation algorithm](http://deeplearning.cs.cmu.edu/pdfs/Werbos.backprop.pdf):

```cpp
// add to objective function
error.grad();
// backpropagate
graph::backward();

} // </get_error>
```

The network's parameters now contain the gradients needed for
improvement.

### Train

Our objective function describes the current error our network
is making. Using backpropagation we were able to obtain a gradient
for the weights controlling the network that will reduce the error.

To apply those updates to our weights we use an optimizer that controls
our gradient descent. There are many different optimizers to choose from. Here we use [Adadelta](http://arxiv.org/abs/1212.5701) to simplify the number of hyperparameters to tune:

```cpp
auto solver = Solver::AdaDelta<REAL_t>>(
    params, // what to train
    0.95,   // rho value (controls AdaDelta)
    1e-9,   // epsilon to avoid division by zero
    100.0,  // clip gradients larger than this
    1e-5    // L2 penalty on weights
)
```

We can now loop through our examples, collecting gradient updates,
and finally call `step` on the solver to apply updates to our weights:

```cpp
int epoch      = 0;
int max_epochs = 300;
while (epoch < 300) {
    auto indices = utils::random_arange(examples.size());
    auto indices_begin = indices.begin();

    // one minibatch
    for (auto indices_begin = indices.begin(); indices_begin < indices.begin() + std::min((size_t)FLAGS_minibatch, examples.size()); indices_begin++) {
        get_error(model, examples[*indices_begin]);
    }
    // One step of gradient descent
    solver.step(params);
    epoch++;
}
```

### Output beam search predictions

Coming soon
