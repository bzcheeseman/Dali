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

### Construct Model

Here are some Stacked LSTMs like those used in [Wojciech Zaremba, and Ilya Sutskever's "Learning to Execute"](http://arxiv.org/abs/1410.4615):

```cpp
#include "core/models/StackedModel.h"

auto model = StackedModel<REAL_t>(
     vocab.index2word.size(), // vocabulary size
     100, // embedding dimension
     100, // # of LSTM hidden cells
     3, // How many LSTM layers are connected up
     vocab.index2word.size(), // How many output dimensions in the decoder
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

Coming soon

### Train

Coming soon

### Output beam search predictions

Coming soon
