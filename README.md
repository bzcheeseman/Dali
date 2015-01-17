RecurrentJS in C++, Python
--------------------------

Following in the footsteps of [Andrej Kaparthy](http://cs.stanford.edu/people/karpathy/), here is a re-imagination in C++ and in Python of [recurrentJS](http://cs.stanford.edu/people/karpathy/recurrentjs/) ([Github](https://github.com/karpathy/recurrentjs)).

The C++ version has an emphasis on clean API and speed, while the Python version remains faithful to its Javascript cousin.

### Why ?

While Python has great automatic differentiation libraries, a no-compile version is lacking. In particular recurrentJS makes great use of callbacks and garbage collection to enable backprop through time. In this implementation the goal is to reduce reliance on these abstractions and have a simple backprop step class.

In Python use of a specialized `Backward` class wraps backpropagation steps, while C++ uses its own `Backward` class but this time garbage collection and tracking is done using `C++11`'s excellent `std::shared_ptr`.

### Usage in C++

#### Installation

Go into the `cpp` folder and:

    > make

#### Stacked LSTMs

Let's build a stacked LSTM network. We start by including the right header:

    #include "Layers.h"

And let's populate our namespace with some goodies:

    using std::pair;
    using std::vector;
    using std::shared_ptr;
    using std::make_shared;

Let's define some types we'll use to simplify notation:

    typedef float REAL_t;
    typedef LSTM<REAL_t> lstm;
    typedef Graph<REAL_t> graph_t;
    typedef Mat<REAL_t> mat;
    typedef shared_ptr<mat> shared_mat;
    typedef vector<shared_mat> cell_outputs;
    typedef pair<cell_outputs, cell_outputs> paired_cell_outputs;

Okay. Let's now build a set of stacked cells inside our main function with 3 layers of 100 hidden units and an input of 50:

    auto input_size          = 50;
    vector<int> hidden_sizes = {100, 100, 100};

    auto cells = StackedCells<lstm>(input_size, hidden_sizes);

We now collect the model parameters into one vector for optimization:
    
    vector<shared_mat> parameters;

    for (auto& layer : cells) {
        parameters.push_back( cells.input_layer.Wh )
        parameters.push_back( cells.input_layer.Wx )
        parameters.push_back( cells.input_layer.b )

        parameters.push_back( cells.forget_layer.Wh )
        parameters.push_back( cells.forget_layer.Wx )
        parameters.push_back( cells.forget_layer.b )

        parameters.push_back( cells.output_layer.Wh )
        parameters.push_back( cells.output_layer.Wx )
        parameters.push_back( cells.output_layer.b )

        parameters.push_back( cells.cell_layer.Wh )
        parameters.push_back( cells.cell_layer.Wx )
        parameters.push_back( cells.cell_layer.b )
    }

For backpropagation we need a `Graph`:

    graph_t G;

Let's create some random input, using a gaussian with standard deviation 2, with 100 different examples in this batch:

    auto batch_size          = 100;
    REAL_t std               = 2.0;

    auto input_vector = make_shared<mat>(input_size, batch_size, std);

To run our network forward in time we'll need some initial states and a propagation function. Let's start with the propagation. This function takes the *graph* to keep track of computations, an *input vector*, and the *previous hidden and cell states* for the LSTM layers, and the *LSTMs* themselves:

    paired_cell_outputs forward_lstms(
        graph_t& G,
        shared_mat input_vector,
        paired_cell_outputs& previous_state,
        vector<lstm>& cells) {

        // in our previous_state pair cells are first
        auto previous_state_cells = previous_state.first;

        // next we have hidden activations:
        auto previous_state_hiddens = previous_state.second;


        // let's iterate through both as we visit each layer:
        auto cell_iter = previous_state_cells.begin();
        auto hidden_iter = previous_state_hiddens.begin();

        // this will be our output states:
        paired_cell_outputs out_state;

        // this is the current input to the lowest LSTM layer
        auto layer_input = input_vector;

        for (auto& layer : cells) {
            auto layer_out = layer.activate(G, layer_input, *cell_iter, *hidden_iter);

            // keep track of cells and their outputs:
            out_state.first.push_back(layer_out.first);
            out_state.second.push_back(layer_out.second);

            ++cell_iter;
            ++hidden_iter;

            // the current layer's hidden activation is passed upwards
            layer_input = layer_out.second;
        }
        return out_state;
    }

Now that we can propagate our network forward let's run this forward in time, and start off with a blank cell activation and hidden activation for each LSTM, here we use the LSTM class's `initial_states` static method:

    paired_cell_outputs initial_state = lstm::initial_states(hidden_sizes);

And we can now run this network forward:
    
    auto timesteps = 20;

    for (auto i = 0; i < timesteps; ++i)
        initial_state = forward_lstms(G, input_vector, initial_state, cells);

The input_vector isn't changing, but this is just an example. We could have instead used indices and plucked rows from an embedding matrix, or taken audio or video inputs.

#### Training

Now that we've built up this computation graph we should assign errors in the outputs we can about, and backpropagate it:

    G.backward();

`G` tallies all the gradients for each model parameter. We now use a solver to take a gradient step in the direction that reduces the error (here we use RMSprop with a decay rate of 0.95, an epsilon value for numerical stability of 1e-6, and we clip gradients larger than 5. element-wise):


    Solver<REAL_t> solver(0.95, 1e-6, 5.0);

Now at every minibatch we can call `step` on the solver to reduce the error (here using a learning rate of 0.01, and an L2 penalty of 1e-7):

    solver.step(parameters, 0.01, 1e-7)

### Usage in Python

#### Installation

Simple:

    pip install recurrent-js-python

#### Stacked LSTMS, aka. Character model demo in Python

Below we follow the same steps as in the character generation demo, and we import the same text for character model learning. Perplexity drops quickly to around 7-8, (mirroring the behavior found in the Javascript version).

    from recurrentjs import *

    input_size  = -1
    output_size = -1
    epoch_size  = -1
    letter_size = 5
    letterToIndex = {}
    indexToLetter = {}
    hidden_sizes = [20,20]
    generator = "lstm"
    vocab = []
    regc = 0.000001 # L2 regularization strength
    learning_rate = 0.01 # learning rate
    clipval = 5.0

    solver = Solver()

    def initVocab(sents, count_threshold):
        global input_size
        global output_size
        global epoch_size
        global vocab
        global letterToIndex
        global indexToLetter
        # count up all characters
        d = {}
        for sent in sents:
            for c in sent:
                if c in d:
                    d[c] += 1
                else:
                    d[c] = 1

        # filter by count threshold and create pointers
        letterToIndex = {}
        indexToLetter = {}
        vocab = []
        # NOTE: start at one because we will have START and END tokens!
        # that is, START token will be index 0 in model letter vectors
        # and END token will be index 0 in the next character softmax
        q = 1
        for ch in d.keys():
            if d[ch] >= count_threshold:
                # add character to vocab
                letterToIndex[ch] = q
                indexToLetter[q] = ch
                vocab.append(ch)
                q += 1
        # globals written: indexToLetter, letterToIndex, vocab (list), and:
        input_size  = len(vocab) + 1;
        output_size = len(vocab) + 1;
        epoch_size  = len(sents)

    def forwardIndex(G, model, ix, prev):
        x = G.row_pluck(model['Wil'], ix)
        # forward prop the sequence learner
        if generator == "rnn":
            out_struct = forwardRNN(G, model, hidden_sizes, x, prev)
        else:
            out_struct = forwardLSTM(G, model, hidden_sizes, x, prev)   
        return out_struct

    def initModel():
        model = {}
        lstm = initLSTM(letter_size, hidden_sizes, output_size) if generator == "lstm" else initRNN(letter_size, hidden_sizes, output_size)
        model['Wil'] = RandMat(input_size, letter_size , 0.08)
        model.update(lstm)

        return model

    def costfun(model, sent):
        # takes a model and a sentence and
        # calculates the loss. Also returns the Graph
        # object which can be used to do backprop
        n = len(sent)
        G = Graph()
        log2ppl = 0.0;
        cost = 0.0;
        prev = None
        for i in range(-1, n):
            # start and end tokens are zeros
            ix_source = 0 if i == -1 else letterToIndex[sent[i]] # first step: start with START token
            ix_target = 0 if i == n-1 else letterToIndex[sent[i+1]] # last step: end with END token

            lh = forwardIndex(G, model, ix_source, prev)
            prev = lh

            # set gradients into logprobabilities
            logprobs = lh.output # interpret output as logprobs
            probs = softmax(logprobs) # compute the softmax probabilities

            log2ppl += -np.log(probs.w[ix_target,0]) # accumulate base 2 log prob and do smoothing
            cost += -np.log(probs.w[ix_target,0])

            # write gradients into log probabilities
            logprobs.dw = probs.w
            logprobs.dw[ix_target] -= 1

        ppl = np.power(2, log2ppl / (n - 1))

        return G, ppl, cost

    text_data = open("paulgraham_text.txt", "rt").readlines()
    initVocab(text_data, 1)
    model = initModel()
    ppl_list = []
    median_ppl = []
    tick_iter = 0

    def tick():
        global tick_iter
        global ppl_list
        global median_ppl
        sentix = np.random.randint(0, len(text_data))
        sent = text_data[sentix]
        G, ppl, cost = costfun(model, sent)
        G.backward()
        solver.step(model, learning_rate, regc, clipval)
        ppl_list.append(ppl)
        tick_iter += 1

        if tick_iter % 100 == 0:
            median = np.median(ppl_list)
            ppl_list = []
            median_ppl.append(median)

And the training loop (no fancy prediction and sampling implemented here, but fairly straightforward conversion from the javascript code)
  
    for i in range(1000):
        tick()