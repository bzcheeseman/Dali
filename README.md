RecurrentJS in C++, Python
--------------------------

Following in the footsteps of [Andrej Kaparthy](http://cs.stanford.edu/people/karpathy/), here is a re-imagination in C++ and in Python of [recurrentJS](http://cs.stanford.edu/people/karpathy/recurrentjs/) ([Github](https://github.com/karpathy/recurrentjs)).

The C++ version has an emphasis on clean API and speed, while the Python version remains faithful to its Javascript cousin.

### Why ?

While Python has great automatic differentiation libraries, a no-compile version is lacking. In particular recurrentJS makes great use of callbacks and garbage collection to enable backprop through time. In this implementation the goal is to reduce reliance on these abstractions and have a simple backprop step class.

In Python use of a specialized `Backward` class wraps backpropagation steps, while C++ uses the `<functional>` lambda functions but this time garbage collection and tracking is done using `C++11`'s excellent `std::shared_ptr`.

### Features

* Automatic differentiation
* Matrix Broadcasting (elementwise multiply, elementwise product)
* Multiple index slicing
* Speed
* Clarity of API

### Usage in C++

#### Run a super duper simple example

Create two 3x3 matrices filled with uniform random noise between -2 and 2:

    Mat<float> A(3,3, -2.0, 2.0);
    Mat<float> B(3,3, -2.0, 2.0);

Now let's multiply them:

    auto C = A * B;

Now's let take the gradient of the squared sum of this operation:

    auto error = (C ^ 2).sum();

And get the gradient of error with respect to A and B:

    error.grad();
    graph::backward();

    auto A_gradient = A.dw();
    auto B_gradient = B.dw();


##### Behind the scenes:

Each matrix has another matrix called `dw` that holds the elementwise gradients for each
matrix. When we multiply the matrices together we create a new output matrix called `C`,
**and** we also add this operation to our computational graph (held by a thread local
variable in `graph::tape`). When we reach `C.sum()` we also add this operation to our graph.

Computing the gradient is done in 2 steps, first we tell our graph what the objective
function is:

    error.grad();

`error` needs to be a scalar (a 1x1 matrix in this implementation) to use `grad()`.
Step 2 is to call `graph::backward()` and go through every operation executed so far
in reverse using `graph::tape`'s record. When we run through the operations backward
we update the gradients of each intermediary object until `A` and `B`'s `dw`s get
updated. Those are now [the gradients we we're looking for](http://youtu.be/DIzAaY2Jm-s?t=3m12s).

#### Run a simple example

Let's run a simple example. We will use data from [Paul Graham's blog](http://paulgraham.com) to train a language model. This way we can generate random pieces of startup wisdom at will! After about 5-10 minutes of training time you should see it generate sentences that sort of make sense. To do this go to cpp/build and execute

    examples/language_model --flagfile ../flags/language_model_simple.flags

That's it. Don't forget to checkout `examples/language_model.cpp`. It's not that scary!

#### Installation

Get **[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)** ([Download Link](http://bitbucket.org/eigen/eigen/get/3.2.4.tar.bz2)), **Clang**, and **protobuf**, then head to the `cpp/build` folder and use `cmake` to configure and create the appropriate Makefiles:

    > brew install clang
    > brew install eigen
    > brew install cmake
    > HOMEBREW_CC=clang HOMEBREW_CXX=clang++ brew install protobuf
    > cmake ..


The run `make` to compile the code:


    > make -j 9


That's it. Now built examples will be stored in `cpp/build/examples`.
For instance a character prediction model using Stacked LSTMs is built under `cpp/build/examples/character_prediction`.

#### Tests

To compile and run tests you need [Google Tests](https://code.google.com/p/googletest/). Download it [here](https://code.google.com/p/googletest/downloads/detail?name=gtest-1.7.0.zip).

#### 1. Compile and run tests

From the build folder do the following:

    cmake ..
    make -j 9 run_tests

###### 2.a Install Gtest on Mac OSX

Homebrew does not offer a way of installing gtest, however in a few steps you can get it running. First go to the directory where you downloaded Gtests:

    cd gtest-1.7.0
    mkdir mybuild
    cd mybuild
    cmake ..
    cp libgtest_main.a /usr/local/lib/libgtest_main.a
    cp libgtest.a /usr/local/lib/libgtest.a
    cp -R ../include/* /usr/local/include/

Now cmake should be able to find gtest (go back to step 1).

###### 2.b Install Gtest on Fedora Linux

Using `yum` it's a piece of cake:

    sudo yum install gtest gtest-devel

#### MKL Zaziness Problems

On Mac OSX, or more generally when using [Intel's gracious MKL Library](https://software.intel.com/en-us/intel-mkl) you may encounter an interesting bug with [`Eigen`](http://eigen.tuxfamily.org/bz/show_bug.cgi?id=874) where `MKL_BLAS` is shown as undefined during compilation.

To fix this bug (feature?) make the modifications listed [here](https://bitbucket.org/eigen/eigen/pull-request/82/fix-for-mkl_blas-not-defined-in-mkl-112/diff) to your Eigen header files and everything should be back to normal.

#### Stacked LSTMs

Let's build a stacked LSTM network. We start by including the right header:

    #include "core/Layers.h"

And let's populate our namespace with some goodies:

    using std::tuple;
    using std::vector;

We also define some types we'll use to simplify notation:

    typedef Mat<float> mat;
    typedef vector<mat> cell_outputs;
    typedef tuple<cell_outputs, cell_outputs> paired_cell_outputs;

Let's build a set of stacked cells inside our main function: 3 layers of 100 hidden units with an original input of size 50 (Note how we template the static method `StackedCells<T>` to take a `LSTM<float>` -- this is how we create multiple layers of cells with the same type: `LSTM`, `RNN`, `ShortcutRNN`, `GatedInput`, etc..):

    auto input_size   = 50;
    auto hidden_sizes = {100, 100, 100};

    auto cells = StackedCells<LSTM<float>>(input_size, hidden_sizes);

We now collect the model parameters into one vector for optimization:

    vector<mat> parameters;

    for (auto& layer : cells) {
        auto layer_params = layer.parameters();
        parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
    }

For backpropagation we use a computation `Graph` (this object orchestrates backpropagation computations):

    Graph<float> G;

Let's create some random input, using a multivariate gaussian with a standard deviation of 2.0, and create a batch of 100 samples from this distribution:

    auto batch_size          = 100;
    float std                = 2.0;

    auto input_vector = mat(input_size, batch_size, std);

To run our network forward in time we'll need some initial states and a forward propagation function. Let's start with the propagation. This function takes the `Graph` to keep track of computations, an `input_vector`, and the *previous hidden and cell states* for the LSTM layers, and the `LSTM`s themselves:

    pair<vector<Mat<float>>, vector<Mat<float>>> forward_LSTMs(
        Graph<float>&,
        Mat<float>,
        pair<vector<Mat<float>>, vector<Mat<float>>>&,
        vector<LSTM<float>>&);

Now that we can propagate our network forward let's run this forward in time, and start off with a blank cell activation and hidden activation for each LSTM, here we use the LSTM class's `initial_states` static method:

    paired_cell_outputs initial_state = LSTM<float>::initial_states(hidden_sizes);

And we can now run this network forward:

    auto timesteps = 20;

    for (auto i = 0; i < timesteps; ++i)
        initial_state = forward_LSTMs(G, input_vector, initial_state, cells);

The input_vector isn't changing, but this is just an example. We could have instead used indices and plucked rows from an embedding matrix, or taken audio or video inputs.

### Character model extension:

Suppose we want to assign error using a prediction with a additional decoding layer that gets exponentially normalized via a *Softmax*:

    auto vocab_size = 300;
    Layer<float> classifier(hidden_sizes[hidden_sizes.size() - 1], vocab_size);

Each character gets a vector in an embedding:

    auto embedding = mat(vocab_size, input_size, 0.08);

Then our forward function can be changed to:

    float cost_fun(
        Graph<float>& G, // the graph for the computation
        vector<int>& hidden_sizes, // re-initialize the hidden cell states at each new sentence
        vector<LSTM<float>>& cells,  // the LSTMs
        mat embedding, // the embedding matrix
        classifier_t& classifier, // the classifier we just defined
        vector<int>& indices // the indices in a sentence whose perplexity we'd like to reduce
        ) {
        // construct hidden cell states:
        auto initial_state = lstm::initial_states(hidden_sizes);
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
            initial_state = forward_LSTMs(input_vector, initial_state, cells);
            // classifier takes as input the final hidden layer's activation:
            logprobs      = classifier.activate(initial_state[num_hidden_sizes-1].hidden);
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


#### Training

Now that we've built up this computation graph we should assign errors in the outputs we can about, and backpropagate it:

    graph::backward();

`graph::backward()` tallies all the gradients for each model parameter. We now use a solver to take a gradient step in the direction that reduces the error (here we use RMSprop with a decay rate of 0.95, an epsilon value for numerical stability of 1e-6, and we clip gradients larger than 5. element-wise):


    Solver::RMSProp<float> solver(0.95, 1e-6, 5.0);

Now at every minibatch we can call `step` on the solver to reduce the error (here using a learning rate of 0.01, and an L2 penalty of 1e-7):

    solver.step(parameters, 0.01, 1e-7);

This would look like the following. First we load the sentences using `fstream`:

    auto sentences = get_character_sequences("data/paulgraham_text.txt", prepad, postpad, vocab_size);

Get a random number generator to uniformly sample from these sentences:

    static std::random_device rd;
    static std::mt19937 seed(rd());
    static std::uniform_int_distribution<> uniform(0, sentences.size() - 1);

Then we train by looping through the sentences:

    // Main training loop:
    for (auto i = 0; i < epochs; ++i) {
        auto G = Graph<float>(true);      // create a new graph for each loop
        auto cost = cost_fun(
            G,                       // to keep track of computation
            hidden_sizes,            // to construct initial states
            cells,                   // LSTMs
            embedding,               // character embedding
            classifier,              // decoder for LSTM final hidden layer
            sentences[uniform(seed)] // the sequence to predict
        );
        G.backward();                // backpropagate
        // progress by one step
        solver.step(parameters, 0.01, 0.0);
        if (i % report_frequency == 0)
            std::cout << "epoch (" << i << ") perplexity = " << cost << std::endl;
    }

### Loading the file of characters:

To get the character sequence we can use this simple function and point it at the Paul Graham text:

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
