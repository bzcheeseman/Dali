# RecurrentJS in Python

Following in the footsteps of [Andrej Kaparthy](http://cs.stanford.edu/people/karpathy/), here is a  Python implementation of [recurrentJS](http://cs.stanford.edu/people/karpathy/recurrentjs/) ([Github](https://github.com/karpathy/recurrentjs)).

### Why ?

While Python has great automatic differentiation libraries, a no-compile version is lacking. In particular recurrentJS makes great use of callbacks and garbage collection to enable backprop through time. In this implementation the goal is to reduce reliance on these abstractions and have a simple backprop step class. Finally if this is easily feasible, then ultimately we will implement a C++ version that will make the majority of the computation steps fast except for some intermediary allocations of backprop steps, but that do not make up the bulk of the computation, and keep the API and the syntax clear.

### Planned extensions to Javascript version

In this implementation the goal is simple:

* Move away from callbacks
* Enable batch processing (through masked losses, tensors, and advanced indexing when plucking rows from matrices)
* Form a baseline for implementation in non-scripting languages


### Usage

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