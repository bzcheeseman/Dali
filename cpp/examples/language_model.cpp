#include <fstream>
#include <iterator>
#include <algorithm>
#include <Eigen>
#include <thread>
#include <atomic>
#include <algorithm>
#include "../utils.h"
#include "../SST.h"
#include "../gzstream.h"
#include "../StackedModel.h"
#include "../OptionParser/OptionParser.h"
using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::istringstream;
using std::stringstream;
using std::string;
using std::min;
using std::thread;
using std::ref;
using utils::Vocab;
using utils::from_string;
using utils::OntologyBranch;
using utils::tokenized_uint_labeled_dataset;
using std::atomic;


typedef float REAL_t;
typedef Graph<REAL_t> graph_t;
typedef Mat<REAL_t> mat;
typedef float price_t;
typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
typedef Eigen::Matrix<REAL_t, Eigen::Dynamic, 1> float_vector;
typedef std::pair<vector<string>, uint> labeled_pair;

const string START = "**START**";

/**
Databatch
---------

Datastructure handling the storage of training
data, length of each example in a minibatch,
and total number of prediction instances
within a single minibatch.

**/
class Databatch {
    typedef shared_ptr<index_mat> shared_index_mat;
    public:
        shared_index_mat data;
        shared_eigen_index_vector codelens;
        shared_ptr<vector<uint>> row_keys;
        int total_codes;
        Databatch(int n, int d) {
            data        = make_shared<index_mat>(n, d);
            codelens    = make_shared<eigen_index_vector>(n);
            row_keys    = make_shared<vector<uint>>();
            total_codes = 0;
            data->fill(0);
        };
};

void insert_example_indices_into_matrix(
    Vocab& word_vocab,
    Databatch& databatch,
    const vector<string>& example,
    size_t& row) {
    auto description_length = example.size();
    (*databatch.data)(row, 0) = word_vocab.word2index[START];
    for (size_t j = 0; j < description_length; j++) {
        (*databatch.data)(row, j + 1) = word_vocab.word2index.find(example[j]) != word_vocab.word2index.end() ? word_vocab.word2index[example[j]] : word_vocab.unknown_word;
        utils::add_to_set(*databatch.row_keys, (*databatch.data)(row, j + 1));
    }
    (*databatch.data)(row, description_length + 1) = word_vocab.word2index[utils::end_symbol];
    (*databatch.codelens)(row) = description_length + 1;
    databatch.total_codes += description_length + 1;
}

Databatch convert_sentences_to_indices(
    const vector<vector<string>>& examples,
    Vocab& word_vocab,
    size_t num_elements,
    vector<size_t>::iterator indices,
    vector<size_t>::iterator lengths_sorted) {

    auto indices_begin = indices;
    Databatch databatch(num_elements, *std::max_element(lengths_sorted, lengths_sorted + num_elements));
    utils::add_to_set(*databatch.row_keys, word_vocab.word2index[utils::end_symbol]);
    utils::add_to_set(*databatch.row_keys, word_vocab.word2index[START]);
    for (size_t k = 0; k < num_elements; k++)
        insert_example_indices_into_matrix(
            word_vocab,
            databatch,
            examples[*(indices++)],
            k);
    return databatch;
}

vector<Databatch> create_dataset(
    const vector<vector<string>>& examples,
    Vocab& word_vocab,
    size_t minibatch_size) {

    vector<Databatch> dataset;
    vector<size_t> lengths = vector<size_t>(examples.size());
    for (size_t i = 0; i != lengths.size(); ++i) lengths[i] = examples[i].size() + 2;
    vector<size_t> lengths_sorted(lengths);

    auto shortest = utils::argsort(lengths);
    std::sort(lengths_sorted.begin(), lengths_sorted.end());
    size_t piece_size = minibatch_size;
    size_t so_far = 0;

    auto shortest_ptr = lengths_sorted.begin();
    auto end_ptr = lengths_sorted.end();
    auto indices_ptr = shortest.begin();

    while (shortest_ptr != end_ptr) {
        dataset.emplace_back( convert_sentences_to_indices(
            examples,
            word_vocab,
            min(piece_size, lengths.size() - so_far),
            indices_ptr,
            shortest_ptr) );
        shortest_ptr += min(piece_size,          lengths.size() - so_far);
        indices_ptr  += min(piece_size,          lengths.size() - so_far);
        so_far        = min(so_far + piece_size, lengths.size());
    }
    return dataset;
}

/**
get word vocab
--------------

Collect a mapping from words to unique indices
from a collection of Annnotate Parse Trees
from the Stanford Sentiment Treebank, and only
keep words ocurring more than some threshold
number of times `min_occurence`

Inputs
------

std::vector<SST::AnnotatedParseTree::shared_tree>& trees : Stanford Sentiment Treebank trees
                                       int min_occurence : cutoff appearance of words to include
                                                           in vocabulary.


Outputs
-------

Vocab vocab : the vocabulary extracted from the trees with the
              addition of a special "**START**" word.

**/
Vocab get_word_vocab(const vector<vector<string>>& examples, int min_occurence) {
    auto index2word  = utils::get_vocabulary(examples, min_occurence);
    Vocab vocab(index2word);
    vocab.word2index[START] = vocab.index2word.size();
    vocab.index2word.emplace_back(START);
    return vocab;
}

/**
Reconstruct
-----------

Condition on the special "**START**" word, generate
a sentence from the language model and output it
to standard out.

Inputs
------
               StackedModel<T>& model : language model to query
           const Databatch& minibatch : where to pull the sample from
                         const int& i : what row to use from the databatch
              const Vocab& word_vocab : the word vocabulary with a lookup table
                                        mapping unique words to an index and
                                        vice-versa.                        


**/
void reconstruct(
    StackedModel<REAL_t>& model,
    const Databatch& minibatch,
    const int& i,
    const Vocab& word_vocab) {
    std::cout << "Reconstruction \"";
    for (int j = 0; j < 1; j++)
        std::cout << word_vocab.index2word[(*minibatch.data)(i, j)] << " ";
    std::cout << "\"\n => ";
    std::cout << model.reconstruct_string(
        minibatch.data->row(i).head(1),
        word_vocab,
        (*minibatch.codelens)(i),
        0) << std::endl;
}

template<typename T>
void cost_update(
    StackedModel<T>& model,
    graph_t& G,
    const Databatch& minibatch,
    T& cost) {
    cost = model.masked_predict_cost(
        G,
        minibatch.data, // the sequence to draw from
        minibatch.data, // what to predict (the words offset by 1)
        1,
        minibatch.codelens,
        0
    );
}

template<typename T>
T average_error(StackedModel<T>& model,
    const vector<Databatch>& dataset) {
	T cost = 0.0;
	auto G = graph_t(false); // create a new graph for each loop)
    int full_code_size(0);
	vector<thread> workers;
    vector<T> costs(dataset.size());
    for (int i = 0; i < dataset.size(); i++) {
        workers.emplace_back(
            cost_update<REAL_t>,
            ref(model),
            ref(G),
            ref(dataset[i]),
            ref(cost)
            );
        full_code_size += dataset[i].total_codes;
    }
    for (auto& worker : workers)
        worker.join();
    for (auto& v : costs) cost += v;
    return cost / full_code_size;
}

template REAL_t average_error(StackedModel<REAL_t>&, const vector<Databatch>&);

/**
Training Loop
-------------

Go through a single epoch of training by updating
parameters once for each minibatch in the dataset.
Takes an optimizer, model, and dataset and performs
several steps of gradient descent.
Moreover every `report_frequency` epochs this will
output the current training error and perform
a sentence reconstruction from the current model.

See `reconstruct`

Inputs
------
               StackedModel<T>& model : language model to train
     const vector<Databatch>& dataset : sentences broken into minibatches to
                                        train model on.
              const Vocab& word_vocab : the word vocabulary with a lookup table
                                        mapping unique words to an index and
                                        vice-versa.
                         const T& rho : rho parameter to control Adadelta decay rate
                            S& Solver : Solver handling updates to parameters using
                                        a specific regimen (SGD, Adadelta, AdaGrad, etc.)
std::vector<std::shared_ptr<Mat<T>>>& : references to model parameters.
          const int& report_frequency : how often to reconstruct a sentence and display
                                        training progress (KL divergence w.r.t. training
                                        data)
                     const int& epoch : how many epochs of training has been done so far.
                              T& cost : store the KL divergence w.r.t. training data here.

**/
template<typename T, typename S>
void training_loop(StackedModel<T>& model,
    const vector<Databatch>& dataset,
    const Vocab& word_vocab,
    S& solver,
    vector<shared_ptr<mat>>& parameters,
    const int& report_frequency,
    const int& epoch,
    T& cost,
    const int& patience) {
    int full_code_size = 0;
    int i = 0;
    auto random_batch_order = utils::random_arange(dataset.size());
    for (auto& j : random_batch_order) {
    	auto& minibatch = dataset[j];
    	auto G = graph_t(true);      // create a new graph for each loop
        cost += model.masked_predict_cost(
            G,
            minibatch.data, // the sequence to draw from
            minibatch.data, // what to predict (the words offset by 1)
            1,
            minibatch.codelens,
            0
        );
        model.embedding->sparse_row_keys = minibatch.row_keys;
        full_code_size += minibatch.total_codes;
        G.backward(); // backpropagate
        solver.step(parameters, 0.0); // One step of gradient descent

        std::cout << "epoch (" << epoch << "." << i++ << ") KL error = " << std::fixed
                                  << std::setw( 5 ) // keep 7 digits
                                  << std::setprecision( 3 ) // use 3 decimals
                                  << std::setfill( ' ' ) << cost / full_code_size << "\r" << std::flush;
    }
    if (epoch % report_frequency == 0) {
        std::cout << "epoch (" << epoch << ") KL error = "
                                  << std::fixed
                                  << std::setw( 5 ) // keep 7 digits
                                  << std::setprecision( 3 ) // use 3 decimals
                                  << std::setfill( ' ' ) << cost / full_code_size << " patience = " << patience << std::endl;
        auto& random_batch = dataset[utils::randint(0, dataset.size() - 1)]; 
        auto random_example_index = utils::randint(0, random_batch.data->rows() - 1);
        reconstruct(model, random_batch, random_example_index, word_vocab);
    }
}


/**
Train Model
-----------

Train a single language model on a corpus of minibatches of sentences
predicting the next word given the current sequence. Stops training
when max number of epochs is reached, training stops decreasing error
for 5 epochs, or cost dips below the cutoff.

Inputs
------

       optparse::Values& options : CLI arguments controling input size,
                                   hidden size, and number of stacked LSTMs
const vector<Databatch>& dataset : sentences broken into minibatches to
                                   train model on.
         const Vocab& word_vocab : the word vocabulary with a lookup table
                                   mapping unique words to an index and
                                   vice-versa.
                    const T& rho : rho parameter to control Adadelta decay rate
     const int& report_frequency : how often to reconstruct a sentence and display
                                   training progress (KL divergence w.r.t. training
                                   data)
               const int& epochs : maximum number of epochs to train for.
     const string& save_location : where to save the model after training.

**/
template<typename T, class S>
void train_model(
    optparse::Values& options, 
    const vector<Databatch>& dataset,
    const vector<Databatch>& validation_set,
    const Vocab& word_vocab,
    const T& rho,
    const int& report_frequency,
    const T& cutoff,
    const int& epochs,
    const string& save_location) {
    // Build Model:
    StackedModel<T> model(word_vocab.index2word.size(),
            from_string<int>(options["input_size"]),
            from_string<int>(options["hidden"]),
            from_string<int>(options["stack_size"]) < 1 ? 1 : from_string<int>(options["stack_size"]),
            word_vocab.index2word.size());
    model.embedding->sparse = from_string<int>(options["sparse"]) > 0;
    auto parameters = model.parameters();
    S solver(parameters, rho, 1e-9, 5.0);
    int i = 0;
    auto cost = std::numeric_limits<REAL_t>::infinity();
    T new_cost = 0.0;
    int patience = 0;
    while (cost > cutoff && i < epochs && patience < 5) {
        new_cost = 0.0;
        training_loop(model, dataset, word_vocab, solver, parameters, report_frequency, i, new_cost, patience);
        new_cost = average_error<T>(model, validation_set);
        if (new_cost >= cost) patience++;
        cost = new_cost;
        i++;
    }
    if (save_location != "") {
        std::cout << "Saving model to \""
                  << save_location << "/\"" << std::endl;
        model.save(save_location);
    }
}

std::pair<Vocab, vector<Databatch>> load_dataset_and_vocabulary(const string& fname, int min_occurence, int minibatch_size) {
	auto text_corpus        = utils::load_tokenized_unlabeled_corpus(fname);
	std::pair<Vocab, vector<Databatch>> pair;
	pair.first = get_word_vocab(text_corpus, min_occurence);
	pair.second = create_dataset(text_corpus, pair.first, minibatch_size);
	return pair;
}

vector<Databatch> load_dataset_with_vocabulary(const string& fname, Vocab& vocab, int minibatch_size) {
	auto text_corpus        = utils::load_tokenized_unlabeled_corpus(fname);
	return create_dataset(text_corpus, vocab, minibatch_size);
}

int main( int argc, char* argv[]) {
    auto parser = optparse::OptionParser()
        .usage("usage: --dataset [corpus_directory] --minibatch [minibatch size]")
        .description(
            "RNN Language Model using Stacked LSTMs\n"
            "--------------------------------------\n"
            "\n"
            "Predict next word in sentence using Stacked LSTMs.\n"
            "\n"
            " @author Jonathan Raiman\n"
            " @date February 15th 2015"
            );
    // Command Line Setup:
    StackedModel<REAL_t>::add_options_to_CLI(parser);
    utils::training_corpus_to_CLI(parser);
    parser.set_defaults("minibatch", "100");
    parser
        .add_option("--minibatch")
        .help("What size should be used for the minibatches ?").metavar("INT");
    parser.set_defaults("validation", "");
    parser
        .add_option("--validation")
        .help("Location of the validation dataset").metavar("FILE");
    parser.set_defaults("sparse", "1");
    parser
        .add_option("--sparse")
        .help("Use sparse embedding").metavar("INT");
    parser.set_defaults("cutoff", "2.0");
    parser
        .add_option("-ct", "--cutoff")
        .help("KL Divergence error where stopping is acceptable").metavar("FLOAT");
    auto& options = parser.parse_args(argc, argv);
    auto args = parser.args();
    if (options["dataset"] == "")    utils::exit_with_message("Error: Dataset (--dataset) keyword argument requires a value.");
    if (options["validation"] == "") utils::exit_with_message("Error: Validation (--validation) keyword argument requires a value.");
    auto report_frequency   = from_string<int>(options["report_frequency"]);
    auto rho                = from_string<REAL_t>(options["rho"]);
    auto epochs             = from_string<int>(options["epochs"]);
    auto cutoff             = from_string<REAL_t>(options["cutoff"]);
    auto dataset_vocab      = load_dataset_and_vocabulary(
    	options["dataset"],
    	from_string<int>(options["min_occurence"]),
    	from_string<int>(options["minibatch"]));
    auto validation_set     = load_dataset_with_vocabulary(
    	options["validation"],
    	dataset_vocab.first,
    	from_string<int>(options["minibatch"]));
    auto vocab_size = dataset_vocab.first.index2word.size();

    std::cout << "    Vocabulary size = " << vocab_size << " (occuring more than " << from_string<int>(options["min_occurence"]) << ")" << std::endl;
    std::cout << "Max training epochs = " << epochs << std::endl;
    std::cout << "    Training cutoff = " << cutoff << std::endl;

    train_model<REAL_t, Solver::AdaDelta<REAL_t>>(options, dataset_vocab.second, validation_set, dataset_vocab.first, rho, report_frequency, cutoff, epochs, options["save"]);

    return 0;
}