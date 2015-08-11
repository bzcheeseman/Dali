#include <algorithm>
#include <fstream>
#include <gflags/gflags.h>
#include <iterator>

#include "dali/core.h"
#include "dali/utils.h"
#include "dali/utils/stacked_model_builder.h"
#include "dali/utils/NlpUtils.h"
#include "dali/models/StackedGatedModel.h"


DEFINE_string(lattice, "", "Where to load a lattice / Ontology from ?");

static bool dummy3 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_lattice,
                                               &utils::validate_flag_nonempty);
static bool dummy4 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_train,
                                               &utils::validate_flag_nonempty);

DEFINE_int32(memory_rampup, 1000, "Over how many epochs should the memory grow ?");
DEFINE_double(cutoff,       10.0, "KL Divergence error where stopping is acceptable");
DEFINE_int32(minibatch,     100,  "What size should be used for the minibatches ?");
DEFINE_int32(repeats,       1,    "How many alternate paths should be shown.");

using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::istringstream;
using std::string;
using std::min;
using utils::Vocab;
using utils::OntologyBranch;
using utils::tokenized_labeled_dataset;

typedef float REAL_t;
typedef Mat<REAL_t> mat;
typedef float price_t;
typedef std::pair<vector<string>, string> labeled_pair;
typedef OntologyBranch lattice_t;
typedef std::shared_ptr<lattice_t> shared_lattice_t;

template<typename R>
struct LatticeBatch : public Batch<R> {
    LatticeBatch() = default;
    LatticeBatch(int max_example_length, int num_examples) {
        this->data   = Mat<int>(max_example_length, num_examples);
        // in a language task, data is our target.
        this->target = Mat<int>(max_example_length, num_examples);
        this->mask   = Mat<R>(max_example_length, num_examples);
        this->code_lengths.clear();
        this->code_lengths.resize(num_examples);
        this->total_codes = 0;
    };

    void add_example(
            const Vocab& lattice_vocab,
            const Vocab& word_vocab,
            shared_lattice_t lattice,
            const vector<vector<string>>& example,
            const OntologyBranch::path_t& path,
            size_t& example_idx) {
        auto lattice_label = utils::join(example[1], " ");
        auto& tokens       = example.at(0);
        // example 1 is the sentence
        // example 2 is the label from the lattice
        auto description_length = tokens.size();
        for (size_t j = 0; j < description_length; j++)
            this->data.w(j, example_idx) = word_vocab[tokens[j]];
        this->data.w(description_length, example_idx) = word_vocab.word2index.at(utils::end_symbol);

        size_t j = 0;
        for (auto& node : path.first) {
            // lattice index is offset by all words +
            // offset using lattice_vocab indexing
            this->data.w(description_length   + j + 1, example_idx) = lattice_vocab.word2index.at(node->name) + word_vocab.word2index.size();
            this->target.w(description_length + j, example_idx)     = path.second.at(j);
            this->mask.w(description_length   + j, example_idx)     = 1.0;
            j++;
        }
        // **END** for tokens is the next dimension after all the categories (the last one)
        this->data.w(description_length   + j + 1, example_idx) = word_vocab.word2index.at(utils::end_symbol);
        this->target.w(description_length + j, example_idx) = 0; // end predictions for path
        this->mask.w(description_length   + j, example_idx) = 1.0;

        this->code_lengths[example_idx] = path.first.size() + 1;
        this->total_codes += path.first.size() + 1;
    }
};

vector<string> ontology_path_to_pathnames(const vector<OntologyBranch::shared_branch>& path) {
    std::vector<string> names(path.size());
    auto steal_names = [](const string& a, const OntologyBranch::shared_branch b) { return b->name; };
    std::transform (names.begin(), names.end(), path.begin(), names.begin(), steal_names);
    return names;
}

template<typename R>
LatticeBatch<R> convert_sentences_to_indices(
        const vector<vector<vector<string>>*>& examples,
        const vector<OntologyBranch::path_t>& paths,
        const Vocab& lattice_vocab,
        const Vocab& word_vocab,
        shared_lattice_t lattice,
        size_t batch_size,
        vector<size_t>::iterator indices,
        vector<size_t>::iterator lengths_sorted) {

    auto indices_begin = indices;

    LatticeBatch<R> batch(
        *std::max_element(lengths_sorted, lengths_sorted + batch_size),
        batch_size
    );
    for (size_t example_idx = 0; example_idx < batch_size; example_idx++) {
        batch.add_example(
            lattice_vocab,
            word_vocab,
            lattice,
            *examples[*(indices)],
            paths[*(indices)],
            example_idx
        );
        indices++;
    }
    return batch;
}

template<typename R>
vector<LatticeBatch<R>> create_labeled_dataset(
        const vector<vector<vector<string>>*>& examples,
        Vocab& lattice_vocab,
        Vocab& word_vocab,
        shared_lattice_t lattice,
        int minibatch_size) {

    vector<LatticeBatch<R>> dataset;
    vector<size_t> lengths = vector<size_t>(examples.size());
    vector<OntologyBranch::path_t> paths(examples.size());
    for (size_t i = 0; i != lengths.size(); ++i) {
        const auto& example = (*examples[i]);
        auto lattice_label = utils::join(example[1], " ");
        paths[i]   = lattice->random_path_from_root(lattice_label, 1);
        lengths[i] = example[0].size() + paths[i].first.size() + 2;
    }

    vector<size_t> lengths_sorted(lengths);
    auto shortest = utils::argsort(lengths);
    std::sort(lengths_sorted.begin(), lengths_sorted.end());
    int so_far = 0;

    auto shortest_ptr = lengths_sorted.begin();
    auto end_ptr      = lengths_sorted.end();
    auto indices_ptr  = shortest.begin();

    while (shortest_ptr != end_ptr) {
        dataset.emplace_back(
            convert_sentences_to_indices<R>(
                examples,
                paths,
                lattice_vocab,
                word_vocab,
                lattice,
                min(minibatch_size, (int) (lengths.size() - so_far)),
                indices_ptr,
                shortest_ptr
            )
        );
        shortest_ptr += min(minibatch_size, (int) (lengths.size() - so_far));
        indices_ptr  += min(minibatch_size, (int) (lengths.size() - so_far));
        so_far       = min(so_far + minibatch_size, (int)lengths.size());
    }
    return dataset;
}

template<typename R>
void reconstruct(
        StackedGatedModel<R>& model,
        const LatticeBatch<R>& minibatch,
        int& example_idx,
        const Vocab& word_vocab,
        shared_lattice_t lattice) {
    std::cout << "Reconstruction \"";
    std::vector<typename utils::Vocab::ind_t> primer;
    for (int t = 0; t < minibatch.max_length(); t++) {
        if (minibatch.mask.w(t, example_idx) == 0.0) {
            std::cout << word_vocab.index2word[minibatch.data.w(t, example_idx)] << " ";
            primer.emplace_back(minibatch.data.w(t, example_idx));
        } else {
            break;
        }
    }
    std::cout << "\"\n => ";

    std::cout << model.reconstruct_lattice_string(
        &primer,
        lattice,
        minibatch.code_lengths[example_idx]
    ) << std::endl;
}

template<typename T>
void training_loop(StackedGatedModel<T>& model,
        const vector<LatticeBatch<T>>& dataset,
        const Vocab& word_vocab,
        shared_lattice_t lattice,
        std::shared_ptr<Solver::AbstractSolver<T>> solver,
        vector<mat>& parameters,
        int& epoch,
        std::tuple<T, T>& cost) {
    mat prediction_error, memory_error;

    ReportProgress<double> journalist(utils::MS() << "Training epoch " << epoch, dataset.size());
    int progress = 0;

    vector<const LatticeBatch<T>*> shuffled_dataset(dataset.size());
    for (int i = 0; i < dataset.size();i++) {
        shuffled_dataset[i] = &dataset[i];
    }
    std::random_shuffle(shuffled_dataset.begin(), shuffled_dataset.end());

    for (auto batch : shuffled_dataset) {
        std::tie(prediction_error, memory_error) = model.masked_predict_cost(
            batch->data,
            batch->target,
            batch->mask,
            0.0,
            0, // predict current point in time
            0
        );

        std::get<0>(cost) = std::get<0>(cost) + prediction_error.w(0);
        std::get<1>(cost) = std::get<1>(cost) + memory_error.w(0);

        prediction_error.grad();
        memory_error.grad();

        graph::backward(); // backpropagate
        solver->step(parameters); // One step of gradient descent
        journalist.tick(++progress, prediction_error.w(0) );
    }
    journalist.done();
    std::cout << "epoch (" << epoch << ") KL error = " << std::get<0>(cost)
                             << ", Memory cost = " << std::get<1>(cost) << std::endl;

    for (int i = 0; i < 10; i++) {
        auto& random_batch = dataset[utils::randint(0, dataset.size() - 1)];
        auto random_example_index = utils::randint(0, random_batch.size() - 1);
        reconstruct(model, random_batch, random_example_index, word_vocab, lattice);
    }
}

int main( int argc, char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Lattice Prediction\n"
        "------------\n"
        "Teach a network to navigate a lattice "
        " from text examples and lattice positions."
        "\n"
        " @author Jonathan Raiman\n"
        " @date February 4th 2015"
    );

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    auto lattice = OntologyBranch::load(FLAGS_lattice)[0];
    auto max_branching_factor = lattice->max_branching_factor();
    std::cout << "Max branching factor in lattice = " << max_branching_factor << std::endl;

    int number_of_columns = 2;
    auto examples  = utils::load_tsv(
        FLAGS_train,
        number_of_columns
    );

    int fact_column = 0;
    auto index2word  = utils::get_vocabulary(examples, FLAGS_min_occurence, fact_column);
    auto index2label = utils::get_lattice_vocabulary(lattice);

    {
        auto index2label2  = utils::get_vocabulary(examples, 0, 1);
        std::cout << "# unique labels = " << index2label2.size() << std::endl;
        std::cout << "# total labels  = " << index2label.size()  << std::endl;
    }

    vector<vector<vector<string>>*> examples_ptr(FLAGS_repeats * examples.size());

    int pos = 0;
    for (int i = 0; i < examples.size(); i++) {
        for (int rep = 0; rep < FLAGS_repeats; rep++) {
            examples_ptr[pos] = &examples[i];
            pos++;
        }
    }

    Vocab word_vocab(index2word);
    Vocab lattice_vocab(index2label, false);
    utils::assign_lattice_ids(lattice->lookup_table, lattice_vocab, word_vocab.size());
    auto dataset = create_labeled_dataset<REAL_t>(
        examples_ptr,
        lattice_vocab,
        word_vocab,
        lattice,
        FLAGS_minibatch);
    {
        int total_num_examples = 0;
        for (int i = 0; i < dataset.size(); i++) {
            total_num_examples += dataset[i].size();
        }
        std::cout << "# examples            = " << examples.size()    << std::endl;
        std::cout << "# examples with paths = " << total_num_examples << std::endl;
    }

    auto vocab_size = word_vocab.size() + lattice_vocab.size();
    auto model = stacked_gated_model_from_CLI<REAL_t>(FLAGS_load, vocab_size, max_branching_factor + 1, true);
    auto memory_penalty = FLAGS_memory_penalty;

    auto rho = FLAGS_rho;
    auto epochs = FLAGS_epochs;
    auto cutoff = FLAGS_cutoff;
    auto memory_rampup = FLAGS_memory_rampup;
    // with a rampup model we start with zero memory penalty and gradually increase the memory
    // L1 penalty until it reaches the desired level.
    // this allows early exploration, but only later forces sparsity on the model
    model.memory_penalty = 0.0;
    // Store all parameters in a vector:
    auto parameters = model.parameters();

    //Gradient descent optimizer:
    auto solver = Solver::construct(FLAGS_solver, parameters);

    // Main training loop:
    std::tuple<REAL_t,REAL_t> cost(std::numeric_limits<REAL_t>::infinity(), std::numeric_limits<REAL_t>::infinity());
    int i = 0;
    std::cout << "Max training epochs = " << epochs << std::endl;
    std::cout << "Training cutoff     = " << cutoff << std::endl;
    while (std::get<0>(cost) > cutoff && i < epochs) {
        std::get<0>(cost) = 0.0;
        std::get<1>(cost) = 0.0;
        model.memory_penalty = (memory_penalty / dataset[0].size()) * std::min((REAL_t)1.0, ((REAL_t) (i*i) / ((REAL_t) memory_rampup * memory_rampup)));
        training_loop(model, dataset, word_vocab, lattice, solver, parameters, i, cost);
        i++;
    }
    maybe_save_model(&model);
    std::cout <<"\nFinal Results\n=============\n" << std::endl;
    for (auto& minibatch : dataset)
        for (int i = 0; i < minibatch.size(); i++)
            reconstruct(model, minibatch, i, word_vocab, lattice);
    return 0;
}
