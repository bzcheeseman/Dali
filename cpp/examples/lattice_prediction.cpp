#include <fstream>
#include <iterator>
#include <algorithm>
#include <gflags/gflags.h>

#include <Eigen/Eigen>

#include "core/utils.h"
#include "core/gzstream.h"
#include "core/StackedGatedModel.h"


DEFINE_string(lattice, "", "Where to load a lattice / Ontology from ?");

static bool dummy1 = gflags::RegisterFlagValidator(&FLAGS_lattice,
                                               &utils::validate_flag_nonempty);

DEFINE_int32(memory_rampup, 1000, "Over how many epochs should the memory grow ?");
DEFINE_double(cutoff, 10.0, "KL Divergence error where stopping is acceptable");

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
typedef Graph<REAL_t> graph_t;
typedef Mat<REAL_t> mat;
typedef float price_t;
typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
typedef Eigen::Matrix<REAL_t, Eigen::Dynamic, 1> float_vector;
typedef std::pair<vector<string>, string> labeled_pair;
typedef OntologyBranch lattice_t;
typedef std::shared_ptr<lattice_t> shared_lattice_t;

class Databatch {
	typedef shared_ptr<index_mat> shared_index_mat;
	public:
		shared_index_mat data;
		shared_index_mat target_data;
		shared_eigen_index_vector codelens;
		shared_eigen_index_vector start_loss;
		Databatch(int n, int d) {
			data        = make_shared<index_mat>(n, d);
			target_data = make_shared<index_mat>(n, d);
			codelens    = make_shared<eigen_index_vector>(n);
			start_loss  = make_shared<eigen_index_vector>(n);
			data->fill(0);
			target_data->fill(-1);
		};
};

vector<string> ontology_path_to_pathnames(const vector<OntologyBranch::shared_branch>& path) {
	std::vector<string> names(path.size());
	auto steal_names = [](const string& a, const OntologyBranch::shared_branch b) { return b->name; };
	std::transform (names.begin(), names.end(), path.begin(), names.begin(), steal_names);
	return names;
}

void insert_example_indices_into_matrix(
	Vocab& lattice_vocab,
	Vocab& word_vocab,
	shared_lattice_t lattice,
	Databatch& databatch,
	labeled_pair& example,
	size_t& row) {
	auto description_length = example.first.size();
	for (size_t j = 0; j < description_length; j++)
		(*databatch.data)(row, j) = word_vocab.word2index.find(example.first[j]) != word_vocab.word2index.end() ? word_vocab.word2index[example.first[j]] : word_vocab.unknown_word;
	(*databatch.data)(row, description_length) = word_vocab.word2index[utils::end_symbol];

	auto path = lattice->random_path_from_root(example.second, 1);

	size_t j = 0;
	for (auto& node : path.first) {
		(*databatch.data)(row,        description_length + 1 + j)   = lattice_vocab.word2index[node->name] + word_vocab.word2index.size();
		(*databatch.target_data)(row, description_length + 1 + j++) = path.second[j];
	}
	// **END** for tokens is the next dimension after all the categories (the last one)
	(*databatch.data)(row, description_length + j + 1) = word_vocab.word2index[utils::end_symbol];
	(*databatch.target_data)(row, description_length + j + 1) = lattice_vocab.word2index[utils::end_symbol];
	(*databatch.codelens)(row)   = path.first.size() + 1;
	(*databatch.start_loss)(row) = description_length;
}

Databatch convert_sentences_to_indices(
	tokenized_labeled_dataset& examples,
	Vocab& lattice_vocab,
	Vocab& word_vocab,
	shared_lattice_t lattice,
	size_t num_elements,
	vector<size_t>::iterator indices,
	vector<size_t>::iterator lengths_sorted) {

	auto indices_begin = indices;
	Databatch databatch(num_elements, *std::max_element(lengths_sorted, lengths_sorted + num_elements));
	for (size_t k = 0; k < num_elements; k++)
		insert_example_indices_into_matrix(
			lattice_vocab,
			word_vocab,
			lattice,
			databatch,
			examples[*(indices++)],
			k);
	return databatch;
}

vector<Databatch> create_labeled_dataset(
	tokenized_labeled_dataset& examples,
	Vocab& lattice_vocab,
	Vocab& word_vocab,
	shared_lattice_t lattice,
	size_t subpieces) {

	vector<Databatch> dataset;
	vector<size_t> lengths = vector<size_t>(examples.size());
	for (size_t i = 0; i != lengths.size(); ++i) lengths[i] = examples[i].first.size() + lattice->max_depth() + 2;
	vector<size_t> lengths_sorted(lengths);

	auto shortest = utils::argsort(lengths);
	std::sort(lengths_sorted.begin(), lengths_sorted.end());
	size_t piece_size = ceil(((float)lengths.size()) / (float)subpieces);
	size_t so_far = 0;

	auto shortest_ptr = lengths_sorted.begin();
	auto end_ptr = lengths_sorted.end();
	auto indices_ptr = shortest.begin();

	while (shortest_ptr != end_ptr) {
		dataset.emplace_back( convert_sentences_to_indices(
			examples,
			lattice_vocab,
			word_vocab,
			lattice,
			min(piece_size, lengths.size() - so_far),
			indices_ptr,
			shortest_ptr) );
		shortest_ptr += min(piece_size, lengths.size() - so_far);
		indices_ptr  += min(piece_size, lengths.size() - so_far);
		so_far       = min(so_far + piece_size, lengths.size());
	}
	return dataset;
}

void reconstruct(
	StackedGatedModel<REAL_t>& model,
	Databatch& minibatch,
	int& i,
	const Vocab& word_vocab,
	shared_lattice_t lattice) {
	std::cout << "Reconstruction \"";
	for (int j = 0; j < (*minibatch.start_loss)(i); j++)
		std::cout << word_vocab.index2word[(*minibatch.data)(i, j)] << " ";
	std::cout << "\"\n => ";
	std::cout << model.reconstruct_lattice_string(
		minibatch.data->row(i).head((*minibatch.start_loss)(i) + 1),
		lattice,
		(*minibatch.codelens)(i)) << std::endl;
}

template<typename T, typename S>
void training_loop(StackedGatedModel<T>& model,
	vector<Databatch>& dataset,
	const Vocab& word_vocab,
	shared_lattice_t lattice,
	S& solver,
	vector<shared_ptr<mat>>& parameters,
	int& report_frequency,
	int& epoch,
	std::tuple<T, T>& cost) {
	for (auto& minibatch : dataset) {
		auto G = graph_t(true);      // create a new graph for each loop
		utils::tuple_sum(cost, model.masked_predict_cost(
			G,
			minibatch.data, // the sequence to draw from
			minibatch.target_data, // what to predict (the path down the lattice)
			minibatch.start_loss,
			minibatch.codelens,
			0
		));
		G.backward(); // backpropagate
		solver.step(parameters, 0.0); // One step of gradient descent
	}
	if (epoch % report_frequency == 0) {
		std::cout << "epoch (" << epoch << ") KL error = " << std::get<0>(cost)
		                         << ", Memory cost = " << std::get<1>(cost) << std::endl;
		auto& random_batch = dataset[utils::randint(0, dataset.size() - 1)];
		auto random_example_index = utils::randint(0, random_batch.data->rows() - 1);

		reconstruct(model, random_batch, random_example_index, word_vocab, lattice);
	}
}

int main( int argc, char* argv[]) {
    gflags::SetUsageMessage(
        "\n"
		"Lattice Prediction\n"
    	"------------\n"
    	"Teach a network to navigate a lattice "
    	" from text examples and lattice positions."
    	"\n"
    	" @author Jonathan Raiman\n"
    	" @date February 4th 2015"
    );

    gflags::ParseCommandLineFlags(&argc, &argv, true);


	auto lattice     = OntologyBranch::load(FLAGS_lattice)[0];
	auto examples    = utils::load_tokenized_labeled_corpus(FLAGS_dataset);
	auto index2word  = utils::get_vocabulary(examples, FLAGS_min_occurence);
	auto index2label = utils::get_lattice_vocabulary(lattice);
	Vocab word_vocab(index2word);
	Vocab lattice_vocab(index2label, false);
	utils::assign_lattice_ids(lattice->lookup_table, lattice_vocab, word_vocab.index2word.size());
	auto dataset = create_labeled_dataset(
		examples,
		lattice_vocab,
		word_vocab,
		lattice,
		FLAGS_subsets);
	auto max_branching_factor = lattice->max_branching_factor();
	auto vocab_size = word_vocab.index2word.size() + lattice_vocab.index2word.size();
	auto model = StackedGatedModel<REAL_t>::build_from_CLI(vocab_size, max_branching_factor + 1, true);
	auto memory_penalty = FLAGS_memory_penalty;
	auto save_destination = FLAGS_save;
	auto report_frequency = FLAGS_report_frequency;
	auto rho = FLAGS_rho;
	auto epochs = FLAGS_epochs;
	auto cutoff = FLAGS_cutoff;
	auto memory_rampup = FLAGS_memory_rampup;
	// with a rampup model we start with zero memory penalty and gradually increase the memory
	// L1 penalty until it reaches the desired level.
	// this allows early exploration, but only later forces sparsity on the model
	model.memory_penalty = 0.0;
	std::cout << "Save location         = " << ((save_destination != "") ? save_destination : "N/A") << std::endl;
	// Store all parameters in a vector:
	auto parameters = model.parameters();

	//Gradient descent optimizer:
	Solver::AdaDelta<REAL_t> solver(parameters, rho, 1e-9, 5.0);
	// Main training loop:
	std::tuple<REAL_t,REAL_t> cost(std::numeric_limits<REAL_t>::infinity(), std::numeric_limits<REAL_t>::infinity());
	int i = 0;
	std::cout << "Max training epochs = " << epochs << std::endl;
	std::cout << "Training cutoff     = " << cutoff << std::endl;
	while (std::get<0>(cost) > cutoff && i < epochs) {
		std::get<0>(cost) = 0.0;
		std::get<1>(cost) = 0.0;
		model.memory_penalty = (memory_penalty / dataset[0].data->cols()) * std::min((REAL_t)1.0, ((REAL_t) (i*i) / ((REAL_t) memory_rampup * memory_rampup)));
		training_loop(model, dataset, word_vocab, lattice, solver, parameters, report_frequency, i, cost);
		i++;
	}
	if (save_destination != "") {
		model.save(save_destination);
		std::cout << "Saved Model in \"" << save_destination << "\"" << std::endl;
	}
	std::cout <<"\nFinal Results\n=============\n" << std::endl;
	for (auto& minibatch : dataset)
		for (int i = 0; i < minibatch.data->rows(); i++)
			reconstruct(model, minibatch, i, word_vocab, lattice);
	return 0;

}
