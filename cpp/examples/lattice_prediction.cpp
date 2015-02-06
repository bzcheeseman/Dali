#include <fstream>
#include <iterator>
#include <algorithm>
#include <Eigen>
#include "../utils.h"
#include "../gzstream.h"
#include "../StackedGatedModel.h"
#include "../OptionParser/OptionParser.h"
using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::istringstream;
using std::string;
using std::min;
using utils::Vocab;
using utils::from_string;
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
	typedef shared_ptr< index_mat > shared_index_mat;
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

int main( int argc, char* argv[]) {
	auto parser = optparse::OptionParser()
	    .usage("usage: [lattice_path] [corpus_path] -s [# of minibatches]")
	    .description(
	    	"Lattice Prediction\n"
	    	"------------\n"
	    	"Teach a network to navigate a lattice "
	    	" from text examples and lattice positions."
	    	"\n"
	    	" @author Jonathan Raiman\n"
	    	" @date February 4th 2015"
	    	);
	parser.set_defaults("subsets", "10");
	parser
		.add_option("-s", "--subsets")
		.help("Break up dataset into how many minibatches ? \n(Note: reduces batch sparsity)").metavar("INT");
	parser.set_defaults("min_occurence", "2");
	parser
		.add_option("-m", "--min_occurence")
		.help("How often a word must appear to be included in the Vocabulary \n"
			"(Note: other words replaced by special **UNKNONW** word)").metavar("INT");
	optparse::Values& options = parser.parse_args(argc, argv);
	vector<string> args = parser.args();

	int subpieces     = from_string<int>(options["subsets"]);
	int min_occurence = from_string<int>(options["min_occurence"]);

	if (args.size() > 1) {
		auto lattice    = OntologyBranch::load(args[0])[0];
		auto examples   = utils::load_tokenized_labeled_corpus(args[1]);
		auto index2word = utils::get_vocabulary(examples, min_occurence);
		auto index2label = utils::get_lattice_vocabulary(lattice);
		Vocab word_vocab(index2word);
		Vocab lattice_vocab(index2label, false);
		auto dataset = create_labeled_dataset(
			examples,
			lattice_vocab,
			word_vocab,
			lattice,
			subpieces);


	} else {
		std::cout << "usage: [lattice_path] [corpus_path]" << std::endl;
	}
}