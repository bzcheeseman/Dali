#include "Layers.h"
#include <fstream>
#include <random>
#include <iterator>
#include <algorithm>
#include <Eigen>
#include "gzstream.h"
// test file for character prediction
using std::vector;
using std::make_shared;
using std::shared_ptr;
using utils::assign_cli_argument;
using std::ifstream;
using std::istringstream;
using std::string;
using std::min;
using utils::Vocab;

typedef float REAL_t;
typedef LSTM<REAL_t> lstm;
typedef Graph<REAL_t> graph_t;
typedef Layer<REAL_t> classifier_t;
typedef Mat<REAL_t> mat;
typedef shared_ptr<mat> shared_mat;
typedef float price_t;
typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
typedef Eigen::Matrix<uint, Eigen::Dynamic, 1> index_vector;
typedef Eigen::Matrix<REAL_t, Eigen::Dynamic, 1> float_vector;
typedef std::tuple<shared_ptr<index_mat>, shared_ptr<index_vector>, shared_ptr<index_vector>, shared_ptr<float_vector>> databatch_tuple;

class Product {
	public:
		string sku;
		string name;
		vector<string> description;
		vector<string> categories;
		price_t price;
		Product(string _sku, string _name, vector<string> _description, vector<string> _categories, float _price) :
			sku(_sku), name(_name), description(_description), categories(_categories), price(_price) {}
};

std::ostream& operator<<(std::ostream& strm, const Product& product) {
	return strm << "<#Product sku=\"" << product.sku
			           << "\" name=\"" << product.name
			           << "\" description=\"" << product.description
			           << "\" categories=" << product.categories
			           << " price=" << product.price << " >";
}

class Databatch {
	typedef shared_ptr< index_mat > shared_index_mat;
	typedef shared_ptr< index_vector > shared_index_vector;
	typedef shared_ptr< float_vector > shared_float_vector;
	public:
		shared_index_mat data;
		shared_index_vector codelens;
		shared_index_vector sequence_lengths;
		shared_float_vector prices;
		Databatch(databatch_tuple databatch) {
			data = std::get<0>(databatch);
			codelens = std::get<1>(databatch);
			sequence_lengths = std::get<2>(databatch);
			prices = std::get<3>(databatch);
		};
		Databatch(
			shared_index_mat _data,
			shared_index_vector _codelens,
			shared_index_vector _sequence_lengths,
			shared_float_vector _prices) :
				prices(_prices),
				sequence_lengths(_sequence_lengths), 
				data(_data),
				codelens(_codelens) {};
};

void insert_product_indices_into_matrix(
	Vocab& category_vocab,
	Vocab& word_vocab,
	shared_ptr<index_mat>& mat,
	shared_ptr<index_vector>& codelens,
	shared_ptr<index_vector>& sequence_lengths,
	Product& product,
	size_t& row) {
	auto description_length = product.description.size();
	auto categories_length  = product.categories.size();
	for (size_t j = 0; j < description_length; j++)
		(*mat)(row, j) = word_vocab.word2index.find(product.description[j]) != word_vocab.word2index.end() ? word_vocab.word2index[product.description[j]] : word_vocab.unknown_word;
	(*mat)(row, description_length) = word_vocab.word2index[utils::end_symbol];
	for (size_t j = 0; j < categories_length; j++)
		(*mat)(row, description_length + j + 1) = category_vocab.word2index[product.categories[j]] + word_vocab.word2index.size();
	// **END** for tokens is the next dimension after all the categories (the last one)
	(*mat)(row, description_length + categories_length + 1) = word_vocab.word2index.size() + category_vocab.word2index.size();
	(*codelens)(row)                                        = categories_length + 1;
	(*sequence_lengths)(row)                                = description_length;
}

databatch_tuple convert_sentences_to_indices(
	vector<Product>& products,
	Vocab& category_vocab,
	Vocab& word_vocab,
	size_t num_elements,
	vector<size_t>::iterator indices,
	vector<size_t>::iterator lengths_sorted) {

	auto indices_begin = indices;
	auto max_len_example = *std::max_element(lengths_sorted, lengths_sorted + num_elements);
	databatch_tuple databatch;
	std::get<0>(databatch) = make_shared<index_mat>(num_elements, max_len_example);
	std::get<1>(databatch) = make_shared<index_vector>(num_elements);
	std::get<2>(databatch) = make_shared<index_vector>(num_elements);
	std::get<3>(databatch) = make_shared<float_vector>(num_elements);
	auto data             = std::get<0>(databatch);
	auto codelens         = std::get<1>(databatch);
	auto sequence_lengths = std::get<2>(databatch);
	auto prices           = std::get<3>(databatch);
	data->fill(0);
	for (size_t k = 0; k < num_elements; k++) {
		(*prices)(k) = products[*indices].price;
		insert_product_indices_into_matrix(
			category_vocab,
			word_vocab,
			data,
			codelens,
			sequence_lengths,
			products[*indices],
			k);
		indices++;
	}
	return databatch;
}

vector<Databatch> create_labeled_dataset(vector<Product>& products,
	Vocab& category_vocab,
	Vocab& word_vocab,
	size_t subpieces) {

	vector<Databatch> dataset;
	vector<size_t> lengths = vector<size_t>(products.size());
	for (size_t i = 0; i != lengths.size(); ++i) lengths[i] = products[i].description.size() + products[i].categories.size() + 2;
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
			products,
			category_vocab,
			word_vocab,
			min(piece_size, lengths.size() - so_far),
			indices_ptr,
			shortest_ptr) );
		shortest_ptr += min(piece_size, lengths.size() - so_far);
		indices_ptr  += min(piece_size, lengths.size() - so_far);
		so_far       = min(so_far + piece_size, lengths.size());
	}
	return dataset;
}

#include <set>
vector<string> get_category_vocabulary(vector<Product>& products) {
	std::set<string> categories;
	string word;
	for (auto& product : products)
		for (auto& category : product.categories)
			categories.insert(category);
	vector<string> list;
	for (auto& key_val : categories)
		list.emplace_back(key_val);
	return list;
}

vector<string> get_vocabulary(vector<Product>& products, int min_occurence) {
	std::unordered_map<string, uint> word_occurences;
	string word;
	for (auto& product : products)
		for (auto& word : product.description) word_occurences[word] += 1;
	vector<string> list;
	for (auto& key_val : word_occurences)
		if (key_val.second >= min_occurence)
			list.emplace_back(key_val.first);
	return list;
}

template<typename T>
void stream_to_products(T& ostream, vector<Product>& products) {
	string line;
	string sku;
	string name;
	vector<string> description;
	vector<string> categories;
	price_t price;
	int args = 0;
	while (std::getline(ostream, line)) {
		if (args == 0) {
			sku = line;
		} else if (args == 1) {
			name = line;
		} else if (args == 2) {
			istringstream ss(line);
			string word;
			while (ss >> word) description.push_back(word);
		} else if (args == 3) {
			istringstream ss(line);
			string category;
			while (ss >> category) categories.push_back(category);
		} else if (args == 4) {
			istringstream ss(line);
			ss >> price;
		}
		args++;
		if (args == 5) {
			products.emplace_back(sku, name, description, categories, price);
			args = 0;
			categories.clear();
			description.clear();
		}
	}
}

/**
Load products from textfile
into memory, and create a vector of
Product objects.
*/
vector<Product> get_products(const string& filename) {
	vector<Product> products;
	if (utils::is_gzip(filename)) {
		igzstream infilegz(filename.c_str());
		stream_to_products(infilegz, products);
	} else {
		ifstream infile(filename);
		stream_to_products(infile, products);
	}
	return products;
}

REAL_t cost_fun(
	graph_t& G,
	vector<int>& hidden_sizes,
    vector<lstm>& cells,
    shared_mat embedding,
    classifier_t& classifier,
    Databatch& minibatch) {

	auto initial_state = lstm::initial_states(hidden_sizes);
	auto num_hidden_sizes = hidden_sizes.size();

	// shared_mat input_vector;
	// shared_mat logprobs;
	// shared_mat probs;

	REAL_t cost = 0.0;
	auto n = minibatch.data->cols();

	std::cout << "number of cols " << n << std::endl;

	// for (int i = 0; i < n-1; ++i) {
	// 	// pick this letter from the embedding
	// 	input_vector  = G.row_pluck(embedding, indices[i]);
	// 	// pass this letter to the LSTM for processing
	// 	initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
	// 	// classifier takes as input the final hidden layer's activation:
	// 	logprobs      = classifier.activate(G, initial_state.second[num_hidden_sizes-1]);
	// 	cost -= cross_entropy(logprobs, indices[i+1]);
	// }
	return cost / (n-1);
}

int main(int argc, char *argv[]) {
	// default parameters:
	int min_occurence = 2;
	int subsets       = 10;
	int input_size    = 100;
	int epochs        = 5;
	int report_frequency = 2;
	std::string dataset_path("sparkfun_dataset.txt");
	// modified with command line
	if (argc > 1) assign_cli_argument(argv[1], dataset_path,  "path");
	if (argc > 2) assign_cli_argument(argv[2], min_occurence, "min_occurence");
	if (argc > 3) assign_cli_argument(argv[3], subsets,       "subsets");
	if (argc > 4) assign_cli_argument(argv[4], input_size,    "input_size");
	if (argc > 5) assign_cli_argument(argv[5], epochs,        "epochs");
	if (argc > 6) assign_cli_argument(argv[6], report_frequency, "report_frequency");
	if (min_occurence <= 0) min_occurence = 1;

	auto products       = get_products(dataset_path);
	auto index2word     = get_vocabulary(products, min_occurence);
	auto index2category = get_category_vocabulary(products);

	Vocab word_vocab(index2word);
	Vocab category_vocab(index2category);

	auto dataset = create_labeled_dataset(products, category_vocab, word_vocab, subsets);

	std::cout << "Loaded Dataset"                                    << std::endl;
	std::cout << "Vocabulary size       = " << index2word.size()     << std::endl;
	std::cout << "Category size         = " << index2category.size() << std::endl;
	std::cout << "Number of Minibatches = " << dataset.size()        << std::endl;

	// Construct the model
	vector<int> hidden_sizes = {100, 100, 100, 100};
	auto vocab_size = index2word.size() + index2category.size() + 1;
	auto cells = StackedCells<lstm>(input_size, hidden_sizes);
	classifier_t decoder(hidden_sizes[hidden_sizes.size() - 1], index2category.size() + 1);
	auto embedding = make_shared<mat>(vocab_size, input_size, (REAL_t) 0.05);
	vector<shared_mat> parameters;
	parameters.push_back(embedding);
	auto decoder_params = decoder.parameters();
	parameters.insert(parameters.end(), decoder_params.begin(), decoder_params.end());

	for (auto& cell : cells) {
		auto cell_params = cell.parameters();
		parameters.insert(parameters.end(), cell_params.begin(), cell_params.end());
	}
	// Done constructing model;
	std::cout << "Constructed Stacked LSTMs" << std::endl;

	//Gradient descent optimizer:
	Solver<REAL_t> solver(parameters, 0.999, 1e-9, 5.0);

	// Main training loop:
	// TODO:
	// 1. Allow row pluck for many values (and many use sparse gradient updates)
	//
	// 2. Use masked loss (implement it here) for KL divergence penalty
	//
	// 3. Use masked sum (implement it here) for memory L1 penalty
	//
	// 4. Use L2 norm to penalize price predictions
	for (auto i = 0; i < epochs; ++i) {
		REAL_t cost = 0.0;
		for (auto& minibatch : dataset) {
			auto G = graph_t(true);      // create a new graph for each loop
			cost += cost_fun(
				G,                       // to keep track of computation
				hidden_sizes,            // to construct initial states
				cells,                   // LSTMs
				embedding,               // word embedding
				decoder,                 // decoder for LSTM final hidden layer
				minibatch // the sequence to predict
			);
			G.backward();                // backpropagate
			// solve it.
			solver.step(parameters, 0.01, 0.0);
		}
		if (i % report_frequency == 0)
			std::cout << "epoch (" << i << ") error = " << cost << std::endl;
	}
	return 0;
}