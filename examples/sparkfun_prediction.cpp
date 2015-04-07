#include <algorithm>
#include <Eigen/Eigen>
#include <fstream>
#include <iterator>
#include <set>

#include "dali/core.h"
#include "dali/utils.h"
#include "dali/utils/NlpUtils.h"
#include "dali/models/StackedGatedModel.h"

using std::ifstream;
using std::istringstream;
using std::make_shared;
using std::min;
using std::shared_ptr;
using std::string;
using std::vector;
using utils::Vocab;

typedef float REAL_t;
typedef Mat<REAL_t> mat;
typedef float price_t;
typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
typedef Eigen::Matrix<REAL_t, Eigen::Dynamic, 1> float_vector;
typedef std::tuple<shared_ptr<index_mat>, shared_eigen_index_vector, shared_eigen_index_vector, shared_ptr<float_vector>> databatch_tuple;

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
    return strm << "<#Product sku=\""     << product.sku
                   << "\" name=\""        << product.name
                   << "\" description=\"" << product.description
                   << "\" categories="    << product.categories
                   << " price="           << product.price
                   << " >";
}

class Databatch {
    typedef shared_ptr< index_mat > shared_index_mat;
    typedef shared_ptr< float_vector > shared_float_vector;
    public:
        shared_index_mat data;
        shared_eigen_index_vector codelens;
        shared_eigen_index_vector start_loss;
        shared_float_vector prices;
        Databatch(databatch_tuple databatch) {
            data = std::get<0>(databatch);
            codelens = std::get<1>(databatch);
            start_loss = std::get<2>(databatch);
            prices = std::get<3>(databatch);
        };
        Databatch(
            shared_index_mat _data,
            shared_eigen_index_vector _codelens,
            shared_eigen_index_vector _start_loss,
            shared_float_vector _prices) :
                prices(_prices),
                start_loss(_start_loss),
                data(_data),
                codelens(_codelens) {};
};

void insert_product_indices_into_matrix(
    Vocab& category_vocab,
    Vocab& word_vocab,
    shared_ptr<index_mat>& mat,
    shared_eigen_index_vector& codelens,
    shared_eigen_index_vector& start_loss,
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
    (*start_loss)(row)                                      = description_length;
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
    std::get<1>(databatch) = make_shared<eigen_index_vector>(num_elements);
    std::get<2>(databatch) = make_shared<eigen_index_vector>(num_elements);
    std::get<3>(databatch) = make_shared<float_vector>(num_elements);
    auto data             = std::get<0>(databatch);
    auto codelens         = std::get<1>(databatch);
    auto start_loss       = std::get<2>(databatch);
    auto prices           = std::get<3>(databatch);
    data->fill(0);
    for (size_t k = 0; k < num_elements; k++) {
            (*prices)(k) = products[*indices].price;
            insert_product_indices_into_matrix(
                    category_vocab,
                    word_vocab,
                    data,
                    codelens,
                    start_loss,
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
    std::map<string, uint> word_occurences;
    string word;
    for (auto& product : products)
            for (auto& word : product.description) word_occurences[word] += 1;
    vector<string> list;
    for (auto& key_val : word_occurences)
            if (key_val.second >= min_occurence)
                    list.emplace_back(key_val.first);
    list.emplace_back(utils::end_symbol);
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
            while (ss >> word)
                description.push_back(word);
        } else if (args == 3) {
            istringstream ss(line);
            string category;
            while (ss >> category)
                categories.push_back(category);
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

void reconstruct(
        StackedGatedModel<REAL_t>& model,
        Databatch& minibatch,
        int& i,
        const Vocab& word_vocab,
        const Vocab& category_vocab) {
    std::cout << "Reconstruction \"";
    for (int j = 0; j < (*minibatch.start_loss)(i); j++)
        std::cout << word_vocab.index2word[(*minibatch.data)(i, j)] << " ";
    std::cout << "\"\n => ";
    std::cout << model.reconstruct_string(
            minibatch.data->row(i).head((*minibatch.start_loss)(i) + 1),
            category_vocab,
            (*minibatch.codelens)(i),
            word_vocab.index2word.size()) << std::endl;
}

template<typename T, typename S>
void training_loop(StackedGatedModel<T>& model,
            vector<Databatch>& dataset,
            const Vocab& word_vocab,
            const Vocab& category_vocab,
            S& solver,
            vector<mat>& parameters,
            int& epoch) {
        std::tuple<REAL_t, REAL_t> cost(0.0, 0.0);

        for (auto& minibatch : dataset) {
            utils::tuple_sum(cost, model.masked_predict_cost(
                    minibatch.data, // the sequence to draw from
                    minibatch.data, // what to predict
                    minibatch.start_loss,
                    minibatch.codelens,
                    word_vocab.index2word.size()
            ));
            graph::backward();// backpropagate
            solver.step(parameters); // One step of gradient descent
        }
        std::cout << "epoch (" << epoch << ") KL error = " << std::get<0>(cost)
                  << ", Memory cost = " << std::get<1>(cost) << std::endl;
        auto&        random_batch = dataset[utils::randint(0, std::min(3, (int)dataset.size() - 1))];
        auto random_example_index = utils::randint(0, random_batch.data->rows() - 1);

        reconstruct(model, random_batch, random_example_index, word_vocab, category_vocab);

}

int main(int argc, char *argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Sparkfun Dataset Prediction\n"
        "---------------------------\n"
        "Use StackedLSTMs to predict SparkFun categories in"
        " sequential fashion. Moreover, use a Multi Layer Perceptron "
        " reading hidden LSTM activations to predict pricing."
        " Final network can read product description and predict it's category"
        " and price, or provide a topology for the products on SparkFun's website:\n"
        " > https://www.sparkfun.com "
        "\n"
        " @author Jonathan Raiman\n"
        " @date January 31st 2015"
    );


    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

        // TODO(sidor): Here dataset should defaults to: examples/sparkfun_dataset.txt

        int epochs           = FLAGS_epochs;
        std::string dataset_path(FLAGS_train);

        // Collect Dataset from File:
        auto products       = get_products(dataset_path);
        auto index2word     = get_vocabulary(products, FLAGS_min_occurence < 1 ? 1 : FLAGS_min_occurence);
        auto index2category = get_category_vocabulary(products);
        Vocab word_vocab(index2word);
        Vocab category_vocab(index2category, false);
        auto dataset = create_labeled_dataset(products, category_vocab, word_vocab, FLAGS_subsets);
        std::cout << "Loaded Dataset" << std::endl;
        auto vocab_size = word_vocab.index2word.size() + index2category.size() + 1;
        // TODO: renable
        auto model = StackedGatedModel<REAL_t>::build_from_CLI(FLAGS_load, vocab_size, index2category.size() + 1, true);
        auto memory_penalty = FLAGS_memory_penalty;
        model.memory_penalty = memory_penalty / dataset[0].data->cols();
        // Store all parameters in a vector:
        auto parameters = model.parameters();
        //Gradient descent optimizer:
        Solver::AdaDelta<REAL_t> solver(parameters, (REAL_t) FLAGS_rho);
        // Main training loop:
        for (int i = 0; i < epochs; ++i)
            training_loop(model, dataset, word_vocab, category_vocab, solver, parameters, i);

        maybe_save_model(&model);
        std::cout <<"\nFinal Results\n=============\n" << std::endl;
        for (auto& minibatch : dataset)
            for (int i = 0; i < minibatch.data->rows(); i++)
                reconstruct(model, minibatch, i, word_vocab, category_vocab);
        return 0;
}
