#include <fstream>
#include <iterator>
#include <algorithm>
#include <Eigen>
#include "../utils.h"
#include "../SST.h"
#include "../gzstream.h"
#include "../StackedGatedModel.h"
#include "../OptionParser/OptionParser.h"
using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::istringstream;
using std::stringstream;
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

int main( int argc, char* argv[]) {
    auto parser = optparse::OptionParser()
        .usage("usage: --dataset [corpus_directory] -s --index2target [index2target_file] [# of minibatches]")
        .description(
            "Sentiment Analysis as Competition amongst Language Models\n"
            "---------------------------------------------------------\n"
            "\n"
            "We present a dual formulation of the word sequence classification\n"
            "task: we treat each label’s examples as originating from different\n"
            "languages and we train language models for each label; at test\n"
            "time we compare the likelihood of a sequence under each label’s\n"
            "language model to find the most likely assignment.\n"
            "\n"
            " @author Jonathan Raiman\n"
            " @date February 13th 2015"
            );
    utils::training_corpus_to_CLI(parser);
    auto& options = parser.parse_args(argc, argv);
    auto args = parser.args();
    if (options["dataset"] == "")
        utils::exit_with_message("Error: Dataset (--dataset) keyword argument requires a value.");
    auto sentiment_treebank = SST::load(options["dataset"]);
    std::cout << "Loaded " << sentiment_treebank.size() << " unique trees " << std::endl;
    std::cout << "Example tree : " << *sentiment_treebank[sentiment_treebank.size()-1] << std::endl;



    return 0;
}