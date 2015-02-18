#include <fstream>
#include <iterator>
#include <algorithm>
#include <Eigen/Eigen>
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

int main( int argc, char* argv[]) {
	auto parser = optparse::OptionParser()
	    .usage("usage: --dataset [corpus_directory] -s --index2target [index2target_file] [# of minibatches]")
	    .description(
	    	"Lattice Prediction from Protobuff\n"
	    	"---------------------------------\n"
	    	"Load a labeled corpus from Protocol Buffers\n"
	    	" @author Jonathan Raiman\n"
	    	" @date February 10th 2015"
	    	);
	
	utils::training_corpus_to_CLI(parser);
	parser.set_defaults("index2target", "");
	parser
		.add_option("--index2target")
		.help("Location of Index2Target file with mapping from integer to target name.").metavar("FILE");
	parser
		.add_option("-lt", "--lattice")
		.help("Where to load a lattice / Ontology from ?").metavar("FILE");

	auto& options = parser.parse_args(argc, argv);
	auto args = parser.args();
	if (options["dataset"] == "")
		utils::exit_with_message("Error: Dataset (--dataset) keyword argument requires a value.");
	if (options["lattice"] == "")
		utils::exit_with_message("Error: Lattice (--lattice) keyword argument requires a value.");
	if (options["index2target"] == "")
		utils::exit_with_message("Error: Index2Target (--index2target) keyword argument requires a value.");

	auto index2concept = utils::load_list(options["index2target"]);
	std::cout << "Loaded " << index2concept.size() << " unique concept names " << std::endl;
	auto examples      = utils::load_protobuff_dataset(options["dataset"], index2concept);
	std::cout << "Loaded " << examples.size() << " examples,";
	auto index2word    = utils::get_vocabulary(examples, from_string<int>(options["min_occurence"]));
	Vocab word_vocab(index2word);
	std::cout <<" with a total vocabulary of " << word_vocab.index2word.size()
	          <<" words (occuring more than "  << from_string<int>(options["min_occurence"]) << " times)"<< std::endl << std::endl;
	std::cout <<"First examples:" << std::endl;
	std::cout <<"---------------" << std::endl;
	for (int i = 0; i < 5 ; i++) {
		std::cout << examples[i].first << std::endl;
		std::cout << examples[i].second << std::endl;
		std::cout << std::endl;
	}
	std::cout << "Loading Lattice" << std::endl;
	auto lattice_roots = OntologyBranch::load(options["lattice"]);

	
	int num_fixpoints            = 0;
	int num_concepts             = 0;
	double mean_fixpoint_degree  = 0.0;
	double mean_concept_indegree = 0.0;
	int min_concept_indegree     = std::numeric_limits<int>::infinity();
	int max_concept_indegree     = 0;
	int min_fixpoint_degree      = std::numeric_limits<int>::infinity();
	int max_fixpoint_degree      = 0;
	string max_fixpoint;
	string max_concept;
	int num_disconnected_concepts = 0;
	vector<string> disconnected;
	for (auto& kv : *lattice_roots[0]->lookup_table) {
		if (kv.first == "__ROOT__") continue;
		if (utils::startswith(kv.first, "fp:")) {
			num_fixpoints++;
			mean_fixpoint_degree += kv.second->children.size();
			if (kv.second->children.size() < min_fixpoint_degree) min_fixpoint_degree = kv.second->children.size();
			if (kv.second->children.size() > max_fixpoint_degree) {
				max_fixpoint_degree = kv.second->children.size();
				max_fixpoint = kv.first;
			}
		} else {
			num_concepts++;
			mean_concept_indegree += kv.second->parents.size();
			if (kv.second->parents.size() < min_concept_indegree) {
				min_concept_indegree = kv.second->parents.size();
			}
			if (kv.second->parents.size() == 0) {
				num_disconnected_concepts++;
				disconnected.emplace_back(kv.first);
			}
			if (kv.second->parents.size() > max_concept_indegree) {
				max_concept_indegree = kv.second->parents.size();
				max_concept = kv.first;
			}
		}
	}
	mean_fixpoint_degree /= num_fixpoints;
	mean_concept_indegree /= num_concepts;
	std::cout << "Lattice Statistics\n"
			  << "------------------\n\n"
			  << "    Number of lattice roots : " << lattice_roots.size()  << "\n"
	          << " Number of nodes in lattice : " << lattice_roots[0]->lookup_table->size() << "\n"
	          << "Number of lattice fixpoints : " << num_fixpoints         << "\n"
	          << " Number of lattice concepts : " << num_concepts          << "\n"
	          << " Mean in-degree of concepts : " << mean_concept_indegree << "\n"
	          << "  Max in-degree of concepts : " << max_concept_indegree  << " (\"" << max_concept << "\")\n"
	          << "  Min in-degree of concepts : " << min_concept_indegree  << "\n"
	          << "   Mean degree of fixpoints : " << mean_fixpoint_degree  << "\n"
	          << "    Max degree of fixpoints : " << max_fixpoint_degree   << " (\"" << max_fixpoint << "\")\n"
	          << "    Min degree of fixpoints : " << min_fixpoint_degree   << std::endl;

	std::cout << "Disconnected Concepts (" << num_disconnected_concepts << ")\n"
			  << "---------------------\n"
			  << disconnected << std::endl;
	// TODO: bug with certain concepts not being exported
	// from Neo4j and there might be a similar issue with fixpoints.
	return 0;
}