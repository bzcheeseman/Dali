#include <algorithm>
#include <fstream>
#include <gflags/gflags.h>
#include <iterator>

#include "dali/core.h"
#include "dali/utils.h"
#include "dali/models/StackedGatedModel.h"

DEFINE_string(index2target, "", "Location of Index2Target file with mapping from integer to target name.");
DEFINE_string(lattice, utils::dir_join({ STR(DALI_DATA_DIR), "mini_wiki.txt"}), "Where to load a lattice / Ontology from ?");
DEFINE_string(train, utils::dir_join({ STR(DALI_DATA_DIR) , "protobuf_sample"}), "Where should the protobuff dataset be loaded from?");
DEFINE_int32(min_occurence, 2, "Minimum occurence of a word to be included in Vocabulary.");
DEFINE_string(root_name, "__ROOT__", "How is the root called in the loaded lattice.");
DEFINE_string(branch_prefix, "Category:", "With what string do branches in the lattice begin with.");

static bool dummy3 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_lattice,
                                                             &utils::validate_flag_nonempty);
static bool dummy4 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_index2target,
                                                             &utils::validate_flag_nonempty);

using std::ifstream;
using std::istringstream;
using std::make_shared;
using std::min;
using std::shared_ptr;
using std::string;
using std::vector;
using utils::OntologyBranch;
using utils::Vocab;

typedef float REAL_t;
typedef Mat<REAL_t> mat;
typedef float price_t;
typedef std::pair<vector<string>, string> labeled_pair;
typedef OntologyBranch lattice_t;
typedef std::shared_ptr<lattice_t> shared_lattice_t;

int main( int argc, char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Lattice Prediction from Protobuff\n"
        "---------------------------------\n"
        "Load a labeled corpus from Protocol Buffers\n"
        " @author Jonathan Raiman\n"
        " @date February 10th 2015"
    );
    ELOG(default_preferred_device == DEVICE_GPU);

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

        auto index2concept = utils::load_list(FLAGS_index2target);
        std::cout << "Loaded " << index2concept.size() << " unique concept names " << std::endl;
        auto examples      = utils::load_protobuff_dataset(FLAGS_train, index2concept);
        std::cout << "Loaded " << examples.size() << " examples,";
        auto index2word    = utils::get_vocabulary(examples, FLAGS_min_occurence, 0);
        Vocab word_vocab(index2word);
        std::cout <<" with a total vocabulary of " << word_vocab.size()
                  <<" words (occuring more than "  << FLAGS_min_occurence << " times)"<< std::endl << std::endl;
        std::cout <<"First examples:" << std::endl;
        std::cout <<"---------------" << std::endl;
        for (int i = 0; i < 5 ; i++) {
                std::cout << examples[i][0] << std::endl;
                std::cout << examples[i][1] << std::endl;
                std::cout << std::endl;
        }
        std::cout << "Loading Lattice" << std::endl;
        auto lattice_roots = OntologyBranch::load(FLAGS_lattice);

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
            if (kv.first == FLAGS_root_name)
                continue;
            if (utils::startswith(kv.first, FLAGS_branch_prefix)) {
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

        if (num_fixpoints > 0)
            mean_fixpoint_degree /= num_fixpoints;
        if (num_concepts > 0)
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
        return 0;
}
