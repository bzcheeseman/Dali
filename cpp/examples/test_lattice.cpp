#include <gflags/gflags.h>

#include "dali/core.h"
#include "dali/utils.h"


DEFINE_string(input_path, "", "Path to test tree (optional)");
DEFINE_string(corpus_path, "", "Path to labeled pairs (optional)");

using std::vector;
using std::string;
using std::make_shared;
using utils::OntologyBranch;
using std::stringstream;

void show_tree(std::vector<std::shared_ptr<OntologyBranch>> tree) {
        for (auto& r : tree)
                std::cout << *r << std::endl;
}

void test_lattice () {
        string data_folder;
        {
            auto file_path_split = utils::split(__FILE__, '/');
            stringstream ss;
            ss << "/";
            for (int i = 0; i < file_path_split.size() - 2; i++) {
                ss << file_path_split[i] << "/";
            }
            ss << "data/";
            data_folder = ss.str();
        }

        std::cout << "Constructing a new lattice" << std::endl;

        auto root = make_shared<OntologyBranch>("root");

        for (auto& v : {"Joe", "Bob", "Max", "Mary", "Jane", "Goodwin"})
                make_shared<OntologyBranch>(v)->add_parent(root);

        auto root2 = make_shared<OntologyBranch>("root 2");

        for (auto& child : root->children)
                child->add_parent(root2);

        // Visualize root 1's children
        std::cout << *root << std::endl;
        // Visualize root 2's children
        std::cout << *root2 << std::endl;

        // Output the name of both parents (verify they are indeed 2)
        for (auto& par : root->children[2]->parents)
                std::cout << "parent name => \"" << par.lock()->name << "\"" << std::endl;

        std::cout << "Saving a lattice to \"" << data_folder << "lattice2.txt\""     << std::endl;

        // Test saving a lattice file
        root->save(data_folder + "lattice2.txt");
        // Test saving a gzipped file:
        root->save(data_folder + "lattice2.txt.gz");


        std::cout << "Load a lattice from \"" << data_folder << "lattice.txt\"" << std::endl;
        // Test loading a lattice in from a text file:
        auto loaded_tree = OntologyBranch::load(data_folder + "lattice.txt");
        std::cout << data_folder + "lattice.txt" << std::endl;
        show_tree(loaded_tree);

        assert(loaded_tree.size() > 0);
        assert(loaded_tree[0]->lookup_table->find("root 2") != loaded_tree[0]->lookup_table->end());

        OntologyBranch::shared_branch root2_loaded = loaded_tree[0]->lookup_table->at("root 2");

        int found = 0;
        for (auto& child : root->children) {
            for (auto& subchild : root2_loaded->children) {
                if (subchild->name == child->name) {
                    found += 1;
                    break;
                }
            }
        }
        std::cout << "Found " << found << "/" << root->children.size() << " children in loaded root 2 " << std::endl;
}

int main (int argc, char *argv[]) {
        GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Test Lattice\n"
        "------------\n"
        "Visualize a lattice (or generalized ontology)"
        " by loading it into using this parser. "
        " Format for a lattice is one edge declared per"
        " line, with relationships of the form \"A\"->\"B\""
        " meaning \"A\" is the parent of \"B\"."
        " As a lattice, \"B\" may have multiple parents, and \"A\""
        " can of course have multiple children. "
        "\n"
        " @author Jonathan Raiman\n"
        " @date February 3rd 2015"
    );

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);


        if (!FLAGS_input_path.empty()) {
                show_tree(OntologyBranch::load(FLAGS_input_path));
                if (!FLAGS_corpus_path.empty()) {
                        std::cout << "Loading labeled corpus pairs" << std::endl;
                        auto corpus = utils::load_labeled_corpus(FLAGS_corpus_path);
                        std::cout << "Found " << corpus.size() << " labeled pairs:" << std::endl;
                        for (auto& v : corpus)
                                std::cout << "\"" << v.first << "\" => \"" << v.second << "\""<< std::endl;
                }
        } else test_lattice();

        return 0;
}
