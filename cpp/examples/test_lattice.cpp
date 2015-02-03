#include "lattice_prediction.h"
#include "../OptionParser/OptionParser.h"
using std::vector;
using std::string;
using std::make_shared;

void show_tree(std::vector<std::shared_ptr<OntologyBranch>> tree) {
	for (auto& r : tree)
		std::cout << *r << std::endl;
}

void test_lattice () {
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

	std::cout << "Saving a lattice to \"examples/lattice2.txt\""     << std::endl;
	// Test saving a lattice file
	root->save("examples/lattice2.txt");


	std::cout << "Load a lattice from \"examples/lattice.txt\"" << std::endl;
	// Test loading a lattice in from a text file:
	show_tree(OntologyBranch::load("examples/lattice.txt"));
}

int main (int argc, char *argv[]) {
	auto parser = optparse::OptionParser()
	    .usage("usage: [input_path]")
	    .description(
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

	optparse::Values& options = parser.parse_args(argc, argv);
	vector<string> args = parser.args();

	if (args.size() > 0)
		show_tree(OntologyBranch::load(args[0]));
	else test_lattice();

	return 0;
}