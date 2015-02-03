#include "lattice_prediction.h"
using std::vector;
using std::string;
using std::make_shared;

int main () {

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

	return 0;
}