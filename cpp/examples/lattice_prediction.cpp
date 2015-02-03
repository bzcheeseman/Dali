#include "lattice_prediction.h"

using std::string;
using std::vector;
using std::shared_ptr;

std::ostream& operator<<(std::ostream& strm, const OntologyBranch& a) {
	strm << "<#OntologyBranch name=\"" << a.name << "\"";
	if (a.children.size() > 0) {
		strm << " children={ ";
		for (auto& v : a.children)
			strm  << *v << ", ";
		strm << "}";
	}
	return strm << ">";
}

OntologyBranch::OntologyBranch(const string& _name) : name(_name) {}

void OntologyBranch::add_parent(OntologyBranch::shared_branch parent) {
	parents.emplace_back(parent);
	parent->add_child(shared_from_this());
}

void OntologyBranch::add_child(OntologyBranch::shared_branch child) {
	children.emplace_back(child);
}