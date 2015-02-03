#ifndef LATTICE_PREDICTION_MAT_H
#define LATTICE_PREDICTION_MAT_H

#include <iostream>
#include <vector>
#include <string>
#include <ostream>
#include <memory>

/**
Ontology Branch
---------------

Small utility class for dealing with lattices with circular
references. Useful for loading in ontologies and lattices.

**/
class OntologyBranch : public std::enable_shared_from_this<OntologyBranch> {
	typedef std::shared_ptr<OntologyBranch> shared_branch;
	typedef std::weak_ptr<OntologyBranch> shared_weak_branch;
	public:
		std::vector<shared_weak_branch> parents;
		std::vector<shared_branch> children;
		std::string name;
		OntologyBranch(const std::string&);
		void add_child(shared_branch);
		void add_parent(shared_branch);
};

std::ostream& operator<<(std::ostream&, const OntologyBranch&);

#endif