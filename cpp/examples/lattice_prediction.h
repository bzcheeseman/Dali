#ifndef LATTICE_PREDICTION_MAT_H
#define LATTICE_PREDICTION_MAT_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <ostream>
#include <memory>
#include <set>
#include <map>
#include "../utils.h"

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
		void save(std::string, std::ios_base::openmode = std::ios::out);
		static std::vector<shared_branch> load(std::string);
		static void add_lattice_edge(const std::string&, const std::string&,
			std::map<std::string, shared_branch>&, std::vector<shared_branch>& parentless);
		OntologyBranch(const std::string&);
		void add_child(shared_branch);
		void add_parent(shared_branch);
		static std::vector<std::string> split_str(const std::string&, const std::string&);
};

// define hash code for OntologyBranch
namespace std {
	template <> struct hash<OntologyBranch> {
		std::size_t operator()(const OntologyBranch&) const;
	};
}

std::ostream& operator<<(std::ostream&, const OntologyBranch&);

#endif