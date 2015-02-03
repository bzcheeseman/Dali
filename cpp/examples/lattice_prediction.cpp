#include "lattice_prediction.h"

using std::string;
using std::vector;
using std::shared_ptr;
using std::set;
using std::ofstream;
using std::stringstream;
using std::ifstream;
using std::make_shared;
using utils::trim;

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

std::size_t std::hash<OntologyBranch>::operator()(const OntologyBranch& k) const {
	size_t seed = 0;
	std::hash<std::string> str_hasher;
	seed ^= str_hasher(k.name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	return seed;
}

void OntologyBranch::save(string fname, std::ios_base::openmode mode) {
	auto hasher = std::hash<OntologyBranch>();
	set<size_t> visited_list;
	vector<shared_branch> open_list;

	ofstream fp;
	fp.open(fname.c_str(), mode);
	open_list.push_back(shared_from_this());

	while (open_list.size() > 0) {
		auto el = open_list[0];
		open_list.erase(open_list.begin());
		if (visited_list.count(hasher(*el)) == 0) {
			visited_list.insert(hasher(*el));
			for (auto& child : el->children) {
				fp << el->name << "->" << child->name << "\n";
				open_list.emplace_back(child);
			}
			for (auto& parent : el->parents)
				open_list.emplace_back(parent.lock());
		}
	}
}

void OntologyBranch::add_lattice_edge(
	const std::string& parent,
	const std::string& child,
	std::map<std::string, std::shared_ptr<OntologyBranch>>& map,
	std::vector<std::shared_ptr<OntologyBranch>>& parentless) {
	if (map.count(child) == 0)
		map[child] = make_shared<OntologyBranch>(child);

	if (map.count(parent) == 0) {
		map[parent] = make_shared<OntologyBranch>(parent);
		parentless.emplace_back(map[parent]);
	}
	map[child]->add_parent(map[parent]);
}

std::vector<std::shared_ptr<OntologyBranch>> OntologyBranch::load(string fname) {
	std::map<std::string, std::shared_ptr<OntologyBranch>> branch_map;
	std::vector<std::shared_ptr<OntologyBranch>> roots;
	std::vector<std::shared_ptr<OntologyBranch>> parentless;

	ifstream fp;
	fp.open(fname.c_str(), std::ios::in);
	const string right_arrow = "->";
	const string left_arrow = "<-";
	string line;

	while (std::getline(fp, line)) {
		auto tokens = utils::split_str(line, right_arrow);
		if (tokens.size() >= 2) {
			for (int i = 0; i < tokens.size()-1; i++)
				add_lattice_edge(trim(tokens[i]), trim(tokens[i+1]), branch_map, parentless);
		} else {
			tokens = utils::split_str(line, left_arrow);
			if (tokens.size() >= 2)
				for (int i = 0; i < tokens.size()-1; i++)
					add_lattice_edge(trim(tokens[i+1]), trim(tokens[i]), branch_map, parentless);
		}
	}

	for (auto& k : parentless)
		if (k->parents.size() == 0)
			roots.emplace_back(k);

	return roots;
}

OntologyBranch::OntologyBranch(const string& _name) : name(_name) {}

void OntologyBranch::add_parent(OntologyBranch::shared_branch parent) {
	parents.emplace_back(parent);
	parent->add_child(shared_from_this());
}

void OntologyBranch::add_child(OntologyBranch::shared_branch child) {
	children.emplace_back(child);
}