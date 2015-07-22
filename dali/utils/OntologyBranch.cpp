#include "dali/utils/OntologyBranch.h"
using std::string;
using std::vector;
using std::make_shared;
using std::ifstream;
using std::set;
using std::ofstream;

namespace utils {
    template<typename T>
    void OntologyBranch::save_to_stream(T& fp) {
        auto hasher = std::hash<OntologyBranch>();
        set<size_t> visited_list;
        vector<shared_branch> open_list;
        open_list.push_back(shared_from_this());
        while (open_list.size() > 0) {
            auto el = open_list[0];
            open_list.erase(open_list.begin());
            if (visited_list.count(hasher(*el)) == 0) {
                visited_list.insert(hasher(*el));
                if (el->children.size() > 1) {
                    auto child_ptr = el->children.begin();
                    open_list.emplace_back(*child_ptr);
                    fp << el->name << "->" << (*(child_ptr++))->name << "\n";
                    while (child_ptr != el->children.end()) {
                        open_list.emplace_back(*child_ptr);
                        fp << (*(child_ptr++))->name << "\n";
                    }
                } else {
                    for (auto& child : el->children) {
                        fp << el->name << "->" << child->name << "\n";
                        open_list.emplace_back(child);
                    }
                }
                for (auto& parent : el->parents)
                    open_list.emplace_back(parent.lock());
            }
        }
    }

    void OntologyBranch::save(string fname, std::ios_base::openmode mode) {
        if (endswith(fname, ".gz")) {
            ogzstream fpgz(fname.c_str(), mode);
            save_to_stream(fpgz);
        } else {
            ofstream fp(fname.c_str(), mode);
            save_to_stream(fp);
        }
    }
    void OntologyBranch::compute_max_depth() {
        _max_depth = 0;
        for (auto&v : children)
            if (v->max_depth() + 1 > _max_depth)
                 _max_depth = v->max_depth() + 1;
    }

    int& OntologyBranch::max_depth() {
        if (_max_depth > -1) return _max_depth;
        else {
                compute_max_depth();
                return _max_depth;
        }
    }

    std::pair<vector<OntologyBranch::shared_branch>, vector<uint>> OntologyBranch::random_path_to_root(const string& nodename) {
            auto node = lookup_table->at(nodename);
            auto up_node = node;
            uint direction;
            std::pair<vector<OntologyBranch::shared_branch>, vector<uint>> path_pair;
            // path_pair.first = vector<OntologyBranch::shared_branch>();
            while ( &(*up_node) != this) {
                    direction = randint(0, up_node->parents.size()-1);
                    path_pair.first.emplace_back(up_node);
                    path_pair.second.emplace_back(direction);
                    up_node = up_node->parents[direction].lock();
            }
            return path_pair;
    }

    std::pair<vector<OntologyBranch::shared_branch>, vector<uint>> OntologyBranch::random_path_to_root(const string& nodename, const int offset) {
            auto node = lookup_table->at(nodename);
            auto up_node = node;
            uint direction;
            std::pair<vector<OntologyBranch::shared_branch>, vector<uint>> path_pair;
            // path_pair.first = vector<OntologyBranch::shared_branch>();
            while ( &(*up_node) != this) {
                    direction = randint(0, up_node->parents.size()-1);
                    path_pair.first.emplace_back(up_node);
                    path_pair.second.emplace_back(direction + offset);
                    up_node = up_node->parents[direction].lock();
            }
            return path_pair;
    }

    std::pair<vector<OntologyBranch::shared_branch>, vector<uint>> OntologyBranch::random_path_from_root(const string& nodename) {
        return random_path_from_root(nodename, 0);
    }

    int OntologyBranch::get_index_of(OntologyBranch::shared_branch node) const {
        int i = 0;
        auto ptr = &(*node);
        for (auto& child : children)
                if (&(*child) == ptr) return i;
                else i++;
        return -1;
    }

    std::pair<vector<OntologyBranch::shared_branch>, vector<uint>> OntologyBranch::random_path_from_root(const string& nodename, const int offset) {
        auto up_node = lookup_table->at(nodename);
        auto parent = up_node;
        uint direction;
        std::pair<vector<OntologyBranch::shared_branch>, vector<uint>> path_pair;
        while ( &(*up_node) != this) {
            // find the parent:
            direction = randint(0, up_node->parents.size()-1);
            parent = up_node->parents[direction].lock();
            direction = parent->get_index_of(up_node);
            // assign an replace current with parent:
            path_pair.second.emplace(path_pair.second.begin(), direction + offset);
            path_pair.first.emplace(path_pair.first.begin(), up_node);
            up_node = parent;
        }
        return path_pair;
    }

    std::pair<OntologyBranch::shared_branch, OntologyBranch::shared_branch> OntologyBranch::add_lattice_edge(
            const std::string& parent,
            const std::string& child,
            lookup_t& map,
            std::vector<OntologyBranch::shared_branch>& parentless) {
            if (map->count(child) == 0)
                    (*map)[child] = make_shared<OntologyBranch>(child);

            if (map->count(parent) == 0) {
                    (*map)[parent] = make_shared<OntologyBranch>(parent);
                    parentless.emplace_back((*map)[parent]);
            }
            (*map)[child]->add_parent((*map)[parent]);
            return std::pair<OntologyBranch::shared_branch, OntologyBranch::shared_branch>((*map)[parent], (*map)[child]);
    }

    std::pair<OntologyBranch::shared_branch, OntologyBranch::shared_branch> OntologyBranch::add_lattice_edge(
            OntologyBranch::shared_branch parent,
            const std::string& child,
            lookup_t& map,
            std::vector<OntologyBranch::shared_branch>& parentless) {
            if (map->count(child) == 0)
                    (*map)[child] = make_shared<OntologyBranch>(child);
            (*map)[child]->add_parent(parent);

            return std::pair<OntologyBranch::shared_branch, OntologyBranch::shared_branch>(parent, (*map)[child]);
    }

    std::pair<OntologyBranch::shared_branch, OntologyBranch::shared_branch> OntologyBranch::add_lattice_edge(
            const std::string& parent,
            OntologyBranch::shared_branch child,
            lookup_t& map,
            std::vector<OntologyBranch::shared_branch>& parentless) {
            if (map->count(parent) == 0) {
                    (*map)[parent] = make_shared<OntologyBranch>(parent);
                    parentless.emplace_back((*map)[parent]);
            }
            child->add_parent((*map)[parent]);
            return std::pair<OntologyBranch::shared_branch, OntologyBranch::shared_branch>((*map)[parent], child);
    }

    template<typename T>
    void OntologyBranch::load_branches_from_stream(
                T& fp,
                std::vector<OntologyBranch::shared_branch>& roots) {

            std::vector<OntologyBranch::shared_branch> parentless;
            auto branch_map = make_shared<std::unordered_map<std::string, OntologyBranch::shared_branch>>();
            const string right_arrow = "->";
            const string left_arrow  = "<-";
            string line;

            OntologyBranch::shared_branch marked_branch = nullptr;
            bool last_edge_is_right_arrow = true;

            while (std::getline(fp, line)) {
                    auto tokens = utils::split_str(line, right_arrow);
                    if (tokens.size() >= 2) {
                        for (int i = 0; i < tokens.size()-1; i++) {
                            marked_branch = add_lattice_edge(trim(tokens[i]), trim(tokens[i+1]), branch_map, parentless).first;
                            last_edge_is_right_arrow = true;
                        }
                    } else {
                            tokens = utils::split_str(line, left_arrow);
                            if (tokens.size() >= 2)
                                for (int i = 0; i < tokens.size()-1; i++) {
                                        marked_branch = add_lattice_edge(trim(tokens[i+1]), trim(tokens[i]), branch_map, parentless).second;
                                        last_edge_is_right_arrow = false;
                                }
                            else if (marked_branch != nullptr) {
                                auto trimmed = trim(tokens[0]);
                                if (!trimmed.empty()) {
                                    if (last_edge_is_right_arrow)
                                            add_lattice_edge(marked_branch, trim(tokens[0]), branch_map, parentless);
                                    else
                                            add_lattice_edge(trim(tokens[0]), marked_branch, branch_map, parentless);
                                }
                            }
                    }
            }

            for (auto& k : parentless)
                    if (k->parents.size() == 0) {
                            roots.emplace_back(k);
                            k->lookup_table = branch_map;
                    }
    }

    std::vector<OntologyBranch::shared_branch> OntologyBranch::load(string fname) {
        std::vector<OntologyBranch::shared_branch> roots;

        if (utils::is_gzip(fname)) {
            igzstream fpgz(fname.c_str());
            load_branches_from_stream(fpgz, roots);
        } else {
            ifstream fp(fname);
            load_branches_from_stream(fp, roots);
        }

        return roots;
    }

    OntologyBranch::OntologyBranch(const string& _name) : name(_name), _max_depth(-1) {}

    void OntologyBranch::add_parent(OntologyBranch::shared_branch parent) {
            parents.emplace_back(parent);
            parent->add_child(shared_from_this());
    }

    void OntologyBranch::add_child(OntologyBranch::shared_branch child) {
            children.emplace_back(child);
    }

    int OntologyBranch::max_branching_factor() const {
            if (children.size() == 0) return 0;
            std::vector<int> child_maxes(children.size());
            auto child_factors = [](const int& a, const shared_branch b) { return b->max_branching_factor(); };
            std::transform (child_maxes.begin(), child_maxes.end(), children.begin(), child_maxes.begin(), child_factors);
            return std::max((int)children.size(), *std::max_element(child_maxes.begin(), child_maxes.end()));
    }

    std::vector<string> get_lattice_vocabulary(const OntologyBranch::shared_branch lattice) {
        std::vector<std::string> index2label;
        index2label.emplace_back(end_symbol);
        for (auto& kv : *lattice->lookup_table) {
            index2label.emplace_back(kv.first);
        }
        return index2label;
    }

    void assign_lattice_ids(OntologyBranch::lookup_t lookup_table, Vocab& lattice_vocab, int offset) {
        for(auto& kv : *lookup_table)
            kv.second->id = lattice_vocab.word2index.at(kv.first) + offset;
    }

    std::string OntologyBranch::to_string(int indent) const {
        std::stringstream ss;
        for (int j = 0; j < indent; j++) {
            ss << " ";
        }
        ss << "<OntologyBranch name=\"" << name << "\"";
        if (children.size() > 0) {
            ss << " children={";
            ss << "\n";
            int i = 0;
            for (int i = 0; i < children.size();i++) {
                for (int j = 0; j < indent; j++) {
                    ss << " ";
                }
                ss << children[i]->to_string(indent + 4);
                if (i != children.size() - 1)
                    ss << ",\n";
            }
            ss << "}";
        }
        ss << ">";
        return ss.str();
    }
}

std::ostream& operator<<(std::ostream& strm, const utils::OntologyBranch& a) {
    return strm << a.to_string();
}

std::ostream& operator<<(std::ostream& strm, const std::unordered_map<std::string, std::shared_ptr<utils::OntologyBranch>>& a) {
    int i = 0;
    strm << "{";
    for (auto& kv : a) {
        strm << kv.first << " => \"" << kv.second->name << "\" (" << kv.second->id << ")";
        if (i != a.size()) {
            strm << ",";
        }
        strm << "\n";
        i++;
    }
    strm << "}";
    return strm;
}

std::size_t std::hash<utils::OntologyBranch>::operator()(const utils::OntologyBranch& k) const {
    size_t seed = 0;
    std::hash<std::string> str_hasher;
    seed ^= str_hasher(k.name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}
