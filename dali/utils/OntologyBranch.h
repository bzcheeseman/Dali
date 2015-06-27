#ifndef ONTOLOGY_BRANCH_DALI_H
#define ONTOLOGY_BRANCH_DALI_H

#include <memory>
#include <string>
#include <vector>
#include "dali/utils/core_utils.h"
#include "dali/utils/random.h"
#include <set>
#include <fstream>
#include <ostream>

namespace utils {
    /**
    Ontology Branch
    ---------------

    Small utility class for dealing with lattices with circular
    references. Useful for loading in ontologies and lattices.

    **/
    class OntologyBranch : public std::enable_shared_from_this<OntologyBranch> {
    int _max_depth;
    /**
    OntologyBranch::compute_max_depth
    ---------------------------------

    Memoization computation for `OntologyBranch::max_depth`

    **/
    void compute_max_depth();
    public:
            typedef std::shared_ptr<OntologyBranch> shared_branch;
            typedef std::weak_ptr<OntologyBranch> shared_weak_branch;
            typedef std::shared_ptr<std::map<std::string, shared_branch>> lookup_t;

            std::vector<shared_weak_branch> parents;
            std::vector<shared_branch> children;
            lookup_t lookup_table;
            std::string name;
            /**
            OntologyBranch::max_depth
            -------------------------

            A memoized value for the deepest leaf's distance from the OntologyBranch
            that called the method. A leaf returns 0, a branch returns the max of its
            children's values + 1.

            Outputs
            -------
            int max_depth : Maximum number of nodes needed to traverse to reach a leaf.

            **/
            int& max_depth();
            int id;
            int max_branching_factor() const;
            /**
            OntologyBranch::save
            --------------------

            Serialize a lattice to a file by saving each edge on a separate line.

            Inputs
            ------

            std::string fname : file to saved the lattice to
            std::ios_base::openmode mode : what file open mode to use (defaults to write,
                                                                       but append can also be useful).
            **/
            void save(std::string, std::ios_base::openmode = std::ios::out);
            template<typename T>
            void save_to_stream(T&);
            static std::vector<shared_branch> load(std::string);
            template<typename T>
            static void load_branches_from_stream(T&, std::vector<shared_branch>&);
            static std::pair<shared_branch, shared_branch> add_lattice_edge(const std::string&, const std::string&,
                    std::shared_ptr<std::map<std::string, shared_branch>>&, std::vector<shared_branch>& parentless);
            static std::pair<shared_branch, shared_branch> add_lattice_edge(shared_branch, const std::string&,
                    std::shared_ptr<std::map<std::string, shared_branch>>&, std::vector<shared_branch>& parentless);
            static std::pair<shared_branch, shared_branch> add_lattice_edge(const std::string&, shared_branch,
                    std::shared_ptr<std::map<std::string, shared_branch>>&, std::vector<shared_branch>& parentless);
            OntologyBranch(const std::string&);
            void add_child(shared_branch);
            void add_parent(shared_branch);
            int get_index_of(shared_branch) const;
            std::pair<std::vector<std::shared_ptr<OntologyBranch>>, std::vector<uint>> random_path_to_root(const std::string&);
            std::pair<std::vector<std::shared_ptr<OntologyBranch>>, std::vector<uint>> random_path_to_root(const std::string&, const int);
            std::pair<std::vector<std::shared_ptr<OntologyBranch>>, std::vector<uint>> random_path_from_root(const std::string&);
            std::pair<std::vector<std::shared_ptr<OntologyBranch>>, std::vector<uint>> random_path_from_root(const std::string&, const int);
            static std::vector<std::string> split_str(const std::string&, const std::string&);
    };

    str_sequence get_lattice_vocabulary(const OntologyBranch::shared_branch);
    void assign_lattice_ids(OntologyBranch::lookup_t, Vocab&, int offset = 0);
}


// define hash code for OntologyBranch
namespace std {
    template <> struct hash<utils::OntologyBranch> {
        std::size_t operator()(const utils::OntologyBranch&) const;
    };
}
std::ostream& operator<<(std::ostream&, const utils::OntologyBranch&);

#endif
