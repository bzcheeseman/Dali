#ifndef RECURRENT_MAT_UTILS_H
#define RECURRENT_MAT_UTILS_H


#include <iomanip>
#include <random>
#include <sstream>
#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <sys/stat.h>
#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>
#include <memory>
#include <errno.h>
#include "OptionParser/OptionParser.h"
// Default writing mode useful for default argument to
// makedirs
#define DEFAULT_MODE S_IRWXU | S_IRGRP |  S_IXGRP | S_IROTH | S_IXOTH

std::ostream& operator<<(std::ostream&, const std::vector<std::string>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, uint>&);

template<typename T>
std::ostream& operator<<(std::ostream&, const std::vector<T>&);

namespace utils {
	/** Utility function to create directory tree */
	bool makedirs(const char* path, mode_t mode = DEFAULT_MODE);
	typedef std::vector<std::pair<std::vector<std::string>, std::string>> tokenized_labeled_dataset;

	extern const char* end_symbol;
	extern const char* unknown_word_symbol;

	class Vocab {
		typedef uint ind_t;
		private:
			void construct_word2index();
			void add_unknown_word();
		public:
			ind_t unknown_word;
			std::unordered_map<std::string, ind_t> word2index;
			std::vector<std::string> index2word;
			Vocab();
			Vocab(std::vector<std::string>&);
			Vocab(std::vector<std::string>&, bool);
	};

	template<typename T>
	void tuple_sum(std::tuple<T, T>&, std::tuple<T,T>);

	/**
	Ontology Branch
	---------------

	Small utility class for dealing with lattices with circular
	references. Useful for loading in ontologies and lattices.

	**/
	class OntologyBranch : public std::enable_shared_from_this<OntologyBranch> {
		int _max_depth;
		void compute_max_depth();
		public:
			typedef std::shared_ptr<OntologyBranch> shared_branch;
			typedef std::weak_ptr<OntologyBranch> shared_weak_branch;
			typedef std::shared_ptr<std::map<std::string, shared_branch>> lookup_t;

			std::vector<shared_weak_branch> parents;
			std::vector<shared_branch> children;
			lookup_t lookup_table;
			std::string name;
			int& max_depth();
			int id;
			int max_branching_factor() const;
			void save(std::string, std::ios_base::openmode = std::ios::out);
			static std::vector<shared_branch> load(std::string);
			static void add_lattice_edge(const std::string&, const std::string&,
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

	std::vector<std::pair<std::string, std::string>> load_labeled_corpus(const std::string&);
	tokenized_labeled_dataset load_tokenized_labeled_corpus(const std::string&);
	std::vector<std::string> tokenize(const std::string&);
	std::vector<std::string> get_vocabulary(const tokenized_labeled_dataset&, int);
	std::vector<std::string> get_lattice_vocabulary(const OntologyBranch::shared_branch);
	std::vector<std::string> get_label_vocabulary(const tokenized_labeled_dataset&);
	void assign_lattice_ids(OntologyBranch::lookup_t, Vocab&, int offset = 0);

	std::string& trim(std::string&);
	std::string& ltrim(std::string&);
	std::string& rtrim(std::string&);

	void map_to_file(const std::unordered_map<std::string, std::vector<std::string>>&, const std::string&);

	void ensure_directory(std::string&);

	std::vector<std::string> split_str(const std::string&, const std::string&);

	std::unordered_map<std::string, std::vector<std::string>> text_to_map(const std::string&);

	int randint(int, int);

	bool is_gzip(const std::string&);
	template<typename T>
	struct sigmoid_operator {
		T operator() (T) const;
	};
	template<typename T>
	struct tanh_operator {
		T operator() (T) const;
	};
	template<typename T>
	struct relu_operator {
		T operator() (T) const;
	};
	template<typename T>
	struct sign_operator {
		T operator() (T) const;
	};
	template<typename T>
	struct dtanh_operator {
		T operator() (T) const;
	};

	template<typename T>
    T from_string(const std::string&);

    template<typename T>
    void assert_map_has_key(std::unordered_map<std::string, T>&, const std::string&);

    std::vector<std::string> split(const std::string &, char);

	template <typename T>
	std::vector<size_t> argsort(const std::vector<T> &);

	template<typename T>
	void assign_cli_argument(char *, T&, T, std::string);

	template<typename T>
	void assign_cli_argument(char *, T&, std::string);

	void training_corpus_to_CLI(optparse::OptionParser&);

	template <class T> inline void hash_combine(std::size_t &, const T &);
	std::size_t get_random_id();
	namespace ops {
		static const uint add                             = 0;
		static const uint eltmul                          = 1;
		static const uint sigmoid                         = 2;
		static const uint tanh                            = 3;
		static const uint mul                             = 4;
		static const uint relu                            = 5;
		static const uint row_pluck                       = 6;
		static const uint add_broadcast                   = 7;
		static const uint eltmul_broadcast                = 8;
		static const uint mul_with_bias                   = 9;
		static const uint mul_add_mul_with_bias           = 10;
		static const uint mul_add_broadcast_mul_with_bias = 11;
		static const uint rows_pluck                      = 12;
		static const uint transpose                       = 13;
		static const uint eltmul_broadcast_rowwise        = 14;
		static const uint eltmul_rowwise                  = 15;
	}

	void exit_with_message(const std::string&, int error_code = 1);
}

// define hash code for OntologyBranch
namespace std {
	template <> struct hash<utils::OntologyBranch> {
		std::size_t operator()(const utils::OntologyBranch&) const;
	};
}
std::ostream& operator<<(std::ostream&, const utils::OntologyBranch&);
#endif