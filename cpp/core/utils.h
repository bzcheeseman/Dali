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
#include <cstring>
#include <vector>
#include <map>
#include <set>
#include <sys/stat.h>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>
#include <memory>
#include "gzstream.h"
#include <sys/stat.h>
#include <errno.h>
#include <dirent.h>
#include <gflags/gflags.h>

#include "protobuf/corpus.pb.h"

DECLARE_int32(subsets);
DECLARE_int32(min_occurence);
DECLARE_int32(epochs);
DECLARE_int32(report_frequency);
DECLARE_string(dataset);

// Default writing mode useful for default argument to
// makedirs
#define DEFAULT_MODE S_IRWXU | S_IRGRP |  S_IXGRP | S_IROTH | S_IXOTH

std::ostream& operator<<(std::ostream&, const std::vector<std::string>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, uint>&);

template<typename T>
std::ostream& operator<<(std::ostream&, const std::vector<T>&);

namespace utils {
	/** Utility function to create directory tree */
	bool makedirs(const char* path, mode_t mode = DEFAULT_MODE);
	typedef std::vector<std::string> str_sequence;

	typedef std::vector<std::pair<str_sequence, std::string>> tokenized_labeled_dataset;
	typedef std::vector<std::pair<str_sequence, uint>> tokenized_uint_labeled_dataset;
	typedef std::vector<std::pair<str_sequence, str_sequence>> tokenized_multilabeled_dataset;

	extern const char* end_symbol;
	extern const char* unknown_word_symbol;

	class Vocab {
		typedef uint ind_t;
		private:
			void construct_word2index();
			void add_unknown_word();
		public:
			ind_t unknown_word;
			std::map<std::string, ind_t> word2index;
			str_sequence index2word;
			Vocab();
			Vocab(str_sequence&);
			Vocab(str_sequence&, bool);
	};

	template<typename T>
	void tuple_sum(std::tuple<T, T>&, std::tuple<T,T>);

	bool endswith(const std::string&, const std::string&);
	bool startswith(const std::string&, const std::string&);

	template<typename T>
	bool add_to_set(std::vector<T>&, T&);

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

	std::vector<std::pair<std::string, std::string>> load_labeled_corpus(const std::string&);
	tokenized_labeled_dataset load_tokenized_labeled_corpus(const std::string&);
	std::vector<str_sequence> load_tokenized_unlabeled_corpus(const std::string&);
	str_sequence tokenize(const std::string&);
	str_sequence get_vocabulary(const tokenized_labeled_dataset&, int);
	str_sequence get_vocabulary(const std::vector<str_sequence>&, int);
	str_sequence get_vocabulary(const tokenized_multilabeled_dataset&, int);
	str_sequence get_vocabulary(const tokenized_uint_labeled_dataset&, int);
	str_sequence get_lattice_vocabulary(const OntologyBranch::shared_branch);
	str_sequence get_label_vocabulary(const tokenized_labeled_dataset&);
	str_sequence get_label_vocabulary(const tokenized_multilabeled_dataset&);
	void assign_lattice_ids(OntologyBranch::lookup_t, Vocab&, int offset = 0);

	Corpus load_corpus_protobuff(const std::string&);
	tokenized_multilabeled_dataset load_protobuff_dataset(std::string, const std::vector<std::string>&);

	std::string& trim(std::string&);
	std::string& ltrim(std::string&);
	std::string& rtrim(std::string&);

	void map_to_file(const std::map<std::string, str_sequence>&, const std::string&);

	void ensure_directory(std::string&);

	std::vector<std::string> split_str(const std::string&, const std::string&);

	std::map<std::string, str_sequence> text_to_map(const std::string&);
	template<typename T, typename K>
	void stream_to_hashmap(T&, std::map<std::string, K>&);
	template<typename T>
	std::map<std::string, T> text_to_hashmap(const std::string&);

	template<typename T>
	void stream_to_list(T&, str_sequence&);

	str_sequence load_list(const std::string&);

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
    void assert_map_has_key(std::map<std::string, T>&, const std::string&);

    str_sequence split(const std::string &, char);

    str_sequence triggers_to_strings(const google::protobuf::RepeatedPtrField<Example::Trigger>&, const str_sequence&);

	template <typename T>
	std::vector<size_t> argsort(const std::vector<T> &);

	str_sequence listdir(const std::string&);

	template<typename T>
	void assign_cli_argument(char *, T&, T, std::string);

	template<typename T>
	void assign_cli_argument(char *, T&, std::string);

	std::vector<size_t> random_arange(size_t);

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

	bool file_exists(const std::string&);

	void exit_with_message(const std::string&, int error_code = 1);

	bool validate_flag_nonempty(const char* flagname, const std::string& value);
}

// define hash code for OntologyBranch
namespace std {
	template <> struct hash<utils::OntologyBranch> {
		std::size_t operator()(const utils::OntologyBranch&) const;
	};
}
std::ostream& operator<<(std::ostream&, const utils::OntologyBranch&);
#endif
