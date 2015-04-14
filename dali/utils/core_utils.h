#ifndef RECURRENT_MAT_UTILS_H
#define RECURRENT_MAT_UTILS_H

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <functional>
#include <gflags/gflags.h>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <ostream>
#include <random>
#include <set>
#include <sstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
#include "SQLiteCpp/Database.h"


#include "dali/utils/gzstream.h"
#include "protobuf/corpus.pb.h"
#include "dali/utils/ThreadPool.h"

// MACRO DEFINITIONS
#define ELOG(EXP) std::cout << #EXP "\t=\t" << (EXP) << std::endl
#define SELOG(STR,EXP) std::cout << #STR "\t=\t" << (EXP) << std::endl

// Useful for expanding macros. Obviously two levels of macro
// are needed....
#define STR(x) __THIS_IS_VERY_ABNOXIOUS(x)
#define __THIS_IS_VERY_ABNOXIOUS(tok) #tok

#ifdef NDEBUG
    #define DEBUG_ASSERT_POSITIVE(X) ;
    #define DEBUG_ASSERT_BOUNDS(X,a,b) ;
    #define DEBUG_ASSERT_NONZERO(X) ;
    #define DEBUG_ASSERT_NOT_NAN(X) ;
    #define DEBUG_ASSERT_MAT_NOT_NAN(X)
#else
    #define DEBUG_ASSERT_POSITIVE(X) assert(((X).array() >= 0).all())
    #define DEBUG_ASSERT_BOUNDS(X,a,b) assert(((X).array() >= (a)).all()  &&  ((X).array() <=(b)).all())
    #define DEBUG_ASSERT_NONZERO(X) assert(((X).array().abs() >= 1e-10).all())
    #define DEBUG_ASSERT_NOT_NAN(X) assert(!utils::contains_NaN(((X).array().abs().sum())))
    #define DEBUG_ASSERT_MAT_NOT_NAN(X) if ( utils::contains_NaN((X).w().array().square().sum()) ) { \
        (X).print(); \
        throw std::runtime_error(utils::explain_mat_bug((((X).name != nullptr) ? *(X).name : "?"), __FILE__,  __LINE__)); \
    }
#endif

// Default writing mode useful for default argument to
// makedirs
#define DEFAULT_MODE S_IRWXU | S_IRGRP |  S_IXGRP | S_IROTH | S_IXOTH

std::ostream& operator<<(std::ostream&, const std::vector<std::string>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, uint>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, float>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, double>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, std::string>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, uint>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, float>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, double>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, std::string>&);

template<typename T>
std::ostream& operator<<(std::ostream&, const std::vector<T>&);

typedef std::vector<std::string> VS;


namespace utils {

        #ifndef NDEBUG
            std::string explain_mat_bug(const std::string&, const char*, const int&);
            template<typename T>
            bool contains_NaN(T);
        #endif

        /** Utility function to create directory tree */
        bool makedirs(const char* path, mode_t mode = DEFAULT_MODE);
        typedef std::vector<std::string> str_sequence;

        typedef std::vector<std::pair<str_sequence, std::string>> tokenized_labeled_dataset;
        typedef std::vector<std::pair<str_sequence, uint>> tokenized_uint_labeled_dataset;
        typedef std::vector<std::pair<str_sequence, str_sequence>> tokenized_multilabeled_dataset;

        extern const char* end_symbol;
        extern const char* unknown_word_symbol;

        class Vocab {
                private:
                        void construct_word2index();
                        void add_unknown_word();
                public:
                    typedef uint ind_t;
                    ind_t unknown_word;
                    std::map<std::string, ind_t> word2index;
                    str_sequence index2word;

                    std::vector<ind_t> transform(const str_sequence& words, bool with_end_symbol = false) const;

                    // create vocabulary from many vectors of words. Vectors do not
                    // need to contain unique words. They need not be sorted.
                    // Standard use case is combining train/validate/test sets.
                    static Vocab from_many_nonunique(std::initializer_list<str_sequence>,
                                                     bool add_unknown_word=true);
                    Vocab();
                    Vocab(str_sequence&);
                    Vocab(str_sequence&, bool);
        };

        template<typename T>
        void tuple_sum(std::tuple<T, T>&, std::tuple<T,T>);
        /**
        Ends With
        ---------

        Check whether a string ends with the same contents as another.

        Inputs
        ------

        std::string const& full: where to look
        std::string const& ending: what to look for

        Outputs
        -------

        bool endswith : whether the second string ends the first.
        */
        bool endswith(const std::string&, const std::string&);
        /**
        Starts With
        -----------

        Check whether a string ends with the same contents as another.

        Inputs
        ------

        std::string const& full: where to look
        std::string const& beginning: what to look for

        Outputs
        -------

        bool startswith : whether the second string starts the first.

        */
        bool startswith(const std::string&, const std::string&);

        template<typename T>
        bool add_to_set(std::vector<T>&, T&);

        template<typename T>
        bool in_vector(const std::vector<T>&, T&);

        template<typename IN, typename OUT>
        std::vector<OUT> fmap(std::vector<IN> in_list,
                              std::function<OUT(IN)> f);

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
        /**
        Load Labeled Corpus
        -------------------

        Read text file line by line and extract pairs of labeled text
        by splitting on the first space character. Whatever is before
        the first space is the label and is stored second in the returned
        pairs, and after the space is the example, and this is returned
        first in the pairs.

        Inputs
        ------

        const std::string& fname : labeled corpus file

        Outputs
        -------

        std::vector<std::pair<std::string, std::string>> pairs : labeled corpus pairs

        **/
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

        template<typename T>
        void load_corpus_from_stream(Corpus& corpus, T& stream);
        Corpus load_corpus_protobuff(const std::string&);
        /**
        Load Protobuff Dataset
        ----------------------

        Load a set of protocol buffer serialized files from ordinary
        or gzipped files, and conver their labels from an index
        to their string representation using an index2label mapping.

        Inputs
        ------

        std::string directory : where the protocol buffer files are stored
        const std::vector<std::string>& index2label : mapping from numericals to
                                                      string labels

        Outputs
        -------

        utils::tokenized_multilabeled_dataset dataset : pairs of tokenized strings
                                                        and vector of string labels

        **/
        tokenized_multilabeled_dataset load_protobuff_dataset(std::string, const std::vector<std::string>&);
        tokenized_multilabeled_dataset load_protobuff_dataset(SQLite::Statement& query, const std::vector<std::string>&, int max_elements = 100, int column = 0);

        std::string& trim(std::string&);
        std::string& ltrim(std::string&);
        std::string& rtrim(std::string&);

        void map_to_file(const std::map<std::string, str_sequence>&, const std::string&);

        void ensure_directory(std::string&);

        std::vector<std::string> split_str(const std::string&, const std::string&);

        /**
        Text To Map
        -----------

        Read a text file, extract all key value pairs and
        ignore markdown decoration characters such as =, -,
        and #

        Inputs
        ------

        std::string fname : the file to read

        Outputs
        -------

        std::map<string, std::vector<string> > map : the extracted key value pairs.

        **/
        std::map<std::string, str_sequence> text_to_map(const std::string&);
        template<typename T, typename K>
        void stream_to_hashmap(T&, std::map<std::string, K>&);
        template<typename T>
        std::map<std::string, T> text_to_hashmap(const std::string&);

        template<typename T>
        void stream_to_list(T&, str_sequence&);

        str_sequence load_list(const std::string&);
        void save_list(const std::vector<std::string>& list, std::string fname, std::ios_base::openmode = std::ios::out);

        template<typename T>
        void save_list_to_stream(const std::vector<std::string>& list, T&);

        template<typename T>
        void stream_to_redirection_list(T&, std::map<std::string, std::string>&);

        std::map<std::string, std::string> load_redirection_list(const std::string&);

        template<typename T>
        void stream_to_redirection_list(T&, std::map<std::string, std::string>&, std::function<std::string(std::string&&)>&, int num_threads = 1);

        std::map<std::string, std::string> load_redirection_list(const std::string&, std::function<std::string(std::string&&)>&&, int num_threads = 1);

        /**
        randint
        -------

        Sample integers from a uniform distribution between (and including)
        lower and upper int values.

        Inputs
        ------
        int lower
        int upper

        Outputs
        -------
        int sample
        **/
        int randint(int, int);
        /**
        Is Gzip ?
        ---------

        Check whether a file has the header information for a GZIP
        file or not.

        Inputs
        ------
        std::string& fname : potential gzip file's path

        Outputs
        -------

        bool gzip? : whether header matches gzip header
        **/
        bool is_gzip(const std::string&);

        /* Used for Eigen CwiseExpr - wraps a lambda expression */
        template<typename T>
        struct LambdaOperator {
            std::function<T(T)> lambda_expr;
            LambdaOperator(std::function<T(T)> lambda_expr);
            T operator() (T) const;
        };


        template<typename T>
        struct sigmoid_operator {
                T operator() (T) const;
        };
        template<typename T>
        struct steep_sigmoid_operator {
                // Sourced from Indico's Passage library's activations
                // (Theano Python module)
                // https://github.com/IndicoDataSolutions/Passage
                // motivation in this Youtube video:
                // https://www.youtube.com/watch?v=VINCQghQRuM
                const T aggressiveness;
                steep_sigmoid_operator(T aggressiveness);
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

    bool is_number(const std::string&);

    template<typename T>
    void assert_map_has_key(std::map<std::string, T>&, const std::string&);

    /**
    Split
    -----

    Split a string at a character into a vector of strings.
    Optionally choose to keep empty strings, e.g.:

        > utils::split("//hello//world", '/');

        => vector<string>({"hello", "world"})

    vs.

        > utils::split("//hello//world", '/', true);

        => vector<string>({"", "", "hello", "", "world", "", ""})

    Inputs
    ------
      std::string& text : text to split using char
                 char c : character used to cut up text
bool keep_empty_strings : keep empty strings [see above], defaults to false.

    Outputs
    -------

    std::vector<std::string> split : split string sequence (tokenized)

    **/
    str_sequence split(const std::string &, char, bool keep_empty_strings = false);


    std::string join(const std::vector<std::string>& vs,
                     const std::string& in_between="");

    bool vs_equal(const VS& a, const VS& b);
    /**
    Triggers To Strings
    -------------------

    Convert triggers from an example to their
    string representation using an index2label
    string vector.

    Inputs
    ------

    const google::protobuf::RepeatedPtrField<Example::Trigger>& triggers : list of Trigger protobuff objects
    const vector<string>& index2target : mapping from numerical to string representation

    Outputs
    -------

    std::vector<std::string> data : the strings corresponding to the trigger targets
    **/
    str_sequence triggers_to_strings(const google::protobuf::RepeatedPtrField<Example::Trigger>&, const str_sequence&);

    template <typename T>
    std::vector<size_t> argsort(const std::vector<T> &);

    str_sequence listdir(const std::string&);

    std::vector<size_t> random_arange(size_t);

    std::vector<std::vector<size_t>> random_minibatches(size_t total_elements, size_t minibatch_size);


    std::vector<uint> arange(uint, uint);

    template <class T> inline void hash_combine(std::size_t &, const T &);

    bool file_exists(const std::string&);

    std::string dir_parent(const std::string& path, int levels_up = 1);

    std::string dir_join(const std::vector<std::string>&);

    template<typename T>
    std::vector<T> normalize_weights(const std::vector<T>& weights);

    /**
    Exit With Message
    -----------------

    Exit the program with a message printed to std::cerr

    Inputs
    ------

    std::string& message : error message
    int error_code : code for the error, defaults to 1

    **/
    void exit_with_message(const std::string&, int error_code = 1);

    bool validate_flag_nonempty(const char* flagname, const std::string& value);

    template<typename T>
    T vsum(const std::vector<T>& vec);

    template<typename T>
    std::vector<T> reversed(const std::vector<T>& v);

    class ThreadError {
        /* Small utility class used to safely average error contributions
           from different threads. */
        public:
            const int num_threads;
            std::vector<double> thread_error;
            std::vector<int>    thread_error_updates;

            ThreadError(int num_threads);

            // should be called from a thread (this internally uses thread pool's number.)
            void update(double error);

            double this_thread_average();

            double average();

            void reset();

    };


    class Timer {
        typedef std::chrono::system_clock clock_t;

        static std::unordered_map<std::string, std::atomic<int>> timers;
        static std::mutex timers_mutex;

        std::string name;
        bool stopped;
        bool started;
        std::chrono::time_point<clock_t> start_time;

        public:
            // creates timer and starts measuring time.
            Timer(std::string name, bool autostart=true);
            // destroys timer and stops counting if the timer was not previously stopped.
            ~Timer();

            // explicitly start the timer
            void start();
            // explicitly stop the timer
            void stop();

            static void report();
    };

    class MS {
        public:
            std::stringstream stream;
            operator std::string() const { return stream.str(); }

            template<class T>
            MS& operator<<(T const& VAR) { stream << VAR; return *this; }
    };

    template<typename T> class Generator;

    template<typename T>
    class Iter {
        Generator<T>* gen;
        T next;
        bool is_done = true;

        void advance() {
            if (gen->done()) {
                is_done = true;
            } else {
                next = std::move(gen->next());

            }
        }

        public:
            Iter (Generator<T>* gen, bool is_end) : gen(gen) {
                if (!is_end) {
                    advance();
                    is_done = false;
                } else {
                    is_done = true;
                }
            }

            // this function only compares regular iterators with end iterators...
            bool operator!= (const Iter& other) const {
                // assume we comparing regular iterator to end iterator.
                return is_done != other.is_done;
            }

            // this method must be defined after the definition of IntVector
            // since it needs to use it
            T& operator* () {
                return next;
            }

            const Iter& operator++ () {
                assert(!is_done);
                advance();
            }
    };

    class ConfusionMatrix {
        public:
            std::vector<std::vector<std::atomic<int>>> grid;
            std::vector<std::atomic<int>> totals;
            const std::vector<std::string>& names;
            ConfusionMatrix(int classes, const std::vector<std::string>& _names);
            void classified_a_when_b(int a, int b);
            void report() const;
    };

    std::string capitalize(const std::string& s);

    template<typename T>
    class Generator {
        public:
            virtual T next() = 0;
            virtual bool done() = 0;

            Iter<T> begin() {
                return Iter<T>(this, false);
            }

            Iter<T> end() {
                return Iter<T>(this, true);
            }
    };
    void assert2(bool condition);
    void assert2(bool condition, std::string message);
}

// define hash code for OntologyBranch
namespace std {
    template <> struct hash<utils::OntologyBranch> {
        std::size_t operator()(const utils::OntologyBranch&) const;
    };
}
std::ostream& operator<<(std::ostream&, const utils::OntologyBranch&);
#endif
