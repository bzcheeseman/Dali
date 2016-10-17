#ifndef DALI_CORE_UTILS_H
#define DALI_CORE_UTILS_H

#include <atomic>
#include <chrono>
#include <string>
#include <iostream>
#include <sstream>
#include <iterator>
#include <functional>
#include <unordered_map>
#include <vector>


// Useful for expanding macros. Obviously two levels of macro
// are needed....
#define STR(x) __THIS_IS_VERY_ABNOXIOUS(x)
#define __THIS_IS_VERY_ABNOXIOUS(tok) #tok

typedef std::vector<std::string> VS;

namespace utils {
    extern const mode_t DEFAULT_MODE;


    #ifndef NDEBUG
        std::string explain_mat_bug(const std::string&, const char*, const int&);
        template<typename T>
        bool contains_NaN(T);
    #endif

    /** Utility function to create directory tree */
    bool makedirs(const char* path, mode_t mode = DEFAULT_MODE);
    std::string expanduser(const std::string& path);
    typedef std::vector<std::string> str_sequence;

    /**
    type tokenized_labeled_dataset
    a vector of vectors of string sequences (e.g. loading a tsv with many rows,
    and each row has whitespace separated words for each column)
    **/
    typedef std::vector<std::vector<str_sequence>> tokenized_labeled_dataset;
    typedef std::vector<std::pair<str_sequence, int>> tokenized_uint_labeled_dataset;
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
    bool in_vector(const std::vector<T>&, const T&);

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


    std::vector<str_sequence> load_tokenized_unlabeled_corpus(const std::string&);
    str_sequence tokenize(const std::string&);
    str_sequence get_vocabulary(const tokenized_labeled_dataset&, int min_occurence, int data_column);
    str_sequence get_vocabulary(const std::vector<str_sequence>&, int);
    str_sequence get_vocabulary(const tokenized_uint_labeled_dataset&, int);
    str_sequence get_label_vocabulary(const tokenized_labeled_dataset&);




    std::string& trim(std::string&);
    std::string& ltrim(std::string&);
    std::string& rtrim(std::string&);

    void ensure_directory(std::string&);



    std::vector<std::string> split_str(const std::string&, const std::string&);

    void save_list(const std::vector<std::string>& list, std::string fname, std::ios_base::openmode = std::ios::out);

    template<typename T>
    void save_list_to_stream(const std::vector<std::string>& list, T&);

    template<typename T>
    void stream_to_redirection_list(T&, std::unordered_map<std::string, std::string>&);

    std::unordered_map<std::string, std::string> load_redirection_list(const std::string&);

    template<typename T>
    void stream_to_redirection_list(T&, std::unordered_map<std::string, std::string>&, std::function<std::string(std::string&&)>&, int num_threads = 1);

    std::unordered_map<std::string, std::string> load_redirection_list(const std::string&, std::function<std::string(std::string&&)>&&, int num_threads = 1);


    /**
    Is Gzip ?
    ---------
    Check whether a file has the header information for a GZIP
    file or not.

    Inputs
    ------
    fname : potential gzip file's path

    Outputs
    -------
    whether header matches gzip header
    **/
    bool is_gzip(const std::string& fname);


    template<typename T>
    T from_string(const std::string& s) {
        std::istringstream stream (s);
        T t;
        stream >> t;
        return t;
    }

    bool is_number(const std::string&);

    std::string find_and_replace(std::string subject,
                                 const std::string& search,
                                 const std::string& replace);


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



    template <typename T>
    std::vector<size_t> argsort(const std::vector<T> &v) {
        // initialize original index locations
        std::vector<size_t> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
        return idx;
    }

    str_sequence listdir(const std::string&);

    std::vector<uint> arange(uint, uint);

    template <class T> inline void hash_combine(std::size_t &, const T &);

    bool file_exists(const std::string&);

    std::string dir_parent(const std::string& path, int levels_up = 1);

    std::string dir_join(const std::vector<std::string>&);


    // returns candidate from candidates with longest common prefix with input.
    // if longest prefix is ambiguous, returns the first one from the list.
    std::string prefix_match(std::vector<std::string> candidates, std::string input);

    /**
    Exit With Message
    -----------------
    Exit the program with a message printed to std::cerr

    Inputs
    ------
    error message
    error_code : code for the error, defaults to 1

    **/
    void exit_with_message(const std::string&, int error_code = 1);

    bool validate_flag_nonempty(const char* flagname, const std::string& value);

    template<typename T>
    T vsum(const std::vector<T>& vec) {
        T res = 0;
        for(T item: vec) res += item;
        return res;
    }

    template<typename T>
    std::vector<T> reversed(const std::vector<T>& v) {
        std::vector<T> ret(v.rbegin(), v.rend());
        return ret;
    }

    class ThreadAverage {
        /* Small utility class used to safely average error contributions
           from different threads. */
        public:
            const int num_threads;
            std::vector<double> thread_error;
            std::atomic<int>    total_updates;

            ThreadAverage(int num_threads);

            // should be called from a thread (this internally uses thread pool's number.)
            void update(double error);
            double average();
            int size();
            void reset();

    };


    std::string capitalize(const std::string& s);

}

#endif
