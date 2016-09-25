#include "core_utils.h"

#include <algorithm>
#include <dirent.h>
#include <iomanip>
#include <iterator>
#include <set>
#include <sys/stat.h>
#include <pwd.h>

#include "dali/utils/assert2.h"
#include "dali/utils/ThreadPool.h"
#include "dali/utils/vocab.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/gzstream.h"

using std::vector;
using std::string;
using std::ifstream;
using std::stringstream;
using std::ofstream;
using std::set;
using std::make_shared;
using std::function;
using std::initializer_list;


namespace utils {
    const mode_t DEFAULT_MODE = S_IRWXU | S_IRGRP |  S_IXGRP | S_IROTH | S_IXOTH;

    #ifndef NDEBUG
    string explain_mat_bug(const string& mat_name, const char* file, const int& line) {
        stringstream ss;
        ss << "Matrix \"" << mat_name << "\" has NaNs in file:\"" << file << "\" and line: " << line;
        return ss.str();
    }
    template<typename T>
    bool contains_NaN(T val) {return !(val == val);}

    template bool contains_NaN(float);
    template bool contains_NaN(double);
    #endif

    vector<int> arange(int start, int end) {
        vector<int> indices(end - start);
        for (int i=0; i < indices.size();i++) indices[i] = i;
        return indices;
    }

    void ensure_directory (std::string& dirname) {
        if (dirname.back() != '/') dirname += "/";
    }

    vector<string> split(const std::string &s, char delim, bool keep_empty_strings) {
        std::vector<std::string> elems;
        std::stringstream ss(s);
        string item;
        while (std::getline(ss, item, delim))
            if (!item.empty() || keep_empty_strings)
                elems.push_back(item);
        return elems;
    }

    string join(const vector<string>& vs, const string& in_between) {
        std::stringstream ss;
        for (int i = 0; i < vs.size(); ++i) {
            ss << vs[i];
            if (i + 1 != vs.size())
                ss << in_between;
        }
        return ss.str();
    }


    template<typename T>
    bool add_to_set(vector<T>& set, T& el) {
        if(std::find(set.begin(), set.end(), el) != set.end()) {
                return false;
        } else {
                set.emplace_back(el);
                return true;
        }
    }

    template bool add_to_set(vector<int>&,    int&);
    template bool add_to_set(vector<uint>&,   uint&);
    template bool add_to_set(vector<string>&, string&);


    template<typename T>
    bool in_vector(const std::vector<T>& set, const T& el) {
        return std::find(set.begin(), set.end(), el) != set.end();
    }

    template bool in_vector(const vector<int>&,    const int&);
    template bool in_vector(const vector<uint>&,   const uint&);
    template bool in_vector(const vector<string>&, const string&);
    template bool in_vector(const vector<char>&, const char&);

    vector<string> listdir(const string& folder) {
            vector<string> filenames;
            DIR *dp;
        struct dirent *dirp;
        if ((dp  = opendir(folder.c_str())) == nullptr) {
            stringstream error_msg;
                    error_msg << "Error: could not open directory \"" << folder << "\"";
                    throw std::runtime_error(error_msg.str());
        }
        // list all contents of directory:
        while ((dirp = readdir(dp)) != nullptr)
            // exclude "current directory" (.) and "parent directory" (..)
            if (std::strcmp(dirp->d_name, ".") != 0 && std::strcmp(dirp->d_name, "..") != 0)
                    filenames.emplace_back(dirp->d_name);
        closedir(dp);
        return filenames;
    }

    vector<string> split_str(const string& original, const string& delimiter) {
            std::vector<std::string> tokens;
            auto delimiter_ptr = delimiter.begin();
            stringstream ss(original);
            int inside = 0;
            std::vector<char> token;
            char ch;
            while (ss) {
                ch = ss.get();
                if (ch == *delimiter_ptr) {
                    delimiter_ptr++;
                    inside++;
                    if (delimiter_ptr == delimiter.end()) {
                        tokens.emplace_back(token.begin(), token.end());
                        token.clear();
                        inside = 0;
                        delimiter_ptr = delimiter.begin();
                    }
                } else {
                    if (inside > 0) {
                        token.insert(token.end(), delimiter.begin(), delimiter_ptr);
                        delimiter_ptr = delimiter.begin();
                        inside = 0;
                    }
                    token.push_back(ch);
                }
            }
            if (inside > 0) {
                    token.insert(token.end(), delimiter.begin(), delimiter_ptr);
                    tokens.emplace_back(token.begin(), token.end());
                    token.clear();
            } else {
                    if (token.size() > 0)
                            tokens.emplace_back(token.begin(), token.end()-1);
            }
            return tokens;
    }

    template<typename T>
    void save_list_to_stream(const vector<string>& list, T& fp) {
        for (auto& el : list) {
            fp << el << "\n";
        }
    }

    void save_list(const vector<string>& list, string fname, std::ios_base::openmode mode) {
        if (endswith(fname, ".gz")) {
            ogzstream fpgz(fname.c_str(), mode);
            save_list_to_stream(list, fpgz);
        } else {
            ofstream fp(fname, mode);
            save_list_to_stream(list, fp);
        }
    }

    template<typename T>
    void stream_to_redirection_list(T& fp, std::unordered_map<string, string>& mapping, std::function<std::string(std::string&&)>& preprocessor, int num_threads) {
        string line;
        const char dash = '-';
        const char arrow = '>';
        bool saw_dash = false;
        auto checker = [&saw_dash, &arrow, &dash](const char& ch) {
            if (saw_dash) {
                if (ch == arrow) {
                    return true;
                } else {
                    saw_dash = (ch == dash);
                    return false;
                }
            } else {
                saw_dash = (ch == dash);
                return false;
            }
        };
        if (num_threads > 1) {
            ThreadPool pool(num_threads);
            while (std::getline(fp, line)) {
                pool.run([&mapping, &preprocessor, &checker, line]() {
                    auto pos_end_arrow = std::find_if(line.begin(), line.end(), checker);
                    if (pos_end_arrow != line.end()) {
                        mapping.emplace(
                            std::piecewise_construct,
                            std::forward_as_tuple(
                                preprocessor(
                                    std::string(
                                        line.begin(),
                                        pos_end_arrow-1
                                    )
                                )
                            ),
                            std::forward_as_tuple(
                                preprocessor(
                                    std::string(
                                        pos_end_arrow+1,
                                        line.end()
                                    )
                                )
                            )
                        );
                    }
                });
            }
        } else {
            while (std::getline(fp, line)) {
                auto pos_end_arrow = std::find_if(line.begin(), line.end(), checker);
                if (pos_end_arrow != line.end()) {
                    mapping.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple( preprocessor(std::string(line.begin(), pos_end_arrow-1))),
                        std::forward_as_tuple( preprocessor(std::string(pos_end_arrow+1, line.end())))
                    );
                }
            }
        }
    }

    template<typename T>
    void stream_to_redirection_list(T& fp, std::unordered_map<string, string>& mapping) {
        string line;
        const char dash = '-';
        const char arrow = '>';
        bool saw_dash = false;
        auto checker = [&saw_dash, &arrow, &dash](const char& ch) {
            if (saw_dash) {
                if (ch == arrow) {
                    return true;
                } else {
                    saw_dash = (ch == dash);
                    return false;
                }
            } else {
                saw_dash = (ch == dash);
                return false;
            }
        };
        while (std::getline(fp, line)) {
            auto pos_end_arrow = std::find_if(line.begin(), line.end(), checker);
            if (pos_end_arrow != line.end()) {
                mapping.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple( line.begin(), pos_end_arrow-1),
                    std::forward_as_tuple( pos_end_arrow+1, line.end())
                );
            }
        }
    }

    template void stream_to_redirection_list(stringstream&, std::unordered_map<string, string>&, std::function<std::string(std::string&&)>&, int);
    template void stream_to_redirection_list(stringstream&, std::unordered_map<string, string>&);

    std::unordered_map<string, string> load_redirection_list(const string& fname, std::function<std::string(std::string&&)>&& preprocessor, int num_threads) {
        std::unordered_map<string, string> mapping;
        if (is_gzip(fname)) {
            igzstream fpgz(fname.c_str(), std::ios::in | std::ios::binary);
            stream_to_redirection_list(fpgz, mapping, preprocessor, num_threads);
        } else {
            std::fstream fp(fname, std::ios::in | std::ios::binary);
            stream_to_redirection_list(fp, mapping, preprocessor, num_threads);
        }
        return mapping;
    }

    std::unordered_map<std::string, std::string> load_redirection_list(
            const std::string& fname) {
        std::unordered_map<std::string, std::string> mapping;
        if (is_gzip(fname)) {
            igzstream fpgz(fname.c_str(), std::ios::in | std::ios::binary);
            stream_to_redirection_list(fpgz, mapping);
        } else {
            std::fstream fp(fname, std::ios::in | std::ios::binary);
            stream_to_redirection_list(fp, mapping);
        }
        return mapping;
    }

    std::vector<std::string> tokenize(const std::string& s) {
        stringstream ss(s);
        std::istream_iterator<string> begin(ss);
        std::istream_iterator<string> end;
        return std::vector<std::string>(begin, end);
    }

    std::vector<std::vector<std::string>> load_tokenized_unlabeled_corpus(
            const std::string& fname) {
        ifstream fp(fname.c_str());
        std::string l;
        std::vector<std::vector<std::string>> list;
        while (std::getline(fp, l)) {
            auto tokenized = tokenize(std::string(l.begin(), l.end()));
            if (tokenized.size() > 0 ) {
                list.emplace_back(tokenized);
            }
        }
        return list;
    }

    std::vector<std::string> get_vocabulary(
            const tokenized_labeled_dataset& examples,
            int min_occurence,
            int data_column) {
        std::unordered_map<std::string, uint> word_occurences;
        string word;
        for (auto& example : examples)
            for (auto& word : example[data_column]) word_occurences[word] += 1;
        std::vector<std::string> list;
        for (auto& key_val : word_occurences)
            if (key_val.second >= min_occurence)
                list.emplace_back(key_val.first);
        list.emplace_back(utils::end_symbol);
        return list;
    }

    std::vector<std::string> get_vocabulary(
            const std::vector<std::vector<std::string>>& examples,
            int min_occurence) {
        std::unordered_map<std::string, uint> word_occurences;
        std::string word;
        for (auto& example : examples) {
            for (auto& word : example) {
                word_occurences[word] += 1;
            }
        }
        std::vector<std::string> list;
        for (auto& key_val : word_occurences) {
            if (key_val.second >= min_occurence) {
                list.emplace_back(key_val.first);
            }
        }
        list.emplace_back(utils::end_symbol);
        return list;
    }

    std::vector<std::string> get_vocabulary(
            const tokenized_uint_labeled_dataset& examples,
            int min_occurence) {
        std::unordered_map<std::string, uint> word_occurences;
        std::string word;
        for (auto& example : examples) {
            for (auto& word : example.first) {
                word_occurences[word] += 1;
            }
        }
        std::vector<std::string> list;
        for (auto& key_val : word_occurences) {
            if (key_val.second >= min_occurence) {
                list.emplace_back(key_val.first);
            }
        }
        list.emplace_back(utils::end_symbol);
        return list;
    }

    std::vector<std::string> get_label_vocabulary(const tokenized_labeled_dataset& examples) {
        std::set<std::string> labels;
        std::string word;
        for (auto& example : examples) {
            ASSERT2(example.size() > 1, "Examples must have at least 2 columns.");
            labels.insert(example[1].begin(), example[1].end());
        }
        return std::vector<std::string>(labels.begin(), labels.end());
    }

    // Trimming text from StackOverflow:
    // http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring

    // trim from start
    std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
    }

    // trim from end
    std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
    }

    // trim from both ends
    std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
    }

    // From this StackOverflow:
    // http://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux
    bool makedirs(const char* path, mode_t mode) {
        // const cast for hack
        char* p = const_cast<char*>(path);

        // Do mkdir for each slash until end of string or error
        while (*p != '\0') {
            // Skip first character
            p++;

            // Find first slash or end
            while(*p != '\0' && *p != '/') p++;

            // Remember value from p
            char v = *p;

            // Write end of string at p
            *p = '\0';

            // Create folder from path to '\0' inserted at p
            if(mkdir(path, mode) == -1 && errno != EEXIST) {
                *p = v;
                return false;
            }

            // Restore path to it's former glory
            *p = v;
        }
        return true;
    }

    // From this StackOverflow:
    // http://stackoverflow.com/questions/3020187/getting-home-directory-in-mac-os-x-using-c-language
    std::string expanduser(const std::string& path) {
        if (path.size() == 0 || path[0] != '~') {
            return path;
        }
        // on windows a different environment variable
        // controls the home directory, but mac and linux
        // use HOME
        const char *homeDir = getenv("HOME");
        bool got_home_dir = false;
        if (!homeDir) {
            struct passwd* pwd = getpwuid(getuid());
            if (pwd) {
               homeDir = pwd->pw_dir;
               got_home_dir = true;
            }
        } else {
            got_home_dir = true;
        }
        if (got_home_dir) {
            return std::string(homeDir) + path.substr(1);
        } else {
            // path could not be expanded
            return path;
        }
    }

    bool is_number(const std::string& s) {
        bool is_negative = *s.begin() == '-';
        bool is_decimal  = false;
        if (is_negative && s.size() == 1) {
            return false;
        }
        return !s.empty() && std::find_if(s.begin() + (is_negative ? 1 : 0),
            s.end(), [&is_decimal](char c) {
                if (is_decimal) {
                    if (c == '.') {
                        return true;
                    } else {
                        return !std::isdigit(c);
                    }
                } else {
                    if (c == '.') {
                        is_decimal = true;
                        return false;
                    } else {
                        return !std::isdigit(c);
                    }
                }
            }) == s.end();
    }

    bool is_gzip(const std::string& fname) {
        const unsigned char gzip_code = 0x1f;
        const unsigned char gzip_code2 = 0x8b;
        unsigned char ch;
        std::ifstream file;
        file.open(fname);
        if (!file) return false;
        file.read(reinterpret_cast<char*>(&ch), 1);
        if (ch != gzip_code)
                return false;
        if (!file) return false;
        file.read(reinterpret_cast<char*>(&ch), 1);
        if (ch != gzip_code2)
                return false;
        return true;
    }

    void exit_with_message(const std::string& message, int error_code) {
            std::cerr << message << std::endl;
            exit(error_code);
    }

    bool endswith(std::string const & full, std::string const & ending) {
        if (full.length() >= ending.length()) {
            return (0 == full.compare(full.length() - ending.length(), ending.length(), ending));
        } else {
            return false;
        }
    }

    bool startswith(std::string const & full, std::string const & beginning) {
        if (full.length() >= beginning.length()) {
            return (0 == full.compare(0, beginning.length(), beginning));
        } else {
            return false;
        }
    }

    bool file_exists(const std::string& fname) {
        struct stat buffer;
        return (stat (fname.c_str(), &buffer) == 0);
    }

    std::string dir_parent(const std::string& path, int levels_up) {
        auto file_path_split = split(path, '/');
        assert(levels_up < file_path_split.size());
        stringstream ss;
        if (path[0] == '/')
            ss << '/';
        for (int i = 0; i < file_path_split.size() - levels_up; ++i) {
            ss << file_path_split[i];
            if (i + 1 != file_path_split.size() - levels_up)
                ss << '/';
        }
        return ss.str();
    }

    std::string dir_join(const std::vector<std::string>& paths) {
        stringstream ss;
        for (int i = 0; i < paths.size(); ++i) {
            ss << paths[i];
            bool found_slash_symbol = false;
            if (paths[i].size() == 0 || paths[i][paths[i].size() - 1] == '/') found_slash_symbol = true;
            if (!found_slash_symbol && i + 1 != paths.size())
                ss << '/';
        }
        return ss.str();
    }

    std::string prefix_match(std::vector<string> candidates, std::string input) {
        ASSERT2(!candidates.empty(), "Empty set of candidates for prefix matching.");
        int best_match_idx = -1;
        for (auto& candidate: candidates) {
            if (candidate.size() < input.size())
                continue;
            if (startswith(candidate, input))
                return candidate;
        }
        ASSERT2(false, MS() << "Could not find match for " << input << " in " << candidates <<".");
        return "";
    }

    bool validate_flag_nonempty(const char* flagname, const std::string& value) {
        if (value.empty()) {
            std::cout << "Invalid value for --" << flagname << " (can't be empty)" << std::endl;
        }
        return not value.empty();
    }

    std::string capitalize(const std::string& s) {
        std::string capitalized = s;
        // capitalize
        if (capitalized[0] >= 'a' && capitalized[0] <= 'z') {
            capitalized[0] += ('A' - 'a');
        }
        return capitalized;
    }

    ThreadAverage::ThreadAverage(int num_threads) :
            num_threads(num_threads),
            thread_error(num_threads),
            total_updates(0) {
        reset();
    }

    void ThreadAverage::update(double error) {
        thread_error[ThreadPool::get_thread_number()] += error;
        total_updates += 1;
    }

    double ThreadAverage::average() {
        return vsum(thread_error) / total_updates;
    }

    int ThreadAverage::size() {
        return total_updates;
    }

    void ThreadAverage::reset() {
        for (int tidx = 0; tidx < num_threads; ++tidx) {
            thread_error[tidx] = 0;
        }
        total_updates.store(0);
    }
}
