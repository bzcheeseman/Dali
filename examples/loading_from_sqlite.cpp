#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "dali/utils.h"
#include "SQLiteCpp/Database.h"

DEFINE_string(index2target, "", "Location of Index2target file with mapping from integer to target name.");
DEFINE_string(dbpath, "", "Location of SQLite Database with protobuf elements");
DEFINE_string(redirections, "", "Set of redirections from article names to other articles.");
DEFINE_string(clean_index2target, "", "Clean version of index2target.");
DEFINE_int32(j, 1, "How many threads should be used ?");

static bool dummy = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_index2target,
                                                            &utils::validate_flag_nonempty);
static bool dummy2 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_dbpath,
                                                            &utils::validate_flag_nonempty);
static bool dummy3 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_redirections,
                                                            &utils::validate_flag_nonempty);
static bool dummy4 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_clean_index2target,
                                                            &utils::validate_flag_nonempty);

using std::string;
using std::vector;
using std::stringstream;
using utils::MS;
using utils::load_corpus_from_stream;
using utils::load_protobuff_dataset;
using std::cout;
using std::endl;

int main(int argc, char * argv[]) {
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    cout << "dbpath = " << FLAGS_dbpath << endl;
    // Open a database file
    SQLite::Database    db(FLAGS_dbpath);
    // Load mapping from concepts to names
    vector<string> index2concept;
    if (!utils::file_exists(FLAGS_clean_index2target)) {
        std::atomic<int> i(0);
        auto concept_redirections = utils::load_redirection_list(FLAGS_redirections, [&i](std::string&& s)->std::string {
            if (++i % 1000 == 0) {
                std::cout << i << " cleaned redirection names \r" << std::flush;
            }
            return utils::join(
                utils::xml_cleaner::split_punct_keep_brackets(s),
                " ");
        }, FLAGS_j);
        index2concept = utils::load_list(FLAGS_index2target);

        for (auto& concept : index2concept) {
            if (concept_redirections.find(concept) != concept_redirections.end()) {
                concept = utils::capitalize(concept_redirections.at(concept));
            } else {
                concept = utils::capitalize(concept);
            }
        }
        utils::save_list(index2concept, FLAGS_clean_index2target);
    } else {
        index2concept = utils::load_list(FLAGS_clean_index2target);
    }
    // Load some examples from DB
    SQLite::Statement   query(db, "SELECT lines FROM articles");
    // Convert protobuf -> vector<string>
    auto els = load_protobuff_dataset(query, index2concept, 100);
    cout << "got labeled examples" << endl;
    for (auto& el : els) {
        std::cout << utils::join(el[0], " ")
                  << " (\033[4m" << utils::join(el[1], "\x1B[m, \033[4m") << "\x1B[m)" << std::endl;
    }
}
