#include "sqlite3.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include "dali/utils.h"
#include "SQLiteCpp/Database.h"

DEFINE_string(index2target, "", "Location of Index2Target file with mapping from integer to target name.");
DEFINE_string(dbpath, "", "Location of SQLite Database with protobuf elements");
DEFINE_string(redirections, "", "Set of redirections from article names to other articles.");

static bool dummy = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_index2target,
                                                            &utils::validate_flag_nonempty);
static bool dummy2 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_dbpath,
                                                            &utils::validate_flag_nonempty);
static bool dummy3 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_redirections,
                                                            &utils::validate_flag_nonempty);

using std::string;
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
    auto concept_redirections = utils::load_redirection_list(FLAGS_redirections);
    std::cout << "loaded redirections" << std::endl;
    auto index2concept        = utils::load_list(FLAGS_index2target);
    std::cout << "loaded index2concept" << std::endl;
    int corrected = 0;
    int total_concepts = index2concept.size();
    for (auto& concept : index2concept) {
        if (concept_redirections.find(concept) != concept_redirections.end()) {
            concept = concept_redirections.at(concept);
            corrected++;
            total_concepts--;
        }
    }
    std::cout << "num corrected els " << corrected << std::endl;
    std::cout << "total els now " << total_concepts << std::endl;
    // Load some examples from DB
    SQLite::Statement   query(db, "SELECT lines FROM articles");
    // Convert protobuf -> vector<string>
    auto els = load_protobuff_dataset(query, index2concept, 100);
    cout << "got labeled examples" << endl;
}
