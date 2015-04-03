#include <chrono>
#include <vector>
#include <memory>
#include <gtest/gtest.h>
#include <sstream>
#include <cstdio>

#include "dali/utils.h"
using std::stringstream;
using std::string;
using utils::OntologyBranch;
using std::make_shared;
#include <string>

TEST(utils, stream_to_redirection_list) {
    stringstream ss(
        "hello->world\n"
        "what->happened\n"
        "who->is this?\n");

    std::map<string, string> mapping;
    utils::stream_to_redirection_list(ss, mapping);

    ASSERT_TRUE(mapping.find("hello") != mapping.end());
    ASSERT_TRUE(mapping.find("what") != mapping.end());
    ASSERT_TRUE(mapping.find("who") != mapping.end());
    ASSERT_EQ(mapping.at("hello"), "world");
    ASSERT_EQ(mapping.at("what"), "happened");
    ASSERT_EQ(mapping.at("who"), "is this?");
}

TEST(utils, stream_to_list) {
    stringstream ss(
        "hello\n"
        "what\n"
        "who\n");

    std::vector<string> list;
    utils::stream_to_list(ss, list);
    ASSERT_EQ(list.size(), 3);
    ASSERT_EQ(list[0], "hello");
    ASSERT_EQ(list[1], "what");
    ASSERT_EQ(list[2], "who");
}

TEST(utils, split_str) {
    string input = "hellobobmeisterhowareyou?";

    auto tokens = utils::split_str(input, "bobmeister");

    ASSERT_EQ(tokens.size(), 2);
    ASSERT_EQ(tokens[0], "hello");
    ASSERT_EQ(tokens[1], "howareyou?");
}

TEST(utils, trim) {
    string input = "     hello_world thus ?     ";
    auto trimmed = utils::trim(input);
    ASSERT_EQ(trimmed, "hello_world thus ?");
}

TEST(xml_cleaner, preprocess) {
    // TODO add fail conditions for this test
    string input = "{{Other uses}}\n"
    "{{Redirect|Foo|FOO (Forward Observation Officer)|Artillery observer}}\n"
    "\n"
    "The terms '''foobar''' ({{IPAc-en|ˈ|f|uː|b|ɑr}}), '''fubar''', or "
    "'''foo''', '''bar''', '''baz''' and '''qux''' (alternatively, '''"
    "quux''') are sometimes used as [[placeholder name]]s (also referred"
    " to as [[metasyntactic variable]]s) in [[computer programming]] or "
    "computer-related documentation.<ref name=\"rfc3092\" /> They have been "
    "used to name entities such as [[Variable (computer science)|variable]]s,"
    " [[Function (computer science)|functions]], and [[command (computing)|command]]s"
    " whose purpose is unimportant and serve only to demonstrate a concept."
    "  The words themselves have no meaning in this usage.  ''Foobar'' is "
    "sometimes used alone; ''foo'', ''bar'', and ''baz'' are sometimes used"
    " in that order, when multiple entities are needed.\n"
    "\n"
    "The usage in [[computer programming]] examples and [[pseudocode]] "
    "varies; in certain circles, it is used extensively, but many prefer "
    "descriptive names, while others prefer to use single letters.  "
    "[[Eric S. Raymond]] has called it an \"important hackerism\" "
    "alongside [[kludge]] and [[cruft]].<ref name=\"Raymond\">{{cite "
    "book|url=http://books.google.com/books?id=g80P_4v4QbIC&amp;lpg=PP1"
    "&amp;pg=PA5|title=The New Hacker's Dictionary|author=[[Eric S. "
    "Raymond]]|publisher=[[MIT Press]]|year=1996|isbn=0-262-68092-0}}</ref>Okay\n";

    auto cleaned = utils::xml_cleaner::process_text_keeping_brackets(input);
}

TEST(utils, construct_lattice) {
    auto root = make_shared<OntologyBranch>("root");
    for (auto& v : {"Joe", "Bob", "Max", "Mary", "Jane", "Goodwin"})
        make_shared<OntologyBranch>(v)->add_parent(root);
    auto root2 = make_shared<OntologyBranch>("root 2");

    for (auto& child : root->children) {
        child->add_parent(root2);
    }

    stringstream ss;
    // Output the name of both parents (verify they are indeed 2)
    for (auto& par : root->children[2]->parents)
        ss << par.lock()->name << " ";

    ASSERT_EQ(ss.str(), "root root 2 ");
    ASSERT_EQ(root->children[2]->parents.size(), 2);
}

TEST(utils, load_save_lattice) {
    auto root = make_shared<OntologyBranch>("root");
    for (auto& v : {"Joe", "Bob", "Max", "Mary", "Jane", "Goodwin"})
        make_shared<OntologyBranch>(v)->add_parent(root);
    auto root2 = make_shared<OntologyBranch>("root 2");

    for (auto& child : root->children)
        child->add_parent(root2);

    string data_folder = STR(DALI_DATA_DIR) "/";
    string fname       = data_folder + "/lattice2.txt";
    string fname_gz    = fname + ".gz";

    if (utils::file_exists(fname))
        std::remove(fname.c_str());
    // Test saving a lattice file
    ASSERT_FALSE(utils::file_exists(fname));
    root->save(fname);
    ASSERT_TRUE(utils::file_exists(fname));
    ASSERT_FALSE(utils::is_gzip(fname));

    if (utils::file_exists(fname_gz))
        std::remove(fname_gz.c_str());
    // Test saving a gzipped file:
    ASSERT_FALSE(utils::file_exists(fname_gz));
    root->save(fname_gz);
    ASSERT_TRUE(utils::file_exists(fname_gz));
    ASSERT_TRUE(utils::is_gzip(fname_gz));
}

TEST(utils, load_lattice) {
    string data_folder = STR(DALI_DATA_DIR) "/";
    auto loaded_tree = OntologyBranch::load(data_folder + "lattice.txt");

    // find 2 roots
    ASSERT_EQ(loaded_tree.size(), 2);

    stringstream ss;
    for (auto& r : loaded_tree)
        ss << r->name << " ";
    ASSERT_EQ(ss.str(), "root 3 root 4 ");
    // find root 2
    ASSERT_TRUE(loaded_tree[0]->lookup_table->find("root 2") != loaded_tree[0]->lookup_table->end());

    // compare with the original tree
    auto root = make_shared<OntologyBranch>("root");
    for (auto& v : {"Joe", "Bob", "Max", "Mary", "Jane", "Goodwin"})
        make_shared<OntologyBranch>(v)->add_parent(root);

    // iterate through children of root 2
    auto root2_loaded = loaded_tree[0]->lookup_table->at("root 2");

    int found = 0;
    for (auto& child : root->children) {
        for (auto& subchild : root2_loaded->children) {
            if (subchild->name == child->name) {
                found += 1;
                break;
            }
        }
    }
    ASSERT_EQ(found, root->children.size());
}
