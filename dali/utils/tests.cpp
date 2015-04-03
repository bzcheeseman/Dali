#include <chrono>
#include <vector>
#include <gtest/gtest.h>
#include <sstream>

#include "dali/utils.h"
using std::stringstream;
using std::string;
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
