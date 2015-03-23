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
