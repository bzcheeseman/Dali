#include "dali/utils/unpack_tuple.h"

#include <string>
#include <gtest/gtest.h>

std::string hello_world(const int& x, const std::string& y) {
    return y + std::string(x, '!');
}

std::string hello_world_explicit(const std::tuple<int, std::string>& stuff) {
    return hello_world(std::get<0>(stuff), std::get<1>(stuff));
}

TEST(UnpackTuple, unpack_calls_correctly_when_explicit) {
    auto packaged_input = std::make_tuple(3, std::string("hello world"));
    auto result = hello_world_explicit(packaged_input);
    EXPECT_EQ("hello world!!!", result);
}

TEST(UnpackTuple, unpack_calls_correctly) {
    auto packaged_input = std::make_tuple(3, std::string("hello world"));
    auto result = unpack_tuple(hello_world, packaged_input);
    EXPECT_EQ("hello world!!!", result);
}
