#include <gtest/gtest.h>

#include "dali/data_processing/Glove.h"
#include "dali/data_processing/Arithmetic.h"

TEST(Glove, load) {
    auto embedding = glove::load<double>( STR(DALI_DATA_DIR) "/glove/test_data.txt");
    ASSERT_EQ(std::get<1>(embedding).index2word.size(), 21);
    ASSERT_EQ(std::get<0>(embedding).dims(0), 21);
    ASSERT_EQ(std::get<0>(embedding).dims(1), 300);
}

// exposing internal functions from arithmetic for testing.
namespace arithmetic {
    std::tuple<std::vector<int>, std::vector<std::string>> remove_multiplies(const std::vector<int>& numbers, const std::vector<std::string>& ops);
    std::tuple<std::vector<int>, std::vector<std::string>> generate_example(int expression_length, int& min, int& max);
    std::vector<std::string> convert_to_chars(const std::vector<int>& numbers, const std::vector<std::string>& ops);
    int compute_result(const std::vector<int>& numbers, const std::vector<std::string>& ops);
}

TEST(arithmetic, generate) {
    int min = 0;
    int max = 9;
    auto example = arithmetic::generate_example(5, min, max);
    ASSERT_EQ(std::get<0>(example).size(), 3);
    ASSERT_EQ(std::get<1>(example).size(), 2);

    auto example2     = std::make_tuple(std::vector<int>({12, 9, 3}), std::vector<std::string>({"*", "+"}));
    auto demultiplied = arithmetic::remove_multiplies(std::get<0>(example2), std::get<1>(example2));
    auto example3     = std::make_tuple(std::vector<int>({108, 3}), std::vector<std::string>({"+"}));

    ASSERT_EQ(demultiplied, example3);
    ASSERT_EQ(arithmetic::compute_result(std::get<0>(example3), std::get<1>(example3)), 111);


    example2     = std::make_tuple(std::vector<int>({12, 9, 3, 5, 5}), std::vector<std::string>({"*", "-","+", "*"}));
    demultiplied = arithmetic::remove_multiplies(std::get<0>(example2), std::get<1>(example2));
    example3     = std::make_tuple(std::vector<int>({108, 3, 25}), std::vector<std::string>({"-", "+"}));

    ASSERT_EQ(demultiplied, example3);
    ASSERT_EQ(arithmetic::compute_result(std::get<0>(example3), std::get<1>(example3)), 130);
}

