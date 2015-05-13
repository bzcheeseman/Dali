#include "dali/data_processing/Arithmetic.h"

using std::vector;
using std::string;
using std::to_string;
using utils::Vocab;

namespace arithmetic {

    std::vector<std::string> symbols = {"+", "*", "-"};
    int NUM_SYMBOLS = symbols.size();

    vector<string> convert_to_chars(const vector<int>& numbers, const vector<string>& ops) {
        utils::assert2(numbers.size() - 1 == ops.size(), "Should be 1 less operation than element");
        vector<string> character_list;
        bool use_operator = false;
        int op_idx = 0;
        int el_idx = 0;
        for (int j = 0; j < numbers.size() + ops.size(); j++) {
            if (use_operator) {
                character_list.push_back(ops[op_idx]);
                op_idx++;
            } else {
                auto num = to_string(numbers[el_idx]);
                for (int c_offset = 0; c_offset < num.size(); c_offset++) {
                    character_list.emplace_back(
                        num.begin() + c_offset,
                        num.begin() + c_offset + 1
                    );
                }
            }
            use_operator = use_operator ? false : true;
        }
        return character_list;
    }

    std::tuple<vector<int>, vector<string>> generate_example(int expression_length, int& min, int& max) {
        std::tuple<vector<int>, vector<string>> example;
        bool use_operator = false;
        for (int j = 0; j < expression_length; j++) {
            if (use_operator) {
                auto operation = symbols[utils::randint(0, NUM_SYMBOLS-1)];
                std::get<1>(example).push_back(operation);
                use_operator = false;
            } else {
                std::get<0>(example).push_back(utils::randint(min, max));
                use_operator = true;
            }
        }
        return example;
    }

    std::tuple<vector<int>, vector<string>> remove_multiplies(const vector<int>& numbers, const vector<string>& ops) {
        std::tuple<vector<int>, vector<string>> example;
        int op_idx         = 0;
        int el_idx         = 0;
        bool use_operator  = false;
        int product_so_far = 1;
        for (int read_idx = 0; read_idx < numbers.size() + ops.size(); read_idx++) {
            if (use_operator) {
                if (ops[op_idx] == "*") {
                    // do nothing
                } else {
                    std::get<0>(example).push_back(product_so_far);
                    std::get<1>(example).push_back(ops[op_idx]);
                    product_so_far = 1;
                }
                op_idx++;
            } else {
                product_so_far *= numbers[el_idx];
                el_idx++;
            }
            // reverse operator
            use_operator = use_operator ? false : true;
        }
        std::get<0>(example).push_back(product_so_far);
        return example;
    }

    int compute_result(const vector<int>& numbers, const vector<string>& ops) {
        int result           = 0;
        bool use_operator    = false;
        int op_idx           = 0;
        int el_idx           = 0;
        string last_operator = "";
        for (int read_idx = 0; read_idx < (numbers.size() + ops.size()); read_idx++) {
            if (use_operator) {
                last_operator = ops[op_idx];
                op_idx++;
                use_operator = false;
            } else {
                if (last_operator == "") {
                    result = numbers[el_idx];
                } else if (last_operator == "+") {
                    result += numbers[el_idx];
                } else if (last_operator == "-") {
                    result -= numbers[el_idx];
                } else {
                    utils::assert2(false, "Unknown operator.");
                }
                el_idx++;
                use_operator = true;
            }
        }
        return result;
    }

    vector<std::pair<vector<string>, vector<string>>> generate(int num, int expression_length, int min, int max) {
        utils::assert2(max > min, "Max must be greater than min number in generation");
        utils::assert2(min >= 0,  "Smallest number produced must be non-negative.");
        utils::assert2(max > 0,   "Largest number produced must be strictly positive.");
        vector<std::pair<vector<string>, vector<string>>> examples;
        // make sure it is odd
        if (expression_length % 2 == 0) {
            expression_length = expression_length + 1;
        }
        /**
        4 steps:
        --------

        1. Create a random set of operation,
        2. Remove multiplies to leave only + & -
        3. Compute result
        4. Convert to string

        **/
        int i = 0;
        while (i < num) {

            // 1. Create a random set of operation,
            vector<int> numbers;
            vector<string> ops;
            std::tie(numbers, ops) = generate_example(expression_length, min, max);

            int result = 0;
            {
                vector<int> simple_numbers;
                vector<string> simple_ops;

                // 2. Remove multiplies to leave only + & -
                std::tie(simple_numbers, simple_ops) = remove_multiplies(numbers, ops);

                // 3. Compute result
                result = compute_result(simple_numbers, simple_ops);
            }

            // 4. Convert to string
            if (result > -500000 && result < 500000) {
                i++;
                examples.emplace_back(
                    convert_to_chars(numbers, ops),
                    convert_to_chars({result}, {})
                );
            }
        }
        return examples;
    }

    vector<NumericalExample> generate_numerical(int num, int expression_length, int min, int max, bool with_end_symbol) {
        auto examples = generate(num, expression_length, min, max);
        vector<NumericalExample> numerical_examples(examples.size());
        for (size_t i = 0; i < examples.size();i++) {
            numerical_examples[i].first  = arithmetic::vocabulary.encode(examples[i].first, with_end_symbol);
            numerical_examples[i].second = arithmetic::vocabulary.encode(examples[i].second, true);
        }
        return numerical_examples;
    }

    Vocab create_vocabulary() {
        // define symbols:
        vector<string> symbols;
        for (int i = 0; i < 10; i++) {
            symbols.push_back(to_string(i));
        }
        symbols.insert(symbols.end(), arithmetic::symbols.begin(), arithmetic::symbols.end());
        symbols.push_back(utils::end_symbol);
        return Vocab(symbols, false);
    }

    Vocab vocabulary = create_vocabulary();
}
