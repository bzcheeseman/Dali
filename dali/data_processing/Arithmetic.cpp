#include "dali/data_processing/Arithmetic.h"

using std::vector;
using std::string;
using std::to_string;
using utils::Vocab;

namespace arithmetic {
    std::vector<std::string> symbols = {"+", "*", "-"};
    int NUM_SYMBOLS = symbols.size();

    vector<std::pair<vector<string>, vector<string>>> generate(int num, int expression_length) {
        vector<std::pair<vector<string>, vector<string>>> examples;
        int i = 0;
        while (i < num) {
            vector<string> example;
            bool use_operator = false;
            for (int j = 0; j < expression_length; j++) {
                if (use_operator) {
                    auto operation = symbols[utils::randint(0, NUM_SYMBOLS-1)];
                    example.push_back(operation);
                    use_operator = false;
                } else {
                    auto value = std::to_string(utils::randint(0, 9));
                    example.push_back(value);
                    use_operator = true;
                }
            }
            if (!use_operator) {
                auto value = std::to_string(utils::randint(0, 9));
                example.push_back(value);
                use_operator = true;
            }
            int result = 0;
            {
                int product_so_far = 1;
                vector<string> multiplied;
                for (auto& character : example) {
                    if (utils::in_vector(symbols, character)) {
                        if (character == "*") {
                            // do nothing
                        } else {
                            multiplied.push_back(to_string(product_so_far));
                            multiplied.push_back(character);
                            product_so_far = 1;
                        }
                    } else {
                        product_so_far *= character[0] - '0';
                    }
                }
                multiplied.push_back(to_string(product_so_far));

                string last_operator = "";
                for (auto& character: multiplied) {
                    if (utils::in_vector(symbols, character)) {
                        last_operator = character;
                    } else {
                        if (last_operator == "") {
                            result = std::stoi(character);
                        } else if (last_operator == "+") {
                            result += std::stoi(character);
                        } else if (last_operator == "-") {
                            result -= std::stoi(character);
                        } else {
                            assert(NULL == "Unknown operator.");
                        }
                    }
                }
            }
            if (result > -500000 && result < 500000) {
                i++;
                auto res = to_string(result);
                vector<string> character_result;
                for (int j = 0; j < res.size(); j++) {
                    character_result.emplace_back(res.begin()+j, res.begin()+j+1);
                }
                examples.emplace_back(
                    example,
                    character_result
                );
            }
        }
        return examples;
    }

    vector<NumericalExample> generate_numerical(int num, int expression_length, bool with_end_symbol) {
        auto examples = generate(num, expression_length);
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
