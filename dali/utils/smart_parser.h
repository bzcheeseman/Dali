#ifndef DALI_UTILS_PARSE_UTILS_H
#define DALI_UTILS_PARSE_UTILS_H

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

namespace utils {
    class SmartParser {
        std::shared_ptr<std::istream> stream;
        std::vector<std::string> line_tokens;
        int token_in_line;

        std::string next_line_internal();

        public:
            SmartParser(std::shared_ptr<std::istream> steam);

            static SmartParser from_path(std::string path);

            int next_int();
            std::string next_token();
            std::string next_string();
            std::string next_line();
    };
}  // namespace utils

#endif
