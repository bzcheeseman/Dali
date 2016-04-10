#include "ParseUtils.h"

#include "dali/utils/core_utils.h"
#include "dali/utils/print_utils.h"

using std::vector;
using std::string;
using utils::assert2;
using utils::MS;
using std::getline;


string SmartParser::next_line_internal() {
    string line_buffer;
    getline(*stream, line_buffer);
    assert2(!stream->eof(), "EOF encountered while parsing a file.");
    return line_buffer;
}

SmartParser SmartParser::from_path(std::string filename) {
        assert2(utils::file_exists(filename),
        MS() << "Error: File \"" << filename
             << "\" does not exist, did you run generate script?");

        return SmartParser(std::make_shared<std::ifstream>(filename));
}

SmartParser::SmartParser(std::shared_ptr<std::istream> stream) : stream(stream) {
}

std::string SmartParser::next_token() {
    while (token_in_line >= line_tokens.size()) {
        string line_buffer = next_line_internal();
        // split by space, don't keep empty strings.
        line_tokens = utils::split(line_buffer, ' ', false);
        token_in_line = 0;
    }
    return line_tokens[token_in_line++];
}
std::string SmartParser::next_string() {
    return next_token();
}

std::string SmartParser::next_line() {
    assert2(token_in_line >= line_tokens.size(), "You must finish reading tokens from current line, before requesting next one.");
    return next_line_internal();
}

int SmartParser::next_int() {
    return std::stoi(next_token());
}
