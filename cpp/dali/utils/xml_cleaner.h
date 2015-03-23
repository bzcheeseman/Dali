#ifndef CORE_UTILS_XML_CLEANER_H
#define CORE_UTILS_XML_CLEANER_H

#include <vector>
#include <string>
#include <regex>
#include "dali/utils/core_utils.h"

namespace utils {

    void inplace_regex_replace(std::string& s, std::regex& reg, const std::string& replacement);

    namespace xml_cleaner {
        extern std::regex mvar_parser;
        extern std::regex html_remover;
        extern std::regex markup_normalizer;
        extern std::regex remove_wikipedia_link;
        extern std::regex table_parser;
        extern std::regex squiggly_bracket_parser;
        extern std::regex remove_bullets_nbsps;
        extern std::regex math_source_sections;
        extern std::regex greater_than;
        extern std::regex less_than;
        extern std::regex period_mover;
        extern std::regex shifted_standard_punctuation;
        extern std::regex english_specific_appendages;
        extern std::regex english_nots;
        extern std::regex semicolon_shifter;
        extern std::regex french_appendages;
        extern std::regex shifted_parenthesis_squiggly_brackets;
        extern std::regex left_single_quote_converter;
        extern std::regex remaining_quote_converter;
        extern std::regex left_quote_shifter;
        extern std::regex left_quote_converter;
        extern std::regex english_contractions;
        extern std::regex right_single_quote_converter;
        extern std::regex dash_converter;
        extern std::regex no_punctuation;
        extern std::regex comma_shifter;
        extern std::regex shifted_ellipses;
        std::vector<std::string> process_text_keeping_brackets(
            const std::string& original);
        std::vector<std::string> split_punct_keep_brackets(
            const std::string& original);
    }
}

#endif
