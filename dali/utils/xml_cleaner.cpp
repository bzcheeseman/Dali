#include "dali/utils/xml_cleaner.h"

using std::string;
using std::vector;
using std::regex;

namespace utils {

    void inplace_regex_replace(string& s, regex& reg, const string& replacement) {
        string result;
        std::regex_replace(std::back_inserter(result),
            s.begin(),
            s.end(),
            reg, replacement);
        s = result;
    }

    namespace xml_cleaner {

        regex mvar_parser("\\{\\{\\d*mvar\\d*\\|([^\\}]+)\\}\\}");
        regex html_remover("<[^>]+>");
        regex markup_normalizer("[',/\\*_=-]{2,5}");
        regex remove_wikipedia_link("\\[\\W*http[^\\] ]+\\b*(?[^\\]]+)\\]");
        regex table_parser("\\{\\|[^\\}]+\\|\\}");
        regex squiggly_bracket_parser("\\{\\{([^\\}]+)\\}\\}");
        regex remove_bullets_nbsps("(&amp;nbsp;|&nbsp;|[\\^\n]\\*{1,}|[\\^\n]\\#{1,}|[\\^\n]:{1,})");
        regex math_source_sections("<(math|source|code|sub|sup)[^>]*>([^<]*)</(math|source|code|sub|sup)>");
        regex greater_than("(\\W)>(\\W)");
        regex less_than("<([^\\w/])");
        regex period_mover("([a-zA-ZÀ-Þ]{2})([\\./])\\s+([a-zA-ZÀ-Þ]{2})");
        regex shifted_standard_punctuation("([\\!\?#\\$%;~|])");
        regex english_specific_appendages("([A-Za-z])['’]([dms])\\b");
        regex english_nots("n['’]t\\b");
        regex semicolon_shifter("(.):([^/])");
        regex french_appendages("(\\b[tjnlsmdclTJNLSMLDC]|qu)['’](?=[^tdms])");
        regex shifted_parenthesis_squiggly_brackets("([\\(\\{\\}\\)])");
        regex left_single_quote_converter("(?:(\\W|^))'(?=.*\\w)");
        regex remaining_quote_converter("[\"“”»]");
        regex left_quote_shifter("`(?!`)(?=.*\\w)");
        regex left_quote_converter("[«\"](?=.*\\w)");
        regex english_contractions("['’](ve|ll|re)\\b");
        regex right_single_quote_converter("(\\w)'(?!')(?=\\W|$)");
        regex dash_converter("–|--+|â\x80\x93|‐|‑|‒|—|―");
        regex any_punctuation("\\W+");
        regex comma_shifter(",(?!\\d)");
        regex shifted_ellipses("(\\.\\.\\.+|…)");

        vector<string> split_punct_keep_brackets(const string& original) {
            // if no punctuation, return
            if (!std::regex_search(original.begin(), original.end(), any_punctuation)) {
                return {original};
            }
            // normalize and simplify punctuation:
            string text = original;
            inplace_regex_replace(text, period_mover, "$1 $2 $3");
            inplace_regex_replace(text, left_quote_shifter, "` ");
            inplace_regex_replace(text, left_quote_converter, "`` ");
            inplace_regex_replace(text, left_single_quote_converter, "$1 ` ");
            inplace_regex_replace(text, remaining_quote_converter, " '' ");
            inplace_regex_replace(text, english_nots, " n't");
            inplace_regex_replace(text, english_contractions, " '$1");
            inplace_regex_replace(text, english_specific_appendages, "$1 '$2");
            inplace_regex_replace(text, right_single_quote_converter, "$1 ' ");
            inplace_regex_replace(text, dash_converter, " - ");
            inplace_regex_replace(text, comma_shifter, " , ");
            inplace_regex_replace(text, semicolon_shifter, "$1 : $2");
            inplace_regex_replace(text, shifted_ellipses, " ...");
            inplace_regex_replace(text, shifted_parenthesis_squiggly_brackets, " $1 ");
            inplace_regex_replace(text, shifted_standard_punctuation, " $1 ");
            inplace_regex_replace(text, french_appendages, "$1' ");

            // replace crazy symbols by common ones:
            auto special_symbol1_ptr = text.find("œ");
            while (special_symbol1_ptr != std::string::npos) {
                text.replace(special_symbol1_ptr, 1, "oe");
                special_symbol1_ptr = text.find("œ");
            }

            special_symbol1_ptr = text.find("æ");
            while (special_symbol1_ptr != std::string::npos) {
                text.replace(special_symbol1_ptr, 1, "ae");
                special_symbol1_ptr = text.find("æ");
            }
            // replace newlines by space
            std::replace(text.begin(), text.end(), '\n', ' ');
            return utils::split(
                text,
                ' ',
                false // don't keep empty strings
            );
        }

        std::vector<string> process_text_keeping_brackets(const string& original) {
            string text = original;
            inplace_regex_replace(text, mvar_parser, "$1");
            inplace_regex_replace(text, squiggly_bracket_parser, "");
            inplace_regex_replace(text, table_parser, "");
            inplace_regex_replace(text, markup_normalizer, "");
            inplace_regex_replace(text, remove_wikipedia_link, "$&");
            inplace_regex_replace(text, remove_bullets_nbsps, "");
            inplace_regex_replace(text, math_source_sections, "");
            inplace_regex_replace(text, greater_than, "$1&gt;$2");
            inplace_regex_replace(text, less_than, "&lt;$1");
            inplace_regex_replace(text, html_remover, " ");

            return split_punct_keep_brackets(text);
        }
    } // namespace xml_cleaner
} // namespace utils
