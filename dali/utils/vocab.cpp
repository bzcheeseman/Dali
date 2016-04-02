#include "vocab.h"

#include "dali/utils/core_utils.h"
#include <sstream>

using std::string;
using std::vector;
using std::stringstream;

const char* utils::end_symbol          = "**END**";
const char* utils::unknown_word_symbol = "███████";

#ifdef DONT_COMPILE

namespace utils {
	typename Vocab::ind_t Vocab::operator[](const string& word) const {
        auto found = word2index.find(word);
        if (found != word2index.end()) {
            return found->second;
        }
        return unknown_word;
    }

    void Vocab::construct_word2index() {
        uint i = 0;
        for (auto& s : index2word)
            word2index[s] = i++;
    }
    void Vocab::add_unknown_word() {
        if (word2index.find(unknown_word_symbol) == word2index.end()) {
            index2word.emplace_back(unknown_word_symbol);
            word2index[unknown_word_symbol] = index2word.size() - 1;
            unknown_word = index2word.size() - 1;
        } else {
            unknown_word = word2index.at(unknown_word_symbol);
        }
    }

    size_t Vocab::size() const {
        return index2word.size();
    }

    vector<typename Vocab::ind_t> Vocab::encode(const vector<string>& words, bool with_end_symbol) const {
        vector<ind_t> result;
        result.reserve(words.size() + (with_end_symbol ? 1 : 0));
        std::transform(words.begin(), words.end(),
                       std::back_inserter(result), [this](const string& word) {
            if (word2index.find(word) == word2index.end()) {
                return unknown_word;
            } else {
                return word2index.at(word);
            }
        });
        if (with_end_symbol) {
            result.emplace_back( word2index.at(utils::end_symbol) );
        }
        return result;
    }

    void Vocab::add(const vector<string>& words) {
        for (auto& word : words) {
            add(word);
        }
    }

    void Vocab::add(const string& word) {
        auto found = word2index.find(word) != word2index.end();
        if (!found) {
            uint next_index = word2index.size();
            word2index[word] = next_index;
            index2word.emplace_back(word);
        }
    }

    vector<string> Vocab::decode(Indexing::Index indices, bool remove_end_symbol) const {
        vector<string> result;
        result.reserve(indices.size());
        // either the decoding must remove the end symbol
        // if there is one, or we assume there is none to remove.
        bool has_end_symbol = remove_end_symbol ?
            indices[indices.size() - 1] == word2index.at(utils::end_symbol) :
            false;
        auto index_end = indices.data() + indices.size();
        if (has_end_symbol) index_end--;
        std::transform(indices.data(), index_end,
                       std::back_inserter(result), [this](const ind_t& idx) {
            if (idx < index2word.size()) {
                return index2word[idx];
            } else {
                return index2word[unknown_word];
            }
        });
        return result;
    }

    Vocab::Vocab() : unknown_word(-1) {add_unknown_word();};

    Vocab::Vocab(vector<string>& _index2word) : index2word(_index2word), unknown_word(-1) {
        construct_word2index();
        add_unknown_word();
    }
    Vocab::Vocab(vector<string>& _index2word, bool _unknown_word) : index2word(_index2word), unknown_word(-1) {
        construct_word2index();
        if (_unknown_word) add_unknown_word();
    }

    Vocab::Vocab(vector<string>&& _index2word) : Vocab(_index2word) {
    }
    Vocab::Vocab(vector<string>&& _index2word, bool _unknown_word) : Vocab(_index2word, _unknown_word) {
    }

    CharacterVocab::CharacterVocab(int min_char, int max_char)
        : min_char(min_char), max_char(max_char) {
        assert2(max_char > min_char, MS() << "Maximum character (" << max_char << ") must be larger than minimum character (" << min_char << ").");
        assert2(max_char >= 0 && min_char >= 0, "Cannot have negative characters in mapping");
    }

    size_t CharacterVocab::size() const {
        return (size_t) ((max_char - min_char) + 1);
    }

    vector<typename Vocab::ind_t> CharacterVocab::encode(const vector<string>& words) const {
        vector<ind_t> result;
        int char_size = 0;
        // add all characters:
        for (auto& w : words) char_size += w.size();
        // for spaces:
        if (words.size() > 0) char_size += words.size() - 1;

        result.reserve(char_size);
        int unknown_char = max_char - min_char;
        int space_char = ' ' - min_char;

        if (space_char < 0) space_char = unknown_char;

        int word_idx = 0;
        for (auto& w : words) {
            for (auto& c : w) {
                if ((int)c >= min_char && (int)c < max_char) {
                    result.emplace_back(c - min_char);
                } else {
                    // all unknown get replaced by max_char
                    result.emplace_back(unknown_char);
                }
            }
            word_idx++;
            if (word_idx < words.size()) result.emplace_back(space_char);
        }
        return result;
    }

    vector<string> CharacterVocab::decode(Indexing::Index indices) const {
        vector<string> result;
        result.reserve(indices.size());

        stringstream stream;
        for (auto& index : indices) {
            if (index == max_char) {
                stream << "█";
            } else {
                stream << ((char) (index + min_char));
            }
        }
        return tokenize(stream.str());
    }

    vector<string> CharacterVocab::decode_characters(Indexing::Index indices) const {
        vector<string> result(indices.size());
        int char_idx = 0;
        for (auto& index : indices) {
            if (index == max_char) {
                result[char_idx++] = "█";
            } else {
                result[char_idx++] = (char) (index + min_char);
            }
        }
        return result;
    }
}

#endif
