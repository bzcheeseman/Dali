#ifndef DALI_UTILS_VOCAB_H
#define DALI_UTILS_VOCAB_H

#include <string>
#include <vector>
#include <unordered_map>

namespace utils {
    extern const char* end_symbol;
    extern const char* unknown_word_symbol;

    class Vocab {
        private:
            void construct_word2index();
        public:

            void add(const std::vector<std::string>& words);
            void add(const std::string& word);

            void add_unknown_word();
            typedef int ind_t;
            ind_t unknown_word;
            std::unordered_map<std::string, ind_t> word2index;
            std::vector<std::string> index2word;

            std::vector<ind_t> encode(const std::vector<std::string>& words, bool with_end_symbol = false) const;
            std::vector<std::string> decode(const std::vector<int>& indices, bool remove_end_symbol = false) const;

            Vocab();
            Vocab(std::vector<std::string>&);
            Vocab(std::vector<std::string>&, bool);
            Vocab(std::vector<std::string>&&, bool);
            Vocab(std::vector<std::string>&&);
            ind_t operator[](const std::string&) const;
            size_t size() const;
    };

    class CharacterVocab {
        public:
            typedef int ind_t;
            ind_t min_char;
            ind_t max_char;

            std::vector<ind_t>       encode(const std::vector<std::string>& words) const;
            std::vector<std::string> decode(const std::vector<int>&) const;
            std::vector<std::string> decode_characters(const std::vector<int>&) const;

            CharacterVocab(int min_char, int max_char);
            size_t size() const;
    };
}

#endif
