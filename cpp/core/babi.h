#ifndef BABI_H
#define BABI_H

#include <cassert>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>

#include "core/utils.h"

namespace babi {
    class Item {
        protected:
            virtual void to_stream(std::ostream& str) const;
        public:
            friend std::ostream& operator<<(std::ostream& str, const Item& data);
    };

    class QA: public Item {
        protected:
            virtual void to_stream(std::ostream& str) const;
        public:
            const std::vector<std::string> question;
            const std::vector<std::string> answer;
            const std::vector<int> supporting_facts;

            QA(const std::vector<std::string> question,
               const std::vector<std::string> answer,
               const std::vector<int> supporting_facts);
    };

    class Fact: public Item {
        protected:
            virtual void to_stream(std::ostream& str) const;
        public:
            const std::vector<std::string> fact;

            Fact(const std::vector<std::string> fact);
    };

    typedef std::vector<std::shared_ptr<Item> > Story;

    class Parser {
        static std::vector<Story> parse_file(const std::string& file,
                                             const int& num_questions);
        public:
                static std::string data_dir();
                static std::vector<std::string> tasks();
                // When facebook tested their algorithms with only 1000
                // training examples, but the dataset they provide is 10000
                // examples (number of examples = number of questions).
                static std::vector<Story> training_data(
                        const std::string& task,
                        const int& num_questions=1000,
                        bool shuffled=false);
    };
};


#endif
