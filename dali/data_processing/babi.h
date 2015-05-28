#ifndef BABI_H
#define BABI_H

#include <atomic>
#include <cassert>
#include <dirent.h>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "dali/utils.h"


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
            const VS question;
            const VS answer;
            const std::vector<int> supporting_facts;

            QA(const VS question,
               const VS answer,
               const std::vector<int> supporting_facts);
    };

    class Fact: public Item {
        protected:
            virtual void to_stream(std::ostream& str) const;
        public:
            const VS fact;

            Fact(const VS fact);
    };

    typedef std::vector<std::shared_ptr<Item> > Story;

    class Parser {
        static std::vector<Story> parse_file(const std::string& file,
                                             const int& num_questions);
        public:
            static std::string data_dir();
            // When facebook tested their algorithms with only 1000
            // training examples, but the dataset they provide is 10000
            // examples (number of examples = number of questions).
            static std::vector<Story> training_data(
                    const std::string& task,
                    const int& num_questions=1000,
                    bool shuffled=false);

            static std::vector<Story> testing_data(
                    const std::string& task,
                    const int& num_questions=1000,
                    bool shuffled=false);
    };


    typedef std::tuple<std::vector<std::vector<std::string>>, QA*> sp_ret_t;

    utils::Generator<sp_ret_t> story_parser(const Story& story);

    // List of all the babi tasks
    std::vector<std::string> tasks();

    // Evaluates accuracy of a model on a task using generic predict function.
    // prediction_fun: (fact_list, question) -> answer
    typedef std::function<VS(const std::vector<VS>&, const VS&)> prediction_fun;
    double accuracy(const std::vector<Story>& data, prediction_fun predict, int num_threads=1);
    double task_accuracy(const std::string& task, prediction_fun predict, int num_threads=1);

    // Extract list of words used in all the facts, questions and answers
    // in particular dataset.
    std::vector<std::string> vocab_from_data(const std::vector<Story>& data);

    // Takes as argument list of results for all the 20 tasks
    // and prints them side by side with facebook results.
    void compare_results(std::vector<double> our_results);
};



#endif
