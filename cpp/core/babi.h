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
                static VS tasks();
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

    class Model {
        public:
            // training
            virtual void train(std::vector<Story> data) = 0;

            // testing
            virtual void new_story() = 0;
            virtual void fact(const VS& fact) = 0;
            virtual VS question(const VS quesiton) = 0;
    };

    template<typename T>
    void benchmark(bool verbose=false) {

        for (auto& task : Parser::tasks()) {
            // It's important to explicitly state time here rather
            // using auto to ensure we get a compiler error if T
            // is not a subclass of babi::Model.
            std::shared_ptr<Model> m = std::make_shared<T>();

            std::cout << task << "... ";

            utils::Timer train_timer("Training");
            m->train(Parser::training_data(task));
            train_timer.stop();

            int correct_questions = 0;
            int total_questions = 0;
            utils::Timer test_timer("Testing");
            for (auto& story : Parser::testing_data(task)) {
                m->new_story();
                for (auto& item_ptr : story) {
                    if (Fact* f = dynamic_cast<Fact*>(item_ptr.get())) {
                        m->fact(f->fact);
                    } else if (QA* qa = dynamic_cast<QA*>(item_ptr.get())) {
                        VS answer = m->question(qa->question);
                        if(utils::vs_equal(answer, qa->answer)) {
                            ++correct_questions;
                        }
                        ++total_questions;
                    } else {
                        assert(NULL == "Unknown subclass of babi::Item");
                    }
                }
            }
            test_timer.stop();
            std::cout << (double)correct_questions/total_questions << std::endl;
            if (verbose) {
                utils::Timer::report();
            }
        }
    }
};


#endif
