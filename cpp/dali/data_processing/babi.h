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
            virtual void train(const std::vector<Story>& data) = 0;

            // testing
            virtual void new_story() = 0;
            virtual void fact(const VS& fact) = 0;
            virtual VS question(const VS& quesiton) = 0;
    };

    template<typename T>
    double benchmark_task(const std::string& task) {
        std::stringstream ss;
        ss << task << " training commenced!";
        ThreadPool::print_safely(ss.str());

        // It's important to explicitly state time here rather
        // using auto to ensure we get a compiler error if T
        // is not a subclass of babi::Model.
        std::shared_ptr<Model> m = std::make_shared<T>();

        m->train(Parser::training_data(task));

        std::stringstream ss2;
        ss2 << task << "training finished. Calculating accuracy on test set.";
        ThreadPool::print_safely(ss2.str());

        int correct_questions = 0;
        int total_questions = 0;
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

        double accuracy = 100.0*(double)correct_questions/total_questions;
        std::stringstream ss3;
        ss3 << task << " all done - accuracy on test set: "<< accuracy;
        ThreadPool::print_safely(ss3.str());
        return accuracy;
    }

    template<typename T>
    void benchmark() {
        VS tasks = Parser::tasks();

        std::vector<double> facebook_lstm_results = {
            50,  20,  20,  61,  70, 48,  49, 45, 64,  44, 72,  74,  94,  27, 21,  23,  51, 52, 8,  91
        };

        std::vector<double> facebook_best_results = {
            100, 100, 100, 100, 98, 100, 85, 91, 100, 98, 100, 100, 100, 99, 100, 100, 65, 95, 36, 100
        };

        std::vector<double> facebook_multitask_results = {
            100, 100, 98,  80,  99, 100, 86, 93, 100, 98, 100, 100, 100, 99, 100, 94,  72, 93, 19, 100
        };

        std::vector<double> our_results = {
            15,  18,  15,  17,  18, 49,  57, 33, 63,  43, 16,  17,  15,  17, 21,  23,  48, 53, 0,  36
        };

        std::atomic<int> tasks_remaining(tasks.size());

        for (int task_id = 0; task_id < tasks.size(); ++task_id) {
                const std::string& task = tasks[task_id];

                double accuracy = benchmark_task<T>(task);

                our_results[task_id] = accuracy;

                tasks_remaining += -1;

                std::stringstream ss;
                ss << task << "Remaining tasks: "<< tasks_remaining;
                ThreadPool::print_safely(ss.str());
        }

        std::cout << "Babi Benchmark Results" << std::endl
                  << std::endl
                  << "Columns are (from left): task name, our result," << std::endl
                  << "Facebook's LSTM result, Facebook's best result," << std::endl
                  << "Facebook's multitask result." << std::endl
                  << "===============================================" << std::endl;

        std::function<std::string(std::string)> moar_whitespace = [](std::string input,
                                                                     int num_whitespace=30) {
            assert(input.size() < num_whitespace);
            std::stringstream ss;
            ss << input;
            num_whitespace -= input.size();
            while(num_whitespace--) ss << ' ';
            return ss.str();
        };

        for (int task_id = 0; task_id < tasks.size(); ++task_id) {
            char special = ' ';
            if (our_results[task_id] >= facebook_lstm_results[task_id]) special = '*';
            if (our_results[task_id] >= facebook_best_results[task_id]) special = '!';


            std::cout << std::setprecision(1) << std::fixed
                      << moar_whitespace(tasks[task_id]) << "\t"
                      << special << our_results[task_id] << special << "\t"
                      << facebook_lstm_results[task_id] << "\t"
                      << facebook_best_results[task_id] << "\t"
                      << facebook_multitask_results[task_id] << std::endl;
        }
    }

    typedef std::tuple<std::vector<std::vector<std::string>>, QA*> sp_ret_t;

    class StoryParser : public utils::Generator<sp_ret_t> {
        std::vector<std::vector<std::string>> facts_so_far;
        int story_idx;
        const Story* story;

        public:
            StoryParser(const Story* story);

            sp_ret_t next();

            bool done();
    };
};



#endif
