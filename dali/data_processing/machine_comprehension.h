#ifndef DALI_DATA_PROCESSING_MACHINE_COMPREHENSION_H
#define DALI_DATA_PROCESSING_MACHINE_COMPREHENSION_H

#include <set>
#include <vector>
#include <string>


namespace mc {
    struct Question {
        std::string type;
        std::vector<std::string> text;

        std::vector<std::vector<std::string>> answers;

        int correct_answer;

        void print();
    };

    struct Section {
        std::string name;
        std::string turk_info;
        std::vector<std::vector<std::string>> text;
        std::vector<Question> questions;

        void print();
    };

    extern std::string data_dir;

    std::tuple<std::vector<Section>, std::vector<Section>> load();

    std::vector<std::string> extract_vocabulary(const std::vector<Section>&);
}
#endif
