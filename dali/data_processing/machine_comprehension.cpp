#include "machine_comprehension.h"

#include "dali/utils.h"

using std::vector;
using std::string;
using utils::join;

namespace mc {
    void Section::print() {
        std::cout << name << std::endl;
        std::cout << turk_info << std::endl;
        std::cout << std::endl;
        std::cout << join(text, " ") << std::endl << std::endl;
        std::cout << std::endl;
        for (int qidx = 0; qidx < questions.size(); ++qidx) {
            std::cout << "Question " << qidx+1 << ": ";
            questions[qidx].print();
            std::cout << std::endl;
        }
    }

    void Question::print() {
        if (type == "one")  {
            std::cout << "(single supporting fact) ";
        } else {
            std::cout << "(multiple supporting facts) ";
        }
        std::cout << join(text, " ") << std::endl;
        for (int aidx = 0; aidx < answers.size(); ++aidx) {
            char c = aidx == correct_answer ? '@' : '*';
            std::cout << "    " << c << " " << join(answers[aidx], " ") << std::endl;
        }
    }

    string data_dir = utils::dir_join({ STR(DALI_DATA_DIR), "machine_comprehension/" });

    vector<Section> parse_file(string filename) {
        vector<Section> result;

        SmartParser sp = SmartParser::from_path(filename);
        int num_sections = sp.next_int();
        while(num_sections--) {
            Section section;
            section.name = sp.next_line();
            section.turk_info = sp.next_line();
            section.text = utils::split(sp.next_line(), ' ', false);
            int num_questions = sp.next_int();
            while(num_questions--) {
                Question question;
                question.text = utils::split(sp.next_line(), ' ', false);
                question.type = sp.next_string();
                int num_answers = sp.next_int();
                while(num_answers--) {
                    vector<string> answer = utils::split(sp.next_line(), ' ', false);
                    question.answers.push_back(answer);
                }
                question.correct_answer = sp.next_int();
                section.questions.push_back(question);
            }
            result.push_back(section);
        }
        return result;
    }

    std::tuple<vector<Section>, vector<Section>> load() {
        string train_file = utils::join({data_dir, "mc_train.txt"});
        string test_file = utils::join({data_dir, "mc_test.txt"});
        return std::make_tuple(parse_file(train_file), parse_file(test_file));
    }
}
