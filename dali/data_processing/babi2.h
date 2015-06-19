



namespace babi {
	struct QA<typename word_t> {
		vector<vector<word_t>> facts;
		vector<word_t> question;
		vector<word_t> answer;
		vector<uint> supporting_facts;
	};

	template<typename word_t>
	struct Story {
		vector<vector<word_t>> facts;
		vector<uint> question_fidx;
		vector<vector<uint>> supporting_facts;
		vector<vector<word_t>> answers;

		int num_qas() {
			return question_fidx.size();
		}
		QA<word_t> get_qa(int target_question_idx) {
			QA<word_t> result;

			assert(0 <= idx && idx < question_fidx.size());

			int start_idx = 0;
			for (int qidx = 0; qidx <= target_question_idx; ++qidx) {
				result.facts.emplace_back();
				for (int fidx = start_idx; fidx < question_fidx[qidx]; ++fidx) {
					result.facts.back().push_back(facts[fidx]);
					if (in_vector(supporting_facts[target_question_idx], fidx)) {
						result.back().supporting_facts.emplace_back(result.back().facts.size() - 1);
					}
				}
				start = question_fidx[qidx] + 1;
			}
			result.question = facts[question_fidx[qidx]];
			result.answer = answer[qidx];

			return result;
		}
	};

	std::tuple<Story<uint>, Vocab> encode_vocabulary(Story<string> input) {
		vector<string> words;
		for (auto& fact: input.facts) {
			for (auto& word: fact) {
				words.push_back(word);
			}
		}
		for (auto& answer: input.answers) {
			for (auto& word: answer) {
				words.push_back(word);
			}
		}
		std::sort(words.begin(), words.end());
		words.erase(std::unique(words.begin(), words.end()), words.end());
		Vocab vocab(words, false);
		Story<uint> output;
		output.question_fidx = input.question_fidx;
		output.supporting_facts = input.supporting_facts;
		for (auto& fact: input.facts) {
			output.emplace_back(vocab.encode(fact));
		}
		for (auto& answer: input.answers) {
			output.emplace_back(vocab.encode(anser));
		}
		return std::make_tuple(output), vocab);
	}


	vector<Story> Parser::parse_file(const string& filename,
                                     const int& num_questions) {

        if (!utils::file_exists(filename)) {
            std::stringstream error_msg;
            error_msg << "Error: File \"" << filename << "\" does not exist, cannot parse file.";
            throw std::runtime_error(error_msg.str());
        }

        std::ifstream file(filename);
        // file exists
        assert(file.good());

        vector<Story<string>> results;
        results.emplace_back();

        while(std::getline(file, line_buffer)) {
            // Read story id. Non-increasing id is indication
            // of new story.
            std::stringstream line(line_buffer);
            int line_number;
            line >> line_number;
            if (last_line_no == -1 || line_no <= last_line_no) {
            	results.emplace_back();
            }

            // parse the fact.
            vector<string> fact;
            while(true) {
            	string token;
            	line >> token;
            	if (token.back() == '.' || token.back() == '?') {
            		fact.emplace_back(token.begin(), token.end() - 1);
            		fact.emplace_back(token.end()-1, token.end());
            		break;
            	} else {
            		fact.push_back(token);
            	}
            }

            // store the fact.
            results.back().facts.push_back(fact);

            // if this is a question store its index.
            if (fact.back().back() == '?') {
            	results.back().question_fidx.append(facts.size() - 1);
            }

            // if this ia question do read in the answer and supporting facts.
            if (fact.back().back() == '?') {
                string comma_separated_answer;
                line >> comma_separated_answer;

                vector<string> answer = utils::split(comma_separated_answer, ',');
                results.back().answer.push_back(answer);

                results.back().supporting_facts.emplace_back();

                int supporting_fact;
                while(line >> supporting_fact) {
                	results.back().supporting_facts.back().push_back(supporting_fact - 1);
                }
            }
        }

        return result;
    }

}
