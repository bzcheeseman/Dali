#ifndef DALI_DATA_PROCESSING_BABI2_H
#define DALI_DATA_PROCESSING_BABI2_H

#include <map>
#include <string>
#include <vector>

#include "dali/utils/core_utils.h"



namespace babi {
    template<typename word_t>
	struct QA {
		std::vector<std::vector<word_t>> facts;
		std::vector<word_t> question;
		std::vector<word_t> answer;
		std::vector<uint> supporting_facts;
	};

	template<typename word_t>
	struct Story {
		std::vector<std::vector<word_t>> facts;
		std::vector<uint> question_fidx;
		std::vector<std::vector<uint>> supporting_facts;
		std::vector<std::vector<word_t>> answers;

        uint size();

		QA<word_t> get(int target_question_idx);
	};

	std::tuple<std::vector<Story<uint>>, utils::Vocab> encode_dataset(
			const std::vector<Story<std::string>>& input);
    std::vector<Story<std::string>> parse_file(const std::string& filename);
}


#endif
