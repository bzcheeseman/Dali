
#include <vector>
#include <gtest/gtest.h>

#include "dali/data_processing/Glove.h"
#include "dali/data_processing/Arithmetic.h"
#include "dali/data_processing/NER.h"
#include "dali/data_processing/Paraphrase.h"
#include "dali/data_processing/babi.h"

using std::string;
using std::vector;


TEST(Glove, load) {
    auto embedding = glove::load<double>( STR(DALI_DATA_DIR) "/glove/test_data.txt");
    ASSERT_EQ(std::get<1>(embedding).size(), 21);
    ASSERT_EQ(std::get<0>(embedding).dims(0), 21);
    ASSERT_EQ(std::get<0>(embedding).dims(1), 300);
}

// exposing internal functions from arithmetic for testing.
namespace arithmetic {
    std::tuple<std::vector<int>, std::vector<std::string>> remove_multiplies(const std::vector<int>& numbers, const std::vector<std::string>& ops);
    std::tuple<std::vector<int>, std::vector<std::string>> generate_example(int expression_length, int& min, int& max);
    std::vector<std::string> convert_to_chars(const std::vector<int>& numbers, const std::vector<std::string>& ops);
    int compute_result(const std::vector<int>& numbers, const std::vector<std::string>& ops);
}

TEST(arithmetic, generate) {
    int min = 0;
    int max = 9;
    auto example = arithmetic::generate_example(5, min, max);
    ASSERT_EQ(std::get<0>(example).size(), 3);
    ASSERT_EQ(std::get<1>(example).size(), 2);

    auto example2 = std::make_tuple(
        std::vector<int>({12, 9, 3}),
        std::vector<std::string>({"*", "+"})
    );
    auto demultiplied = arithmetic::remove_multiplies(
        std::get<0>(example2),
        std::get<1>(example2)
    );
    auto example3     = std::make_tuple(
        std::vector<int>({108, 3}),
        std::vector<std::string>({"+"})
    );

    ASSERT_EQ(demultiplied, example3);
    ASSERT_EQ(arithmetic::compute_result(
        std::get<0>(example3),
        std::get<1>(example3)),
        111);
    example2     = std::make_tuple(
        std::vector<int>({12, 9, 3, 5, 5}),
        std::vector<std::string>({"*", "-","+", "*"})
    );
    demultiplied = arithmetic::remove_multiplies(
        std::get<0>(example2),
        std::get<1>(example2)
    );
    example3     = std::make_tuple(
        std::vector<int>({108, 3, 25}),
        std::vector<std::string>({"-", "+"})
    );
    ASSERT_EQ(demultiplied, example3);
    ASSERT_EQ(
        arithmetic::compute_result(
            std::get<0>(example3),
            std::get<1>(example3)
        ),
        130
    );
}

TEST(NER, load) {
    auto NER_data = NER::load( STR(DALI_DATA_DIR) "/Stanford_NER/NER_dummy_dataset.tsv");
    ASSERT_EQ(NER_data.size(), 2);

    auto loader = NER::NER_Loader();
    loader.data_column = 0;
    loader.label_column = -1;
    NER_data = loader.convert_tsv(
        utils::load_tsv(
            STR(DALI_DATA_DIR) "/CoNLL_NER/NER_dummy_dataset.tsv", 4, '\t'
        )
    );
    // how many examples
    ASSERT_EQ(NER_data.size(), 1);
    // how many words in first example
    ASSERT_EQ(NER_data.front().first.size(), 7);
    // last word of first example is period
    ASSERT_EQ(NER_data.front().first.back(), ".");
    // before last word of first example is a location
    ASSERT_EQ(NER_data.front().second[5], "I-LOC");
}

TEST(paraphrase, load) {
    auto paraphrase_data = paraphrase::STS_2015::load_train( STR(DALI_DATA_DIR) "/paraphrase_STS_2015/paraphrase_dummy_data.tsv");
    ASSERT_EQ(paraphrase_data.size(), 4);

    ASSERT_EQ(std::get<2>(paraphrase_data[0]), 1.0);
    ASSERT_EQ(std::get<2>(paraphrase_data[1]), 0.0);
    ASSERT_EQ(std::get<2>(paraphrase_data[2]), 1.0);
    ASSERT_EQ(std::get<2>(paraphrase_data[3]), 0.5);

    auto expected = utils::tokenize(
        "But my bro from the 757 EJ Manuel is the 1st QB gone"
    );

    ASSERT_EQ(std::get<1>(paraphrase_data[0]), expected);

    auto vocab       = paraphrase::get_vocabulary(paraphrase_data, 1);
    auto minibatches = paraphrase::convert_to_indexed_minibatches(vocab, paraphrase_data, 2);

    ASSERT_EQ(minibatches.size(), 2);

    auto char_vocab = utils::CharacterVocab(10, 255);
    auto char_minibatches = paraphrase::convert_to_indexed_minibatches(char_vocab, paraphrase_data, 2);

    ASSERT_EQ(char_minibatches.size(), 2);
    auto sentence = char_vocab.decode(&std::get<0>(char_minibatches[0][0]));
    ASSERT_EQ(sentence, std::get<0>(paraphrase_data[0]));
}


TEST(paraphrase, generator_load) {
    auto gen_data = paraphrase::STS_2015::generate_train(STR(DALI_DATA_DIR) "/paraphrase_STS_2015/paraphrase_dummy_data.tsv");
    int ex_seen = 0;
    for (auto ex : gen_data) ex_seen++;
    ASSERT_EQ(ex_seen, 4);
    // TODO: find why when gen_data is overwritten a segfault occurs!!
    auto gen_data2 = paraphrase::STS_2015::generate_train(STR(DALI_DATA_DIR) "/paraphrase_STS_2015/paraphrase_dummy_data.tsv");
    auto paraphrase_data = paraphrase::STS_2015::load_train( STR(DALI_DATA_DIR) "/paraphrase_STS_2015/paraphrase_dummy_data.tsv");
    auto vocab_gen = paraphrase::get_vocabulary(gen_data2, 1);
    auto vocab     = paraphrase::get_vocabulary(paraphrase_data, 1);
    ASSERT_EQ(vocab_gen, vocab);
}

TEST(paraphrase, convert_to_indexed_minibatches) {
    auto gen_data = paraphrase::STS_2015::generate_train(STR(DALI_DATA_DIR) "/paraphrase_STS_2015/paraphrase_dummy_data.tsv");
    auto vocab_gen = utils::Vocab(paraphrase::get_vocabulary(gen_data, 1));

    // deadlock when generator inside generator is run.
    auto gen_minibatches = paraphrase::convert_to_indexed_minibatches(vocab_gen, gen_data, 3);

    vector<vector<paraphrase::numeric_example_t>> minibatches;
    for (auto minibatch : gen_minibatches) {
        minibatches.emplace_back(minibatch);
    }

    ASSERT_EQ(minibatches.size(), 2);

}

template<typename T>
::testing::AssertionResult VECTORS_EQUAL(const vector<T>& a,
                                         const vector<T>& b) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure() << "Vectors of different length "
                << "first is " << a.size() << ", but second is " << b.size() << ".";
    }

    for (int idx = 0; idx < a.size(); ++idx) {
        if (a[idx] != b[idx]) {
            return ::testing::AssertionFailure() << "Vectors differ at "
                    << idx << "-th index: first is " << a[idx]
                    << ", but second is " << b[idx] << ".";
        }
    }
    return ::testing::AssertionSuccess();
}


TEST(babi, parse) {
    auto test_file = utils::dir_join({ STR(DALI_DATA_DIR),
                                      "tests",
                                      "babi.sample" });

    auto datasets = babi::parse_file(test_file);
    ASSERT_EQ(datasets.size(), 2);

    vector<vector<string>> facts[2] = {
        {
            {"The", "office", "is", "east", "of", "the", "hallway", "."},
            {"The", "kitchen", "is", "north", "of", "the", "office", "."},
            {"The", "garden", "is", "west", "of", "the", "bedroom", "."},
            {"The", "office", "is", "west", "of", "the", "garden", "."},
            {"The", "bathroom", "is", "north", "of", "the", "garden", "."},
            {"How", "do", "you", "go", "from", "the", "kitchen", "to", "the", "garden", "?"}
        },
        {
            {"This", "morning", "Mary", "moved", "to", "the", "kitchen", "."},
            {"This", "afternoon", "Mary", "moved", "to", "the", "cinema", "."},
            {"Yesterday", "Bill", "went", "to", "the", "bedroom", "."},
            {"Yesterday", "Mary", "journeyed", "to", "the", "school", "."},
            {"Where", "was", "Mary", "before", "the", "cinema", "?"},
            {"Yesterday", "Fred", "went", "back", "to", "the", "cinema", "."},
            {"Bill", "journeyed", "to", "the", "office", "this", "morning", "."},
            {"Where", "was", "Bill", "before", "the", "office", "?"},
            {"Mary", "went", "to", "the", "school", "this", "evening", "."},
            {"This", "afternoon", "Bill", "journeyed", "to", "the", "kitchen", "."},
            {"Where", "was", "Bill", "before", "the", "office", "?"}
        }
    };
    vector<uint> question_fidx[2] = {
        { 5 },
        { 4, 7, 10 },
    };
    vector<vector<uint>> supporting_facts[2] = {
        {
            { 1, 3 }
        },
        {
            { 1, 0 },
            { 6, 2 },
            { 6, 2 }
        },
    };
    vector<vector<string>> answers[2] = {
        {
            {"s", "e"}
        },
        {
            {"kitchen"},
            {"bedroom"},
            {"bedroom"}
        }
    };

    vector<string> scope = { "dataset1", "dataset2"};
    for (int didx = 0; didx < 2; ++didx) {
        SCOPED_TRACE(scope[didx]);
        auto& dataset = datasets[didx];
        auto& expected_facts            = facts[didx];
        auto& expected_question_fidx    = question_fidx[didx];
        auto& expected_supporting_facts = supporting_facts[didx];
        auto& expected_answers          = answers[didx];

        ASSERT_EQ(dataset.facts.size(), expected_facts.size());
        for (int fidx = 0; fidx < expected_facts.size(); ++fidx) {
            EXPECT_TRUE(VECTORS_EQUAL(dataset.facts[fidx], expected_facts[fidx]));
        };

        EXPECT_TRUE(VECTORS_EQUAL(dataset.question_fidx, expected_question_fidx));

        ASSERT_EQ(dataset.supporting_facts.size(), expected_supporting_facts.size());
        for (int sidx = 0; sidx < dataset.supporting_facts.size(); ++sidx) {
            EXPECT_TRUE(VECTORS_EQUAL(dataset.supporting_facts[sidx],
                                      expected_supporting_facts[sidx]));
        }

        ASSERT_EQ(dataset.answers.size(), expected_answers.size());
        for (int aidx = 0; aidx < expected_answers.size(); ++aidx) {
            EXPECT_TRUE(VECTORS_EQUAL(dataset.answers[aidx], expected_answers[aidx]));
        };
    };
};

TEST(babi, extract_qa) {
    auto test_file = utils::dir_join({ STR(DALI_DATA_DIR),
                                      "tests",
                                      "babi.sample" });
    babi::QA<string> qa1;
    qa1.facts = {
        {"The", "office", "is", "east", "of", "the", "hallway", "."},
        {"The", "kitchen", "is", "north", "of", "the", "office", "."},
        {"The", "garden", "is", "west", "of", "the", "bedroom", "."},
        {"The", "office", "is", "west", "of", "the", "garden", "."},
        {"The", "bathroom", "is", "north", "of", "the", "garden", "."},
    };
    qa1.question = {"How", "do", "you", "go", "from", "the", "kitchen", "to", "the", "garden", "?"};
    qa1.answer = {"s", "e"};
    qa1.supporting_facts = { 1, 3 };


    babi::QA<string> qa2;
    qa2.facts = {
        {"This", "morning", "Mary", "moved", "to", "the", "kitchen", "."},
        {"This", "afternoon", "Mary", "moved", "to", "the", "cinema", "."},
        {"Yesterday", "Bill", "went", "to", "the", "bedroom", "."},
        {"Yesterday", "Mary", "journeyed", "to", "the", "school", "."},
    };
    qa2.question = {"Where", "was", "Mary", "before", "the", "cinema", "?"};
    qa2.answer = { "kitchen" };
    qa2.supporting_facts = { 1, 0 };

    babi::QA<string> qa3;
    qa3.facts = {
        {"This", "morning", "Mary", "moved", "to", "the", "kitchen", "."},
        {"This", "afternoon", "Mary", "moved", "to", "the", "cinema", "."},
        {"Yesterday", "Bill", "went", "to", "the", "bedroom", "."},
        {"Yesterday", "Mary", "journeyed", "to", "the", "school", "."},
        {"Yesterday", "Fred", "went", "back", "to", "the", "cinema", "."},
        {"Bill", "journeyed", "to", "the", "office", "this", "morning", "."},
    };
    qa3.question = {"Where", "was", "Bill", "before", "the", "office", "?"};
    qa3.answer = { "bedroom" };
    qa3.supporting_facts = { 5, 2 };


    babi::QA<string> qa4;
    qa4.facts = {
        {"This", "morning", "Mary", "moved", "to", "the", "kitchen", "."},
        {"This", "afternoon", "Mary", "moved", "to", "the", "cinema", "."},
        {"Yesterday", "Bill", "went", "to", "the", "bedroom", "."},
        {"Yesterday", "Mary", "journeyed", "to", "the", "school", "."},

        {"Yesterday", "Fred", "went", "back", "to", "the", "cinema", "."},
        {"Bill", "journeyed", "to", "the", "office", "this", "morning", "."},

        {"Mary", "went", "to", "the", "school", "this", "evening", "."},
        {"This", "afternoon", "Bill", "journeyed", "to", "the", "kitchen", "."},
    };
    qa4.question = {"Where", "was", "Bill", "before", "the", "office", "?"};
    qa4.answer = { "bedroom" };
    qa4.supporting_facts = { 5, 2 };

    auto compare_qa = [](babi::QA<string> given, babi::QA<string> expected) {
        ASSERT_EQ(given.facts.size(), expected.facts.size());
        for (int fidx = 0; fidx < given.facts.size(); ++fidx) {
            EXPECT_TRUE(VECTORS_EQUAL(given.facts[fidx], expected.facts[fidx]));
        };

        EXPECT_TRUE(VECTORS_EQUAL(given.question, expected.question));

        EXPECT_TRUE(VECTORS_EQUAL(given.supporting_facts, expected.supporting_facts));

        EXPECT_TRUE(VECTORS_EQUAL(given.answer, expected.answer));
    };

    auto datasets = babi::parse_file(test_file);
    ASSERT_EQ(datasets.size(), 2);
    ASSERT_EQ(datasets[0].size(), 1);
    ASSERT_EQ(datasets[1].size(), 3);

    {
        SCOPED_TRACE("Story 1 - question 1");
        compare_qa(datasets[0].get(0), qa1);
    }
    {
        SCOPED_TRACE("Story 2 - question 1");
        compare_qa(datasets[1].get(0), qa2);
    }
    {
        SCOPED_TRACE("Story 2 - question 2");
        compare_qa(datasets[1].get(1), qa3);
    }
    {
        SCOPED_TRACE("Story 2 - question 3");
        compare_qa(datasets[1].get(2), qa4);
    }
}

TEST(babi, encode) {
    auto test_file = utils::dir_join({ STR(DALI_DATA_DIR),
                                      "tests",
                                      "babi.sample" });
    auto datasets = babi::parse_file(test_file);

    utils::Vocab vocab;
    vector<babi::Story<uint>> encoded_datasets;

    encoded_datasets = babi::encode_dataset(datasets, &vocab);

    ASSERT_EQ(datasets.size(), encoded_datasets.size());
    for (int i = 0; i < datasets.size(); ++i) {
        auto& dataset         = datasets[i];
        auto& encoded_dataset = encoded_datasets[i];
        ASSERT_EQ(dataset.facts.size(), encoded_dataset.facts.size());
        for (int fidx = 0; fidx < dataset.facts.size(); ++fidx) {
            auto decoded_fact = vocab.decode(&encoded_dataset.facts[fidx]);
            EXPECT_TRUE(VECTORS_EQUAL(decoded_fact, dataset.facts[fidx]));
        };

        EXPECT_TRUE(VECTORS_EQUAL(dataset.question_fidx, encoded_dataset.question_fidx));

        ASSERT_EQ(dataset.supporting_facts.size(), encoded_dataset.supporting_facts.size());
        for (int sidx = 0; sidx < dataset.supporting_facts.size(); ++sidx) {
            EXPECT_TRUE(VECTORS_EQUAL(dataset.supporting_facts[sidx],
                                      encoded_dataset.supporting_facts[sidx]));
        }

        ASSERT_EQ(dataset.answers.size(), encoded_dataset.answers.size());
        for (int aidx = 0; aidx < dataset.answers.size(); ++aidx) {
            auto decoded_answer = vocab.decode(&encoded_dataset.answers[aidx]);
            EXPECT_TRUE(VECTORS_EQUAL(dataset.answers[aidx], decoded_answer));
        };
    }

}

