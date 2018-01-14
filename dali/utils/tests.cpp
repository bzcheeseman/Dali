#include <chrono>
#include <vector>
#include <memory>
#include <gtest/gtest.h>
#include <sstream>
#include <cstdio>
#include <string>

#include "dali/config.h"
#include "dali/utils/core_utils.h"
#include "dali/utils/performance_report.h"
#include "dali/utils/scope.h"
#include "dali/utils/ThreadPool.h"
#include "dali/utils/xml_cleaner.h"
#include "dali/utils/vocab.h"

using std::chrono::milliseconds;
using std::make_shared;
using std::string;
using std::stringstream;
using std::vector;


TEST(utils, performance_report) {
    PerformanceReport report;
    report.start_capture();
    {
        auto s1 = Scope(std::make_shared<std::string>("yay1"));
        {
            auto s2 = Scope(std::make_shared<std::string>("siema1"));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        {
            auto s3 = Scope(std::make_shared<std::string>("siemasi2"));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    report.stop_capture();
    // report.print();
}


TEST(ThreadPool, wait_until_idle) {
    const int NUM_THREADS = 10;
    const int NUM_JOBS = 100;
    ThreadPool t(NUM_THREADS);

    bool barrier = true;

    for(int j=0; j<NUM_JOBS; ++j) {
        t.run([&barrier]() {
            while(barrier) {
                std::this_thread::yield();
            };
        });
    }

    // wait for threads to pick up work.
    while(t.active_workers() < NUM_THREADS);

    // Threads are currently waiting for barrier.
    // Ensure that wait until idle returns false..
    ASSERT_FALSE(t.wait_until_idle(milliseconds(1)));
    // Remove barrier and hope they come back.
    barrier = false;

    // Assert all threads will be done exentually.
    ASSERT_TRUE(t.wait_until_idle());
}

TEST(ThreadPool, thread_number) {
    const int NUM_THREADS = 4;
    const int JOBS_PER_ATTEMPT = 10;
    ThreadPool pool(NUM_THREADS);


    for(int t = 0; t < NUM_THREADS; ++t) {
        // Try to get thread t to manifest itself by checking
        // it's thread_number.
        bool thread_t_seen = false;
        while (!thread_t_seen) {
            for(int job = 0; job < JOBS_PER_ATTEMPT; ++job) {
                pool.run([&t, &thread_t_seen]() {
                    for (int i=0; i<10000; ++i) {
                        if(t == ThreadPool::get_thread_number()) {
                            thread_t_seen = true;
                        }
                    }
                });
            }
            pool.wait_until_idle();
        }
    }
}

TEST(utils, stream_to_redirection_list) {
    stringstream ss(
        "hello->world\n"
        "what->happened\n"
        "who->is this?\n");

    std::unordered_map<string, string> mapping;
    utils::stream_to_redirection_list(ss, mapping);
    ASSERT_TRUE(mapping.find("hello") != mapping.end());
    ASSERT_TRUE(mapping.find("what") != mapping.end());
    ASSERT_TRUE(mapping.find("who") != mapping.end());
    ASSERT_EQ(mapping.at("hello"), "world");
    ASSERT_EQ(mapping.at("what"), "happened");
    ASSERT_EQ(mapping.at("who"), "is this?");
}

TEST(utils, split_str) {
    string input = "hellobobmeisterhowareyou?";

    auto tokens = utils::split_str(input, "bobmeister");

    ASSERT_EQ(tokens.size(), 2);
    ASSERT_EQ(tokens[0], "hello");
    ASSERT_EQ(tokens[1], "howareyou?");

    string dashed_input = "Category:Plantain-eaters->Western plantain-eater";

    tokens = utils::split_str(dashed_input, "->");
    ASSERT_EQ(tokens[0], "Category:Plantain-eaters");
    ASSERT_EQ(tokens[1], "Western plantain-eater");
    ASSERT_EQ(tokens.size(), 2);
}

TEST(utils, trim) {
    string input = "     hello_world thus ?     ";
    auto trimmed = utils::trim(input);
    ASSERT_EQ(trimmed, "hello_world thus ?");
}

TEST(xml_cleaner, preprocess) {
    // TODO add fail conditions for this test
    string input = "{{Other uses}}\n"
    "{{Redirect|Foo|FOO (Forward Observation Officer)|Artillery observer}}\n"
    "\n"
    "The terms '''foobar''' ({{IPAc-en|ˈ|f|uː|b|ɑr}}), '''fubar''', or "
    "'''foo''', '''bar''', '''baz''' and '''qux''' (alternatively, '''"
    "quux''') are sometimes used as [[placeholder name]]s (also referred"
    " to as [[metasyntactic variable]]s) in [[computer programming]] or "
    "computer-related documentation.<ref name=\"rfc3092\" /> They have been "
    "used to name entities such as [[Variable (computer science)|variable]]s,"
    " [[Function (computer science)|functions]], and [[command (computing)|command]]s"
    " whose purpose is unimportant and serve only to demonstrate a concept."
    "  The words themselves have no meaning in this usage.  ''Foobar'' is "
    "sometimes used alone; ''foo'', ''bar'', and ''baz'' are sometimes used"
    " in that order, when multiple entities are needed.\n"
    "\n"
    "The usage in [[computer programming]] examples and [[pseudocode]] "
    "varies; in certain circles, it is used extensively, but many prefer "
    "descriptive names, while others prefer to use single letters.  "
    "[[Eric S. Raymond]] has called it an \"important hackerism\" "
    "alongside [[kludge]] and [[cruft]].<ref name=\"Raymond\">{{cite "
    "book|url=http://books.google.com/books?id=g80P_4v4QbIC&amp;lpg=PP1"
    "&amp;pg=PA5|title=The New Hacker's Dictionary|author=[[Eric S. "
    "Raymond]]|publisher=[[MIT Press]]|year=1996|isbn=0-262-68092-0}}</ref>Okay\n";

    auto cleaned = utils::xml_cleaner::process_text_keeping_brackets(input);
}

TEST(utils, CharacterVocab) {
    auto seq      = vector<string>{"bob", "ate", "an", "apple"};
    auto vocab    = utils::CharacterVocab(0, 255);
    auto chars    = vocab.encode(seq);
    auto seq_size = utils::join(seq, " ").size();
    ASSERT_EQ(chars.size(), seq_size);
    ASSERT_EQ(seq, vocab.decode(chars));


    auto char_decoded_seq = vocab.decode_characters(chars);
    ASSERT_EQ(char_decoded_seq.size(), seq_size);
    ASSERT_EQ(utils::join(seq, " "), utils::join(char_decoded_seq));


    // space char is char 32, if we start at 33 we lose it, and
    // spaces get replaced by "█":
    auto spaceless_vocab = utils::CharacterVocab(33, 255);
    auto spaceless_chars = spaceless_vocab.encode(seq);
    ASSERT_NE(seq, spaceless_vocab.decode(spaceless_chars));
    auto special_seq = utils::join(seq, "\xFF");
    ASSERT_EQ(special_seq, utils::join(spaceless_vocab.decode(spaceless_chars)));
}

TEST(utils, dir_join) {
    auto joined = utils::dir_join({"", "hey", "yo"});
    ASSERT_EQ("hey/yo", joined);
    auto joined2 = utils::dir_join({"/", "hey/", "yo"});
    ASSERT_EQ("/hey/yo", joined2);
}

TEST(utils, prefix_match) {
    using utils::prefix_match;
    vector<string> candidates = {
        "siema",
        "lol",
        "we_hit_a_wall",
    };
    // candidates match with themselvers
    for (auto& candidate : candidates) {
        ASSERT_EQ(candidate, prefix_match(candidates, candidate));
    }

    EXPECT_THROW(prefix_match(candidates, "low"), std::runtime_error);
    EXPECT_THROW(prefix_match(candidates, "lol2"), std::runtime_error);

    EXPECT_EQ(prefix_match(candidates, ""), "siema");
    EXPECT_EQ(prefix_match(candidates, "lo"), "lol");
    EXPECT_EQ(prefix_match(candidates, "we_hit"), "we_hit_a_wall");
}
