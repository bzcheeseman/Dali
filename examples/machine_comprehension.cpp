#include <gflags/gflags.h>
#include <tuple>
#include <vector>
#include <string>
#include <functional>

#include "dali/core.h"
#include "dali/data_processing/machine_comprehension.h"
#include "dali/data_processing/Glove.h"
#include "dali/utils.h"
#include "dali/models/QuestionAnswering.h"

using mc::Section;
using mc::Question;
using std::vector;
using std::string;
using utils::Vocab;
using utils::random_minibatches;
using utils::vsum;
using utils::assert2;
using std::function;

vector<Section> train_data, validate_data, test_data;
std::shared_ptr<Vocab> vocab;
ThreadPool* pool;

DEFINE_int32(j,                  1,     "How many threads should be used ?");
DEFINE_int32(minibatch, 50,             "Number of sections considered in every minibatch gradient step.");
DEFINE_string(pretrained_vectors, "",   "Path to pretrained word vectors (Glove etc.)?");
DEFINE_double(validation_fraction, 0.1, "How much of training set to use for validation.");

typedef GatedLstmsModel<double> model_t;

double calculate_accuracy(model_t& model, const vector<Section>& data) {
    std::atomic<int> correct(0);
    std::atomic<int> total(0);

    ReportProgress<double> journalist("Computing accuracy", data.size()*4);

    for (int sidx = 0; sidx < data.size(); ++sidx) {
        pool->run([sidx, &data, &model, &correct, &total, &journalist]() {
            graph::NoBackprop nb;
            auto& section = data[sidx];
            for (auto& question: section.questions) {
                    int ans = model.predict(section.text, question.text, question.answers);
                    if (ans == question.correct_answer) ++correct;
                    ++total;
                    journalist.tick(total);
            }
        });
    }
    pool->wait_until_idle();
    journalist.done();
    return (double)correct/total;
}

void init() {
    // Load the data set
    std::tie(train_data, test_data) = mc::load();
    // shuffle examples
    std::random_shuffle(train_data.begin(), train_data.end());
    // separate validation dataset
    int num_validation = train_data.size() * FLAGS_validation_fraction;
    validate_data      = vector<Section>(
        train_data.begin(),
        train_data.begin() + num_validation
    );
    train_data.erase(
        train_data.begin(),
        train_data.begin() + num_validation
    );
    // extract vocabulary
    // only consider common words.
    vector<vector<string>> wrapper;
    wrapper.emplace_back(mc::extract_vocabulary(train_data));
    auto index2word = utils::get_vocabulary(wrapper, 2);
    vocab = std::make_shared<Vocab>(index2word);

    std::cout << "Datasets : " << "train ("    << train_data.size()    << " items), "
                               << "validate (" << validate_data.size() << " items), "
                               << "test ("     << test_data.size()     << " items)" << std::endl;
    std::cout << "vocabulary size : " << vocab->word2index.size() << std::endl;

}

int main(int argc, char** argv) {
    sane_crashes::activate();

    GFLAGS_NAMESPACE::SetUsageMessage(
        "\nMicrosoft Machine Comprehension Task"
    );

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    init();

    pool = new ThreadPool(FLAGS_j);

    auto training = LSTV(0.9, 0.3, 3);

    model_t model(vocab);

    auto params = model.parameters();
    auto solver = Solver::AdaDelta<double>(params);

    vector<model_t> thread_models;

    for (int tmidx = 0; tmidx < FLAGS_j; tmidx++)
        thread_models.emplace_back(model.shallow_copy());

    utils::ThreadAverage thread_error(FLAGS_j);

    model_t best_model(model, false, false);
    float best_accuracy = 0.0;

    while(true) {
        thread_error.reset();

        auto minibatches = random_minibatches(train_data.size(), FLAGS_minibatch);

        ReportProgress<double> journalist("Training", train_data.size());
        std::atomic<int> processed_sections(0);

        for (int bidx = 0; bidx < minibatches.size(); ++bidx) {
            pool->run([bidx, &minibatches, &thread_models, &solver, &thread_error,
                       &processed_sections, &journalist]() {
                auto& batch = minibatches[bidx];
                model_t& thread_model = thread_models[ThreadPool::get_thread_number()];

                double partial_error = 0.0;
                int partial_error_updates = 0;
                for (auto& example_idx: batch) {
                    Section& section = train_data[example_idx];
                    for (auto& question: section.questions) {

                        auto e = thread_model.error(section.text,
                                                    question.text,
                                                    question.answers,
                                                    question.correct_answer);

                        partial_error += e.w(0,0);
                        partial_error_updates += 1;
                        e.grad();
                        graph::backward();
                    }
                    processed_sections += 1;

                    journalist.tick(processed_sections, thread_error.average());

                }

                auto params = thread_model.parameters();

                solver.step(params);
                thread_error.update(partial_error / (double)partial_error_updates);

            });
        }
        pool->wait_until_idle();
        journalist.done();
        double val_acc = 100.0 * calculate_accuracy(model, validate_data);
        std::cout << "Training error = " << thread_error.average()
                  << ", Validation accuracy = " << val_acc << "%" << std::endl;

        if (val_acc > best_accuracy) {
            std::cout << "NEW WORLD RECORD!" << std::endl;
            best_accuracy = val_acc;
            best_model = model_t(model, true, true);
        }

        if (training.should_stop(100.0-val_acc)) break;
    }
    double final_val_acc = 100.0 * calculate_accuracy(best_model, validate_data);
    std::cout << "Final validation accuracy = " << final_val_acc << "%" << std::endl;

    double test_acc = 100.0 * calculate_accuracy(best_model, test_data);
    std::cout << "Test accuracy = " << test_acc << "%" << std::endl;
}
