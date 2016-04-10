#include "dali/utils/tsv_utils.h"

#include "dali/utils/core_utils.h"
#include "dali/utils/generator.h"
#include "dali/utils/print_utils.h"

using std::vector;
using std::string;
using std::ifstream;

namespace utils {

    Generator<row_t> generate_tsv_rows(const std::string& fname, const char& delimiter) {
        assert2(file_exists(fname), utils::MS() << "Cannot open tsv file: " << fname);
        if (utils::is_gzip(fname)) {
            auto fpgz = std::make_shared<igzstream>(fname.c_str());
            // igzstream fpgz(fname.c_str());
            return generate_tsv_rows_from_stream(fpgz, delimiter);
        } else {
            auto fp = std::make_shared<ifstream>(fname);
            // ifstream fp(fname);
            return generate_tsv_rows_from_stream(fp,   delimiter);
        }
    }

    template<typename T>
    Generator<row_t> generate_tsv_rows_from_stream(std::shared_ptr<T> stream, const char& delimiter) {
        return utils::Generator<row_t>([stream, delimiter](utils::yield_t<row_t> yield) {
            string::size_type n;
            string l, cell;
            while (std::getline(*stream, l)) {
                std::stringstream ss(l);
                row_t row;
                while(std::getline(ss, cell, delimiter))
                    row.push_back(tokenize(cell));

                if (row.size() > 0) yield(row);
            }
        });
    }

    tokenized_labeled_dataset load_tsv(const string& fname, int expected_columns, const char& delimiter) {
        auto tsv = generate_tsv_rows(fname, delimiter);
        // row by row
        tokenized_labeled_dataset rows;
        int row_number = 0;
        for (auto row : tsv) {
            row_number++;
            if (expected_columns > 0) {
                assert2(
                    row.size() == expected_columns,
                    MS() << "File TSV Row at row "
                         << row_number
                         << " has unexpected number of columns (" << row.size() << ")."
                );
            }
            rows.emplace_back(row);
        }
        return rows;
    }

    template Generator<row_t> generate_tsv_rows_from_stream(std::shared_ptr<igzstream>,         const char& delimiter);
    template Generator<row_t> generate_tsv_rows_from_stream(std::shared_ptr<std::fstream>,      const char& delimiter);
    template Generator<row_t> generate_tsv_rows_from_stream(std::shared_ptr<std::stringstream>, const char& delimiter);
    template Generator<row_t> generate_tsv_rows_from_stream(std::shared_ptr<std::istream>,      const char& delimiter);
}
