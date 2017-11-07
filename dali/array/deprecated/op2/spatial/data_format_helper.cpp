#include "data_format_helper.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"

void check_data_format(const std::string& data_format,
                       int* n_dim, int* c_dim,
                       int* h_dim, int* w_dim) {
    ASSERT2(data_format.size() == 4, utils::make_message("data_format"
        " should be 4 character string containing letters N, C, H and W ("
        "got ", data_format, ")."));
    *n_dim = data_format.find('N');
    ASSERT2(*n_dim != -1, utils::make_message("data_format"
        " should contain character 'N' (got ", data_format, ")."));
    *c_dim = data_format.find('C');
    ASSERT2(*c_dim != -1, utils::make_message("data_format"
        " should contain character 'C' (got ", data_format, ")."));
    *h_dim = data_format.find('H');
    ASSERT2(*h_dim != -1, utils::make_message("data_format"
        " should contain character 'H' (got ", data_format, ")."));
    *w_dim = data_format.find('W');
    ASSERT2(*w_dim != -1, utils::make_message("data_format"
        " should contain character 'W' (got ", data_format, ")."));
}
