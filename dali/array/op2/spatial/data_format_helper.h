#ifndef DALI_ARRAY_OP2_DATA_FORMAT_HELPER_H
#define DALI_ARRAY_OP2_DATA_FORMAT_HELPER_H

#include <string>

void check_data_format(const std::string& data_format,
					   int* n_dim,
					   int* c_dim,
					   int* h_dim,
					   int* w_dim);

#endif  // DALI_ARRAY_OP2_DATA_FORMAT_HELPER_H
