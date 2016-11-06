#include "nervana_gemm.h"

#include "dali/config.h"

#ifdef DALI_USE_CUDA

#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/expression/array_wrapper.h"
#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"
#include "dali/utils/core_utils.h"
#include <unordered_map>
#include <string>
#include <cuda.h>
#include <fstream>
#include <mutex>

namespace {
    void check_cuda_status(cudaError_t err, const std::string& msg) {
        ASSERT2(err == cudaSuccess, utils::make_message("could not ",
            msg, " (error = ", cudaGetErrorString(err), ")"));
    }

    const char* cuGetErrorString(CUresult result) {
        switch (result) {
            case CUDA_SUCCESS:                              return "No errors";
            case CUDA_ERROR_INVALID_VALUE:                  return "Invalid value";
            case CUDA_ERROR_OUT_OF_MEMORY:                  return "Out of memory";
            case CUDA_ERROR_NOT_INITIALIZED:                return "Driver not initialized";
            case CUDA_ERROR_DEINITIALIZED:                  return "Driver deinitialized";
            case CUDA_ERROR_PROFILER_DISABLED:              return "Profiler disabled";
            case CUDA_ERROR_PROFILER_NOT_INITIALIZED:       return "Profiler not initialized";
            case CUDA_ERROR_PROFILER_ALREADY_STARTED:       return "Profiler already started";
            case CUDA_ERROR_PROFILER_ALREADY_STOPPED:       return "Profiler already stopped";
            case CUDA_ERROR_NO_DEVICE:                      return "No CUDA-capable device available";
            case CUDA_ERROR_INVALID_DEVICE:                 return "Invalid device";
            case CUDA_ERROR_INVALID_IMAGE:                  return "Invalid kernel image";
            case CUDA_ERROR_INVALID_CONTEXT:                return "Invalid context";
            case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:        return "Context already current";
            case CUDA_ERROR_MAP_FAILED:                     return "Map failed";
            case CUDA_ERROR_UNMAP_FAILED:                   return "Unmap failed";
            case CUDA_ERROR_ARRAY_IS_MAPPED:                return "Array is mapped";
            case CUDA_ERROR_ALREADY_MAPPED:                 return "Already mapped";
            case CUDA_ERROR_NO_BINARY_FOR_GPU:              return "No binary for GPU";
            case CUDA_ERROR_ALREADY_ACQUIRED:               return "Already acquired";
            case CUDA_ERROR_NOT_MAPPED:                     return "Not mapped";
            case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:            return "Not mapped as array";
            case CUDA_ERROR_NOT_MAPPED_AS_POINTER:          return "Not mapped as pointer";
            case CUDA_ERROR_ECC_UNCORRECTABLE:              return "Uncorrectable ECC error";
            case CUDA_ERROR_UNSUPPORTED_LIMIT:              return "Unsupported CUlimit";
            case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:         return "Context already in use";
            case CUDA_ERROR_INVALID_SOURCE:                 return "Invalid source";
            case CUDA_ERROR_FILE_NOT_FOUND:                 return "File not found";
            case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "Shared object symbol not found";
            case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:      return "Shared object initialization failed";
            case CUDA_ERROR_OPERATING_SYSTEM:               return "Operating System call failed";
            case CUDA_ERROR_INVALID_HANDLE:                 return "Invalid handle";
            case CUDA_ERROR_NOT_FOUND:                      return "Not found";
            case CUDA_ERROR_NOT_READY:                      return "CUDA not ready";
            case CUDA_ERROR_LAUNCH_FAILED:                  return "Launch failed";
            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:        return "Launch exceeded resources";
            case CUDA_ERROR_LAUNCH_TIMEOUT:                 return "Launch exceeded timeout";
            case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:  return "Launch with incompatible texturing";
            case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:    return "Peer access already enabled";
            case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:        return "Peer access not enabled";
            case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:         return "Primary context active";
            case CUDA_ERROR_CONTEXT_IS_DESTROYED:           return "Context is destroyed";
            case CUDA_ERROR_ASSERT:                         return "Device assert failed";
            case CUDA_ERROR_TOO_MANY_PEERS:                 return "Too many peers";
            case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: return "Host memory already registered";
            case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:     return "Host memory not registered";
            case CUDA_ERROR_UNKNOWN:                        return "Unknown error";
            default:                                        return "Unknown error code";
        }
    }

    void check_cuda_status(CUresult err, const std::string& msg) {
        cuGetErrorString(err);
        ASSERT2(err == CUDA_SUCCESS, utils::make_message("could not ",
            msg, " (error = ", cuGetErrorString(err), ")"));
    }

    void write_file(const std::string& contents, const std::string& fname) {
        std::ofstream out(fname.c_str(), std::ofstream::out);
        ASSERT2(!out.bad(), utils::make_message("cannot open ", fname, "."));
        // add header to code (and extern c to avoid name mangling)
        out << contents;
        out.flush();
        out.close();
    }

    std::tuple<int, int, int> get_grid_dimensions(int grid, int m, int n, int sm_count, const std::string& trans) {
        int sizeA, sizeB, threads;
        if (grid >= 0) {
            if (grid == 0) {
                sizeA = 32;
                sizeB = 128;
                threads = 128;
            } else if (grid == 1) {
                sizeA = 128;
                sizeB = 32;
                threads = 128;
            } else if (grid == 2) {
                sizeA = 128;
                sizeB = 64;
                threads = 128;
            } else if (grid == 3) {
                sizeA = 128;
                sizeB = 128;
                threads = 256;
            }
        } else {
            int sh = std::min(m, n);

            int size;
            if (sh < 384 - 16) {
                int sh128 = sh % 128;
                if (sh128 > 0 && sh128 < 112) {
                    if (sh128 > 48 && sh128 <= 64) {
                        int sh64 = sh / 64;
                        int wide = std::max(m, n);
                        sh64 *= (wide / 128 + (wide % 128 != 0)) / sm_count;
                        if (sh64 > 1) {
                            size = 64;
                        }
                        else {
                            size = 32;
                        }
                    }
                    else {
                        size = 32;
                    }
                }
                else {
                    size = 128;
                }
            } else {
                size = 128;
            }

            if (m >= n) {
                if (trans == "nt") {
                    size = 128;
                }
                sizeA = 128;
                sizeB = size;
            } else {
                if (trans == "tn") {
                    size = 128;
                } else if (size == 64) {
                    //temporary until kernels exist
                    size = 32;
                }
                sizeA = size;
                sizeB = 128;
            }
            threads = (sizeA == 128 && sizeB == 128) ? 256 : 128;
        }

        return std::make_tuple(sizeA, sizeB, threads);
    }
}

struct SassKernel {
    int threads_;
    std::string sass_;
    std::string params_;
    int share_;

    SassKernel(int threads,
               const std::string& sass,
               const std::string& params,
               int share) :
        threads_(threads), sass_(sass), params_(params), share_(share) {}
};

std::unordered_map<std::string, SassKernel> kernels = {
    {"sgemm_nn_128x128", SassKernel(/*threads=*/ 256, /*sass=*/"sgemm_nn_128x128", /*params=*/ "gemm", /*share=*/ 128*8*2 + 128*8*2 + 4)},
    {"sgemm_nt_128x128", SassKernel(/*threads=*/ 256, /*sass=*/"sgemm_nt_128x128", /*params=*/ "gemm", /*share=*/ 128*8*2 + 128*8*2 + 4)},
    {"sgemm_tn_128x128", SassKernel(/*threads=*/ 256, /*sass=*/"sgemm_tn_128x128", /*params=*/ "gemm", /*share=*/ 128*8*2 + 128*8*2 + 4)},
    {"hgemm_nn_128x128", SassKernel(/*threads=*/ 256, /*sass=*/"hgemm_nn_128x128", /*params=*/ "gemm", /*share=*/ 128*8*2 + 128*8*2 + 4)},
    {"hgemm_nt_128x128", SassKernel(/*threads=*/ 256, /*sass=*/"hgemm_nt_128x128", /*params=*/ "gemm", /*share=*/ 128*8*2 + 128*8*2 + 4)},
    {"hgemm_tn_128x128", SassKernel(/*threads=*/ 256, /*sass=*/"hgemm_tn_128x128", /*params=*/ "gemm", /*share=*/ 128*8*2 + 128*8*2 + 4)},

    {"sgemm_nn_128x64", SassKernel(/*threads=*/ 128, /*sass=*/"sgemm_nn_128x64", /*params=*/ "gemm", /*share=*/ 128*8*2 +  64*8*2 + 4)},
    {"sgemm_tn_128x64", SassKernel(/*threads=*/ 128, /*sass=*/"sgemm_tn_128x64", /*params=*/ "gemm", /*share=*/ 128*8*2 +  64*8*2 + 4)},
    {"hgemm_nn_128x64", SassKernel(/*threads=*/ 128, /*sass=*/"hgemm_nn_128x64", /*params=*/ "gemm", /*share=*/ 128*8*2 +  64*8*2 + 4)},
    {"hgemm_tn_128x64", SassKernel(/*threads=*/ 128, /*sass=*/"hgemm_tn_128x64", /*params=*/ "gemm", /*share=*/ 128*8*2 +  64*8*2 + 4)},

    {"sgemm_nn_128x32", SassKernel(/*threads=*/ 128, /*sass=*/ "sgemm_nn_128x32", /*params=*/ "gemm", /*share=*/ (128*16 + 32)*2 + 32*16*2 + 4)},
    {"sgemm_tn_128x32", SassKernel(/*threads=*/ 128, /*sass=*/ "sgemm_tn_128x32", /*params=*/ "gemm", /*share=*/ (128*16 +  0)*2 + 32*16*2 + 4)},
    {"hgemm_nn_128x32", SassKernel(/*threads=*/ 128, /*sass=*/ "hgemm_nn_128x32", /*params=*/ "gemm", /*share=*/ (128*16 + 32)*2 + 32*16*2 + 4)},
    {"hgemm_tn_128x32", SassKernel(/*threads=*/ 128, /*sass=*/ "hgemm_tn_128x32", /*params=*/ "gemm", /*share=*/ (128*16 +  0)*2 + 32*16*2 + 4)},

    {"sgemm_nn_32x128", SassKernel(/*threads=*/ 128, /*sass=*/ "sgemm_nn_32x128", /*params=*/"gemm", /*share=*/ (32*16 + 32)*2 + (128*16 +  0)*2 + 4)},
    {"sgemm_nt_32x128", SassKernel(/*threads=*/ 128, /*sass=*/ "sgemm_nt_32x128", /*params=*/"gemm", /*share=*/ (32*16 + 32)*2 + (128*16 + 32)*2 + 4)},
    {"hgemm_nn_32x128", SassKernel(/*threads=*/ 128, /*sass=*/ "hgemm_nn_32x128", /*params=*/"gemm", /*share=*/ (32*16 + 32)*2 + (128*16 +  0)*2 + 4)},
    {"hgemm_nt_32x128", SassKernel(/*threads=*/ 128, /*sass=*/ "hgemm_nt_32x128", /*params=*/"gemm", /*share=*/ (32*16 + 32)*2 + (128*16 + 32)*2 + 4)},

    {"hgemm_nt_32x32", SassKernel(/*threads=*/ 128, /*sass=*/ "hgemm_nt_32x32", /*params=*/"gemm", /*share=*/ 32*65*4 + 4)},
    {"hgemm_nt_16x64", SassKernel(/*threads=*/ 128, /*sass=*/ "hgemm_nt_16x64", /*params=*/"gemm", /*share=*/ (16*64 + 32)*2 + (64*64 + 32)*2 + 4)},
    {"hgemm_nn_32x64", SassKernel(/*threads=*/ 128, /*sass=*/ "hgemm_nn_32x64", /*params=*/"gemm", /*share=*/ 32*33*2 + 64*32*2 + 2048)},// artificially limit occpancy
    {"hgemm_nn_16x64", SassKernel(/*threads=*/ 128, /*sass=*/ "hgemm_nn_16x64", /*params=*/"gemm", /*share=*/ (16*64 + 32)*2 + 64*64*2 + 4)}
};

std::string kSassDir = STR(DALI_COMPILE_DIRECTORY) "/nervana_kernels/sass/";
std::string kMaxasDir = STR(DALI_COMPILE_DIRECTORY) "/nervana_kernels/maxas/";
std::string kCubinDir = STR(DALI_COMPILE_DIRECTORY) "/nervana_kernels/cubin";
#ifdef DALI_CACHE_DIR
std::string kPtxDir    = utils::expanduser(STR(DALI_CACHE_DIR)) + "nervana_kernels/ptx/";
#else
std::string kPtxDir    = utils::expanduser("~/.dali/cache/nervana_kernels/ptx/");
#endif

struct LoadedSassKernel {
    CUmodule module_;
    CUfunction function_;

    LoadedSassKernel(const std::string cubin_file, const std::string& kernel_name) {
        check_cuda_status(cuModuleLoad(&module_, cubin_file.c_str()), "load cubin file");
        check_cuda_status(cuModuleGetFunction(&function_, module_, kernel_name.c_str()), "get function from cuda module");
    }

    void operator()(
            float* C,
            float* A,
            float* B,
            float alpha,
            float beta,
            unsigned flags,
            unsigned lda,
            unsigned ldb,
            unsigned ldc,
            unsigned m,
            unsigned n,
            unsigned k,
            unsigned param_ldaz,
            unsigned param_ldbz,
            unsigned param_ldcz,
            unsigned param_batch_loops
        ) const {
        // int sharedMemBytes = 0;

        // to call
        // cuLaunchKernel(
        //     function_,
        //     grid_dim_x,
        //     grid_dim_y,
        //     grid_dim_z,
        //     block_dim_x,
        //     block_dim_y,
        //     block_dim_z,
        //     sharedMemBytes,
        //     kernelParams (void** to arguments)
        //     config (can be null)
        // )
    }
};


std::unordered_map<std::string, std::vector<std::string>> spec_params = {
    {
        "gemm",
        {
            "float* param_C",
            "float* param_A",
            "float* param_B",
            "float param_alpha",
            "float param_beta",
            "unsigned param_flags",
            "unsigned param_lda",
            "unsigned param_ldb",
            "unsigned param_ldc",
            "unsigned param_m",
            "unsigned param_n",
            "unsigned param_k",
            "unsigned param_ldaz",
            "unsigned param_ldbz",
            "unsigned param_ldcz",
            "unsigned param_batch_loops"
        }
    }
};

std::string get_ptx_file(const SassKernel& kernel_spec,
                         const std::string& kernel_name,
                         const std::string& arch,
                         const std::string& ptx_version) {
    const int& thread_spec = kernel_spec.threads_;
    // args_spec, not used now
    std::string args_spec = "";
    const auto& param_spec = spec_params.at(kernel_spec.params_);

    std::vector<std::string> kernel_params_unjoined;
    for (const auto& p : param_spec) {
        auto split_p = utils::split(p, ' ', /*keep_empty=*/false);
        ASSERT2(split_p.size() == 2, utils::make_message("split by space "
            "should result in two elements for each param but got ",
            split_p.size(), " instead (p = ", p, ")."));
        auto& ptype = split_p[0];
        auto& pname = split_p[1];
        if (ptype[ptype.size() - 1] == '*') {
            ptype = ".u64";
        } else if (ptype == "float") {
            ptype = ".f32";
        } else {
            ptype = ".u32";
        }
        kernel_params_unjoined.emplace_back(
            utils::make_message(
                "    .param ", ptype, " ", pname
            )
        );
    }

    std::string kernel_params = utils::join(kernel_params_unjoined, ",\n");

    // if share
    std::string share = utils::make_message(
        "\n"
        ".shared .align 4 .b32 share[", kernel_spec.share_, "];\n"
    );

    std::string kernel_text = utils::make_message(
        "\n"
        ".version ", ptx_version, "\n"
        ".target ", arch, "\n"
        ".address_size 64\n"
        "// args: ", args_spec, "\n"
        ".visible .entry  ", kernel_name, "(\n",
        kernel_params, "\n"
        ")\n"
        ".reqntid ", thread_spec, "\n"
        "{\n",
        share, "\n"
        "    ret;\n"
        "}\n"
    );

    if (!utils::file_exists(kPtxDir)) {
        utils::makedirs(kPtxDir.c_str());
    }

    std::string kernel_ptx = utils::make_message(
        kPtxDir, kernel_name, ".ptx"
    );

    // check if contents have changed (for now let's assume yes)
    // then write out file contents.
    write_file(/*contents=*/kernel_text, /*fname=*/kernel_ptx);

    return kernel_ptx;
}

std::map<CUdevice, int> nervana_sm_counts_;
std::mutex nervana_sm_count_mutex_;


std::shared_ptr<LoadedSassKernel> get_sass_gemm_kernel(const std::string& base_name, CUdevice dev_number) {
    // based off of the get_kernel method of "neon/backends/kernel_specs.py" in the
    // repository: https://github.com/NervanaSystems/neon

    // get device attributes and check that they fit
    cudaDeviceProp prop;
    check_cuda_status(cudaGetDeviceProperties(&prop, dev_number), "get device properties");
    int major = prop.major;
    int minor = prop.minor;
    ASSERT2(major >= 5, "sass kernels require Maxwell or greater class hardware.");
    std::string arch = utils::make_message("sm_", major, minor);

    std::string libprefix = utils::make_message("PERL5LIB=", kMaxasDir);

    // maxas_i

    const auto& kernel_spec = kernels.at(base_name);
    std::string kernel_name = base_name;

    // TODO: check if sass kernel has args (not the case for gemm,
    // but is the case for conv, etc...)

    std::string sass_name = kernel_spec.sass_ + ".sass";
    std::string cubin_name = kernel_name + ".cubin";

    std::string ptx_version = major < 6 ? "4.2" : "5.0";
    std::string ptx_file = get_ptx_file(kernel_spec, kernel_name, arch, ptx_version);

    std::string sass_file = kSassDir + sass_name;
    std::string cubin_file = kCubinDir + cubin_name;

    ASSERT2(utils::file_exists(sass_file), utils::make_message(
        "Missing: ", sass_file, " for kernel: ", kernel_name, "."
    ));

    std::string ptxas_cmd = utils::make_message(
        "ptxas -v -arch ", arch, " -o ", cubin_file, " ", ptx_file
    );
    int ptxas_ret = system(ptxas_cmd.c_str());
    ASSERT2(WEXITSTATUS(ptxas_ret) == EXIT_SUCCESS, utils::make_message("error running "
        " command : ", ptxas_cmd, "."));

    std::string maxas_cmd = utils::make_message(
        libprefix, " ", kMaxasDir, "maxas.pl -i -w -k ", kernel_name,
        " ", sass_file, " ", cubin_file
    );
    int maxas_ret = system(maxas_cmd.c_str());
    ASSERT2(WEXITSTATUS(maxas_ret) == EXIT_SUCCESS, utils::make_message("error running "
        " command : ", maxas_cmd, "."));

    auto kernel = std::make_shared<LoadedSassKernel>(
        cubin_file, kernel_name
    );
    return kernel;
}

void nervana_hgemm(bool a_t, bool b_t,
                   int m, int n, int k,
                   short alpha,
                   const short *A, int lda,
                   const short *B, int ldb,
                   short beta,
                   short *C, int ldc,
                   unsigned int *rand_state,
                   bool stochastic_round, bool apply_relu,
                   CUstream stream, int grid=-1) {
    int sm_count;
    CUdevice dev_number;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);
        check_cuda_status(cuCtxGetDevice(&dev_number), "get current device");
        auto count = nervana_sm_counts_.find(dev_number);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        } else {
            int pi;
            check_cuda_status(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_number), "get device multiprocessor count");
            sm_count = pi;
            nervana_sm_counts_[dev_number] = pi;
        }
    }

    std::string name = "hgemm_";
    std::string trans;
    trans += a_t ? 't' : 'n';
    trans += b_t ? 't' : 'n';
    name += trans;
    int sizeA, sizeB, threads;
    std::tie(sizeA, sizeB, threads) = get_grid_dimensions(grid, m, n, sm_count, trans);

    int gridA = m / sizeA + (m % sizeA != 0);
    int gridB = n / sizeB + (n % sizeB != 0);
    std::stringstream ss;
    ss << "_" << sizeA << "x" << sizeB;
    name += ss.str();

    int flags = 0;
    flags |= (stochastic_round << 0);
    flags |= (apply_relu << 1);

    CUresult res;

    if (a_t)
        lda *= (8 * sizeof(short));

    if (!b_t)
        ldb *= (8 * sizeof(short));

    int zero = 0;
    int one  = 1;
    void *args[16] = {&C, &A, &B, &alpha, &beta, &flags, &lda, &ldb, &ldc, &m, &n, &k,
                      &zero, &zero, &zero, &one};

    auto kernel = get_sass_gemm_kernel(name, dev_number);
    check_cuda_status(cuLaunchKernel(kernel->function_,
                      1, gridA, gridB,
                      threads, 1, 1,
                      0,
                      stream, args, NULL), "kernel launch");
}

void nervana_sgemm(bool a_t, bool b_t,
                   int m, int n, int k,
                   float alpha,
                   const float *A, int lda,
                   const float *B, int ldb,
                   float beta,
                   float *C, int ldc,
                   unsigned int *rand_state,
                   bool stochastic_round, bool apply_relu,
                   CUstream stream, int grid=-1) {
    int sm_count;
    CUdevice dev_number;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);
        check_cuda_status(cuCtxGetDevice(&dev_number), "get current device");
        auto count = nervana_sm_counts_.find(dev_number);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        } else {
            int pi;
            check_cuda_status(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_number), "get device multiprocessor count");
            sm_count = pi;
            nervana_sm_counts_[dev_number] = pi;
        }
    }

    std::string name = "sgemm_";
    std::string trans;
    trans += a_t ? 't' : 'n';
    trans += b_t ? 't' : 'n';
    name += trans;
    int sizeA, sizeB, threads;
    std::tie(sizeA, sizeB, threads) = get_grid_dimensions(grid, m, n, sm_count, trans);

    int gridA = m / sizeA + (m % sizeA != 0);
    int gridB = n / sizeB + (n % sizeB != 0);
    std::stringstream ss;
    ss << "_" << sizeA << "x" << sizeB;
    name += ss.str();

    int flags = 0;
    flags |= (stochastic_round << 0);
    flags |= (apply_relu << 1);

    CUresult res;

    if (a_t)
        lda *= (8 * sizeof(float));

    if (!b_t)
        ldb *= (8 * sizeof(float));

    int zero = 0;
    int one  = 1;
    void *args[16] = {&C, &A, &B, &alpha, &beta, &flags, &lda, &ldb, &ldc, &m, &n, &k,
                      &zero, &zero, &zero, &one};

    auto kernel = get_sass_gemm_kernel(name, dev_number);
    check_cuda_status(cuLaunchKernel(kernel->function_,
                      1, gridA, gridB,
                      threads, 1, 1,
                      0,
                      stream, args, NULL), "kernel launch");
}

namespace expression {

void NervanaGemmAssignExpressionState::run() const {
    Array dst = dest_->array_;
    Array lhs = left_->destination_op()->as_rvalue()->as_array()->array_;
    Array rhs = right_->destination_op()->as_rvalue()->as_array()->array_;
    auto op_dtype = dtype();
    void* dst_ptr = destination_multiplier_ == 0 ?
        dst.memory()->overwrite_data(device_) : dst.memory()->mutable_data(device_);
    const void* rhs_ptr = rhs.memory()->readonly_data(device_);
    const void* lhs_ptr = lhs.memory()->readonly_data(device_);
    bool rhs_transpose, lhs_transpose, dst_transpose;
    int rhs_stride, lhs_stride, dst_stride;
    std::tie(rhs_transpose, rhs_stride) = gemm_stride_transpose(rhs);
    std::tie(lhs_transpose, lhs_stride) = gemm_stride_transpose(lhs);
    std::tie(dst_transpose, dst_stride) = gemm_stride_transpose(dst);

    // in row major:
    // dst = result_multiplier * left * right + destination_multiplier * dst
    // in col major:
    // dst.T = result_multiplier * right.T * left.T + destination_multiplier * dst.T
    int m = rhs.shape()[1],
        n = lhs.shape()[0],
        k = rhs.shape()[0];

     // use null stream for now
    cudaStream_t stream = NULL;
    const float result_multiplier_float = result_multiplier_;
    const float destination_multiplier_float = destination_multiplier_;

    nervana_sgemm(
        rhs_transpose, lhs_transpose,
        m, n, k,
        /*alpha=*/result_multiplier_float,
        (const float*)rhs_ptr, rhs_stride,
        (const float*)lhs_ptr, lhs_stride,
        /*beta=*/destination_multiplier_float,
        (float*)dst_ptr, dst_stride,
        /*rand_state=*/0,
        /*stochastic_round=*/false,
        /*apply_relu=*/false,
        /*stream=*/stream,
        /*grid=*/-1
    );
}


std::vector<int> device_major_capabilities_cache(DALI_MAX_GPU_DEVICES, -1);

int device_major_capabilities(const memory::Device& device) {
    int dev_number = device.number();
    if (device_major_capabilities_cache[dev_number] == -1) {
        cudaDeviceProp prop;
        check_cuda_status(cudaGetDeviceProperties(&prop, dev_number), "get device properties");
        device_major_capabilities_cache[dev_number] = prop.major;
    }
    return device_major_capabilities_cache[dev_number];
}

bool device_compatible_with_nervana(const memory::Device& device) {
    return device_major_capabilities(device) >= 3;
}

}  // namespace expression

#endif  // DALI_USE_CUDA
