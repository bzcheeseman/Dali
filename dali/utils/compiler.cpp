#include "compiler.h"
#include "dali/config.h"

#include <sys/stat.h>
#include <cxxabi.h>

#include "dali/utils/assert2.h"
#include "dali/utils/core_utils.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"

std::string get_call_args(std::size_t num_args) {
    std::string call_args;
    for (int i = 0; i < num_args; i++) {
        if (i > 0) {
            call_args = call_args + ", ";
        }
        call_args = call_args + (char)(((int)'a') + i);
    }
    return call_args;
}

std::string get_class_name(const char* name) {
    int status;
    char * demangled = abi::__cxa_demangle(
        name,
        0,
        0,
        &status
    );
    return std::string(demangled);
}

ModulePointer::ModulePointer(const std::string& libname) : module_(NULL), libname_(libname) {
    module_ = dlopen(libname_.c_str(), RTLD_LAZY);
    if(!module_) {
        std::cerr << "error loading library:\n" << dlerror() << std::endl;
        exit(EXIT_FAILURE);
    }
}

ModulePointer::~ModulePointer() {
    if (module_) {
        dlclose(module_);
    }
}

Module::Module() : module_ptr_(NULL) {}
Module::Module(const std::string& libname) :
        module_ptr_(std::make_shared<ModulePointer>(libname)) {
}

void* Module::module() {
    return module_ptr_->module_;
}



Compiler::Compiler(std::vector<std::string> headers, std::string outpath, std::string include_path) :
        headers_(headers), outpath_(outpath), include_path_(include_path) {

    if (!utils::file_exists(outpath)) {
        utils::makedirs(outpath.c_str());
    }
}

std::string Compiler::kCxxExecutable  = STR(DALI_CXX_COMPILER);
std::string Compiler::kCudaExecutable = "nvcc";
std::string Compiler::kCudaCxxExecutable = STR(DALI_CUDA_CXX_COMPILER);
std::string Compiler::kCompilerId     = STR(DALI_CXX_COMPILER_ID);


bool Compiler::load(hash_t hash) {
    if (hash_to_f_ptr_.find(hash) != hash_to_f_ptr_.end()) {
        return true;
    }

    auto module_path = utils::make_message(outpath_, hash, ".so");

    if (!utils::file_exists(module_path)) {
        return false;
    }

    modules_.emplace_back(module_path);

    auto ptr = modules_.back().get_symbol<void*>("maker");
    hash_to_f_ptr_[hash] = ptr;

    return true;
}


std::string Compiler::header_file_includes() const {
    std::stringstream ss;
    for (auto& header : headers_) {
        ss << "#include \"" << header << "\"\n";
    }
    return ss.str();
}

std::string Compiler::compiler_command(const std::string& source,
                                       const std::string& dest,
                                       const std::string& logfile,
                                       const std::string& extra_args,
                                       const memory::DeviceT& device_type) {
    std::string cxx_compiler_os_specific_flags;
    if (Compiler::kCompilerId == "clang") {
        cxx_compiler_os_specific_flags = (
            " -undefined dynamic_lookup"
            " -Rpass=loop-vectorize"
            " -Rpass-analysis=loop-vectorize"
            " -ffast-math"
            " -fslp-vectorize-aggressive"
        );
    } else if (Compiler::kCompilerId == "gnu") {
        cxx_compiler_os_specific_flags = (
            " -shared"
            " -fPIC"
            " -ftree-vectorize"
            " -msse2"
            " \"-Wl,--unresolved-symbols=ignore-in-object-files\""
        );
    } else {
        utils::assert2(false, utils::make_message("Compiler::kCompilerId == ", Compiler::kCompilerId, " is not supported."));
    }

    auto cxx_compile_flags = utils::make_message(
        " -I" , include_path_,
        " -I", STR(DALI_BLAS_INCLUDE_DIRECTORY),
        " -DDALI_ARRAY_HIDE_LAZY=1",
        " ", extra_args,
        cxx_compiler_os_specific_flags
    );

    if (device_type == memory::DEVICE_T_CPU) {
        return utils::make_message(Compiler::kCxxExecutable,
                                   " ", source,
                                   " -o ", dest,
                                   " -std=c++11 ",
                                   " ", cxx_compile_flags,
                                   " -O3 &> ", logfile);
    }
#ifdef DALI_USE_CUDA
    else if (device_type == memory::DEVICE_T_GPU) {
        return utils::make_message(Compiler::kCudaExecutable,
                                   " ", source,
                                   " -o ", dest,
                                   " -std=c++11",
                                   " -ccbin ", Compiler::kCudaCxxExecutable,
                                   " --compiler-options ", "\"", utils::find_and_replace(cxx_compile_flags, "\"", "\\\""), "\"",
                                   " -O3 &> ", logfile);
    }
#endif
    else {
        ASSERT2(false, "TPUs are not yet supported.");
    }

}

void Compiler::write_code(const std::string& fname,
                          const std::string& code,
                          const std::string& function_arguments,
                          const std::string& call_args) {
    std::ofstream out(fname.c_str(), std::ofstream::out);
    if (out.bad()) {
        std::cout << "cannot open " << fname << std::endl;
        exit(EXIT_FAILURE);
    }
    // add header to code (and extern c to avoid name mangling)
    out << header_file_includes()
        << code
        << "\nextern \"C\" void maker ("
        << function_arguments
        << "){\nrun("
        << call_args
        << ");\n}\n";
    out.flush();
    out.close();
}

bool Compiler::compile_code(const std::string& source,
                            const std::string& dest,
                            const std::string& logfile,
                            const std::string& extra_args,
                            const memory::DeviceT& device_type) {
    auto cmd = compiler_command(source, dest, logfile, extra_args, device_type);
    int ret = system(cmd.c_str());
    std::cout << "cmd = " << cmd << std::endl;
    std::cout << "utils::listdir(outpath) = " << utils::listdir(array_op_compiler.outpath_) << std::endl;

    std::cout << "printing logfile:" << std::endl;
    std::ifstream fp(logfile.c_str());
    std::string l;
    while (std::getline(fp, l)) {
        std::cout << l << std::endl;
    }
    std::cout << "done printing logfile." << std::endl;

    return WEXITSTATUS(ret) == EXIT_SUCCESS;
}

std::vector<std::string> kHeaders = {
    "dali/array/jit/array_view.h",
    "dali/array/jit/reducer_kernels.h",
    "dali/array/functor.h"
};
#ifdef DALI_CACHE_DIR
std::string kOutpath    = utils::expanduser(STR(DALI_CACHE_DIR));
#else
std::string kOutpath    = utils::expanduser("~/.dali/cache/");
#endif
std::string kIncludeDir = STR(DALI_COMPILE_DIRECTORY);

Compiler array_op_compiler(kHeaders, kOutpath, kIncludeDir);
