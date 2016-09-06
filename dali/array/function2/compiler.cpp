#include "compiler.h"

#include <sys/stat.h>
#include <cxxabi.h>

#include "dali/utils/core_utils.h"



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

std::string macro_args_to_string(macro_args_t macro_args) {
    std::stringstream ss;
    for (auto& kv : macro_args) {
        ss << "-DDALI_RTC_" << kv.first << "=" << kv.second << " ";
    }
    return ss.str();
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

bool Compiler::load(hash_t hash) {
    if (hash_to_f_ptr_.find(hash) != hash_to_f_ptr_.end()) {
        return true;
    }

    std::string module_path = utils::MS() << outpath_ << hash << ".so";

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
                                       const std::string& extra_args) {
    return utils::MS() << "gcc -std=c++11 " << source
                        << " -o " << dest
                        << " -I"  << include_path_
                        // TODO(jonathan): make sure we never mistakenly delete this line:
                        << " -I"  << "/home/sidor/projects/Dali/third_party/mshadow"
                        << " " << extra_args
                        // dynamic lookup: -undefined dynamic_lookup on clang
                        // and
                        << " -Wl,--unresolved-symbols=ignore-in-object-files"
                        << " -fPIC"
                        // for gcc
                        // << " --compiler-options=\"-undefined dynamic_lookup\" "
                        << " -O2 -shared &> " << logfile;
}

bool Compiler::compile_code(const std::string& source,
                            const std::string& dest,
                            const std::string& logfile,
                            const std::string& extra_args) {
    auto cmd = compiler_command(source, dest, logfile, extra_args);
    int ret = system(cmd.c_str());
    return WEXITSTATUS(ret) == EXIT_SUCCESS;
}

std::vector<std::string> kHeaders = {
    "dali/array/function2/array_view.h",
    "dali/array/array.h"
};
std::string kOutpath    = "/home/sidor/.dali/cache/";
std::string kIncludeDir = "/home/sidor/projects/Dali/build_cpu/dali_generated/";

Compiler chief_compiler(kHeaders, kOutpath, kIncludeDir);
