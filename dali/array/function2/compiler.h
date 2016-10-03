#ifndef DALI_ARRAY_FUNCTION2_COMPILER_H
#define DALI_ARRAY_FUNCTION2_COMPILER_H

#include <cstdlib> // EXIT_FAILURE, etc
#include <dlfcn.h>      // dynamic library loading, dlopen() etc
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "dali/utils/hash_utils.h"
#include "dali/utils/print_utils.h"

typedef std::unordered_map<std::string, std::string> macro_args_t;


std::string get_call_args(std::size_t num_args);
std::string get_class_name(const char* name);
std::string macro_args_to_string(macro_args_t macro_args);

namespace {
    // is_const:
    // this functions checks if a type is const qualified or reference
    // to a const-qualified
    template<typename T>
    constexpr bool is_const() {
        return std::is_const<typename std::remove_reference<T>::type>::value;
    }
}

template<typename... Args, typename std::enable_if<sizeof... (Args) == 0, int>::type = 0>
void get_function_arguments(int i, std::string* call_ptr) {}

template<typename Arg, typename... Args>
void get_function_arguments(int i, std::string* call_ptr) {
    std::string& call = *call_ptr;
    if (i > 0) {
        call = call + ", ";
    }
    if (is_const<Arg>()) {
        call = call + "const ";
    }
    call += get_class_name(typeid(Arg).name());
    if (std::is_lvalue_reference<Arg>::value) {
        call = call + "&";
    }
    if (std::is_rvalue_reference<Arg>::value) {
        call = call + "&&";
    }
    call = call + " " + (char)(((int)'a') + i);
    get_function_arguments<Args...>(i+1, call_ptr);
}

template<typename... Args>
std::string get_function_arguments() {
    std::string s;
    get_function_arguments<Args...>(0, &s);
    return s;
}


struct ModulePointer {
    void* module_;
    std::string libname_;
    ModulePointer(const std::string& libname);
    ~ModulePointer();
};

struct Module {
    std::shared_ptr<ModulePointer> module_ptr_;

    Module();
    Module(const std::string& libname);
    void* module();

    template<typename T>
    T get_symbol(const std::string& name) {
        void* symbol = dlsym(module(), name.c_str());
        const char* dlsym_error = dlerror();
        if (dlsym_error != NULL) {
            std::cerr << "error loading symbol:\n" << dlsym_error << std::endl;
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<T>(symbol);
    }
};

class Compiler {
    std::vector<std::string>           headers_;
    std::string                        outpath_;
    std::string                        include_path_;
    std::vector<Module> modules_;
    std::unordered_map<hash_t, void*> hash_to_f_ptr_;

  public:
    Compiler(std::vector<std::string> headers,
             std::string outpath,
             std::string include_path);

    static std::string kExecutable;
    static std::string kCompilerId;

    std::string header_file_includes() const;

    bool load(hash_t hash);

    void write_code(const std::string& fname,
                    const std::string& code,
                    const std::string& function_arguments,
                    const std::string& call_args);

    template<typename... Args>
    void compile(hash_t hash, std::string code_template, macro_args_t macro_args) {
        std::string module_path = utils::MS() << outpath_ << hash << ".so";

        std::string cppfile = utils::MS() << outpath_ << hash << ".cpp";
        std::string logfile = utils::MS() << outpath_ << hash << ".log";

        write_code(
            cppfile,
            code_template,
            get_function_arguments<Args...>(),
            get_call_args(sizeof...(Args))
        );

        auto macro_args_str = macro_args_to_string(macro_args);
        bool success = compile_code(
            cppfile,
            module_path,
            logfile,
            macro_args_str
        );

        if (!success) {
            std::cout << "Failure encoutered when running the following command:" << std::endl;
            std::cout << compiler_command(cppfile, module_path, logfile, macro_args_str) << std::endl;
            std::cout << "See details in " << logfile << std::endl;
            exit(EXIT_FAILURE);
        }

        modules_.emplace_back(module_path);

        auto ptr = modules_.back().get_symbol<void*>("maker");
        hash_to_f_ptr_[hash] = ptr;
    }

    template<typename... Args>
    std::function<void(Args...)> get_function(hash_t hash) {
        auto f_ptr = reinterpret_cast<void(*)(Args...)>(hash_to_f_ptr_[hash]);
        return std::function<void(Args...)>(f_ptr);
    }
  private:
    std::string compiler_command(const std::string& source,
                                 const std::string& dest,
                                 const std::string& logfile,
                                 const std::string& extra_args);

    bool compile_code(const std::string& source,
                      const std::string& dest,
                      const std::string& logfile,
                      const std::string& extra_args);
};


extern Compiler array_op_compiler;

#endif  // DALI_ARRAY_FUNCTION2_COMPILER_H
