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

#include "dali/utils/print_utils.h"

typedef uint64_t hash_t;
typedef std::unordered_map<std::string, std::string> macro_args_t;


std::string get_call_args(std::size_t num_args);
std::string get_class_name(const char* name);
std::string macro_args_to_string(macro_args_t macro_args);

template<typename... Args, typename std::enable_if<sizeof... (Args) == 0, int>::type = 0>
void get_function_arguments(int i, std::string* call_ptr) {}

template<typename Arg, typename... Args>
void get_function_arguments(int i, std::string* call_ptr) {
    std::string& call = *call_ptr;
    if (i > 0) {
        call = call + ", ";
    }
    call = call + get_class_name(typeid(Arg).name()) + " " + (char)(((int)'a') + i);
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

    std::string header_file_includes() const;

    bool load(hash_t hash);

    template<typename... Args>
    void write_code(const std::string& fname,
                    const std::string& code) {
        std::ofstream out(fname.c_str(), std::ofstream::out);
        if (out.bad()) {
            std::cout << "cannot open " << fname << std::endl;
            exit(EXIT_FAILURE);
        }
        // add header to code (and extern c to avoid name mangling)
        std::string newcode =
            utils::MS() << header_file_includes()
                        << code << "\n" << "extern \"C\" void maker ("
                        << get_function_arguments<Args...>()
                        << "){\nrun("
                        << get_call_args(sizeof...(Args))
                        << ");}";
        out << newcode;
        out.flush();
        out.close();
    }


    template<typename... Args>
    void compile(hash_t hash, std::string code_template, macro_args_t macro_args) {
        std::string module_path = utils::MS() << outpath_ << hash << ".so";

        std::string cppfile = utils::MS() << outpath_ << hash << ".cpp";
        std::string logfile = utils::MS() << outpath_ << hash << ".log";

        write_code<Args...>(cppfile, code_template);

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
