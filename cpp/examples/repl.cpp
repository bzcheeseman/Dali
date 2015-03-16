#include <unordered_map>
#include <string>
#include <memory>
#include <exception>
#include <sstream>
#include <functional>

#include "dali/core.h"

using std::string;
using std::unordered_map;
using std::cout;
using std::endl;
using std::make_shared;
using std::stringstream;
using std::shared_ptr;
using std::vector;
using utils::startswith;
using utils::is_number;
using utils::trim;
using utils::from_string;
using std::function;

typedef double REAL_t;
typedef Mat<REAL_t> mat;
typedef std::shared_ptr<mat> shared_mat;

#define THROW_INCOMPATIBLE_TYPE(X) stringstream ss; \
        ss << "Cannot cast " << type_name() << " to " << #X; \
        throw std::runtime_error(ss.str());

/**
Very rudimentary REPL for Dali / RecurrentJS.

Goal
----

Support super fast prototyping by creating
a super feature restricted REPL for matrix operations
and graph operations.

Key goal is to allow interactive editing of models
without compilation.


Features:
---------

* Create a Graph using Graph(true) or Graph()
* Create primitive types ( bool, integer, float)
* Create Matrix using any of the basic inits methods (
empty, using integers, using random numbers, etc...)


TODO:
-----

-Add ways of adding methods to classes and calling these dynamically.
-Have simple way of declaring classes in this REPL.
-Enable method calls.
-Support multi-line statements.
-Support strings for loading numpy objects, etc..


@author Jonathan Raiman

**/

class Expression {
    public:
        function<shared_ptr<Expression>(Expression&, vector<shared_ptr<Expression>>&)> call = 0;

        class ExpressionConstructor {
            public:
                typedef function<shared_ptr<void>(vector<shared_ptr<Expression>>&)> constructor_t;
                typedef function<bool(const vector<shared_ptr<Expression>>&)> validator_t;
                validator_t validator;
                constructor_t constructor;
                ExpressionConstructor(validator_t _validator, constructor_t _constructor) : constructor(_constructor), validator(_validator) {}
        };

        class WrappedCppClass {
            public:
                vector<ExpressionConstructor> constructors;
                unordered_map<string, Expression> methods;
                string name;
                const int TYPE_T;
                static function<string(const Expression&)> default_to_string;
                function<string(const Expression&)> to_string;
                static vector< shared_ptr<WrappedCppClass> > classes;
                static shared_ptr<WrappedCppClass> bool_class;
                static shared_ptr<WrappedCppClass> int_class;
                static shared_ptr<WrappedCppClass> float_class;
                static shared_ptr<WrappedCppClass> lambda_class;
                WrappedCppClass (string _name, vector<ExpressionConstructor> _constructors, uint type_t) : name(_name), constructors(_constructors), TYPE_T(type_t), to_string(default_to_string) {}
                WrappedCppClass (string _name, vector<ExpressionConstructor> _constructors, uint type_t, function<string(const Expression&)> to_s) : name(_name), constructors(_constructors), TYPE_T(type_t), to_string(to_s) {}

                void add_method (const string& name, function<shared_ptr<Expression>(Expression&, vector<shared_ptr<Expression>>&)> _call) {
                    methods.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple(name),
                        std::forward_as_tuple(lambda_class, nullptr, _call));
                }
        };

        class ArgumentList {
            public:
                string call;
                vector<shared_ptr<Expression>> arguments;
                vector<string> get_arguments(const string& line) {
                    vector<string> args;
                    stringstream ss(line);
                    vector<char> arg_tokens;
                    vector<char> call_tokens;
                    int depth = 0;
                    char ch;
                    bool creating_method_name = true;
                    while (ss) {
                        ch = ss.get();
                        if (ch == EOF) {
                            break;
                        }
                        if (ch == '(') {
                            if (depth > 0) {
                                arg_tokens.emplace_back(ch);
                            }
                            depth += 1;
                            creating_method_name = false;
                            continue;
                        }
                        if (ch == ')') {
                            depth -= 1;
                            if (depth < 0) {
                                throw std::runtime_error("SyntaxError: Unequal number of opening and closing parentheses.");
                            }
                            if (depth == 0) {
                                if (arg_tokens.size() > 0) {
                                    args.emplace_back(arg_tokens.begin(), arg_tokens.end());
                                    arg_tokens.clear();
                                    break;
                                }
                            }
                            if (depth > 0) {
                                arg_tokens.emplace_back(ch);
                            }
                            continue;
                        }
                        if (ch == ',') {
                            if (depth == 1) {
                                if (arg_tokens.size() == 0) {
                                    throw std::runtime_error("ParseError: Empty argument to function call.");
                                } else {
                                    args.emplace_back(arg_tokens.begin(), arg_tokens.end());
                                    arg_tokens.clear();
                                }
                            } else {
                                arg_tokens.emplace_back(ch);
                            }
                            continue;
                        }
                        if (ch != ' ') {
                            if (depth > 0) {
                                arg_tokens.emplace_back(ch);
                            } else if (creating_method_name) {
                                call_tokens.emplace_back(ch);
                            }
                        }
                    }
                    call = string(call_tokens.begin(), call_tokens.end());
                    return args;
                }
                ArgumentList(const string& line, unordered_map<string, shared_ptr<Expression>> locals) {
                    auto string_args = get_arguments(line);
                    arguments.reserve(string_args.size());
                    for (auto& arg : string_args)
                        arguments.emplace_back(parseExpression(arg, locals));
                }
        };

        shared_ptr<void> internal_t;
        shared_ptr<WrappedCppClass> type;

        static string get_starting_element (const string& input) {
            stringstream ss(input);
            char ch;
            int end   = 0;
            while (ss) {
                ch = ss.get();
                if (ch != '.' && ch != '(' && ch != ')' && ch != ',') {
                    end += 1;
                } else {
                    break;
                }
            }
            return string(input.begin(), input.begin() + end);
        }

        bool is_numeric() const {
            return (type == WrappedCppClass::int_class || type == WrappedCppClass::float_class);
        }

        bool is_int() const {
            return type == WrappedCppClass::int_class;
        }

        bool is_boolean() const {
            return (type == WrappedCppClass::bool_class || type == WrappedCppClass::int_class);
        }

        static bool is_literal (const string& line) {
            if (line.find('(') == std::string::npos) {
                return true;
            }
            int closing_paren = line.find(')');
            if (closing_paren != std::string::npos) {
                int opening_paren = line.find('(');
                if (opening_paren > closing_paren) {
                    return true;
                }
            }
            return false;
        }
        static bool is_protected_literal( const string& line) {
            if (line == "true"  || line == "false" ||
                is_number(line) || line == "exit"  || line == "type") {
                return true;
            }
            return false;
        }
        string type_name () const {
            return type->name;
        }

        bool to_bool() const  {
            if (type == WrappedCppClass::bool_class) {
                return *reinterpret_cast<bool*>(internal_t.get());
            } else if (type == WrappedCppClass::int_class) {
                return int_as_bool();
            } else {
                THROW_INCOMPATIBLE_TYPE(Boolean)
            }
            return *reinterpret_cast<bool*>(internal_t.get());
        }
        int to_int () const  {
            if (type == WrappedCppClass::int_class) {
                return *reinterpret_cast<int*>(internal_t.get());
            } else if (type == WrappedCppClass::float_class) {
                return (int) *reinterpret_cast<REAL_t*>(internal_t.get());
            } else {
                THROW_INCOMPATIBLE_TYPE(Integer)
            }
        }
        bool int_as_bool() const  {
            if (type == WrappedCppClass::int_class) {
                return (*reinterpret_cast<int*>(internal_t.get())) > 0;
            } else {
                THROW_INCOMPATIBLE_TYPE(Boolean)
            }
        }
        REAL_t to_float () const  {
            if (type == WrappedCppClass::float_class) {
                return *reinterpret_cast<REAL_t*>(internal_t.get());
            } else if (type == WrappedCppClass::int_class) {
                return (REAL_t) *reinterpret_cast<int*>(internal_t.get());
            } else {
                THROW_INCOMPATIBLE_TYPE(Float)
            }
        }
        string to_string () const  {
            return type->to_string(*this);
        }

        static shared_ptr<Expression> parseExpression(string _repr, unordered_map<string, std::shared_ptr<Expression>>& locals);

        Expression(shared_ptr<WrappedCppClass> _type, shared_ptr<void> _internal_t) : internal_t(_internal_t), type(_type) {}
        Expression(shared_ptr<WrappedCppClass> _type, shared_ptr<void> _internal_t, function<shared_ptr<Expression>(Expression&, vector<shared_ptr<Expression>>&)> _call) : internal_t(_internal_t), type(_type), call(_call) {}
};

function<string(const Expression&)> Expression::WrappedCppClass::default_to_string = [](const Expression& self) {
    stringstream ss;
    ss << "<" << self.type->name << " >";
    return ss.str();
};
vector< shared_ptr<Expression::WrappedCppClass> > Expression::WrappedCppClass::classes;
shared_ptr<Expression::WrappedCppClass> Expression::WrappedCppClass::bool_class = make_shared<WrappedCppClass>("Boolean", vector<Expression::ExpressionConstructor>(), -1, [](const Expression& expr) {
    if (*reinterpret_cast<bool*>(expr.internal_t.get())) {
        return "true";
    }
    return "false";
});
shared_ptr<Expression::WrappedCppClass> Expression::WrappedCppClass::float_class = make_shared<WrappedCppClass>("Float", vector<Expression::ExpressionConstructor>(), -2, [](const Expression& expr) {
    stringstream ss;
    ss << (REAL_t) *reinterpret_cast<REAL_t*>(expr.internal_t.get());
    return ss.str();
});
shared_ptr<Expression::WrappedCppClass> Expression::WrappedCppClass::int_class = make_shared<WrappedCppClass>("Integer", vector<Expression::ExpressionConstructor>(), -3, [](const Expression& expr) {
    stringstream ss;
    ss << (int) *reinterpret_cast<int*>(expr.internal_t.get());
    return ss.str();
});
shared_ptr<Expression::WrappedCppClass> Expression::WrappedCppClass::lambda_class = make_shared<WrappedCppClass>("Lambda", vector<Expression::ExpressionConstructor>(), -4);

shared_ptr<Expression> Expression::parseExpression(string _repr, unordered_map<string, std::shared_ptr<Expression>>& locals) {
    if (is_literal(_repr)) {
        if (startswith(_repr, "exit")) {
            exit(0);
        }
        if (startswith(_repr, "true")) {
            return make_shared<Expression>(
                WrappedCppClass::bool_class,
                make_shared<bool>(true)
            );
        } else if (startswith(_repr, "false")) {
            return make_shared<Expression>(
                WrappedCppClass::bool_class,
                make_shared<bool>(false)
            );
        } else if (is_number(_repr)) {
            if (_repr.find('.') != std::string::npos) {
                return make_shared<Expression>(
                    WrappedCppClass::float_class,
                    make_shared<REAL_t>( from_string<REAL_t>(_repr) )
                );
            } else {
                return make_shared<Expression>(
                    WrappedCppClass::int_class,
                    make_shared<int>( from_string<int>(_repr) )
                );
            }
        } else {
            if (locals.find(_repr) != locals.end()) {
                return locals.at(_repr);
            } else {
                stringstream ss;
                ss << "Error: Unknown variable name or function \""
                   << _repr << "\"";
                throw std::runtime_error(ss.str());
            }
        }
    } else {
        auto expression = get_starting_element(_repr);
        shared_ptr<WrappedCppClass> detected_class = nullptr;
        for (auto cpp_class : WrappedCppClass::classes ) {
            if (expression == cpp_class->name) {
                detected_class = cpp_class;
                break;
            }
        }
        if ( detected_class != nullptr) {
            auto arglist = ArgumentList(_repr, locals);
            auto good_constructor = std::find_if(detected_class->constructors.begin(), detected_class->constructors.end(), [&arglist](ExpressionConstructor& cc) {
                return cc.validator(arglist.arguments);
            });
            if (good_constructor != detected_class->constructors.end()) {
                return make_shared<Expression>(
                    detected_class,
                    good_constructor->constructor(arglist.arguments)
                );
            } else {
                stringstream ss;
                ss << "Error: No compatible constructor found for \"" << detected_class->name << "\"";
                throw std::runtime_error(ss.str());
            }
        } else {
            if (locals.find(expression) != locals.end()) {
                // deal with method calls now
                auto& variable = *locals.at(expression);
                detected_class = variable.type;

                auto arglist = ArgumentList(
                    string(_repr.begin() + expression.size() + (_repr.at(expression.size()) == '.' ? 1 : 0), _repr.end()),
                    locals);

                if (detected_class->methods.find(arglist.call) != detected_class->methods.end()) {
                    auto found_method = detected_class->methods.at(arglist.call);
                    return found_method.call(variable, arglist.arguments);
                } else {
                    stringstream ss;
                    ss << "Error: Method \"" << arglist.call << "\" does not exist for type \"" << detected_class->name << "\".";
                    throw std::runtime_error(ss.str());
                }
            } else {
                stringstream ss;
                ss << "Error: Unknown type: \"" << expression << "\"";
                throw std::runtime_error(ss.str());
            }
        }
    }
}

std::ostream& operator<<(std::ostream& strm, const Expression& expr) {
    return strm << expr.to_string();
}

// The type of shared expression variable,
// the lingua franca of this REPL.
shared_ptr<Expression> parseLine (const string& line, unordered_map<string, shared_ptr<Expression>>& locals) {
    if (!line.empty()) {
        int assignment_position = line.find('=');
        if (assignment_position != std::string::npos) {
            string varname(line.begin(), line.begin() + assignment_position);
            varname = trim(varname);
            if (!varname.empty()) {
                if (Expression::is_protected_literal(varname)) {
                    throw std::runtime_error("Error: Cannot make an assignment to a primitive type.");
                } else {
                    auto assignment_string = string(
                            line.begin() + assignment_position + 1,
                            line.end()
                        );
                    assignment_string = trim(assignment_string);
                    auto variable = Expression::parseExpression(
                        assignment_string, locals);
                    locals[string(varname)] = variable;
                    return nullptr;
                }
            } else {
                throw std::runtime_error("SyntaxError: Empty variable name for assignment.");
            }
        } else {
            return Expression::parseExpression(line, locals);
        }
    } else {
        return nullptr;
    }
}

void declare_classes() {

    auto zero_arg = [](const vector<shared_ptr<Expression>>& args) {
        return args.size() == 0;
    };
    auto single_arg_boolean = [](const vector<shared_ptr<Expression>>& args) {
        return (args.size() == 1 && args[0]->is_boolean());
    };
    auto two_int_arg = [](const vector<shared_ptr<Expression>>& args) {
        return (args.size() == 2 && args[0]->is_int() && args[1]->is_int());
    };
    auto two_int_arg_and_one_float = [](const vector<shared_ptr<Expression>>& args) {
        return (args.size() == 3 && args[0]->is_int() && args[1]->is_int()
                                 && args[2]->is_numeric()
        );
    };
    auto two_int_arg_and_two_floats = [](const vector<shared_ptr<Expression>>& args) {
        return (args.size() == 4 && args[0]->is_int() && args[1]->is_int()
                                 && args[2]->is_numeric() && args[3]->is_numeric()
        );
    };

    auto mat_constructors = {
        Expression::ExpressionConstructor( two_int_arg, [](vector<shared_ptr<Expression>>& args) {return make_shared<Mat<REAL_t>>(args[0]->to_int(), args[1]->to_int());} ),
        Expression::ExpressionConstructor( two_int_arg_and_one_float, [](vector<shared_ptr<Expression>>& args) {return make_shared<Mat<REAL_t>>(args[0]->to_int(), args[1]->to_int(), args[2]->to_float());} ),
        Expression::ExpressionConstructor( two_int_arg_and_two_floats, [](vector<shared_ptr<Expression>>& args) {return make_shared<Mat<REAL_t>>(args[0]->to_int(), args[1]->to_int(), args[2]->to_float(), args[3]->to_float());} )
    };

    auto MAT = make_shared<Expression::WrappedCppClass>("Mat", mat_constructors, (uint) Expression::WrappedCppClass::classes.size(), [](const Expression& self) {
        stringstream ss;
        ss << * reinterpret_cast<Mat<REAL_t>*>(self.internal_t.get());
        return ss.str();
    });

    Expression::WrappedCppClass::classes.emplace_back(MAT);

    auto graph_constructors = {
        Expression::ExpressionConstructor( zero_arg, [](vector<shared_ptr<Expression>>& args) {return make_shared<Graph<REAL_t>>();} ),
        Expression::ExpressionConstructor( single_arg_boolean, [](vector<shared_ptr<Expression>>& args) { return make_shared<Graph<REAL_t>>(args[0]->to_bool());}),
    };

    auto GRAPH = make_shared<Expression::WrappedCppClass>("Graph", graph_constructors, (uint) Expression::WrappedCppClass::classes.size(), [](const Expression& self) {
        stringstream ss;
        ss << * reinterpret_cast<Graph<REAL_t>*>(self.internal_t.get());
        return ss.str();
    });

    Expression::WrappedCppClass::classes.emplace_back(GRAPH);

    #define GRAPH_BINARY_MAT_METHOD(X) GRAPH->add_method(#X, [MAT](Expression& expr, vector<shared_ptr<Expression>>& args) -> shared_ptr<Expression> { \
        if (args.size() == 2 && args[0]->type == MAT && args[1]->type == MAT ) { \
            return make_shared<Expression>( \
                MAT, \
                reinterpret_cast<Graph<REAL_t>*>(expr.internal_t.get())->X( \
                    std::static_pointer_cast<mat>(args[0]->internal_t), \
                    std::static_pointer_cast<mat>(args[1]->internal_t) \
                ) \
            ); \
        } else { \
            throw std::runtime_error("Error: Wrong number of arguments for method " #X "."); \
        } \
    });

    #define GRAPH_UNARY_MAT_METHOD(X) GRAPH->add_method(#X, [MAT](Expression& expr, vector<shared_ptr<Expression>>& args) -> shared_ptr<Expression> { \
        if (args.size() == 1 && args[0]->type == MAT) { \
            return make_shared<Expression>( \
                MAT, \
                reinterpret_cast<Graph<REAL_t>*>(expr.internal_t.get())->X( \
                    std::static_pointer_cast<mat>(args[0]->internal_t) \
                ) \
            ); \
        } else { \
            throw std::runtime_error("Error: Wrong number of arguments for method " #X "."); \
        } \
    });

    #define ARGUMENTlESS_VOID_METHOD_ON_CLASS(CLS, X, CAST) CLS->add_method(#X, [](Expression& expr, vector<shared_ptr<Expression>>& args) -> shared_ptr<Expression> { \
        if (args.size() > 0) { \
            throw std::runtime_error("Error: Wrong number of arguments for method " #X "."); \
        } \
        reinterpret_cast<CAST*>(expr.internal_t.get())->X(); \
        return nullptr; \
    });

    ARGUMENTlESS_VOID_METHOD_ON_CLASS(MAT, print, Mat<REAL_t>)
    ARGUMENTlESS_VOID_METHOD_ON_CLASS(MAT, grad, Mat<REAL_t>)
    ARGUMENTlESS_VOID_METHOD_ON_CLASS(GRAPH, backward, Graph<REAL_t>)

    GRAPH_BINARY_MAT_METHOD(mul)
    GRAPH_BINARY_MAT_METHOD(eltmul)
    GRAPH_BINARY_MAT_METHOD(add)
    GRAPH_BINARY_MAT_METHOD(sub)

    GRAPH_UNARY_MAT_METHOD(exp)
    GRAPH_UNARY_MAT_METHOD(log)
    GRAPH_UNARY_MAT_METHOD(tanh)
    GRAPH_UNARY_MAT_METHOD(sigmoid)
    GRAPH_UNARY_MAT_METHOD(sum)
    GRAPH_UNARY_MAT_METHOD(mean)

    GRAPH->add_method("softmax", [MAT](Expression& expr, vector<shared_ptr<Expression>>& args) -> shared_ptr<Expression> {
        if (args.size() == 1 && args[0]->type == MAT ) {
            return make_shared<Expression>(
                MAT,
                reinterpret_cast<Graph<REAL_t>*>(expr.internal_t.get())->softmax(
                    std::static_pointer_cast<mat>(args[0]->internal_t)
                )
            );
        } else if (args.size() == 2 && args[0]->type == MAT && args[1]->is_numeric()) {
            return make_shared<Expression>(
                MAT,
                reinterpret_cast<Graph<REAL_t>*>(expr.internal_t.get())->softmax(
                    std::static_pointer_cast<mat>(args[0]->internal_t),
                    args[1]->to_float()
                )
            );
        } else {
            throw std::runtime_error("Error: Wrong number of arguments for method softmax.");
        }
    });
}

int main () {
    declare_classes();
    unordered_map<string, shared_ptr<Expression>> locals;
    locals["G"] = Expression::parseExpression("Graph(true)", locals);


    // We will implement type checking:
    /*locals["type"] = make_shared<Expression>() {

    };*/

    string line;
    shared_ptr<Expression> expr;
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, line);
        line = trim(line);
        try {
            expr = parseLine(line, locals);
            if (expr != nullptr) {
                locals["$0"] = expr;
                std::cout << *expr << endl;
            }
        } catch (std::exception& e) {
            std::cout << e.what() << std::endl;
        }
    }
}
