#include <unordered_map>
#include <string>
#include <memory>
#include <exception>
#include <sstream>
#include <functional>
#include "core/Mat.h"
#include "core/Graph.h"

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

Very rudimentary REPL for Dalhi / RecurrentJS.

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
                    vector<char> tokens;
                    vector<char> call_tokens;
                    int depth = 0;
                    char ch;
                    while (ss) {
                        ch = ss.get();
                        if (ch == '(') {
                            if (depth > 0) {
                                tokens.emplace_back(ch);
                            }
                            depth += 1;
                            continue;
                        }
                        if (ch == ')') {
                            depth -= 1;
                            if (depth < 0) {
                                throw std::runtime_error("SyntaxError: Unequal number of opening and closing parentheses.");
                            }
                            if (depth == 0) {
                                if (tokens.size() > 0) {
                                    args.emplace_back(tokens.begin(), tokens.end());
                                    tokens.clear();
                                    break;
                                }
                            }
                            if (depth > 0) {
                                tokens.emplace_back(ch);
                            }
                            continue;
                        }
                        if (ch == ',') {
                            if (depth == 1) {
                                if (tokens.size() == 0) {
                                    throw std::runtime_error("ParseError: Empty argument to function call.");
                                } else {
                                    args.emplace_back(tokens.begin(), tokens.end());
                                    tokens.clear();
                                }
                            } else {
                                tokens.emplace_back(ch);
                            }
                            continue;
                        }
                        if (ch != ' ') {
                            if (depth > 0) {
                                tokens.emplace_back(ch);
                            } else {
                                std::cout << ch << std::endl;
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
                return *reinterpret_cast<int*>(internal_t.get());
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
                return (REAL_t)*reinterpret_cast<REAL_t*>(internal_t.get());
            } else {
                THROW_INCOMPATIBLE_TYPE(Float)
            }
        }
        string to_string () const  {
            return type->to_string(*this);
        }

        // Graph<REAL_t>* to_graph () const  {
        //     if (type == GRAPH_T) {
        //         return reinterpret_cast<Graph<REAL_t>*>(internal_t.get());
        //     } else {
        //         THROW_INCOMPATIBLE_TYPE(Graph)
        //     }
        // }
        // Mat<REAL_t>* to_mat () const  {
        //     if (type == MAT_T) {
        //         return reinterpret_cast<mat*>(internal_t.get());
        //     } else {
        //         THROW_INCOMPATIBLE_TYPE(Mat)
        //     }
        // }
        static shared_ptr<Expression> parseExpression(string _repr, unordered_map<string, std::shared_ptr<Expression>>& locals);

        Expression(shared_ptr<WrappedCppClass> _type, shared_ptr<void> _internal_t) : internal_t(_internal_t), type(_type) {}
        Expression(shared_ptr<WrappedCppClass> _type, shared_ptr<void> _internal_t, function<shared_ptr<Expression>(Expression&, vector<shared_ptr<Expression>>&)> _call) : internal_t(_internal_t), type(_type), call(_call) {}
};

/*class Expression::LambdaExpression : public Expression {
    public:
        function<Expression(vector<Expression>&)> call;
        LambdaExpression(string _repr, unordered_map<string, std::shared_ptr<Expression>>& locals, function<Expression(vector<Expression>&)> _call) : Expression(_repr, locals) : call(_call) {
        }
        Expression operator()(vector<Expression>& args) {
            return call(args);
        }
};*/

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
        /*
        if (startswith(_repr, "Graph")) {
            auto argnum = eval_arguments(_repr, locals);
            if (argnum.size() == 0) {
                type = GRAPH_T;
                internal_t = make_shared<Graph<REAL_t>>();
            } else if (argnum.size() == 1) {
                type = GRAPH_T;
                internal_t = make_shared<Graph<REAL_t>>( argnum[0].to_bool() );
            } else {
                throw std::runtime_error("Error: incompatible number or arguments for Graph( bool ): 0 to 1.");
            }
        } else if (startswith( _repr, "Mat")) {
            auto argnum = eval_arguments(_repr, locals);
            
            if (argnum.size() < 2 || argnum.size() > 5) {
                throw std::runtime_error("Error: incompatible number or arguments for Mat( int, int ): 2 to 5.");
            } else if (argnum.size() == 2) {
                if (argnum[0].type == INT_T && argnum[1].type == INT_T) {
                    type = MAT_T;
                    internal_t = make_shared<Mat<REAL_t>>( argnum[0].to_int(), argnum[1].to_int() );
                } else {
                    throw std::runtime_error("Error: incompatible arguments for Mat( int, int )");
                }
            } else if (argnum.size() == 3) {
                if (argnum[0].type == INT_T && argnum[1].type == INT_T && argnum[2].type == BOOL_T) {
                    type = MAT_T;
                    internal_t = make_shared<Mat<REAL_t>>(
                        argnum[0].to_int(),
                        argnum[1].to_int(),
                        argnum[2].to_bool()
                    );
                } else if (argnum[0].type == INT_T && argnum[1].type == INT_T && argnum[2].is_numeric()) {
                    type = MAT_T;
                    internal_t = make_shared<Mat<REAL_t>>(
                        argnum[0].to_int(),
                        argnum[1].to_int(),
                        argnum[2].to_float()
                    );
                } else {
                    throw std::runtime_error("Error: incompatible arguments for Mat( int, int, float) / Mat( int, int, bool)");
                }
            } else if (argnum.size() == 4) {
                if (argnum[0].type == INT_T && argnum[1].type == INT_T && argnum[2].is_numeric() && argnum[3].is_numeric()) {
                    type = MAT_T;
                    internal_t = make_shared<Mat<REAL_t>>(
                        argnum[0].to_int(),
                        argnum[1].to_int(),
                        argnum[2].to_float(),
                        argnum[3].to_float()
                    );
                } else {
                    throw std::runtime_error("Error: incompatible arguments for Mat( int, int, float, float)");
                }
            } else {
                throw std::runtime_error("Error: incompatible number or arguments for Mat( int, int ): 2 to 5.");
            }
        } else {
            // deal with method calls now
            stringstream ss;
            ss << "Error: Unknown type: \"" << _repr << "\"";
            throw std::runtime_error(ss.str());
        }*/
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

    /*auto LAMBDA = make_shared<Expression::WrappedCppClass>("Lambda",
        vector<Expression::ExpressionConstructor>(), (uint) Expression::WrappedCppClass::classes.size())

    Expression::WrappedCppClass::classes.emplace_back(LAMBDA);*/

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

    auto graph_constructors = {
        Expression::ExpressionConstructor( zero_arg, [](vector<shared_ptr<Expression>>& args) {return make_shared<Graph<REAL_t>>();} ),
        Expression::ExpressionConstructor( single_arg_boolean, [](vector<shared_ptr<Expression>>& args) { return make_shared<Graph<REAL_t>>(args[0]->to_bool());}),
    };

    Expression::WrappedCppClass::classes.emplace_back(make_shared<Expression::WrappedCppClass>("Graph", graph_constructors, (uint) Expression::WrappedCppClass::classes.size(), [](const Expression& self) {
        stringstream ss;
        ss << * reinterpret_cast<Graph<REAL_t>*>(self.internal_t.get());
        return ss.str();
    }));

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

    MAT->add_method("print", [](Expression& expr, vector<shared_ptr<Expression>>& args) -> shared_ptr<Expression> {
        if (args.size() > 0) {
            throw std::runtime_error("Error: Wrong number of arguments for method print.");
        }
        reinterpret_cast<mat*>(expr.internal_t.get())->print();
        return nullptr;
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