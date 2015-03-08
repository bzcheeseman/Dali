#include <unordered_map>
#include <string>
#include <memory>
#include <exception>
#include <sstream>
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
	static const uint NIL_T   = 0;
	static const uint MAT_T   = 1;
	static const uint GRAPH_T = 2;
	static const uint INT_T   = 3;
	static const uint FLOAT_T = 4;
	static const uint BOOL_T  = 5;
	public:
		shared_ptr<void> internal_t;
		unsigned int type;


		static vector<Expression> eval_arguments(const string& line, unordered_map<string, shared_ptr<Expression>> locals) {
			// get the relevant arguments.
			auto args = get_arguments(line);
			vector<Expression> evaled_args;
			evaled_args.reserve(args.size());

			for (auto& arg : args) {
				evaled_args.emplace_back(arg, locals);
			}

			return evaled_args;

		}

		static vector<string> get_arguments(const string& line) {
			vector<string> args;
			stringstream ss(line);
			vector<char> tokens;
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
					}
				}
			}
			return args;
		}

		bool is_numeric() const {
			return (type == INT_T || type == FLOAT_T);
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
			if (line == "true" || line == "false" ||
				is_number(line) ) {
				return true;
			}
			return false;
		}

		Expression(string _repr, unordered_map<string, std::shared_ptr<Expression>>& locals) {
			if (is_literal(_repr)) {
				if (startswith(_repr, "true")) {
					type = BOOL_T;
					internal_t = make_shared<bool>(true);
				} else if (startswith(_repr, "false")) {
					type = BOOL_T;
					internal_t = make_shared<bool>(false);
				} else if (is_number(_repr)) {
					if (_repr.find('.') != std::string::npos) {
						type = FLOAT_T;
						internal_t = make_shared<REAL_t>( from_string<REAL_t>(_repr) );
					} else {
						type = INT_T;
						internal_t = make_shared<int>( from_string<int>(_repr) );
					}
				} else {
					if (locals.find(_repr) != locals.end()) {
						auto prev_var = locals.at(_repr);
						type       = prev_var->type;
						internal_t = prev_var->internal_t;
					} else {
						stringstream ss;
						ss << "Error: Unknown variable name or function \""
						   << _repr << "\"";
						throw std::runtime_error(ss.str());
					}
				}
			} else {
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
				}
			}
		}

		string type_name () const {
			switch (type) {
				case MAT_T:
					return "Mat";
				case GRAPH_T:
					return "Graph";
				case INT_T:
					return "Int";
				case FLOAT_T:
					return "Float";
				case BOOL_T:
					return "Boolean";
				default:
					return "??";
			}
		}
		Mat<REAL_t>* to_mat () const  {
			if (type == MAT_T) {
				return reinterpret_cast<mat*>(internal_t.get());
			} else {
				THROW_INCOMPATIBLE_TYPE(Mat)
			}
		}
		bool to_bool() const  {
			if (type == BOOL_T) {
				return *reinterpret_cast<int*>(internal_t.get());
			} else if (type == INT_T) {
				return int_as_bool();
			} else {
				THROW_INCOMPATIBLE_TYPE(Boolean)
			}
			return *reinterpret_cast<bool*>(internal_t.get());
		}
		int to_int () const  {
			if (type == INT_T) {
				return *reinterpret_cast<int*>(internal_t.get());
			} else if (type == FLOAT_T) {
				return (int) *reinterpret_cast<REAL_t*>(internal_t.get());
			} else {
				THROW_INCOMPATIBLE_TYPE(Integer)
			}
		}
		bool int_as_bool() const  {
			if (type == INT_T) {
				return (*reinterpret_cast<int*>(internal_t.get())) > 0;
			} else {
				THROW_INCOMPATIBLE_TYPE(Boolean)
			}
		}
		REAL_t to_float () const  {
			if (type == FLOAT_T) {
				return *reinterpret_cast<REAL_t*>(internal_t.get());
			} else if (type == INT_T) {
				return (REAL_t)*reinterpret_cast<REAL_t*>(internal_t.get());
			} else {
				THROW_INCOMPATIBLE_TYPE(Float)
			}
		}
		Graph<REAL_t>* to_graph () const  {
			if (type == GRAPH_T) {
				return reinterpret_cast<Graph<REAL_t>*>(internal_t.get());
			} else {
				THROW_INCOMPATIBLE_TYPE(Graph)
			}
		}
		string to_string () const  {
			stringstream ss;
			switch (type) {
				case MAT_T:
					ss << * this->to_mat();
					break;
				case GRAPH_T:
					ss << * this->to_graph();
					break;
				case INT_T:
					ss << this->to_int();
					break;
				case FLOAT_T:
					ss << this->to_float();
					break;
				case BOOL_T:
					ss << this->to_bool();
					break;
				default:
					ss << "??";
			}
			return ss.str();
		}
};

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
					auto variable = make_shared<Expression>(
						assignment_string, locals);
					locals[string(varname)] = variable;
					return nullptr;
				}
			} else {
				throw std::runtime_error("SyntaxError: Empty variable name for assignment.");
			}
		} else {
			return make_shared<Expression>(line, locals);
		}
	} else {
		return nullptr;
	}
}

int main () {
	Graph<REAL_t> G(true);
	unordered_map<string, shared_ptr<Expression>> locals;
	string line;
	shared_ptr<Expression> expr;
	while (true) {
		std::cout << "> ";
		std::getline(std::cin, line);
		line = trim(line);
		try {
			expr = parseLine(line, locals);
			if (expr != nullptr) {
				std::cout << *expr << endl;
			}
		} catch (std::exception& e) {
			std::cout << e.what() << std::endl;
		}
	}
}