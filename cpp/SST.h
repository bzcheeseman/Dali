#ifndef SST_MAT_H
#define SST_MAT_H

#include <memory>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include "gzstream.h"
#include "utils.h"

namespace SST {
	class AnnotatedParseTree {
        public:
            typedef std::shared_ptr<AnnotatedParseTree> shared_tree;
            std::vector<shared_tree> children;
            std::weak_ptr<AnnotatedParseTree> parent;
            uint label;
            uint depth;
            uint udepth;
            bool has_parent;
            std::string sentence;
            AnnotatedParseTree(uint);
            AnnotatedParseTree(uint, shared_tree);
    };
    void replace_char_by_char(std::vector<char>&, const char&, const char&);
	AnnotatedParseTree::shared_tree create_tree_from_string(const std::string&);
	template<typename T>
	void stream_to_sentiment_treebank(T&, std::vector<AnnotatedParseTree::shared_tree>&);
	std::vector<AnnotatedParseTree::shared_tree> load(const std::string&);
}

std::ostream &operator <<(std::ostream &, const SST::AnnotatedParseTree&);

#endif