#include "SST.h"

using std::string;
using std::vector;
using std::stringstream;
using std::make_shared;

namespace SST {

    AnnotatedParseTree::AnnotatedParseTree(uint _depth) : depth(_depth), udepth(1), has_parent(false) {}
    AnnotatedParseTree::AnnotatedParseTree(uint _depth, shared_tree _parent) : depth(_depth), udepth(1), has_parent(true), parent(_parent) {}

    void replace_char_by_char(std::vector<char>& list, const char& target, const char& replacement) {
        for (auto& c : list)
            if (c == target) c = replacement;
    }

    void AnnotatedParseTree::add_general_child(AnnotatedParseTree::shared_tree child) {
        general_children.emplace_back(child);
    }

    void AnnotatedParseTree::add_words_to_vector(vector<string>& list) const {
        if (children.size() == 0) {
            list.emplace_back(sentence);
        } else {
            for (auto& child : children)
                child->add_words_to_vector(list);
        }
    }

    std::pair<vector<string>, uint> AnnotatedParseTree::to_labeled_pair() const {
        std::pair<vector<string>, uint> pair;
        pair.second = label;
        add_words_to_vector(pair.first);
        return pair;
    }

    AnnotatedParseTree::shared_tree create_tree_from_string(const string& line) {
        int depth = 0;
        bool awaiting_num = false;
        std::vector<char> current_word;
        AnnotatedParseTree::shared_tree root = nullptr;
        auto current_node = root;
        stringstream ss(line);
        char ch;
        const char left_parenthesis  = '(';
        const char right_parenthesis = ')';
        const char space             = ' ';

        while (ss) {
            ch = ss.get();
            if (awaiting_num) {
                current_node->label = (uint)((int) (ch - '0'));
                awaiting_num = false;
            } else {
                if (ch == left_parenthesis) {
                    if (++depth > 1) {
                        // replace current head node by this node:
                        current_node->children.emplace_back(make_shared<AnnotatedParseTree>(depth, current_node));
                        current_node = current_node->children.back();
                        root->add_general_child(current_node);
                    } else {
                        root = make_shared<AnnotatedParseTree>(depth);
                        current_node = root;
                    }
                    awaiting_num = true;
                } else if (ch == right_parenthesis) {
                    // assign current word:
                    if (current_word.size() > 0) {
                        replace_char_by_char(current_word, '\xa0', space);
                        current_node->sentence = string(current_word.begin(), current_word.end());
                        current_node->udepth   = 1;
                        // erase current word
                        current_word.clear();
                    }
                    // go up a level:
                    depth--;
                    if (current_node->has_parent) {
                        uint& current_node_udepth = current_node->udepth;
                        current_node = current_node->parent.lock();
                        current_node->udepth = std::max(current_node_udepth+1, current_node->udepth);
                    } else {
                        current_node = nullptr;
                    }
                } else if (ch == space) {
                    // ignore spacing
                    continue;
                } else {
                    // add to current read word
                    current_word.emplace_back(ch);
                }
            }
        }
        if (depth != 0)
            throw std::invalid_argument("ParseError: Not an equal amount of closing and opening parentheses");
        return root;
    }

    template<typename T>
    void stream_to_sentiment_treebank(T& fp, vector<AnnotatedParseTree::shared_tree>& trees) {
        string line;
        while (std::getline(fp, line))
            trees.emplace_back( create_tree_from_string(line) );
    }

    vector<AnnotatedParseTree::shared_tree> load(const string& fname) {
        vector<AnnotatedParseTree::shared_tree> trees;
        if (utils::file_exists(fname)) {
            if (utils::is_gzip(fname)) {
                igzstream fpgz(fname.c_str(), std::ios::in | std::ios::binary);
                stream_to_sentiment_treebank(fpgz, trees);
            } else {
                std::fstream fp(fname, std::ios::in | std::ios::binary);
                stream_to_sentiment_treebank(fp, trees);
            }
        } else {
            stringstream error_msg;
            error_msg << "FileNotFound: No file found at \"" << fname << "\"";
            throw std::runtime_error(error_msg.str());
        }
        return trees;
    }
}

std::ostream &operator <<(std::ostream &os, const SST::AnnotatedParseTree&v) {
    if (v.children.size() > 0) {
       os << "(" << v.label << " ";
       for (auto& child : v.children)
           os << *child;
       return os << ")";
    } else {
        return os << "(" << v.label << " " << v.sentence << ") ";
    }
}