#include "jit_runner.h"

JITRunner::JITRunner(Array root, const std::vector<Array>& leaves) :
	JITNode(root.shape(), root.dtype()),
	root_(root), leaves_(leaves) {
	if (std::dynamic_pointer_cast<JITRunner>(root.expression())) {
		throw std::runtime_error("JITRunner should not contain a JITRunner.");
	}
}

std::vector<Array> JITRunner::arguments() const {
	return leaves_;
}
// TODO(jonathan): add pretty-printing here to keep track of what was jitted or not.

expression_ptr JITRunner::copy() const {
    return std::make_shared<JITRunner>(*this);
}

memory::Device JITRunner::preferred_device() const {
	return root_.preferred_device();
}
