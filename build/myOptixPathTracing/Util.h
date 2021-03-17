#pragma once

#include <string>
#include <stdexcept>

// Throw error 
inline void Throw(const std::string& msg) {
	throw std::runtime_error(msg);
}

// Assertion
/** \brief Make sure the `condition` is true. */
inline void Assert(bool condition, const std::string& msg) {
	if (!condition) Throw(msg);
}