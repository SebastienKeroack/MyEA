// pch.h: This is a precompiled header file.
// Files listed below are compiled only once, improving build performance for future builds.
// This also affects IntelliSense performance, including code completion and many code browsing features.
// However, files listed here are ALL re-compiled if any one of them is updated between builds.
// Do not add files here that you will be updating frequently as this negates the performance advantage.

#ifndef PCH_H
#define PCH_H

// add headers that you want to pre-compile here
#include "framework.hpp"

// Boost.
#include <boost/locale.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/algorithm/string/replace.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

#if defined(DLL_EXPORTS)
    #define DLL_API __declspec(dllexport)
#else
    #define DLL_API __declspec(dllimport)
#endif // COMPILE_DLL_EXPORTS

#include <fstream>

#endif //PCH_H
