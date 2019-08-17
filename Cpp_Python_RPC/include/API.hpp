#pragma once

#include <Configuration/Exports.hpp>

#include <Configuration/DTypes.hpp>

DLL_API bool API__Cpp_Python_RPC__Initialize(void);

DLL_API bool API__Cpp_Python_RPC__Open(void);

DLL_API bool API__Cpp_Python_RPC__Close(void);

DLL_API T_ API__Cpp_Python_RPC__Predict(void);