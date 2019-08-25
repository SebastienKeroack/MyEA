#pragma once

// Common_Library.
#include <Configuration/Exports.hpp>
#include <Configuration/DTypes.hpp>
#include <Enums/Enum_Type_Dataset.hpp>

DLL_API bool API__Cpp_Python_RPC__Initialize(void);

DLL_API bool API__Cpp_Python_RPC__Open(void);

DLL_API bool API__Cpp_Python_RPC__Close(void);

DLL_API bool API__Cpp_Python_RPC__Merge_Y(T_ const *const inputs, size_t const length);

DLL_API size_t API__Cpp_Python_RPC__Sizeof_T(void);

DLL_API T_ API__Cpp_Python_RPC__Predict(T_ const *const inputs, size_t const length);

DLL_API T_ API__Cpp_Python_RPC__Metric_Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset);

DLL_API T_ API__Cpp_Python_RPC__Metric_Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset);