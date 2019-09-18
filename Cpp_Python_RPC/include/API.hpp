#pragma once

// Common_Library.
#include <Configuration/Exports.hpp>
#include <Configuration/DTypes.hpp>
#include <Enums/Enum_Type_Dataset.hpp>

DLL_API bool API__Cpp_Python_RPC__Initialize(wchar_t const *const host, wchar_t const *const script);

DLL_API bool API__Cpp_Python_RPC__Open(void);

DLL_API bool API__Cpp_Python_RPC__Close(void);

DLL_API bool API__Cpp_Python_RPC__Concatenate_X(T_ const *const inputs, size_t const length);

DLL_API bool API__Cpp_Python_RPC__Concatenate_Y(T_ const *const inputs, size_t const length);

DLL_API size_t API__Cpp_Python_RPC__Sizeof_T(void);

DLL_API T_ API__Cpp_Python_RPC__Predict(unsigned int const past_action,
                                        T_ const *const inputs,
                                        size_t const length,
                                        size_t const seq_w);

DLL_API T_ API__Cpp_Python_RPC__Metric_Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset);

DLL_API T_ API__Cpp_Python_RPC__Metric_Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset);