#pragma once

// This.
#include <Configuration/Configuration.hpp>

// Forward declaration.
namespace MyEA::Memory
{
    template<class T, bool STD = true>
    void Copy(T const *ptr_array_source_received,
              T const *ptr_array_last_source_received,
              T *ptr_array_destination_received);

    template<class T, bool STD = true>
    void Fill(T *ptr_array_received,
              T *const ptr_array_last_received,
              T const value_received);
}

// This.
#include <../src/Reallocate/CPU/Reallocate.cpp>
