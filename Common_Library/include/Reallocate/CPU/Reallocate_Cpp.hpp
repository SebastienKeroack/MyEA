#pragma once

// This.
#include <Configuration/Configuration.hpp>

// Forward declaration.
namespace MyEA::Memory::Cpp
{
    template<class T>
    void Fill_Nullptr(T *ptr_array_received, T const *const ptr_array_last_received);

    template<class T, bool CPY = true, bool SET = true>
    T* Reallocate(T *ptr_array_received,
                  size_t const new_size_received,
                  size_t const old_size_received);

    template<class T, bool CPY = true>
    T* Reallocate_Objects(T *ptr_array_received,
                          size_t const new_size_received,
                          size_t const old_size_received);

    template<class T, bool CPY = true, bool SET = true>
    T* Reallocate_PtOfPt(T *ptr_array_received,
                         size_t const new_size_received,
                         size_t const old_size_received);
}

// This.
#include <../src/Reallocate/CPU/Reallocate_Cpp.cpp>
