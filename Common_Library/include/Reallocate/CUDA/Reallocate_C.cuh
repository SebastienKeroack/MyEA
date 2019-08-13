#pragma once

// This.
#include <Configuration/Configuration.hpp>

// Forward declaration.
namespace MyEA::Memory::C
{
    // TODO: CUDA Parallelism.
    template<class T, bool CPY = true, bool SET = true> __host__ __device__
    T* Reallocate(T *ptr_array_received,
                  size_t const new_size_t_received,
                  size_t const old_size_t_received);
}

// This.
#include <../src/Reallocate/CUDA/Reallocate_C.cu>
