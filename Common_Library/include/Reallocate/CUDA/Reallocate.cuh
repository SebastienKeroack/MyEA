#pragma once

// Standard.
#include <device_launch_parameters.h>

// This.
#include <Configuration/Configuration.hpp>

// Forward declaration.
namespace MyEA::Memory
{
    // TODO: CUDA Parallelism.
    template<class T, bool STD = true> __host__ __device__
    void Copy(T const *ptr_array_source_received,
              T const *ptr_array_last_source_received,
              T *ptr_array_destination_received);

    // TODO: CUDA Parallelism.
    template<class T, bool STD = true> __host__ __device__
    void Fill(T *ptr_array_received,
              T *const ptr_array_last_received,
              T const value_received);
}

// This.
#include <../src/Reallocate/CUDA/Reallocate.cu>
