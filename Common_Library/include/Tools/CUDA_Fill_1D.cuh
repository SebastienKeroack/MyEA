#pragma once

#include <Configuration/Configuration.hpp>

#include <device_launch_parameters.h>

namespace MyEA::Memory
{
    template<typename T> __global__
    void kernel__Fill_1D(T *const ptr_array_outputs_received, T const constant_received);

    template<typename T> __global__
    void kernel__Fill_1D(size_t const size_received,
                         T *const ptr_array_outputs_received,
                         T const constant_received);

    template<typename T> __global__
    void kernel_while__Fill_1D(size_t const size_received,
                               T *const ptr_array_outputs_received,
                               T const constant_received);

    template<typename T> __device__
    void Fill_1D(size_t const size_received,
                 T *ptr_array_outputs_received,
                 T const constant_received,
                 struct dim3 const *const ptr_dimension_grid_received,
                 struct dim3 const *const ptr_dimension_block_received);
}

#include <../src/Tools/CUDA_Fill_1D.cu>
