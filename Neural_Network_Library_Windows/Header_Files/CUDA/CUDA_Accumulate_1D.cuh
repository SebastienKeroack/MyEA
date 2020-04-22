#pragma once

#include <Configuration/Configuration.hpp>

#include <device_launch_parameters.h>

namespace Accumulate
{
    template<typename T>
    __global__ void kernel__Accumulate_X_X_1D(T *ptr_array_outputs_received, T const *ptr_array_inputs_received);

    template<typename T>
    __global__ void kernel__Accumulate_X_X_1D(size_t const size_received,
                                                                      T *ptr_array_outputs_received,
                                                                      T const *ptr_array_inputs_received);

    template<typename T>
    __global__ void kernel_while__Accumulate_X_X_1D(size_t const size_received,
                                                                               T *ptr_array_outputs_received,
                                                                               T const *ptr_array_inputs_received);

    template<typename T>
    __device__ void Accumulate_X_X_1D(size_t const size_received,
                                                          T *ptr_array_outputs_received,
                                                          T const *ptr_array_inputs_received,
                                                          struct dim3 const *const ptr_dimension_grid_received,
                                                          struct dim3 const *const ptr_dimension_block_received);
}

#include <../Source_Files/CUDA/CUDA_Accumulate_1D.cu>
