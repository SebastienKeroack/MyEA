#pragma once

#include <Configuration/Configuration.hpp>

#include <device_launch_parameters.h>

namespace MyEA::Memory
{
    template<typename T>
    __global__ void kernel__Memory_Copy_1D( T *const ptr_array_destination_received, T const *const ptr_array_source_received);

    template<typename T>
    __global__ void kernel__Memory_Copy_1D(size_t const size_received,
                                                                   T *const ptr_array_destination_received,
                                                                   T const *const ptr_array_source_received);

    template<typename T>
    __global__ void kernel_while__Memory_Copy_1D(size_t const size_received,
                                                                            T *const ptr_array_destination_received,
                                                                            T const *const ptr_array_source_received);

    template<typename T>
    __device__ void Memory_Copy_1D(size_t const size_received,
                                                        T *const ptr_array_destination_received,
                                                        T const *const ptr_array_source_received,
                                                        struct dim3 const *const ptr_dimension_grid_received,
                                                        struct dim3 const *const ptr_dimension_block_received);
}

#include <../Source_Files/Tools/CUDA_Memory_Copy_1D.cu>
