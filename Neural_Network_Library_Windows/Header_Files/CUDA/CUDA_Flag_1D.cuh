#pragma once

#include <Configuration/Configuration.hpp>

#include <device_launch_parameters.h>

template<typename T>
__global__ void kernel__Flag_1D(bool const *const ptr_array_flag_received, T *const ptr_array_to_one_received);
    
template<typename T>
__global__ void kernel__Flag_1D(size_t const size_received,
                                                bool const *const ptr_array_flag_received,
                                                T *const ptr_array_to_one_received);

template<typename T>
__global__ void kernel_while__Flag_1D(size_t const size_received,
                                                         bool const *const ptr_array_flag_received,
                                                         T *const ptr_array_to_one_received);

template<typename T>
__device__ void Flag_1D(size_t const size_received,
                                    bool const *ptr_array_flag_received,
                                    T *ptr_array_to_flag_received,
                                    struct dim3 const *const ptr_dimension_grid_received,
                                    struct dim3 const *const ptr_dimension_block_received);

#include <../Source_Files/CUDA/CUDA_Flag_1D.cu>
