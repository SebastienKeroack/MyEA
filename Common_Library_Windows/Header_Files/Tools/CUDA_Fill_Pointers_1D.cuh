#pragma once

#include <Configuration/Configuration.hpp>

#include <device_launch_parameters.h>

template<typename T>
__global__ void kernel__Fill_Pointers_1D(T *const ptr_array_outputs_received);

template<typename T>
__global__ void kernel__Fill_Pointers_1D(size_t const size_received, T *const ptr_array_outputs_received);

template<typename T>
__global__ void kernel_while__Fill_Pointers_1D(size_t const size_received, T *const ptr_array_outputs_received);

template<typename T>
__device__ void Fill_Pointers_1D(size_t const size_received,
                                                T *ptr_array_outputs_received,
                                                struct dim3 const *const ptr_dimension_grid_received,
                                                struct dim3 const *const ptr_dimension_block_received);

#include <../Source_Files/Tools/CUDA_Fill_Pointers_1D.cu>
