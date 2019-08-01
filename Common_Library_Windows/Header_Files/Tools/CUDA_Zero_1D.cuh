#pragma once

#include <Configuration/Configuration.hpp>

#include <device_launch_parameters.h>

template<typename T>
__global__ void kernel__Zero_1D(T *const ptr_array_outputs_received);

template<typename T>
__global__ void kernel__Zero_1D(size_t const size_received, T *const ptr_array_outputs_received);

template<typename T>
__global__ void kernel_while__Zero_1D(size_t const size_received, T *const ptr_array_outputs_received);

template<typename T>
__device__ void Zero_1D(size_t const size_received,
                                    T *ptr_array_outputs_received,
                                    struct dim3 const *const ptr_dimension_grid_received,
                                    struct dim3 const *const ptr_dimension_block_received);

template<typename T>
__global__ void kernel__Zero_Struct_1D(T *const ptr_array_outputs_received);

template<typename T>
__global__ void kernel__Zero_Struct_1D(size_t const size_received, T *const ptr_array_outputs_received);

template<typename T>
__global__ void kernel_while__Zero_Struct_1D(size_t const size_received, T *const ptr_array_outputs_received);

template<typename T>
__device__ void Zero_Struct_1D(size_t const size_received,
                                                T *ptr_array_outputs_received,
                                                struct dim3 const *const ptr_dimension_grid_received,
                                                struct dim3 const *const ptr_dimension_block_received);

template<typename T>
__global__ void kernel__Zero_1D__Inc(size_t const increment_step_received, T *const ptr_array_outputs_received);

template<typename T>
__global__ void kernel__Zero_1D__Inc(size_t const size_received,
                                                        size_t const increment_step_received,
                                                        T *const ptr_array_outputs_received);

template<typename T>
__global__ void kernel_while__Zero_1D__Inc(size_t const size_received,
                                                                 size_t const increment_step_received,
                                                                 T *const ptr_array_outputs_received);

template<typename T>
__device__ void Zero_1D(size_t const size_received,
                                    size_t const increment_step_received,
                                    T *ptr_array_outputs_received,
                                    struct dim3 const *const ptr_dimension_grid_received,
                                    struct dim3 const *const ptr_dimension_block_received);

#include <../Source_Files/Tools/CUDA_Zero_1D.cu>
