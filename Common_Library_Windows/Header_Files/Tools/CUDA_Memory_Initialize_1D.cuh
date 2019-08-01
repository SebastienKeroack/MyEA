#pragma once

#include <Configuration/Configuration.hpp>

#include <device_launch_parameters.h>

namespace MyEA::Memory
{
    template<typename T>
    __device__ void Memory_Initialize_Index(size_t const size_received,
                                                               T *const ptr_array_outputs_received,
                                                               struct dim3 const *const ptr_dimension_grid_received,
                                                               struct dim3 const *const ptr_dimension_block_received);
    
    template<typename T>
    __device__ void Memory_Initialize_Index_Shift(size_t const size_received,
                                                                       size_t const shift_received,
                                                                       T *const ptr_array_outputs_received,
                                                                       struct dim3 const *const ptr_dimension_grid_received,
                                                                       struct dim3 const *const ptr_dimension_block_received);

    template<typename T>
    __device__ void Memory_Initialize_Index_Offset(size_t const size_received,
                                                                          size_t const offSet__received,
                                                                          T *const ptr_array_outputs_received,
                                                                          struct dim3 const *const ptr_dimension_grid_received,
                                                                          struct dim3 const *const ptr_dimension_block_received);

    template<typename T>
    __device__ void Memory_Initialize_Index_Transposed(size_t const size_received,
                                                                                  T *const ptr_array_outputs_received,
                                                                                  struct dim3 const *const ptr_dimension_grid_received,
                                                                                  struct dim3 const *const ptr_dimension_block_received);
}

#include <../Source_Files/Tools/CUDA_Memory_Initialize_1D.cu>
