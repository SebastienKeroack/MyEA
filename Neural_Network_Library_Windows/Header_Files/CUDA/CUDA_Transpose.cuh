#pragma once

#include <Tools/Configuration.hpp>

#include <device_launch_parameters.h>

namespace Transpose
{
    template<typename T>
    __device__ void Transpose_Square(size_t const size_received,
                                                        size_t const width_received,
                                                        T *const ptr_array_outputs_received,
                                                        T const *const ptr_array_inputs_received,
                                                        struct dim3 const *const ptr_dimension_grid_recieved,
                                                        struct dim3 const *const ptr_dimension_block_recieved);

    template<typename T>
    __device__ void Transpose_Rectangular(size_t const size_received,
                                                            size_t const columns_length_received,
                                                            size_t const rows_length_received,
                                                            T *const ptr_array_outputs_received,
                                                            T const *const ptr_array_inputs_received,
                                                            struct dim3 const *const ptr_dimension_grid_recieved,
                                                            struct dim3 const *const ptr_dimension_block_recieved);

    template<typename T>
    __device__ void Transpose(size_t const size_received,
                                            size_t const columns_length_received,
                                            size_t const rows_length_received,
                                            T *const ptr_array_outputs_received,
                                            T const *const ptr_array_inputs_received,
                                            struct dim3 const *const ptr_dimension_grid_recieved,
                                            struct dim3 const *const ptr_dimension_block_recieved);
}

#include <../Source_Files/CUDA/CUDA_Transpose.cu>
