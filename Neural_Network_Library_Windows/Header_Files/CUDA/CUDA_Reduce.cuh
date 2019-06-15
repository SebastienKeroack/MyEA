#pragma once

#include <Tools/Configuration.hpp>

#include <device_launch_parameters.h>

namespace Reduce
{
    template<typename T>
    __device__ inline void Launch_Reduce(size_t const size_received,
                                                            T *const ptr_array_outputs_received,
                                                            T const *const ptr_array_inputs_received,
                                                            struct dim3 const *const ptr_dimension_grid_recieved,
                                                            struct dim3 const *const ptr_dimension_block_recieved);

    template<typename T>
    __device__ void Reduce(size_t const size_received,
                                        size_t const stride_dim3_received,
                                        T *const ptr_array_outputs_received,
                                        T const *const ptr_array_inputs_received,
                                        struct dim3 const *const ptr_dimension_grid_recieved,
                                        struct dim3 const *const ptr_dimension_block_recieved);

    template<typename T>
    __device__ inline void Launch_Reduce_Square(size_t const size_received,
                                                                        T *const ptr_array_outputs_received,
                                                                        T const *const ptr_array_to_reduce_square_received,
                                                                        struct dim3 const *const ptr_dimension_grid_recieved,
                                                                        struct dim3 const *const ptr_dimension_block_recieved);

    template<typename T>
    __device__ void Reduce_Square(size_t const size_received,
                                                    size_t const stride_dim3_received,
                                                    T *const ptr_array_outputs_received,
                                                    T const *const ptr_array_to_reduce_square_received,
                                                    struct dim3 const *const ptr_dimension_grid_recieved,
                                                    struct dim3 const *const ptr_dimension_block_recieved);
    
    template<typename T>
    __device__ inline void Launch_Reduce_XX(size_t const size_received,
                                                                T *const ptr_array_outputs_received,
                                                                T const *const ptr_array_X0_received,
                                                                T const *const ptr_array_X1_received,
                                                                struct dim3 const *const ptr_dimension_grid_recieved,
                                                                struct dim3 const *const ptr_dimension_block_recieved);

    template<typename T>
    __device__ void Reduce_XX(size_t const size_received,
                                            size_t const stride_dim3_received,
                                            T *const ptr_array_outputs_received,
                                            T const *const ptr_array_X0_received,
                                            T const *const ptr_array_X1_received,
                                            struct dim3 const *const ptr_dimension_grid_recieved,
                                            struct dim3 const *const ptr_dimension_block_recieved);
    
    template<typename T>
    __device__ inline void Launch_Reduce_XZ(size_t const size_received,
                                                                size_t const stride_Z_received,
                                                                T *const ptr_array_outputs_received,
                                                                T const *const ptr_array_X_received,
                                                                T const *const ptr_array_Z_received,
                                                                struct dim3 const *const ptr_dimension_grid_recieved,
                                                                struct dim3 const *const ptr_dimension_block_recieved);

    template<typename T>
    __device__ void Reduce_XZ(size_t const size_received,
                                            size_t const stride_dim3_received,
                                            size_t const stride_Z_received,
                                            T *const ptr_array_outputs_received,
                                            T const *const ptr_array_X_received,
                                            T const *const ptr_array_Z_received,
                                            struct dim3 const *const ptr_dimension_grid_recieved,
                                            struct dim3 const *const ptr_dimension_block_recieved);
    
    template<typename T>
    __device__ inline void Launch_Reduce_Z0Z1(size_t const size_received,
                                                                    size_t const stride_Z0_received,
                                                                    size_t const stride_Z1_received,
                                                                    T *const ptr_array_outputs_received,
                                                                    T const *const ptr_array_Z0_received,
                                                                    T const *const ptr_array_Z1_received,
                                                                    struct dim3 const *const ptr_dimension_grid_recieved,
                                                                    struct dim3 const *const ptr_dimension_block_recieved);

    template<typename T>
    __device__ void Reduce_Z0Z1(size_t const size_received,
                                                size_t const stride_dim3_received,
                                                size_t const stride_Z0_received,
                                                size_t const stride_Z1_received,
                                                T *const ptr_array_outputs_received,
                                                T const *const ptr_array_Z0_received,
                                                T const *const ptr_array_Z1_received,
                                                struct dim3 const *const ptr_dimension_grid_recieved,
                                                struct dim3 const *const ptr_dimension_block_recieved);
    
    template<typename T>
    __device__ inline void Launch_Reduce_Array(size_t const size_received,
                                                                     size_t const stride_array_received,
                                                                     T *const ptr_array_IO_received,
                                                                     struct dim3 const *const ptr_dimension_grid_recieved,
                                                                     struct dim3 const *const ptr_dimension_block_recieved,
                                                                     struct dim3 const *const ptr_dimension_grid_reduce_array_recieved,
                                                                     struct dim3 const *const ptr_dimension_block_reduce_array_recieved);

    template<typename T>
    __device__ void Reduce_Array(size_t const size_received,
                                                 size_t const stride_array_received,
                                                 size_t const stride_dim3_received,
                                                 T *const ptr_array_IO_received,
                                                 struct dim3 const *const ptr_dimension_grid_recieved,
                                                 struct dim3 const *const ptr_dimension_block_recieved,
                                                 struct dim3 const *const ptr_dimension_grid_reduce_array_recieved,
                                                 struct dim3 const *const ptr_dimension_block_reduce_array_recieved);
}

#include <../Source_Files/CUDA/CUDA_Reduce.cu>
