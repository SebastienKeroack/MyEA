#pragma once

#include <Configuration/Configuration.hpp>

#include <device_launch_parameters.h>

namespace Multiply
{
    template<typename T>
    __global__ void kernel__Multiply_Z_Y_1D(size_t const stride_Z_received,
                                                           T const constant_received,
                                                           T *const ptr_array_Z_received);

    template<typename T>
    __global__ void kernel__Multiply_Z_Y_1D(size_t const size_received,
                                                           size_t const stride_Z_received,
                                                           T const constant_received,
                                                           T *const ptr_array_Z_received);

    template<typename T>
    __global__ void kernel_while__Multiply_Z_Y_1D(size_t const size_received,
                                                                    size_t const stride_Z_received,
                                                                    T const constant_received,
                                                                    T *const ptr_array_Z_received);

    template<typename T>
    __device__ void Multiply_Z_Y_1D(size_t const size_received,
                                                    size_t const stride_Z_received,
                                                    T const constant_received,
                                                    T *ptr_array_Z_received,
                                                    struct dim3 const *const ptr_dim3_grid_received,
                                                    struct dim3 const *const ptr_dim3_block_received);

    template<typename T>
    __global__ void kernel__Multiply_X_Y_1D(T const constant_received, T *const ptr_array_X_received);

    template<typename T>
    __global__ void kernel__Multiply_X_Y_1D(size_t const size_received,
                                                               T const constant_received,
                                                               T *const ptr_array_X_received);

    template<typename T>
    __global__ void kernel_while__Multiply_X_Y_1D(size_t const size_received,
                                                                        T const constant_received,
                                                                        T *const ptr_array_X_received);

    template<typename T>
    __device__ void Multiply_X_Y_1D(size_t const size_received,
                                                     T const constant_received,
                                                     T *ptr_array_X_received,
                                                     struct dim3 const *const ptr_dim3_grid_received,
                                                     struct dim3 const *const ptr_dim3_block_received);

    template<typename T>
    __device__ void Multiply_X_Y_1D(bool &ref_synchronized_received,
                                                     size_t const size_received,
                                                     T const constant_received,
                                                     T *ptr_array_X_received,
                                                     struct dim3 const *const ptr_dim3_grid_received,
                                                     struct dim3 const *const ptr_dim3_block_received);

    template<typename T>
    __global__ void kernel__FMAC_Z_YX_1D(size_t const stride_Z_received,
                                                                T *const ptr_array_Z_received,
                                                                T const constant_received,
                                                                T const *const ptr_array_X_received);
    
    template<typename T>
    __global__ void kernel__FMAC_Z_YX_1D(size_t const size_received,
                                                                size_t const stride_Z_received,
                                                                T *const ptr_array_Z_received,
                                                                T const constant_received,
                                                                T const *const ptr_array_X_received);

    template<typename T>
    __global__ void kernel_while__FMAC_Z_YX_1D(size_t const size_received,
                                                                        size_t const stride_Z_received,
                                                                        T *const ptr_array_Z_received,
                                                                        T const constant_received,
                                                                        T const *const ptr_array_X_received);

    template<typename T>
    __device__ void FMAC_Z_YX_1D(size_t const size_received,
                                                    size_t const stride_Z_received,
                                                    T *ptr_array_Z_received,
                                                    T const constant_received,
                                                    T const *ptr_array_X_received,
                                                    struct dim3 const *const ptr_dimension_grid_received,
                                                    struct dim3 const *const ptr_dimension_block_received);

    template<typename T>
    __global__ void kernel__FMAC_X_YZ_1D(size_t const stride_Z_received,
                                                                T *const ptr_array_X_received,
                                                                T const constant_received,
                                                                T const *const ptr_array_Z_received);
    
    template<typename T>
    __global__ void kernel__FMAC_X_YZ_1D(size_t const size_received,
                                                                size_t const stride_Z_received,
                                                                T *const ptr_array_X_received,
                                                                T const constant_received,
                                                                T const *const ptr_array_Z_received);

    template<typename T>
    __global__ void kernel_while__FMAC_X_YZ_1D(size_t const size_received,
                                                                         size_t const stride_Z_received,
                                                                         T *const ptr_array_X_received,
                                                                         T const constant_received,
                                                                         T const *const ptr_array_Z_received);

    template<typename T>
    __device__ void FMAC_X_YZ_1D(size_t const size_received,
                                                    size_t const stride_Z_received,
                                                    T *ptr_array_X_received,
                                                    T const constant_received,
                                                    T const *ptr_array_Z_received,
                                                    struct dim3 const *const ptr_dimension_grid_received,
                                                    struct dim3 const *const ptr_dimension_block_received);

    template<typename T>
    __global__ void kernel__FMAC_X_YX_1D(T *const ptr_array_outputs_X_received,
                                                                T const constant_received,
                                                                T const *const ptr_array_inputs_X_received);
    
    template<typename T>
    __global__ void kernel__FMAC_X_YX_1D(size_t const size_received,
                                                                T *const ptr_array_outputs_X_received,
                                                                T const constant_received,
                                                                T const *const ptr_array_inputs_X_received);

    template<typename T>
    __global__ void kernel_while__FMAC_X_YX_1D(size_t const size_received,
                                                                         T *const ptr_array_outputs_X_received,
                                                                         T const constant_received,
                                                                         T const *const ptr_array_inputs_X_received);

    template<typename T>
    __device__ void FMAC_X_YX_1D(size_t const size_received,
                                                    T *ptr_array_outputs_X_received,
                                                    T const constant_received,
                                                    T const *ptr_array_inputs_X_received,
                                                    struct dim3 const *const ptr_dimension_grid_received,
                                                    struct dim3 const *const ptr_dimension_block_received);

    template<typename T>
    __device__ void FMAC_X_YX_1D__atomic(size_t const size_received,
                                                                  T *ptr_array_outputs_X_received,
                                                                  T const constant_received,
                                                                  T const *ptr_array_inputs_X_received,
                                                                  struct dim3 const *const ptr_dimension_grid_received,
                                                                  struct dim3 const *const ptr_dimension_block_received);
}

#include <../Source_Files/CUDA/CUDA_Multiply_1D.cu>
