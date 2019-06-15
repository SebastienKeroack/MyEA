#pragma once

#include <Tools/Configuration.hpp>

#if defined(COMPILE_CUDA)
    #include <device_launch_parameters.h>
    
    namespace Memory
    {
        template<class T>
        __host__ __device__ void Copy(T const *ptr_array_source_received,
                                                      T const *ptr_array_last_source_received,
                                                      T *ptr_array_destination_received);
        
        template<class T>
        __host__ __device__ void Copy_Loop(T const *ptr_array_source_received,
                                                               T const *ptr_array_last_source_received,
                                                               T *ptr_array_destination_received);
        
        template<class T>
        __host__ __device__ void Fill(T *ptr_array_received,
                                                   T *ptr_array_last_received,
                                                   T const value_received);
        
        template<class T>
        __host__ __device__ void Fill_Nullptr(T *ptr_array_received, T *ptr_array_last_received);

        template<class T>
        __host__ __device__ T* reallocate(T *ptr_array_received,
                                                          size_t const size_t_new_received,
                                                          size_t const size_t_old_received,
                                                          bool const memcpy_received = true);

        template<class T>
        __host__ __device__ T* reallocate_cpp(T *ptr_array_received,
                                                                 size_t const new_size_received,
                                                                 size_t const old_size_received,
                                                                 bool const memcpy_received = true);
    
        template<class T>
        __device__ T* reallocate_cpp(T *ptr_array_received,
                                                   size_t const new_size_received,
                                                   size_t const old_size_received,
                                                   struct dim3 const *const ptr_dimension_grid_zero_received,
                                                   struct dim3 const *const ptr_dimension_block_zero_received,
                                                   struct dim3 const *const ptr_dimension_grid_copy_received,
                                                   struct dim3 const *const ptr_dimension_block_copy_received,
                                                   bool const memcpy_received = true);

        template<class T>
        __host__ __device__ T* reallocate_objects_cpp(T *ptr_array_received,
                                                                              size_t const new_size_received,
                                                                              size_t const old_size_received,
                                                                              bool const memcpy_received = true);
    
        template<class T>
        __device__ T* reallocate_objects_cpp(T *ptr_array_received,
                                                               size_t const new_size_received,
                                                               size_t const old_size_received,
                                                               struct dim3 const *const ptr_dimension_grid_zero_received,
                                                               struct dim3 const *const ptr_dimension_block_zero_received,
                                                               struct dim3 const *const ptr_dimension_grid_copy_received,
                                                               struct dim3 const *const ptr_dimension_block_copy_received,
                                                               bool const memcpy_received = true);

        template<class T>
        __host__ __device__ T* reallocate_pointers_array_cpp(T *ptr_array_received,
                                                                                       size_t const new_size_received,
                                                                                       size_t const old_size_received,
                                                                                       bool const memcpy_received = true);
    
        template<class T>
        __device__ T* reallocate_pointers_array_cpp(T *ptr_array_received,
                                                                         size_t const new_size_received,
                                                                         size_t const old_size_received,
                                                                         struct dim3 const *const ptr_dimension_grid_zero_received,
                                                                         struct dim3 const *const ptr_dimension_block_zero_received,
                                                                         struct dim3 const *const ptr_dimension_grid_copy_received,
                                                                         struct dim3 const *const ptr_dimension_block_copy_received,
                                                                         bool const memcpy_received = true);
    }

    #include <../Source_Files/Tools/CUDA_Reallocate.cu>
#endif
