#pragma once

#include <Tools/Configuration.hpp>

#include <device_launch_parameters.h>

#define PREPROCESSED_STRING_(x) #x
#define PREPROCESSED_STRING(x) PREPROCESSED_STRING_(x)

#define PREPROCESSED_CONCAT_(x, y) x##y
#define PREPROCESSED_CONCAT(x, y) PREPROCESSED_CONCAT_(x, y)

#define DECLARE_EXTERN_SHARED_MEMORY_TEMPLATE(struct_name_received) \
            template<typename T> \
            struct struct_name_received \
            { \
                __device__ T* operator()(void) \
                { \
                    extern __shared__ T PREPROCESSED_CONCAT(tmp_ptr_pointer_T_, struct_name_received)[]; \
                    return(PREPROCESSED_CONCAT(tmp_ptr_pointer_T_, struct_name_received)); \
                } \
            }; \
            template <> \
            struct struct_name_received<int> \
            { \
                __device__ int* operator()(void) \
                { \
                    extern __shared__ int PREPROCESSED_CONCAT(tmp_ptr_pointer_int_, struct_name_received)[]; \
                    return(PREPROCESSED_CONCAT(tmp_ptr_pointer_int_, struct_name_received)); \
                } \
            }; \
            template <> \
            struct struct_name_received<unsigned int> \
            { \
                __device__ unsigned int* operator()(void) \
                { \
                    extern __shared__ unsigned int PREPROCESSED_CONCAT(tmp_ptr_pointer_unsigned_int_, struct_name_received)[]; \
                    return(PREPROCESSED_CONCAT(tmp_ptr_pointer_unsigned_int_, struct_name_received)); \
                } \
            }; \
            template <> \
            struct struct_name_received<float> \
            { \
                __device__ float* operator()(void) \
                { \
                    extern __shared__ float PREPROCESSED_CONCAT(tmp_ptr_pointer_float_, struct_name_received)[]; \
                    return(PREPROCESSED_CONCAT(tmp_ptr_pointer_float_, struct_name_received)); \
                } \
            }; \
            template <> \
            struct struct_name_received<double> \
            { \
                __device__ double* operator()(void) \
                { \
                    extern __shared__ double PREPROCESSED_CONCAT(tmp_ptr_pointer_double_, struct_name_received)[]; \
                    return(PREPROCESSED_CONCAT(tmp_ptr_pointer_double_, struct_name_received)); \
                } \
            };

#define EXTERN_SHARED_MEMORY_TEMPLATE(type_received, assignation_received, struct_name_received) \
            struct struct_name_received<type_received> tmp_struct_shared_memory; \
            assignation_received = tmp_struct_shared_memory();
