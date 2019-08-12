#pragma once

#include <Configuration/Configuration.hpp>

#include <Enums/Enum_Type_Device_Synchronized.hpp>
    
#include <Strings/String.hpp>
    
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <cooperative_groups.hpp>
//#include <cooperative_groups_helpers.hpp>
    
#include <cstdio>

#define LAUNCH_KERNEL_1D(kernel_name_received, \
                                              grid_received, \
                                              block_received, \
                                              size_t_shared_memory_received, \
                                              size_received, ...) \
            if(grid_received.x * block_received.x < size_received) { PREPROCESSED_CONCAT(kernel_while__, kernel_name_received) <<< grid_received, block_received, size_t_shared_memory_received >>> (size_received, __VA_ARGS__); } \
            else if(grid_received.x * block_received.x > size_received) { PREPROCESSED_CONCAT(kernel__, kernel_name_received) <<< grid_received, block_received, size_t_shared_memory_received >>> (size_received, __VA_ARGS__); } \
            else { PREPROCESSED_CONCAT(kernel__, kernel_name_received) <<< grid_received, block_received, size_t_shared_memory_received >>> (__VA_ARGS__); }

#define LAUNCH_KERNEL_POINTER_1D(kernel_name_received, \
                                                              ptr_grid_received, \
                                                              ptr_block_received, \
                                                              size_t_shared_memory_received, \
                                                              size_received, ...) \
            if(ptr_grid_received->x * ptr_block_received->x < size_received) { PREPROCESSED_CONCAT(kernel_while__, kernel_name_received) <<< *ptr_grid_received, *ptr_block_received, size_t_shared_memory_received >>> (size_received, __VA_ARGS__); } \
            else if(ptr_grid_received->x * ptr_block_received->x > size_received) { PREPROCESSED_CONCAT(kernel__, kernel_name_received) <<< *ptr_grid_received, *ptr_block_received, size_t_shared_memory_received >>> (size_received, __VA_ARGS__); } \
            else { PREPROCESSED_CONCAT(kernel__, kernel_name_received) <<< *ptr_grid_received, *ptr_block_received, size_t_shared_memory_received >>> (__VA_ARGS__); }

enum ENUM_TYPE_MEMORY_ALLOCATE : unsigned int
{
    TYPE_MEMORY_ALLOCATE_UNKNOW = 0,
    TYPE_MEMORY_ALLOCATE_CPU = 1,
    TYPE_MEMORY_ALLOCATE_GPU = 2,
    TYPE_MEMORY_ALLOCATE_MANAGED = 3
};

void CUDA__Initialize__Device(struct cudaDeviceProp const &ref_device_prop_received, size_t const memory_allocate_received);
void CUDA__Set__Device(int const index_device_received);
void CUDA__Set__Synchronization_Depth(size_t const depth_received);
void CUDA__Reset(void);
void CUDA__Print__Device_Property(struct cudaDeviceProp const &ref_device_prop_received, int const index_device_received);

__host__ __device__ static inline void CUDA__Safe_Call(cudaError const cudaError_received,
                                                                                                        char const *const ptr_file_received,
                                                                                                        int const line_received)
{
    if(cudaError::cudaSuccess != cudaError_received)
    {
        PRINT_FORMAT("%s: ERROR: Failed at %s:%i: %s _%d" NEW_LINE,
                                 __FUNCTION__,
                                 ptr_file_received,
                                 line_received,
                                 cudaGetErrorString(cudaError_received),
                                 cudaError_received);
    }
}
__host__ __device__ static inline void CUDA__Check_Error(char const *const ptr_file_received, int const line_received)
{
    cudaError tmp_cudaError(cudaDeviceSynchronize());
    if(cudaError::cudaSuccess != tmp_cudaError)
    {
        PRINT_FORMAT("%s: ERROR: Synchronization failed at %s:%i: %s _%d" NEW_LINE,
                                 __FUNCTION__,
                                 ptr_file_received,
                                 line_received,
                                 cudaGetErrorString(tmp_cudaError),
                                 tmp_cudaError);
    }
    else if(cudaError::cudaSuccess != (tmp_cudaError = cudaGetLastError()))
    {
        PRINT_FORMAT("%s: ERROR: Failed at %s:%i: %s _%d" NEW_LINE,
                                 __FUNCTION__,
                                 ptr_file_received,
                                 line_received,
                                 cudaGetErrorString(tmp_cudaError),
                                 tmp_cudaError);
    }
}
__host__ __device__ static inline void CUDA__Last_Error(char const *const ptr_file_received, int const line_received)
{
    cudaError const tmp_error(cudaGetLastError());
    if(cudaError::cudaSuccess != tmp_error)
    {
        PRINT_FORMAT("%s: ERROR: Failed at %s:%i: %s _%d" NEW_LINE,
                                 __FUNCTION__,
                                 ptr_file_received,
                                 line_received,
                                 cudaGetErrorString(tmp_error),
                                 tmp_error);
    }
}

#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    #define CUDA__Safe_Call(error_received) CUDA__Safe_Call(error_received, __FILE__, __LINE__)
    #define CUDA__Check_Error() CUDA__Check_Error(__FILE__, __LINE__)
    #define CUDA__Last_Error() CUDA__Last_Error(__FILE__, __LINE__)
#else
    #define CUDA__Safe_Call(error_received) error_received
    #define CUDA__Check_Error() cudaDeviceSynchronize()
    #define CUDA__Last_Error()
#endif

__host__ __device__ static bool CUDA__Required_Compatibility_Device(struct cudaDeviceProp const &ref_device_prop_received)
{
    return((ref_device_prop_received.major == 3 && ref_device_prop_received.minor >= 5)
             ||
             ref_device_prop_received.major >= 4);
}

int CUDA__Device_Count(void);
int CUDA__Maximum_Concurrent_Kernel(struct cudaDeviceProp const &ref_device_prop_received);

size_t CUDA__Number_CUDA_Cores(struct cudaDeviceProp const &ref_device_prop_received);
    
__device__ __forceinline__ void CUDA__ThreadBlockSynchronize(void)
{
#if defined(__CUDA_ARCH__)
    __syncthreads();

    if(threadIdx.x == 0u) { CUDA__Check_Error(); }

    __syncthreads();
#endif
}

__device__ __forceinline__ void CUDA__Device_Synchronise(enum MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED const type_device_synchronise_received)
{
    switch(type_device_synchronise_received)
    {
        case MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD: CUDA__Check_Error(); break;
        case MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK: CUDA__ThreadBlockSynchronize(); break;
        default: break;
    }
}

__device__ __forceinline__ void CUDA__Device_Synchronise(bool &ref_synchronized_received, enum MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED const type_device_synchronise_received)
{
    switch(type_device_synchronise_received)
    {
        case MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD:
            if(ref_synchronized_received == false)
            {
                ref_synchronized_received = true;

                CUDA__Check_Error();
            }
                break;
        case MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK: 
            if(ref_synchronized_received == false)
            {
                ref_synchronized_received = true;

                CUDA__ThreadBlockSynchronize();
            }
                break;
        default: break;
    }
}

bool CUDA__Input__Use__CUDA(int &ref_index_device_received, size_t &ref_maximum_allowable_memory_bytes_received);