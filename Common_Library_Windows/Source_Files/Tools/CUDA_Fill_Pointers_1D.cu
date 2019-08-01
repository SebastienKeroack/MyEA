#include <Configuration/CUDA/Configuration.cuh>

#define LAUNCH_KERNEL_POINTER_1D(kernel_name_received, \
                                 ptr_grid_received, \
                                 ptr_block_received, \
                                 size_t_shared_memory_received, \
                                 size_received, ...) \
    if(ptr_grid_received->x * ptr_block_received->x < size_received) { PREPROCESSED_CONCAT(kernel_while__, kernel_name_received) <<< *ptr_grid_received, *ptr_block_received, size_t_shared_memory_received >>> (size_received, __VA_ARGS__); } \
    else if(ptr_grid_received->x * ptr_block_received->x > size_received) { PREPROCESSED_CONCAT(kernel__, kernel_name_received) <<< *ptr_grid_received, *ptr_block_received, size_t_shared_memory_received >>> (size_received, __VA_ARGS__); } \
    else { PREPROCESSED_CONCAT(kernel__, kernel_name_received) <<< *ptr_grid_received, *ptr_block_received, size_t_shared_memory_received >>> (__VA_ARGS__); }

template<typename T>
__global__ void kernel__Fill_Pointers_1D(T *const ptr_array_outputs_received)
{ ptr_array_outputs_received[blockIdx.x * blockDim.x + threadIdx.x] = nullptr; }

template<typename T>
__global__ void kernel__Fill_Pointers_1D(size_t const size_received, T *const ptr_array_outputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_global_index < size_received)
    { ptr_array_outputs_received[tmp_thread_global_index] = nullptr; }
}

template<typename T>
__global__ void kernel_while__Fill_Pointers_1D(size_t const size_received, T *const ptr_array_outputs_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    do
    {
        ptr_array_outputs_received[tmp_thread_global_index] = nullptr;
        
        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__device__ void Fill_Pointers_1D(size_t const size_received,
                                                T *ptr_array_outputs_received,
                                                struct dim3 const *const ptr_dimension_grid_received,
                                                struct dim3 const *const ptr_dimension_block_received)
{
    if(USE_PARALLEL && size_received >= warpSize * warpSize)
    {
        LAUNCH_KERNEL_POINTER_1D(Fill_Pointers_1D<T>,
                                                          ptr_dimension_grid_received,
                                                          ptr_dimension_block_received,
                                                          0_zu,
                                                          size_received,
                                                          ptr_array_outputs_received)
    }
    else
    {
        for(T const *const tmp_ptr_output_end(ptr_array_outputs_received + size_received); ptr_array_outputs_received != tmp_ptr_output_end; ++ptr_array_outputs_received)
        { *ptr_array_outputs_received = nullptr; }
    }
}