#include <Tools/CUDA_Configuration.cuh>

template<typename T>
__global__ void kernel__Flag_1D(bool const *const ptr_array_flag_received, T *const ptr_array_to_one_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    ptr_array_to_one_received[tmp_thread_global_index] = static_cast<T>(ptr_array_flag_received[tmp_thread_global_index]);
}
    
template<typename T>
__global__ void kernel__Flag_1D(size_t const size_received,
                                                bool const *const ptr_array_flag_received,
                                                T *const ptr_array_to_one_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_global_index < size_received)
    { ptr_array_to_one_received[tmp_thread_global_index] = static_cast<T>(ptr_array_flag_received[tmp_thread_global_index]); }
}

template<typename T>
__global__ void kernel_while__Flag_1D(size_t const size_received,
                                                            bool const *const ptr_array_flag_received,
                                                            T *const ptr_array_to_one_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    do
    {
        ptr_array_to_one_received[tmp_thread_global_index] = static_cast<T>(ptr_array_flag_received[tmp_thread_global_index]);
            
        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__device__ void Flag_1D(size_t const size_received,
                                    bool const *ptr_array_flag_received,
                                    T *ptr_array_to_flag_received,
                                    struct dim3 const *const ptr_dimension_grid_received,
                                    struct dim3 const *const ptr_dimension_block_received)
{
    if(USE_PARALLEL && size_received >= warpSize)
    {
        LAUNCH_KERNEL_POINTER_1D(Flag_1D<T>,
                                                          ptr_dimension_grid_received,
                                                          ptr_dimension_block_received,
                                                          0_zu,
                                                          size_received,
                                                          ptr_array_flag_received,
                                                          ptr_array_to_flag_received)
    }
    else
    {
        for(T const *const tmp_ptr_one_end(ptr_array_to_flag_received + size_received); ptr_array_to_flag_received != tmp_ptr_one_end; ++ptr_array_to_flag_received,
                                                                                                                                                                                                    ++ptr_array_flag_received)
        { *ptr_array_to_flag_received = static_cast<T>(*ptr_array_flag_received); }
    }
}
