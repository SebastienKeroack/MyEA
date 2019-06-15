#include <Tools/CUDA_Configuration.cuh>

namespace Accumulate
{
    template<typename T>
    __global__ void kernel__Accumulate_X_X_1D(T *const ptr_array_outputs_received, T const *const ptr_array_inputs_received)
    {
        size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
        ptr_array_outputs_received[tmp_thread_global_index] += ptr_array_inputs_received[tmp_thread_global_index];
    }

    template<typename T>
    __global__ void kernel__Accumulate_X_X_1D(size_t const size_received,
                                                                      T *const ptr_array_outputs_received,
                                                                      T const *const ptr_array_inputs_received)
    {
        size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
        if(tmp_thread_global_index < size_received)
        { ptr_array_outputs_received[tmp_thread_global_index] += ptr_array_inputs_received[tmp_thread_global_index]; }
    }

    template<typename T>
    __global__ void kernel_while__Accumulate_X_X_1D(size_t const size_received,
                                                                              T *const ptr_array_outputs_received,
                                                                              T const *const ptr_array_inputs_received)
    {
        size_t const tmp_grid_stride(gridDim.x * blockDim.x);
        size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
        do
        {
            ptr_array_outputs_received[tmp_thread_global_index] += ptr_array_inputs_received[tmp_thread_global_index];
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }

    template<typename T>
    __device__ void Accumulate_X_X_1D(size_t const size_received,
                                                          T *ptr_array_outputs_received,
                                                          T const *ptr_array_inputs_received,
                                                          struct dim3 const *const ptr_dimension_grid_received,
                                                          struct dim3 const *const ptr_dimension_block_received)
    {
        if(USE_PARALLEL && size_received >= warpSize)
        {
            LAUNCH_KERNEL_POINTER_1D(Accumulate_X_X_1D<T>,
                                                              ptr_dimension_grid_received,
                                                              ptr_dimension_block_received,
                                                              0_zu,
                                                              size_received,
                                                              ptr_array_outputs_received,
                                                              ptr_array_inputs_received)
        }
        else
        {
            for(T const *const ptr_output_end(ptr_array_outputs_received + size_received); ptr_array_outputs_received != ptr_output_end; ++ptr_array_outputs_received,
                                                                                                                                                                                                 ++ptr_array_inputs_received)
            { *ptr_array_outputs_received += *ptr_array_inputs_received; }
        }
    }

}