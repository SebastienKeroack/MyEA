#include <CUDA/CUDA_cuRAND.cuh>

#include <Tools/CUDA_Configuration.cuh>

#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#include <chrono>

__global__ void kernel__cuRAND__Memcpy_cuRAND_State_MTGP32(struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_destination_received,
                                                                                                       struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_source_received,
                                                                                                       struct mtgp32_kernel_params *const ptr_array_mtgp32_kernel_params_t_source_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index].k = ptr_array_mtgp32_kernel_params_t_source_received + tmp_thread_global_index;
    
    *ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index].k = *ptr_array_cuRAND_State_MTGP32_source_received[tmp_thread_global_index].k;
}

__global__ void kernel__cuRAND__Memcpy_cuRAND_State_MTGP32(size_t const size_received,
                                                                                                       struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_destination_received,
                                                                                                       struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_source_received,
                                                                                                       struct mtgp32_kernel_params *const ptr_array_mtgp32_kernel_params_t_source_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index].k = ptr_array_mtgp32_kernel_params_t_source_received + tmp_thread_global_index;
    
        *ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index].k = *ptr_array_cuRAND_State_MTGP32_source_received[tmp_thread_global_index].k;
    }
}

__global__ void kernel_while__cuRAND__Memcpy_cuRAND_State_MTGP32(size_t const size_received,
                                                                                                               struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_destination_received,
                                                                                                               struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_source_received,
                                                                                                               struct mtgp32_kernel_params *const ptr_array_mtgp32_kernel_params_t_source_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    do
    {
        ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index].k = ptr_array_mtgp32_kernel_params_t_source_received + tmp_thread_global_index;
        
        *ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index].k = *ptr_array_cuRAND_State_MTGP32_source_received[tmp_thread_global_index].k;

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuRAND__Memcpy_cuRAND_State_MTGP32(size_t const size_received,
                                                                                           struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_destination_received,
                                                                                           struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_source_received,
                                                                                           struct mtgp32_kernel_params *const ptr_array_mtgp32_kernel_params_t_source_received,
                                                                                           struct dim3 const *const ptr_dimension_grid_received,
                                                                                           struct dim3 const *const ptr_dimension_block_received)
{
    if(USE_PARALLEL && size_received >= warpSize)
    {
        LAUNCH_KERNEL_POINTER_1D(cuRAND__Memcpy_cuRAND_State_MTGP32,
                                                          ptr_dimension_grid_received,
                                                          ptr_dimension_block_received,
                                                          0_zu,
                                                          size_received,
                                                          ptr_array_cuRAND_State_MTGP32_destination_received,
                                                          ptr_array_cuRAND_State_MTGP32_source_received,
                                                          ptr_array_mtgp32_kernel_params_t_source_received)
    }
    else
    {
        for(size_t i(0_zu); i != size_received; ++i)
        {
            ptr_array_cuRAND_State_MTGP32_destination_received[i].k = ptr_array_mtgp32_kernel_params_t_source_received + i;
            
            *ptr_array_cuRAND_State_MTGP32_destination_received[i].k = *ptr_array_cuRAND_State_MTGP32_source_received[i].k;
        }
    }

}

__host__ bool Allocate_cuRAND_MTGP32(int const number_states_MTGP32_received,
                                                              size_t seed_received,
                                                              struct mtgp32_kernel_params *&ptr_mtgp32_kernel_params_received,
                                                              struct curandStateMtgp32 *&ptr_curandStateMtgp32_t_received)
{
    if(number_states_MTGP32_received == 0)
    {
        PRINT_FORMAT("%s: ERROR: Number of states for MTGP32 equal zero." NEW_LINE,
                                __FUNCTION__);

        return(false);
    }

    CUDA__Safe_Call(cudaMalloc((void**)&ptr_mtgp32_kernel_params_received, sizeof(struct mtgp32_kernel_params)));
    
    curandStatus_t tmp_curandStatus_t(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, ptr_mtgp32_kernel_params_received));
    
    if(tmp_curandStatus_t != curandStatus::CURAND_STATUS_SUCCESS)
    {
        PRINT_FORMAT("%s: ERROR: curandMakeMTGP32Constants failed at %s:%i: _%d" NEW_LINE,
                                 __FUNCTION__,
                                 __FILE__,
                                 __LINE__,
                                 tmp_curandStatus_t);

        CUDA__Safe_Call(cudaFree(ptr_mtgp32_kernel_params_received));

        return(false);
    }
    
    CUDA__Safe_Call(cudaMalloc((void**)&ptr_curandStateMtgp32_t_received, static_cast<size_t>(number_states_MTGP32_received) * sizeof(struct curandStateMtgp32)));

    for(int tmp_number_states_MTGP32_allocate,
             tmp_number_states_MTGP32_offset(0),
             tmp_length_i(static_cast<int>(ceil(static_cast<double>(number_states_MTGP32_received) / 200.0))),
             i(0); i != tmp_length_i; ++i,
                                              tmp_number_states_MTGP32_offset += static_cast<size_t>(tmp_number_states_MTGP32_allocate))
    {
        if(i + 1 != tmp_length_i) { tmp_number_states_MTGP32_allocate = 200; }
        else { tmp_number_states_MTGP32_allocate = number_states_MTGP32_received - 200 * i; }

        tmp_curandStatus_t = curandMakeMTGP32KernelState(ptr_curandStateMtgp32_t_received + tmp_number_states_MTGP32_offset,
                                                                                          mtgp32dc_params_fast_11213, 
                                                                                          ptr_mtgp32_kernel_params_received,
                                                                                          tmp_number_states_MTGP32_allocate, // 200 Maximum states
                                                                                          seed_received);
                
        if(tmp_curandStatus_t != curandStatus::CURAND_STATUS_SUCCESS)
        {
            PRINT_FORMAT("%s: ERROR: curandMakeMTGP32KernelState(ptr + %d, args, args, %d, %zu) failed at %s:%i: _%d" NEW_LINE,
                                     __FUNCTION__,
                                     tmp_number_states_MTGP32_offset,
                                     tmp_number_states_MTGP32_allocate,
                                     seed_received,
                                     __FILE__,
                                     __LINE__,
                                     tmp_curandStatus_t);
            
            CUDA__Safe_Call(cudaFree(ptr_mtgp32_kernel_params_received));

            CUDA__Safe_Call(cudaFree(ptr_curandStateMtgp32_t_received));
            
            return(false);
        }

        seed_received = seed_received == 0_zu ? static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) : seed_received - 1_zu;
    }

    return(true);
}

__host__ void Cleanup_cuRAND_MTGP32(struct mtgp32_kernel_params *&ptr_mtgp32_kernel_params_received, struct curandStateMtgp32 *&ptr_curandStateMtgp32_t_received)
{
    CUDA__Safe_Call(cudaFree(ptr_mtgp32_kernel_params_received));

    CUDA__Safe_Call(cudaFree(ptr_curandStateMtgp32_t_received));
}
