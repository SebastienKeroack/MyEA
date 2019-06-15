#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

__global__ void kernel__CNeural_Network__Set__Regularization__L2(T_ const regularization__l2_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Regularization__L2(regularization__l2_received); }

__host__ __device__ bool CUDA_Neural_Network::Set__Regularization__L2(T_ const regularization__l2_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Regularization__L2 <<< 1u, 1u >>> (regularization__l2_received, this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    if(this->regularization__l2 != regularization__l2_received)
    {
        bool const tmp_use_regularization(this->Use__Regularization_Parameter()),
                        tmp_not_initialized_regularization(this->ptr_array_mask_regularized_parameters == nullptr);

        this->regularization__l2 = regularization__l2_received;

        if(tmp_use_regularization == false && regularization__l2_received != 0_T)
        {
            if(this->Allocate__Parameter__Regularization() == false)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate regularization connections!" NEW_LINE, __FUNCTION__);
        
                return(false);
            }

            if(tmp_not_initialized_regularization) { this->Indexing_Regularization_Parameters(); }
        }

        if(this->Use__Regularization_Parameter() == false)
        { this->Deallocate__Parameter__Regularization(); }
    }
#endif

    return(true);
}

template<typename T>
__global__ void kernel__Update_Derivative_Weight__Regularization__L2(T const regularization__l2_received,
                                                                                        T *const ptr_array_gradients_received,
                                                                                        T const *const ptr_array_parameters_received,
                                                                                        T const *const ptr_array_connections_mask_regularization_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    ptr_array_gradients_received[tmp_thread_global_index] += ptr_array_connections_mask_regularization_received[tmp_thread_global_index] * ptr_array_parameters_received[tmp_thread_global_index] * regularization__l2_received;
}
    
template<typename T>
__global__ void kernel__Update_Derivative_Weight__Regularization__L2(size_t const size_received,
                                                                                        T const regularization__l2_received,
                                                                                        T *const ptr_array_gradients_received,
                                                                                        T const *const ptr_array_parameters_received,
                                                                                        T const *const ptr_array_connections_mask_regularization_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
        
    if(tmp_thread_global_index < size_received)
    { ptr_array_gradients_received[tmp_thread_global_index] += ptr_array_connections_mask_regularization_received[tmp_thread_global_index] * ptr_array_parameters_received[tmp_thread_global_index] * regularization__l2_received; }
}
    
template<typename T>
__global__ void kernel_while__Update_Derivative_Weight__Regularization__L2(size_t const size_received,
                                                                                                T const regularization__l2_received,
                                                                                                T *const ptr_array_gradients_received,
                                                                                                T const *const ptr_array_parameters_received,
                                                                                                T const *const ptr_array_connections_mask_regularization_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_gradients_received[tmp_thread_global_index] += ptr_array_connections_mask_regularization_received[tmp_thread_global_index] * ptr_array_parameters_received[tmp_thread_global_index] * regularization__l2_received;

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Update_Derivative_Weight__Regularization__L2(size_t const batch_size_received)
{
    T_ *tmp_ptr_gradient_it(this->ptr_array_derivatives_parameters);
    T_ const tmp_regularization__l2(this->regularization__l2 * batch_size_received),
                 *tmp_ptr_weight_it(this->ptr_array_parameters),
                 *tmp_ptr_connections_mask_regularization_it(this->ptr_array_mask_regularized_parameters);

    if(USE_PARALLEL && this->total_weights >= warpSize)
    {
        LAUNCH_KERNEL_1D(Update_Derivative_Weight__Regularization__L2<T_>,
                                          this->ptr_array_dim3_grid[2u],
                                          this->ptr_array_dim3_block[2u],
                                          0_zu,
                                          this->total_weights,
                                          tmp_regularization__l2,
                                          tmp_ptr_gradient_it,
                                          tmp_ptr_weight_it,
                                          tmp_ptr_connections_mask_regularization_it)

        CUDA__Check_Error();
    }
    else
    {
        for(T_ const *const tmp_ptr_last_gradient(tmp_ptr_gradient_it + this->total_weights); tmp_ptr_gradient_it != tmp_ptr_last_gradient; ++tmp_ptr_gradient_it,
                                                                                                                                                                                                    ++tmp_ptr_weight_it,
                                                                                                                                                                                                    ++tmp_ptr_connections_mask_regularization_it)
        { *tmp_ptr_gradient_it += *tmp_ptr_connections_mask_regularization_it * *tmp_ptr_weight_it * tmp_regularization__l2; }
    }
}

__host__ __device__ T_ CUDA_Neural_Network::Get__Regularization__L2(void) const { return(this->regularization__l2); }
