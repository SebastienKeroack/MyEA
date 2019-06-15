#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

template<typename T>
__global__ void kernel__Update_Parameter__Gradient_Descent_Momentum(T const learning_rate_received,
                                                                                                            T const learning_momentum_received,
                                                                                                            T *const ptr_array_partial_derivative_received,
                                                                                                            T *const ptr_array_parameters_received,
                                                                                                            T *const ptr_array_previous_delta_weights_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_delta_weigth;
    
    ptr_array_previous_delta_weights_received[tmp_thread_global_index] = tmp_delta_weigth = learning_rate_received * ptr_array_partial_derivative_received[tmp_thread_global_index] + learning_momentum_received * ptr_array_previous_delta_weights_received[tmp_thread_global_index];

    ptr_array_parameters_received[tmp_thread_global_index] -= tmp_delta_weigth;

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template<typename T>
__global__ void kernel__Update_Parameter__Gradient_Descent_Momentum(size_t const size_received,
                                                                                                            T const learning_rate_received,
                                                                                                            T const learning_momentum_received,
                                                                                                            T *const ptr_array_partial_derivative_received,
                                                                                                            T *const ptr_array_parameters_received,
                                                                                                            T *const ptr_array_previous_delta_weights_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_delta_weigth;

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_previous_delta_weights_received[tmp_thread_global_index] = tmp_delta_weigth = learning_rate_received * ptr_array_partial_derivative_received[tmp_thread_global_index] + learning_momentum_received * ptr_array_previous_delta_weights_received[tmp_thread_global_index];

        ptr_array_parameters_received[tmp_thread_global_index] -= tmp_delta_weigth;

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
    }
}

template<typename T>
__global__ void kernel_while__Update_Parameter__Gradient_Descent_Momentum(size_t const size_received,
                                                                                                                    T const learning_rate_received,
                                                                                                                    T const learning_momentum_received,
                                                                                                                    T *const ptr_array_partial_derivative_received,
                                                                                                                    T *const ptr_array_parameters_received,
                                                                                                                    T *const ptr_array_previous_delta_weights_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_delta_weigth;

    do
    {
        ptr_array_previous_delta_weights_received[tmp_thread_global_index] = tmp_delta_weigth = learning_rate_received * ptr_array_partial_derivative_received[tmp_thread_global_index] + learning_momentum_received * ptr_array_previous_delta_weights_received[tmp_thread_global_index];

        ptr_array_parameters_received[tmp_thread_global_index] -= tmp_delta_weigth;

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Update_Parameter__Gradient_Descent_Momentum(T const weight_decay_received,
                                                                                                            T const learning_rate_received,
                                                                                                            T const learning_momentum_received,
                                                                                                            T const *const ptr_array_connections_mask_rergularization_received,
                                                                                                            T *const ptr_array_partial_derivative_received,
                                                                                                            T *const ptr_array_parameters_received,
                                                                                                            T *const ptr_array_previous_delta_weights_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_delta_weigth;
    
    ptr_array_previous_delta_weights_received[tmp_thread_global_index] = tmp_delta_weigth = learning_rate_received * ptr_array_partial_derivative_received[tmp_thread_global_index] + learning_momentum_received * ptr_array_previous_delta_weights_received[tmp_thread_global_index];

    ptr_array_parameters_received[tmp_thread_global_index] -= tmp_delta_weigth + ptr_array_connections_mask_rergularization_received[tmp_thread_global_index] * weight_decay_received * ptr_array_parameters_received[tmp_thread_global_index];

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template<typename T>
__global__ void kernel__Update_Parameter__Gradient_Descent_Momentum(size_t const size_received,
                                                                                                            T const weight_decay_received,
                                                                                                            T const learning_rate_received,
                                                                                                            T const learning_momentum_received,
                                                                                                            T const *const ptr_array_connections_mask_rergularization_received,
                                                                                                            T *const ptr_array_partial_derivative_received,
                                                                                                            T *const ptr_array_parameters_received,
                                                                                                            T *const ptr_array_previous_delta_weights_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_delta_weigth;

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_previous_delta_weights_received[tmp_thread_global_index] = tmp_delta_weigth = learning_rate_received * ptr_array_partial_derivative_received[tmp_thread_global_index] + learning_momentum_received * ptr_array_previous_delta_weights_received[tmp_thread_global_index];

        ptr_array_parameters_received[tmp_thread_global_index] -= tmp_delta_weigth + ptr_array_connections_mask_rergularization_received[tmp_thread_global_index] * weight_decay_received * ptr_array_parameters_received[tmp_thread_global_index];

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
    }
}

template<typename T>
__global__ void kernel_while__Update_Parameter__Gradient_Descent_Momentum(size_t const size_received,
                                                                                                                    T const weight_decay_received,
                                                                                                                    T const learning_rate_received,
                                                                                                                    T const learning_momentum_received,
                                                                                                                    T const *const ptr_array_connections_mask_rergularization_received,
                                                                                                                    T *const ptr_array_partial_derivative_received,
                                                                                                                    T *const ptr_array_parameters_received,
                                                                                                                    T *const ptr_array_previous_delta_weights_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    T tmp_delta_weigth;

    do
    {
        ptr_array_previous_delta_weights_received[tmp_thread_global_index] = tmp_delta_weigth = learning_rate_received * ptr_array_partial_derivative_received[tmp_thread_global_index] + learning_momentum_received * ptr_array_previous_delta_weights_received[tmp_thread_global_index];

        ptr_array_parameters_received[tmp_thread_global_index] -= tmp_delta_weigth + ptr_array_connections_mask_rergularization_received[tmp_thread_global_index] * weight_decay_received * ptr_array_parameters_received[tmp_thread_global_index];

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Update_Parameter__Gradient_Descent_Momentum__CUDA(size_t const batch_size_received, size_t const training_size_received, size_t const start_index_received, size_t const end_index_received)
{
    T_ const tmp_learning_rate_scale(this->use_Warm_Restarts ? this->Warm_Restarts_Decay() / this->learning_rate : 1_T);

    ++this->optimizer_time_step;

    T_ const *const tmp_ptr_array_connections_mask_rergularization(this->ptr_array_mask_regularized_parameters),
                  tmp_learning_rate(tmp_learning_rate_scale * this->learning_rate),
                  tmp_learning_momentum(this->learning_momentum),
                  tmp_weight_decay(tmp_learning_rate_scale * (this->use_normalized_weight_decay ? this->Normalized_Weight_Decay(batch_size_received, training_size_received) : this->regularization__weight_decay));
    T_ *const tmp_ptr_array_partial_derivative(this->ptr_array_derivatives_parameters),
         *const tmp_ptr_array_parameters(this->ptr_array_parameters),
         *const tmp_ptr_array_previous_delta_weights(this->ptr_array_previous_delta_parameters),
         tmp_delta_weigth;
        
    if(tmp_weight_decay != 0_T)
    {
        if(USE_PARALLEL && end_index_received - start_index_received >= warpSize)
        {
            // KERNEL LAUNCH
            //    1: Launching do-while elements.
            if(this->ptr_array_dim3_grid[1u].x * this->ptr_array_dim3_block[1u].x < end_index_received - start_index_received)
            {
                kernel_while__Update_Parameter__Gradient_Descent_Momentum<T_> <<< this->ptr_array_dim3_grid[1u], this->ptr_array_dim3_block[1u] >>> (end_index_received - start_index_received,
                                                                                                                                                                                                                                tmp_weight_decay,
                                                                                                                                                                                                                                tmp_learning_rate,
                                                                                                                                                                                                                                tmp_learning_momentum,
                                                                                                                                                                                                                                tmp_ptr_array_connections_mask_rergularization + start_index_received,
                                                                                                                                                                                                                                tmp_ptr_array_partial_derivative + start_index_received,
                                                                                                                                                                                                                                tmp_ptr_array_parameters + start_index_received,
                                                                                                                                                                                                                                tmp_ptr_array_previous_delta_weights + start_index_received);
            }
            //    2: Launching size condition.
            else if(this->ptr_array_dim3_grid[1u].x * this->ptr_array_dim3_block[1u].x > end_index_received - start_index_received)
            {
                kernel__Update_Parameter__Gradient_Descent_Momentum<T_> <<< this->ptr_array_dim3_grid[1u], this->ptr_array_dim3_block[1u] >>> (end_index_received - start_index_received,
                                                                                                                                                                                                                        tmp_weight_decay,
                                                                                                                                                                                                                        tmp_learning_rate,
                                                                                                                                                                                                                        tmp_learning_momentum,
                                                                                                                                                                                                                        tmp_ptr_array_connections_mask_rergularization + start_index_received,
                                                                                                                                                                                                                        tmp_ptr_array_partial_derivative + start_index_received,
                                                                                                                                                                                                                        tmp_ptr_array_parameters + start_index_received,
                                                                                                                                                                                                                        tmp_ptr_array_previous_delta_weights + start_index_received);
            }
            //    3: Standard.
            else
            {
                kernel__Update_Parameter__Gradient_Descent_Momentum<T_> <<< this->ptr_array_dim3_grid[1u], this->ptr_array_dim3_block[1u] >>> (tmp_weight_decay,
                                                                                                                                                                                                                        tmp_learning_rate,
                                                                                                                                                                                                                        tmp_learning_momentum,
                                                                                                                                                                                                                        tmp_ptr_array_connections_mask_rergularization + start_index_received,
                                                                                                                                                                                                                        tmp_ptr_array_partial_derivative + start_index_received,
                                                                                                                                                                                                                        tmp_ptr_array_parameters + start_index_received,
                                                                                                                                                                                                                        tmp_ptr_array_previous_delta_weights + start_index_received);
            }
            // |END| KERNEL LAUNCH |END|

            CUDA__Check_Error();
        }
        else
        {
            for(size_t tmp_thread_global_index(start_index_received); tmp_thread_global_index != end_index_received; ++tmp_thread_global_index)
            {
                tmp_ptr_array_previous_delta_weights[tmp_thread_global_index] = tmp_delta_weigth = tmp_learning_rate * tmp_ptr_array_partial_derivative[tmp_thread_global_index] + tmp_learning_momentum * tmp_ptr_array_previous_delta_weights[tmp_thread_global_index];

                tmp_ptr_array_parameters[tmp_thread_global_index] -= tmp_delta_weigth - tmp_ptr_array_connections_mask_rergularization[tmp_thread_global_index] * tmp_weight_decay * tmp_ptr_array_parameters[tmp_thread_global_index];

                tmp_ptr_array_partial_derivative[tmp_thread_global_index] = 0_T;
            }
        }
    }
    else
    {
        if(USE_PARALLEL && end_index_received - start_index_received >= warpSize)
        {
            // KERNEL LAUNCH
            //    1: Launching do-while elements.
            if(this->ptr_array_dim3_grid[1u].x * this->ptr_array_dim3_block[1u].x < end_index_received - start_index_received)
            {
                kernel_while__Update_Parameter__Gradient_Descent_Momentum<T_> <<< this->ptr_array_dim3_grid[1u], this->ptr_array_dim3_block[1u] >>> (end_index_received - start_index_received,
                                                                                                                                                                                                                                tmp_learning_rate,
                                                                                                                                                                                                                                tmp_learning_momentum,
                                                                                                                                                                                                                                tmp_ptr_array_partial_derivative + start_index_received,
                                                                                                                                                                                                                                tmp_ptr_array_parameters + start_index_received,
                                                                                                                                                                                                                                tmp_ptr_array_previous_delta_weights + start_index_received);
            }
            //    2: Launching size condition.
            else if(this->ptr_array_dim3_grid[1u].x * this->ptr_array_dim3_block[1u].x > end_index_received - start_index_received)
            {
                kernel__Update_Parameter__Gradient_Descent_Momentum<T_> <<< this->ptr_array_dim3_grid[1u], this->ptr_array_dim3_block[1u] >>> (end_index_received - start_index_received,
                                                                                                                                                                                                                        tmp_learning_rate,
                                                                                                                                                                                                                        tmp_learning_momentum,
                                                                                                                                                                                                                        tmp_ptr_array_partial_derivative + start_index_received,
                                                                                                                                                                                                                        tmp_ptr_array_parameters + start_index_received,
                                                                                                                                                                                                                        tmp_ptr_array_previous_delta_weights + start_index_received);
            }
            //    3: Standard.
            else
            {
                kernel__Update_Parameter__Gradient_Descent_Momentum<T_> <<< this->ptr_array_dim3_grid[1u], this->ptr_array_dim3_block[1u] >>> (tmp_learning_rate,
                                                                                                                                                                                                                        tmp_learning_momentum,
                                                                                                                                                                                                                        tmp_ptr_array_partial_derivative + start_index_received,
                                                                                                                                                                                                                        tmp_ptr_array_parameters + start_index_received,
                                                                                                                                                                                                                        tmp_ptr_array_previous_delta_weights + start_index_received);
            }
            // |END| KERNEL LAUNCH |END|

            CUDA__Check_Error();
        }
        else
        {
            for(size_t tmp_thread_global_index(start_index_received); tmp_thread_global_index != end_index_received; ++tmp_thread_global_index)
            {
                tmp_ptr_array_previous_delta_weights[tmp_thread_global_index] = tmp_delta_weigth = tmp_learning_rate * tmp_ptr_array_partial_derivative[tmp_thread_global_index] + tmp_learning_momentum * tmp_ptr_array_previous_delta_weights[tmp_thread_global_index];

                tmp_ptr_array_parameters[tmp_thread_global_index] -= tmp_delta_weigth; // Gradient descent

                tmp_ptr_array_partial_derivative[tmp_thread_global_index] = 0_T;
            }
        }
    }
}
