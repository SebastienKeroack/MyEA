#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

#include <Math/Mathematic.hpp>

__device__ void CUDA_Neural_Network::Update_Parameter__iRPROP_plus(size_t const start_index_received, size_t const end_index_received)
{
    if(this->use_Dropout)
    { this->Update_Parameter__iRPROP_plus__CUDA__Dropout(start_index_received, end_index_received); }
    else
    { this->Update_Parameter__iRPROP_plus__CUDA(start_index_received, end_index_received); }
}

template<typename T>
__global__ void kernel__Update_Parameter__iRPROP_plus(bool const error_is_worst_received,
                                                                                    T const increase_factor_received,
                                                                                    T const decrease_factor_received,
                                                                                    T const minimum_delta_received,
                                                                                    T const maximum_delta_received,
                                                                                    T *const ptr_array_partial_derivative_received,
                                                                                    T *const ptr_array_weight_received,
                                                                                    T *const ptr_array_previous_delta_weight_received,
                                                                                    T *const ptr_array_previous_steps_received,
                                                                                    T *const ptr_array_previous_partial_derivative_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative[];
    /* Index map:
        0: delta_step
        1: partial_derivative
        2: delta_weight */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;

    tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_partial_derivative_received[tmp_thread_global_index];

    T const tmp_sign(ptr_array_previous_partial_derivative_received[tmp_thread_global_index] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]);

    if(tmp_sign > T(0))
    {
        tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * increase_factor_received;
        ptr_array_previous_steps_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = MyEA::Math::Minimum<T_>(tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);
        
        ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * tmp_ptr_array_smem[threadIdx.x];

        ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

        ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
    }
    else if(tmp_sign < T(0))
    {
        tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * decrease_factor_received;
        ptr_array_previous_steps_received[tmp_thread_global_index] = MyEA::Math::Maximum<T_>(tmp_ptr_array_smem[threadIdx.x], minimum_delta_received);
                
        if(error_is_worst_received) { ptr_array_weight_received[tmp_thread_global_index] -= ptr_array_previous_delta_weight_received[tmp_thread_global_index]; }
                
        ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = T(0);
    }
    else // if(tmp_sign == T(0))
    {
        ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * ptr_array_previous_steps_received[tmp_thread_global_index];
                
        ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

        ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
    }

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template<typename T>
__global__ void kernel__Update_Parameter__iRPROP_plus(size_t const size_received,
                                                                                    bool const error_is_worst_received,
                                                                                    T const increase_factor_received,
                                                                                    T const decrease_factor_received,
                                                                                    T const minimum_delta_received,
                                                                                    T const maximum_delta_received,
                                                                                    T *const ptr_array_partial_derivative_received,
                                                                                    T *const ptr_array_weight_received,
                                                                                    T *const ptr_array_previous_delta_weight_received,
                                                                                    T *const ptr_array_previous_steps_received,
                                                                                    T *const ptr_array_previous_partial_derivative_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative[];
    /* Index map:
        0: delta_step
        1: partial_derivative
        2: delta_weight */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_partial_derivative_received[tmp_thread_global_index];
        
        T const tmp_sign(ptr_array_previous_partial_derivative_received[tmp_thread_global_index] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]);

        if(tmp_sign > T(0))
        {
            tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * increase_factor_received;
            ptr_array_previous_steps_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = MyEA::Math::Minimum<T_>(tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);
        
            ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * tmp_ptr_array_smem[threadIdx.x];

            ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

            ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
        }
        else if(tmp_sign < T(0))
        {
            tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * decrease_factor_received;
            ptr_array_previous_steps_received[tmp_thread_global_index] = MyEA::Math::Maximum<T_>(tmp_ptr_array_smem[threadIdx.x], minimum_delta_received);
                
            if(error_is_worst_received) { ptr_array_weight_received[tmp_thread_global_index] -= ptr_array_previous_delta_weight_received[tmp_thread_global_index]; }
                
            ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = T(0);
        }
        else // if(tmp_sign == T(0))
        {
            ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * ptr_array_previous_steps_received[tmp_thread_global_index];
                
            ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

            ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
        }

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
    }
}

template<typename T>
__global__ void kernel_while__Update_Parameter__iRPROP_plus(size_t const size_received,
                                                                                            bool const error_is_worst_received,
                                                                                            T const increase_factor_received,
                                                                                            T const decrease_factor_received,
                                                                                            T const minimum_delta_received,
                                                                                            T const maximum_delta_received,
                                                                                            T *const ptr_array_partial_derivative_received,
                                                                                            T *const ptr_array_weight_received,
                                                                                            T *const ptr_array_previous_delta_weight_received,
                                                                                            T *const ptr_array_previous_steps_received,
                                                                                            T *const ptr_array_previous_partial_derivative_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative[];
    /* Index map:
        0: delta_step
        1: partial_derivative
        2: delta_weight */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;
    T tmp_sign;

    do
    {
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_partial_derivative_received[tmp_thread_global_index];
        
        tmp_sign = ptr_array_previous_partial_derivative_received[tmp_thread_global_index] * tmp_ptr_array_smem[threadIdx.x + blockDim.x];

        if(tmp_sign > T(0))
        {
            tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * increase_factor_received;
            ptr_array_previous_steps_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = MyEA::Math::Minimum<T_>(tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);
        
            ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * tmp_ptr_array_smem[threadIdx.x];

            ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

            ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
        }
        else if(tmp_sign < T(0))
        {
            tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * decrease_factor_received;
            ptr_array_previous_steps_received[tmp_thread_global_index] = MyEA::Math::Maximum<T_>(tmp_ptr_array_smem[threadIdx.x], minimum_delta_received);
                
            if(error_is_worst_received) { ptr_array_weight_received[tmp_thread_global_index] -= ptr_array_previous_delta_weight_received[tmp_thread_global_index]; }
                
            ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = T(0);
        }
        else // if(tmp_sign == T(0))
        {
            ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * ptr_array_previous_steps_received[tmp_thread_global_index];
                
            ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

            ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
        }

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Update_Parameter__iRPROP_plus__CUDA(size_t const start_index_received, size_t const end_index_received)
{
    bool const tmp_error_is_worst(this->loss_rprop > this->previous_loss_rprop);

    float const tmp_increase_factor(this->rprop_increase_factor), // 1.2
                    tmp_decrease_factor(this->rprop_decrease_factor), // 0.5
                    tmp_delta_minimum(this->rprop_delta_min), // 1e-6
                    tmp_delta_maximum(this->rprop_delta_max); // 50.0
    
    T_ *const tmp_ptr_array_partial_derivative(this->ptr_array_derivatives_parameters),
         *const tmp_ptr_array_parameters(this->ptr_array_parameters),
         *const tmp_ptr_array_previous_delta_weight(this->ptr_array_previous_delta_parameters),
         *const tmp_ptr_array_previous_step(this->ptr_array_previous_steps),
         *const tmp_ptr_array_previous_partial_derivative(this->ptr_array_previous_derivatives_parameters),
         tmp_partial_derivative,
         tmp_delta_weight,
         tmp_delta_step;

#if defined(COMPILE_DEBUG_PRINT)
    PRINT_FORMAT("rprop error : %f" NEW_LINE, this->loss_rprop);
    PRINT_FORMAT("rprop previous error : %f" NEW_LINE, this->previous_loss_rprop);
#endif
    
    if(USE_PARALLEL && end_index_received - start_index_received >= warpSize)
    {
        LAUNCH_KERNEL_1D(Update_Parameter__iRPROP_plus<T_>,
                                            this->ptr_array_dim3_grid[1u],
                                            this->ptr_array_dim3_block[1u],
                                            this->ptr_array_dim3_block[1u].x * 3u * sizeof(T_),
                                            end_index_received - start_index_received,
                                            tmp_error_is_worst,
                                            tmp_increase_factor,
                                            tmp_decrease_factor,
                                            tmp_delta_minimum,
                                            tmp_delta_maximum,
                                            tmp_ptr_array_partial_derivative + start_index_received,
                                            tmp_ptr_array_parameters + start_index_received,
                                            tmp_ptr_array_previous_delta_weight + start_index_received,
                                            tmp_ptr_array_previous_step + start_index_received,
                                            tmp_ptr_array_previous_partial_derivative + start_index_received)

        CUDA__Check_Error();
    }
    else
    {
        for(size_t i(start_index_received); i != end_index_received; ++i)
        {
            tmp_partial_derivative = tmp_ptr_array_partial_derivative[i];  // Gradient descent
                
            if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative > 0_T)
            {
                tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_increase_factor;
                tmp_ptr_array_previous_step[i] = tmp_delta_step = MyEA::Math::Minimum<T_>(tmp_delta_step, tmp_delta_maximum);

                tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight = -MyEA::Math::Sign<T_>(tmp_partial_derivative) * tmp_delta_step;
                    
                tmp_ptr_array_parameters[i] += tmp_delta_weight;

                tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
            }
            else if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative < 0_T)
            {
                tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_decrease_factor;
                tmp_ptr_array_previous_step[i] = MyEA::Math::Maximum<T_>(tmp_delta_step, tmp_delta_minimum);
                    
                if(tmp_error_is_worst) { tmp_ptr_array_parameters[i] -= tmp_ptr_array_previous_delta_weight[i]; }
                    
                tmp_ptr_array_previous_partial_derivative[i] = 0_T;
            }
            else // if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative == 0_T)
            {
                tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight = -MyEA::Math::Sign<T_>(tmp_partial_derivative) * tmp_ptr_array_previous_step[i];
                    
                tmp_ptr_array_parameters[i] += tmp_delta_weight;

                tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
            }

            tmp_ptr_array_partial_derivative[i] = 0_T; // tmp_partial_derivative
        }
    }
}
    
template<typename T>
__global__ void kernel__Update_Parameter__iRPROP_plus__Dropout(bool const error_is_worst_received,
                                                                                                T const increase_factor_received,
                                                                                                T const decrease_factor_received,
                                                                                                T const minimum_delta_received,
                                                                                                T const maximum_delta_received,
                                                                                                T const *const ptr_array_mask_dropout_parameters_received,
                                                                                                T *const ptr_array_partial_derivative_received,
                                                                                                T *const ptr_array_weight_received,
                                                                                                T *const ptr_array_previous_delta_weight_received,
                                                                                                T *const ptr_array_previous_steps_received,
                                                                                                T *const ptr_array_previous_partial_derivative_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative[];
    /* Index map:
        0: delta_step
        1: partial_derivative
        2: delta_weight */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;
    
    tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_partial_derivative_received[tmp_thread_global_index];

    T const tmp_sign(ptr_array_previous_partial_derivative_received[tmp_thread_global_index] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]);

    if(ptr_array_mask_dropout_parameters_received[tmp_thread_global_index] == T(0)) { ptr_array_previous_delta_weight_received[tmp_thread_global_index] = T(0); }
    else
    {
        if(tmp_sign > T(0))
        {
            tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * increase_factor_received;
            ptr_array_previous_steps_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = MyEA::Math::Minimum<T_>(tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);
            
            ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * tmp_ptr_array_smem[threadIdx.x];

            ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

            ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
        }
        else if(tmp_sign < T(0))
        {
            tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * decrease_factor_received;
            ptr_array_previous_steps_received[tmp_thread_global_index] = MyEA::Math::Maximum<T_>(tmp_ptr_array_smem[threadIdx.x], minimum_delta_received);
                
            if(error_is_worst_received) { ptr_array_weight_received[tmp_thread_global_index] -= ptr_array_previous_delta_weight_received[tmp_thread_global_index]; }
                
            ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = T(0);
        }
        else if(ptr_array_previous_partial_derivative_received[tmp_thread_global_index] == T(0))
        {
            ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * ptr_array_previous_steps_received[tmp_thread_global_index];
                
            ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

            ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
        }
        else { ptr_array_previous_delta_weight_received[tmp_thread_global_index] = T(0); }
    }

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template<typename T>
__global__ void kernel__Update_Parameter__iRPROP_plus__Dropout(size_t const size_received,
                                                                                                bool const error_is_worst_received,
                                                                                                T const increase_factor_received,
                                                                                                T const decrease_factor_received,
                                                                                                T const minimum_delta_received,
                                                                                                T const maximum_delta_received,
                                                                                                T const *const ptr_array_mask_dropout_parameters_received,
                                                                                                T *const ptr_array_partial_derivative_received,
                                                                                                T *const ptr_array_weight_received,
                                                                                                T *const ptr_array_previous_delta_weight_received,
                                                                                                T *const ptr_array_previous_steps_received,
                                                                                                T *const ptr_array_previous_partial_derivative_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative[];
    /* Index map:
        0: delta_step
        1: partial_derivative
        2: delta_weight */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;
    
    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_partial_derivative_received[tmp_thread_global_index];

        T const tmp_sign(ptr_array_previous_partial_derivative_received[tmp_thread_global_index] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]);

        if(ptr_array_mask_dropout_parameters_received[tmp_thread_global_index] == T(0)) { ptr_array_previous_delta_weight_received[tmp_thread_global_index] = T(0); }
        else
        {
            if(tmp_sign > T(0))
            {
                tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * increase_factor_received;
                ptr_array_previous_steps_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = MyEA::Math::Minimum<T_>(tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);
                
                ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * tmp_ptr_array_smem[threadIdx.x];

                ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

                ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
            }
            else if(tmp_sign < T(0))
            {
                tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * decrease_factor_received;
                ptr_array_previous_steps_received[tmp_thread_global_index] = MyEA::Math::Maximum<T_>(tmp_ptr_array_smem[threadIdx.x], minimum_delta_received);
                
                if(error_is_worst_received) { ptr_array_weight_received[tmp_thread_global_index] -= ptr_array_previous_delta_weight_received[tmp_thread_global_index]; }
                
                ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = T(0);
            }
            else if(ptr_array_previous_partial_derivative_received[tmp_thread_global_index] == T(0))
            {
                ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * ptr_array_previous_steps_received[tmp_thread_global_index];
                
                ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

                ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
            }
            else { ptr_array_previous_delta_weight_received[tmp_thread_global_index] = T(0); }
        }

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
    }
}

template<typename T>
__global__ void kernel_while__Update_Parameter__iRPROP_plus__Dropout(size_t const size_received,
                                                                                                        bool const error_is_worst_received,
                                                                                                        T const increase_factor_received,
                                                                                                        T const decrease_factor_received,
                                                                                                        T const minimum_delta_received,
                                                                                                        T const maximum_delta_received,
                                                                                                        T const *const ptr_array_mask_dropout_parameters_received,
                                                                                                        T *const ptr_array_partial_derivative_received,
                                                                                                        T *const ptr_array_weight_received,
                                                                                                        T *const ptr_array_previous_delta_weight_received,
                                                                                                        T *const ptr_array_previous_steps_received,
                                                                                                        T *const ptr_array_previous_partial_derivative_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative[];
    /* Index map:
        0: delta_step
        1: partial_derivative
        2: delta_weight */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;
    
    do
    {
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_partial_derivative_received[tmp_thread_global_index];

        T const tmp_sign(ptr_array_previous_partial_derivative_received[tmp_thread_global_index] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]);

        if(ptr_array_mask_dropout_parameters_received[tmp_thread_global_index] == T(0)) { ptr_array_previous_delta_weight_received[tmp_thread_global_index] = T(0); }
        else
        {
            if(tmp_sign > T(0))
            {
                tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * increase_factor_received;
                ptr_array_previous_steps_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = MyEA::Math::Minimum<T_>(tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);
                
                ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * tmp_ptr_array_smem[threadIdx.x];

                ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

                ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
            }
            else if(tmp_sign < T(0))
            {
                tmp_ptr_array_smem[threadIdx.x] = ptr_array_previous_steps_received[tmp_thread_global_index] * decrease_factor_received;
                ptr_array_previous_steps_received[tmp_thread_global_index] = MyEA::Math::Maximum<T_>(tmp_ptr_array_smem[threadIdx.x], minimum_delta_received);
                
                if(error_is_worst_received) { ptr_array_weight_received[tmp_thread_global_index] -= ptr_array_previous_delta_weight_received[tmp_thread_global_index]; }
                
                ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = T(0);
            }
            else if(ptr_array_previous_partial_derivative_received[tmp_thread_global_index] == T(0))
            {
                ptr_array_previous_delta_weight_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -MyEA::Math::Sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) * ptr_array_previous_steps_received[tmp_thread_global_index];
                
                ptr_array_weight_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

                ptr_array_previous_partial_derivative_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + blockDim.x];
            }
            else { ptr_array_previous_delta_weight_received[tmp_thread_global_index] = T(0); }
        }

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}
    
__device__ void CUDA_Neural_Network::Update_Parameter__iRPROP_plus__CUDA__Dropout(size_t const start_index_received, size_t const end_index_received)
{
    bool const tmp_error_is_worst(this->loss_rprop > this->previous_loss_rprop);

    float const tmp_increase_factor(this->rprop_increase_factor), // 1.2
                    tmp_decrease_factor(this->rprop_decrease_factor), // 0.5
                    tmp_delta_minimum(this->rprop_delta_min), // 1e-6
                    tmp_delta_maximum(this->rprop_delta_max); // 50.0
        
    T_ const *const tmp_ptr_array_mask_dropout_parameters(this->ptr_array_mask_dropout_parameters);
    T_ *const tmp_ptr_array_partial_derivative(this->ptr_array_derivatives_parameters),
         *const tmp_ptr_array_parameters(this->ptr_array_parameters),
         *const tmp_ptr_array_previous_delta_weight(this->ptr_array_previous_delta_parameters),
         *const tmp_ptr_array_previous_step(this->ptr_array_previous_steps),
         *const tmp_ptr_array_previous_partial_derivative(this->ptr_array_previous_derivatives_parameters),
         tmp_partial_derivative,
         tmp_delta_weight,
         tmp_delta_step;

    if(USE_PARALLEL && end_index_received - start_index_received >= warpSize)
    {
        LAUNCH_KERNEL_1D(Update_Parameter__iRPROP_plus__Dropout<T_>,
                                            this->ptr_array_dim3_grid[1u],
                                            this->ptr_array_dim3_block[1u],
                                            this->ptr_array_dim3_block[1u].x * 3u * sizeof(T_),
                                            end_index_received - start_index_received,
                                            tmp_error_is_worst,
                                            tmp_increase_factor,
                                            tmp_decrease_factor,
                                            tmp_delta_minimum,
                                            tmp_delta_maximum,
                                            tmp_ptr_array_mask_dropout_parameters + start_index_received,
                                            tmp_ptr_array_partial_derivative + start_index_received,
                                            tmp_ptr_array_parameters + start_index_received,
                                            tmp_ptr_array_previous_delta_weight + start_index_received,
                                            tmp_ptr_array_previous_step + start_index_received,
                                            tmp_ptr_array_previous_partial_derivative + start_index_received)

        CUDA__Check_Error();
    }
    else
    {
        for(size_t i(start_index_received); i != end_index_received; ++i)
        {
            tmp_partial_derivative = tmp_ptr_array_partial_derivative[i];  // Gradient descent

            if(tmp_ptr_array_mask_dropout_parameters[i] == 0_T) { tmp_ptr_array_previous_delta_weight[i] = 0_T; }
            else
            {
                if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative > 0_T)
                {
                    tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_increase_factor;
                    tmp_ptr_array_previous_step[i] = tmp_delta_step = MyEA::Math::Minimum<T_>(tmp_delta_step, tmp_delta_maximum);
                    
                    tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight = -MyEA::Math::Sign<T_>(tmp_partial_derivative) * tmp_delta_step;

                    tmp_ptr_array_parameters[i] += tmp_delta_weight;

                    tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
                }
                else if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative < 0_T)
                {
                    tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_decrease_factor;
                    tmp_ptr_array_previous_step[i] = MyEA::Math::Maximum<T_>(tmp_delta_step, tmp_delta_minimum);
                        
                    if(tmp_error_is_worst) { tmp_ptr_array_parameters[i] -= tmp_ptr_array_previous_delta_weight[i]; }
                    
                    tmp_ptr_array_previous_partial_derivative[i] = 0_T;
                }
                else if(tmp_ptr_array_previous_partial_derivative[i] == 0_T)
                {
                    tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight = -MyEA::Math::Sign<T_>(tmp_partial_derivative) * tmp_ptr_array_previous_step[i];

                    tmp_ptr_array_parameters[i] += tmp_delta_weight;

                    tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
                }
                else { tmp_ptr_array_previous_delta_weight[i] = 0_T; }
            }

            tmp_ptr_array_partial_derivative[i] = 0_T; // tmp_partial_derivative
        }
    }
}
