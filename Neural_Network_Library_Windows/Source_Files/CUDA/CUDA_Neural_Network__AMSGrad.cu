#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

#include <Math/Mathematic.hpp>

template<typename T>
__global__ void kernel__Update_Parameter__AMSGrad(T const beta1_received,
                                                                                T const beta2_received,
                                                                                T const epsilon_received,
                                                                                T const learning_rate_at_time_t_received,
                                                                                T *const ptr_array_partial_derivative_received,
                                                                                T *const ptr_array_parameters_received,
                                                                                T *const ptr_array_previous_biased_first_moment_received,
                                                                                T *const ptr_array_previous_biased_second_moment_received,
                                                                                T *const ptr_array_previous_biased_second_moment_hat_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_partial_derivative(ptr_array_partial_derivative_received[tmp_thread_global_index]),
       tmp_biased_first_moment,
       tmp_biased_second_moment,
       tmp_previous_biased_second_moment_hat(ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index]);

    ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] = tmp_biased_first_moment = beta1_received * ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] + (T(1) - beta1_received) * tmp_partial_derivative;
    ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] = tmp_biased_second_moment = beta2_received * ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] + (T(1) - beta2_received) * tmp_partial_derivative * tmp_partial_derivative;

    ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index] = tmp_previous_biased_second_moment_hat = MyEA::Math::Maximum<T_>(tmp_previous_biased_second_moment_hat, tmp_biased_second_moment);

    ptr_array_parameters_received[tmp_thread_global_index] -= learning_rate_at_time_t_received * tmp_biased_first_moment / (sqrt(tmp_previous_biased_second_moment_hat) + epsilon_received);

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template<typename T>
__global__ void kernel__Update_Parameter__AMSGrad(size_t const size_received,
                                                                                T const beta1_received,
                                                                                T const beta2_received,
                                                                                T const epsilon_received,
                                                                                T const learning_rate_at_time_t_received,
                                                                                T *const ptr_array_partial_derivative_received,
                                                                                T *const ptr_array_parameters_received,
                                                                                T *const ptr_array_previous_biased_first_moment_received,
                                                                                T *const ptr_array_previous_biased_second_moment_received,
                                                                                T *const ptr_array_previous_biased_second_moment_hat_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    T tmp_partial_derivative,
       tmp_biased_first_moment,
       tmp_biased_second_moment,
       tmp_previous_biased_second_moment_hat;

    if(tmp_thread_global_index < size_received)
    {
        tmp_partial_derivative = ptr_array_partial_derivative_received[tmp_thread_global_index];

        ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] = tmp_biased_first_moment = beta1_received * ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] + (T(1) - beta1_received) * tmp_partial_derivative;
        ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] = tmp_biased_second_moment = beta2_received * ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] + (T(1) - beta2_received) * tmp_partial_derivative * tmp_partial_derivative;

        tmp_previous_biased_second_moment_hat = ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index];
        ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index] = tmp_previous_biased_second_moment_hat = MyEA::Math::Maximum<T_>(tmp_previous_biased_second_moment_hat, tmp_biased_second_moment);

        ptr_array_parameters_received[tmp_thread_global_index] -= learning_rate_at_time_t_received * tmp_biased_first_moment / (sqrt(tmp_previous_biased_second_moment_hat) + epsilon_received);

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
    }
}

template<typename T>
__global__ void kernel_while__Update_Parameter__AMSGrad(size_t const size_received,
                                                                                        T const beta1_received,
                                                                                        T const beta2_received,
                                                                                        T const epsilon_received,
                                                                                        T const learning_rate_at_time_t_received,
                                                                                        T *const ptr_array_partial_derivative_received,
                                                                                        T *const ptr_array_parameters_received,
                                                                                        T *const ptr_array_previous_biased_first_moment_received,
                                                                                        T *const ptr_array_previous_biased_second_moment_received,
                                                                                        T *const ptr_array_previous_biased_second_moment_hat_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    T tmp_partial_derivative,
       tmp_biased_first_moment,
       tmp_biased_second_moment,
       tmp_previous_biased_second_moment_hat;

    do
    {
        tmp_partial_derivative = ptr_array_partial_derivative_received[tmp_thread_global_index];

        ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] = tmp_biased_first_moment = beta1_received * ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] + (T(1) - beta1_received) * tmp_partial_derivative;
        ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] = tmp_biased_second_moment = beta2_received * ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] + (T(1) - beta2_received) * tmp_partial_derivative * tmp_partial_derivative;

        tmp_previous_biased_second_moment_hat = ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index];
        ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index] = tmp_previous_biased_second_moment_hat = MyEA::Math::Maximum<T_>(tmp_previous_biased_second_moment_hat, tmp_biased_second_moment);

        ptr_array_parameters_received[tmp_thread_global_index] -= learning_rate_at_time_t_received * tmp_biased_first_moment / (sqrt(tmp_previous_biased_second_moment_hat) + epsilon_received);

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Update_Parameter__AMSGrad(T const weight_decay_received,
                                                                                T const beta1_received,
                                                                                T const beta2_received,
                                                                                T const epsilon_received,
                                                                                T const learning_rate_at_time_t_received,
                                                                                T const *const ptr_array_connections_mask_rergularization_received,
                                                                                T *const ptr_array_partial_derivative_received,
                                                                                T *const ptr_array_parameters_received,
                                                                                T *const ptr_array_previous_biased_first_moment_received,
                                                                                T *const ptr_array_previous_biased_second_moment_received,
                                                                                T *const ptr_array_previous_biased_second_moment_hat_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_partial_derivative(ptr_array_partial_derivative_received[tmp_thread_global_index]),
       tmp_biased_first_moment,
       tmp_biased_second_moment,
       tmp_previous_biased_second_moment_hat(ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index]);

    ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] = tmp_biased_first_moment = beta1_received * ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] + (T(1) - beta1_received) * tmp_partial_derivative;
    ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] = tmp_biased_second_moment = beta2_received * ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] + (T(1) - beta2_received) * tmp_partial_derivative * tmp_partial_derivative;

    ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index] = tmp_previous_biased_second_moment_hat = MyEA::Math::Maximum<T_>(tmp_previous_biased_second_moment_hat, tmp_biased_second_moment);

    ptr_array_parameters_received[tmp_thread_global_index] -= learning_rate_at_time_t_received * tmp_biased_first_moment / (sqrt(tmp_previous_biased_second_moment_hat) + epsilon_received) + ptr_array_connections_mask_rergularization_received[tmp_thread_global_index] * weight_decay_received * ptr_array_parameters_received[tmp_thread_global_index];

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template<typename T>
__global__ void kernel__Update_Parameter__AMSGrad(size_t const size_received,
                                                                                T const weight_decay_received,
                                                                                T const beta1_received,
                                                                                T const beta2_received,
                                                                                T const epsilon_received,
                                                                                T const learning_rate_at_time_t_received,
                                                                                T const *const ptr_array_connections_mask_rergularization_received,
                                                                                T *const ptr_array_partial_derivative_received,
                                                                                T *const ptr_array_parameters_received,
                                                                                T *const ptr_array_previous_biased_first_moment_received,
                                                                                T *const ptr_array_previous_biased_second_moment_received,
                                                                                T *const ptr_array_previous_biased_second_moment_hat_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    T tmp_partial_derivative,
       tmp_biased_first_moment,
       tmp_biased_second_moment,
       tmp_previous_biased_second_moment_hat;

    if(tmp_thread_global_index < size_received)
    {
        tmp_partial_derivative = ptr_array_partial_derivative_received[tmp_thread_global_index];

        ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] = tmp_biased_first_moment = beta1_received * ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] + (T(1) - beta1_received) * tmp_partial_derivative;
        ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] = tmp_biased_second_moment = beta2_received * ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] + (T(1) - beta2_received) * tmp_partial_derivative * tmp_partial_derivative;

        tmp_previous_biased_second_moment_hat = ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index];
        ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index] = tmp_previous_biased_second_moment_hat = MyEA::Math::Maximum<T_>(tmp_previous_biased_second_moment_hat, tmp_biased_second_moment);

        ptr_array_parameters_received[tmp_thread_global_index] -= learning_rate_at_time_t_received * tmp_biased_first_moment / (sqrt(tmp_previous_biased_second_moment_hat) + epsilon_received) + ptr_array_connections_mask_rergularization_received[tmp_thread_global_index] * weight_decay_received * ptr_array_parameters_received[tmp_thread_global_index];

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
    }
}

template<typename T>
__global__ void kernel_while__Update_Parameter__AMSGrad(size_t const size_received,
                                                                                        T const weight_decay_received,
                                                                                        T const beta1_received,
                                                                                        T const beta2_received,
                                                                                        T const epsilon_received,
                                                                                        T const learning_rate_at_time_t_received,
                                                                                        T const *const ptr_array_connections_mask_rergularization_received,
                                                                                        T *const ptr_array_partial_derivative_received,
                                                                                        T *const ptr_array_parameters_received,
                                                                                        T *const ptr_array_previous_biased_first_moment_received,
                                                                                        T *const ptr_array_previous_biased_second_moment_received,
                                                                                        T *const ptr_array_previous_biased_second_moment_hat_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    T tmp_partial_derivative,
       tmp_biased_first_moment,
       tmp_biased_second_moment,
       tmp_previous_biased_second_moment_hat;

    do
    {
        tmp_partial_derivative = ptr_array_partial_derivative_received[tmp_thread_global_index];

        ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] = tmp_biased_first_moment = beta1_received * ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] + (T(1) - beta1_received) * tmp_partial_derivative;
        ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] = tmp_biased_second_moment = beta2_received * ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] + (T(1) - beta2_received) * tmp_partial_derivative * tmp_partial_derivative;

        tmp_previous_biased_second_moment_hat = ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index];
        ptr_array_previous_biased_second_moment_hat_received[tmp_thread_global_index] = tmp_previous_biased_second_moment_hat = MyEA::Math::Maximum<T_>(tmp_previous_biased_second_moment_hat, tmp_biased_second_moment);

        ptr_array_parameters_received[tmp_thread_global_index] -= learning_rate_at_time_t_received * tmp_biased_first_moment / (sqrt(tmp_previous_biased_second_moment_hat) + epsilon_received) + ptr_array_connections_mask_rergularization_received[tmp_thread_global_index] * weight_decay_received * ptr_array_parameters_received[tmp_thread_global_index];

        ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Update_Parameter__AMSGrad(size_t const batch_size_received, size_t const training_size_received, size_t const start_index_received, size_t const end_index_received)
{
    size_t i;
    
    T_ const tmp_learning_rate_scale(this->use_Warm_Restarts ? this->Warm_Restarts_Decay() / this->adam_learning_rate : 1_T);

    ++this->optimizer_time_step;

    T_ const *const tmp_ptr_array_connections_mask_rergularization(this->ptr_array_mask_regularized_parameters),
                  tmp_learning_rate(tmp_learning_rate_scale * this->adam_learning_rate),
                  tmp_weight_decay(this->use_normalized_weight_decay ? this->Normalized_Weight_Decay(batch_size_received, training_size_received) : this->regularization__weight_decay),
                  tmp_beta1(this->adam_beta1),
                  tmp_beta2(this->adam_beta2),
                  tmp_epsilon(this->adam_epsilon),
                  tmp_adam_epochs(this->optimizer_time_step),
                  tmp_learning_rate_at_time_t(tmp_learning_rate * sqrt(1_T - pow(tmp_beta2, tmp_adam_epochs)) / (1_T - pow(tmp_beta1, tmp_adam_epochs)));
    T_ *const tmp_ptr_array_partial_derivative(this->ptr_array_derivatives_parameters),
         *const tmp_ptr_array_parameters(this->ptr_array_parameters),
         *const tmp_ptr_array_previous_biased_first_moment(this->ptr_array_previous_biased_first_moment),
         *const tmp_ptr_array_previous_biased_second_moment(this->ptr_array_previous_biased_second_moment),
         *const tmp_ptr_array_previous_biased_second_moment_hat(this->ptr_array_previous_biased_second_moment_hat),
         tmp_partial_derivative,
         tmp_biased_first_moment,
         tmp_biased_second_moment,
         tmp_previous_biased_second_moment_hat;
    
    if(tmp_weight_decay != 0_T)
    {
        if(USE_PARALLEL && end_index_received - start_index_received >= warpSize)
        {
            LAUNCH_KERNEL_1D(Update_Parameter__AMSGrad<T_>,
                                            this->ptr_array_dim3_grid[1u],
                                            this->ptr_array_dim3_block[1u],
                                            0_zu,
                                            end_index_received - start_index_received,
                                            tmp_weight_decay,
                                            tmp_beta1,
                                            tmp_beta2,
                                            tmp_epsilon,
                                            tmp_learning_rate_at_time_t,
                                            tmp_ptr_array_connections_mask_rergularization + start_index_received,
                                            tmp_ptr_array_partial_derivative + start_index_received,
                                            tmp_ptr_array_parameters + start_index_received,
                                            tmp_ptr_array_previous_biased_first_moment + start_index_received,
                                            tmp_ptr_array_previous_biased_second_moment + start_index_received,
                                            tmp_ptr_array_previous_biased_second_moment_hat + start_index_received)

            CUDA__Check_Error();
        }
        else
        {
            for(i = start_index_received; i != end_index_received; ++i)
            {
                tmp_partial_derivative = tmp_ptr_array_partial_derivative[i];

                tmp_ptr_array_previous_biased_first_moment[i] = tmp_biased_first_moment = tmp_beta1 * tmp_ptr_array_previous_biased_first_moment[i] + (1_T - tmp_beta1) * tmp_partial_derivative;
                tmp_ptr_array_previous_biased_second_moment[i] = tmp_biased_second_moment = tmp_beta2 * tmp_ptr_array_previous_biased_second_moment[i] + (1_T - tmp_beta2) * tmp_partial_derivative * tmp_partial_derivative;

                tmp_previous_biased_second_moment_hat = tmp_ptr_array_previous_biased_second_moment_hat[i];
                tmp_ptr_array_previous_biased_second_moment_hat[i] = tmp_previous_biased_second_moment_hat = MyEA::Math::Maximum<T_>(tmp_previous_biased_second_moment_hat, tmp_biased_second_moment);

                tmp_ptr_array_parameters[i] -= tmp_learning_rate_at_time_t * tmp_biased_first_moment / (sqrt(tmp_previous_biased_second_moment_hat) + tmp_epsilon) + tmp_ptr_array_connections_mask_rergularization[i] * tmp_weight_decay * tmp_ptr_array_parameters[i];

                tmp_ptr_array_partial_derivative[i] = 0_T;
            }
        }
    }
    else
    {
        if(USE_PARALLEL && end_index_received - start_index_received >= warpSize)
        {
            LAUNCH_KERNEL_1D(Update_Parameter__AMSGrad<T_>,
                                            this->ptr_array_dim3_grid[1u],
                                            this->ptr_array_dim3_block[1u],
                                            0_zu,
                                            end_index_received - start_index_received,
                                            tmp_beta1,
                                            tmp_beta2,
                                            tmp_epsilon,
                                            tmp_learning_rate_at_time_t,
                                            tmp_ptr_array_partial_derivative + start_index_received,
                                            tmp_ptr_array_parameters + start_index_received,
                                            tmp_ptr_array_previous_biased_first_moment + start_index_received,
                                            tmp_ptr_array_previous_biased_second_moment + start_index_received,
                                            tmp_ptr_array_previous_biased_second_moment_hat + start_index_received)

            CUDA__Check_Error();
        }
        else
        {
            for(i = start_index_received; i != end_index_received; ++i)
            {
                tmp_partial_derivative = tmp_ptr_array_partial_derivative[i];

                tmp_ptr_array_previous_biased_first_moment[i] = tmp_biased_first_moment = tmp_beta1 * tmp_ptr_array_previous_biased_first_moment[i] + (1_T - tmp_beta1) * tmp_partial_derivative;
                tmp_ptr_array_previous_biased_second_moment[i] = tmp_biased_second_moment = tmp_beta2 * tmp_ptr_array_previous_biased_second_moment[i] + (1_T - tmp_beta2) * tmp_partial_derivative * tmp_partial_derivative;

                tmp_previous_biased_second_moment_hat = tmp_ptr_array_previous_biased_second_moment_hat[i];
                tmp_ptr_array_previous_biased_second_moment_hat[i] = tmp_previous_biased_second_moment_hat = MyEA::Math::Maximum<T_>(tmp_previous_biased_second_moment_hat, tmp_biased_second_moment);

                tmp_ptr_array_parameters[i] -= tmp_learning_rate_at_time_t * tmp_biased_first_moment / (sqrt(tmp_previous_biased_second_moment_hat) + tmp_epsilon);

                tmp_ptr_array_partial_derivative[i] = 0_T;
            }
        }
    }
}
