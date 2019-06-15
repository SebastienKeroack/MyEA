#include <Tools/CUDA_Fill_1D.cuh>
#include <Tools/CUDA_Zero_1D.cuh>

#include <CUDA/CUDA_Neural_Network.cuh>
#include <CUDA/CUDA_Reduce.cuh>
#include <CUDA/CUDA_Transpose.cuh>
#include <CUDA/CUDA_Multiply_1D.cuh>
#include <CUDA/CUDA_Activation_Functions.cuh>
    
#include <Neural_Network/Neural_Network.hpp>

__device__ void CUDA_Neural_Network::Merge__Post__Training(void)
{
    Reduce::Reduce<T_>(this->number_threads,
                                     1_zu,
                                     this->ptr_array_reduce_loss_values,
                                     this->ptr_array_loss_values,
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);

    Reduce::Reduce<T_>(this->number_threads,
                                     1_zu,
                                     this->ptr_array_reduce_accuracy_values[0u],
                                     this->ptr_array_accuracy_values[0u],
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);
    
    Reduce::Reduce<T_>(this->number_threads,
                                     1_zu,
                                     this->ptr_array_reduce_accuracy_values[1u],
                                     this->ptr_array_accuracy_values[1u],
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);

    Reduce::Reduce<T_>(this->number_threads,
                                     1_zu,
                                     this->ptr_array_reduce_accuracy_values[2u],
                                     this->ptr_array_accuracy_values[2u],
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);

    Reduce::Reduce<T_>(this->number_threads,
                                     1_zu,
                                     this->ptr_array_reduce_accuracy_values[3u],
                                     this->ptr_array_accuracy_values[3u],
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);
    
    Reduce::Reduce<T_>(this->number_threads,
                                     1_zu,
                                     this->ptr_array_reduce_accuracy_values[4u],
                                     this->ptr_array_accuracy_values[4u],
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);

    if(this->type_loss_function == MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_BIT)
    {
        Reduce::Reduce<size_t>(this->number_threads,
                                              1_zu,
                                              this->ptr_array_reduce_bit_fail_values,
                                              this->ptr_array_number_bit_fail,
                                              this->ptr_array_dim3_grid_reduce_threads,
                                              this->ptr_array_dim3_block_reduce_threads);
    }
    
    // Synchronize to see the variable reduced.
    if(this->number_threads >= static_cast<size_t>(warpSize * 2)) { CUDA__Check_Error(); }

    if(this->type_loss_function == MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_BIT)
    {
        *this->ptr_array_number_bit_fail = *this->ptr_array_reduce_bit_fail_values;
    }

    *this->ptr_array_loss_values = *this->ptr_array_reduce_loss_values;

    this->ptr_array_accuracy_values[0u][0u] = this->ptr_array_reduce_accuracy_values[0u][0u];
    this->ptr_array_accuracy_values[1u][0u] = this->ptr_array_reduce_accuracy_values[1u][0u];
    this->ptr_array_accuracy_values[2u][0u] = this->ptr_array_reduce_accuracy_values[2u][0u];
    this->ptr_array_accuracy_values[3u][0u] = this->ptr_array_reduce_accuracy_values[3u][0u];
    this->ptr_array_accuracy_values[4u][0u] = this->ptr_array_reduce_accuracy_values[4u][0u];
}
    
__global__ void kernel__CNeural_Network__Reset__Loss(class CUDA_Neural_Network *ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Reset__Loss(); }

__host__ __device__ void CUDA_Neural_Network::Reset__Loss(void)
{
#if defined(__CUDA_ARCH__)
    this->number_accuracy_trial = 0u;
    
    Zero_1D<size_t>(this->number_threads,
                                      this->ptr_array_number_loss,
                                      this->ptr_array_dim3_grid,
                                      this->ptr_array_dim3_block);
    
    Zero_1D<size_t>(this->number_threads,
                                      this->ptr_array_number_bit_fail,
                                      this->ptr_array_dim3_grid,
                                      this->ptr_array_dim3_block);
    
    Zero_1D<T_>(this->number_threads,
                         this->ptr_array_loss_values,
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
    
    Zero_1D<T_>(this->number_threads,
                         this->ptr_array_accuracy_values[0u],
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
    
    Zero_1D<T_>(this->number_threads,
                         this->ptr_array_accuracy_values[1u],
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
    
    Zero_1D<T_>(this->number_threads,
                         this->ptr_array_accuracy_values[2u],
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
    
    Zero_1D<T_>(this->number_threads,
                         this->ptr_array_accuracy_values[3u],
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
    
    Zero_1D<T_>(this->number_threads,
                         this->ptr_array_accuracy_values[4u],
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
#else
    kernel__CNeural_Network__Reset__Loss <<< 1u, 1u >>> (this);
#endif
}
    
__device__ T_ const * CUDA_Neural_Network::Get__Outputs(size_t const thread_index_received) const
{
    return(&(this->ptr_last_layer - 1)->ptr_array_neuron_units->ptr_array_values[thread_index_received * (this->number_outputs + 1u)]); // Add bias
}

__device__ T_ const * CUDA_Neural_Network::Get__Outputs(size_t const thread_index_received, size_t const time_step_index_received) const
{
    return(&(this->ptr_last_layer - 1)->ptr_array_neuron_units->ptr_array_values[thread_index_received + this->batch_size * this->total_neuron_units_allocated * time_step_index_received]);
}

__device__ T_ Activation_Derived(T_ const activation_steepness_received,
                                                T_ const summation_received,
                                                T_ const value_received,
                                                enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const activation_function_received)
{
    switch(activation_function_received)
    {
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_LINEAR: return(Activation_Function_LINEAR_derive_t(1_T));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_LINEAR_PIECE: return(Activation_Function_LINEAR_PIECE_derive_t(1_T));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_LINEAR_PIECE_SYMMETRIC: return(Activation_Function_LINEAR_PIECE_SYMMETRIC_derive_t(1_T));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SIGMOID: return(Activation_Function_SIGMOID_derive_t(1_T, value_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SIGMOID_STEPWISE: return(Activation_Function_SIGMOID_STEPWISE_derive_t(1_T, value_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_TANH: return(Activation_Function_TANH_derive_t(1_T, value_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_TANH_STEPWISE: return(Activation_Function_TANH_STEPWISE_derive_t(1_T, value_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_THRESHOLD:
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_THRESHOLD_SYMMETRIC:
            PRINT_FORMAT("%s: ERROR: Can not training the neural network with this type (%u) of activation function." NEW_LINE,
                                    __FUNCTION__,
                                    static_cast<size_t>(activation_function_received));
                break;
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_GAUSSIAN:
            return(Activation_Function_GAUSSIAN_derive_t(1_T,
                                                                                  value_received,
                                                                                  summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_GAUSSIAN_SYMMETRIC:
            return(Activation_Function_GAUSSIAN_SYMMETRIC_derive_t(1_T,
                                                                                                      value_received,
                                                                                                      summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_ELLIOT: return(Activation_Function_ELLIOT_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_ELLIOT_SYMMETRIC: return(Activation_Function_ELLIOT_SYMMETRIC_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SINE: return(Activation_Function_SIN_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SINE_SYMMETRIC: return(Activation_Function_SIN_SYMMETRIC_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_COSINE: return(Activation_Function_COS_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_COSINE_SYMMETRIC: return(Activation_Function_COS_SYMMETRIC_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_RELU: return(Activation_Function_RELU_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_LEAKY_RELU: return(Activation_Function_LRELU_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_PARAMETRIC_RELU: return(Activation_Function_PRELU_derive_t(1_T, summation_received));
        default:
            PRINT_FORMAT("%s: ERROR: Can not find the derivative of activation function (type=%u)." NEW_LINE,
                                    __FUNCTION__,
                                    static_cast<size_t>(activation_function_received));
                break;
    }

    return(0_T);
}

__device__ T_ Activation_Derived(T_ const activation_steepness_received,
                                                T_ const summation_received,
                                                T_ const value_received,
                                                enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const activation_function_received,
                                                enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const type_loss_function_received)
{
    switch(activation_function_received)
    {
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_LINEAR: return(Activation_Function_LINEAR_derive_t(1_T));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_LINEAR_PIECE: return(Activation_Function_LINEAR_PIECE_derive_t(1_T));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_LINEAR_PIECE_SYMMETRIC: return(Activation_Function_LINEAR_PIECE_SYMMETRIC_derive_t(1_T));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SIGMOID: return(Activation_Function_SIGMOID_derive_t(1_T, value_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SIGMOID_STEPWISE: return(Activation_Function_SIGMOID_STEPWISE_derive_t(1_T, value_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_TANH: return(Activation_Function_TANH_derive_t(1_T, value_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_TANH_STEPWISE: return(Activation_Function_TANH_STEPWISE_derive_t(1_T, value_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_THRESHOLD:
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_THRESHOLD_SYMMETRIC:
            PRINT_FORMAT("%s: ERROR: Can not training the neural network with this type (%u) of activation function." NEW_LINE,
                                    __FUNCTION__,
                                    static_cast<size_t>(activation_function_received));
                break;
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_GAUSSIAN:
            return(Activation_Function_GAUSSIAN_derive_t(1_T,
                                                                                value_received,
                                                                                summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_GAUSSIAN_SYMMETRIC:
            return(Activation_Function_GAUSSIAN_SYMMETRIC_derive_t(1_T,
                                                                                                    value_received,
                                                                                                    summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_ELLIOT: return(Activation_Function_ELLIOT_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_ELLIOT_SYMMETRIC: return(Activation_Function_ELLIOT_SYMMETRIC_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SINE: return(Activation_Function_SIN_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SINE_SYMMETRIC: return(Activation_Function_SIN_SYMMETRIC_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_COSINE: return(Activation_Function_COS_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_COSINE_SYMMETRIC: return(Activation_Function_COS_SYMMETRIC_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_RELU: return(Activation_Function_RELU_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_LEAKY_RELU: return(Activation_Function_LRELU_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_PARAMETRIC_RELU: return(Activation_Function_PRELU_derive_t(1_T, summation_received));
        case  MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SOFTMAX: return(activation_steepness_received);
        default:
            PRINT_FORMAT("%s: ERROR: Can not find the derivative of activation function (type=%u)." NEW_LINE,
                                    __FUNCTION__,
                                    static_cast<size_t>(activation_function_received));
                break;
    }

    return(0_T);
}

__device__ void Update_Accuracy(T_ const error_received,
                                                  T_ const accuracy_variance_received,
                                                  float *const ptr_accuracy_value_received)
{
    if(MyEA::Math::Absolute<T_>(error_received) <= accuracy_variance_received)
    { ++*ptr_accuracy_value_received; }
}

__device__ void Update_Accuracy__atomic(T_ const error_received,
                                                                T_ const accuracy_variance_received,
                                                                float *const ptr_accuracy_value_received)
{
    if(MyEA::Math::Absolute<T_>(error_received) <= accuracy_variance_received)
    { atomicAdd(ptr_accuracy_value_received, 1.0f); }
}

__device__ void Update_Error(T_ const observed_output_received,
                                            T_ const desired_output_received,
                                            T_ const error_received,
                                            float *const ptr_loss_values_received,
                                            enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const type_loss_function_received)
{
    T_ tmp_error;
        
    switch(type_loss_function_received)
    {
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_ME:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L1: tmp_error = error_received; break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MAE: tmp_error = MyEA::Math::Absolute<T_>(error_received); break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L2:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MSE:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_RMSE:
            tmp_error = error_received * error_received; // (Û - U)2, square the difference
                break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MAPE:
            tmp_error = error_received / observed_output_received;

            tmp_error = MyEA::Math::Absolute<T_>(tmp_error);
                break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_SMAPE:
            tmp_error = MyEA::Math::Absolute<T_>(error_received);

            tmp_error /= MyEA::Math::Absolute<T_>(desired_output_received) + MyEA::Math::Absolute<T_>(observed_output_received);
                break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MASE_SEASONAL:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MASE_NON_SEASONAL:
            PRINT_FORMAT("%s: [FUNCTION DEPRECATED] Need to Fix MASE algorithm." NEW_LINE, __FUNCTION__);
                return;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_CROSS_ENTROPY:
            tmp_error = observed_output_received != 0_T ? observed_output_received : MyEA::Math::Maximum<T_>(observed_output_received, 1.0e-6_T); // Numerical stability.
            
            // TODO: Make cross-entropy multi label / binary.
            //if(this->Use__Multi_Label() || this->number_outputs == 1_zu)
            //{
            //    tmp_error = -(desired_output_received * log(tmp_error) + (1_T - desired_output_received) * log(1_T - tmp_error));
            //}
            //else
            {
                tmp_error = -(desired_output_received * log(tmp_error));
            }
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Loss type (%u) is not managed in the switch." NEW_LINE,
                                     __FUNCTION__,
                                     type_loss_function_received);
                return;
    }

    *ptr_loss_values_received += static_cast<float>(tmp_error);
}

__device__ void Update_Error__atomic(T_ const observed_output_received,
                                                         T_ const desired_output_received,
                                                         T_ const error_received,
                                                         float *const ptr_loss_values_received,
                                                         enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const type_loss_function_received)
{
    T_ tmp_error;
        
    switch(type_loss_function_received)
    {
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_ME:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L1: tmp_error = error_received; break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MAE: tmp_error = MyEA::Math::Absolute<T_>(error_received); break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L2:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MSE:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_RMSE:
            tmp_error = error_received * error_received; // (Û - U)2, square the difference
                break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MAPE:
            tmp_error = error_received / observed_output_received;

            tmp_error = MyEA::Math::Absolute<T_>(tmp_error);
                break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_SMAPE:
            tmp_error = MyEA::Math::Absolute<T_>(error_received);

            tmp_error /= MyEA::Math::Absolute<T_>(desired_output_received) + MyEA::Math::Absolute<T_>(observed_output_received);
                break;
                /*
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MASE: // Non seasonal time series
            PRINT_FORMAT("%s: [FUNCTION DEPRECATED] Need to Fix MASE algorithm." NEW_LINE, __FUNCTION__);
                return;
                */
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_CROSS_ENTROPY:
            tmp_error = observed_output_received != 0_T ? observed_output_received : MyEA::Math::Maximum<T_>(observed_output_received, 1.0e-6_T); // Numerical stability.
            
            // TODO: Make cross-entropy multi label / binary.
            //if(this->Use__Multi_Label() || this->number_outputs == 1_zu)
            //{
            //    tmp_error = -(desired_output_received * log(tmp_error) + (1_T - desired_output_received) * log(1_T - tmp_error));
            //}
            //else
            {
                tmp_error = -(desired_output_received * log(tmp_error));
            }
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Loss type (%u) is not managed in the switch." NEW_LINE,
                                     __FUNCTION__,
                                     type_loss_function_received);
                return;
    }

    atomicAdd(ptr_loss_values_received, static_cast<float>(tmp_error));
}

__device__ void Update_Error__Binary_Cross_Entropy(T_ const observed_output_received,
                                                                                T_ const desired_output_received,
                                                                                float *const ptr_loss_values_received)
{
    T_ tmp_error(desired_output_received * -log(observed_output_received) + (1_T - desired_output_received) * logf(1_T - observed_output_received));

    *ptr_loss_values_received += static_cast<float>(tmp_error);
}

__device__ void Update_Error__Binary_Cross_Entropy__atomic(T_ const observed_output_received,
                                                                                             T_ const desired_output_received,
                                                                                             float *const ptr_loss_values_received)
{
    T_ tmp_error(desired_output_received * -log(observed_output_received) + (1_T - desired_output_received) * logf(1_T - observed_output_received));

    atomicAdd(ptr_loss_values_received, static_cast<float>(tmp_error));
}

__device__ void Update_Error__Bit_Fail(T_ const error_received,
                                                          T_ const bit_fail_limit_received,
                                                          size_t *const ptr_bit_fail_values_received)
{
    if(MyEA::Math::Absolute<T_>(error_received) >= bit_fail_limit_received)
    { ++*ptr_bit_fail_values_received; }
}

__device__ void Update_Error__Bit_Fail__atomic(T_ const error_received,
                                                                       T_ const bit_fail_limit_received,
                                                                       size_t *const ptr_bit_fail_values_received)
{
    if(MyEA::Math::Absolute<T_>(error_received) >= bit_fail_limit_received)
    { atomicAdd(ptr_bit_fail_values_received, 1_zu); }
}


__global__ void kernel__CNeural_Network__Clear_Train_Arrays(class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->device__Clear_Train_Arrays(); }

void Neural_Network::Clear_Training_Arrays__CUDA(void)
{ kernel__CNeural_Network__Clear_Train_Arrays <<< 1u, 1u >>> (this->ptr_device_Neural_Network); }

__device__ void CUDA_Neural_Network::device__Clear_Train_Arrays(void)
{
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;
    
    // Weights slope.
    if(this->ptr_array_derivatives_parameters == nullptr)
    {
        T_ *tmp_ptr_array_derivate_weights(new T_[this->number_threads * this->total_parameters_allocated]);
        if(tmp_ptr_array_derivate_weights == nullptr)
        {
            PRINT_FORMAT("ERROR: Can not allocate memory." NEW_LINE);

            return;
        }
        this->ptr_array_derivatives_parameters = tmp_ptr_array_derivate_weights;
    }

    this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(this->number_threads * this->total_parameters_allocated,
                                                                                                                                           0u,
                                                                                                                                           tmp_dim3_grid,
                                                                                                                                           tmp_dim3_block);

    Zero_1D<T_>(this->number_threads * this->total_parameters_allocated,
                        this->ptr_array_derivatives_parameters,
                        &tmp_dim3_grid,
                        &tmp_dim3_block);
    // |END| Weights slope. |END|
    
    this->Clear_Optimizer();
    
    this->warm_restarts_maximum_learning_rate = this->warm_restarts_initial_maximum_learning_rate;
    this->warm_restarts_T_i = this->warm_restarts_initial_T_i;
}

__device__ void CUDA_Neural_Network::Clear_Optimizer(void)
{
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NONE: break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
            if(this->learning_momentum != 0.0f && this->ptr_array_previous_delta_parameters != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_delta_parameters,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
            // Previous train slopes.
            if(this->ptr_array_previous_derivatives_parameters != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_derivatives_parameters,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
            // |END| Previous train slopes. |END|
                
            // Previous steps.
            if(this->ptr_array_previous_steps != nullptr)
            {
                Memory::Fill_1D<T_>(this->total_parameters_allocated,
                                                                     this->ptr_array_previous_steps,
                                                                     this->rprop_delta_zero,
                                                                     this->ptr_array_dim3_grid + 1,
                                                                     this->ptr_array_dim3_block + 1);
            }
            // |END| Previous steps. |END|
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
            this->loss_rprop = FLT_MAX;
            this->previous_loss_rprop = FLT_MAX;
                
            // Previous train slopes.
            if(this->ptr_array_previous_derivatives_parameters != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_derivatives_parameters,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
            // |END| Previous train slopes. |END|
                
            // Previous steps.
            if(this->ptr_array_previous_steps != nullptr)
            {
                Memory::Fill_1D<T_>(this->total_parameters_allocated,
                                                                     this->ptr_array_previous_steps,
                                                                     this->rprop_delta_zero,
                                                                     this->ptr_array_dim3_grid + 1,
                                                                     this->ptr_array_dim3_block + 1);
            }
            // |END| Previous steps. |END|

            // Previous delta weights.
            if(this->ptr_array_previous_delta_parameters != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_delta_parameters,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
            // |END| Previous delta weights. |END|
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP: break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP: break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad:
            if(this->ptr_array_previous_biased_first_moment != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_first_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                        
            if(this->ptr_array_previous_biased_second_moment != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_second_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }

            if(this->ptr_array_previous_biased_second_moment_hat != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_second_moment_hat,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
            if(this->ptr_array_previous_biased_first_moment != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_first_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                
            if(this->ptr_array_previous_biased_second_moment != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_second_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM:
            if(this->ptr_array_previous_biased_first_moment != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_first_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                
            if(this->ptr_array_previous_biased_second_moment != nullptr)
            {
                Zero_1D<T_>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_second_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }

            this->adam_previous_beta2 = 0_T;
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Can not reset parameters of the optimizer (%u)." NEW_LINE,
                        __FUNCTION__,
                        this->type_optimizer_function);
                break;
    }
        
    this->optimizer_time_step = 0_T;
    this->epoch_time_step = 1_T;
}
    
__global__ void kernel__CNeural_Network__Set__Loss_Function(enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const type_loss_function_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Loss_Function(type_loss_function_received); }

__host__ __device__ void CUDA_Neural_Network::Set__Loss_Function(enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const type_loss_function_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Loss_Function <<< 1u, 1u >>> (type_loss_function_received, this);
#else
    this->type_loss_function = type_loss_function_received;
#endif
}

__global__ void kernel__CNeural_Network__Set__Accuracy_Function(enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS const type_accuracy_function_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Accuracy_Function(type_accuracy_function_received); }

__host__ __device__ void CUDA_Neural_Network::Set__Accuracy_Function(enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS const type_accuracy_function_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Accuracy_Function <<< 1u, 1u >>> (type_accuracy_function_received, this);
#else
    this->type_accuracy_function = type_accuracy_function_received;
#endif
}

__global__ void kernel__CNeural_Network__Set__Bit_Fail_Limit(T_ const bit_fail_limit_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Bit_Fail_Limit(bit_fail_limit_received); }

__host__ __device__ void CUDA_Neural_Network::Set__Bit_Fail_Limit(T_ const bit_fail_limit_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Bit_Fail_Limit <<< 1u, 1u >>> (bit_fail_limit_received, this);
#else
    this->bit_fail_limit = bit_fail_limit_received;
#endif
}
    
__global__ void kernel__CNeural_Network__Set__Optimizer_Function(enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS const type_optimizer_function_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Optimizer_Function(type_optimizer_function_received); }

__host__ __device__ void CUDA_Neural_Network::Set__Optimizer_Function(enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS const type_optimizer_function_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Optimizer_Function <<< 1u, 1u >>> (type_optimizer_function_received, this);
#else
    if(this->type_optimizer_function == type_optimizer_function_received) { return; }
        
    // Deallocate old optimizer array.
    if(this->type_optimizer_function != MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NONE)
    { this->Deallocate__Parameter__Optimizer(); }
    // |END| Deallocate old optimizer array. |END|

    // Allocate optimizer array.
    this->type_optimizer_function = type_optimizer_function_received;

    if(this->Allocate__Parameter__Optimizer() == false)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate connections for optimizer function." NEW_LINE, __FUNCTION__);

        return;
    }
    // |END| Allocate optimizer array. |END|

    this->device__Clear_Train_Arrays();
#endif
}

__device__ T_ CUDA_Neural_Network::Warm_Restarts_Decay(void)
{
    T_ const tmp_learning_rate_decay(this->warm_restarts_minimum_learning_rate + 0.5_T * (this->warm_restarts_maximum_learning_rate - this->warm_restarts_minimum_learning_rate) * (1_T + cos(this->optimizer_time_step / this->warm_restarts_T_i * MyEA::Math::PI<T_>)));
    
    if(this->optimizer_time_step >= this->warm_restarts_T_i)
    {
        this->Clear_Optimizer();

        this->warm_restarts_T_i *= this->warm_restarts_multiplier;

        this->warm_restarts_maximum_learning_rate *= this->warm_restarts_decay_learning_rate;
    }

    return(tmp_learning_rate_decay);
}
    
// https://arxiv.org/pdf/1711.05101.pdf: Fixing Weight Decay Regularization in Adam
__device__ T_  CUDA_Neural_Network::Normalized_Weight_Decay(size_t const batch_size_received, size_t const training_size_received)
{ return(this->regularization__weight_decay * sqrt(batch_size_received / (training_size_received * this->epoch_time_step))); }

__device__ void CUDA_Neural_Network::Update_Parameter(size_t const batch_size_received, size_t const training_size_received)
{
    if(this->Get__Regularization__L1() != 0_T)
    { this->Update_Derivative_Weight__Regularization__L1(batch_size_received); }

    if(this->Get__Regularization__L2() != 0_T)
    { this->Update_Derivative_Weight__Regularization__L2(batch_size_received); }
    
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: 
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Merge_Derivatives_Parameters(); break;
        default:
            PRINT_FORMAT("%s: ERROR: Unknow type optimizer function (%u)." NEW_LINE,
                                    __FUNCTION__,
                                    this->type_optimizer_function);
                break;
    }

    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD: this->Update_Parameter__Gradient_Descent(batch_size_received, training_size_received, 0u, this->total_parameters); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: this->Update_Parameter__iRPROP_plus(0u, this->total_parameters); break;
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP: Update_Weight_QuickProp(this, this->Get__Number_Examples(), 0u, this->total_parameters); break;
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP: Update_Weight_SARProp(this, this->sarprop_epoch, 0u, this->total_parameters); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM: this->Update_Parameter__Adam(batch_size_received, training_size_received, 0u, this->total_parameters); break;
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SADAMAX: this->Update_Weight_AdaMax(0u, this->total_parameters); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: this->Update_Parameter__AMSGrad(batch_size_received, training_size_received, 0u, this->total_parameters); break;
        default:
            PRINT_FORMAT("%s: ERROR: Unknow type optimizer function (%u)." NEW_LINE,
                        __FUNCTION__,
                        this->type_optimizer_function);
                break;
    }

    if(this->Get__Regularization__Max_Norm_Constraints() != 0_T)
    { this->Update_Weight_Regularization__Max_Norm_Constraints(); }

    this->Transpose_Weights();
}

__global__ void kernel__CNeural_Network__Set__Accurancy_Variance(float const accurancy_variance_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Accurancy_Variance(accurancy_variance_received); }

__host__ __device__ void CUDA_Neural_Network::Set__Accurancy_Variance(float const accurancy_variance_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Accurancy_Variance <<< 1u, 1u >>> (accurancy_variance_received, this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    if(this->accuracy_variance == accurancy_variance_received) { return; }

    this->accuracy_variance = accurancy_variance_received;
#endif
}

__global__ void kernel__CNeural_Network__Set__Time_Delays(size_t const time_delays_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Time_Delays(time_delays_received); }

__host__ __device__ void CUDA_Neural_Network::Set__Time_Delays(size_t const time_delays_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Time_Delays <<< 1u, 1u >>> (time_delays_received, this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    if(this->number_time_delays == time_delays_received) { return; }

    this->number_time_delays = time_delays_received;
#endif
}
    
__device__ void CUDA_Neural_Network::Set__Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_accuracy_received, float const accurancy_received)
{
    switch(type_accuracy_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: this->accuracy_training = accurancy_received; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: this->accuracy_validating = accurancy_received; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: this->accuracy_testing = accurancy_received; break;
    }
}

__device__ void CUDA_Neural_Network::Set__Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, float const loss_received)
{
    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: this->loss_training = loss_received; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: this->loss_validating = loss_received; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: this->loss_testing = loss_received; break;
    }
}

__host__ __device__ float CUDA_Neural_Network::Get__Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received) const
{
#if defined(__CUDA_ARCH__) == false
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] TODO: Fix \"Get__Accuracy\" algorithm." NEW_LINE, __FUNCTION__);

    // TODO: Fix "Get__Accuracy".

    return(100_T);
#else
    T_ tmp_accurancy;
        
    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: tmp_accurancy = this->accuracy_training; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: tmp_accurancy = this->accuracy_validating; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: tmp_accurancy = this->accuracy_testing; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE: tmp_accurancy = this->number_accuracy_trial == 0_zu ? 0_T : this->ptr_array_accuracy_values[0u][0u] / static_cast<T_>(this->number_accuracy_trial) * 100_T; break;
    }

    return(tmp_accurancy);
#endif
}

__host__ __device__ float CUDA_Neural_Network::Get__Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, size_t const number_digits_received) const
{
#if defined(__CUDA_ARCH__) == false
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] TODO: Fix \"Get__Loss\" algorithm." NEW_LINE, __FUNCTION__);

    // TODO: Fix "Get__Loss".

    return(1_T);
#else
    float tmp_loss;

    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: tmp_loss = this->loss_training; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: tmp_loss = this->loss_validating; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: tmp_loss = this->loss_testing; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE:
            switch(this->type_loss_function)
            {
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_ME: tmp_loss = this->Get__ME(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L1: tmp_loss = this->Get__Loss_L1(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MAE: tmp_loss = this->Get__MAE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L2: tmp_loss = this->Get__Loss_L2(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MSE: tmp_loss = this->Get__MSE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_RMSE: tmp_loss = this->Get__RMSE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MAPE: tmp_loss = this->Get__MAPE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_SMAPE: tmp_loss = this->Get__SMAPE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MASE_SEASONAL: tmp_loss = this->Get__MASE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MASE_NON_SEASONAL: tmp_loss = this->Get__MASE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_CROSS_ENTROPY: tmp_loss = this->Get__ACE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_BIT: tmp_loss = this->Get__BITFAIL(); break;
                default: tmp_loss = 1_T;; break;
            }
                break;
        default: tmp_loss = 1_T;; break;
    }

    return(tmp_loss);
#endif
}
    
__device__ float CUDA_Neural_Network::Get__ME(void) const // https://en.wikipedia.org/wiki/Mean_absolute_error
{
    if(*this->ptr_array_number_loss != 0u)
    { return(*this->ptr_array_loss_values / static_cast<float>(*this->ptr_array_number_loss)); }
    else
    { return(1.0f); }
}
    
__device__ float CUDA_Neural_Network::Get__Loss_L1(void) const
{ return(*this->ptr_array_loss_values); }
    
__device__ float CUDA_Neural_Network::Get__MAE(void) const // https://en.wikipedia.org/wiki/Mean_absolute_error
{
    if(*this->ptr_array_number_loss != 0u)
    { return(*this->ptr_array_loss_values / static_cast<float>(*this->ptr_array_number_loss)); }
    else
    { return(1.0f); }
}
    
__device__ float CUDA_Neural_Network::Get__Loss_L2(void) const
{ return(*this->ptr_array_loss_values); }
    
__device__ float CUDA_Neural_Network::Get__MSE(void) const // https://en.wikipedia.org/wiki/Mean_squared_error
{
    if(*this->ptr_array_number_loss != 0u)
    { return(1.0f / static_cast<float>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values); }
    else
    { return(1.0f); }
}
    
__device__ float CUDA_Neural_Network::Get__RMSE(void) const // https://en.wikipedia.org/wiki/Root-mean-square_deviation
{
    if(*this->ptr_array_number_loss != 0u)
    { return(sqrt(1.0f / static_cast<float>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values)); }
    else
    { return(1.0f); }
}
    
__device__ float CUDA_Neural_Network::Get__MAPE(void) const // https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
{
    if(*this->ptr_array_number_loss != 0u)
    { return(1.0f / static_cast<float>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values); }
    else
    { return(1.0f); }
}
    
__device__ float CUDA_Neural_Network::Get__SMAPE(void) const // https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
{
    if(*this->ptr_array_number_loss != 0u)
    { return(1.0f / static_cast<float>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values); }
    else
    { return(1.0f); }
}

__device__ float CUDA_Neural_Network::Get__MASE(void) const // https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
{
    // Non seasonal time series
    //if(*this->ptr_array_number_loss != 0u
    //   &&
    //   this->mean_absolute_error_denominator != 0.0f
    //   &&
    //   *this->ptr_array_number_loss > 1u)
    //{ return(1.0f / this->number_recurrent_depth * (*this->ptr_array_loss_values / ((1.0f / static_cast<float>(this->number_recurrent_depth - 1_zu)) * this->mean_absolute_error_denominator))); }
    //else    { return(1.0f); }

    return(1.0f);
}

// TODO: optimize and check for the structure
__device__ float CUDA_Neural_Network::Get__ACE(void) const // https://en.wikipedia.org/wiki/Cross_entropy
{
    if(*this->ptr_array_number_loss != 0u)
    { return(*this->ptr_array_loss_values / static_cast<float>(*this->ptr_array_number_loss)); }
    else
    { return(std::numeric_limits<float>::max()); }
}
//__device__ float CUDA_Neural_Network::Get__CE(void) const // https://en.wikipedia.org/wiki/Cross_entropy
//{ return(*this->ptr_array_loss_values); }
    
__device__ float CUDA_Neural_Network::Get__BITFAIL(void) const // link
{ return(static_cast<float>(*this->ptr_array_number_bit_fail)); }

