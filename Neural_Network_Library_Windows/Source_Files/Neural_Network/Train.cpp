#include "stdafx.hpp"

#include <Files/File.hpp>

#include <omp.h>

    void Neural_Network::Set__Bit_Fail_Limit(T_ const bit_fail_limit_received)
    {
        if(this->bit_fail_limit == bit_fail_limit_received) { return; }

        this->bit_fail_limit = bit_fail_limit_received;

    #if defined(COMPILE_CUDA)
        if(this->is_device_initialized)
        { this->ptr_device_Neural_Network->Set__Bit_Fail_Limit(bit_fail_limit_received); }
    #endif
    }
    
    void Neural_Network::Set__Maximum_Allowable_Memory(size_t const maximum_allowable_memory_bytes_received)
    { this->maximum_allowable_memory_bytes = maximum_allowable_memory_bytes_received; }
    
    void Neural_Network::Set__Loss_Function(enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const loss_function_received)
    {
        if(this->type_loss_function == loss_function_received) { return; }

        this->type_loss_function = loss_function_received;

    #if defined(COMPILE_CUDA)
        if(this->is_device_initialized)
        { this->ptr_device_Neural_Network->Set__Loss_Function(loss_function_received); }
    #endif
    }
    
    void Neural_Network::Set__Accuracy_Function(enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS const type_accuracy_function_received)
    {
        if(this->type_accuracy_function == type_accuracy_function_received) { return; }

        this->type_accuracy_function = type_accuracy_function_received;

    #if defined(COMPILE_CUDA)
        if(this->is_device_initialized)
        { this->ptr_device_Neural_Network->Set__Accuracy_Function(type_accuracy_function_received); }
    #endif
    }

    void Neural_Network::Set__Optimizer_Function(enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS const optimizer_function_received)
    {
        if(this->type_optimizer_function == optimizer_function_received) { return; }

        // Deallocate old optimizer array.
        if(this->type_optimizer_function != MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NONE)
        { this->Deallocate__Parameter__Optimizer(); }
        // |END| Deallocate old optimizer array. |END|
        
        // Store type optimizer function.
        this->type_optimizer_function = optimizer_function_received;
        
        // Allocate optimizer array(s).
        if(this->Allocate__Parameter__Optimizer() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate connections for optimizer function." NEW_LINE,
                         MyEA::Time::Date_Time_Now().c_str(),
                         __FUNCTION__);

            return;
        }
        // |END| Allocate optimizer array(s). |END|

        // Clear all derivative array(s).
        this->Clear_Training_Arrays();

    #if defined(COMPILE_CUDA)
        if(this->is_device_initialized)
        { this->ptr_device_Neural_Network->Set__Optimizer_Function(optimizer_function_received); }
    #endif
    }

    void Neural_Network::Clear_Training_Arrays(void)
    {
    #if defined(COMPILE_CUDA)
        if(this->use_CUDA && this->is_device_initialized)
        { this->Clear_Training_Arrays__CUDA(); }
        else
    #endif
        {
            if(this->ptr_array_derivatives_parameters == nullptr)
            {
                if((this->ptr_array_derivatives_parameters = new T_[this->number_threads * this->total_parameters_allocated]) == nullptr)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             this->number_threads * this->total_parameters_allocated * sizeof(T_),
                                             __LINE__);

                    return;
                }

                if(this->Use__Normalization())
                { this->Reset__Derivative_Parameter__Normalized_Unit(); }
            }
            MEMSET(this->ptr_array_derivatives_parameters,
                        0,
                        this->number_threads * this->total_parameters_allocated * sizeof(T_));

            this->Clear_Optimizer();

            this->warm_restarts_maximum_learning_rate = this->warm_restarts_initial_maximum_learning_rate;
            this->warm_restarts_T_i = this->warm_restarts_initial_T_i;
        }
    }
    
    void Neural_Network::Clear_Optimizer(void)
    {
        size_t i;

        switch(this->type_optimizer_function)
        {
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NONE: break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
                if(this->learning_momentum != 0_T && this->ptr_array_previous_delta_parameters != nullptr)
                {
                    MEMSET(this->ptr_array_previous_delta_parameters,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }
                    break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
                // Previous train slopes.
                if(this->ptr_array_previous_derivatives_parameters != nullptr)
                {
                    MEMSET(this->ptr_array_previous_derivatives_parameters,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }
                // |END| Previous train slopes. |END|

                // Previous steps.
                if(this->ptr_array_previous_steps != nullptr)
                {
                    for(i = 0u; i != this->total_parameters_allocated; ++i)
                    {
                        this->ptr_array_previous_steps[i] = this->rprop_delta_zero;
                    }
                }
                // |END| Previous steps. |END|
                    break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
                this->loss_rprop = (std::numeric_limits<ST_>::max)();
                this->previous_loss_rprop = (std::numeric_limits<ST_>::max)();

                // Previous train slopes.
                if(this->ptr_array_previous_derivatives_parameters != nullptr)
                {
                    MEMSET(this->ptr_array_previous_derivatives_parameters,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }
                // |END| Previous train slopes. |END|

                // Previous steps.
                if(this->ptr_array_previous_steps != nullptr)
                {
                    for(i = 0u; i != this->total_parameters_allocated; ++i)
                    {
                        this->ptr_array_previous_steps[i] = this->rprop_delta_zero;
                    }
                }
                // |END| Previous steps. |END|

                // Previous delta weights.
                if(this->ptr_array_previous_delta_parameters != nullptr)
                {
                    MEMSET(this->ptr_array_previous_delta_parameters,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }
                // |END| Previous delta weights. |END|
                    break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP: break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP: break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad:
                if(this->ptr_array_previous_biased_first_moment != nullptr)
                {
                    MEMSET(this->ptr_array_previous_biased_first_moment,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }

                if(this->ptr_array_previous_biased_second_moment != nullptr)
                {
                    MEMSET(this->ptr_array_previous_biased_second_moment,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }

                if(this->ptr_array_previous_biased_second_moment_hat != nullptr)
                {
                    MEMSET(this->ptr_array_previous_biased_second_moment_hat,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }
                    break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
                if(this->ptr_array_previous_biased_first_moment != nullptr)
                {
                    MEMSET(this->ptr_array_previous_biased_first_moment,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }

                if(this->ptr_array_previous_biased_second_moment != nullptr)
                {
                    MEMSET(this->ptr_array_previous_biased_second_moment,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }
                    break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM:
                if(this->ptr_array_previous_biased_first_moment != nullptr)
                {
                    MEMSET(this->ptr_array_previous_biased_first_moment,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }

                if(this->ptr_array_previous_biased_second_moment != nullptr)
                {
                    MEMSET(this->ptr_array_previous_biased_second_moment,
                                   0,
                                   this->total_parameters_allocated * sizeof(T_));
                }

                this->adam_previous_beta2 = 0_T;
                    break;
            default:
                PRINT_FORMAT("%s: ERROR: Can not reset the optimizer parameters with (%u | %s) as the current optimizer." NEW_LINE,
                                         __FUNCTION__,
                                         this->type_optimizer_function,
                                         MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
                    break;
        }

        this->optimizer_time_step = 0_T;
        this->epoch_time_step = 1_T;
    }

    T_ Neural_Network::Warm_Restarts_Decay(void)
    {
        T_ const tmp_learning_rate_decay(this->warm_restarts_minimum_learning_rate + 0.5_T * (this->warm_restarts_maximum_learning_rate - this->warm_restarts_minimum_learning_rate) * (1_T + cos(this->optimizer_time_step / this->warm_restarts_T_i * MyEA::Math::PI<T_>)));
        
        if(this->optimizer_time_step >= this->warm_restarts_T_i)
        {
            this->Clear_Optimizer();

            this->warm_restarts_T_i *= this->warm_restarts_multiplier;

            this->warm_restarts_maximum_learning_rate *= this->warm_restarts_decay_learning_rate;

            this->warm_restarts_maximum_learning_rate = MyEA::Math::Maximum<T_>(this->warm_restarts_maximum_learning_rate, this->warm_restarts_minimum_learning_rate);
        }

        return(tmp_learning_rate_decay);
    }

    // https://arxiv.org/pdf/1711.05101.pdf: Fixing Weight Decay Regularization in Adam
    T_  Neural_Network::Normalized_Weight_Decay(size_t const batch_size_received, size_t const training_size_received)
    { return(this->regularization__weight_decay * sqrt(static_cast<T_>(batch_size_received) / (static_cast<T_>(training_size_received) * this->epoch_time_step))); }

    void Update_Weight_Batch(class Neural_Network *ptr_Neural_Network_received,
                                            size_t const number_examples_received,
                                            size_t const first_weight_received,
                                            size_t const past_end_received)
    {
        T_ *const tmp_ptr_array_partial_derivative(ptr_Neural_Network_received->ptr_array_derivatives_parameters),
             *const tmp_ptr_array_parameters(ptr_Neural_Network_received->ptr_array_parameters);

        T_ const tmp_epsilon(ptr_Neural_Network_received->learning_rate / static_cast<T_>(number_examples_received));
    
        for(size_t i(first_weight_received); i != past_end_received; ++i)
        {
            //tmp_ptr_array_parameters[i] += tmp_ptr_array_partial_derivative[i] * tmp_epsilon; // Gradient acsent
            tmp_ptr_array_parameters[i] -= tmp_ptr_array_partial_derivative[i] * tmp_epsilon; // Gradient descent
            tmp_ptr_array_partial_derivative[i] = 0_T;
        }
    }

    void Update_Weight_QuickProp(class Neural_Network *ptr_Neural_Network_received,
                                                           size_t const number_examples_received,
                                                           size_t const first_weight_received,
                                                           size_t const past_end_received)
    {
        T_ const tmp_epsilon(ptr_Neural_Network_received->learning_rate / static_cast<T_>(number_examples_received)),
                      tmp_decay(ptr_Neural_Network_received->quickprop_decay), // -0.0001;
                      tmp_mu(ptr_Neural_Network_received->quickprop_mu), // 1.75;
                      tmp_shrink_factor(tmp_mu / (1_T + tmp_mu));

        T_ *tmp_ptr_array_partial_derivative(ptr_Neural_Network_received->ptr_array_derivatives_parameters),
             *tmp_ptr_array_parameters(ptr_Neural_Network_received->ptr_array_parameters),
             *tmp_ptr_prev_steps(ptr_Neural_Network_received->ptr_array_previous_steps),
             *tmp_ptr_prev_trainer_slopes(ptr_Neural_Network_received->ptr_array_previous_derivatives_parameters);
    
        for(size_t i(first_weight_received); i != past_end_received; ++i)
        {
            T_ const tmp_prev_step(tmp_ptr_prev_steps[i]),
                                       tmp_prev_slope(tmp_ptr_prev_trainer_slopes[i]);
            T_ tmp_weight(tmp_ptr_array_parameters[i]),
                             tmp_next_step(0.0);
            T_ const tmp_slope(tmp_ptr_array_partial_derivative[i] + tmp_decay * tmp_weight);
        
            if(tmp_prev_step > 0.001)
            {
                if(tmp_slope > 0.0) { tmp_next_step += tmp_epsilon * tmp_slope; }

                if(tmp_slope > (tmp_shrink_factor * tmp_prev_slope)) { tmp_next_step += tmp_mu * tmp_prev_step; }
                else { tmp_next_step += tmp_prev_step * tmp_slope / (tmp_prev_slope - tmp_slope); }
            }
            else if(tmp_prev_step < -0.001)
            {
                if(tmp_slope < 0.0) { tmp_next_step += tmp_epsilon * tmp_slope; }

                if(tmp_slope < (tmp_shrink_factor * tmp_prev_slope)) { tmp_next_step += tmp_mu * tmp_prev_step; }
                else { tmp_next_step += tmp_prev_step * tmp_slope / (tmp_prev_slope - tmp_slope); }
            }
            else { tmp_next_step += tmp_epsilon * tmp_slope; }

            tmp_ptr_prev_steps[i] = tmp_next_step;

            tmp_weight += tmp_next_step;

            if(tmp_weight > 1500) { tmp_ptr_array_parameters[i] = 1500; }
            else if(tmp_weight < -1500) { tmp_ptr_array_parameters[i] = -1500; }
            else { tmp_ptr_array_parameters[i] = tmp_weight; }

            tmp_ptr_prev_trainer_slopes[i] = tmp_slope;
            tmp_ptr_array_partial_derivative[i] = 0.0;
        }
    }
    
    void Update_Weight_SARProp(class Neural_Network *ptr_Neural_Network_received,
                                                 size_t const epoch_received,
                                                 size_t const first_weight_received,
                                                 size_t const past_end_received)
    {
        T_ *tmp_ptr_array_partial_derivative(ptr_Neural_Network_received->ptr_array_derivatives_parameters),
            *tmp_ptr_array_parameters(ptr_Neural_Network_received->ptr_array_parameters),
            *tmp_ptr_prev_steps(ptr_Neural_Network_received->ptr_array_previous_steps),
            *tmp_ptr_prev_trainer_slopes(ptr_Neural_Network_received->ptr_array_previous_derivatives_parameters);

        T_ const tmp_increase_factor(ptr_Neural_Network_received->rprop_increase_factor), // 1.2;
                      tmp_decrease_factor(ptr_Neural_Network_received->rprop_decrease_factor), // 0.5;
                      tmp_delta_min(0.000001f),
                      tmp_delta_max(ptr_Neural_Network_received->rprop_delta_max), // 50.0;
                      tmp_weight_decay_shift(ptr_Neural_Network_received->sarprop_weight_decay_shift), // ld 0.01 = -6.644
                      tmp_step_error_threshold_factor(ptr_Neural_Network_received->sarprop_step_error_threshold_factor), // 0.1
                      tmp_step_error_shift(ptr_Neural_Network_received->sarprop_step_error_shift), // ld 3 = 1.585
                      tmp_T(ptr_Neural_Network_received->sarprop_temperature),
                      tmp_MSE(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE)),
                      tmp_RMSE(sqrt(tmp_MSE)); // TODO: MSE?

        for(size_t i(first_weight_received); i != past_end_received; ++i)
        {
            T_ tmp_prev_step(MyEA::Math::Maximum<T_>(tmp_ptr_prev_steps[i], (T_) 0.000001)), // prev_step may not be zero because then the training will stop.
                 tmp_slope(-tmp_ptr_array_partial_derivative[i] - tmp_ptr_array_parameters[i] * (T_)fann_exp2(-tmp_T * static_cast<T_>(epoch_received) + tmp_weight_decay_shift)),
                 tmp_prev_slope(tmp_ptr_prev_trainer_slopes[i]),
                 tmp_same_sign(tmp_prev_slope * tmp_slope),
                 tmp_next_step(0);

            if(tmp_same_sign > 0.0)
            {
                tmp_next_step = MyEA::Math::Minimum<T_>(tmp_prev_step * tmp_increase_factor, tmp_delta_max);

                if(tmp_slope < 0.0) { tmp_ptr_array_parameters[i] += tmp_next_step; }
                else { tmp_ptr_array_parameters[i] -= tmp_next_step; }
            }
            else if(tmp_same_sign < 0.0)
            {
                if(tmp_prev_step < tmp_step_error_threshold_factor * tmp_MSE) { tmp_next_step = tmp_prev_step * tmp_decrease_factor + static_cast<T_>(rand()) / static_cast<T_>(RAND_MAX) * tmp_RMSE * static_cast<T_>(fann_exp2(-tmp_T * static_cast<T_>(epoch_received) + tmp_step_error_shift)); }
                else { tmp_next_step = MyEA::Math::Maximum<T_>(tmp_prev_step * tmp_decrease_factor, tmp_delta_min); }

                tmp_slope = 0.0;
            }
            else
            {
                if(tmp_slope < 0.0) { tmp_ptr_array_parameters[i] += tmp_prev_step; }
                else { tmp_ptr_array_parameters[i] -= tmp_prev_step; }
            }
        
            tmp_ptr_prev_steps[i] = tmp_next_step;
            tmp_ptr_prev_trainer_slopes[i] = tmp_slope;
            tmp_ptr_array_partial_derivative[i] = 0.0;
        }

        ++ptr_Neural_Network_received->sarprop_epoch;
    }

bool Neural_Network::Set__Layer_Activation_Function(size_t const index_layer_received, enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received)
{
    if(type_activation_function_received == MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_NONE
      ||
      type_activation_function_received == MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: Type activation function can not be set to (%u | %s). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_activation_function_received,
                                 MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[type_activation_function_received].c_str(),
                                 __LINE__);

        return(false);
    }
    else if(this->ptr_array_layers == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_array_layers\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(index_layer_received >= this->total_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer index (%zu) overflow the number of layers in the network (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 index_layer_received,
                                 this->total_layers,
                                 __LINE__);

        return(false);
    }

    return(this->Set__Layer_Activation_Function(this->ptr_array_layers + index_layer_received, type_activation_function_received));
}

enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION Neural_Network::Activation_Function__To__Class_Activation_Function(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received) const
{
    enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION tmp_class_activation_function;

    switch(type_activation_function_received)
    {
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_COSINE_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELLIOT_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR_PIECE_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SINE_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH_STEPWISE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_THRESHOLD_SYMMETRIC: tmp_class_activation_function = MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC; break;
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_COSINE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELU:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELLIOT:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN_STEPWISE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ISRU:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ISRLU:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR_PIECE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SINE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID_STEPWISE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_THRESHOLD: tmp_class_activation_function = MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC; break;
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LEAKY_RELU:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_PARAMETRIC_RELU:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_RELU: tmp_class_activation_function = MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER; break;
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SELU: tmp_class_activation_function = MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION; break;
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SOFTMAX: tmp_class_activation_function = MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX; break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Activation function type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     type_activation_function_received,
                                     MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[type_activation_function_received].c_str(),
                                     __LINE__);
                tmp_class_activation_function = MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_NONE;
    }
    
    return(tmp_class_activation_function);
}

bool Neural_Network::Set__Layer_Activation_Function(struct Layer *const ptr_layer_it_received, enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received)
{
    if(type_activation_function_received == MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_NONE
      ||
      type_activation_function_received >= MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: Type activation function can not be set to (%u | %s). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_activation_function_received,
                                 MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[type_activation_function_received].c_str(),
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_it_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_it_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
             ||
             ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING
             ||
             ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
    {
        PRINT_FORMAT("%s: %s: WARNING: Type layer (%u | %s) is useless in this function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_layer_it_received->type_layer,
                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str(),
                                 __LINE__);

        return(true);
    }

    // Regularization on recurrent connection(s) (Independently RNN).
    bool const tmp_layer_use_regularization_constraint_recurrent_weight_default(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT
                                                                                                                    &&
                                                                                                                    this->Check__Use__Regularization__Constraint_Recurrent_Weight__Default(ptr_layer_it_received));

    switch((ptr_layer_it_received->type_activation = this->Activation_Function__To__Class_Activation_Function(type_activation_function_received)))
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC: break;
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX:
            if(ptr_layer_it_received != this->ptr_last_layer - 1) // If is not the output layer
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not use a softmax layer in a hidden layer. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            else if(*ptr_layer_it_received->ptr_number_outputs == 1u)
            {
                PRINT_FORMAT("%s: %s: ERROR: Softmax activation functions is only for multi class. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer activation type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_activation,
                                     MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str(),
                                     __LINE__);
                return(false);
    }

    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_SHORTCUT:
            if(this->Set__Layer_Activation_Function__AF(ptr_layer_it_received, type_activation_function_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function__AF(ptr, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         type_activation_function_received,
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            if(this->Set__Layer_Activation_Function__AF_Ind_Recurrent(ptr_layer_it_received, type_activation_function_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function__AF_Ind_Recurrent(ptr, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         type_activation_function_received,
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            if(this->Set__Layer_Activation_Function__LSTM(ptr_layer_it_received, type_activation_function_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function__LSTM(ptr, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         type_activation_function_received,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str(),
                                     __LINE__);
            return(false);
    }

    // Regularization on recurrent connection(s) (Independently RNN).
    if(tmp_layer_use_regularization_constraint_recurrent_weight_default
      &&
      this->Set__Regularization__Constraint_Recurrent_Weight__Default(ptr_layer_it_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight__Default(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }

    return(true);
}

bool Neural_Network::Set__Layer_Activation_Function__AF(struct Layer *const ptr_layer_it_received, enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received)
{
    struct AF_unit const *const tmp_ptr_last_AF_unit(ptr_layer_it_received->ptr_last_AF_unit);
    struct AF_unit *tmp_ptr_AF_unit_it(ptr_layer_it_received->ptr_array_AF_units);

    for(; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it) { *tmp_ptr_AF_unit_it->ptr_type_activation_function = type_activation_function_received; }

    return(true);
}

bool Neural_Network::Set__Layer_Activation_Function__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received, enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit);
    struct AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);

    for(; tmp_ptr_AF_Ind_recurrent_unit_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_AF_Ind_recurrent_unit_it) { *tmp_ptr_AF_Ind_recurrent_unit_it->ptr_type_activation_function = type_activation_function_received; }

    return(true);
}

bool Neural_Network::Set__Layer_Activation_Function__LSTM(struct Layer *const ptr_layer_it_received, enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received)
{
    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);

    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it) { tmp_ptr_block_unit_it->activation_function_io = type_activation_function_received; }

    return(true);
}

bool Neural_Network::Set__Layer_Activation_Steepness(size_t const index_layer_received, T_ const activation_steepness_received)
{
    if(activation_steepness_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Activation steepness (%f) can not be less than zero. At line %d." NEW_LINE,
                                MyEA::Time::Date_Time_Now().c_str(),
                                __FUNCTION__,
                                Cast_T(activation_steepness_received),
                                __LINE__);

        return(false);
    }
    else if(activation_steepness_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Activation steepness (%f) can not be greater than one. At line %d." NEW_LINE,
                                MyEA::Time::Date_Time_Now().c_str(),
                                __FUNCTION__,
                                Cast_T(activation_steepness_received),
                                __LINE__);

        return(false);
    }
    else if(this->ptr_array_layers == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_array_layers\" is a nullptr. At line %d." NEW_LINE,
                                MyEA::Time::Date_Time_Now().c_str(),
                                __FUNCTION__,
                                __LINE__);

        return(false);
    }
    else if(index_layer_received >= this->total_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer index (%zu) overflow the number of layers in the network (%zu). At line %d." NEW_LINE,
                                MyEA::Time::Date_Time_Now().c_str(),
                                __FUNCTION__,
                                index_layer_received,
                                this->total_layers,
                                __LINE__);

        return(false);
    }

    return(this->Set__Layer_Activation_Steepness(this->ptr_array_layers + index_layer_received, activation_steepness_received));
}

bool Neural_Network::Set__Layer_Activation_Steepness(struct Layer *const ptr_layer_it_received, T_ const activation_steepness_received)
{
    if(activation_steepness_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Activation steepness (%f) can not be less than zero. At line %d." NEW_LINE,
                                MyEA::Time::Date_Time_Now().c_str(),
                                __FUNCTION__,
                                Cast_T(activation_steepness_received),
                                __LINE__);

        return(false);
    }
    else if(activation_steepness_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Activation steepness (%f) can not be greater than one. At line %d." NEW_LINE,
                                MyEA::Time::Date_Time_Now().c_str(),
                                __FUNCTION__,
                                Cast_T(activation_steepness_received),
                                __LINE__);

        return(false);
    }
    else if(ptr_layer_it_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_it_received\" is a nullptr. At line %d." NEW_LINE,
                                MyEA::Time::Date_Time_Now().c_str(),
                                __FUNCTION__,
                                __LINE__);

        return(false);
    }
    else if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
             ||
             ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING
             ||
             ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM
             ||
             ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
    {
        PRINT_FORMAT("%s: %s: WARNING: Type layer (%u | %s) is useless in this function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_layer_it_received->type_layer,
                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str(),
                                 __LINE__);

        return(true);
    }
    
    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_SHORTCUT:
            if(this->Set__Layer_Activation_Steepness__AF(ptr_layer_it_received, activation_steepness_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Steepness__AF(ptr, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         Cast_T(activation_steepness_received),
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            if(this->Set__Layer_Activation_Steepness__AF_Ind_Recurrent(ptr_layer_it_received, activation_steepness_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Steepness__AF_Ind_Recurrent(ptr, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         Cast_T(activation_steepness_received),
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str(),
                                     __LINE__);
                break;
    }

    return(true);
}

bool Neural_Network::Set__Layer_Activation_Steepness__AF(struct Layer *const ptr_layer_it_received, T_ const activation_steepness_received)
{
    struct AF_unit const *const tmp_ptr_last_AF_unit(ptr_layer_it_received->ptr_last_AF_unit);
    struct AF_unit *tmp_ptr_AF_unit_it(ptr_layer_it_received->ptr_array_AF_units);

    for(; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it) { *tmp_ptr_AF_unit_it->ptr_activation_steepness = activation_steepness_received; }

    return(true);
}

bool Neural_Network::Set__Layer_Activation_Steepness__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received, T_ const activation_steepness_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit);
    struct AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);

    for(; tmp_ptr_AF_Ind_recurrent_unit_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_AF_Ind_recurrent_unit_it) { *tmp_ptr_AF_Ind_recurrent_unit_it->ptr_activation_steepness = activation_steepness_received; }

    return(true);
}

// Ctrl-x then ctrl-v then ctrl-shift-b