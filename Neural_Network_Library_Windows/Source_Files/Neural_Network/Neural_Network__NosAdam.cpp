#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::Update_Parameters__NosAdam(size_t const batch_size_received,
                                                                               size_t const training_size_received,
                                                                               size_t const start_index_received,
                                                                               size_t const end_index_received)
{
    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        this->Update_Parameters__NosAdam__OpenMP(batch_size_received,
                                                                                training_size_received,
                                                                                start_index_received,
                                                                                end_index_received);
    }
    else
    {
        this->Update_Parameters__NosAdam__Loop(batch_size_received,
                                                                          training_size_received,
                                                                          start_index_received,
                                                                          end_index_received);
    }
}

void Neural_Network::Update_Parameters__NosAdam__Loop(size_t const batch_size_received,
                                                                                         size_t const training_size_received,
                                                                                         size_t const start_index_received,
                                                                                         size_t const end_index_received)
{
    size_t tmp_connection_index;
    
    T_ const tmp_learning_rate_scale(this->use_Warm_Restarts ? this->Warm_Restarts_Decay() / this->adam_learning_rate : 1_T);

    this->optimizer_time_step += 1_T;

    T_ const *const tmp_ptr_array_connections_mask_rergularization(this->ptr_array_mask_regularized_parameters),
                  tmp_learning_rate(tmp_learning_rate_scale * this->adam_learning_rate),
                  tmp_weight_decay(this->use_normalized_weight_decay ? this->Normalized_Weight_Decay(batch_size_received, training_size_received) : this->regularization__weight_decay),
                  tmp_beta1(this->adam_beta1),
                  tmp_beta2(this->adam_beta2),
                  tmp_beta2_at_power_gamma(pow(this->optimizer_time_step, -this->adam_gamma)),
                  tmp_beta2_at_time_t(this->adam_previous_beta2 / (this->adam_previous_beta2 + tmp_beta2_at_power_gamma)),
                  tmp_epsilon(this->adam_epsilon),
                  tmp_learning_rate_t(this->use_adam_bias_correction ? tmp_learning_rate * sqrt(1_T - pow(tmp_beta2, this->optimizer_time_step)) / (1_T - pow(tmp_beta1, this->optimizer_time_step)) : tmp_learning_rate);
    T_ *const tmp_ptr_array_partial_derivative(this->ptr_array_derivatives_parameters),
         *const tmp_ptr_array_parameters(this->ptr_array_parameters),
         *const tmp_ptr_array_previous_biased_first_moment(this->ptr_array_previous_biased_first_moment),
         *const tmp_ptr_array_previous_biased_second_moment(this->ptr_array_previous_biased_second_moment),
         tmp_partial_derivative,
         tmp_biased_first_moment,
         tmp_biased_second_moment;

    this->adam_previous_beta2 += tmp_beta2_at_power_gamma;

    if(tmp_weight_decay != 0_T)
    {
        for(tmp_connection_index = start_index_received; tmp_connection_index != end_index_received; ++tmp_connection_index)
        {
            tmp_partial_derivative = tmp_ptr_array_partial_derivative[tmp_connection_index];

            tmp_ptr_array_previous_biased_first_moment[tmp_connection_index] = tmp_biased_first_moment = tmp_beta1 * tmp_ptr_array_previous_biased_first_moment[tmp_connection_index] + (1_T - tmp_beta1) * tmp_partial_derivative;
            tmp_ptr_array_previous_biased_second_moment[tmp_connection_index] = tmp_biased_second_moment = tmp_beta2_at_time_t * tmp_ptr_array_previous_biased_second_moment[tmp_connection_index] + (1_T - tmp_beta2_at_time_t) * tmp_partial_derivative * tmp_partial_derivative;

            tmp_ptr_array_parameters[tmp_connection_index] -= tmp_learning_rate_t * tmp_biased_first_moment / (sqrt(tmp_biased_second_moment) + tmp_epsilon) + tmp_ptr_array_connections_mask_rergularization[tmp_connection_index] * tmp_weight_decay * tmp_ptr_array_parameters[tmp_connection_index];

            tmp_ptr_array_partial_derivative[tmp_connection_index] = 0_T;
        }
    }
    else
    {
        for(tmp_connection_index = start_index_received; tmp_connection_index != end_index_received; ++tmp_connection_index)
        {
            tmp_partial_derivative = tmp_ptr_array_partial_derivative[tmp_connection_index];

            tmp_ptr_array_previous_biased_first_moment[tmp_connection_index] = tmp_biased_first_moment = tmp_beta1 * tmp_ptr_array_previous_biased_first_moment[tmp_connection_index] + (1_T - tmp_beta1) * tmp_partial_derivative;
            tmp_ptr_array_previous_biased_second_moment[tmp_connection_index] = tmp_biased_second_moment = tmp_beta2_at_time_t * tmp_ptr_array_previous_biased_second_moment[tmp_connection_index] + (1_T - tmp_beta2_at_time_t) * tmp_partial_derivative * tmp_partial_derivative;

            tmp_ptr_array_parameters[tmp_connection_index] -= tmp_learning_rate_t * tmp_biased_first_moment / (sqrt(tmp_biased_second_moment) + tmp_epsilon);

            tmp_ptr_array_partial_derivative[tmp_connection_index] = 0_T;
        }
    }
}
    
void Neural_Network::Update_Parameters__NosAdam__OpenMP(size_t const batch_size_received,
                                                                                               size_t const training_size_received,
                                                                                               size_t const start_index_received,
                                                                                               size_t const end_index_received)
{
    int const tmp_end_index(static_cast<int>(end_index_received));
    
    T_ const tmp_learning_rate_scale(this->use_Warm_Restarts ? this->Warm_Restarts_Decay() / this->adam_learning_rate : 1_T);

    this->optimizer_time_step += 1_T;

    T_ const *const tmp_ptr_array_connections_mask_rergularization(this->ptr_array_mask_regularized_parameters),
                  tmp_learning_rate(tmp_learning_rate_scale * this->adam_learning_rate),
                  tmp_weight_decay(this->use_normalized_weight_decay ? this->Normalized_Weight_Decay(batch_size_received, training_size_received) : this->regularization__weight_decay),
                  tmp_beta1(this->adam_beta1),
                  tmp_beta2(this->adam_beta2),
                  tmp_beta2_at_power_gamma(pow(this->optimizer_time_step, -this->adam_gamma)),
                  tmp_beta2_at_time_t(this->adam_previous_beta2 / (this->adam_previous_beta2 + tmp_beta2_at_power_gamma)),
                  tmp_epsilon(this->adam_epsilon),
                  tmp_learning_rate_t(this->use_adam_bias_correction ? tmp_learning_rate * sqrt(1_T - pow(tmp_beta2, this->optimizer_time_step)) / (1_T - pow(tmp_beta1, this->optimizer_time_step)) : tmp_learning_rate);
    T_ *const tmp_ptr_array_partial_derivative(this->ptr_array_derivatives_parameters),
        *const tmp_ptr_array_parameters(this->ptr_array_parameters),
        *const tmp_ptr_array_previous_biased_first_moment(this->ptr_array_previous_biased_first_moment),
        *const tmp_ptr_array_previous_biased_second_moment(this->ptr_array_previous_biased_second_moment),
        tmp_partial_derivative(0),
        tmp_biased_first_moment(0),
        tmp_biased_second_moment(0);
    
    this->adam_previous_beta2 += tmp_beta2_at_power_gamma;

    if(tmp_weight_decay != 0_T)
    {
        #pragma omp parallel for schedule(static) private(tmp_partial_derivative, \
                                                                               tmp_biased_first_moment, \
                                                                               tmp_biased_second_moment)
        for(int tmp_connection_index = static_cast<int>(start_index_received); tmp_connection_index < tmp_end_index; ++tmp_connection_index)
        {
            tmp_partial_derivative = tmp_ptr_array_partial_derivative[tmp_connection_index];

            tmp_ptr_array_previous_biased_first_moment[tmp_connection_index] = tmp_biased_first_moment = tmp_beta1 * tmp_ptr_array_previous_biased_first_moment[tmp_connection_index] + (1_T - tmp_beta1) * tmp_partial_derivative;
            tmp_ptr_array_previous_biased_second_moment[tmp_connection_index] = tmp_biased_second_moment = tmp_beta2_at_time_t * tmp_ptr_array_previous_biased_second_moment[tmp_connection_index] + (1_T - tmp_beta2_at_time_t) * tmp_partial_derivative * tmp_partial_derivative;

            tmp_ptr_array_parameters[tmp_connection_index] -= tmp_learning_rate_t * tmp_biased_first_moment / (sqrt(tmp_biased_second_moment) + tmp_epsilon) + tmp_ptr_array_connections_mask_rergularization[tmp_connection_index] * tmp_weight_decay * tmp_ptr_array_parameters[tmp_connection_index];

            tmp_ptr_array_partial_derivative[tmp_connection_index] = 0_T;
        }
    }
    else
    {
        #pragma omp parallel for schedule(static) private(tmp_partial_derivative, \
                                                                               tmp_biased_first_moment, \
                                                                               tmp_biased_second_moment)
        for(int tmp_connection_index = static_cast<int>(start_index_received); tmp_connection_index < tmp_end_index; ++tmp_connection_index)
        {
            tmp_partial_derivative = tmp_ptr_array_partial_derivative[tmp_connection_index];

            tmp_ptr_array_previous_biased_first_moment[tmp_connection_index] = tmp_biased_first_moment = tmp_beta1 * tmp_ptr_array_previous_biased_first_moment[tmp_connection_index] + (1_T - tmp_beta1) * tmp_partial_derivative;
            tmp_ptr_array_previous_biased_second_moment[tmp_connection_index] = tmp_biased_second_moment = tmp_beta2_at_time_t * tmp_ptr_array_previous_biased_second_moment[tmp_connection_index] + (1_T - tmp_beta2_at_time_t) * tmp_partial_derivative * tmp_partial_derivative;

            tmp_ptr_array_parameters[tmp_connection_index] -= tmp_learning_rate_t * tmp_biased_first_moment / (sqrt(tmp_biased_second_moment) + tmp_epsilon);

            tmp_ptr_array_partial_derivative[tmp_connection_index] = 0_T;
        }
    }
}
