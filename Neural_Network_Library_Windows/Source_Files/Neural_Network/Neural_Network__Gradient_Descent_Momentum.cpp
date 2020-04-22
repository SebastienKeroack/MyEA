/* Copyright 2020 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::Update_Parameter__Gradient_Descent_Momentum__Loop(size_t const batch_size_received,
                                                                                                                    size_t const training_size_received,
                                                                                                                    size_t const start_index_received,
                                                                                                                    size_t const end_index_received)
{
    size_t tmp_connection_index;

    T_ const tmp_learning_rate_scale(this->use_Warm_Restarts ? this->Warm_Restarts_Decay() / this->learning_rate : 1_T);

    this->optimizer_time_step += 1_T;

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
        for(tmp_connection_index = start_index_received; tmp_connection_index != end_index_received; ++tmp_connection_index)
        {
            tmp_ptr_array_previous_delta_weights[tmp_connection_index] = tmp_delta_weigth = tmp_learning_rate * tmp_ptr_array_partial_derivative[tmp_connection_index] + tmp_learning_momentum * tmp_ptr_array_previous_delta_weights[tmp_connection_index];

            tmp_ptr_array_parameters[tmp_connection_index] -= tmp_delta_weigth + tmp_ptr_array_connections_mask_rergularization[tmp_connection_index] * tmp_weight_decay * tmp_ptr_array_parameters[tmp_connection_index];

            tmp_ptr_array_partial_derivative[tmp_connection_index] = 0_T;
        }
    }
    else
    {
        for(tmp_connection_index = start_index_received; tmp_connection_index != end_index_received; ++tmp_connection_index)
        {
            tmp_ptr_array_previous_delta_weights[tmp_connection_index] = tmp_delta_weigth = tmp_learning_rate * tmp_ptr_array_partial_derivative[tmp_connection_index] + tmp_learning_momentum * tmp_ptr_array_previous_delta_weights[tmp_connection_index];

            tmp_ptr_array_parameters[tmp_connection_index] -= tmp_delta_weigth; // Gradient descent

            tmp_ptr_array_partial_derivative[tmp_connection_index] = 0_T;
        }
    }
}
    
void Neural_Network::Update_Parameter__Gradient_Descent_Momentum__OpenMP(size_t const batch_size_received,
                                                                                                                         size_t const training_size_received,
                                                                                                                         size_t const start_index_received,
                                                                                                                         size_t const end_index_received)
{
    int const tmp_end_index__int(static_cast<int>(end_index_received));
    
    T_ const tmp_learning_rate_scale(this->use_Warm_Restarts ? this->Warm_Restarts_Decay() / this->learning_rate : 1_T);

    this->optimizer_time_step += 1_T;

    T_ const *const tmp_ptr_array_connections_mask_rergularization(this->ptr_array_mask_regularized_parameters),
                  tmp_learning_rate(tmp_learning_rate_scale * this->learning_rate),
                  tmp_learning_momentum(this->learning_momentum),
                  tmp_weight_decay(tmp_learning_rate_scale * (this->use_normalized_weight_decay ? this->Normalized_Weight_Decay(batch_size_received, training_size_received) : this->regularization__weight_decay));
    T_ *const tmp_ptr_array_partial_derivative(this->ptr_array_derivatives_parameters),
        *const tmp_ptr_array_parameters(this->ptr_array_parameters),
        *const tmp_ptr_array_previous_delta_weights(this->ptr_array_previous_delta_parameters),
        tmp_delta_weigth(0);

    if(tmp_weight_decay != 0_T)
    {
        #pragma omp parallel for schedule(static) private(tmp_delta_weigth)
        for(int tmp_connection_index = static_cast<int>(start_index_received); tmp_connection_index < tmp_end_index__int; ++tmp_connection_index)
        {
            tmp_ptr_array_previous_delta_weights[tmp_connection_index] = tmp_delta_weigth = tmp_learning_rate * tmp_ptr_array_partial_derivative[tmp_connection_index] + tmp_learning_momentum * tmp_ptr_array_previous_delta_weights[tmp_connection_index];

            tmp_ptr_array_parameters[tmp_connection_index] -= tmp_delta_weigth + tmp_ptr_array_connections_mask_rergularization[tmp_connection_index] * tmp_weight_decay * tmp_ptr_array_parameters[tmp_connection_index];

            tmp_ptr_array_partial_derivative[tmp_connection_index] = 0_T;
        }
    }
    else
    {
        #pragma omp parallel for schedule(static) private(tmp_delta_weigth)
        for(int tmp_connection_index = static_cast<int>(start_index_received); tmp_connection_index < tmp_end_index__int; ++tmp_connection_index)
        {
            tmp_ptr_array_previous_delta_weights[tmp_connection_index] = tmp_delta_weigth = tmp_learning_rate * tmp_ptr_array_partial_derivative[tmp_connection_index] + tmp_learning_momentum * tmp_ptr_array_previous_delta_weights[tmp_connection_index];

            tmp_ptr_array_parameters[tmp_connection_index] -= tmp_delta_weigth;

            tmp_ptr_array_partial_derivative[tmp_connection_index] = 0_T;
        }
    }
}
