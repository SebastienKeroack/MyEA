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

void Neural_Network::FF__Compute__Accuracy__R__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received)
{
    struct Layer const *const tmp_ptr_output_layer(this->Get__Output_Layer());

    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int,
        tmp_thread_index__int;
    
    size_t const tmp_output_size(static_cast<size_t>(tmp_ptr_output_layer->ptr_last_AF_unit - tmp_ptr_output_layer->ptr_array_AF_units));
    size_t tmp_output_data_index;

    T_ const tmp_desired_mean(this->ptr_array_accuracy_values[0u][0u]),
                  tmp_predicted_mean(this->ptr_array_accuracy_values[1u][0u]),
                  *tmp_ptr_array_desireds_outputs;
    T_ *const tmp_ptr_array_numerator(this->ptr_array_accuracy_values[2u]),
        *const tmp_ptr_array_denominator_desired(this->ptr_array_accuracy_values[3u]),
        *const tmp_ptr_array_denominator_predicted(this->ptr_array_accuracy_values[4u]),
        tmp_desired_mean_difference,
        tmp_predicted_mean_difference;

    struct AF_unit const *const tmp_ptr_last_AF_unit(tmp_ptr_output_layer->ptr_last_AF_unit); 
    struct AF_unit *tmp_ptr_AF_unit_it;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        // Desired output(s): If the NN is in pre-training mode use depending on the level, the dataset array or the hidden-input layer to the AE. Else take the dataset array.
        tmp_ptr_array_desireds_outputs = this->pre_training_level <= 1_zu ? ptr_array_desireds_outputs_received[tmp_example_index__int] : this->Get__Outputs(this->ptr_array_layers + (this->pre_training_level - 1_zu), static_cast<size_t>(tmp_example_index__int));

        tmp_output_data_index = static_cast<size_t>(tmp_example_index__int) * tmp_output_size;

        tmp_ptr_AF_unit_it = tmp_ptr_output_layer->ptr_array_AF_units;

        for(; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it,
                                                                            ++tmp_ptr_array_desireds_outputs)
        {
            tmp_desired_mean_difference = *tmp_ptr_array_desireds_outputs - tmp_desired_mean;

            tmp_predicted_mean_difference = tmp_ptr_AF_unit_it->ptr_array_values[tmp_output_data_index] - tmp_predicted_mean;
            
            tmp_ptr_array_numerator[tmp_thread_index__int] += tmp_desired_mean_difference * tmp_predicted_mean_difference;
            
            tmp_ptr_array_denominator_desired[tmp_thread_index__int] += tmp_desired_mean_difference * tmp_desired_mean_difference;
            tmp_ptr_array_denominator_predicted[tmp_thread_index__int] += tmp_predicted_mean_difference * tmp_predicted_mean_difference;
        }
    }
}
