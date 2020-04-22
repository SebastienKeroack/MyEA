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

void Neural_Network::Dropout_Bernoulli(void)
{
    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        #pragma omp parallel
        this->Dropout_Bernoulli__OpenMP();
    }
    else
    { this->Dropout_Bernoulli__Loop(); }
}

void Neural_Network::Dropout_Bernoulli__Loop(void)
{
    size_t tmp_number_outputs;

    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

    // Input layer.
    if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI) { this->Dropout_Bernoulli__Layer__Loop(this->number_inputs, tmp_ptr_layer_it); }

    for(++tmp_ptr_layer_it; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI)
        {
            switch(tmp_ptr_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: tmp_number_outputs = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_unit - tmp_ptr_layer_it->ptr_array_AF_units); break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: tmp_number_outputs = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units); break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_layer_it->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                        return;
            }

            this->Dropout_Bernoulli__Layer__Loop(tmp_number_outputs, tmp_ptr_layer_it);
        }
    }
}

void Neural_Network::Dropout_Bernoulli__Layer__Loop(size_t const number_outputs_received, struct Layer *const ptr_layer_it_received)
{
    T_ const tmp_retained_probability(ptr_layer_it_received->dropout_values[0u]);

    if(tmp_retained_probability != 0_T)
    {
        size_t tmp_unit_index,
                  tmp_time_step_index,
                  tmp_timed_mask_index;
        
        this->ptr_array_Class_Generator_Bernoulli->Probability(tmp_retained_probability);
        
        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_timed_mask_index = tmp_time_step_index * number_outputs_received;

            for(tmp_unit_index = 0_zu; tmp_unit_index != number_outputs_received; ++tmp_unit_index)
            {
                if((*this->ptr_array_Class_Generator_Bernoulli)()) // Keep unit.
                { ptr_layer_it_received->ptr_array__mask__dropout__bernoulli[tmp_timed_mask_index + tmp_unit_index] = true; }
                else // Drop unit.
                { ptr_layer_it_received->ptr_array__mask__dropout__bernoulli[tmp_timed_mask_index + tmp_unit_index] = false; }
            }
        }
    }
    else
    {
        MyEA::Memory::Fill<bool>(ptr_layer_it_received->ptr_array__mask__dropout__bernoulli,
                                     ptr_layer_it_received->ptr_array__mask__dropout__bernoulli + number_outputs_received * this->number_recurrent_depth,
                                     false);
    }
}
