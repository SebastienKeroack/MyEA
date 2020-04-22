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

void Neural_Network::Set__Clip_Gradient(bool const use_clip_gradient_received) { this->use_clip_gradient = use_clip_gradient_received; }

bool Neural_Network::Set__Clip_Gradient(T_ const clip_gradient_received)
{
    if(this->clip_gradient == clip_gradient_received) { return(true); }

    bool *tmp_ptr_array_layers_use_default(new bool[this->total_layers - 2_zu]);
    if(tmp_ptr_array_layers_use_default == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 (this->total_layers - 2_zu) * sizeof(bool),
                                 __LINE__);

        return(false);
    }
    memset(tmp_ptr_array_layers_use_default,
                 0,
                 (this->total_layers - 2_zu) * sizeof(bool));

    size_t const tmp_number_layers(this->total_layers - 2_zu);
    size_t tmp_layer_index;

    struct Layer *tmp_ptr_layer_it;

    for(tmp_layer_index = 0_zu; tmp_layer_index != tmp_number_layers; ++tmp_layer_index)
    {
        tmp_ptr_layer_it = this->ptr_array_layers + tmp_layer_index + 1;

        // Regularization on recurrent connection(s) (Independently RNN).
        tmp_ptr_array_layers_use_default[tmp_layer_index] = tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT
                                                                                      &&
                                                                                      this->Check__Use__Regularization__Constraint_Recurrent_Weight__Default(tmp_ptr_layer_it);
    }
    
    this->clip_gradient = clip_gradient_received;
    
    for(tmp_layer_index = 0_zu; tmp_layer_index != tmp_number_layers; ++tmp_layer_index)
    {
        tmp_ptr_layer_it = this->ptr_array_layers + tmp_layer_index + 1;

        // Regularization on recurrent connection(s) (Independently RNN).
        if(tmp_ptr_array_layers_use_default[tmp_layer_index]
          &&
          this->Set__Regularization__Constraint_Recurrent_Weight__Default(tmp_ptr_layer_it) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight__Default(ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);
        
            return(false);
        }
    }

    return(true);
}

void Neural_Network::Clip_Gradient__Loop(size_t const start_index_received, size_t const end_index_received)
{
    this->Euclidean_Norm__Loop(start_index_received,
                                                end_index_received,
                                                this->clip_gradient,
                                                this->ptr_array_derivatives_parameters);
}

void Neural_Network::Clip_Gradient__OpenMP(size_t const start_index_received, size_t const end_index_received)
{
    this->Euclidean_Norm__OpenMP(start_index_received,
                                                     end_index_received,
                                                     this->clip_gradient,
                                                     this->ptr_array_derivatives_parameters);
}