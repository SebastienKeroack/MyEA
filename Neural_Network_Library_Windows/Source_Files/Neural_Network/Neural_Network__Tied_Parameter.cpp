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

bool Layer::Use__Tied_Parameter(void) const { return(this->use_tied_parameter); }

bool Neural_Network::Set__Tied_Parameter(size_t const index_layer_received,
                                                                bool const use_tied_parameter_received,
                                                                bool const transpose_received)
{
    if(index_layer_received >= this->total_layers)
    {
        PRINT_FORMAT("%s: ERROR: Layer received (%zu) overflow the number of layers (%zu) in the neural network." NEW_LINE,
                                 __FUNCTION__,
                                 index_layer_received,
                                 this->total_layers);

        return(false);
    }
    else if(this->ptr_array_layers == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_array_layers\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(this->Set__Tied_Parameter(this->ptr_array_layers + index_layer_received,
                                                      use_tied_parameter_received,
                                                      transpose_received));
}

bool Neural_Network::Set__Tied_Parameter(struct Layer *const ptr_layer_received,
                                                                bool const use_tied_parameter_received,
                                                                bool const transpose_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the input layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the output layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received->use_tied_parameter == use_tied_parameter_received) { return(true); }

    // Mirror.
    if(ptr_layer_received < this->Get__End_Layer__Active() - 1) // Get last active layer.
    {
        struct Layer *const tmp_ptr_mirror_layer(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1);

        if(ptr_layer_received->type_layer != tmp_ptr_mirror_layer->type_layer)
        {
            PRINT_FORMAT("%s: %s: ERROR: The layer type (%u | %s) differ from the mirror layer type (%u | %s). At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_received->type_layer].c_str(),
                                     tmp_ptr_mirror_layer->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_mirror_layer->type_layer].c_str(),
                                     __LINE__);

            return(false);
        }
        else if(*ptr_layer_received->ptr_number_outputs != *tmp_ptr_mirror_layer->ptr_number_outputs)
    {
        PRINT_FORMAT("%s: %s: ERROR: The layer size (%zu) differ from the mirror layer size (%zu). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 *ptr_layer_received->ptr_number_outputs,
                                 *tmp_ptr_mirror_layer->ptr_number_outputs,
                                 __LINE__);

        return(false);
    }
        else if(this->Set__Tied_Parameter(tmp_ptr_mirror_layer,
                                                          use_tied_parameter_received,
                                                          false))
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Tied_Parameter(ptr, %s, false)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     use_tied_parameter_received ? "true" : "false",
                                     __LINE__);

            return(false);
        }
    }
    // |END| Mirror. |END|
    
    if(ptr_layer_received->use_tied_parameter == false && use_tied_parameter_received)
    {
        ++this->total_tied_weight_layers;

        if(transpose_received) { this->Tied__Transpose(ptr_layer_received); }
    }
    else if(ptr_layer_received->use_tied_parameter && use_tied_parameter_received == false) { --this->total_tied_weight_layers; }

    ptr_layer_received->use_tied_parameter = use_tied_parameter_received;
    
    return(true);
}

void Neural_Network::Tied__Transpose(void)
{
    struct Layer const *const tmp_ptr_end_layer(this->ptr_array_layers + (this->total_layers - 3_zu) / 2_zu + 2_zu);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
    for(; tmp_ptr_layer_it != tmp_ptr_end_layer; ++tmp_ptr_layer_it)
    {
        if(tmp_ptr_layer_it->Use__Tied_Parameter())
        {
            this->Tied__Transpose(tmp_ptr_layer_it);
        }
    }
}

void Neural_Network::Tied__Transpose(struct Layer *const ptr_layer_received)
{
    this->Tied__Transpose__Weight(ptr_layer_received);
}