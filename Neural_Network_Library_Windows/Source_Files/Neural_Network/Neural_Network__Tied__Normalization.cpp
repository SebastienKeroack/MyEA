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

void Neural_Network::Tied__Transpose__Normalization(struct Layer *const ptr_layer_received)
{
    struct Layer *const tmp_ptr_mirror_layer_it(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1);
    
    if(ptr_layer_received != tmp_ptr_mirror_layer_it)
    {
        switch(ptr_layer_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION: this->Tied__Transpose__Normalization__Batch_Normalization(ptr_layer_received, tmp_ptr_mirror_layer_it); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                            MyEA::Time::Date_Time_Now().c_str(),
                                            __FUNCTION__,
                                            ptr_layer_received->type_normalization,
                                            MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_received->type_normalization].c_str());
                    break;
        }
    }
}

void Neural_Network::Tied__Transpose__Normalization__Batch_Normalization(struct Layer const *const ptr_encoded_layer_it_received, struct Layer const *const ptr_mirror_layer_it_received)
{
    struct Normalized_batch_unit const *const tmp_ptr_encoded_layer_it_first_normalized_batch_unit(&ptr_encoded_layer_it_received->ptr_array_normalized_units->normalized_batch_units),
                                                        *const tmp_ptr_mirror_layer_it_first_normalized_batch_unit(&ptr_mirror_layer_it_received->ptr_array_normalized_units->normalized_batch_units);
    
    size_t const tmp_number_units(static_cast<size_t>(ptr_encoded_layer_it_received->ptr_last_normalized_unit - ptr_encoded_layer_it_received->ptr_array_normalized_units));
    
    MEMCPY(tmp_ptr_mirror_layer_it_first_normalized_batch_unit->ptr_scale,
                   tmp_ptr_encoded_layer_it_first_normalized_batch_unit->ptr_scale,
                   tmp_number_units * sizeof(T_));
    
    MEMCPY(tmp_ptr_mirror_layer_it_first_normalized_batch_unit->ptr_shift,
                   tmp_ptr_encoded_layer_it_first_normalized_batch_unit->ptr_shift,
                   tmp_number_units * sizeof(T_));
}
