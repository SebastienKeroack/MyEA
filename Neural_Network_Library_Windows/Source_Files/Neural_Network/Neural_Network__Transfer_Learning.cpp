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

#include <iostream>

bool Neural_Network::Transfer_Learning(class Neural_Network *&ptr_destination_Neural_Network_received) const
{
    // Copy connections.
    struct Layer const *tmp_ptr_source_last_layer,
                               *tmp_ptr_destination_last_layer,
                               *tmp_ptr_source_layer_it;
    struct Layer *tmp_ptr_destination_layer_it;

    if(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
      &&
      ptr_destination_Neural_Network_received->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER)
    {
        // Input ... Coded.
        tmp_ptr_source_last_layer = this->ptr_array_layers + MyEA::Math::Minimum<size_t>((this->total_layers - 3_zu) / 2_zu + 2_zu, (ptr_destination_Neural_Network_received->total_layers - 3_zu) / 2_zu + 2_zu); // First decoded layer.
        tmp_ptr_destination_last_layer = ptr_destination_Neural_Network_received->ptr_array_layers + MyEA::Math::Minimum<size_t>((this->total_layers - 3_zu) / 2_zu + 2_zu, (ptr_destination_Neural_Network_received->total_layers - 3_zu) / 2_zu + 2_zu); // First decoded layer.

        //  Compare dimensions, type activation, bias.
        for(tmp_ptr_destination_layer_it = ptr_destination_Neural_Network_received->ptr_array_layers,
            tmp_ptr_source_layer_it = this->ptr_array_layers; tmp_ptr_source_layer_it != tmp_ptr_source_last_layer; ++tmp_ptr_source_layer_it,
                                                                                                                                                                    ++tmp_ptr_destination_layer_it)
        {
            if(tmp_ptr_source_layer_it->Compare__Dimensions(*tmp_ptr_destination_layer_it) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: Layer dimensions unequal. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            tmp_ptr_destination_layer_it->type_activation = tmp_ptr_source_layer_it->type_activation;
            
            //  Bias.
            if(tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index != 0_zu)
            {
                MEMCPY(ptr_destination_Neural_Network_received->ptr_array_parameters + tmp_ptr_destination_layer_it->first_bias_connection_index,
                               this->ptr_array_parameters + tmp_ptr_source_layer_it->first_bias_connection_index,
                               (tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index) * sizeof(T_));
            }
            //  |END| Bias. |END|
        }
        //  |END| Compare dimensions, type activation, bias. |END|

        //  Unit(s).
        ptr_destination_Neural_Network_received->Copy__AF_Units(0_zu,
                                                                                                static_cast<size_t>(tmp_ptr_source_last_layer->ptr_array_AF_units - this->ptr_array_AF_units),
                                                                                                this->ptr_array_AF_units);
        
        ptr_destination_Neural_Network_received->Copy__AF_Ind_Recurrent_Units(0_zu,
                                                                                                                      static_cast<size_t>(tmp_ptr_source_last_layer->ptr_array_AF_Ind_recurrent_units - this->ptr_array_AF_Ind_recurrent_units),
                                                                                                                      this->ptr_array_AF_Ind_recurrent_units,
                                                                                                                      false);
        
        ptr_destination_Neural_Network_received->Copy__Blocks(0_zu,
                                                                                            static_cast<size_t>(tmp_ptr_source_last_layer->ptr_array_block_units - this->ptr_array_block_units),
                                                                                            this->ptr_array_block_units,
                                                                                            false);
        //  |END| Unit(s). |END|
        
        //  Dropout.
        ptr_destination_Neural_Network_received->Copy__Dropout(this->ptr_array_layers,
                                                                                              tmp_ptr_source_last_layer,
                                                                                              ptr_destination_Neural_Network_received->ptr_array_layers);
        //  |END| Dropout. |END|
        
        //  Normalization.
        ptr_destination_Neural_Network_received->Copy__Normalization(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                      tmp_ptr_source_last_layer,
                                                                                                      ptr_destination_Neural_Network_received->ptr_array_layers + 1); // Skip input layer.

        ptr_destination_Neural_Network_received->Copy__Normalized_Units(0_zu,
                                                                                                            static_cast<size_t>(tmp_ptr_source_last_layer->ptr_array_normalized_units - this->ptr_array_normalized_units),
                                                                                                            this->ptr_array_normalized_units);
        //  |END| Normalization. |END|

        //  k-Sparse.
        ptr_destination_Neural_Network_received->Copy__Sparse_K_Filters(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                           tmp_ptr_source_last_layer,
                                                                                                           ptr_destination_Neural_Network_received->ptr_array_layers + 1); // Skip input layer.
        //  |END| k-Sparse. |END|
        
        //  Constraint recurrent weight.
        ptr_destination_Neural_Network_received->Copy__Constraint_Recurrent_Weight(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                                             tmp_ptr_source_last_layer,
                                                                                                                             ptr_destination_Neural_Network_received->ptr_array_layers + 1); // Skip input layer.
        //  |END| Constraint recurrent weight. |END|
        
        //  Weights.
        MEMCPY(ptr_destination_Neural_Network_received->ptr_array_parameters,
                        this->ptr_array_parameters,
                        *tmp_ptr_source_last_layer->ptr_first_connection_index * sizeof(T_));
        //  |END| Weights. |END|
        // |END| Input ... Coded. |END|
        
        // Coded ... Output.
        struct Layer const *tmp_ptr_source_layer_begin(this->ptr_last_layer - MyEA::Math::Minimum<size_t>((this->total_layers - 3_zu) / 2_zu + 1_zu, (ptr_destination_Neural_Network_received->total_layers - 3_zu) / 2_zu + 1_zu)); // First decoded layer.
        struct Layer *tmp_ptr_destination_layer_begin(ptr_destination_Neural_Network_received->ptr_last_layer - MyEA::Math::Minimum<size_t>((this->total_layers - 3_zu) / 2_zu + 1_zu, (ptr_destination_Neural_Network_received->total_layers - 3_zu) / 2_zu + 1_zu)); // First decoded layer.

        tmp_ptr_source_last_layer = this->ptr_last_layer - 1; // Get output layer.
        tmp_ptr_destination_last_layer = ptr_destination_Neural_Network_received->ptr_last_layer - 1; // Get output layer.
        
        // Compare dimensions, type activation, bias.
        for(tmp_ptr_destination_layer_it = tmp_ptr_destination_layer_begin,
            tmp_ptr_source_layer_it = tmp_ptr_source_layer_begin; tmp_ptr_source_layer_it != tmp_ptr_source_last_layer; ++tmp_ptr_source_layer_it,
                                                                                                                                                                             ++tmp_ptr_destination_layer_it)
        {
            if(tmp_ptr_source_layer_it->Compare__Dimensions(*tmp_ptr_destination_layer_it) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: Layer dimensions unequal. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            tmp_ptr_destination_layer_it->type_activation = tmp_ptr_source_layer_it->type_activation;
            
            //  Bias.
            if(tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index != 0_zu)
            {
                MEMCPY(ptr_destination_Neural_Network_received->ptr_array_parameters + tmp_ptr_destination_layer_it->first_bias_connection_index,
                               this->ptr_array_parameters + tmp_ptr_source_layer_it->first_bias_connection_index,
                               (tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index) * sizeof(T_));
            }
            //  |END| Bias. |END|
        }

        //  Output layer, Bias.
        if(tmp_ptr_source_last_layer->Compare__Dimensions(*tmp_ptr_destination_last_layer))
        {
            MEMCPY(ptr_destination_Neural_Network_received->ptr_array_parameters + tmp_ptr_destination_last_layer->first_bias_connection_index,
                            this->ptr_array_parameters + tmp_ptr_source_last_layer->first_bias_connection_index,
                            (tmp_ptr_source_last_layer->last_bias_connection_index - tmp_ptr_source_last_layer->first_bias_connection_index) * sizeof(T_));
        }
        //  |END| Output layer, Bias. |END|
        // |END| Compare dimensions, type activation, bias. |END|
        
        //  Unit(s).
        if(tmp_ptr_source_last_layer != tmp_ptr_source_layer_begin)
        {
            ptr_destination_Neural_Network_received->Copy__AF_Units(static_cast<size_t>(tmp_ptr_destination_layer_begin->ptr_array_AF_units - ptr_destination_Neural_Network_received->ptr_array_AF_units),
                                                                                                    static_cast<size_t>(tmp_ptr_destination_last_layer->ptr_array_AF_units - ptr_destination_Neural_Network_received->ptr_array_AF_units),
                                                                                                    this->ptr_array_AF_units + static_cast<size_t>(tmp_ptr_source_layer_begin->ptr_array_AF_units - this->ptr_array_AF_units));
            
            ptr_destination_Neural_Network_received->Copy__AF_Ind_Recurrent_Units(static_cast<size_t>(tmp_ptr_destination_layer_begin->ptr_array_AF_Ind_recurrent_units - ptr_destination_Neural_Network_received->ptr_array_AF_Ind_recurrent_units),
                                                                                                                          static_cast<size_t>(tmp_ptr_destination_last_layer->ptr_array_AF_Ind_recurrent_units - ptr_destination_Neural_Network_received->ptr_array_AF_Ind_recurrent_units),
                                                                                                                          this->ptr_array_AF_Ind_recurrent_units + static_cast<size_t>(tmp_ptr_source_layer_begin->ptr_array_AF_Ind_recurrent_units - this->ptr_array_AF_Ind_recurrent_units),
                                                                                                                          false);
            
            ptr_destination_Neural_Network_received->Copy__Blocks(static_cast<size_t>(tmp_ptr_destination_layer_begin->ptr_array_block_units - ptr_destination_Neural_Network_received->ptr_array_block_units),
                                                                                                static_cast<size_t>(tmp_ptr_destination_last_layer->ptr_array_block_units - ptr_destination_Neural_Network_received->ptr_array_block_units),
                                                                                                this->ptr_array_block_units + static_cast<size_t>(tmp_ptr_source_layer_begin->ptr_array_block_units - this->ptr_array_block_units),
                                                                                                false);
        }
        //  |END| Unit(s). |END|
        
        //  Weights.
        if(tmp_ptr_source_last_layer->Compare__Dimensions(*tmp_ptr_destination_last_layer))
        {
            MEMCPY(ptr_destination_Neural_Network_received->ptr_array_parameters + *tmp_ptr_destination_layer_begin->ptr_first_connection_index,
                            this->ptr_array_parameters + *tmp_ptr_source_layer_begin->ptr_first_connection_index,
                            (*tmp_ptr_source_last_layer->ptr_last_connection_index - *tmp_ptr_source_layer_begin->ptr_first_connection_index) * sizeof(T_));
        }
        else if(*tmp_ptr_source_last_layer->ptr_first_connection_index - *tmp_ptr_source_layer_begin->ptr_first_connection_index != 0_zu)
        {
            MEMCPY(ptr_destination_Neural_Network_received->ptr_array_parameters + *tmp_ptr_destination_layer_begin->ptr_first_connection_index,
                            this->ptr_array_parameters + *tmp_ptr_source_layer_begin->ptr_first_connection_index,
                            (*tmp_ptr_source_last_layer->ptr_first_connection_index - *tmp_ptr_source_layer_begin->ptr_first_connection_index) * sizeof(T_));
        }
        //  |END| Weights. |END|
        // |END| Coded ... Output. |END|
    }
    else
    {
        tmp_ptr_source_last_layer = this->ptr_array_layers + MyEA::Math::Minimum<size_t>(this->total_layers, ptr_destination_Neural_Network_received->total_layers) - 1_zu; // Get output layer.
        tmp_ptr_destination_last_layer = ptr_destination_Neural_Network_received->ptr_array_layers + MyEA::Math::Minimum<size_t>(this->total_layers, ptr_destination_Neural_Network_received->total_layers) - 1_zu; // Get output layer.

        // Compare dimensions, type activation, bias.
        for(tmp_ptr_destination_layer_it = ptr_destination_Neural_Network_received->ptr_array_layers,
            tmp_ptr_source_layer_it = this->ptr_array_layers; tmp_ptr_source_layer_it != tmp_ptr_source_last_layer; ++tmp_ptr_source_layer_it,
                                                                                                                                                                    ++tmp_ptr_destination_layer_it)
        {
            if(tmp_ptr_source_layer_it->Compare__Dimensions(*tmp_ptr_destination_layer_it) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: Layer dimensions unequal. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            tmp_ptr_destination_layer_it->type_activation = tmp_ptr_source_layer_it->type_activation;
            
            //  Bias.
            if(tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index != 0_zu)
            {
                MEMCPY(ptr_destination_Neural_Network_received->ptr_array_parameters + tmp_ptr_destination_layer_it->first_bias_connection_index,
                               this->ptr_array_parameters + tmp_ptr_source_layer_it->first_bias_connection_index,
                               (tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index) * sizeof(T_));
            }
            //  |END| Bias. |END|
        }

        //  Output layer, Bias.
        if(tmp_ptr_source_last_layer->Compare__Dimensions(*tmp_ptr_destination_last_layer))
        {
            MEMCPY(ptr_destination_Neural_Network_received->ptr_array_parameters + tmp_ptr_destination_last_layer->first_bias_connection_index,
                            this->ptr_array_parameters + tmp_ptr_source_last_layer->first_bias_connection_index,
                            (tmp_ptr_source_last_layer->last_bias_connection_index - tmp_ptr_source_last_layer->first_bias_connection_index) * sizeof(T_));
        }
        //  |END| Output layer, Bias. |END|
        // |END| Compare dimensions, type activation, bias. |END|
        
        // Unit(s).
        ptr_destination_Neural_Network_received->Copy__AF_Units(0_zu,
                                                                                                ptr_destination_Neural_Network_received->total_AF_units,
                                                                                                this->ptr_array_AF_units);
        
        ptr_destination_Neural_Network_received->Copy__AF_Ind_Recurrent_Units(0_zu,
                                                                                                                      ptr_destination_Neural_Network_received->total_AF_Ind_recurrent_units,
                                                                                                                      this->ptr_array_AF_Ind_recurrent_units,
                                                                                                                      false);
        
        ptr_destination_Neural_Network_received->Copy__Blocks(0_zu,
                                                                                            ptr_destination_Neural_Network_received->total_block_units,
                                                                                            this->ptr_array_block_units,
                                                                                            false);
        // |END| Unit(s). |END|

        // Dropout.
        ptr_destination_Neural_Network_received->Copy__Dropout(this->ptr_array_layers,
                                                                                              tmp_ptr_source_last_layer,
                                                                                              ptr_destination_Neural_Network_received->ptr_array_layers);
        // |END| Dropout. |END|
        
        // Normalization.
        ptr_destination_Neural_Network_received->Copy__Normalization(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                      tmp_ptr_source_last_layer,
                                                                                                      ptr_destination_Neural_Network_received->ptr_array_layers + 1); // Skip input layer.

        ptr_destination_Neural_Network_received->Copy__Normalized_Units(0_zu,
                                                                                                            ptr_destination_Neural_Network_received->total_normalized_units,
                                                                                                            this->ptr_array_normalized_units);
        // |END| Normalization. |END|
        
        // k-Sparse.
        ptr_destination_Neural_Network_received->Copy__Sparse_K_Filters(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                           tmp_ptr_source_last_layer,
                                                                                                           ptr_destination_Neural_Network_received->ptr_array_layers + 1); // Skip input layer.
        // |END| k-Sparse. |END|
        
        // Constraint recurrent weight.
        ptr_destination_Neural_Network_received->Copy__Constraint_Recurrent_Weight(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                                             tmp_ptr_source_last_layer,
                                                                                                                             ptr_destination_Neural_Network_received->ptr_array_layers + 1); // Skip input layer.
        // |END| Constraint recurrent weight. |END|
        
        // Weights.
        if(tmp_ptr_source_last_layer->Compare__Dimensions(*tmp_ptr_destination_last_layer))
        {
            MEMCPY(ptr_destination_Neural_Network_received->ptr_array_parameters,
                           this->ptr_array_parameters,
                           *tmp_ptr_source_last_layer->ptr_last_connection_index * sizeof(T_));
        }
        else
        {
            MEMCPY(ptr_destination_Neural_Network_received->ptr_array_parameters,
                           this->ptr_array_parameters,
                           *tmp_ptr_source_last_layer->ptr_first_connection_index * sizeof(T_));
        }
        // |END| Weights. |END|
    }
    // |END| Copy connections. |END|
    
    // Normalization.
    ptr_destination_Neural_Network_received->Copy__Normalization(this);
    // |END| Normalization. |END|
    
    // Initializer weight parameters.
    ptr_destination_Neural_Network_received->Copy__Initializer__Weight_Parameter(*this);
    // |END| Initializer weight parameters. |END|
    
    // Training parameters.
    ptr_destination_Neural_Network_received->Copy__Training_Parameters(this);
    // |END| Training parameters. |END|
    
    // Optimizer parameters.
    if(ptr_destination_Neural_Network_received->Copy__Optimizer_Parameters(this) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Optimizer_Parameters(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| Optimizer parameters. |END|

    // Regularization parameters.
    ptr_destination_Neural_Network_received->Copy__Regularization(this);
    // |END| Regularization parameters. |END|
    
    // Compute parameters.
    ptr_destination_Neural_Network_received->maximum_allowable_memory_bytes = this->maximum_allowable_memory_bytes;
    
    if(ptr_destination_Neural_Network_received->Set__Maximum_Thread_Usage(this->percentage_maximum_thread_usage) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Maximum_Thread_Usage(%f)\" function. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    this->percentage_maximum_thread_usage,
                                    __LINE__);

        return(false);
    }
    else if(ptr_destination_Neural_Network_received->Set__Maximum__Batch_Size(this->maximum_batch_size) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Maximum__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->maximum_batch_size,
                                 __LINE__);

        return(false);
    }
    // |END| Compute parameters. |END|

    return(true);
}
