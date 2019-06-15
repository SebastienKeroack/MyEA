#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

void Neural_Network::Order__Layers__Connection(void)
{
    size_t tmp_state_previous_layer_index(0_zu),
              tmp_state_layer_index(0_zu),
              tmp_state_next_layer_index(0_zu);
    
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                               *tmp_ptr_previous_layer_state(nullptr),
                               *tmp_ptr_layer_state(nullptr),
                               *tmp_ptr_next_layer_state(nullptr);
    struct Layer *tmp_ptr_layer_it;
    
    for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        this->Organize__Previous_Layers_Connected(tmp_state_previous_layer_index,
                                                                           tmp_ptr_layer_it,
                                                                           tmp_ptr_previous_layer_state);
        
        this->Organize__Next_Layers_Connected(tmp_state_next_layer_index,
                                                                     tmp_ptr_layer_it,
                                                                     tmp_ptr_next_layer_state);
        
        this->Organize__Layer__Group(tmp_state_layer_index,
                                                      tmp_ptr_layer_it,
                                                      tmp_ptr_layer_state);

    #if defined(COMPILE_DEBUG_PRINT)
        PRINT_FORMAT(NEW_LINE);
        PRINT_FORMAT("Layer[%zu] %s, {%zu}" NEW_LINE,
                                 (size_t)(tmp_ptr_layer_it - this->ptr_array_layers),
                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                 *tmp_ptr_layer_it->ptr_number_outputs);
        for(size_t i=0_zu; i != tmp_ptr_layer_it->previous_connected_layers.size(); ++i)
        {
            PRINT_FORMAT("\t<< [%zu], %s, {%zu}" NEW_LINE,
                                     (size_t)(tmp_ptr_layer_it->previous_connected_layers[i] - this->ptr_array_layers),
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->previous_connected_layers[i]->type_layer].c_str(),
                                     *tmp_ptr_layer_it->previous_connected_layers[i]->ptr_number_outputs);
        }
        for(size_t i=0_zu; i != tmp_ptr_layer_it->next_connected_layers.size(); ++i)
        {
            PRINT_FORMAT("\t>> [%zu], %s, {%zu}" NEW_LINE,
                                     (size_t)(tmp_ptr_layer_it->next_connected_layers[i] - this->ptr_array_layers),
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->next_connected_layers[i]->type_layer].c_str(),
                                     *tmp_ptr_layer_it->next_connected_layers[i]->ptr_number_outputs);
        }
    #endif
    }
}

void Neural_Network::Order__Layers__Output(void)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
    
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it) { this->Order__Layer__Output(true, tmp_ptr_layer_it); }
}

void Neural_Network::Order__Layer__Output(bool const is_sequentiel_received, struct Layer *const ptr_layer_received)
{
    if(this->pre_training_level != 0_zu) { return(this->Order__Layer__Output__Pre_Training(is_sequentiel_received, ptr_layer_received)); }

    struct Layer *const tmp_ptr_previous_layer_connected(ptr_layer_received->previous_connected_layers.size() != 0_zu ? const_cast<struct Layer *>(ptr_layer_received->previous_connected_layers[0u]) : ptr_layer_received);

    switch(ptr_layer_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            ptr_layer_received->ptr_array_pre_summations = nullptr;

            ptr_layer_received->ptr_array_pre_normalization = nullptr;
            
            ptr_layer_received->ptr_array_pre_activation_functions = nullptr;

            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_basic_units->ptr_array_values;
            
            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_basic_units->ptr_array_errors;
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            // Update the previous layer connected before continuing.
            if(is_sequentiel_received == false) { this->Order__Layer__Output(true, tmp_ptr_previous_layer_connected); }

            if(ptr_layer_received->type_group == MyEA::Common::ENUM_TYPE_GROUP::TYPE_GROUP_RESIDUAL)
            {
                bool const tmp_is_block_input_layer(static_cast<size_t>(ptr_layer_received->ptr_last_AF_unit - ptr_layer_received->ptr_array_AF_units) + static_cast<size_t>(ptr_layer_received->ptr_last_AF_Ind_recurrent_unit - ptr_layer_received->ptr_array_AF_Ind_recurrent_units) == 0_zu);
                
                if(ptr_layer_received->Use__Normalization())
                {
                    switch(ptr_layer_received->type_normalization)
                    {
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
                            ptr_layer_received->ptr_array_pre_normalization = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                            
                            if(tmp_is_block_input_layer == false)
                            {
                                ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_values_normalizes;
                                
                                ptr_layer_received->ptr_array_pre_summations = ptr_layer_received->ptr_array_AF_units->ptr_array_values;
                            }
                            else
                            {
                                ptr_layer_received->ptr_array_pre_activation_functions = nullptr;
                                
                                ptr_layer_received->ptr_array_pre_summations = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_values_normalizes;
                            }

                            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_neuron_units->ptr_array_errors;
                                break;
                        default:
                            PRINT_FORMAT("%s: %s: ERROR: Layer normalization type (%u | %s) is not managed in the switch." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     ptr_layer_received->type_normalization,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_received->type_normalization].c_str());
                                return;
                    }
                }
                else
                {
                    ptr_layer_received->ptr_array_pre_normalization = nullptr;
                    
                    if(tmp_is_block_input_layer == false)
                    {
                        ptr_layer_received->ptr_array_pre_activation_functions = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                        
                        ptr_layer_received->ptr_array_pre_summations = ptr_layer_received->ptr_array_AF_units->ptr_array_values;
                    }
                    else
                    {
                        ptr_layer_received->ptr_array_pre_activation_functions = nullptr;
                        
                        ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                    }

                    ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                    ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_neuron_units->ptr_array_errors;
                }
                
                // If the forward connection is a residual block. Update the residual block (normalization).
                if(ptr_layer_received->next_connected_layers[0u]->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { this->Order__Layer__Output(is_sequentiel_received, const_cast<struct Layer *>(ptr_layer_received->next_connected_layers[0u])); }
            }
            else
            {
                if(ptr_layer_received->Use__Normalization())
                {
                    switch(ptr_layer_received->type_normalization)
                    {
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
                            if(ptr_layer_received->use_layer_normalization_before_activation)
                            {
                                ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                                
                                ptr_layer_received->ptr_array_pre_normalization = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                                ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_values_normalizes;
                                
                                ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_AF_units->ptr_array_values;

                                ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_AF_units->ptr_array_errors;
                            }
                            else
                            {
                                ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                                
                                ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                                ptr_layer_received->ptr_array_pre_normalization = ptr_layer_received->ptr_array_AF_units->ptr_array_values;

                                ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_values_normalizes;

                                ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_errors;
                            }
                                break;
                        default:
                            PRINT_FORMAT("%s: %s: ERROR: Layer normalization type (%u | %s) is not managed in the switch." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     ptr_layer_received->type_normalization,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_received->type_normalization].c_str());
                                return;
                    }
                }
                else
                {
                    ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;

                    ptr_layer_received->ptr_array_pre_normalization = nullptr;

                    ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                    ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_AF_units->ptr_array_values;

                    ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_AF_units->ptr_array_errors;
                }
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            // Update the previous layer connected before continuing.
            if(is_sequentiel_received == false) { this->Order__Layer__Output(true, tmp_ptr_previous_layer_connected); }

            if(ptr_layer_received->type_group == MyEA::Common::ENUM_TYPE_GROUP::TYPE_GROUP_RESIDUAL)
            {
                bool const tmp_is_block_input_layer(static_cast<size_t>(ptr_layer_received->ptr_last_AF_unit - ptr_layer_received->ptr_array_AF_units) + static_cast<size_t>(ptr_layer_received->ptr_last_AF_Ind_recurrent_unit - ptr_layer_received->ptr_array_AF_Ind_recurrent_units) == 0_zu);
                
                if(ptr_layer_received->Use__Normalization())
                {
                    switch(ptr_layer_received->type_normalization)
                    {
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
                            ptr_layer_received->ptr_array_pre_normalization = tmp_ptr_previous_layer_connected->ptr_array_outputs;

                            if(tmp_is_block_input_layer == false)
                            {
                                ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_pre_AFs;
                                
                                ptr_layer_received->ptr_array_pre_summations = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_AFs;
                            }
                            else
                            {
                                ptr_layer_received->ptr_array_pre_activation_functions = nullptr;
                                
                                ptr_layer_received->ptr_array_pre_summations = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_values_normalizes;
                            }

                            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_neuron_units->ptr_array_errors;
                                break;
                        default:
                            PRINT_FORMAT("%s: %s: ERROR: Layer normalization type (%u | %s) is not managed in the switch." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     ptr_layer_received->type_normalization,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_received->type_normalization].c_str());
                                return;
                    }
                }
                else
                {
                    ptr_layer_received->ptr_array_pre_normalization = nullptr;

                    if(tmp_is_block_input_layer == false)
                    {
                        ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_pre_AFs;
                        
                        ptr_layer_received->ptr_array_pre_summations = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_AFs;
                    }
                    else
                    {
                        ptr_layer_received->ptr_array_pre_activation_functions = nullptr;
                        
                        ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                    }

                    ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                    ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_neuron_units->ptr_array_errors;
                }
                
                // If the forward connection is a residual block. Update the residual block (normalization).
                if(ptr_layer_received->next_connected_layers[0u]->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { this->Order__Layer__Output(is_sequentiel_received, const_cast<struct Layer *>(ptr_layer_received->next_connected_layers[0u])); }
            }
            else
            {
                if(ptr_layer_received->Use__Normalization())
                {
                    switch(ptr_layer_received->type_normalization)
                    {
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
                            if(ptr_layer_received->use_layer_normalization_before_activation)
                            {
                                ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                                
                                ptr_layer_received->ptr_array_pre_normalization = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                                ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_pre_AFs;
                                
                                ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_AFs;

                                ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_errors;
                            }
                            else
                            {
                                ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                                
                                ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_pre_AFs;

                                ptr_layer_received->ptr_array_pre_normalization = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_AFs;

                                ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_values_normalizes;

                                ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_errors;
                            }
                                break;
                        default:
                            PRINT_FORMAT("%s: %s: ERROR: Layer normalization type (%u | %s) is not managed in the switch." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     ptr_layer_received->type_normalization,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_received->type_normalization].c_str());
                                return;
                    }
                }
                else
                {
                    ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                    
                    ptr_layer_received->ptr_array_pre_normalization = nullptr;

                    ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_pre_AFs;

                    ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_AFs;

                    ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_errors;
                }
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            ptr_layer_received->ptr_array_pre_summations = nullptr;

            ptr_layer_received->ptr_array_pre_normalization = nullptr;

            ptr_layer_received->ptr_array_pre_activation_functions = nullptr;

            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_cell_units->ptr_cell_output;
            
            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_cell_units->ptr_delta_cell_output;
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            ptr_layer_received->ptr_array_pre_summations = nullptr;

            ptr_layer_received->ptr_array_pre_normalization = nullptr;

            ptr_layer_received->ptr_array_pre_activation_functions = nullptr;

            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_basic_indice_units->ptr_array_values;

            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_basic_indice_units->ptr_array_errors;
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
            ptr_layer_received->ptr_array_pre_summations = nullptr;

            ptr_layer_received->ptr_array_pre_normalization = ptr_layer_received->Use__Normalization() ? (ptr_layer_received + ptr_layer_received->block_depth)->ptr_array_outputs : nullptr;
            
            ptr_layer_received->ptr_array_pre_activation_functions = nullptr;

            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_basic_units->ptr_array_values;
            
            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_basic_units->ptr_array_errors;
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_received->type_layer].c_str());
                return;
    }
}

void Neural_Network::Order__Layer__Output__Pre_Training(bool const is_sequentiel_received, struct Layer *const ptr_layer_received)
{
    struct Layer *tmp_ptr_previous_layer_connected;

    if(ptr_layer_received->previous_connected_layers.size() == 0_zu)
    { tmp_ptr_previous_layer_connected = ptr_layer_received; }
    // If is an encoded/coded layer.
    else if(ptr_layer_received <= this->Get__End_Layer__Active() - 1)
    { tmp_ptr_previous_layer_connected = const_cast<struct Layer *>(ptr_layer_received->previous_connected_layers[0u]); }
    // If is a decoded layer.
    else // if(ptr_layer_received > this->Get__End_Layer__Active() - 1)
    { tmp_ptr_previous_layer_connected = this->ptr_array_layers + static_cast<size_t>(this->ptr_last_layer - ptr_layer_received); }

    switch(ptr_layer_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            ptr_layer_received->ptr_array_pre_summations = nullptr;

            ptr_layer_received->ptr_array_pre_normalization = nullptr;
            
            ptr_layer_received->ptr_array_pre_activation_functions = nullptr;

            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_basic_units->ptr_array_values;
            
            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_basic_units->ptr_array_errors;
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            // Update the previous layer connected before continuing.
            if(is_sequentiel_received == false) { this->Order__Layer__Output__Pre_Training(true, tmp_ptr_previous_layer_connected); }

            if(ptr_layer_received->Use__Normalization())
            {
                switch(ptr_layer_received->type_normalization)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
                        if(ptr_layer_received->use_layer_normalization_before_activation)
                        {
                            ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                                
                            ptr_layer_received->ptr_array_pre_normalization = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                            ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_values_normalizes;
                                
                            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_AF_units->ptr_array_values;

                            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_AF_units->ptr_array_errors;
                        }
                        else
                        {
                            ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                                
                            ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                            ptr_layer_received->ptr_array_pre_normalization = ptr_layer_received->ptr_array_AF_units->ptr_array_values;

                            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_values_normalizes;

                            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_errors;
                        }
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer normalization type (%u | %s) is not managed in the switch." NEW_LINE,
                                                    MyEA::String::Get__Time().c_str(),
                                                    __FUNCTION__,
                                                    ptr_layer_received->type_normalization,
                                                    MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_received->type_normalization].c_str());
                            return;
                }
            }
            else
            {
                ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected == nullptr ? nullptr : tmp_ptr_previous_layer_connected->ptr_array_outputs;

                ptr_layer_received->ptr_array_pre_normalization = nullptr;

                ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_AF_units->ptr_array_values;

                ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_AF_units->ptr_array_errors;
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            // Update the previous layer connected before continuing.
            if(is_sequentiel_received == false) { this->Order__Layer__Output__Pre_Training(true, tmp_ptr_previous_layer_connected); }

            if(ptr_layer_received->Use__Normalization())
            {
                switch(ptr_layer_received->type_normalization)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
                        if(ptr_layer_received->use_layer_normalization_before_activation)
                        {
                            ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                                
                            ptr_layer_received->ptr_array_pre_normalization = ptr_layer_received->ptr_array_neuron_units->ptr_array_summations;

                            ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_pre_AFs;
                                
                            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_AFs;

                            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_errors;
                        }
                        else
                        {
                            ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                                
                            ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_pre_AFs;

                            ptr_layer_received->ptr_array_pre_normalization = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_AFs;

                            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_values_normalizes;

                            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_errors;
                        }
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer normalization type (%u | %s) is not managed in the switch." NEW_LINE,
                                                    MyEA::String::Get__Time().c_str(),
                                                    __FUNCTION__,
                                                    ptr_layer_received->type_normalization,
                                                    MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_received->type_normalization].c_str());
                            return;
                }
            }
            else
            {
                ptr_layer_received->ptr_array_pre_summations = tmp_ptr_previous_layer_connected->ptr_array_outputs;
                    
                ptr_layer_received->ptr_array_pre_normalization = nullptr;

                ptr_layer_received->ptr_array_pre_activation_functions = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_pre_AFs;

                ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_AFs;

                ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_AF_Ind_recurrent_units->ptr_array_errors;
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            ptr_layer_received->ptr_array_pre_summations = nullptr;

            ptr_layer_received->ptr_array_pre_normalization = nullptr;

            ptr_layer_received->ptr_array_pre_activation_functions = nullptr;

            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_cell_units->ptr_cell_output;
            
            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_cell_units->ptr_delta_cell_output;
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            ptr_layer_received->ptr_array_pre_summations = nullptr;

            ptr_layer_received->ptr_array_pre_normalization = nullptr;

            ptr_layer_received->ptr_array_pre_activation_functions = nullptr;

            ptr_layer_received->ptr_array_outputs = ptr_layer_received->ptr_array_basic_indice_units->ptr_array_values;

            ptr_layer_received->ptr_array_derivative_outputs = ptr_layer_received->ptr_array_basic_indice_units->ptr_array_errors;
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_received->type_layer].c_str());
                return;
    }
}

void Neural_Network::Order__Layer__Basic(struct Layer *const ptr_layer_it_received)
{
    this->Order__Layer__Basic_unit(ptr_layer_it_received);
    
    // TODO: Transpose per layers.
    if(ptr_layer_it_received->Use__K_Sparsity()) { this->Assign__Sparsity_Activities(this->number_threads); }
}

void Neural_Network::Order__Layer__Basic_unit(struct Layer *const ptr_layer_it_received)
{
    struct Basic_unit const *const tmp_ptr_last_basic_unit(ptr_layer_it_received->ptr_last_basic_unit);
    struct Basic_unit *tmp_ptr_basic_unit_it(ptr_layer_it_received->ptr_array_basic_units);
    
    if(static_cast<size_t>(tmp_ptr_last_basic_unit - tmp_ptr_basic_unit_it) != 0_zu)
    {
        size_t const tmp_basic_unit_index_start(static_cast<size_t>(tmp_ptr_basic_unit_it - this->ptr_array_basic_units));

        T_ *tmp_ptr_array_basic_units_values(this->ptr_array_basic_units_values + tmp_basic_unit_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_basic_units_errors(this->ptr_array_basic_units_errors + tmp_basic_unit_index_start * this->batch_size * this->number_recurrent_depth);
        
        for(; tmp_ptr_basic_unit_it != tmp_ptr_last_basic_unit; ++tmp_ptr_basic_unit_it,
                                                                                       ++tmp_ptr_array_basic_units_values,
                                                                                       ++tmp_ptr_array_basic_units_errors)
        {
            tmp_ptr_basic_unit_it->ptr_array_values = tmp_ptr_array_basic_units_values;
            tmp_ptr_basic_unit_it->ptr_array_errors = tmp_ptr_array_basic_units_errors;
        }
    }
}

void Neural_Network::Order__Layer__Basic_indice(struct Layer *const ptr_layer_it_received)
{
    this->Order__Layer__Basic_indice_unit(ptr_layer_it_received);

    // TODO: Transpose per layers.
    if(ptr_layer_it_received->Use__K_Sparsity()) { this->Assign__Sparsity_Activities(this->number_threads); }
}

void Neural_Network::Order__Layer__Basic_indice_unit(struct Layer *const ptr_layer_it_received)
{
    struct Basic_indice_unit const *const tmp_ptr_last_basic_indice_unit(ptr_layer_it_received->ptr_last_basic_indice_unit);
    struct Basic_indice_unit *tmp_ptr_basic_indice_unit_it(ptr_layer_it_received->ptr_array_basic_indice_units);
    
    if(static_cast<size_t>(tmp_ptr_last_basic_indice_unit - tmp_ptr_basic_indice_unit_it) != 0_zu)
    {
        size_t const tmp_basic_indice_unit_index_start(static_cast<size_t>(tmp_ptr_basic_indice_unit_it - this->ptr_array_basic_indice_units));
        size_t *tmp_ptr_array_basic_indice_units_indices(this->ptr_array_basic_indice_units_indices + tmp_basic_indice_unit_index_start * this->batch_size * this->number_recurrent_depth);

        T_ *tmp_ptr_array_basic_indice_units_values(this->ptr_array_basic_indice_units_values + tmp_basic_indice_unit_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_basic_indice_units_errors(this->ptr_array_basic_indice_units_errors + tmp_basic_indice_unit_index_start * this->batch_size * this->number_recurrent_depth);
        
        for(; tmp_ptr_basic_indice_unit_it != tmp_ptr_last_basic_indice_unit; ++tmp_ptr_basic_indice_unit_it,
                                                                                                           ++tmp_ptr_array_basic_indice_units_indices,
                                                                                                           ++tmp_ptr_array_basic_indice_units_values,
                                                                                                           ++tmp_ptr_array_basic_indice_units_errors)
        {
            tmp_ptr_basic_indice_unit_it->ptr_array_indices = tmp_ptr_array_basic_indice_units_indices;
            tmp_ptr_basic_indice_unit_it->ptr_array_values = tmp_ptr_array_basic_indice_units_values;
            tmp_ptr_basic_indice_unit_it->ptr_array_errors = tmp_ptr_array_basic_indice_units_errors;
        }
    }
}

void Neural_Network::Order__Layer__Neuron(struct Layer *const ptr_layer_it_received) { this->Order__Layer__Neuron_Unit(ptr_layer_it_received); }

void Neural_Network::Order__Layer__Neuron_Unit(struct Layer *const ptr_layer_it_received)
{
    struct Neuron_unit const *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit);
    struct Neuron_unit *tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);
    
    if(static_cast<size_t>(tmp_ptr_last_neuron_unit - tmp_ptr_neuron_unit_it) != 0_zu)
    {
        size_t const tmp_neuron_unit_index_start(static_cast<size_t>(tmp_ptr_neuron_unit_it - this->ptr_array_neuron_units));

        T_ *tmp_ptr_array_neuron_units_summations(this->ptr_array_neuron_units_summations + tmp_neuron_unit_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_neuron_units_errors(this->ptr_array_neuron_units_errors + tmp_neuron_unit_index_start * this->batch_size * this->number_recurrent_depth);
        
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                           ++tmp_ptr_array_neuron_units_summations,
                                                                                           ++tmp_ptr_array_neuron_units_errors)
        {
            tmp_ptr_neuron_unit_it->ptr_array_summations = tmp_ptr_array_neuron_units_summations;
            tmp_ptr_neuron_unit_it->ptr_array_errors = tmp_ptr_array_neuron_units_errors;
        }
    }
}

void Neural_Network::Order__Layer__AF(struct Layer *const ptr_layer_it_received)
{
    this->Order__Layer__AF_Unit(ptr_layer_it_received);

    if(this->Use__Dropout__Bernoulli() || this->Use__Dropout__Bernoulli__Inverted())
    { this->Order__Layer__AF_Unit__Dropout_Bernoulli(ptr_layer_it_received); }
    
    // TODO: Transpose per layers.
    if(ptr_layer_it_received->Use__K_Sparsity()) { this->Assign__Sparsity_Activities(this->number_threads); }
}

void Neural_Network::Order__Layer__AF_Unit(struct Layer *const ptr_layer_it_received)
{
    struct AF_unit const *const tmp_ptr_last_AF_unit(ptr_layer_it_received->ptr_last_AF_unit);
    struct AF_unit *tmp_ptr_AF_unit_it(ptr_layer_it_received->ptr_array_AF_units);
    
    if(static_cast<size_t>(tmp_ptr_last_AF_unit - tmp_ptr_AF_unit_it) != 0_zu)
    {
        size_t const tmp_AF_unit_index_start(static_cast<size_t>(tmp_ptr_AF_unit_it - this->ptr_array_AF_units));

        T_ *tmp_ptr_array_AF_units_values(this->ptr_array_AF_units_values + tmp_AF_unit_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_AF_units_errors(this->ptr_array_AF_units_errors + tmp_AF_unit_index_start * this->batch_size * this->number_recurrent_depth);
        
        for(; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it,
                                                                            ++tmp_ptr_array_AF_units_values,
                                                                            ++tmp_ptr_array_AF_units_errors)
        {
            tmp_ptr_AF_unit_it->ptr_array_values = tmp_ptr_array_AF_units_values;
            tmp_ptr_AF_unit_it->ptr_array_errors = tmp_ptr_array_AF_units_errors;
        }
    }
}

void Neural_Network::Order__Layer__AF_Unit__Dropout_Bernoulli(struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_AF_unit_index_start(static_cast<size_t>(ptr_layer_it_received->ptr_array_AF_units - this->ptr_array_AF_units));

    ptr_layer_it_received->ptr_array__mask__dropout__bernoulli = this->ptr_array_units_mask_dropout_bernoulli + tmp_AF_unit_index_start * this->number_recurrent_depth;
}

void Neural_Network::Order__Layer__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received)
{
    this->Order__Layer__AF_Ind_Recurrent_Unit(ptr_layer_it_received);

    if(this->Use__Dropout__Bernoulli() || this->Use__Dropout__Bernoulli__Inverted())
    { this->Order__Layer__AF_Ind_Recurrent_Unit__Dropout_Bernoulli(ptr_layer_it_received); }
    
    // TODO: Transpose per layers.
    if(ptr_layer_it_received->Use__K_Sparsity()) { this->Assign__Sparsity_Activities(this->number_threads); }
}

void Neural_Network::Order__Layer__AF_Ind_Recurrent_Unit(struct Layer *const ptr_layer_it_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit);
    struct AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    if(static_cast<size_t>(tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit - tmp_ptr_AF_Ind_recurrent_unit_it) != 0_zu)
    {
        size_t const tmp_AF_Ind_recurrent_unit_index_start(static_cast<size_t>(tmp_ptr_AF_Ind_recurrent_unit_it - this->ptr_array_AF_Ind_recurrent_units));

        T_ *tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs(this->ptr_array_AF_Ind_recurrent_units_pre_AFs + tmp_AF_Ind_recurrent_unit_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_AF_Ind_recurrent_units_AFs(this->ptr_array_AF_Ind_recurrent_units_AFs + tmp_AF_Ind_recurrent_unit_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_AF_Ind_recurrent_units_errors(this->ptr_array_AF_Ind_recurrent_units_errors + tmp_AF_Ind_recurrent_unit_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_AF_Ind_recurrent_units_dAFs(this->ptr_array_AF_Ind_recurrent_units_dAFs + tmp_AF_Ind_recurrent_unit_index_start * this->batch_size * this->number_recurrent_depth);
        
        for(; tmp_ptr_AF_Ind_recurrent_unit_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_AF_Ind_recurrent_unit_it,
                                                                                        ++tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs,
                                                                                        ++tmp_ptr_array_AF_Ind_recurrent_units_AFs,
                                                                                        ++tmp_ptr_array_AF_Ind_recurrent_units_errors,
                                                                                        ++tmp_ptr_array_AF_Ind_recurrent_units_dAFs)
        {
            tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_pre_AFs = tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs;
            tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_AFs = tmp_ptr_array_AF_Ind_recurrent_units_AFs;
            tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_errors = tmp_ptr_array_AF_Ind_recurrent_units_errors;
            tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_dAFs = tmp_ptr_array_AF_Ind_recurrent_units_dAFs;
        }
    }
}

void Neural_Network::Order__Layer__AF_Ind_Recurrent_Unit__Dropout_Bernoulli(struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_AF_Ind_recurrent_unit_index_start(static_cast<size_t>(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units - this->ptr_array_AF_Ind_recurrent_units));

    ptr_layer_it_received->ptr_array__mask__dropout__bernoulli = this->ptr_array_units_mask_dropout_bernoulli + (this->total_AF_units_allocated + tmp_AF_Ind_recurrent_unit_index_start) * this->number_recurrent_depth;
}

void Neural_Network::Order__Layer__LSTM(struct Layer *const ptr_layer_it_received)
{
    this->Order__Layer__Block_Unit(ptr_layer_it_received);

    if(this->Use__Dropout__Zoneout()) { this->Order__Layer__Block_Unit__Dropout_Zoneout(ptr_layer_it_received); }
    
    if(ptr_layer_it_received->Use__Normalization()
      &&
      ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM)
    { this->Order__Layer__Normalization_Iterator(ptr_layer_it_received); }

    // TODO: Transpose per layers.
    if(ptr_layer_it_received->Use__K_Sparsity()) { this->Assign__Sparsity_Activities(this->number_threads); }
}

void Neural_Network::Order__Layer__Block_Unit(struct Layer *const ptr_layer_it_received)
{
    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);
    
    if(static_cast<size_t>(tmp_ptr_last_block_unit - tmp_ptr_block_unit_it) != 0_zu)
    {
        struct Cell_unit const *tmp_ptr_last_cell_unit;
        struct Cell_unit *tmp_ptr_cell_unit_it;
        
        size_t const tmp_block_index_start(static_cast<size_t>(tmp_ptr_block_unit_it - this->ptr_array_layers->ptr_array_block_units)),
                           tmp_cell_index_start(static_cast<size_t>(ptr_layer_it_received->ptr_array_cell_units - this->ptr_array_layers->ptr_array_cell_units));

        T_ *tmp_ptr_array_summation_cells_inputs(this->ptr_array_cells_summations_cells_inputs + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_input_cells_inputs(this->ptr_array_cells_summations_input_cells_inputs + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_recurrent_cells_inputs(this->ptr_array_cells_summations_recurrent_cells_inputs + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_inputs_gates(this->ptr_array_blocks_summations_inputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_input_inputs_gates(this->ptr_array_blocks_summations_input_inputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_recurrent_inputs_gates(this->ptr_array_blocks_summations_recurrent_inputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_forgets_gates(this->ptr_array_blocks_summations_forgets_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_input_forgets_gates(this->ptr_array_blocks_summations_input_forgets_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_recurrent_forgets_gates(this->ptr_array_blocks_summations_recurrent_forgets_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_outputs_gates(this->ptr_array_blocks_summations_outputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_input_outputs_gates(this->ptr_array_blocks_summations_input_outputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_summation_recurrent_outputs_gates(this->ptr_array_blocks_summations_recurrent_outputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_cells_inputs(this->ptr_array_cells_inputs + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_cells_states(this->ptr_array_cells_states + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_cells_states_activates(this->ptr_array_cells_states_activates + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_cells_outputs(this->ptr_array_cells_outputs + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_inputs_gates(this->ptr_array_blocks_inputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_forgets_gates(this->ptr_array_blocks_forgets_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_outputs_gates(this->ptr_array_blocks_outputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_cells_inputs(this->ptr_array_cells_delta_inputs + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_cells_input_inputs(this->ptr_array_cells_delta_input_inputs + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_cells_recurrent_inputs(this->ptr_array_cells_delta_recurrent_inputs + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_cells_states(this->ptr_array_cells_delta_states + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_cells_outputs(this->ptr_array_cells_delta_outputs + tmp_cell_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_inputs_gates(this->ptr_array_blocks_delta_inputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_input_inputs_gates(this->ptr_array_blocks_delta_input_inputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_recurrent_inputs_gates(this->ptr_array_blocks_delta_recurrent_inputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_forgets_gates(this->ptr_array_blocks_delta_forgets_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_input_forgets_gates(this->ptr_array_blocks_delta_input_forgets_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_recurrent_forgets_gates(this->ptr_array_blocks_delta_recurrent_forgets_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_outputs_gates(this->ptr_array_blocks_delta_outputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_input_outputs_gates(this->ptr_array_blocks_delta_input_outputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_delta_recurrent_outputs_gates(this->ptr_array_blocks_delta_recurrent_outputs_gates + tmp_block_index_start * this->batch_size * this->number_recurrent_depth);
        
        for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                       ++tmp_ptr_array_summation_inputs_gates,
                                                                                       ++tmp_ptr_array_summation_input_inputs_gates,
                                                                                       ++tmp_ptr_array_summation_recurrent_inputs_gates,
                                                                                       ++tmp_ptr_array_summation_forgets_gates,
                                                                                       ++tmp_ptr_array_summation_input_forgets_gates,
                                                                                       ++tmp_ptr_array_summation_recurrent_forgets_gates,
                                                                                       ++tmp_ptr_array_summation_outputs_gates,
                                                                                       ++tmp_ptr_array_summation_input_outputs_gates,
                                                                                       ++tmp_ptr_array_summation_recurrent_outputs_gates,
                                                                                       ++tmp_ptr_array_inputs_gates,
                                                                                       ++tmp_ptr_array_forgets_gates,
                                                                                       ++tmp_ptr_array_outputs_gates,
                                                                                       ++tmp_ptr_array_delta_inputs_gates,
                                                                                       ++tmp_ptr_array_delta_input_inputs_gates,
                                                                                       ++tmp_ptr_array_delta_recurrent_inputs_gates,
                                                                                       ++tmp_ptr_array_delta_forgets_gates,
                                                                                       ++tmp_ptr_array_delta_input_forgets_gates,
                                                                                       ++tmp_ptr_array_delta_recurrent_forgets_gates,
                                                                                       ++tmp_ptr_array_delta_outputs_gates,
                                                                                       ++tmp_ptr_array_delta_input_outputs_gates,
                                                                                       ++tmp_ptr_array_delta_recurrent_outputs_gates)
        {
            tmp_ptr_block_unit_it->ptr_array_summation_cells_inputs = tmp_ptr_array_summation_cells_inputs;
            tmp_ptr_block_unit_it->ptr_array_summation_input_cells_inputs = tmp_ptr_array_summation_input_cells_inputs;
            tmp_ptr_block_unit_it->ptr_array_summation_recurrent_cells_inputs = tmp_ptr_array_summation_recurrent_cells_inputs;
            tmp_ptr_block_unit_it->ptr_summation_inputs_gates = tmp_ptr_array_summation_inputs_gates;
            tmp_ptr_block_unit_it->ptr_summation_input_inputs_gates = tmp_ptr_array_summation_input_inputs_gates;
            tmp_ptr_block_unit_it->ptr_summation_recurrent_inputs_gates = tmp_ptr_array_summation_recurrent_inputs_gates;
            tmp_ptr_block_unit_it->ptr_summation_forgets_gates = tmp_ptr_array_summation_forgets_gates;
            tmp_ptr_block_unit_it->ptr_summation_input_forgets_gates = tmp_ptr_array_summation_input_forgets_gates;
            tmp_ptr_block_unit_it->ptr_summation_recurrent_forgets_gates = tmp_ptr_array_summation_recurrent_forgets_gates;
            tmp_ptr_block_unit_it->ptr_summation_outputs_gates = tmp_ptr_array_summation_outputs_gates;
            tmp_ptr_block_unit_it->ptr_summation_input_outputs_gates = tmp_ptr_array_summation_input_outputs_gates;
            tmp_ptr_block_unit_it->ptr_summation_recurrent_outputs_gates = tmp_ptr_array_summation_recurrent_outputs_gates;
            tmp_ptr_block_unit_it->ptr_array_cells_inputs = tmp_ptr_array_cells_inputs;
            tmp_ptr_block_unit_it->ptr_array_cells_states = tmp_ptr_array_cells_states;
            tmp_ptr_block_unit_it->ptr_array_cells_states_activates = tmp_ptr_array_cells_states_activates;
            tmp_ptr_block_unit_it->ptr_array_cells_outputs = tmp_ptr_array_cells_outputs;
            tmp_ptr_block_unit_it->ptr_inputs_gates = tmp_ptr_array_inputs_gates;
            tmp_ptr_block_unit_it->ptr_forgets_gates = tmp_ptr_array_forgets_gates;
            tmp_ptr_block_unit_it->ptr_outputs_gates = tmp_ptr_array_outputs_gates;
            tmp_ptr_block_unit_it->ptr_array_delta_cells_inputs = tmp_ptr_array_delta_cells_inputs;
            tmp_ptr_block_unit_it->ptr_array_delta_cells_input_inputs = tmp_ptr_array_delta_cells_input_inputs;
            tmp_ptr_block_unit_it->ptr_array_delta_cells_recurrent_inputs = tmp_ptr_array_delta_cells_recurrent_inputs;
            tmp_ptr_block_unit_it->ptr_array_delta_cells_states = tmp_ptr_array_delta_cells_states;
            tmp_ptr_block_unit_it->ptr_array_delta_cells_outputs = tmp_ptr_array_delta_cells_outputs;
            tmp_ptr_block_unit_it->ptr_delta_inputs_gates = tmp_ptr_array_delta_inputs_gates;
            tmp_ptr_block_unit_it->ptr_delta_input_inputs_gates = tmp_ptr_array_delta_input_inputs_gates;
            tmp_ptr_block_unit_it->ptr_delta_recurrent_inputs_gates = tmp_ptr_array_delta_recurrent_inputs_gates;
            tmp_ptr_block_unit_it->ptr_delta_forgets_gates = tmp_ptr_array_delta_forgets_gates;
            tmp_ptr_block_unit_it->ptr_delta_input_forgets_gates = tmp_ptr_array_delta_input_forgets_gates;
            tmp_ptr_block_unit_it->ptr_delta_recurrent_forgets_gates = tmp_ptr_array_delta_recurrent_forgets_gates;
            tmp_ptr_block_unit_it->ptr_delta_outputs_gates = tmp_ptr_array_delta_outputs_gates;
            tmp_ptr_block_unit_it->ptr_delta_input_outputs_gates = tmp_ptr_array_delta_input_outputs_gates;
            tmp_ptr_block_unit_it->ptr_delta_recurrent_outputs_gates = tmp_ptr_array_delta_recurrent_outputs_gates;
                
            for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                   ++tmp_ptr_array_summation_cells_inputs,
                                                                                                                                                                                   ++tmp_ptr_array_summation_input_cells_inputs,
                                                                                                                                                                                   ++tmp_ptr_array_summation_recurrent_cells_inputs,
                                                                                                                                                                                   ++tmp_ptr_array_cells_inputs,
                                                                                                                                                                                   ++tmp_ptr_array_cells_states,
                                                                                                                                                                                   ++tmp_ptr_array_cells_states_activates,
                                                                                                                                                                                   ++tmp_ptr_array_cells_outputs,
                                                                                                                                                                                   ++tmp_ptr_array_delta_cells_inputs,
                                                                                                                                                                                   ++tmp_ptr_array_delta_cells_input_inputs,
                                                                                                                                                                                   ++tmp_ptr_array_delta_cells_recurrent_inputs,
                                                                                                                                                                                   ++tmp_ptr_array_delta_cells_states,
                                                                                                                                                                                   ++tmp_ptr_array_delta_cells_outputs)
            {
                tmp_ptr_cell_unit_it->ptr_summation_cell_input = tmp_ptr_array_summation_cells_inputs;
                tmp_ptr_cell_unit_it->ptr_summation_input_cell_input = tmp_ptr_array_summation_input_cells_inputs;
                tmp_ptr_cell_unit_it->ptr_summation_recurrent_cell_input = tmp_ptr_array_summation_recurrent_cells_inputs;
                tmp_ptr_cell_unit_it->ptr_cell_input = tmp_ptr_array_cells_inputs;
                tmp_ptr_cell_unit_it->ptr_cell_state = tmp_ptr_array_cells_states;
                tmp_ptr_cell_unit_it->ptr_cell_state_activate = tmp_ptr_array_cells_states_activates;
                tmp_ptr_cell_unit_it->ptr_cell_output = tmp_ptr_array_cells_outputs;
                tmp_ptr_cell_unit_it->ptr_delta_cell_input = tmp_ptr_array_delta_cells_inputs;
                tmp_ptr_cell_unit_it->ptr_delta_cell_input_input = tmp_ptr_array_delta_cells_input_inputs;
                tmp_ptr_cell_unit_it->ptr_delta_cell_recurrent_input = tmp_ptr_array_delta_cells_recurrent_inputs;
                tmp_ptr_cell_unit_it->ptr_delta_cell_state = tmp_ptr_array_delta_cells_states;
                tmp_ptr_cell_unit_it->ptr_delta_cell_output = tmp_ptr_array_delta_cells_outputs;
            }
        }
    }
}

void Neural_Network::Order__Layer__Block_Unit__Dropout_Zoneout(struct Layer *const ptr_layer_it_received)
{
    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);
    
    if(static_cast<size_t>(tmp_ptr_last_block_unit - tmp_ptr_block_unit_it) != 0_zu)
    {
        struct Cell_unit const *tmp_ptr_last_cell_unit;
        struct Cell_unit *tmp_ptr_cell_unit_it;
        
        size_t const tmp_cell_index_start(static_cast<size_t>(ptr_layer_it_received->ptr_array_cell_units - this->ptr_array_layers->ptr_array_cell_units));
        
        bool *tmp_ptr_array_cell_units_mask_dropout_zoneout(this->ptr_array_cell_units_mask_dropout_zoneout + tmp_cell_index_start * this->number_recurrent_depth);
        
        for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
        {
            tmp_ptr_block_unit_it->ptr_array_mask_dropout_zoneout = tmp_ptr_array_cell_units_mask_dropout_zoneout;

            for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                   ++tmp_ptr_array_cell_units_mask_dropout_zoneout)
            {
                tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state = tmp_ptr_array_cell_units_mask_dropout_zoneout;
                tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output = tmp_ptr_array_cell_units_mask_dropout_zoneout + this->number_recurrent_depth * this->total_cell_units_allocated;
            }
        }
    }
}

void Neural_Network::Order__Layer__Normalization_Iterator(struct Layer *const ptr_layer_it_received)
{
    union Normalized_unit const *const tmp_ptr_last_normalized_unit(ptr_layer_it_received->ptr_last_normalized_unit);
    union Normalized_unit *tmp_ptr_normalized_unit_it(ptr_layer_it_received->ptr_array_normalized_units);
    
    if(static_cast<size_t>(tmp_ptr_last_normalized_unit - tmp_ptr_normalized_unit_it) != 0_zu)
    {
        struct Block_unit const *tmp_ptr_last_block_unit;
        struct Block_unit *tmp_ptr_block_unit_it;
        
        struct Cell_unit const *tmp_ptr_last_cell_unit;
        struct Cell_unit *tmp_ptr_cell_unit_it;
        
        switch(ptr_layer_it_received->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL: break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units) != 0_zu)
                {
                    for(tmp_ptr_last_block_unit = ptr_layer_it_received->ptr_last_block_unit,
                        tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
                    {
                        // [0]: Block input, input.
                        // [1]: Block input, recurrent.
                        // [2]: Cell state activate.

                        for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                            tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                        {
                            tmp_ptr_cell_unit_it->ptr_array_normalized_units = tmp_ptr_normalized_unit_it;
                            tmp_ptr_normalized_unit_it += 3;
                            tmp_ptr_cell_unit_it->ptr_last_normalized_unit = tmp_ptr_normalized_unit_it;
                        }
                        
                        // [3]: Input gate, input.
                        // [4]: Input gate, recurrent.
                        // [5]: Forget gate, input.
                        // [6]: Forget gate, recurrent.
                        // [7]: Output gate, input.
                        // [8]: Output gate, recurrent.

                        tmp_ptr_block_unit_it->ptr_array_normalized_units = tmp_ptr_normalized_unit_it;
                        tmp_ptr_normalized_unit_it += 6;
                        tmp_ptr_block_unit_it->ptr_last_normalized_unit = tmp_ptr_normalized_unit_it;
                    }
                }
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         ptr_layer_it_received->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str(),
                                         __LINE__);
                    return;
        }
    }
}

void Neural_Network::Order__Layer__Normalization(struct Layer *const ptr_layer_it_received)
{
    this->Order__Layer__Batch_Normalization(ptr_layer_it_received);

    if(ptr_layer_it_received->Use__Batch_Renormalization()) { this->Order__Layer__Batch_Renormalization(ptr_layer_it_received); }
}

void Neural_Network::Order__Layer__Batch_Normalization(struct Layer *const ptr_layer_it_received)
{
    union Normalized_unit const *const tmp_ptr_last_normalized_unit(ptr_layer_it_received->ptr_last_normalized_unit);
    union Normalized_unit *tmp_ptr_normalized_unit_it(ptr_layer_it_received->ptr_array_normalized_units);
    
    if(static_cast<size_t>(tmp_ptr_last_normalized_unit - tmp_ptr_normalized_unit_it) != 0_zu)
    {
        size_t const tmp_normalized_unit_index_start(static_cast<size_t>(ptr_layer_it_received->ptr_array_normalized_units - this->ptr_array_normalized_units));

        T_ *tmp_ptr_array_parameters_scale_it(this->ptr_array_parameters + this->total_weights_allocated + this->total_bias_allocated + tmp_normalized_unit_index_start),
             *tmp_ptr_array_parameters_shift_it(this->ptr_array_parameters + this->total_weights_allocated + this->total_bias_allocated + this->total_normalized_units_allocated + tmp_normalized_unit_index_start),
             *tmp_ptr_array_derivatives_parameters_scale_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated + this->total_bias_allocated + tmp_normalized_unit_index_start),
             *tmp_ptr_array_derivatives_parameters_shift_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated + this->total_bias_allocated + this->total_normalized_units_allocated + tmp_normalized_unit_index_start),
             *tmp_ptr_array_normalized_units_values_hat(this->ptr_array_normalized_batch_units_values_hats + tmp_normalized_unit_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_normalized_units_values_normalize(this->ptr_array_normalized_batch_units_values_normalizes + tmp_normalized_unit_index_start * this->batch_size * this->number_recurrent_depth),
             *tmp_ptr_array_normalized_units_mean_it(this->ptr_array_normalized_batch_units_means + tmp_normalized_unit_index_start * this->number_threads * this->number_recurrent_depth),
             *tmp_ptr_array_normalized_units_variance_it(this->ptr_array_normalized_batch_units_variances + tmp_normalized_unit_index_start * this->number_threads * this->number_recurrent_depth),
             *tmp_ptr_array_normalized_units_derivative_mean_it(this->ptr_array_normalized_batch_units_derivatives_means + tmp_normalized_unit_index_start * this->number_threads * this->number_recurrent_depth),
             *tmp_ptr_array_normalized_units_derivative_variance_it(this->ptr_array_normalized_batch_units_derivatives_variances + tmp_normalized_unit_index_start * this->number_threads * this->number_recurrent_depth),
             *tmp_ptr_array_normalized_units_mean_average_it(this->ptr_array_normalized_batch_units_means_averages + tmp_normalized_unit_index_start * this->number_recurrent_depth),
             *tmp_ptr_array_normalized_units_variance_average_it(this->ptr_array_normalized_batch_units_variances_averages + tmp_normalized_unit_index_start * this->number_recurrent_depth),
             *tmp_ptr_array_normalized_units_errors(this->ptr_array_normalized_batch_units_errors + tmp_normalized_unit_index_start * this->batch_size * this->number_recurrent_depth);
        
        for(; tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit; ++tmp_ptr_normalized_unit_it,
                                                                                                       ++tmp_ptr_array_parameters_scale_it,
                                                                                                       ++tmp_ptr_array_parameters_shift_it,
                                                                                                       ++tmp_ptr_array_derivatives_parameters_scale_it,
                                                                                                       ++tmp_ptr_array_derivatives_parameters_shift_it,
                                                                                                       ++tmp_ptr_array_normalized_units_values_hat,
                                                                                                       ++tmp_ptr_array_normalized_units_values_normalize,
                                                                                                       ++tmp_ptr_array_normalized_units_mean_it,
                                                                                                       ++tmp_ptr_array_normalized_units_variance_it,
                                                                                                       ++tmp_ptr_array_normalized_units_derivative_mean_it,
                                                                                                       ++tmp_ptr_array_normalized_units_derivative_variance_it,
                                                                                                       ++tmp_ptr_array_normalized_units_mean_average_it,
                                                                                                       ++tmp_ptr_array_normalized_units_variance_average_it,
                                                                                                       ++tmp_ptr_array_normalized_units_errors)
        {
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_scale = tmp_ptr_array_parameters_scale_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_shift = tmp_ptr_array_parameters_shift_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_values_hats = tmp_ptr_array_normalized_units_values_hat;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_values_normalizes = tmp_ptr_array_normalized_units_values_normalize;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_mean_average = tmp_ptr_array_normalized_units_mean_average_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_variance_average = tmp_ptr_array_normalized_units_variance_average_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors;
        }
    }
}

void Neural_Network::Order__Layer__Batch_Renormalization(struct Layer *const ptr_layer_it_received)
{
    union Normalized_unit const *const tmp_ptr_last_normalized_unit(ptr_layer_it_received->ptr_last_normalized_unit);
    union Normalized_unit *tmp_ptr_normalized_unit_it(ptr_layer_it_received->ptr_array_normalized_units);
    
    if(static_cast<size_t>(tmp_ptr_last_normalized_unit - tmp_ptr_normalized_unit_it) != 0_zu)
    {
        size_t const tmp_normalized_unit_index_start(static_cast<size_t>(ptr_layer_it_received->ptr_array_normalized_units - this->ptr_array_normalized_units));

        T_ *tmp_ptr_array_normalized_units_r_correction_it(this->ptr_array_normalized_batch_units_r_corrections + tmp_normalized_unit_index_start * this->number_recurrent_depth),
             *tmp_ptr_array_normalized_units_d_correction_it(this->ptr_array_normalized_batch_units_d_corrections + tmp_normalized_unit_index_start * this->number_recurrent_depth);
        
        for(; tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit; ++tmp_ptr_normalized_unit_it,
                                                                                                       ++tmp_ptr_array_normalized_units_r_correction_it,
                                                                                                       ++tmp_ptr_array_normalized_units_d_correction_it)
        {
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_r_correction = tmp_ptr_array_normalized_units_r_correction_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_d_correction = tmp_ptr_array_normalized_units_d_correction_it;
        }
    }
}
