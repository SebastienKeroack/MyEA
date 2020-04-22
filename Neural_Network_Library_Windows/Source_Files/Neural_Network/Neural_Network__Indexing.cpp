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

bool Neural_Network::Set__Pre_Training_Level(size_t const pre_training_level_received)
{
    if(this->pre_training_level == pre_training_level_received) { return(true); }
    else if(pre_training_level_received > (this->total_layers - 3_zu) / 2_zu + 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Pre training level (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 pre_training_level_received,
                                 (this->total_layers - 3_zu) / 2_zu + 1_zu,
                                 __LINE__);

        return(false);
    }

    size_t const tmp_past_pre_training_level(this->pre_training_level);

    this->pre_training_level = pre_training_level_received;

    if((tmp_past_pre_training_level == 0_zu && pre_training_level_received != 0_zu)
      ||
      (tmp_past_pre_training_level != 0_zu && pre_training_level_received == 0_zu))
    { this->Order__Layers__Output(); }

    if(this->Use__Regularization_Parameter())
    {
        if(this->pre_training_level != 0_zu) { this->Indexing_Regularization_Parameters__Pre_training(); }
        else { this->Indexing_Regularization_Parameters(); }
    }

    return(true);
}

void Neural_Network::Indexing_Regularization_Parameters(void)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                               *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(1_T, tmp_ptr_layer_it);
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(1_T, tmp_ptr_layer_it); break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Indexing_Regularization__Weights__LSTM(1_T, tmp_ptr_layer_it); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    break;
        }

        this->Indexing_Regularization__Bias(0_T, tmp_ptr_layer_it);
    }
}

void Neural_Network::Indexing_Regularization_Parameters__Pre_training(void)
{
    if(this->pre_training_level == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The neural network use the pre-training function without the mode pre-training activate. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return;
    }

    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                               *const tmp_ptr_input_layer(this->ptr_array_layers + this->pre_training_level),
                               *const tmp_ptr_output_layer(this->Get__Output_Layer()),
                               *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
    // First layer to coded layer, Mask zero.
    for(; tmp_ptr_layer_it < tmp_ptr_input_layer; ++tmp_ptr_layer_it)
    {
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(0_T, tmp_ptr_layer_it);
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(0_T, tmp_ptr_layer_it); break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Indexing_Regularization__Weights__LSTM(0_T, tmp_ptr_layer_it); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    break;
        }

        this->Indexing_Regularization__Bias(0_T, tmp_ptr_layer_it);
    }
    
    // Coded layer, Mask one.
    switch(tmp_ptr_layer_it->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(1_T, tmp_ptr_layer_it);
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(1_T, tmp_ptr_layer_it); break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Indexing_Regularization__Weights__LSTM(1_T, tmp_ptr_layer_it); break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                        MyEA::Time::Date_Time_Now().c_str(),
                                        __FUNCTION__,
                                        tmp_ptr_layer_it->type_layer,
                                        MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                break;
    }

    this->Indexing_Regularization__Bias(0_T, tmp_ptr_layer_it);
    // |END| Coded layer, Mask one. |END|
    
    // Coded layer to output layer, Mask zero.
    for(++tmp_ptr_layer_it; tmp_ptr_layer_it < tmp_ptr_output_layer; ++tmp_ptr_layer_it)
    {
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(0_T, tmp_ptr_layer_it);
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(0_T, tmp_ptr_layer_it); break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Indexing_Regularization__Weights__LSTM(0_T, tmp_ptr_layer_it); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    break;
        }

        this->Indexing_Regularization__Bias(0_T, tmp_ptr_layer_it);
    }
    
    // Output layer, Mask one.
    switch(tmp_ptr_layer_it->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(1_T, tmp_ptr_layer_it);
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(1_T, tmp_ptr_layer_it); break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Indexing_Regularization__Weights__LSTM(1_T, tmp_ptr_layer_it); break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                        MyEA::Time::Date_Time_Now().c_str(),
                                        __FUNCTION__,
                                        tmp_ptr_layer_it->type_layer,
                                        MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                break;
    }

    this->Indexing_Regularization__Bias(0_T, tmp_ptr_layer_it);
    // |END| Output layer, Mask one. |END|
    
    // Output layer to last layer, Mask zero.
    for(++tmp_ptr_layer_it; tmp_ptr_layer_it < tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(0_T, tmp_ptr_layer_it);
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(0_T, tmp_ptr_layer_it); break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Indexing_Regularization__Weights__LSTM(0_T, tmp_ptr_layer_it); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    break;
        }

        this->Indexing_Regularization__Bias(0_T, tmp_ptr_layer_it);
    }
}

void Neural_Network::Indexing_Regularization__Weights__FC__Forward(T_ const mask_received, struct Layer const *const ptr_layer_it_received)
{
    struct Neuron_unit const *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit),
                                         *tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);

    T_ const *tmp_ptr_last_mask_regularization;
    T_ *tmp_ptr_mask_regularization_it;
    
    for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
    {
        tmp_ptr_mask_regularization_it = this->ptr_array_mask_regularized_parameters + *tmp_ptr_neuron_unit_it->ptr_first_connection_index;
        tmp_ptr_last_mask_regularization = this->ptr_array_mask_regularized_parameters + *tmp_ptr_neuron_unit_it->ptr_last_connection_index;

        for(; tmp_ptr_mask_regularization_it != tmp_ptr_last_mask_regularization; ++tmp_ptr_mask_regularization_it) { *tmp_ptr_mask_regularization_it = mask_received; }
    }
}

void Neural_Network::Indexing_Regularization__Weights__AF_Ind_Recurrent(T_ const mask_received, struct Layer const *const ptr_layer_it_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_number_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit));

    T_ *tmp_ptr_array_mask_regularization_it(this->ptr_array_mask_regularized_parameters + *tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index);
    T_ const *const tmp_ptr_array_mask_regularization_end(tmp_ptr_array_mask_regularization_it + tmp_number_units);
    
    for(; tmp_ptr_array_mask_regularization_it != tmp_ptr_array_mask_regularization_end; ++tmp_ptr_array_mask_regularization_it) { *tmp_ptr_array_mask_regularization_it = mask_received; }
}

void Neural_Network::Indexing_Regularization__Weights__LSTM(T_ const mask_received, struct Layer const *const ptr_layer_it_received)
{
    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit),
                                      *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);
    
    size_t const tmp_number_peephole_connections(tmp_ptr_block_unit_it->last_index_peephole_input_gate - tmp_ptr_block_unit_it->first_index_peephole_input_gate),
                       tmp_number_feedforward_connections(tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrent_connections(tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index;
    
    struct Cell_unit const *tmp_ptr_block_ptr_last_cell_unit,
                                    *tmp_ptr_block_ptr_cell_unit_it;

    T_ *tmp_ptr_array_cell_input_regularized_connections,
         *tmp_ptr_array_input_gate_regularized_connections,
         *tmp_ptr_array_forget_gate_regularized_connections,
         *tmp_ptr_array_output_gate_regularized_connections;

    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        // [0] Cell input.
        for(tmp_ptr_block_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
            tmp_ptr_block_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
        {
            //    [1] Input, cell input.
            tmp_ptr_array_cell_input_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;
            
            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index) { tmp_ptr_array_cell_input_regularized_connections[tmp_connection_index] = mask_received; }
            //    [1] |END| Input, cell input. |END|
            
            //    [1] Recurrent, cell input.
            tmp_ptr_array_cell_input_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;
            
            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index) { tmp_ptr_array_cell_input_regularized_connections[tmp_connection_index] = mask_received; }
            //    [1] |END| Recurrent, cell input. |END|

        }
        // [0] |END| Cell input. |END|
        
        // Input, gates.
        tmp_ptr_array_input_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
        tmp_ptr_array_forget_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
        tmp_ptr_array_output_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;
        
        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_forget_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_output_gate_regularized_connections[tmp_connection_index] = mask_received;
        }
        // |END| Input, gates. |END|
        
        // Recurrent, gates.
        tmp_ptr_array_input_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;
        tmp_ptr_array_forget_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;
        tmp_ptr_array_output_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;
        
        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_forget_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_output_gate_regularized_connections[tmp_connection_index] = mask_received;
        }
        // |END| Recurrent, gates. |END|
        
    #ifndef NO_PEEPHOLE
        // [0] Peepholes.
        tmp_ptr_array_input_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;
        tmp_ptr_array_forget_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;
        tmp_ptr_array_output_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;
        
        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_forget_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_output_gate_regularized_connections[tmp_connection_index] = mask_received;
        }
        // [0] |END| Peepholes. |END|
    #endif
    }
}

void Neural_Network::Indexing_Regularization__Bias(T_ const mask_received, struct Layer const *const ptr_layer_it_received)
{
    T_ const *const tmp_ptr_array_mask_regularization_end(this->ptr_array_mask_regularized_parameters + ptr_layer_it_received->last_bias_connection_index);
    T_ *tmp_ptr_array_mask_regularization_it(this->ptr_array_mask_regularized_parameters + ptr_layer_it_received->first_bias_connection_index);
    
    for(; tmp_ptr_array_mask_regularization_it != tmp_ptr_array_mask_regularization_end; ++tmp_ptr_array_mask_regularization_it) { *tmp_ptr_array_mask_regularization_it = mask_received; }
}
