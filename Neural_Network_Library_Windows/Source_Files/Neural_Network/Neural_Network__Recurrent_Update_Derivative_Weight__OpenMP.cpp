#include "stdafx.hpp"

#include <Math/Mathematic.hpp>

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::RNN__Update_Derivative_Weight_Batch__OpenMP(size_t const batch_size_received,
                                                                                                           struct Layer *ptr_layer_it_received,
                                                                                                           struct Layer const *const ptr_layer_end_received)
{
    size_t tmp_number_units[5u];

    struct Layer const *tmp_ptr_previous_connected_layer;

    for(; ptr_layer_it_received != ptr_layer_end_received; ++ptr_layer_it_received)
    {
        // If the current layer is a pooling/residual layer, continue.
        if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
          ||
          ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING
          ||
          ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { continue; }
        
        tmp_ptr_previous_connected_layer = ptr_layer_it_received->previous_connected_layers[0u];

        switch(ptr_layer_it_received->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Recurrent__Update_Derivative_Weight__FC__OpenMP(batch_size_received,
                                                                                                        *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                        tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                        ptr_layer_it_received);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                tmp_number_units[0u] = static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units);
                tmp_number_units[1u] = static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units);

                if(ptr_layer_it_received->Use__Bidirectional())
                {
                    tmp_number_units[2u] = tmp_number_units[0u] >> 1_zu;
                    tmp_number_units[3u] = tmp_number_units[1u] >> 1_zu;

                    this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                                batch_size_received,
                                                                                                                tmp_number_units[2u],
                                                                                                                tmp_number_units[3u],
                                                                                                                *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                                tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                                &ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer);
                    this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size_received,
                                                                                                                          tmp_number_units[2u],
                                                                                                                          tmp_number_units[3u],
                                                                                                                          ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                          ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                          ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                          ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index,
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3u],
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3u] + tmp_number_units[2u],
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3u] + 2_zu * tmp_number_units[2u]);
                    this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(false,
                                                                                                                batch_size_received,
                                                                                                                tmp_number_units[2u],
                                                                                                                tmp_number_units[3u],
                                                                                                                *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                                tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                                ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                                &ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer);
                    this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size_received,
                                                                                                                          tmp_number_units[2u],
                                                                                                                          tmp_number_units[3u],
                                                                                                                          ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                          ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                          ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                          ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index,
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3u],
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3u] + tmp_number_units[2u],
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3u] + 2_zu * tmp_number_units[2u]);
                }
                else
                {
                    this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                                batch_size_received,
                                                                                                                tmp_number_units[0u],
                                                                                                                tmp_number_units[1u],
                                                                                                                *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                                tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                                ptr_layer_it_received->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                ptr_layer_it_received->Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                                ptr_layer_it_received->Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                                ptr_layer_it_received->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                ptr_layer_it_received->Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                                ptr_layer_it_received->Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                                ptr_layer_it_received->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                ptr_layer_it_received->Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                                ptr_layer_it_received->Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                                ptr_layer_it_received->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                ptr_layer_it_received->Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                                ptr_layer_it_received->Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                                ptr_layer_it_received);
                    this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size_received,
                                                                                                                          tmp_number_units[0u],
                                                                                                                          tmp_number_units[1u],
                                                                                                                          ptr_layer_it_received->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                          ptr_layer_it_received->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                          ptr_layer_it_received->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                          ptr_layer_it_received->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->first_bias_connection_index,
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->first_bias_connection_index + tmp_number_units[1u],
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->first_bias_connection_index + tmp_number_units[1u] + tmp_number_units[0u],
                                                                                                                          this->ptr_array_derivatives_parameters + ptr_layer_it_received->first_bias_connection_index + tmp_number_units[1u] + 2_zu * tmp_number_units[0u]);
                }
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         ptr_layer_it_received->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str());
                    return;
        }
    }
}

void Neural_Network::RNN__Update_Derivative_Weight_Batch__Pre_Training__OpenMP(size_t const batch_size_received)
{
    size_t tmp_number_units[4u];

    struct Layer const *tmp_ptr_previous_connected_layer;
    struct Layer *tmp_ptr_layer_it;
    
    // Coded level part.
    tmp_ptr_layer_it = this->ptr_array_layers + this->pre_training_level;
    tmp_ptr_previous_connected_layer = tmp_ptr_layer_it->previous_connected_layers[0u];
    
    switch(tmp_ptr_layer_it->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Update_Derivative_Weight__FC__OpenMP(batch_size_received,
                                                                                                    *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                    tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                    tmp_ptr_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            tmp_number_units[0u] = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units);
            tmp_number_units[1u] = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units);

            if(tmp_ptr_layer_it->Use__Bidirectional())
            {
                tmp_number_units[2u] = tmp_number_units[0u] >> 1_zu;
                tmp_number_units[3u] = tmp_number_units[1u] >> 1_zu;

                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                            batch_size_received,
                                                                                                            tmp_number_units[2u],
                                                                                                            tmp_number_units[3u],
                                                                                                            *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                            tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            &tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size_received,
                                                                                                                       tmp_number_units[2u],
                                                                                                                       tmp_number_units[3u],
                                                                                                                       tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                       tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                       tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                       tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                       this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index,
                                                                                                                       this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3u],
                                                                                                                       this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3u] + tmp_number_units[2u],
                                                                                                                       this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3u] + 2_zu * tmp_number_units[2u]);
                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(false,
                                                                                                            batch_size_received,
                                                                                                            tmp_number_units[2u],
                                                                                                            tmp_number_units[3u],
                                                                                                            *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                            tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            &tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size_received,
                                                                                                                        tmp_number_units[2u],
                                                                                                                        tmp_number_units[3u],
                                                                                                                        tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                        tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                        tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                        tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index,
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3u],
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3u] + tmp_number_units[2u],
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3u] + 2_zu * tmp_number_units[2u]);
            }
            else
            {
                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                            batch_size_received,
                                                                                                            tmp_number_units[0u],
                                                                                                            tmp_number_units[1u],
                                                                                                            *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                            tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                            tmp_ptr_layer_it->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size_received,
                                                                                                                        tmp_number_units[0u],
                                                                                                                        tmp_number_units[1u],
                                                                                                                        tmp_ptr_layer_it->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                        tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                        tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                        tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->first_bias_connection_index,
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->first_bias_connection_index + tmp_number_units[1u],
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->first_bias_connection_index + tmp_number_units[1u] + tmp_number_units[0u],
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->first_bias_connection_index + tmp_number_units[1u] + 2_zu * tmp_number_units[0u]);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_layer_it->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                return;
    }
    // |END| Coded level part. |END|

    // Decode level part.
    tmp_ptr_previous_connected_layer = tmp_ptr_layer_it;
    tmp_ptr_layer_it = this->ptr_last_layer - static_cast<size_t>(tmp_ptr_layer_it - this->ptr_array_layers);
    
    switch(tmp_ptr_layer_it->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Update_Derivative_Weight__FC__OpenMP(batch_size_received,
                                                                                                    *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                    tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                    tmp_ptr_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            tmp_number_units[0u] = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units);
            tmp_number_units[1u] = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units);

            if(tmp_ptr_layer_it->Use__Bidirectional())
            {
                tmp_number_units[2u] = tmp_number_units[0u] >> 1_zu;
                tmp_number_units[3u] = tmp_number_units[1u] >> 1_zu;

                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                            batch_size_received,
                                                                                                            tmp_number_units[2u],
                                                                                                            tmp_number_units[3u],
                                                                                                            *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                            tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            &tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size_received,
                                                                                                                       tmp_number_units[2u],
                                                                                                                       tmp_number_units[3u],
                                                                                                                       tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                       tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                       tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                       tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                       this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index,
                                                                                                                       this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3u],
                                                                                                                       this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3u] + tmp_number_units[2u],
                                                                                                                       this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3u] + 2_zu * tmp_number_units[2u]);
                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(false,
                                                                                                            batch_size_received,
                                                                                                            tmp_number_units[2u],
                                                                                                            tmp_number_units[3u],
                                                                                                            *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                            tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            &tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size_received,
                                                                                                                        tmp_number_units[2u],
                                                                                                                        tmp_number_units[3u],
                                                                                                                        tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                        tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                        tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                        tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index,
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3u],
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3u] + tmp_number_units[2u],
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3u] + 2_zu * tmp_number_units[2u]);
            }
            else
            {
                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                            batch_size_received,
                                                                                                            tmp_number_units[0u],
                                                                                                            tmp_number_units[1u],
                                                                                                            *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                            tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                            tmp_ptr_layer_it->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            tmp_ptr_layer_it->Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            tmp_ptr_layer_it);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size_received,
                                                                                                                        tmp_number_units[0u],
                                                                                                                        tmp_number_units[1u],
                                                                                                                        tmp_ptr_layer_it->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                        tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                        tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                        tmp_ptr_layer_it->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->first_bias_connection_index,
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->first_bias_connection_index + tmp_number_units[1u],
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->first_bias_connection_index + tmp_number_units[1u] + tmp_number_units[0u],
                                                                                                                        this->ptr_array_derivatives_parameters + tmp_ptr_layer_it->first_bias_connection_index + tmp_number_units[1u] + 2_zu * tmp_number_units[0u]);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_layer_it->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                return;
    }
    // |END| Decode level part. |END|
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Recurrent__Update_Derivative_Weight__FC__OpenMP(size_t const batch_size_received,
                                                                                                               size_t const input_unit_size_received,
                                                                                                               T_ const *const ptr_array_inputs_received,
                                                                                                               struct Layer *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Update_Derivative_Weight__FC__OpenMP(tmp_time_step_index,
                                                                               batch_size_received,
                                                                               input_unit_size_received,
                                                                               ptr_array_inputs_received,
                                                                               ptr_layer_it_received);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Update_Derivative_Weight__FC_Ind_RNN__OpenMP(size_t const time_step_index_received,
                                                                                                            size_t const batch_size_received,
                                                                                                            size_t const derivative_size_received,
                                                                                                            T_ const *const ptr_array_inputs_received,
                                                                                                            T_ const *const ptr_array_derivative_inputs_received,
                                                                                                            T_ *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const tmp_derivative_previous_timed_batched_index(this->batch_size * derivative_size_received * (time_step_index_received - 1_zu)),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_derivative_index;
    
    T_ const *tmp_ptr_array_previous_timed_inputs,
                  *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_previous_timed_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_previous_timed_batched_index;
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(omp_get_thread_num()) * this->total_parameters_allocated;

        for(tmp_derivative_index = 0_zu; tmp_derivative_index != derivative_size_received; ++tmp_derivative_index) { tmp_ptr_array_derivatives[tmp_derivative_index] += tmp_ptr_array_previous_timed_inputs[tmp_derivative_index] * tmp_ptr_array_derivative_inputs[tmp_derivative_index]; }
    }
}

void Neural_Network::Recurrent__Update_Derivative_Weight__LSTM__OpenMP(bool const forward_layer_received,
                                                                                                                    size_t const batch_size_received,
                                                                                                                    size_t const block_unit_size_received,
                                                                                                                    size_t const cell_unit_size_received,
                                                                                                                    size_t const input_unit_size_received,
                                                                                                                    T_ const *const ptr_array_inputs_received,
                                                                                                                    T_ const *const ptr_array_delta_block_inputs_received,
                                                                                                                    T_ const *const ptr_array_delta_input_block_inputs_received,
                                                                                                                    T_ const *const ptr_array_delta_recurrent_block_inputs_received,
                                                                                                                    T_ const *const ptr_array_delta_input_gates_received,
                                                                                                                    T_ const *const ptr_array_delta_input_input_gates_received,
                                                                                                                    T_ const *const ptr_array_delta_recurrent_input_gates_received,
                                                                                                                    T_ const *const ptr_array_delta_forget_gates_received,
                                                                                                                    T_ const *const ptr_array_delta_input_forget_gates_received,
                                                                                                                    T_ const *const ptr_array_delta_recurrent_forget_gates_received,
                                                                                                                    T_ const *const ptr_array_delta_output_gates_received,
                                                                                                                    T_ const *const ptr_array_delta_input_output_gates_received,
                                                                                                                    T_ const *const ptr_array_delta_recurrent_output_gates_received,
                                                                                                                    struct Layer *const ptr_layer_it_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const tmp_number_inputs_connections(ptr_layer_it_received->ptr_array_block_units->last_index_feedforward_connection_input_gate - ptr_layer_it_received->ptr_array_block_units->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrents_connection(ptr_layer_it_received->ptr_array_block_units->last_index_recurrent_connection_input_gate - ptr_layer_it_received->ptr_array_block_units->first_index_recurrent_connection_input_gate),
                       tmp_number_peepholes_connections(ptr_layer_it_received->ptr_array_block_units->last_index_peephole_input_gate - ptr_layer_it_received->ptr_array_block_units->first_index_peephole_input_gate);
    size_t tmp_thread_index,
              tmp_time_step_direction_direction,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_block_data_direction_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_direction_timed_index;
    
    T_ const *tmp_ptr_array_previous_layer_outputs,
                  *tmp_ptr_array_layer_timed_outputs,
                  *tmp_ptr_array_cells_states;
    T_ *tmp_ptr_array_cell_input_derivatives_parameters,
         *tmp_ptr_array_input_gate_derivatives_parameters,
         *tmp_ptr_array_forget_gate_derivatives_parameters,
         *tmp_ptr_array_output_gate_derivatives_parameters,
         tmp_cell_state,
         tmp_cell_input_error,
         tmp_input_gate_error,
         tmp_forget_gate_error,
         tmp_output_gate_error;

    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    long long int tmp_time_step_index,
                       tmp_time_step_start(forward_layer_received ? 0ll : static_cast<long long int>(this->number_recurrent_depth - 1_zu)),
                       tmp_time_step_end(forward_layer_received ? static_cast<long long int>(this->number_recurrent_depth) : -1ll),
                       tmp_time_prediction_end(forward_layer_received ? static_cast<long long int>(this->number_recurrent_depth - 1_zu) : 0ll);
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index = static_cast<size_t>(omp_get_thread_num());

        for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? ++tmp_time_step_index : --tmp_time_step_index)
        {
            if(tmp_time_step_index != tmp_time_prediction_end)
            {
                tmp_time_step_direction_direction = forward_layer_received ? static_cast<size_t>(tmp_time_step_index + 1ll) : static_cast<size_t>(tmp_time_step_index - 1ll);
                
                tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                tmp_block_data_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * tmp_time_step_direction_direction;

                tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                tmp_cell_data_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * tmp_time_step_direction_direction;
                
                tmp_ptr_array_previous_layer_outputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_unit_size_received + this->batch_size * input_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                tmp_ptr_array_layer_timed_outputs = ptr_layer_it_received->ptr_array_cell_units->ptr_cell_output + tmp_cell_data_timed_index;
                
                for(tmp_cell_index = 0_zu,
                    tmp_block_index = 0_zu,
                    tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                                                                                                 ++tmp_block_index)
                {
                    // [0] Cells inputs.
                    for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                        tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                           ++tmp_cell_index)
                    {
                        // Cell inputs.
                        tmp_cell_input_error = ptr_array_delta_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index];
                        
                        tmp_ptr_array_cell_input_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input + tmp_thread_index * this->total_parameters_allocated;

                        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                        {
                            tmp_ptr_array_cell_input_derivatives_parameters[tmp_connection_index] += tmp_cell_input_error * tmp_ptr_array_previous_layer_outputs[tmp_connection_index];
                        }
                        // |END| Cell inputs. |END|

                        // Cell recurrents.
                        tmp_cell_input_error = ptr_array_delta_recurrent_block_inputs_received[tmp_cell_data_direction_timed_index + tmp_cell_index];
                        
                        tmp_ptr_array_cell_input_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input + tmp_thread_index * this->total_parameters_allocated;

                        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
                        {
                            tmp_ptr_array_cell_input_derivatives_parameters[tmp_connection_index] += tmp_cell_input_error * tmp_ptr_array_layer_timed_outputs[tmp_connection_index];
                        }
                        // |END| Cell recurrents. |END|
                    }
                    // [0] |END| Cells inputs. |END|
                    
                    // [0] Gates-inputs.
                    tmp_input_gate_error = ptr_array_delta_input_input_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    tmp_forget_gate_error = ptr_array_delta_input_forget_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    tmp_output_gate_error = ptr_array_delta_input_output_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    
                    tmp_ptr_array_input_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_forget_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_output_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate + tmp_thread_index * this->total_parameters_allocated;

                    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                    {
                        tmp_ptr_array_input_gate_derivatives_parameters[tmp_connection_index] += tmp_input_gate_error * tmp_ptr_array_previous_layer_outputs[tmp_connection_index];
                        tmp_ptr_array_forget_gate_derivatives_parameters[tmp_connection_index] += tmp_forget_gate_error * tmp_ptr_array_previous_layer_outputs[tmp_connection_index];
                        tmp_ptr_array_output_gate_derivatives_parameters[tmp_connection_index] += tmp_output_gate_error * tmp_ptr_array_previous_layer_outputs[tmp_connection_index];
                    }

                    // [0] Output gate, peepholes.
                    tmp_input_gate_error = ptr_array_delta_recurrent_input_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];
                    tmp_forget_gate_error = ptr_array_delta_recurrent_forget_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];

                #ifndef NO_PEEPHOLE
                    tmp_ptr_array_cells_states = tmp_ptr_block_unit_it->ptr_array_cells_states + tmp_cell_data_timed_index;
                    
                    tmp_ptr_array_input_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_forget_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_output_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate + tmp_thread_index * this->total_parameters_allocated;

                    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                    {
                        tmp_cell_state = tmp_ptr_array_cells_states[tmp_connection_index];

                        tmp_ptr_array_input_gate_derivatives_parameters[tmp_connection_index] += tmp_input_gate_error * tmp_cell_state;
                        tmp_ptr_array_forget_gate_derivatives_parameters[tmp_connection_index] += tmp_forget_gate_error * tmp_cell_state;
                        tmp_ptr_array_output_gate_derivatives_parameters[tmp_connection_index] += tmp_output_gate_error * tmp_cell_state;
                    }
                #endif
                    // [0] |END| Output gate, peepholes. |END|
                    // [0] |END| Gates-inputs. |END|

                    // [0] Gates-recurrents.
                    tmp_output_gate_error = ptr_array_delta_recurrent_output_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];
                    
                    tmp_ptr_array_input_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_forget_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_output_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate + tmp_thread_index * this->total_parameters_allocated;

                    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
                    {
                        tmp_ptr_array_input_gate_derivatives_parameters[tmp_connection_index] += tmp_input_gate_error * tmp_ptr_array_layer_timed_outputs[tmp_connection_index];
                        tmp_ptr_array_forget_gate_derivatives_parameters[tmp_connection_index] += tmp_forget_gate_error * tmp_ptr_array_layer_timed_outputs[tmp_connection_index];
                        tmp_ptr_array_output_gate_derivatives_parameters[tmp_connection_index] += tmp_output_gate_error * tmp_ptr_array_layer_timed_outputs[tmp_connection_index];
                    }
                    // [0] |END| Gates-recurrents. |END|
                }
            }
            else
            {
                tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                tmp_ptr_array_previous_layer_outputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_unit_size_received + this->batch_size * input_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                for(tmp_cell_index = 0_zu,
                    tmp_block_index = 0_zu,
                    tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                                                                                                 ++tmp_block_index)
                {
                    // [0] Cells inputs.
                    for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                        tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                           ++tmp_cell_index)
                    {
                        // Cell inputs.
                        tmp_cell_input_error = ptr_array_delta_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index];
                        
                        tmp_ptr_array_cell_input_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input + tmp_thread_index * this->total_parameters_allocated;

                        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                        {
                            tmp_ptr_array_cell_input_derivatives_parameters[tmp_connection_index] += tmp_cell_input_error * tmp_ptr_array_previous_layer_outputs[tmp_connection_index];
                        }
                        // |END| Cell inputs. |END|
                    }
                    // [0] |END| Cells inputs. |END|
                    
                    // [0] Gates-inputs.
                    tmp_input_gate_error = ptr_array_delta_input_input_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    tmp_forget_gate_error = ptr_array_delta_input_forget_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    tmp_output_gate_error = ptr_array_delta_input_output_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    
                    tmp_ptr_array_input_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_forget_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_output_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate + tmp_thread_index * this->total_parameters_allocated;

                    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                    {
                        tmp_ptr_array_input_gate_derivatives_parameters[tmp_connection_index] += tmp_input_gate_error * tmp_ptr_array_previous_layer_outputs[tmp_connection_index];
                        tmp_ptr_array_forget_gate_derivatives_parameters[tmp_connection_index] += tmp_forget_gate_error * tmp_ptr_array_previous_layer_outputs[tmp_connection_index];
                        tmp_ptr_array_output_gate_derivatives_parameters[tmp_connection_index] += tmp_output_gate_error * tmp_ptr_array_previous_layer_outputs[tmp_connection_index];
                    }

                    // [0] Output gate, peepholes.
                #ifndef NO_PEEPHOLE
                    tmp_ptr_array_cells_states = tmp_ptr_block_unit_it->ptr_array_cells_states + tmp_cell_data_timed_index;
                    
                    tmp_ptr_array_output_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate + tmp_thread_index * this->total_parameters_allocated;

                    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                    {
                        tmp_ptr_array_output_gate_derivatives_parameters[tmp_connection_index] += tmp_output_gate_error * tmp_ptr_array_cells_states[tmp_connection_index];
                    }
                #endif
                    // [0] |END| Output gate, peepholes. |END|
                    // [0] |END| Gates-inputs. |END|
                }
            }
        }
    }
}

void Neural_Network::Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(size_t const batch_size_received,
                                                                                                                              size_t const layer_block_unit_size_received,
                                                                                                                              size_t const layer_cell_unit_size_received,
                                                                                                                              T_ const *const ptr_array_delta_block_inputs_received,
                                                                                                                              T_ const *const ptr_array_delta_input_gates_received,
                                                                                                                              T_ const *const ptr_array_delta_forget_gates_received,
                                                                                                                              T_ const *const ptr_array_delta_output_gates_received,
                                                                                                                              T_ *const ptr_array_cell_input_derivatives_bias_received,
                                                                                                                              T_ *const ptr_array_input_gate_derivatives_bias_received,
                                                                                                                              T_ *const ptr_array_forget_gate_derivatives_bias_received,
                                                                                                                              T_ *const ptr_array_output_gate_derivatives_bias_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t tmp_thread_index,
              tmp_time_step_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_block_index,
              tmp_block_data_timed_index;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index = static_cast<size_t>(omp_get_thread_num());

        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(tmp_time_step_index);

            tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(tmp_time_step_index);

            for(tmp_cell_index = 0_zu; tmp_cell_index != layer_cell_unit_size_received; ++tmp_cell_index)
            {
                ptr_array_cell_input_derivatives_bias_received[tmp_thread_index * this->total_parameters_allocated + tmp_cell_index] += ptr_array_delta_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index];
            }

            for(tmp_block_index = 0_zu; tmp_block_index != layer_block_unit_size_received; ++tmp_block_index)
            {
                ptr_array_input_gate_derivatives_bias_received[tmp_thread_index * this->total_parameters_allocated + tmp_block_index] += ptr_array_delta_input_gates_received[tmp_block_data_timed_index + tmp_block_index];
                ptr_array_forget_gate_derivatives_bias_received[tmp_thread_index * this->total_parameters_allocated + tmp_block_index] += ptr_array_delta_forget_gates_received[tmp_block_data_timed_index + tmp_block_index];
                ptr_array_output_gate_derivatives_bias_received[tmp_thread_index * this->total_parameters_allocated + tmp_block_index] += ptr_array_delta_output_gates_received[tmp_block_data_timed_index + tmp_block_index];
            }
        }
    }
}
