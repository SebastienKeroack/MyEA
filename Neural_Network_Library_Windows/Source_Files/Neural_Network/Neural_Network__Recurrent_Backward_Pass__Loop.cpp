#include "stdafx.hpp"

#include <Math/Mathematic.hpp>

#include <Neural_Network/Neural_Network.hpp>

void Neural_Network::RNN__Backward_Pass_Batch__Loop(size_t const batch_size_received)
{
    size_t tmp_layer_number_outputs;
    
    T_ *tmp_ptr_array_layer_gradients;
    
#if defined(COMPILE_ADEPT)
    struct Layer const *const tmp_ptr_second_layer(this->ptr_array_layers);
#else
    struct Layer const *const tmp_ptr_second_layer(this->ptr_array_layers + 1);
#endif

    struct Layer const *tmp_ptr_next_layer_end,
                               *tmp_ptr_next_layer_it;
    struct Layer *tmp_ptr_gradient_layer_it(this->ptr_last_layer - 1),
                      *tmp_ptr_layer_it;

#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->type_state_propagation != MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not back propagate gradient in inference mode. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return;
    }
#endif
    
    // If the network use normalization.
    if(this->Use__Normalization())
    {
        // Set all derivative mean to zero.
        MEMSET(this->ptr_array_normalized_batch_units_derivatives_means,
                    0,
                    this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
        // |END| Set all derivative mean to zero. |END|

        // Set all derivative variance to zero.
        MEMSET(this->ptr_array_normalized_batch_units_derivatives_variances,
                    0,
                    this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
        // |END| Set all derivative variance to zero. |END|
    }

    // Loop through each layer and do a backward propagation.
    for(; tmp_ptr_gradient_layer_it != tmp_ptr_second_layer; --tmp_ptr_gradient_layer_it)
    {
        tmp_ptr_layer_it = this->ptr_array_layers + static_cast<size_t>(tmp_ptr_gradient_layer_it->previous_connected_layers[0u] - this->ptr_array_layers);

        // Clear past error(s).
        tmp_layer_number_outputs = *tmp_ptr_layer_it->ptr_number_outputs;

        tmp_ptr_array_layer_gradients = tmp_ptr_layer_it->ptr_array_derivative_outputs;

        MEMSET(tmp_ptr_array_layer_gradients,
                    0,
                    this->batch_size * tmp_layer_number_outputs * this->number_recurrent_depth * sizeof(T_));
        // |END| Clear past error(s). |END|
        
        // Propagate the error(s) to the layer.
        for(tmp_ptr_next_layer_it = tmp_ptr_layer_it->next_connected_layers[0u],
            tmp_ptr_next_layer_end = tmp_ptr_next_layer_it + tmp_ptr_layer_it->next_connected_layers.size(); tmp_ptr_next_layer_it != tmp_ptr_next_layer_end; ++tmp_ptr_next_layer_it)
        {
            switch(tmp_ptr_next_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                    this->Recurrent__Backward_Pass__Average_Pooling__Loop(batch_size_received,
                                                                                                            tmp_layer_number_outputs,
                                                                                                            tmp_ptr_array_layer_gradients,
                                                                                                            tmp_ptr_next_layer_it);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    this->Recurrent__Backward_Pass__FC__Loop(batch_size_received,
                                                                                         tmp_layer_number_outputs,
                                                                                         tmp_ptr_array_layer_gradients,
                                                                                         tmp_ptr_next_layer_it);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    this->Recurrent__Backward_Pass__LSTM__Loop(batch_size_received,
                                                                                             tmp_layer_number_outputs,
                                                                                             tmp_ptr_array_layer_gradients,
                                                                                             tmp_ptr_next_layer_it);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                    this->Recurrent__Backward_Pass__Max_Pooling__Loop(batch_size_received,
                                                                                                       tmp_layer_number_outputs,
                                                                                                       tmp_ptr_array_layer_gradients,
                                                                                                       tmp_ptr_next_layer_it);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    this->Recurrent__Backward_Pass__Residual__Loop(batch_size_received,
                                                                                                 tmp_layer_number_outputs,
                                                                                                 tmp_ptr_array_layer_gradients,
                                                                                                 tmp_ptr_next_layer_it);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_next_layer_it->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_next_layer_it->type_layer].c_str());
                        return;
            }
        }
        // |END| Propagate the error(s) to the layer. |END|

        // Compute the gradients.
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING: break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Recurrent__Backward_Pass__Gradient__FC__Loop(batch_size_received, tmp_ptr_layer_it); break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(tmp_ptr_layer_it->Use__Bidirectional())
                {
                    this->Recurrent__Backward_Pass__Gradient__LSTM__Loop(true,
                                                                                                    batch_size_received,
                                                                                                    tmp_layer_number_outputs,
                                                                                                    tmp_ptr_array_layer_gradients,
                                                                                                    &tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer);
                    this->Recurrent__Backward_Pass__Gradient__LSTM__Loop(false,
                                                                                                    batch_size_received,
                                                                                                    tmp_layer_number_outputs,
                                                                                                    tmp_ptr_array_layer_gradients,
                                                                                                    &tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer);
                }
                else
                {
                    this->Recurrent__Backward_Pass__Gradient__LSTM__Loop(true,
                                                                                                    batch_size_received,
                                                                                                    tmp_layer_number_outputs,
                                                                                                    tmp_ptr_array_layer_gradients,
                                                                                                    tmp_ptr_layer_it);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                this->Recurrent__Backward_Pass__Gradient__Residual__Loop(batch_size_received, tmp_ptr_layer_it);

                tmp_ptr_gradient_layer_it = tmp_ptr_layer_it + 1;
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    return;
        }
        // |END| Compute the gradients. |END|
    }
}

void Neural_Network::RNN__Backward_Pass_Batch__Pre_Training__Loop(size_t const batch_size_received)
{
    size_t tmp_layer_number_outputs;
    
    T_ *tmp_ptr_array_layer_gradients;
    
    struct Layer *const tmp_ptr_coded_layer(this->ptr_array_layers + this->pre_training_level);
    struct Layer const *const tmp_ptr_decoded_layer(this->ptr_last_layer - static_cast<size_t>(tmp_ptr_coded_layer - this->ptr_array_layers));
    
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->type_state_propagation != MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not back propagate gradient in inference mode. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return;
    }
#endif
    
    // If the network use normalization.
    if(this->Use__Normalization())
    {
        // Set all derivative mean to zero.
        MEMSET(this->ptr_array_normalized_batch_units_derivatives_means,
                    0,
                    this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
        // |END| Set all derivative mean to zero. |END|

        // Set all derivative variance to zero.
        MEMSET(this->ptr_array_normalized_batch_units_derivatives_variances,
                    0,
                    this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
        // |END| Set all derivative variance to zero. |END|
    }

    // Clear past error(s).
    tmp_layer_number_outputs = *tmp_ptr_coded_layer->ptr_number_outputs;

    tmp_ptr_array_layer_gradients = tmp_ptr_coded_layer->ptr_array_derivative_outputs;

    MEMSET(tmp_ptr_array_layer_gradients,
                   0,
                   this->batch_size * tmp_layer_number_outputs * this->number_recurrent_depth * sizeof(T_));
    // |END| Clear past error(s). |END|
    
    // Propagate the error(s) to the layer.
    switch(tmp_ptr_decoded_layer->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            this->Recurrent__Backward_Pass__FC__Loop(batch_size_received,
                                                                                 tmp_layer_number_outputs,
                                                                                 tmp_ptr_array_layer_gradients,
                                                                                 tmp_ptr_decoded_layer);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            this->Recurrent__Backward_Pass__LSTM__Loop(batch_size_received,
                                                                                     tmp_layer_number_outputs,
                                                                                     tmp_ptr_array_layer_gradients,
                                                                                     tmp_ptr_decoded_layer);
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_decoded_layer->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_decoded_layer->type_layer].c_str());
                return;
    }
    // |END| Propagate the error(s) to the layer. |END|

    // Compute the gradients.
    switch(tmp_ptr_coded_layer->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Recurrent__Backward_Pass__Gradient__FC__Loop(batch_size_received, tmp_ptr_coded_layer); break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            if(tmp_ptr_coded_layer->Use__Bidirectional())
            {
                this->Recurrent__Backward_Pass__Gradient__LSTM__Loop(true,
                                                                                                batch_size_received,
                                                                                                tmp_layer_number_outputs,
                                                                                                tmp_ptr_array_layer_gradients,
                                                                                                &tmp_ptr_coded_layer->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Backward_Pass__Gradient__LSTM__Loop(false,
                                                                                                batch_size_received,
                                                                                                tmp_layer_number_outputs,
                                                                                                tmp_ptr_array_layer_gradients,
                                                                                                &tmp_ptr_coded_layer->ptr_Bidirectional_Layer->backward_layer);
            }
            else
            {
                this->Recurrent__Backward_Pass__Gradient__LSTM__Loop(true,
                                                                                                batch_size_received,
                                                                                                tmp_layer_number_outputs,
                                                                                                tmp_ptr_array_layer_gradients,
                                                                                                tmp_ptr_coded_layer);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_coded_layer->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_coded_layer->type_layer].c_str());
                return;
    }
    // |END| Compute the gradients. |END|
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Recurrent__Backward_Pass__Average_Pooling__Loop(size_t const batch_size_received,
                                                                                                               size_t const derivative_size_received,
                                                                                                               T_ *const ptr_array_derivatives_received,
                                                                                                               struct Layer const *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(this->number_recurrent_depth); tmp_time_step_index--;)
    {
        this->Backward_Pass__Average_Pooling__Loop(tmp_time_step_index,
                                                                               batch_size_received,
                                                                               derivative_size_received,
                                                                               ptr_array_derivatives_received,
                                                                               ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Backward_Pass__FC__Loop(size_t const batch_size_received,
                                                                                            size_t const derivative_size_received,
                                                                                            T_ *const ptr_array_derivatives_received,
                                                                                            struct Layer const *const ptr_layer_it_received)
{
    if(ptr_layer_it_received->type_group == MyEA::Common::ENUM_TYPE_GROUP::TYPE_GROUP_RESIDUAL)
    {
        for(size_t tmp_time_step_index(this->number_recurrent_depth); tmp_time_step_index--;)
        {
            this->Backward_Pass__Residual__FC__Loop(tmp_time_step_index,
                                                                               batch_size_received,
                                                                               derivative_size_received,
                                                                               ptr_array_derivatives_received,
                                                                               ptr_layer_it_received);
        }
    }
    else
    {
        for(size_t tmp_time_step_index(this->number_recurrent_depth); tmp_time_step_index--;)
        {
            this->Backward_Pass__FC__Loop(tmp_time_step_index,
                                                               batch_size_received,
                                                               derivative_size_received,
                                                               ptr_array_derivatives_received,
                                                               ptr_layer_it_received);
        }
    }
}

void Neural_Network::Recurrent__Backward_Pass__LSTM__Loop(size_t const batch_size_received,
                                                                                                size_t const derivative_size_received,
                                                                                                T_ *const ptr_array_derivatives_received,
                                                                                                struct Layer const *const ptr_layer_it_received)
{
    size_t tmp_time_step_index;
    
    T_ const *const tmp_ptr_array_delta_input_block_inputs(ptr_layer_it_received->Get__Array_Deltas__Cell__Block_Input__Input()),
                  *const tmp_ptr_array_delta_input_input_gates(ptr_layer_it_received->Get__Array_Deltas__Block__Input_Gate__Input()),
                  *const tmp_ptr_array_delta_input_forget_gates(ptr_layer_it_received->Get__Array_Deltas__Block__Forget_Gate__Input()),
                  *const tmp_ptr_array_delta_input_output_gates(ptr_layer_it_received->Get__Array_Deltas__Block__Output_Gate__Input());

    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Backward_Pass__LSTM__Loop(tmp_time_step_index,
                                                               batch_size_received,
                                                               derivative_size_received,
                                                               tmp_ptr_array_delta_input_block_inputs,
                                                               tmp_ptr_array_delta_input_input_gates,
                                                               tmp_ptr_array_delta_input_forget_gates,
                                                               tmp_ptr_array_delta_input_output_gates,
                                                               ptr_array_derivatives_received,
                                                               ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Backward_Pass__Max_Pooling__Loop(size_t const batch_size_received,
                                                                                                          size_t const derivative_size_received,
                                                                                                          T_ *const ptr_array_derivatives_received,
                                                                                                          struct Layer const *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(this->number_recurrent_depth); tmp_time_step_index--;)
    {
        this->Backward_Pass__Max_Pooling__Loop(tmp_time_step_index,
                                                                          batch_size_received,
                                                                          derivative_size_received,
                                                                          ptr_array_derivatives_received,
                                                                          ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Backward_Pass__Residual__Loop(size_t const batch_size_received,
                                                                                                    size_t const derivative_size_received,
                                                                                                    T_ *const ptr_array_derivatives_received,
                                                                                                    struct Layer const *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(this->number_recurrent_depth); tmp_time_step_index--;)
    {
        this->Backward_Pass__Residual__Loop(tmp_time_step_index,
                                                                   batch_size_received,
                                                                   derivative_size_received,
                                                                   ptr_array_derivatives_received,
                                                                   ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Backward_Pass__Residual__Block__Loop(size_t const batch_size_received,
                                                                                                                size_t const derivative_size_received,
                                                                                                                T_ *const ptr_array_derivatives_received,
                                                                                                                struct Layer const *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(this->number_recurrent_depth); tmp_time_step_index--;)
    {
        this->Backward_Pass__Residual__Block__Loop(tmp_time_step_index,
                                                                               batch_size_received,
                                                                               derivative_size_received,
                                                                               ptr_array_derivatives_received,
                                                                               ptr_layer_it_received);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Recurrent__Backward_Pass__Gradient__FC__Loop(size_t const batch_size_received, struct Layer const *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(this->number_recurrent_depth); tmp_time_step_index--;)
    {
        this->Backward_Pass__Gradient__FC__Loop(tmp_time_step_index,
                                                                           batch_size_received,
                                                                           ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Backward_Pass__Gradient__LSTM__Loop(bool const forward_layer_received,
                                                                                                                size_t const batch_size_received,
                                                                                                                size_t const derivative_input_size_received,
                                                                                                                T_ *const ptr_array_derivative_inputs_received,
                                                                                                                struct Layer *const ptr_layer_it_received)
{
    struct Block_unit *const tmp_ptr_layer_first_block_unit(ptr_layer_it_received->ptr_array_block_units);
    
    struct Cell_unit *const tmp_ptr_layer_first_cell_unit(ptr_layer_it_received->ptr_array_cell_units);
    
    size_t const tmp_number_block_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - tmp_ptr_layer_first_block_unit)),
                       tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - tmp_ptr_layer_first_cell_unit));

    long long int tmp_time_step_index,
                       tmp_time_step_start(forward_layer_received ? static_cast<long long int>(this->number_recurrent_depth - 1_zu) : 0ll),
                       tmp_time_step_end(forward_layer_received ? -1ll : static_cast<long long int>(this->number_recurrent_depth)),
                       tmp_time_prediction_direction_end(forward_layer_received ? 0ll : static_cast<long long int>(this->number_recurrent_depth - 1_zu));
    
    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? --tmp_time_step_index : ++tmp_time_step_index)
    {
        // Output.
        this->Backward_Pass__LSTM_Derivative__Output__Loop(tmp_time_step_index,
                                                                                            forward_layer_received ? (tmp_time_step_index + 1ll) : (tmp_time_step_index - 1ll),
                                                                                            tmp_time_step_start,
                                                                                            batch_size_received,
                                                                                            tmp_number_block_units,
                                                                                            tmp_number_cell_units,
                                                                                            ptr_layer_it_received->Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                            ptr_layer_it_received->Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                            ptr_layer_it_received->Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                            ptr_layer_it_received->Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                            ptr_layer_it_received);
        
        // Output gate normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Output gate, memcpy.
            MEMCPY(tmp_ptr_layer_first_block_unit->ptr_delta_input_outputs_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                         tmp_ptr_layer_first_block_unit->ptr_delta_outputs_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                         batch_size_received * tmp_number_block_units * sizeof(T_));
            
            if(tmp_time_step_index != tmp_time_prediction_direction_end)
            {
                MEMCPY(tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_outputs_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                             tmp_ptr_layer_first_block_unit->ptr_delta_outputs_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                             batch_size_received * tmp_number_block_units * sizeof(T_));
            }
            // |END| Output gate, memcpy. |END|
            
            // Normalization.
            switch(ptr_layer_it_received->type_normalization)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    this->Backward_Pass__Batch_Normalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                batch_size_received,
                                                                                                tmp_number_block_units,
                                                                                                tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_delta_input_outputs_gates,
                                                                                                tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_delta_input_outputs_gates);

                    if(tmp_time_step_index != tmp_time_prediction_direction_end)
                    {
                        this->Backward_Pass__Batch_Normalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size_received,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_outputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_outputs_gates);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    this->Backward_Pass__Batch_Renormalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size_received,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_r_correction,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_input_outputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_input_outputs_gates);
                    
                    if(tmp_time_step_index != tmp_time_prediction_direction_end)
                    {
                        this->Backward_Pass__Batch_Renormalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_r_correction,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_outputs_gates);
                    }
                        break;
                default: break;
            }
        }
        
        // Cell activation state.
        this->Backward_Pass__LSTM_Derivative__Cell_State_AF__Loop(tmp_time_step_index,
                                                                                                       forward_layer_received ? (tmp_time_step_index + 1ll) : (tmp_time_step_index - 1ll),
                                                                                                       tmp_time_prediction_direction_end,
                                                                                                       tmp_time_step_start,
                                                                                                       batch_size_received,
                                                                                                       tmp_number_block_units,
                                                                                                       tmp_number_cell_units,
                                                                                                       ptr_layer_it_received->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                                       ptr_layer_it_received);
        
        // Cell state normalization.
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                this->Backward_Pass__Batch_Normalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                            batch_size_received,
                                                                                            tmp_number_cell_units,
                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_means,
                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                            tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_hats,
                                                                                            tmp_ptr_layer_first_cell_unit->ptr_delta_cell_state,
                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                            tmp_ptr_layer_first_cell_unit->ptr_delta_cell_state);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                this->Backward_Pass__Batch_Renormalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                batch_size_received,
                                                                                                tmp_number_cell_units,
                                                                                                tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_r_correction,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_delta_cell_state,
                                                                                                tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_delta_cell_state);
                    break;
            default: break;
        }

        // CIF gate, activation, state.
        this->Backward_Pass__LSTM_Derivative__Gates_CIF_AF_State__Loop(tmp_time_step_index,
                                                                                                                 forward_layer_received ? (tmp_time_step_index + 1ll) : (tmp_time_step_index - 1ll),
                                                                                                                 forward_layer_received ? (tmp_time_step_index - 1ll) : (tmp_time_step_index + 1ll),
                                                                                                                 tmp_time_prediction_direction_end,
                                                                                                                 tmp_time_step_start,
                                                                                                                 batch_size_received,
                                                                                                                 tmp_number_block_units,
                                                                                                                 tmp_number_cell_units,
                                                                                                                 ptr_layer_it_received);

        // CIF gate normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // memcpy.
            MEMCPY(tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input_input + this->batch_size * tmp_number_cell_units * tmp_time_step_index,
                         tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input + this->batch_size * tmp_number_cell_units * tmp_time_step_index,
                         batch_size_received * tmp_number_cell_units * sizeof(T_));

            MEMCPY(tmp_ptr_layer_first_block_unit->ptr_delta_input_inputs_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                         tmp_ptr_layer_first_block_unit->ptr_delta_inputs_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                         batch_size_received * tmp_number_block_units * sizeof(T_));

            MEMCPY(tmp_ptr_layer_first_block_unit->ptr_delta_input_forgets_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                         tmp_ptr_layer_first_block_unit->ptr_delta_forgets_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                         batch_size_received * tmp_number_block_units * sizeof(T_));
            
            if(tmp_time_step_index != tmp_time_prediction_direction_end)
            {
                MEMCPY(tmp_ptr_layer_first_cell_unit->ptr_delta_cell_recurrent_input + this->batch_size * tmp_number_cell_units * tmp_time_step_index,
                             tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input + this->batch_size * tmp_number_cell_units * tmp_time_step_index,
                             batch_size_received * tmp_number_cell_units * sizeof(T_));

                MEMCPY(tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_inputs_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                             tmp_ptr_layer_first_block_unit->ptr_delta_inputs_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                             batch_size_received * tmp_number_block_units * sizeof(T_));

                MEMCPY(tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_forgets_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                             tmp_ptr_layer_first_block_unit->ptr_delta_forgets_gates + this->batch_size * tmp_number_block_units * tmp_time_step_index,
                             batch_size_received * tmp_number_block_units * sizeof(T_));
            }
            // |END| memcpy. |END|
            
            // Normalization.
            switch(ptr_layer_it_received->type_normalization)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    // Block input, input.
                    this->Backward_Pass__Batch_Normalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                batch_size_received,
                                                                                                tmp_number_cell_units,
                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input_input,
                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input_input);

                    // Input gate, input.
                    this->Backward_Pass__Batch_Normalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                batch_size_received,
                                                                                                tmp_number_block_units,
                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_delta_input_inputs_gates,
                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_delta_input_inputs_gates);

                    if(tmp_time_step_index != tmp_time_prediction_direction_end)
                    {
                        // Forget gate, input.
                        this->Backward_Pass__Batch_Normalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size_received,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_input_forgets_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_input_forgets_gates);

                        // Block input, recurrent.
                        this->Backward_Pass__Batch_Normalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size_received,
                                                                                                    tmp_number_cell_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_delta_cell_recurrent_input,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_delta_cell_recurrent_input);
                        
                        // Input gate, recurrent.
                        this->Backward_Pass__Batch_Normalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size_received,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_inputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_inputs_gates);

                        // Forget gate, recurrent.
                        this->Backward_Pass__Batch_Normalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size_received,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_forgets_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_forgets_gates);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    // Block input, input.
                    this->Backward_Pass__Batch_Renormalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size_received,
                                                                                                    tmp_number_cell_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_r_correction,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input_input,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input_input);
                    
                    // Input gate, input.
                    this->Backward_Pass__Batch_Renormalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size_received,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_r_correction,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_input_inputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_input_inputs_gates);
                    
                    if(tmp_time_step_index != tmp_time_prediction_direction_end)
                    {
                        // Forget gate, input.
                        this->Backward_Pass__Batch_Renormalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_r_correction,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_input_forgets_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_input_forgets_gates);
                        
                        // Block input, recurrent.
                        this->Backward_Pass__Batch_Renormalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_r_correction,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_delta_cell_recurrent_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_delta_cell_recurrent_input);
                        
                        // Input gate, recurrent.
                        this->Backward_Pass__Batch_Renormalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_r_correction,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                        tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_inputs_gates);
                        
                        // Forget gate, recurrent.
                        this->Backward_Pass__Batch_Renormalization__Loop(static_cast<size_t>(tmp_time_step_index),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_r_correction,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_forgets_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                        tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_forgets_gates);
                    }
                        break;
                default: break;
            }
        }
    }
}

void Neural_Network::Recurrent__Backward_Pass__Gradient__Residual__Loop(size_t const batch_size_received, struct Layer const *const ptr_layer_it_received)
{
    struct Layer const *const tmp_ptr_layer_end(ptr_layer_it_received + 1);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + static_cast<size_t>(ptr_layer_it_received - this->ptr_array_layers) + ptr_layer_it_received->block_depth);
    
    // Remaining layer(s).
    for(; tmp_ptr_layer_it != tmp_ptr_layer_end; --tmp_ptr_layer_it)
    {
        this->Recurrent__Backward_Pass__Gradient__Residual__Layer__Loop(false,
                                                                                                                batch_size_received,
                                                                                                                tmp_ptr_layer_it);
    }
    // |END| Remaining layer(s). |END|
    
    // First block layer.
    this->Recurrent__Backward_Pass__Gradient__Residual__Layer__Loop(true,
                                                                                                            batch_size_received,
                                                                                                            tmp_ptr_layer_it);
    // |END| First block layer. |END|
}

void Neural_Network::Recurrent__Backward_Pass__Gradient__Residual__Layer__Loop(bool const is_block_input_layer_received,
                                                                                                                               size_t const batch_size_received,
                                                                                                                               struct Layer *&ptr_layer_it_received)
{
    size_t const tmp_layer_number_outputs(*ptr_layer_it_received->ptr_number_outputs);
    
    T_ *const tmp_ptr_array_layer_gradients(ptr_layer_it_received->ptr_array_derivative_outputs);
    
    struct Layer const *const tmp_ptr_next_layer_it(ptr_layer_it_received->next_connected_layers[0u]);
    
    // Clear past error(s).
    MEMSET(tmp_ptr_array_layer_gradients,
                  0,
                  this->batch_size * tmp_layer_number_outputs * this->number_recurrent_depth * sizeof(T_));
    // |END| Clear past error(s). |END|
    
    // Propagate the error(s) to the layer.
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_layer_it_received->next_connected_layers.size() > 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: In a residual block the layers can not have more than one forward connection. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
    }
#endif
    
    switch(tmp_ptr_next_layer_it->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            this->Recurrent__Backward_Pass__Average_Pooling__Loop(batch_size_received,
                                                                                                    tmp_layer_number_outputs,
                                                                                                    tmp_ptr_array_layer_gradients,
                                                                                                    tmp_ptr_next_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Backward_Pass__FC__Loop(batch_size_received,
                                                                                 tmp_layer_number_outputs,
                                                                                 tmp_ptr_array_layer_gradients,
                                                                                 tmp_ptr_next_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            this->Recurrent__Backward_Pass__LSTM__Loop(batch_size_received,
                                                                                     tmp_layer_number_outputs,
                                                                                     tmp_ptr_array_layer_gradients,
                                                                                     tmp_ptr_next_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            this->Recurrent__Backward_Pass__Max_Pooling__Loop(batch_size_received,
                                                                                               tmp_layer_number_outputs,
                                                                                               tmp_ptr_array_layer_gradients,
                                                                                               tmp_ptr_next_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
            this->Recurrent__Backward_Pass__Residual__Block__Loop(batch_size_received,
                                                                                                    tmp_layer_number_outputs,
                                                                                                    tmp_ptr_array_layer_gradients,
                                                                                                    tmp_ptr_next_layer_it);
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_next_layer_it->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_next_layer_it->type_layer].c_str());
                return;
    }
    // |END| Propagate the error(s) to the layer. |END|

    // Compute the gradients.
    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING: break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Backward_Pass__Gradient__Residual__FC__Loop(is_block_input_layer_received,
                                                                                                                batch_size_received,
                                                                                                                ptr_layer_it_received);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            if(ptr_layer_it_received->Use__Bidirectional())
            {
                this->Recurrent__Backward_Pass__Gradient__LSTM__Loop(true,
                                                                                                        batch_size_received,
                                                                                                        tmp_layer_number_outputs,
                                                                                                        tmp_ptr_array_layer_gradients,
                                                                                                        &ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Backward_Pass__Gradient__LSTM__Loop(false,
                                                                                                        batch_size_received,
                                                                                                        tmp_layer_number_outputs,
                                                                                                        tmp_ptr_array_layer_gradients,
                                                                                                        &ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer);
            }
            else
            {
                this->Recurrent__Backward_Pass__Gradient__LSTM__Loop(true,
                                                                                                        batch_size_received,
                                                                                                        tmp_layer_number_outputs,
                                                                                                        tmp_ptr_array_layer_gradients,
                                                                                                        ptr_layer_it_received);
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
    // |END| Compute the gradients. |END|
}

void Neural_Network::Recurrent__Backward_Pass__Gradient__Residual__FC__Loop(bool const is_block_input_layer_received,
                                                                                                                           size_t const batch_size_received,
                                                                                                                           struct Layer const *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(this->number_recurrent_depth); tmp_time_step_index--;)
    {
        this->Backward_Pass__Gradient__Residual__FC__Loop(is_block_input_layer_received,
                                                                                           tmp_time_step_index,
                                                                                           batch_size_received,
                                                                                           ptr_layer_it_received);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Backward_Pass__LSTM__Loop(size_t const time_step_index_received,
                                                                              size_t const batch_size_received,
                                                                              size_t const derivative_input_size_received,
                                                                              T_ const *const ptr_array_delta_input_block_inputs_received,
                                                                              T_ const *const ptr_array_delta_input_input_gates_received,
                                                                              T_ const *const ptr_array_delta_input_forget_gates_received,
                                                                              T_ const *const ptr_array_delta_input_output_gates_received,
                                                                              T_ *const ptr_array_derivative_inputs_received,
                                                                              struct Layer const *const ptr_layer_it_received)
{
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index;

    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    size_t const tmp_number_blocks(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units)),
                       tmp_number_cells(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units));
    
    T_ const *tmp_ptr_array_parameters;
    T_ *tmp_ptr_array_previous_layer_errors,
         tmp_error;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_block_data_timed_index = tmp_example_index * tmp_number_blocks + this->batch_size * tmp_number_blocks * time_step_index_received;

        tmp_cell_data_timed_index = tmp_example_index * tmp_number_cells + this->batch_size * tmp_number_cells * time_step_index_received;

        tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units;

        tmp_ptr_array_previous_layer_errors = ptr_array_derivative_inputs_received + tmp_example_index * derivative_input_size_received + this->batch_size * derivative_input_size_received * time_step_index_received;

        for(tmp_cell_index = 0_zu,
            tmp_block_index = 0_zu; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                          ++tmp_block_index)
        {
            // Cells inputs to previous neurons.
            for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                   ++tmp_cell_index)
            {
                tmp_error = ptr_array_delta_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index];

                tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;
                
                for(tmp_connection_index = tmp_ptr_cell_unit_it->last_index_feedforward_connection_cell_input - tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input; tmp_connection_index--;)
                {
                    tmp_ptr_array_previous_layer_errors[tmp_connection_index] += tmp_error * tmp_ptr_array_parameters[tmp_connection_index];
                }
            }
            // |END| Cell input to previous neurons. |END|

            // Input gate to previous neurons.
            tmp_error = ptr_array_delta_input_input_gates_received[tmp_block_data_timed_index + tmp_block_index];

            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
            
            for(tmp_connection_index = tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate; tmp_connection_index--;)
            {
                tmp_ptr_array_previous_layer_errors[tmp_connection_index] += tmp_error * tmp_ptr_array_parameters[tmp_connection_index];
            }
            // |END| Input gate to previous neurons. |END|

            // Forget gate to previous neurons.
            tmp_error = ptr_array_delta_input_forget_gates_received[tmp_block_data_timed_index + tmp_block_index];

            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
            
            for(tmp_connection_index = tmp_ptr_block_unit_it->last_index_feedforward_connection_forget_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate; tmp_connection_index--;)
            {
                tmp_ptr_array_previous_layer_errors[tmp_connection_index] += tmp_error * tmp_ptr_array_parameters[tmp_connection_index];
            }
            // |END| Forget gate to previous neurons. |END|

            // Output gate to previous neurons.
            tmp_error = ptr_array_delta_input_output_gates_received[tmp_block_data_timed_index + tmp_block_index];

            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;
            
            for(tmp_connection_index = tmp_ptr_block_unit_it->last_index_feedforward_connection_output_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate; tmp_connection_index--;)
            {
                tmp_ptr_array_previous_layer_errors[tmp_connection_index] += tmp_error * tmp_ptr_array_parameters[tmp_connection_index];
            }
            // |END| Output gate to previous neurons. |END|
        }
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Backward_Pass__LSTM_Derivative__Output__Loop(long long int const time_step_index_received,
                                                                                                           long long int const time_step_direction_received,
                                                                                                           long long int const time_step_prediction_end_received,
                                                                                                           size_t const batch_size_received,
                                                                                                           size_t const block_unit_size_received,
                                                                                                           size_t const cell_unit_size_received,
                                                                                                           T_ const *const ptr_array_delta_recurrent_block_inputs_received,
                                                                                                           T_ const *const ptr_array_delta_recurrent_input_gates_received,
                                                                                                           T_ const *const ptr_array_delta_recurrent_forget_gates_received,
                                                                                                           T_ const *const ptr_array_delta_recurrent_output_gates_received,
                                                                                                           struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_number_recurrents_connection(ptr_layer_it_received->ptr_array_block_units->last_index_recurrent_connection_input_gate - ptr_layer_it_received->ptr_array_block_units->first_index_recurrent_connection_input_gate);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_block_data_direction_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_direction_timed_index;

    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    T_ const *tmp_ptr_array_cell_input_parameters,
                  *tmp_ptr_array_input_gate_parameters,
                  *tmp_ptr_array_forget_gate_parameters,
                  *tmp_ptr_array_output_gate_parameters;
    T_ *tmp_ptr_array_delta_cells_outputs,
         tmp_activation,
         tmp_error,
         tmp_cell_input_error,
         tmp_input_gate_error,
         tmp_forget_gate_error,
         tmp_output_gate_error;
    
    if(time_step_index_received != time_step_prediction_end_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_block_data_direction_timed_index = tmp_example_index * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_direction_received);

            tmp_cell_data_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_cell_data_direction_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_direction_received);

            tmp_ptr_array_delta_cells_outputs = ptr_layer_it_received->ptr_array_block_units->ptr_array_delta_cells_outputs + tmp_cell_data_timed_index;

            // Cells inputs.
            for(tmp_cell_index = 0_zu,
                tmp_ptr_last_cell_unit = ptr_layer_it_received->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = ptr_layer_it_received->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                  ++tmp_cell_index)
            {
                tmp_cell_input_error = ptr_array_delta_recurrent_block_inputs_received[tmp_cell_data_direction_timed_index + tmp_cell_index];
                
                tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

                for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
                {
                    tmp_ptr_array_delta_cells_outputs[tmp_connection_index] += tmp_cell_input_error * tmp_ptr_array_cell_input_parameters[tmp_connection_index];
                }
            }
            // |END| Cells inputs. |END|

            for(tmp_block_index = 0_zu,
                tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                                                                                             ++tmp_block_index)
            {
                // Gates-recurrent.
                tmp_input_gate_error = ptr_array_delta_recurrent_input_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];
                tmp_forget_gate_error = ptr_array_delta_recurrent_forget_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];
                tmp_output_gate_error = ptr_array_delta_recurrent_output_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];
                
                tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;
                tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;
                tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;
                
                for(tmp_connection_index = tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate; tmp_connection_index--;)
                {
                    tmp_error = tmp_input_gate_error * tmp_ptr_array_input_gate_parameters[tmp_connection_index];
                    tmp_error += tmp_forget_gate_error * tmp_ptr_array_forget_gate_parameters[tmp_connection_index];
                    tmp_error += tmp_output_gate_error * tmp_ptr_array_output_gate_parameters[tmp_connection_index];
                    
                    tmp_ptr_array_delta_cells_outputs[tmp_connection_index] += tmp_error;
                }
                // |END| Gates-recurrent. |END|
            }

            for(tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                // Output gate, activation.
                tmp_error = 0_T;

                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    tmp_error += tmp_ptr_cell_unit_it->ptr_delta_cell_output[tmp_cell_data_timed_index] * tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index];
                }
                
                tmp_activation = tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index];

                tmp_ptr_block_unit_it->ptr_delta_outputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_derive(1_T, tmp_activation) * tmp_error;
                // |END| Output gate, activation. |END|
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);

            for(tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                // Output gate, activation.
                tmp_error = 0_T;

                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    tmp_error += tmp_ptr_cell_unit_it->ptr_delta_cell_output[tmp_cell_data_timed_index] * tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index];
                }
                
                tmp_activation = tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index];

                tmp_ptr_block_unit_it->ptr_delta_outputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_derive(1_T, tmp_activation) * tmp_error;
                // |END| Output gate, activation. |END|
            }
        }
    }
}

void Neural_Network::Backward_Pass__LSTM_Derivative__Cell_State_AF__Loop(long long int const time_step_index_received,
                                                                                                                      long long int const time_step_direction_received,
                                                                                                                      long long int const time_step_prediction_start_received,
                                                                                                                      long long int const time_step_prediction_end_received,
                                                                                                                      size_t const batch_size_received,
                                                                                                                      size_t const block_unit_size_received,
                                                                                                                      size_t const cell_unit_size_received,
                                                                                                                      T_ const *const ptr_array_summation_cell_states_received,
                                                                                                                      struct Layer *const ptr_layer_it_received)
{
    size_t tmp_example_index,
              tmp_block_data_timed_index,
              tmp_cell_layer_index,
              tmp_cell_data_timed_index;
    
    T_ tmp_output_gate;
    
    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const tmp_type_activation_function_io(ptr_layer_it_received->ptr_array_block_units->activation_function_io);
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_block_data_timed_index = tmp_example_index * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
        
        tmp_cell_data_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);
        
        for(tmp_cell_layer_index = 0_zu,
            tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
        {
            tmp_output_gate = tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index];

            for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                    ++tmp_cell_layer_index)
            {
                // Cell state.
                tmp_ptr_cell_unit_it->ptr_delta_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_delta_cell_output[tmp_cell_data_timed_index] * tmp_output_gate * this->Activation_Function_Derive(tmp_type_activation_function_io,
                                                                                                                                                                                                                                                                                                               ptr_array_summation_cell_states_received[tmp_cell_data_timed_index + tmp_cell_layer_index],
                                                                                                                                                                                                                                                                                                               1_T,
                                                                                                                                                                                                                                                                                                               tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index]);
                // |END| Cell state. |END|
            }
        }
    }
}

void Neural_Network::Backward_Pass__LSTM_Derivative__Gates_CIF_AF_State__Loop(long long int const time_step_index_received,
                                                                                                                                long long int const time_step_direction_received,
                                                                                                                                long long int const time_step_reverse_direction_received,
                                                                                                                                long long int const time_step_prediction_start_received,
                                                                                                                                long long int const time_step_prediction_end_received,
                                                                                                                                size_t const batch_size_received,
                                                                                                                                size_t const block_unit_size_received,
                                                                                                                                size_t const cell_unit_size_received,
                                                                                                                                struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_number_cells_per_block(static_cast<size_t>(ptr_layer_it_received->ptr_array_block_units->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_block_units->ptr_array_cell_units));
    size_t tmp_example_index,
              tmp_block_data_timed_index,
              tmp_block_data_direction_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_direction_timed_index,
              tmp_cell_data_reverse_direction_timed_index,
              tmp_first_index_peephole_input_gate,
              tmp_first_index_peephole_forget_gate,
              tmp_first_index_peephole_output_gate;
    
    T_ const *tmp_ptr_array_cell_inputs,
                 *tmp_ptr_array_cell_summation_inputs,
                 *tmp_ptr_array_cell_states_reverse_direction_timed,
                 *tmp_ptr_array_delta_cell_states_direction_timed;
    T_ tmp_input_gate_activation,
        tmp_forget_gate_activation,
        tmp_input_gate,
        tmp_forget_gate_dt,
        tmp_delta_input_gate_dt,
        tmp_delta_forget_gate_dt,
        tmp_delta_output_gate,
        tmp_cell_state_error,
        tmp_input_gate_error,
        tmp_forget_gate_error,
        *tmp_ptr_array_delta_cell_inputs,
        *tmp_ptr_array_delta_cell_states;
    
    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it;

    struct Cell_unit *tmp_ptr_block_it_cell_unit;

    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const tmp_type_activation_function_io(ptr_layer_it_received->ptr_array_block_units->activation_function_io);

    if(time_step_index_received != time_step_prediction_end_received && time_step_index_received != time_step_prediction_start_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
            tmp_block_data_direction_timed_index = tmp_example_index * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_direction_received);

            tmp_cell_data_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            tmp_cell_data_direction_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_direction_received);
            tmp_cell_data_reverse_direction_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);

            for(tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                tmp_ptr_block_it_cell_unit = tmp_ptr_block_unit_it->ptr_array_cell_units;

                tmp_first_index_peephole_input_gate = tmp_ptr_block_it_cell_unit->index_peephole_input_gate;
                tmp_first_index_peephole_forget_gate = tmp_ptr_block_it_cell_unit->index_peephole_forget_gate;
                tmp_first_index_peephole_output_gate = tmp_ptr_block_it_cell_unit->index_peephole_output_gate;

                tmp_ptr_array_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_cell_summation_inputs = tmp_ptr_block_it_cell_unit->ptr_summation_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_cell_states_reverse_direction_timed = tmp_ptr_block_it_cell_unit->ptr_cell_state + tmp_cell_data_reverse_direction_timed_index;
                tmp_ptr_array_delta_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_delta_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_states = tmp_ptr_block_it_cell_unit->ptr_delta_cell_state + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_states_direction_timed = tmp_ptr_block_it_cell_unit->ptr_delta_cell_state + tmp_cell_data_direction_timed_index;
                
                tmp_input_gate = tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index];
                tmp_forget_gate_dt = tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_direction_timed_index];
                
                tmp_delta_input_gate_dt = tmp_ptr_block_unit_it->ptr_delta_inputs_gates[tmp_block_data_direction_timed_index];
                tmp_delta_forget_gate_dt = tmp_ptr_block_unit_it->ptr_delta_forgets_gates[tmp_block_data_direction_timed_index];
                tmp_delta_output_gate = tmp_ptr_block_unit_it->ptr_delta_outputs_gates[tmp_block_data_timed_index];
                
                // Cells.
                for(tmp_cell_index = 0_zu; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    // Cell state.
                    tmp_cell_state_error = tmp_ptr_array_delta_cell_states[tmp_cell_index];
                    
                #ifndef NO_PEEPHOLE
                    tmp_cell_state_error += this->ptr_array_parameters[tmp_first_index_peephole_output_gate + tmp_cell_index] * tmp_delta_output_gate;
                    
                    tmp_cell_state_error += this->ptr_array_parameters[tmp_first_index_peephole_input_gate + tmp_cell_index] * tmp_delta_input_gate_dt;

                    tmp_cell_state_error += this->ptr_array_parameters[tmp_first_index_peephole_forget_gate + tmp_cell_index] * tmp_delta_forget_gate_dt;
                #endif
                    
                    tmp_cell_state_error += tmp_ptr_array_delta_cell_states_direction_timed[tmp_cell_index] * tmp_forget_gate_dt;
                    
                    tmp_ptr_array_delta_cell_states[tmp_cell_index] = tmp_cell_state_error;
                    // |END| Cell state. |END|

                    // Cell input.
                    tmp_ptr_array_delta_cell_inputs[tmp_cell_index] = tmp_cell_state_error * tmp_input_gate * this->Activation_Function_Derive(tmp_type_activation_function_io,
                                                                                                                                                                                                        tmp_ptr_array_cell_summation_inputs[tmp_cell_index],
                                                                                                                                                                                                        1_T,
                                                                                                                                                                                                        tmp_ptr_array_cell_inputs[tmp_cell_index]);
                    // |END| Cell input. |END|
                }
                // |END| Cells. |END|
                
                // Gates.
                tmp_input_gate_error = 0_T;
                tmp_forget_gate_error = 0_T;
                
                for(tmp_cell_index = 0_zu; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    tmp_cell_state_error = tmp_ptr_array_delta_cell_states[tmp_cell_index];

                    tmp_input_gate_error += tmp_cell_state_error * tmp_ptr_array_cell_inputs[tmp_cell_index];
                    tmp_forget_gate_error += tmp_cell_state_error * tmp_ptr_array_cell_states_reverse_direction_timed[tmp_cell_index];
                }
                
                tmp_input_gate_activation = tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index];
                tmp_forget_gate_activation = tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index];

                tmp_ptr_block_unit_it->ptr_delta_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_error * AF_SIGMOID_derive(1_T, tmp_input_gate_activation);
                tmp_ptr_block_unit_it->ptr_delta_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_error * AF_SIGMOID_derive(1_T, tmp_forget_gate_activation);
                // |END| Gates. |END|
            }
        }
    }
    else if(time_step_index_received != time_step_prediction_end_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
            tmp_block_data_direction_timed_index = tmp_example_index * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_direction_received);

            tmp_cell_data_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            tmp_cell_data_direction_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_direction_received);

            for(tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                tmp_ptr_block_it_cell_unit = tmp_ptr_block_unit_it->ptr_array_cell_units;

                tmp_first_index_peephole_input_gate = tmp_ptr_block_it_cell_unit->index_peephole_input_gate;
                tmp_first_index_peephole_forget_gate = tmp_ptr_block_it_cell_unit->index_peephole_forget_gate;
                tmp_first_index_peephole_output_gate = tmp_ptr_block_it_cell_unit->index_peephole_output_gate;

                tmp_ptr_array_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_cell_summation_inputs = tmp_ptr_block_it_cell_unit->ptr_summation_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_delta_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_states = tmp_ptr_block_it_cell_unit->ptr_delta_cell_state + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_states_direction_timed = tmp_ptr_block_it_cell_unit->ptr_delta_cell_state + tmp_cell_data_direction_timed_index;
                
                tmp_input_gate = tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index];
                tmp_forget_gate_dt = tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_direction_timed_index];

                tmp_delta_input_gate_dt = tmp_ptr_block_unit_it->ptr_delta_inputs_gates[tmp_block_data_direction_timed_index];
                tmp_delta_forget_gate_dt = tmp_ptr_block_unit_it->ptr_delta_forgets_gates[tmp_block_data_direction_timed_index];
                tmp_delta_output_gate = tmp_ptr_block_unit_it->ptr_delta_outputs_gates[tmp_block_data_timed_index];
                
                // Cells.
                for(tmp_cell_index = 0_zu; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    // Cell state.
                    tmp_cell_state_error = tmp_ptr_array_delta_cell_states[tmp_cell_index];
                    
                #ifndef NO_PEEPHOLE
                    tmp_cell_state_error += this->ptr_array_parameters[tmp_first_index_peephole_output_gate + tmp_cell_index] * tmp_delta_output_gate;
                    
                    tmp_cell_state_error += this->ptr_array_parameters[tmp_first_index_peephole_input_gate + tmp_cell_index] * tmp_delta_input_gate_dt;

                    tmp_cell_state_error += this->ptr_array_parameters[tmp_first_index_peephole_forget_gate + tmp_cell_index] * tmp_delta_forget_gate_dt;
                #endif

                    tmp_cell_state_error += tmp_ptr_array_delta_cell_states_direction_timed[tmp_cell_index] * tmp_forget_gate_dt;
                    
                    tmp_ptr_array_delta_cell_states[tmp_cell_index] = tmp_cell_state_error;
                    // |END| Cell state. |END|

                    // Cell input.
                    tmp_ptr_array_delta_cell_inputs[tmp_cell_index] = tmp_cell_state_error * tmp_input_gate * this->Activation_Function_Derive(tmp_type_activation_function_io,
                                                                                                                                                                                                        tmp_ptr_array_cell_summation_inputs[tmp_cell_index],
                                                                                                                                                                                                        1_T,
                                                                                                                                                                                                        tmp_ptr_array_cell_inputs[tmp_cell_index]);
                    // |END| Cell input. |END|
                }
                // |END| Cells. |END|
                
                // Gates.
                tmp_input_gate_error = 0_T;
                
                for(tmp_cell_index = 0_zu; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    tmp_input_gate_error += tmp_ptr_array_delta_cell_states[tmp_cell_index] * tmp_ptr_array_cell_inputs[tmp_cell_index];
                }
                
                tmp_input_gate_activation = tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index];

                tmp_ptr_block_unit_it->ptr_delta_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_error * AF_SIGMOID_derive(1_T, tmp_input_gate_activation);
                // |END| Gates. |END|
            }
        }
    }
    else if(time_step_index_received != time_step_prediction_start_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            tmp_cell_data_reverse_direction_timed_index = tmp_example_index * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);
            
            for(tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                tmp_ptr_block_it_cell_unit = tmp_ptr_block_unit_it->ptr_array_cell_units;

                tmp_first_index_peephole_output_gate = tmp_ptr_block_it_cell_unit->index_peephole_output_gate;

                tmp_ptr_array_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_cell_summation_inputs = tmp_ptr_block_it_cell_unit->ptr_summation_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_cell_states_reverse_direction_timed = tmp_ptr_block_it_cell_unit->ptr_cell_state + tmp_cell_data_reverse_direction_timed_index;
                tmp_ptr_array_delta_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_delta_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_states = tmp_ptr_block_it_cell_unit->ptr_delta_cell_state + tmp_cell_data_timed_index;
                
                tmp_input_gate = tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index];

                tmp_delta_output_gate = tmp_ptr_block_unit_it->ptr_delta_outputs_gates[tmp_block_data_timed_index];
                
                // Cells.
                for(tmp_cell_index = 0_zu; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    // Cell state.
                    tmp_cell_state_error = tmp_ptr_array_delta_cell_states[tmp_cell_index];
                    
                #ifndef NO_PEEPHOLE
                    tmp_cell_state_error += this->ptr_array_parameters[tmp_first_index_peephole_output_gate + tmp_cell_index] * tmp_delta_output_gate;
                #endif

                    tmp_ptr_array_delta_cell_states[tmp_cell_index] = tmp_cell_state_error;
                    // |END| Cell state. |END|

                    // Cell input.
                    tmp_ptr_array_delta_cell_inputs[tmp_cell_index] = tmp_cell_state_error * tmp_input_gate * this->Activation_Function_Derive(tmp_type_activation_function_io,
                                                                                                                                                                                                        tmp_ptr_array_cell_summation_inputs[tmp_cell_index],
                                                                                                                                                                                                        1_T,
                                                                                                                                                                                                        tmp_ptr_array_cell_inputs[tmp_cell_index]);
                    // |END| Cell input. |END|
                }
                // |END| Cells. |END|
                
                // Gates.
                tmp_input_gate_error = 0_T;
                tmp_forget_gate_error = 0_T;
                
                for(tmp_cell_index = 0_zu; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    tmp_cell_state_error = tmp_ptr_array_delta_cell_states[tmp_cell_index];

                    tmp_input_gate_error += tmp_cell_state_error * tmp_ptr_array_cell_inputs[tmp_cell_index];
                    tmp_forget_gate_error += tmp_cell_state_error * tmp_ptr_array_cell_states_reverse_direction_timed[tmp_cell_index];
                }
                
                tmp_input_gate_activation = tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index];
                tmp_forget_gate_activation = tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index];

                tmp_ptr_block_unit_it->ptr_delta_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_error * AF_SIGMOID_derive(1_T, tmp_input_gate_activation);
                tmp_ptr_block_unit_it->ptr_delta_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_error * AF_SIGMOID_derive(1_T, tmp_forget_gate_activation);
                // |END| Gates. |END|
            }
        }
    }
}
