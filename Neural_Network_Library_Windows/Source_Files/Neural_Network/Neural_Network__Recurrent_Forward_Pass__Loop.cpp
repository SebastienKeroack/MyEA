#include "stdafx.hpp"

#include <Math/Math.hpp>

#include <Neural_Network/Neural_Network.hpp>

void Neural_Network::RNN__Forward_Pass_Batch__Loop(size_t const batch_size_received,
                                                                                    T_ const *const *const ptr_array_inputs_received,
                                                                                    struct Layer *const ptr_first_layer_received,
                                                                                    struct Layer const *const ptr_last_layer_received)
{
    struct Layer const *tmp_ptr_previous_connected_layer;
    struct Layer *tmp_ptr_layer_it(ptr_first_layer_received + 1);
    
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(batch_size_received > this->batch_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Overflow of memory. Unable to process %zu examples out of %zu allocated examples. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 batch_size_received,
                                 this->batch_size,
                                 __LINE__);

        return;
    }
#endif
    
    // Training mode.
    if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // If the network use normalization.
        if(this->Use__Normalization())
        {
            // Set all mean to zero.
            MEMSET(this->ptr_array_normalized_batch_units_means,
                          0,
                          this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
            // |END| Set all mean to zero. |END|

            // Set all variance to zero.
            MEMSET(this->ptr_array_normalized_batch_units_variances,
                          0,
                          this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
            // |END| Set all variance to zero. |END|
        }
    }

    // Input layer.
    this->RNN__Assign_Inputs__Loop(batch_size_received, ptr_array_inputs_received);
    // |END| Input layer. |END|
    
    // Loop through each layer and do a forward propagation.
    for(; tmp_ptr_layer_it != ptr_last_layer_received; ++tmp_ptr_layer_it)
    {
        tmp_ptr_previous_connected_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                this->Recurrent__Forward_Pass__Average_Pooling__Loop(batch_size_received,
                                                                                                      *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                      tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                      tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Recurrent__Forward_Pass__FC__Loop(batch_size_received,
                                                                                  *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                  tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                  tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(tmp_ptr_layer_it->Use__Bidirectional())
                {
                    this->Recurrent__Forward_Pass__LSTM__Loop(true,
                                                                                          batch_size_received,
                                                                                          *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                          tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                          &tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer);
                    this->Recurrent__Forward_Pass__LSTM__Loop(false,
                                                                                          batch_size_received,
                                                                                          *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                          tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                          &tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer);
                }
                else
                {
                    this->Recurrent__Forward_Pass__LSTM__Loop(true,
                                                                                          batch_size_received,
                                                                                          *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                          tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                          tmp_ptr_layer_it);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                this->Recurrent__Forward_Pass__Max_Pooling__Loop(batch_size_received,
                                                                                                *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL: this->Recurrent__Forward_Pass__Residual__Loop(batch_size_received, tmp_ptr_layer_it); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    return;
        }
    }
}

void Neural_Network::RNN__Forward_Pass_Batch__Pre_Training__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_inputs_received)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_array_layers + this->pre_training_level),
                               *tmp_ptr_previous_connected_layer;
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(batch_size_received > this->batch_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Overflow of memory. Unable to process %zu examples out of %zu allocated examples. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 batch_size_received,
                                 this->batch_size,
                                 __LINE__);

        return;
    }
    else if(this->pre_training_level == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The neural network use the pre-training function without the mode pre-training activate. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return;
    }
#endif
    
    // Training mode.
    if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // If the network use normalization.
        if(this->Use__Normalization())
        {
            // Set all mean to zero.
            MEMSET(this->ptr_array_normalized_batch_units_means,
                        0,
                        this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
            // |END| Set all mean to zero. |END|

            // Set all variance to zero.
            MEMSET(this->ptr_array_normalized_batch_units_variances,
                        0,
                        this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
            // |END| Set all variance to zero. |END|
        }
    }

    // Input layer.
    this->RNN__Assign_Inputs__Loop(batch_size_received, ptr_array_inputs_received);
    // |END| Input layer. |END|
    
    // Loop through each encoded layer and do a forward propagation.
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        tmp_ptr_previous_connected_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Recurrent__Forward_Pass__Encode__FC__Loop(batch_size_received,
                                                                                                     *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                     tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                     tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(tmp_ptr_layer_it->Use__Bidirectional())
                {
                    this->Recurrent__Forward_Pass__Encode__LSTM__Loop(true,
                                                                                                         batch_size_received,
                                                                                                         *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                         tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                         &tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer);
                    this->Recurrent__Forward_Pass__Encode__LSTM__Loop(false,
                                                                                                         batch_size_received,
                                                                                                         *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                         tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                         &tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer);
                }
                else
                {
                    this->Recurrent__Forward_Pass__Encode__LSTM__Loop(true,
                                                                                                        batch_size_received,
                                                                                                        *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                        tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                        tmp_ptr_layer_it);
                }
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    return;
        }
    }
    
    // Coded level part.
    tmp_ptr_previous_connected_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

    switch(tmp_ptr_layer_it->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Forward_Pass__Code__FC__Loop(batch_size_received,
                                                                                         *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                         tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                         tmp_ptr_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            if(tmp_ptr_layer_it->Use__Bidirectional())
            {
                this->Recurrent__Forward_Pass__Code__LSTM__Loop(true,
                                                                                                 batch_size_received,
                                                                                                 *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                 tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                 &tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Forward_Pass__Code__LSTM__Loop(false,
                                                                                                 batch_size_received,
                                                                                                 *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                 tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                 &tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer);
            }
            else
            {
                this->Recurrent__Forward_Pass__Code__LSTM__Loop(true,
                                                                                                 batch_size_received,
                                                                                                 *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                 tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                 tmp_ptr_layer_it);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
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
            this->Recurrent__Forward_Pass__Decode__FC__Loop(batch_size_received,
                                                                                                 *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                 tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                 tmp_ptr_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            if(tmp_ptr_layer_it->Use__Bidirectional())
            {
                this->Recurrent__Forward_Pass__Decode__LSTM__Loop(true,
                                                                                                     batch_size_received,
                                                                                                     *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                     tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                     &tmp_ptr_layer_it->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Forward_Pass__Decode__LSTM__Loop(false,
                                                                                                     batch_size_received,
                                                                                                     *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                     tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                     &tmp_ptr_layer_it->ptr_Bidirectional_Layer->backward_layer);
            }
            else
            {
                this->Recurrent__Forward_Pass__Decode__LSTM__Loop(true,
                                                                                                    batch_size_received,
                                                                                                    *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                    tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                    tmp_ptr_layer_it);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
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

void Neural_Network::Recurrent__Forward_Pass__Average_Pooling__Loop(size_t const batch_size_received,
                                                                                                             size_t const input_unit_size_received,
                                                                                                             T_ const *const ptr_array_inputs_received,
                                                                                                             struct Layer *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Forward_Pass__Average_Pooling__Loop(tmp_time_step_index,
                                                                            batch_size_received,
                                                                            input_unit_size_received,
                                                                            ptr_array_inputs_received,
                                                                            ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Dropout__Bernoulli__Inference__Loop(size_t const batch_size_received,
                                                                                                                                 size_t const input_unit_size_received,
                                                                                                                                 T_ const retention_probability_received,
                                                                                                                                 T_ *const ptr_array_inputs_received)
{
    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(tmp_time_step_index,
                                                                                                batch_size_received,
                                                                                                input_unit_size_received,
                                                                                                retention_probability_received,
                                                                                                ptr_array_inputs_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Dropout__ShakeDrop__Loop(size_t const batch_size_received,
                                                                                                                   size_t const input_unit_size_received,
                                                                                                                   bool *const ptr_array_mask_dopout_shakedrop_received,
                                                                                                                   T_ const lower_bound_received,
                                                                                                                   T_ const upper_bound_received,
                                                                                                                   T_ const dropout_probability_received,
                                                                                                                   T_ *const ptr_array_inputs_received)
{
    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Forward_Pass__Dropout__ShakeDrop__Loop(tmp_time_step_index,
                                                                                   batch_size_received,
                                                                                   input_unit_size_received,
                                                                                   ptr_array_mask_dopout_shakedrop_received,
                                                                                   lower_bound_received,
                                                                                   upper_bound_received,
                                                                                   dropout_probability_received,
                                                                                   ptr_array_inputs_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__FC__Loop(size_t const batch_size_received,
                                                                                         size_t const input_unit_size_received,
                                                                                         T_ const *const ptr_array_inputs_received,
                                                                                         struct Layer *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Forward_Pass__FC__Loop(tmp_time_step_index,
                                                        batch_size_received,
                                                        input_unit_size_received,
                                                        ptr_array_inputs_received,
                                                        ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Encode__FC__Loop(size_t const batch_size_received,
                                                                                                            size_t const input_unit_size_received,
                                                                                                            T_ const *const ptr_array_inputs_received,
                                                                                                            struct Layer *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Forward_Pass__Encode__FC__Loop(tmp_time_step_index,
                                                                       batch_size_received,
                                                                       input_unit_size_received,
                                                                       ptr_array_inputs_received,
                                                                       ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Code__FC__Loop(size_t const batch_size_received,
                                                                                                         size_t const input_unit_size_received,
                                                                                                         T_ const *const ptr_array_inputs_received,
                                                                                                         struct Layer *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Forward_Pass__Code__FC__Loop(tmp_time_step_index,
                                                                   batch_size_received,
                                                                   input_unit_size_received,
                                                                   ptr_array_inputs_received,
                                                                   ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Decode__FC__Loop(size_t const batch_size_received,
                                                                                                            size_t const input_unit_size_received,
                                                                                                            T_ const *const ptr_array_inputs_received,
                                                                                                            struct Layer *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Forward_Pass__Decode__FC__Loop(tmp_time_step_index,
                                                                       batch_size_received,
                                                                       input_unit_size_received,
                                                                       ptr_array_inputs_received,
                                                                       ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__LSTM__Loop(bool const forward_layer_received,
                                                                                             size_t const batch_size_received,
                                                                                             size_t const input_unit_size_received,
                                                                                             T_ const *const ptr_array_inputs_received,
                                                                                             struct Layer *const ptr_layer_it_received)
{
    long long int tmp_time_step_index,
                       tmp_time_step_reverse_direction,
                       tmp_time_step_start(forward_layer_received ? 0ll : static_cast<long long int>(this->number_recurrent_depth - 1_zu)),
                       tmp_time_step_end(forward_layer_received ? static_cast<long long int>(this->number_recurrent_depth) : -1ll);
    
    for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? ++tmp_time_step_index : --tmp_time_step_index)
    {
        tmp_time_step_reverse_direction = forward_layer_received ? (tmp_time_step_index - 1ll) : (tmp_time_step_index + 1ll);

        this->Forward_Pass__LSTM__Loop(tmp_time_step_index,
                                                             tmp_time_step_reverse_direction,
                                                             tmp_time_step_start,
                                                             batch_size_received,
                                                             input_unit_size_received,
                                                             ptr_array_inputs_received,
                                                             ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Encode__LSTM__Loop(bool const forward_layer_received,
                                                                                                            size_t const batch_size_received,
                                                                                                            size_t const input_unit_size_received,
                                                                                                            T_ const *const ptr_array_inputs_received,
                                                                                                            struct Layer *const ptr_layer_it_received)
{
    long long int tmp_time_step_index,
                       tmp_time_step_reverse_direction,
                       tmp_time_step_start(forward_layer_received ? 0ll : static_cast<long long int>(this->number_recurrent_depth - 1_zu)),
                       tmp_time_step_end(forward_layer_received ? static_cast<long long int>(this->number_recurrent_depth) : -1ll);
    
    for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? ++tmp_time_step_index : --tmp_time_step_index)
    {
        tmp_time_step_reverse_direction = forward_layer_received ? (tmp_time_step_index - 1ll) : (tmp_time_step_index + 1ll);

        this->Forward_Pass__Encode__LSTM__Loop(tmp_time_step_index,
                                                                           tmp_time_step_reverse_direction,
                                                                           tmp_time_step_start,
                                                                           batch_size_received,
                                                                           input_unit_size_received,
                                                                           ptr_array_inputs_received,
                                                                           ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Code__LSTM__Loop(bool const forward_layer_received,
                                                                                                        size_t const batch_size_received,
                                                                                                        size_t const input_unit_size_received,
                                                                                                        T_ const *const ptr_array_inputs_received,
                                                                                                        struct Layer *const ptr_layer_it_received)
{
    long long int tmp_time_step_index,
                       tmp_time_step_reverse_direction,
                       tmp_time_step_start(forward_layer_received ? 0ll : static_cast<long long int>(this->number_recurrent_depth - 1_zu)),
                       tmp_time_step_end(forward_layer_received ? static_cast<long long int>(this->number_recurrent_depth) : -1ll);
    
    for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? ++tmp_time_step_index : --tmp_time_step_index)
    {
        tmp_time_step_reverse_direction = forward_layer_received ? (tmp_time_step_index - 1ll) : (tmp_time_step_index + 1ll);

        this->Forward_Pass__Code__LSTM__Loop(tmp_time_step_index,
                                                                       tmp_time_step_reverse_direction,
                                                                       tmp_time_step_start,
                                                                       batch_size_received,
                                                                       input_unit_size_received,
                                                                       ptr_array_inputs_received,
                                                                       ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Decode__LSTM__Loop(bool const forward_layer_received,
                                                                                                            size_t const batch_size_received,
                                                                                                            size_t const input_unit_size_received,
                                                                                                            T_ const *const ptr_array_inputs_received,
                                                                                                            struct Layer *const ptr_layer_it_received)
{
    long long int tmp_time_step_index,
                       tmp_time_step_reverse_direction,
                       tmp_time_step_start(forward_layer_received ? 0ll : static_cast<long long int>(this->number_recurrent_depth - 1_zu)),
                       tmp_time_step_end(forward_layer_received ? static_cast<long long int>(this->number_recurrent_depth) : -1ll);
    
    for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? ++tmp_time_step_index : --tmp_time_step_index)
    {
        tmp_time_step_reverse_direction = forward_layer_received ? (tmp_time_step_index - 1ll) : (tmp_time_step_index + 1ll);

        this->Forward_Pass__Decode__LSTM__Loop(tmp_time_step_index,
                                                                           tmp_time_step_reverse_direction,
                                                                           tmp_time_step_start,
                                                                           batch_size_received,
                                                                           input_unit_size_received,
                                                                           ptr_array_inputs_received,
                                                                           ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Max_Pooling__Loop(size_t const batch_size_received,
                                                                                                       size_t const input_unit_size_received,
                                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                                       struct Layer *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Forward_Pass__Max_Pooling__Loop(tmp_time_step_index,
                                                                       batch_size_received,
                                                                       input_unit_size_received,
                                                                       ptr_array_inputs_received,
                                                                       ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Residual__Loop(size_t const batch_size_received, struct Layer *&ptr_layer_it_received)
{
    size_t tmp_time_step_index;

    T_ *tmp_ptr_array_inputs;
    
    struct Layer const *const tmp_ptr_end_block_layer(ptr_layer_it_received + ptr_layer_it_received->block_depth + 1),
                               *tmp_ptr_previous_connected_layer;
    struct Layer *const tmp_ptr_residual_layer(ptr_layer_it_received);
    
    union Normalized_unit *const tmp_ptr_residual_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // First block layer.
    this->Recurrent__Forward_Pass__Residual__Layer__Loop(true,
                                                                                          batch_size_received,
                                                                                          ++ptr_layer_it_received);
    // |END| First block layer. |END|

    // Remaining layer(s).
    for(++ptr_layer_it_received; ptr_layer_it_received != tmp_ptr_end_block_layer; ++ptr_layer_it_received)
    {
        this->Recurrent__Forward_Pass__Residual__Layer__Loop(false,
                                                                                              batch_size_received,
                                                                                              ptr_layer_it_received);
    }
    // |END| Remaining layer(s). |END|
    
    // Assign layer iterator to the last layer inside the block.
    --ptr_layer_it_received;

    // Shortcut.
    //  Assign previous layer iterator to the previously connected layer from the residual layer.
    tmp_ptr_previous_connected_layer = tmp_ptr_residual_layer->previous_connected_layers[0u];
    
    //  Store the input(s) (block, last layer output(s)).
    tmp_ptr_array_inputs = ptr_layer_it_received->ptr_array_outputs;
    
    // Normalization.
    if(tmp_ptr_residual_layer->Use__Normalization())
    {
        // Training mode.
        if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
        {
            switch(tmp_ptr_residual_layer->type_normalization)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                    {
                        this->Forward_Pass__Batch_Normalization__Training__Loop(tmp_time_step_index,
                                                                                                                 batch_size_received,
                                                                                                                 *ptr_layer_it_received->ptr_number_outputs,
                                                                                                                 tmp_ptr_array_inputs,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                    {
                        this->Forward_Pass__Batch_Renormalization__Training__Loop(tmp_time_step_index,
                                                                                                                    batch_size_received,
                                                                                                                    *ptr_layer_it_received->ptr_number_outputs,
                                                                                                                    tmp_ptr_array_inputs,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                    }
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_normalization,
                                             MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                        break;
            }
        }
        // Inference mode.
        else
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
            {
                this->Forward_Pass__Batch_Normalization__Inference__Loop(tmp_time_step_index,
                                                                                                          batch_size_received,
                                                                                                          *ptr_layer_it_received->ptr_number_outputs,
                                                                                                          tmp_ptr_array_inputs,
                                                                                                          tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                          tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                          tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                          tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                          tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            }
        }

        // Store the new inputs (value normalize).
        tmp_ptr_array_inputs = tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
    }
    
    // Dropout.
    if(tmp_ptr_residual_layer->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP)
    {
        // If the state of propagation is strictly at training.
        if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
        {
            this->Recurrent__Forward_Pass__Dropout__ShakeDrop__Loop(batch_size_received,
                                                                                                        *ptr_layer_it_received->ptr_number_outputs,
                                                                                                        tmp_ptr_residual_layer->ptr_array__mask__dropout__shakedrop,
                                                                                                        -1_T,
                                                                                                        1_T,
                                                                                                        tmp_ptr_residual_layer->dropout_values[0u],
                                                                                                        tmp_ptr_array_inputs);
        }
        // Inference mode.
        else
        {
            this->Recurrent__Forward_Pass__Dropout__Bernoulli__Inference__Loop(batch_size_received,
                                                                                                                      *ptr_layer_it_received->ptr_number_outputs,
                                                                                                                      1_T - tmp_ptr_residual_layer->dropout_values[0u],
                                                                                                                      tmp_ptr_array_inputs);
        }
    }

    //  Zero-padded identity-mapping shortcut.
    this->Recurrent__Forward_Pass__Zero_Padded_Identity__Loop(batch_size_received,
                                                                                                 *tmp_ptr_previous_connected_layer->ptr_number_outputs, // Shortcut.
                                                                                                 *ptr_layer_it_received->ptr_number_outputs, // Block, last layer.
                                                                                                 tmp_ptr_previous_connected_layer->ptr_array_outputs, // Shortcut.
                                                                                                 tmp_ptr_array_inputs, // Block, last layer.
                                                                                                 tmp_ptr_residual_layer);
    // |END| Shortcut. |END|
}

void Neural_Network::Recurrent__Forward_Pass__Residual__Layer__Loop(bool const is_block_input_layer_received,
                                                                                                             size_t const batch_size_received,
                                                                                                             struct Layer *&ptr_layer_it_received)
{
    struct Layer const *const tmp_ptr_previous_connected_layer(ptr_layer_it_received->previous_connected_layers[0u]);

    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            this->Recurrent__Forward_Pass__Average_Pooling__Loop(batch_size_received,
                                                                                                  *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                                  tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                                  ptr_layer_it_received);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Forward_Pass__Residual__FC__Loop(is_block_input_layer_received,
                                                                                              batch_size_received,
                                                                                              *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                              tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                              ptr_layer_it_received);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            if(ptr_layer_it_received->Use__Bidirectional())
            {
                this->Recurrent__Forward_Pass__LSTM__Loop(true,
                                                                                      batch_size_received,
                                                                                      *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                      tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                      &ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Forward_Pass__LSTM__Loop(false,
                                                                                      batch_size_received,
                                                                                      *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                      tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                      &ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer);
            }
            else
            {
                this->Recurrent__Forward_Pass__LSTM__Loop(true,
                                                                                      batch_size_received,
                                                                                      *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                      tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                      ptr_layer_it_received);
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            this->Recurrent__Forward_Pass__Max_Pooling__Loop(batch_size_received,
                                                                                            *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                            tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                            ptr_layer_it_received);
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str());
                return;
    }
}

void Neural_Network::Recurrent__Forward_Pass__Residual__FC__Loop(bool const is_block_input_layer_received,
                                                                                                         size_t const batch_size_received,
                                                                                                         size_t const input_unit_size_received,
                                                                                                         T_ const *const ptr_array_inputs_received,
                                                                                                         struct Layer *const ptr_layer_it_received)
{
    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Forward_Pass__Residual__FC__Loop(is_block_input_layer_received,
                                                                        tmp_time_step_index,
                                                                        batch_size_received,
                                                                        input_unit_size_received,
                                                                        ptr_array_inputs_received,
                                                                        ptr_layer_it_received);
    }
}

void Neural_Network::Recurrent__Forward_Pass__Zero_Padded_Identity__Loop(size_t const batch_size_received,
                                                                                                                    size_t const size_A_received,
                                                                                                                    size_t const size_B_received,
                                                                                                                    T_ const *const ptr_array_A_received,
                                                                                                                    T_ const *const ptr_array_B_received,
                                                                                                                    struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_padding(ptr_layer_it_received->pooling_values[2u]);

    T_ *const tmp_ptr_array_outputs(ptr_layer_it_received->ptr_array_basic_units->ptr_array_values);

    for(size_t tmp_time_step_index(0_zu); tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        this->Forward_Pass__Zero_Padded_Identity__Loop(tmp_time_step_index,
                                                                                   batch_size_received,
                                                                                   size_A_received,
                                                                                   size_B_received,
                                                                                   tmp_padding,
                                                                                   ptr_array_A_received,
                                                                                   ptr_array_B_received,
                                                                                   tmp_ptr_array_outputs);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Forward_Pass__FC_Ind_RNN__Loop(size_t const time_step_index_received,
                                                                                      size_t const batch_size_received,
                                                                                      size_t const input_size_received,
                                                                                      T_ const *const ptr_array_parameters_received,
                                                                                      T_ const *const ptr_array_AFs_received,
                                                                                      T_ const *const ptr_array_inputs_received,
                                                                                      T_ *const ptr_array_outputs_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_input_previous_timed_batched_index(this->batch_size * input_size_received * (time_step_index_received - 1_zu));
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_inverse_timed_AFs,
                 *tmp_ptr_array_inputs;
    T_ *tmp_ptr_array_outputs;

    if(time_step_index_received != 0_zu)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            tmp_ptr_array_inverse_timed_AFs = ptr_array_AFs_received + tmp_example_index * input_size_received + tmp_input_previous_timed_batched_index;
            tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_outputs[tmp_input_index] = tmp_ptr_array_inputs[tmp_input_index] + ptr_array_parameters_received[tmp_input_index] * tmp_ptr_array_inverse_timed_AFs[tmp_input_index]; }
        }
    }
    else
    {
    #if defined(COMPILE_AUTODIFF)
        // Identity.
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_outputs[tmp_input_index] = tmp_ptr_array_inputs[tmp_input_index]; }
        }
    #else
        MEMCPY(ptr_array_outputs_received + tmp_input_timed_batched_index,
                       ptr_array_inputs_received + tmp_input_timed_batched_index,
                       this->batch_size * input_size_received * sizeof(T_));
    #endif
    }
}

void Neural_Network::Forward_Pass__LSTM__Gates_CIFO__Loop(long long int const time_step_index_received,
                                                                                                 long long int const time_step_reverse_direction_received,
                                                                                                 long long int const time_step_prediction_start_received,
                                                                                                 size_t const batch_size_received,
                                                                                                 size_t const layer_block_unit_size_received,
                                                                                                 size_t const layer_cell_unit_size_received,
                                                                                                 size_t const input_unit_size_received,
                                                                                                 T_ const *const ptr_array_inputs_received,
                                                                                                 struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_number_inputs_connections(ptr_layer_it_received->ptr_array_block_units->last_index_feedforward_connection_input_gate - ptr_layer_it_received->ptr_array_block_units->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrents_connection(ptr_layer_it_received->ptr_array_block_units->last_index_recurrent_connection_input_gate - ptr_layer_it_received->ptr_array_block_units->first_index_recurrent_connection_input_gate);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_data_timed_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_reverse_direction_timed_index;
    
    T_ const *tmp_ptr_array_previous_layer_outputs,
                  *tmp_ptr_array_layer_reverse_direction_timed_outputs,
                  *tmp_ptr_array_cell_input_parameters,
                  *tmp_ptr_array_input_gate_parameters,
                  *tmp_ptr_array_forget_gate_parameters,
                  *tmp_ptr_array_output_gate_parameters;
    T_ tmp_cell_input_summation,
        tmp_input_gate_summation,
        tmp_forget_gate_summation,
        tmp_output_gate_summation;

    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    if(time_step_index_received != time_step_prediction_start_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_cell_data_reverse_direction_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);

            tmp_ptr_array_previous_layer_outputs = ptr_array_inputs_received + tmp_example_index * input_unit_size_received + this->batch_size * input_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_ptr_array_layer_reverse_direction_timed_outputs = ptr_layer_it_received->ptr_array_cell_units->ptr_cell_output + tmp_cell_data_reverse_direction_timed_index;
            
            for(tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    // Cell-Input.
                    tmp_cell_input_summation = 0_T;

                    tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

                    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                    {
                        tmp_cell_input_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_cell_input_parameters[tmp_connection_index];
                    }

                    tmp_ptr_cell_unit_it->ptr_summation_input_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    // |END| Cell-Input. |END|

                    // Cell-Recurrent.
                    tmp_cell_input_summation = 0_T;

                    tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

                    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
                    {
                        tmp_cell_input_summation += tmp_ptr_array_layer_reverse_direction_timed_outputs[tmp_connection_index] * tmp_ptr_array_cell_input_parameters[tmp_connection_index];
                    }

                    tmp_ptr_cell_unit_it->ptr_summation_recurrent_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    // |END| Cell-Recurrent. |END|
                }

                // Gates-Input.
                tmp_input_gate_summation = 0_T;
                tmp_forget_gate_summation = 0_T;
                tmp_output_gate_summation = 0_T;

                tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
                tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
                tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;
            
                for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                {
                    tmp_input_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_input_gate_parameters[tmp_connection_index];
                    tmp_forget_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_forget_gate_parameters[tmp_connection_index];
                    tmp_output_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_output_gate_parameters[tmp_connection_index];
                }

                tmp_ptr_block_unit_it->ptr_summation_input_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_input_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_input_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;
                // |END| Gates-Input. |END|

                // Gates-Recurrent.
                tmp_input_gate_summation = 0_T;
                tmp_forget_gate_summation = 0_T;
                tmp_output_gate_summation = 0_T;
                
                tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;
                tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;
                tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;
                
                for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
                {
                    tmp_input_gate_summation += tmp_ptr_array_layer_reverse_direction_timed_outputs[tmp_connection_index] * tmp_ptr_array_input_gate_parameters[tmp_connection_index];
                    tmp_forget_gate_summation += tmp_ptr_array_layer_reverse_direction_timed_outputs[tmp_connection_index] * tmp_ptr_array_forget_gate_parameters[tmp_connection_index];
                    tmp_output_gate_summation += tmp_ptr_array_layer_reverse_direction_timed_outputs[tmp_connection_index] * tmp_ptr_array_output_gate_parameters[tmp_connection_index];
                }

                tmp_ptr_block_unit_it->ptr_summation_recurrent_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_recurrent_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_recurrent_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;
                // |END| Gates-Recurrent. |END|
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_ptr_array_previous_layer_outputs = ptr_array_inputs_received + tmp_example_index * input_unit_size_received + this->batch_size * input_unit_size_received * static_cast<size_t>(time_step_index_received);

            for(tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    // Cell-Input.
                    tmp_cell_input_summation = 0_T;

                    tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

                    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                    {
                        tmp_cell_input_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_cell_input_parameters[tmp_connection_index];
                    }

                    tmp_ptr_cell_unit_it->ptr_summation_input_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    // |END| Cell-Input. |END|
                }

                // Gates-Input.
                tmp_input_gate_summation = 0_T;
                tmp_forget_gate_summation = 0_T;
                tmp_output_gate_summation = 0_T;

                tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
                tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
                tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;
            
                for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                {
                    tmp_input_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_input_gate_parameters[tmp_connection_index];
                    tmp_forget_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_forget_gate_parameters[tmp_connection_index];
                    tmp_output_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_output_gate_parameters[tmp_connection_index];
                }

                tmp_ptr_block_unit_it->ptr_summation_input_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_input_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_input_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;
                // |END| Gates-Input. |END|
            }
        }
    }
}

void Neural_Network::Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(long long int const time_step_index_received,
                                                                                                              long long int const time_step_reverse_direction_received,
                                                                                                              long long int const time_step_prediction_start_received,
                                                                                                              size_t const batch_size_received,
                                                                                                              size_t const layer_block_unit_size_received,
                                                                                                              size_t const layer_cell_unit_size_received,
                                                                                                              T_ const *const ptr_array_summation_input_block_inputs_received,
                                                                                                              T_ const *const ptr_array_summation_recurrent_block_inputs_received,
                                                                                                              T_ const *const ptr_array_summation_input_inputs_gates_received,
                                                                                                              T_ const *const ptr_array_summation_recurrent_inputs_gates_received,
                                                                                                              T_ const *const ptr_array_summation_input_forgets_gates_received,
                                                                                                              T_ const *const ptr_array_summation_recurrent_forgets_gates_received,
                                                                                                              struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_number_peepholes_connections(ptr_layer_it_received->ptr_array_block_units->last_index_peephole_input_gate - ptr_layer_it_received->ptr_array_block_units->first_index_peephole_input_gate);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_reverse_direction_timed_index;
    
    T_ const *const tmp_ptr_array_cell_input_bias(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index),
                  *const tmp_ptr_array_input_gate_bias(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index + layer_cell_unit_size_received),
                  *const tmp_ptr_array_forget_gate_bias(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index + layer_cell_unit_size_received + layer_block_unit_size_received),
                  *tmp_ptr_array_peephole_input_gate_parameters,
                  *tmp_ptr_array_peephole_forget_gate_parameters;
    T_ tmp_cell_input_summation,
        tmp_input_gate_summation,
        tmp_forget_gate_summation;

    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    if(time_step_index_received != time_step_prediction_start_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_reverse_direction_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);

            tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units;

            for(tmp_cell_index = 0_zu,
                tmp_block_index = 0_zu; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Gates.        
                tmp_input_gate_summation = ptr_array_summation_input_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                            ptr_array_summation_recurrent_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                            tmp_ptr_array_input_gate_bias[tmp_block_index];
                
                tmp_forget_gate_summation = ptr_array_summation_input_forgets_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                             ptr_array_summation_recurrent_forgets_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                             tmp_ptr_array_forget_gate_bias[tmp_block_index];

            #ifndef NO_PEEPHOLE
                tmp_ptr_array_peephole_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;
                tmp_ptr_array_peephole_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;

                for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                {
                    tmp_input_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_reverse_direction_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_input_gate_parameters[tmp_connection_index];
                    tmp_forget_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_reverse_direction_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_forget_gate_parameters[tmp_connection_index];
                }
            #endif
                
                tmp_ptr_block_unit_it->ptr_summation_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_summation;

                tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_input_gate_summation);
                tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_forget_gate_summation);
                // [0] |END| Gates. |END|
                
                // [0] Cell input/state.        
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                       ++tmp_cell_index)
                {
                    tmp_cell_input_summation = ptr_array_summation_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Input.
                                                               ptr_array_summation_recurrent_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Recurrent.
                                                               tmp_ptr_array_cell_input_bias[tmp_cell_index];

                    tmp_ptr_cell_unit_it->ptr_summation_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    
                    AF_FIRE(tmp_ptr_block_unit_it->activation_function_io,
                                  tmp_cell_input_summation,
                                  tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index]);
                    
                    tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] + tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index] * tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_reverse_direction_timed_index];
                }
                // [0] |END| Cell input/state. |END|
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units;

            for(tmp_cell_index = 0_zu,
                tmp_block_index = 0_zu; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Input gate.        
                tmp_input_gate_summation = ptr_array_summation_input_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                            tmp_ptr_array_input_gate_bias[tmp_block_index];
                
                tmp_ptr_block_unit_it->ptr_summation_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;

                tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_input_gate_summation);
                // [0] |END| Input gate. |END|
                
                // [0] Cell input/state.        
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                       ++tmp_cell_index)
                {
                    tmp_cell_input_summation = ptr_array_summation_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Input.
                                                               tmp_ptr_array_cell_input_bias[tmp_cell_index];

                    tmp_ptr_cell_unit_it->ptr_summation_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    
                    AF_FIRE(tmp_ptr_block_unit_it->activation_function_io,
                                  tmp_cell_input_summation,
                                  tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index]);

                    tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index];
                }
                // [0] |END| Cell input/state. |END|
            }
        }
    }
}

void Neural_Network::Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__Loop(long long int const time_step_index_received,
                                                                                                                             long long int const time_step_reverse_direction_received,
                                                                                                                             long long int const time_step_prediction_start_received,
                                                                                                                             size_t const batch_size_received,
                                                                                                                             size_t const layer_block_unit_size_received,
                                                                                                                             size_t const layer_cell_unit_size_received,
                                                                                                                             T_ const *const ptr_array_summation_input_block_inputs_received,
                                                                                                                             T_ const *const ptr_array_summation_recurrent_block_inputs_received,
                                                                                                                             T_ const *const ptr_array_summation_input_inputs_gates_received,
                                                                                                                             T_ const *const ptr_array_summation_recurrent_inputs_gates_received,
                                                                                                                             T_ const *const ptr_array_summation_input_forgets_gates_received,
                                                                                                                             T_ const *const ptr_array_summation_recurrent_forgets_gates_received,
                                                                                                                             struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_number_peepholes_connections(ptr_layer_it_received->ptr_array_block_units->last_index_peephole_input_gate - ptr_layer_it_received->ptr_array_block_units->first_index_peephole_input_gate),
                       tmp_zoneout_mask_index(static_cast<size_t>(time_step_index_received) * layer_cell_unit_size_received);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_reverse_direction_timed_index;
    
    T_ const *const tmp_ptr_array_cell_input_bias(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index),
                  *const tmp_ptr_array_input_gate_bias(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index + layer_cell_unit_size_received),
                  *const tmp_ptr_array_forget_gate_bias(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index + layer_cell_unit_size_received + layer_block_unit_size_received),
                  *tmp_ptr_array_peephole_input_gate_parameters,
                  *tmp_ptr_array_peephole_forget_gate_parameters;
    T_ tmp_cell_input_summation,
        tmp_input_gate_summation,
        tmp_forget_gate_summation;

    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    if(time_step_index_received != time_step_prediction_start_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_reverse_direction_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);

            tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units;

            for(tmp_cell_index = 0_zu,
                tmp_block_index = 0_zu; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Gates.        
                tmp_input_gate_summation = ptr_array_summation_input_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                            ptr_array_summation_recurrent_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                            tmp_ptr_array_input_gate_bias[tmp_block_index];
                
                tmp_forget_gate_summation = ptr_array_summation_input_forgets_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                             ptr_array_summation_recurrent_forgets_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                             tmp_ptr_array_forget_gate_bias[tmp_block_index];

            #ifndef NO_PEEPHOLE
                tmp_ptr_array_peephole_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;
                tmp_ptr_array_peephole_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;

                for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                {
                    tmp_input_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_reverse_direction_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_input_gate_parameters[tmp_connection_index];
                    tmp_forget_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_reverse_direction_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_forget_gate_parameters[tmp_connection_index];
                }
            #endif
                
                tmp_ptr_block_unit_it->ptr_summation_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_summation;

                tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_input_gate_summation);
                tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_forget_gate_summation);
                // [0] |END| Gates. |END|
                
                // [0] Cell input/state.        
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                       ++tmp_cell_index)
                {
                    tmp_cell_input_summation = ptr_array_summation_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Input.
                                                               ptr_array_summation_recurrent_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Recurrent.
                                                               tmp_ptr_array_cell_input_bias[tmp_cell_index];

                    tmp_ptr_cell_unit_it->ptr_summation_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    
                    AF_FIRE(tmp_ptr_block_unit_it->activation_function_io,
                                  tmp_cell_input_summation,
                                  tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index]);
                    
                    if(tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state[tmp_zoneout_mask_index])
                    { tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] + tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index] * tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_reverse_direction_timed_index]; }
                    else
                    { tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_reverse_direction_timed_index]; }
                }
                // [0] |END| Cell input/state. |END|
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units;

            for(tmp_cell_index = 0_zu,
                tmp_block_index = 0_zu; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Input gate.        
                tmp_input_gate_summation = ptr_array_summation_input_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                            tmp_ptr_array_input_gate_bias[tmp_block_index];
                
                tmp_ptr_block_unit_it->ptr_summation_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;

                tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_input_gate_summation);
                // [0] |END| Input gate. |END|
                
                // [0] Cell input/state.        
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                       ++tmp_cell_index)
                {
                    tmp_cell_input_summation = ptr_array_summation_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Input.
                                                               tmp_ptr_array_cell_input_bias[tmp_cell_index];

                    tmp_ptr_cell_unit_it->ptr_summation_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    
                    AF_FIRE(tmp_ptr_block_unit_it->activation_function_io,
                                  tmp_cell_input_summation,
                                  tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index]);
                    
                    if(tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state[tmp_zoneout_mask_index])
                    { tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index]; }
                    else
                    { tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = 0_T; }
                }
                // [0] |END| Cell input/state. |END|
            }
        }
    }
}

void Neural_Network::Forward_Pass__LSTM__Output__Loop(long long int const time_step_index_received,
                                                                                         size_t const batch_size_received,
                                                                                         size_t const layer_block_unit_size_received,
                                                                                         size_t const layer_cell_unit_size_received,
                                                                                         T_ const *const ptr_array_summation_input_outputs_gates_received,
                                                                                         T_ const *const ptr_array_summation_recurrent_outputs_gates_received,
                                                                                         struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_number_peepholes_connections(ptr_layer_it_received->ptr_array_block_units->last_index_peephole_output_gate - ptr_layer_it_received->ptr_array_block_units->first_index_peephole_output_gate);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_cell_data_timed_index;
    
    T_ const *const tmp_ptr_array_output_gate_bias(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index + layer_cell_unit_size_received + 2_zu * layer_block_unit_size_received),
                  *tmp_ptr_array_peephole_output_gate_parameters;
    T_ tmp_output_gate_summation;

    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it;
    
    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
        
        tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
        
        tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units;
        
        for(tmp_block_index = 0_zu; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                            ++tmp_block_index)
        {
            // [0] Output gate.
            tmp_output_gate_summation = ptr_array_summation_input_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                          ptr_array_summation_recurrent_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                          tmp_ptr_array_output_gate_bias[tmp_block_index];
            
        #ifndef NO_PEEPHOLE
            tmp_ptr_array_peephole_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
            {
                tmp_output_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_output_gate_parameters[tmp_connection_index];
            }
        #endif

            tmp_ptr_block_unit_it->ptr_summation_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;

            tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_output_gate_summation);
            // [0] |END| Output gate. |END|

            // [0] Cell output.
            for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
            {
                tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index];
            }
            // [0] |END| Cell output. |END|
        }
    }
}

void Neural_Network::Forward_Pass__LSTM__Output__Zoneout__Loop(long long int const time_step_index_received,
                                                                                                        long long int const time_step_reverse_direction_received,
                                                                                                        long long int const time_step_prediction_start_received,
                                                                                                        size_t const batch_size_received,
                                                                                                        size_t const layer_block_unit_size_received,
                                                                                                        size_t const layer_cell_unit_size_received,
                                                                                                        T_ const *const ptr_array_summation_input_outputs_gates_received,
                                                                                                        T_ const *const ptr_array_summation_recurrent_outputs_gates_received,
                                                                                                        struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_number_peepholes_connections(ptr_layer_it_received->ptr_array_block_units->last_index_peephole_output_gate - ptr_layer_it_received->ptr_array_block_units->first_index_peephole_output_gate),
                       tmp_zoneout_mask_index(static_cast<size_t>(time_step_index_received) * layer_cell_unit_size_received);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_reverse_direction_timed_index;
    
    T_ const *const tmp_ptr_array_output_gate_bias(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index + layer_cell_unit_size_received + 2_zu * layer_block_unit_size_received),
                  *tmp_ptr_array_peephole_output_gate_parameters;
    T_ tmp_output_gate_summation;

    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it;
    
    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    if(time_step_index_received != time_step_prediction_start_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_reverse_direction_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);
            
            tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units;
            
            for(tmp_block_index = 0_zu; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Output gate.
                tmp_output_gate_summation = ptr_array_summation_input_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                              ptr_array_summation_recurrent_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                              tmp_ptr_array_output_gate_bias[tmp_block_index];
                
            #ifndef NO_PEEPHOLE
                tmp_ptr_array_peephole_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

                for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                {
                    tmp_output_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_output_gate_parameters[tmp_connection_index];
                }
            #endif

                tmp_ptr_block_unit_it->ptr_summation_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;

                tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_output_gate_summation);
                // [0] |END| Output gate. |END|

                // [0] Cell output.
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    if(tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output[tmp_zoneout_mask_index])
                    { tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index]; }
                    else
                    { tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_reverse_direction_timed_index]; }
                }
                // [0] |END| Cell output. |END|
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_ptr_block_unit_it = ptr_layer_it_received->ptr_array_block_units;
            
            for(tmp_block_index = 0_zu; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Output gate.
                tmp_output_gate_summation = ptr_array_summation_input_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                              ptr_array_summation_recurrent_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                              tmp_ptr_array_output_gate_bias[tmp_block_index];
                
            #ifndef NO_PEEPHOLE
                tmp_ptr_array_peephole_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

                for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                {
                    tmp_output_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_output_gate_parameters[tmp_connection_index];
                }
            #endif

                tmp_ptr_block_unit_it->ptr_summation_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;

                tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_output_gate_summation);
                // [0] |END| Output gate. |END|

                // [0] Cell output.
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    if(tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output[tmp_zoneout_mask_index])
                    { tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index]; }
                    else
                    { tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_timed_index] = 0_T; }
                }
                // [0] |END| Cell output. |END|
            }
        }
    }
}

void Neural_Network::Forward_Pass__LSTM__States_AF__Loop(long long int const time_step_index_received,
                                                                                              size_t const batch_size_received,
                                                                                              size_t const layer_block_unit_size_received,
                                                                                              size_t const layer_cell_unit_size_received,
                                                                                              T_ const *const ptr_array_summation_cell_states_received,
                                                                                              struct Layer *const ptr_layer_it_received)
{
    size_t tmp_example_index,
              tmp_cell_index,
              tmp_cell_data_timed_index;
    
    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const tmp_type_activation_function_io(ptr_layer_it_received->ptr_array_block_units->activation_function_io);

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);

        // [0] Cells states.        
        for(tmp_cell_index = 0_zu,
            tmp_ptr_last_cell_unit = ptr_layer_it_received->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = ptr_layer_it_received->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                              ++tmp_cell_index)
        {
            AF_FIRE(tmp_type_activation_function_io,
                          ptr_array_summation_cell_states_received[tmp_cell_data_timed_index + tmp_cell_index],
                          tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index]);
        }
        // [0] |END| Cells states. |END|
    }
}
