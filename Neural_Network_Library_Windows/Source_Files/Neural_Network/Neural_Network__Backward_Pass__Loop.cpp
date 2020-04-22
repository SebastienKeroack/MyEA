#include "stdafx.hpp"

#include <Math/Math.hpp>

#include <Neural_Network/Neural_Network.hpp>

void Neural_Network::Backward_Pass(size_t const batch_size_received)
{
    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            if(this->pre_training_level != 0_zu)
            {
                this->Backward_Pass__Pre_Training(batch_size_received);

                break;
            }
        default:
            if(this->number_recurrent_depth > 1_zu)
            {
                if(this->use_OpenMP && this->is_OpenMP_initialized)
                { this->RNN__Backward_Pass_Batch__OpenMP(batch_size_received); }
                else
                { this->RNN__Backward_Pass_Batch__Loop(batch_size_received); }
            }
            else
            {
                if(this->use_OpenMP && this->is_OpenMP_initialized)
                { this->FF__Backward_Pass_Batch__OpenMP(batch_size_received); }
                else
                { this->FF__Backward_Pass_Batch__Loop(batch_size_received); }
            }
                break;
    }
}

void Neural_Network::Backward_Pass__Pre_Training(size_t const batch_size_received)
{
    if(this->pre_training_level == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The neural network use the pre-training function without the mode pre-training activate. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return;
    }

    if(this->number_recurrent_depth > 1_zu)
    {
        if(this->use_OpenMP && this->is_OpenMP_initialized)
        { this->RNN__Backward_Pass_Batch__Pre_Training__OpenMP(batch_size_received); }
        else
        { this->RNN__Backward_Pass_Batch__Pre_Training__Loop(batch_size_received); }
    }
    else
    {
        if(this->use_OpenMP && this->is_OpenMP_initialized)
        { this->FF__Backward_Pass_Batch__Pre_Training__OpenMP(batch_size_received); }
        else
        { this->FF__Backward_Pass_Batch__Pre_Training__Loop(batch_size_received); }
    }
}

void Neural_Network::FF__Backward_Pass_Batch__Loop(size_t const batch_size_received)
{
    size_t tmp_layer_number_outputs;
    
    T_ *tmp_ptr_array_layer_gradients;
    
#if defined(COMPILE_AUTODIFF)
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
                                 MyEA::Time::Date_Time_Now().c_str(),
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
                    this->number_threads * this->total_normalized_units_allocated * sizeof(T_));
        // |END| Set all derivative mean to zero. |END|

        // Set all derivative variance to zero.
        MEMSET(this->ptr_array_normalized_batch_units_derivatives_variances,
                    0,
                    this->number_threads * this->total_normalized_units_allocated * sizeof(T_));
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
                     this->batch_size * tmp_layer_number_outputs * sizeof(T_));
        // |END| Clear past error(s). |END|
        
        // Propagate the error(s) to the layer.
        for(tmp_ptr_next_layer_it = tmp_ptr_layer_it->next_connected_layers[0u],
            tmp_ptr_next_layer_end = tmp_ptr_next_layer_it + tmp_ptr_layer_it->next_connected_layers.size(); tmp_ptr_next_layer_it != tmp_ptr_next_layer_end; ++tmp_ptr_next_layer_it)
        {
            switch(tmp_ptr_next_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                    this->Backward_Pass__Average_Pooling__Loop(0_zu,
                                                                                           batch_size_received,
                                                                                           tmp_layer_number_outputs,
                                                                                           tmp_ptr_array_layer_gradients,
                                                                                           tmp_ptr_next_layer_it);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    this->Backward_Pass__FC__Loop(batch_size_received,
                                                                       tmp_layer_number_outputs,
                                                                       tmp_ptr_array_layer_gradients,
                                                                       tmp_ptr_next_layer_it);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                    this->Backward_Pass__Max_Pooling__Loop(0_zu,
                                                                                      batch_size_received,
                                                                                      tmp_layer_number_outputs,
                                                                                      tmp_ptr_array_layer_gradients,
                                                                                      tmp_ptr_next_layer_it);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    this->Backward_Pass__Residual__Loop(0_zu,
                                                                               batch_size_received,
                                                                               tmp_layer_number_outputs,
                                                                               tmp_ptr_array_layer_gradients,
                                                                               tmp_ptr_next_layer_it);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
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
                this->Backward_Pass__Gradient__FC__Loop(0_zu,
                                                                                   batch_size_received,
                                                                                   tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                this->Backward_Pass__Gradient__Residual__Loop(batch_size_received, tmp_ptr_layer_it);

                tmp_ptr_gradient_layer_it = tmp_ptr_layer_it + 1;
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    return;
        }
        // |END| Compute the gradients. |END|
    }
}

void Neural_Network::FF__Backward_Pass_Batch__Pre_Training__Loop(size_t const batch_size_received)
{
    size_t tmp_layer_number_outputs;
    
    T_ *tmp_ptr_array_layer_gradients;

    struct Layer *const tmp_ptr_coded_layer(this->ptr_array_layers + this->pre_training_level);
    struct Layer const *const tmp_ptr_decoded_layer(this->ptr_last_layer - static_cast<size_t>(tmp_ptr_coded_layer - this->ptr_array_layers));
    
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->type_state_propagation != MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not back propagate gradient in inference mode. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
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
                    this->number_threads * this->total_normalized_units_allocated * sizeof(T_));
        // |END| Set all derivative mean to zero. |END|

        // Set all derivative variance to zero.
        MEMSET(this->ptr_array_normalized_batch_units_derivatives_variances,
                    0,
                    this->number_threads * this->total_normalized_units_allocated * sizeof(T_));
        // |END| Set all derivative variance to zero. |END|
    }

    // Clear past error(s).
    tmp_layer_number_outputs = *tmp_ptr_coded_layer->ptr_number_outputs;

    tmp_ptr_array_layer_gradients = tmp_ptr_coded_layer->ptr_array_derivative_outputs;

    MEMSET(tmp_ptr_array_layer_gradients,
                   0,
                   this->batch_size * tmp_layer_number_outputs * sizeof(T_));
    // |END| Clear past error(s). |END|
    
    // Propagate the error(s) to the layer.
    switch(tmp_ptr_decoded_layer->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            this->Backward_Pass__FC__Loop(batch_size_received,
                                                               tmp_layer_number_outputs,
                                                               tmp_ptr_array_layer_gradients,
                                                               tmp_ptr_decoded_layer);
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
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
            this->Backward_Pass__Gradient__FC__Loop(0_zu,
                                                                               batch_size_received,
                                                                               tmp_ptr_coded_layer);
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
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

void Neural_Network::Backward_Pass__FC__Loop(size_t const batch_size_received,
                                                                          size_t const derivative_size_received,
                                                                          T_ *const ptr_array_derivatives_received,
                                                                          struct Layer const *const ptr_layer_it_received)
{
    if(ptr_layer_it_received->type_group == MyEA::Common::ENUM_TYPE_GROUP::TYPE_GROUP_RESIDUAL)
    {
        this->Backward_Pass__Residual__FC__Loop(0_zu,
                                                                           batch_size_received,
                                                                           derivative_size_received,
                                                                           ptr_array_derivatives_received,
                                                                           ptr_layer_it_received);
    }
    else
    {
        this->Backward_Pass__FC__Loop(0_zu,
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

void Neural_Network::Backward_Pass__Average_Pooling__Loop(size_t const time_step_index_received,
                                                                                              size_t const batch_size_received,
                                                                                              size_t const derivative_size_received,
                                                                                              T_ *const ptr_array_derivatives_received,
                                                                                              struct Layer const *const ptr_layer_it_received)
{
    this->Backward_Pass__Average_Pooling__Loop(time_step_index_received,
                                                                           batch_size_received,
                                                                           *ptr_layer_it_received->ptr_number_outputs,
                                                                           derivative_size_received,
                                                                           ptr_layer_it_received->pooling_values[0u],
                                                                           ptr_layer_it_received->pooling_values[1u],
                                                                           ptr_layer_it_received->pooling_values[2u],
                                                                           ptr_layer_it_received->pooling_values[3u],
                                                                           ptr_layer_it_received->ptr_array_basic_units->ptr_array_errors,
                                                                           ptr_array_derivatives_received);
}

void Neural_Network::Backward_Pass__FC__Loop(size_t const time_step_index_received,
                                                                          size_t const batch_size_received,
                                                                          size_t const derivative_size_received,
                                                                          T_ *const ptr_array_derivatives_received,
                                                                          struct Layer const *const ptr_layer_it_received)
{
    struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
    
    this->Backward_Pass__FC__Loop(time_step_index_received,
                                                       batch_size_received,
                                                       static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit),
                                                       derivative_size_received,
                                                       tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                       this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                       ptr_array_derivatives_received);
}

void Neural_Network::Backward_Pass__Max_Pooling__Loop(size_t const time_step_index_received,
                                                                                        size_t const batch_size_received,
                                                                                        size_t const derivative_size_received,
                                                                                        T_ *const ptr_array_derivatives_received,
                                                                                        struct Layer const *const ptr_layer_it_received)
{
    struct Basic_indice_unit *const tmp_ptr_layer_first_basic_indice_unit(ptr_layer_it_received->ptr_array_basic_indice_units);
    
    this->Backward_Pass__Max_Pooling__Loop(time_step_index_received,
                                                                      batch_size_received,
                                                                      *ptr_layer_it_received->ptr_number_outputs,
                                                                      derivative_size_received,
                                                                      ptr_layer_it_received->pooling_values[2u],
                                                                      tmp_ptr_layer_first_basic_indice_unit->ptr_array_indices,
                                                                      tmp_ptr_layer_first_basic_indice_unit->ptr_array_errors,
                                                                      ptr_array_derivatives_received);
}

void Neural_Network::Backward_Pass__Residual__Loop(size_t const time_step_index_received,
                                                                                  size_t const batch_size_received,
                                                                                  size_t const derivative_size_received,
                                                                                  T_ *const ptr_array_derivatives_received,
                                                                                  struct Layer const *const ptr_layer_it_received)
{
    this->Backward_Pass__Residual__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               *ptr_layer_it_received->ptr_number_outputs,
                                                               derivative_size_received,
                                                               ptr_layer_it_received->pooling_values[2u],
                                                               ptr_layer_it_received->ptr_array_basic_units->ptr_array_errors,
                                                               ptr_array_derivatives_received);
}

void Neural_Network::Backward_Pass__Residual__Block__Loop(size_t const time_step_index_received,
                                                                                              size_t const batch_size_received,
                                                                                              size_t const derivative_size_received,
                                                                                              T_ *const ptr_array_derivatives_received,
                                                                                              struct Layer const *const ptr_layer_it_received)
{
    union Normalized_unit *const tmp_ptr_residual_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    T_ *tmp_ptr_array_derivatives;
    
    if(ptr_layer_it_received->Use__Normalization())
    {
        tmp_ptr_array_derivatives = tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_errors;

        // Clear past error(s).
        MEMSET(tmp_ptr_array_derivatives + this->batch_size * derivative_size_received * time_step_index_received,
                       0,
                       this->batch_size * derivative_size_received * sizeof(T_));
        // |END| Clear past error(s). |END|
    }
    else { tmp_ptr_array_derivatives = ptr_array_derivatives_received; }
    
    this->Backward_Pass__Residual__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               *ptr_layer_it_received->ptr_number_outputs,
                                                               derivative_size_received,
                                                               ptr_layer_it_received->pooling_values[2u],
                                                               ptr_layer_it_received->ptr_array_basic_units->ptr_array_errors,
                                                               tmp_ptr_array_derivatives);

    // Dropout, ShakeDrop.
    if(ptr_layer_it_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP)
    {
        this->Backward_Pass__Dropout__ShakeDrop__Loop(time_step_index_received,
                                                                                      batch_size_received,
                                                                                      derivative_size_received,
                                                                                      ptr_layer_it_received->ptr_array__mask__dropout__shakedrop,
                                                                                      0_T,
                                                                                      1_T,
                                                                                      tmp_ptr_array_derivatives);
    }
    
    // Normalization.
    if(ptr_layer_it_received->Use__Normalization())
    {
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                this->Backward_Pass__Batch_Normalization__Loop(time_step_index_received,
                                                                                            batch_size_received,
                                                                                            derivative_size_received,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                            ptr_layer_it_received->ptr_array_pre_normalization,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                            tmp_ptr_array_derivatives,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                this->Backward_Pass__Batch_Renormalization__Loop(time_step_index_received,
                                                                                                batch_size_received,
                                                                                                derivative_size_received,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                ptr_layer_it_received->ptr_array_pre_normalization,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_array_derivatives,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         ptr_layer_it_received->type_normalization,
                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                    return;
        }
        
        // Store the new derivative inputs (normalized derivative).
        tmp_ptr_array_derivatives = tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_errors;
        
        //     --------------> FC --> ... --> {FC} --> [ResNet-BN]
        //    /                                                        /
        // FC --> ResNet -----------------------------------------------> ...
        MEMCPY(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                       tmp_ptr_array_derivatives + this->batch_size * derivative_size_received * time_step_index_received,
                       this->batch_size * derivative_size_received * sizeof(T_));
    }
}

void Neural_Network::Backward_Pass__Residual__FC__Loop(size_t const time_step_index_received,
                                                                                          size_t const batch_size_received,
                                                                                          size_t const derivative_size_received,
                                                                                          T_ *const ptr_array_derivatives_received,
                                                                                          struct Layer const *const ptr_layer_it_received)
{
    bool const tmp_is_input_layer(static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_unit - ptr_layer_it_received->ptr_array_AF_units) + static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit - ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units) == 0_zu);
    
    if(ptr_layer_it_received->Use__Normalization())
    {
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                //     --------------> {FC} --> [FC] ...
                //    /
                // FC --> ResNet ---> ...
                if(tmp_is_input_layer == false)
                {
                    MEMCPY(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                                   ptr_layer_it_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_errors + this->batch_size * derivative_size_received * time_step_index_received,
                                   this->batch_size * derivative_size_received * sizeof(T_));
                }
                //     --------------> [FC] --> FC ...
                //    /
                // {FC} --> ResNet ---> ...
                else
                {
                    this->Backward_Pass__Identity__Loop(time_step_index_received,
                                                                             batch_size_received,
                                                                             derivative_size_received,
                                                                             ptr_layer_it_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_errors,
                                                                             ptr_array_derivatives_received);
                }
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         ptr_layer_it_received->type_normalization,
                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                    return;
        }
    }
    //     --------------> {FC} --> [FC] ...
    //    /
    // FC --> ResNet ---> ...
    else if(tmp_is_input_layer == false)
    {
        if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            MEMCPY(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                           ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units->ptr_array_dAFs + this->batch_size * derivative_size_received * time_step_index_received,
                           this->batch_size * derivative_size_received * sizeof(T_));
        }
        else
        {
            MEMCPY(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                           ptr_layer_it_received->ptr_array_AF_units->ptr_array_errors + this->batch_size * derivative_size_received * time_step_index_received,
                           this->batch_size * derivative_size_received * sizeof(T_));
        }
    }
    //     --------------> [FC] --> FC ...
    //    /
    // {FC} --> ResNet ---> ...
    else
    {
        struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
        
        this->Backward_Pass__FC__Loop(time_step_index_received,
                                                           batch_size_received,
                                                           static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit),
                                                           derivative_size_received,
                                                           tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                           this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                           ptr_array_derivatives_received);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Backward_Pass__Gradient__FC__Loop(size_t const time_step_index_received,
                                                                                          size_t const batch_size_received,
                                                                                          struct Layer const *const ptr_layer_it_received)
{
    struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
    
    struct AF_unit *const tmp_ptr_layer_first_AF_unit(ptr_layer_it_received->ptr_array_AF_units);
    struct AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));
    
    T_ const *tmp_ptr_array_derivative_inputs(ptr_layer_it_received->ptr_array_derivative_outputs);

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // k-Sparse.
    // ...

    // Normalization after activation.
    if(ptr_layer_it_received->Use__Normalization()
      &&
      ptr_layer_it_received->use_layer_normalization_before_activation == false)
    {
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                this->Backward_Pass__Batch_Normalization__Loop(time_step_index_received,
                                                                                            batch_size_received,
                                                                                            tmp_output_size,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                            ptr_layer_it_received->ptr_array_pre_normalization,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                            tmp_ptr_array_derivative_inputs,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                this->Backward_Pass__Batch_Renormalization__Loop(time_step_index_received,
                                                                                                batch_size_received,
                                                                                                tmp_output_size,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                ptr_layer_it_received->ptr_array_pre_normalization,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_array_derivative_inputs,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         ptr_layer_it_received->type_normalization,
                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                    return;
        }
        
        // Store the new derivative inputs (normalized derivative).
        tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors;
    }
    
    if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
    {
        // Recurrent activation function.
        this->Backward_Pass__FC__DF_Ind_RNN__Loop(time_step_index_received,
                                                                                 batch_size_received,
                                                                                 tmp_output_size,
                                                                                 this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                                 tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function,
                                                                                 tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_activation_steepness,
                                                                                 ptr_layer_it_received->ptr_array_pre_activation_functions,
                                                                                 tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                                 tmp_ptr_array_derivative_inputs,
                                                                                 tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_dAFs,
                                                                                 tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_errors);

        // Store the new derivative inputs (recurrent activation function derivative).
        tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_dAFs;
    }
    else
    {
        // Activation function.
        this->Backward_Pass__FC__DF__Loop(time_step_index_received,
                                                                   batch_size_received,
                                                                   tmp_output_size,
                                                                   tmp_ptr_layer_first_AF_unit->ptr_type_activation_function,
                                                                   tmp_ptr_layer_first_AF_unit->ptr_activation_steepness,
                                                                   ptr_layer_it_received->ptr_array_pre_activation_functions,
                                                                   tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                   tmp_ptr_array_derivative_inputs,
                                                                   tmp_ptr_layer_first_AF_unit->ptr_array_errors);

        // Store the new derivative inputs (activation function derivative).
        tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_errors;
    }

    // Normalization before activation.
    if(ptr_layer_it_received->Use__Normalization()
      &&
      ptr_layer_it_received->use_layer_normalization_before_activation)
    {
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                this->Backward_Pass__Batch_Normalization__Loop(time_step_index_received,
                                                                                            batch_size_received,
                                                                                            tmp_output_size,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                            ptr_layer_it_received->ptr_array_pre_normalization,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                            tmp_ptr_array_derivative_inputs,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                this->Backward_Pass__Batch_Renormalization__Loop(time_step_index_received,
                                                                                                batch_size_received,
                                                                                                tmp_output_size,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                ptr_layer_it_received->ptr_array_pre_normalization,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_array_derivative_inputs,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         ptr_layer_it_received->type_normalization,
                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                    return;
        }
        
        // Store the new derivative inputs (normalized derivative).
        tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors;
    }
    
    // Copy derivative to derivative neurons.
    MEMCPY(tmp_ptr_layer_first_neuron_unit->ptr_array_errors + this->batch_size * tmp_output_size * time_step_index_received,
                   tmp_ptr_array_derivative_inputs + this->batch_size * tmp_output_size * time_step_index_received,
                   this->batch_size * tmp_output_size * sizeof(T_));
    // |END| Copy derivative to derivative neurons. |END|
}

void Neural_Network::Backward_Pass__Gradient__Residual__Loop(size_t const batch_size_received, struct Layer const *const ptr_layer_it_received)
{
    struct Layer const *const tmp_ptr_layer_end(ptr_layer_it_received + 1);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + static_cast<size_t>(ptr_layer_it_received - this->ptr_array_layers) + ptr_layer_it_received->block_depth);
    
    // Remaining layer(s).
    for(; tmp_ptr_layer_it != tmp_ptr_layer_end; --tmp_ptr_layer_it)
    {
        this->Backward_Pass__Gradient__Residual__Layer__Loop(false,
                                                                                               batch_size_received,
                                                                                               tmp_ptr_layer_it);
    }
    // |END| Remaining layer(s). |END|
    
    // First block layer.
    this->Backward_Pass__Gradient__Residual__Layer__Loop(true,
                                                                                           batch_size_received,
                                                                                           tmp_ptr_layer_it);
    // |END| First block layer. |END|
}

void Neural_Network::Backward_Pass__Gradient__Residual__Layer__Loop(bool const is_block_input_layer_received,
                                                                                                              size_t const batch_size_received,
                                                                                                              struct Layer *&ptr_layer_it_received)
{
    size_t const tmp_layer_number_outputs(*ptr_layer_it_received->ptr_number_outputs);
    
    T_ *const tmp_ptr_array_layer_gradients(ptr_layer_it_received->ptr_array_derivative_outputs);
    
    struct Layer const *const tmp_ptr_next_layer_it(ptr_layer_it_received->next_connected_layers[0u]);
    
    // Clear past error(s).
    MEMSET(tmp_ptr_array_layer_gradients,
                  0,
                  this->batch_size * tmp_layer_number_outputs * sizeof(T_));
    // |END| Clear past error(s). |END|

    // Propagate the error(s) to the layer.
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_layer_it_received->next_connected_layers.size() > 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: In a residual block the layers can not have more than one forward connection. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
    }
#endif

    switch(tmp_ptr_next_layer_it->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            this->Backward_Pass__Average_Pooling__Loop(0_zu,
                                                                                   batch_size_received,
                                                                                   tmp_layer_number_outputs,
                                                                                   tmp_ptr_array_layer_gradients,
                                                                                   tmp_ptr_next_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            this->Backward_Pass__Residual__FC__Loop(0_zu,
                                                                               batch_size_received,
                                                                               tmp_layer_number_outputs,
                                                                               tmp_ptr_array_layer_gradients,
                                                                               tmp_ptr_next_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            this->Backward_Pass__Max_Pooling__Loop(0_zu,
                                                                              batch_size_received,
                                                                              tmp_layer_number_outputs,
                                                                              tmp_ptr_array_layer_gradients,
                                                                              tmp_ptr_next_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
            this->Backward_Pass__Residual__Block__Loop(0_zu,
                                                                                   batch_size_received,
                                                                                   tmp_layer_number_outputs,
                                                                                   tmp_ptr_array_layer_gradients,
                                                                                   tmp_ptr_next_layer_it);
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
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
            this->Backward_Pass__Gradient__Residual__FC__Loop(is_block_input_layer_received,
                                                                                               0_zu,
                                                                                               batch_size_received,
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
    // |END| Compute the gradients. |END|
}

void Neural_Network::Backward_Pass__Gradient__Residual__FC__Loop(bool const is_block_input_layer_received,
                                                                                                          size_t const time_step_index_received,
                                                                                                          size_t const batch_size_received,
                                                                                                          struct Layer const *const ptr_layer_it_received)
{
    struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
    
    struct AF_unit *const tmp_ptr_layer_first_AF_unit(ptr_layer_it_received->ptr_array_AF_units);
    struct AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_input_size(*tmp_ptr_layer_first_neuron_unit->ptr_number_connections),
                       tmp_output_size(static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));
    
    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    T_ *tmp_ptr_array_derivative_inputs;
    
    if(is_block_input_layer_received == false)
    {
        if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT) { tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_errors; }
        else { tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_errors; }
    }
    else if(ptr_layer_it_received->Use__Normalization()) { tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors; }
    else { return; }

    // Clear past error(s).
    MEMSET(tmp_ptr_array_derivative_inputs + this->batch_size * tmp_input_size * time_step_index_received,
                   0,
                   this->batch_size * tmp_input_size * sizeof(T_));
    // |END| Clear past error(s). |END|
    
    this->Backward_Pass__FC__Loop(time_step_index_received,
                                                       batch_size_received,
                                                       tmp_output_size,
                                                       tmp_input_size,
                                                       tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                       this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                       tmp_ptr_array_derivative_inputs);

    if(is_block_input_layer_received == false)
    {
        // k-Sparse.
        // ...
        
        if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Backward_Pass__FC__DF_Ind_RNN__Loop(time_step_index_received,
                                                                                     batch_size_received,
                                                                                     tmp_output_size,
                                                                                     this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                                     tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function,
                                                                                     tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_activation_steepness,
                                                                                     ptr_layer_it_received->ptr_array_pre_activation_functions,
                                                                                     tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                                     tmp_ptr_array_derivative_inputs,
                                                                                     tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_dAFs,
                                                                                     tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_errors);

            // Store the new derivative inputs (recurrent activation function derivative).
            tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_dAFs;
        }
        else
        {
            // Activation function.
            this->Backward_Pass__FC__DF__Loop(time_step_index_received,
                                                                       batch_size_received,
                                                                       tmp_input_size,
                                                                       tmp_ptr_layer_first_AF_unit->ptr_type_activation_function,
                                                                       tmp_ptr_layer_first_AF_unit->ptr_activation_steepness,
                                                                       ptr_layer_it_received->ptr_array_pre_activation_functions,
                                                                       tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                       tmp_ptr_array_derivative_inputs,
                                                                       tmp_ptr_layer_first_AF_unit->ptr_array_errors);

            // Store the new derivative inputs (activation function derivative).
            tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_errors;
        }
    }

    // Normalization.
    if(ptr_layer_it_received->Use__Normalization())
    {
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                this->Backward_Pass__Batch_Normalization__Loop(time_step_index_received,
                                                                                            batch_size_received,
                                                                                            tmp_input_size,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                            ptr_layer_it_received->ptr_array_pre_normalization,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                            tmp_ptr_array_derivative_inputs,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                this->Backward_Pass__Batch_Renormalization__Loop(time_step_index_received,
                                                                                                batch_size_received,
                                                                                                tmp_input_size,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                ptr_layer_it_received->ptr_array_pre_normalization,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_array_derivative_inputs,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         ptr_layer_it_received->type_normalization,
                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                    return;
        }
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Backward_Pass__Average_Pooling__Loop(size_t const time_step_index_received,
                                                                                              size_t const batch_size_received,
                                                                                              size_t const input_size_received,
                                                                                              size_t const derivative_size_received,
                                                                                              size_t const kernel_size_received,
                                                                                              size_t const stride_received,
                                                                                              size_t const padding_received,
                                                                                              size_t const dilation_received,
                                                                                              T_ const *const ptr_array_derivative_inputs_received,
                                                                                              T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received),
                       tmp_derivative_padded_half(derivative_size_received + padding_received);
    size_t tmp_example_index,
              tmp_kernel_index,
              tmp_index,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_derivative_inputs,
                  tmp_scale(1_T / static_cast<T_>(kernel_size_received));
    T_ *tmp_ptr_array_derivatives,
         tmp_error;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * derivative_size_received + tmp_derivative_timed_batched_index;
        
        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = tmp_ptr_array_derivative_inputs[tmp_input_index] * tmp_scale;
            
            for(tmp_kernel_index = 0_zu; tmp_kernel_index != kernel_size_received; ++tmp_kernel_index)
            {
                tmp_index = tmp_input_index * stride_received + tmp_kernel_index * dilation_received;

                if(tmp_index < padding_received || tmp_index >= tmp_derivative_padded_half) { continue; }

                tmp_ptr_array_derivatives[tmp_index - padding_received] += tmp_error;
            }
        }
    }
}

void Neural_Network::Backward_Pass__Dropout__ShakeDrop__Loop(size_t const time_step_index_received,
                                                                                                    size_t const batch_size_received,
                                                                                                    size_t const derivative_size_received,
                                                                                                    bool const *const ptr_array_mask_dopout_shakedrop_received,
                                                                                                    T_ const lower_bound_received,
                                                                                                    T_ const upper_bound_received,
                                                                                                    T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_derivative_timed_index(derivative_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * tmp_derivative_timed_index),
                       tmp_layer_timed_batched_index(this->batch_size * time_step_index_received);
    size_t tmp_example_index,
              tmp_derivative_index;
    
    T_ *tmp_ptr_array_derivatives;

    this->ptr_array_Class_Generator_Real_ShakeDrop->Range(lower_bound_received, upper_bound_received);

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        if(ptr_array_mask_dopout_shakedrop_received[tmp_layer_timed_batched_index + tmp_example_index])
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * derivative_size_received + tmp_derivative_timed_batched_index;

            for(tmp_derivative_index = 0_zu; tmp_derivative_index != derivative_size_received; ++tmp_derivative_index) { tmp_ptr_array_derivatives[tmp_derivative_index] *= (*this->ptr_array_Class_Generator_Real_ShakeDrop)(); }
        }
    }
}

void Neural_Network::Backward_Pass__FC__Loop(size_t const time_step_index_received,
                                                                          size_t const batch_size_received,
                                                                          size_t const input_size_received,
                                                                          size_t const derivative_size_received,
                                                                          T_ const *const ptr_array_derivative_inputs_received,
                                                                          T_ const *const ptr_array_parameters_received,
                                                                          T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_input_index,
              tmp_connection_index;
    
    T_ const *tmp_ptr_array_derivative_inputs,
                  *tmp_ptr_array_parameters;
    T_ *tmp_ptr_array_derivatives,
         tmp_error;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_parameters = ptr_array_parameters_received;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * derivative_size_received + tmp_derivative_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index,
                                                                                                            tmp_ptr_array_parameters += derivative_size_received)
        {
            tmp_error = tmp_ptr_array_derivative_inputs[tmp_input_index];
            
            for(tmp_connection_index = 0_zu; tmp_connection_index != derivative_size_received; ++tmp_connection_index) { tmp_ptr_array_derivatives[tmp_connection_index] += tmp_error * tmp_ptr_array_parameters[tmp_connection_index]; }
        }
    }
}

void Neural_Network::Backward_Pass__Identity__Loop(size_t const time_step_index_received,
                                                                                size_t const batch_size_received,
                                                                                size_t const input_size_received,
                                                                                T_ const *const ptr_array_derivative_inputs_received,
                                                                                T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_derivatives[tmp_input_index] += tmp_ptr_array_derivative_inputs[tmp_input_index]; }
    }
}

void Neural_Network::Backward_Pass__Max_Pooling__Loop(size_t const time_step_index_received,
                                                                                         size_t const batch_size_received,
                                                                                         size_t const input_size_received,
                                                                                         size_t const derivative_size_received,
                                                                                         size_t const padding_received,
                                                                                         size_t const *const ptr_array_indices_received,
                                                                                         T_ const *const ptr_array_derivative_inputs_received,
                                                                                         T_ *const ptr_array_derivatives_received)
{
    size_t const *tmp_ptr_array_indices,
                       tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received),
                       tmp_derivative_padded_half(derivative_size_received + padding_received);
    size_t tmp_example_index,
              tmp_indice,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives,
         tmp_error;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_indices = ptr_array_indices_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * derivative_size_received + tmp_derivative_timed_batched_index;
        
        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_indice = tmp_ptr_array_indices[tmp_input_index];
            
            if(tmp_indice < padding_received || tmp_indice >= tmp_derivative_padded_half) { continue; }

            tmp_error = tmp_ptr_array_derivative_inputs[tmp_input_index];
            
            tmp_ptr_array_derivatives[tmp_indice - padding_received] += tmp_error;
        }
    }
}

void Neural_Network::Backward_Pass__Residual__Loop(size_t const time_step_index_received,
                                                                                  size_t const batch_size_received,
                                                                                  size_t const input_size_received,
                                                                                  size_t const derivative_size_received,
                                                                                  size_t const padding_received,
                                                                                  T_ const *const ptr_array_derivative_inputs_received,
                                                                                  T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives;
    
    if(input_size_received == derivative_size_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * derivative_size_received + tmp_derivative_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            
        #if defined(COMPILE_DEBUG_PRINT)
            PRINT_FORMAT("BACKWARD, TIME[%zu], DATA[%zu]" NEW_LINE, time_step_index_received, tmp_example_index);
            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { PRINT_FORMAT("%+.2f ", tmp_ptr_array_derivatives[tmp_input_index]); }
            PRINT_FORMAT(NEW_LINE);
            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { PRINT_FORMAT("%+.2f ", tmp_ptr_array_derivative_inputs[tmp_input_index]); }
            PRINT_FORMAT(NEW_LINE "\t=" NEW_LINE);
        #endif

            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_derivatives[tmp_input_index] += tmp_ptr_array_derivative_inputs[tmp_input_index];
            }
            
        #if defined(COMPILE_DEBUG_PRINT)
            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { PRINT_FORMAT("%+.2f ", tmp_ptr_array_derivatives[tmp_input_index]); }
            PRINT_FORMAT(NEW_LINE);
        #endif
        }
    }
    else if(input_size_received > derivative_size_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * derivative_size_received + tmp_derivative_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            
            for(tmp_input_index = 0_zu; tmp_input_index != derivative_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_derivatives[tmp_input_index] += tmp_ptr_array_derivative_inputs[tmp_input_index + padding_received];
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * derivative_size_received + tmp_derivative_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            
            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_derivatives[tmp_input_index + padding_received] += tmp_ptr_array_derivative_inputs[tmp_input_index];
            }
        }
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Backward_Pass__FC__DF__Loop(size_t const time_step_index_received,
                                                                                  size_t const batch_size_received,
                                                                                  size_t const input_size_received,
                                                                                  enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *const ptr_array_type_activations_functions_received,
                                                                                  T_ const *const ptr_array_activations_steepness_received,
                                                                                  T_ const *const ptr_array_pre_AFs_received,
                                                                                  T_ const *const ptr_array_AFs_received,
                                                                                  T_ const *const ptr_array_derivative_inputs_received,
                                                                                  T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_pre_AFs,
                 *tmp_ptr_array_AFs,
                 *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives;

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_pre_AFs = ptr_array_pre_AFs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        tmp_ptr_array_AFs = ptr_array_AFs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_ptr_array_derivatives[tmp_input_index] = tmp_ptr_array_derivative_inputs[tmp_input_index] * this->Activation_Function_Derive(ptr_array_type_activations_functions_received[tmp_input_index],
                                                                                                                                                                                                         tmp_ptr_array_pre_AFs[tmp_input_index],
                                                                                                                                                                                                         ptr_array_activations_steepness_received[tmp_input_index],
                                                                                                                                                                                                         tmp_ptr_array_AFs[tmp_input_index]);
        }
    }
}

void Neural_Network::Backward_Pass__FC__DF_Ind_RNN__Loop(size_t const time_step_index_received,
                                                                                                size_t const batch_size_received,
                                                                                                size_t const input_size_received,
                                                                                                T_ const *const ptr_array_parameters_received,
                                                                                                enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *const ptr_array_type_activations_functions_received,
                                                                                                T_ const *const ptr_array_activations_steepness_received,
                                                                                                T_ const *const ptr_array_pre_AFs_received,
                                                                                                T_ const *const ptr_array_AFs_received,
                                                                                                T_ const *const ptr_array_derivative_inputs_received,
                                                                                                T_ *const ptr_array_dAFs_received,
                                                                                                T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_input_next_timed_batched_index(this->batch_size * input_size_received * (time_step_index_received + 1_zu));
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_pre_AFs,
                 *tmp_ptr_array_AFs,
                 *tmp_ptr_array_next_timed_dAFs,
                 *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_dAFs,
         *tmp_ptr_array_derivatives;
    
    if(time_step_index_received + 1_zu != this->number_recurrent_depth)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_pre_AFs = ptr_array_pre_AFs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_AFs = ptr_array_AFs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            tmp_ptr_array_dAFs = ptr_array_dAFs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_next_timed_dAFs = ptr_array_dAFs_received + tmp_example_index * input_size_received + tmp_input_next_timed_batched_index;
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            
            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_dAFs[tmp_input_index] = this->Activation_Function_Derive(ptr_array_type_activations_functions_received[tmp_input_index],
                                                                                                                           tmp_ptr_array_pre_AFs[tmp_input_index],
                                                                                                                           ptr_array_activations_steepness_received[tmp_input_index],
                                                                                                                           tmp_ptr_array_AFs[tmp_input_index]);

                /*
                PRINT_FORMAT("%f from %f, %f" NEW_LINE,
                            Cast_T(tmp_ptr_array_dAFs[tmp_input_index]),
                            Cast_T(tmp_ptr_array_pre_AFs[tmp_input_index]),
                            Cast_T(tmp_ptr_array_AFs[tmp_input_index]));
                */

                tmp_ptr_array_derivatives[tmp_input_index] = tmp_ptr_array_derivative_inputs[tmp_input_index] * tmp_ptr_array_dAFs[tmp_input_index];
                
                tmp_ptr_array_dAFs[tmp_input_index] = tmp_ptr_array_derivatives[tmp_input_index]
                                                                                                        +
                                                                           ptr_array_parameters_received[tmp_input_index]
                                                                                                        *
                                                                           tmp_ptr_array_dAFs[tmp_input_index]
                                                                                                        *
                                                                           tmp_ptr_array_next_timed_dAFs[tmp_input_index];
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_pre_AFs = ptr_array_pre_AFs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_AFs = ptr_array_AFs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            tmp_ptr_array_dAFs = ptr_array_dAFs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
            
            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_dAFs[tmp_input_index] = tmp_ptr_array_derivatives[tmp_input_index] = tmp_ptr_array_derivative_inputs[tmp_input_index] * this->Activation_Function_Derive(ptr_array_type_activations_functions_received[tmp_input_index],
                                                                                                                                                                                                                                                                        tmp_ptr_array_pre_AFs[tmp_input_index],
                                                                                                                                                                                                                                                                        ptr_array_activations_steepness_received[tmp_input_index],
                                                                                                                                                                                                                                                                        tmp_ptr_array_AFs[tmp_input_index]);
            }
        }
    }
}

void Neural_Network::Backward_Pass__Batch_Normalization__Loop(size_t const time_step_index_received,
                                                                                                    size_t const batch_size_received,
                                                                                                    size_t const input_size_received,
                                                                                                    T_ const *const ptr_array_means_received,
                                                                                                    T_ const *const ptr_array_variances_received,
                                                                                                    T_ const *const ptr_array_scales_received,
                                                                                                    T_ const *const ptr_array_inputs_received,
                                                                                                    T_ const *const ptr_array_inputs_hats_received,
                                                                                                    T_ const *const ptr_array_derivative_inputs_received,
                                                                                                    T_ *const ptr_array_derivatives_scales_received,
                                                                                                    T_ *const ptr_array_derivatives_shifts_received,
                                                                                                    T_ *const ptr_array_derivatives_means_received,
                                                                                                    T_ *const ptr_array_derivatives_variances_received,
                                                                                                    T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index,
              tmp_input_data_timed_index;

    T_ const tmp_batch_scale(1_T / static_cast<T_>(batch_size_received));
    T_ tmp_error,
         tmp_variance_b;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_index] += tmp_error * ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index];
            
            // Derivative shift.
            // dShift += dY
            ptr_array_derivatives_shifts_received[tmp_input_index] += tmp_error;

            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= ptr_array_scales_received[tmp_input_index];
            
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] += tmp_error * ( -1_T / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] += tmp_error * (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) * ( -1_T / (tmp_variance_b * tmp_variance_b) );

            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];

            // First
            // dX_h *= r_correction / variance_b
            tmp_error *= 1_T / tmp_variance_b;
            
            // Middle
            // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
            tmp_error += ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] * ( (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) / (static_cast<T_>(batch_size_received) * tmp_variance_b) );

            // Last
            // dX_h += dMean_b * 1 / N
            // dX_h += dMean_b / N
            tmp_error += ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] * tmp_batch_scale;

            // dX = dX_h
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
}

void Neural_Network::Backward_Pass__Batch_Normalization__Loop(size_t const time_step_index_received,
                                                                                                    size_t const batch_size_received,
                                                                                                    size_t const input_size_received,
                                                                                                    T_ const *const ptr_array_means_received,
                                                                                                    T_ const *const ptr_array_variances_received,
                                                                                                    T_ const *const ptr_array_scales_received,
                                                                                                    T_ const *const ptr_array_inputs_received,
                                                                                                    T_ const *const ptr_array_inputs_hats_received,
                                                                                                    T_ const *const ptr_array_derivative_inputs_received,
                                                                                                    T_ *const ptr_array_derivatives_scales_received,
                                                                                                    T_ *const ptr_array_derivatives_means_received,
                                                                                                    T_ *const ptr_array_derivatives_variances_received,
                                                                                                    T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index,
              tmp_input_data_timed_index;

    T_ const tmp_batch_scale(1_T / static_cast<T_>(batch_size_received));
    T_ tmp_error,
         tmp_variance_b;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_index] += tmp_error * ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index];
            
            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= ptr_array_scales_received[tmp_input_index];
            
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] += tmp_error * ( -1_T / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] += tmp_error * (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) * ( -1_T / (tmp_variance_b * tmp_variance_b) );

            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];

            // First
            // dX_h *= r_correction / variance_b
            tmp_error *= 1_T / tmp_variance_b;
            
            // Middle
            // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
            tmp_error += ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] * ( (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) / (static_cast<T_>(batch_size_received) * tmp_variance_b) );

            // Last
            // dX_h += dMean_b * 1 / N
            // dX_h += dMean_b / N
            tmp_error += ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] * tmp_batch_scale;

            // dX = dX_h
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
}

void Neural_Network::Backward_Pass__Batch_Renormalization__Loop(size_t const time_step_index_received,
                                                                                                       size_t const batch_size_received,
                                                                                                       size_t const input_size_received,
                                                                                                       T_ const *const ptr_array_means_received,
                                                                                                       T_ const *const ptr_array_variances_received,
                                                                                                       T_ const *const ptr_array_scales_received,
                                                                                                       T_ const *const ptr_array_r_corrections_received,
                                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                                       T_ const *const ptr_array_inputs_hats_received,
                                                                                                       T_ const *const ptr_array_derivative_inputs_received,
                                                                                                       T_ *const ptr_array_derivatives_scales_received,
                                                                                                       T_ *const ptr_array_derivatives_shifts_received,
                                                                                                       T_ *const ptr_array_derivatives_means_received,
                                                                                                       T_ *const ptr_array_derivatives_variances_received,
                                                                                                       T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index,
              tmp_input_data_timed_index;

    T_ const tmp_batch_scale(1_T / static_cast<T_>(batch_size_received));
    T_ tmp_error,
         tmp_variance_b,
         tmp_negate_r_correction;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];
            tmp_negate_r_correction = -ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index]; // Negate.

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_index] += tmp_error * ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index];
            
            // Derivative shift.
            // dShift += dY
            ptr_array_derivatives_shifts_received[tmp_input_index] += tmp_error;

            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= ptr_array_scales_received[tmp_input_index];
            
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] += tmp_error * ( tmp_negate_r_correction / tmp_variance_b );
            
            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] += tmp_error * (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) * ( tmp_negate_r_correction / (tmp_variance_b * tmp_variance_b) );

            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];
            
            // First
            // dX_h *= r_correction / variance_b
            tmp_error *= ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index] / tmp_variance_b;
            
            // Middle
            // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
            tmp_error += ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] * ( (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) / (static_cast<T_>(batch_size_received) * tmp_variance_b) );
            
            // Last
            // dX_h += dMean_b * 1 / N
            // dX_h += dMean_b / N
            tmp_error += ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] * tmp_batch_scale;
            
            // dX = dX_h
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
}

void Neural_Network::Backward_Pass__Batch_Renormalization__Loop(size_t const time_step_index_received,
                                                                                                       size_t const batch_size_received,
                                                                                                       size_t const input_size_received,
                                                                                                       T_ const *const ptr_array_means_received,
                                                                                                       T_ const *const ptr_array_variances_received,
                                                                                                       T_ const *const ptr_array_scales_received,
                                                                                                       T_ const *const ptr_array_r_corrections_received,
                                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                                       T_ const *const ptr_array_inputs_hats_received,
                                                                                                       T_ const *const ptr_array_derivative_inputs_received,
                                                                                                       T_ *const ptr_array_derivatives_scales_received,
                                                                                                       T_ *const ptr_array_derivatives_means_received,
                                                                                                       T_ *const ptr_array_derivatives_variances_received,
                                                                                                       T_ *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index,
              tmp_input_data_timed_index;

    T_ const tmp_batch_scale(1_T / static_cast<T_>(batch_size_received));
    T_ tmp_error,
         tmp_variance_b,
         tmp_negate_r_correction;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];
            tmp_negate_r_correction = -ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index]; // Negate.

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_index] += tmp_error * ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index];
            
            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= ptr_array_scales_received[tmp_input_index];
            
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] += tmp_error * ( tmp_negate_r_correction / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] += tmp_error * (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) * ( tmp_negate_r_correction / (tmp_variance_b * tmp_variance_b) );

            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];

            // First
            // dX_h *= r_correction / variance_b
            tmp_error *= ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index] / tmp_variance_b;
            
            // Middle
            // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
            tmp_error += ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] * ( (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) / (static_cast<T_>(batch_size_received) * tmp_variance_b) );

            // Last
            // dX_h += dMean_b * 1 / N
            // dX_h += dMean_b / N
            tmp_error += ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] * tmp_batch_scale;

            // dX = dX_h
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
}
