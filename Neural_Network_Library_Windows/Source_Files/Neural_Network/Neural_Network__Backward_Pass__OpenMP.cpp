#include "stdafx.hpp"

#include <Math/Math.hpp>

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::FF__Backward_Pass_Batch__OpenMP(size_t const batch_size_received)
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
    #pragma omp single
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

        #pragma omp single
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
                    this->Backward_Pass__Average_Pooling__OpenMP(0_zu,
                                                                                                 batch_size_received,
                                                                                                 tmp_layer_number_outputs,
                                                                                                 tmp_ptr_array_layer_gradients,
                                                                                                 tmp_ptr_next_layer_it);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    this->Backward_Pass__FC__OpenMP(batch_size_received,
                                                                             tmp_layer_number_outputs,
                                                                             tmp_ptr_array_layer_gradients,
                                                                             tmp_ptr_next_layer_it);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                    this->Backward_Pass__Max_Pooling__OpenMP(0_zu,
                                                                                           batch_size_received,
                                                                                           tmp_layer_number_outputs,
                                                                                           tmp_ptr_array_layer_gradients,
                                                                                           tmp_ptr_next_layer_it);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    this->Backward_Pass__Residual__OpenMP(0_zu,
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
                this->Backward_Pass__Gradient__FC__OpenMP(0_zu,
                                                                                         batch_size_received,
                                                                                         tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                this->Backward_Pass__Gradient__Residual__OpenMP(batch_size_received, tmp_ptr_layer_it);

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

void Neural_Network::FF__Backward_Pass_Batch__Pre_Training__OpenMP(size_t const batch_size_received)
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
    #pragma omp single
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

    #pragma omp single
    MEMSET(tmp_ptr_array_layer_gradients,
                   0,
                   this->batch_size * tmp_layer_number_outputs * sizeof(T_));
    // |END| Clear past error(s). |END|
    
    // Propagate the error(s) to the layer.
    switch(tmp_ptr_decoded_layer->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            this->Backward_Pass__FC__OpenMP(batch_size_received,
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
            this->Backward_Pass__Gradient__FC__OpenMP(0_zu,
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

void Neural_Network::Backward_Pass__FC__OpenMP(size_t const batch_size_received,
                                                                                size_t const derivative_size_received,
                                                                                T_ *const ptr_array_derivatives_received,
                                                                                struct Layer const *const ptr_layer_it_received)
{
    if(ptr_layer_it_received->type_group == MyEA::Common::ENUM_TYPE_GROUP::TYPE_GROUP_RESIDUAL)
    {
        this->Backward_Pass__Residual__FC__OpenMP(0_zu,
                                                                                 batch_size_received,
                                                                                 derivative_size_received,
                                                                                 ptr_array_derivatives_received,
                                                                                 ptr_layer_it_received);
    }
    else
    {
        this->Backward_Pass__FC__OpenMP(0_zu,
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

void Neural_Network::Backward_Pass__Average_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                                    size_t const batch_size_received,
                                                                                                    size_t const derivative_size_received,
                                                                                                    T_ *const ptr_array_derivatives_received,
                                                                                                    struct Layer const *const ptr_layer_it_received)
{
    this->Backward_Pass__Average_Pooling__OpenMP(time_step_index_received,
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

void Neural_Network::Backward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                                                                size_t const batch_size_received,
                                                                                size_t const derivative_size_received,
                                                                                T_ *const ptr_array_derivatives_received,
                                                                                struct Layer const *const ptr_layer_it_received)
{
    struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
    
    this->Backward_Pass__FC__OpenMP(time_step_index_received,
                                                            batch_size_received,
                                                            static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit),
                                                            derivative_size_received,
                                                            tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                            this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                            ptr_array_derivatives_received);
}

void Neural_Network::Backward_Pass__Max_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                              size_t const batch_size_received,
                                                                                              size_t const derivative_size_received,
                                                                                              T_ *const ptr_array_derivatives_received,
                                                                                              struct Layer const *const ptr_layer_it_received)
{
    struct Basic_indice_unit *const tmp_ptr_layer_first_basic_indice_unit(ptr_layer_it_received->ptr_array_basic_indice_units);
    
    this->Backward_Pass__Max_Pooling__OpenMP(time_step_index_received,
                                                                           batch_size_received,
                                                                           *ptr_layer_it_received->ptr_number_outputs,
                                                                           derivative_size_received,
                                                                           ptr_layer_it_received->pooling_values[2u],
                                                                           tmp_ptr_layer_first_basic_indice_unit->ptr_array_indices,
                                                                           tmp_ptr_layer_first_basic_indice_unit->ptr_array_errors,
                                                                           ptr_array_derivatives_received);
}

void Neural_Network::Backward_Pass__Residual__OpenMP(size_t const time_step_index_received,
                                                                                        size_t const batch_size_received,
                                                                                        size_t const derivative_size_received,
                                                                                        T_ *const ptr_array_derivatives_received,
                                                                                        struct Layer const *const ptr_layer_it_received)
{
    this->Backward_Pass__Residual__OpenMP(time_step_index_received,
                                                                     batch_size_received,
                                                                     *ptr_layer_it_received->ptr_number_outputs,
                                                                     derivative_size_received,
                                                                     ptr_layer_it_received->pooling_values[2u],
                                                                     ptr_layer_it_received->ptr_array_basic_units->ptr_array_errors,
                                                                     ptr_array_derivatives_received);
}

void Neural_Network::Backward_Pass__Residual__Block__OpenMP(size_t const time_step_index_received,
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
        #pragma omp single
        MEMSET(tmp_ptr_array_derivatives + this->batch_size * derivative_size_received * time_step_index_received,
                       0,
                       this->batch_size * derivative_size_received * sizeof(T_));
        // |END| Clear past error(s). |END|
    }
    else { tmp_ptr_array_derivatives = ptr_array_derivatives_received; }

    this->Backward_Pass__Residual__OpenMP(time_step_index_received,
                                                                     batch_size_received,
                                                                     *ptr_layer_it_received->ptr_number_outputs,
                                                                     derivative_size_received,
                                                                     ptr_layer_it_received->pooling_values[2u],
                                                                     ptr_layer_it_received->ptr_array_basic_units->ptr_array_errors,
                                                                     tmp_ptr_array_derivatives);
    
    // Dropout, ShakeDrop.
    if(ptr_layer_it_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP)
    {
        this->Backward_Pass__Dropout__ShakeDrop__OpenMP(time_step_index_received,
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
                this->Backward_Pass__Batch_Normalization__OpenMP(time_step_index_received,
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
                this->Backward_Pass__Batch_Renormalization__OpenMP(time_step_index_received,
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
        #pragma omp single
        MEMCPY(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                       tmp_ptr_array_derivatives + this->batch_size * derivative_size_received * time_step_index_received,
                       this->batch_size * derivative_size_received * sizeof(T_));
    }
}

void Neural_Network::Backward_Pass__Residual__FC__OpenMP(size_t const time_step_index_received,
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
                    #pragma omp single
                    MEMCPY(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                                  ptr_layer_it_received->ptr_array_normalized_units->normalized_batch_units.ptr_array_errors + this->batch_size * derivative_size_received * time_step_index_received,
                                  this->batch_size * derivative_size_received * sizeof(T_));
                }
                //     --------------> [FC] --> FC ...
                //    /
                // {FC} --> ResNet ---> ...
                else
                {
                    this->Backward_Pass__Identity__OpenMP(time_step_index_received,
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
            #pragma omp single
            MEMCPY(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                           ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units->ptr_array_dAFs + this->batch_size * derivative_size_received * time_step_index_received,
                           this->batch_size * derivative_size_received * sizeof(T_));
        }
        else
        {
            #pragma omp single
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
        
        this->Backward_Pass__FC__OpenMP(time_step_index_received,
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

void Neural_Network::Backward_Pass__Gradient__FC__OpenMP(size_t const time_step_index_received,
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
                this->Backward_Pass__Batch_Normalization__OpenMP(time_step_index_received,
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
                this->Backward_Pass__Batch_Renormalization__OpenMP(time_step_index_received,
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
        this->Backward_Pass__FC__DF_Ind_RNN__OpenMP(time_step_index_received,
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
        this->Backward_Pass__FC__DF__OpenMP(time_step_index_received,
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
                this->Backward_Pass__Batch_Normalization__OpenMP(time_step_index_received,
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
                this->Backward_Pass__Batch_Renormalization__OpenMP(time_step_index_received,
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
    #pragma omp single
    MEMCPY(tmp_ptr_layer_first_neuron_unit->ptr_array_errors + this->batch_size * tmp_output_size * time_step_index_received,
                  tmp_ptr_array_derivative_inputs + this->batch_size * tmp_output_size * time_step_index_received,
                  this->batch_size * tmp_output_size * sizeof(T_));
    // |END| Copy derivative to derivative neurons. |END|
}

void Neural_Network::Backward_Pass__Gradient__Residual__OpenMP(size_t const batch_size_received, struct Layer const *const ptr_layer_it_received)
{
    struct Layer const *const tmp_ptr_layer_end(ptr_layer_it_received + 1);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + static_cast<size_t>(ptr_layer_it_received - this->ptr_array_layers) + ptr_layer_it_received->block_depth);
    
    // Remaining layer(s).
    for(; tmp_ptr_layer_it != tmp_ptr_layer_end; --tmp_ptr_layer_it)
    {
        this->Backward_Pass__Gradient__Residual__Layer__OpenMP(false,
                                                                                                    batch_size_received,
                                                                                                    tmp_ptr_layer_it);
    }
    // |END| Remaining layer(s). |END|
    
    // First block layer.
    this->Backward_Pass__Gradient__Residual__Layer__OpenMP(true,
                                                                                                batch_size_received,
                                                                                                tmp_ptr_layer_it);
    // |END| First block layer. |END|
}

void Neural_Network::Backward_Pass__Gradient__Residual__Layer__OpenMP(bool const is_block_input_layer_received,
                                                                                                                   size_t const batch_size_received,
                                                                                                                   struct Layer *&ptr_layer_it_received)
{
    size_t const tmp_layer_number_outputs(*ptr_layer_it_received->ptr_number_outputs);
    
    T_ *const tmp_ptr_array_layer_gradients(ptr_layer_it_received->ptr_array_derivative_outputs);
    
    struct Layer const *const tmp_ptr_next_layer_it(ptr_layer_it_received->next_connected_layers[0u]);
    
    // Clear past error(s).
    #pragma omp single
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
            this->Backward_Pass__Average_Pooling__OpenMP(0_zu,
                                                                                         batch_size_received,
                                                                                         tmp_layer_number_outputs,
                                                                                         tmp_ptr_array_layer_gradients,
                                                                                         tmp_ptr_next_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            this->Backward_Pass__Residual__FC__OpenMP(0_zu,
                                                                                     batch_size_received,
                                                                                     tmp_layer_number_outputs,
                                                                                     tmp_ptr_array_layer_gradients,
                                                                                     tmp_ptr_next_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            this->Backward_Pass__Max_Pooling__OpenMP(0_zu,
                                                                                   batch_size_received,
                                                                                   tmp_layer_number_outputs,
                                                                                   tmp_ptr_array_layer_gradients,
                                                                                   tmp_ptr_next_layer_it);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
            this->Backward_Pass__Residual__Block__OpenMP(0_zu,
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
            this->Backward_Pass__Gradient__Residual__FC__OpenMP(is_block_input_layer_received,
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

void Neural_Network::Backward_Pass__Gradient__Residual__FC__OpenMP(bool const is_block_input_layer_received,
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
    #pragma omp single
    MEMSET(tmp_ptr_array_derivative_inputs + this->batch_size * tmp_input_size * time_step_index_received,
                  0,
                  this->batch_size * tmp_input_size * sizeof(T_));
    // |END| Clear past error(s). |END|
    
    this->Backward_Pass__FC__OpenMP(time_step_index_received,
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
            this->Backward_Pass__FC__DF_Ind_RNN__OpenMP(time_step_index_received,
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
            this->Backward_Pass__FC__DF__OpenMP(time_step_index_received,
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
                this->Backward_Pass__Batch_Normalization__OpenMP(time_step_index_received,
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
                this->Backward_Pass__Batch_Renormalization__OpenMP(time_step_index_received,
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

void Neural_Network::Backward_Pass__Average_Pooling__OpenMP(size_t const time_step_index_received,
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
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received),
                       tmp_derivative_padded_half(derivative_size_received + padding_received);
    size_t tmp_kernel_index,
              tmp_index,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_derivative_inputs,
                  tmp_scale(1_T / static_cast<T_>(kernel_size_received));
    T_ *tmp_ptr_array_derivatives,
         tmp_error;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
        
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

void Neural_Network::Backward_Pass__Dropout__ShakeDrop__OpenMP(size_t const time_step_index_received,
                                                                                                          size_t const batch_size_received,
                                                                                                          size_t const derivative_size_received,
                                                                                                          bool const *const ptr_array_mask_dopout_shakedrop_received,
                                                                                                          T_ const lower_bound_received,
                                                                                                          T_ const upper_bound_received,
                                                                                                          T_ *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int,
        tmp_thread_index__int(omp_get_thread_num());
    
    size_t const tmp_derivative_timed_index(derivative_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * tmp_derivative_timed_index),
                       tmp_layer_timed_batched_index(this->batch_size * time_step_index_received);
    size_t tmp_derivative_index;
    
    T_ *tmp_ptr_array_derivatives;

    this->ptr_array_Class_Generator_Real_ShakeDrop[tmp_thread_index__int].Range(lower_bound_received, upper_bound_received);
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();
        
        if(ptr_array_mask_dopout_shakedrop_received[tmp_layer_timed_batched_index + static_cast<size_t>(tmp_example_index__int)])
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;

            for(tmp_derivative_index = 0_zu; tmp_derivative_index != derivative_size_received; ++tmp_derivative_index) { tmp_ptr_array_derivatives[tmp_derivative_index] *= this->ptr_array_Class_Generator_Real_ShakeDrop[tmp_thread_index__int](); }
        }
    }
}

void Neural_Network::Backward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                                                                size_t const batch_size_received,
                                                                                size_t const input_size_received,
                                                                                size_t const derivative_size_received,
                                                                                T_ const *const ptr_array_derivative_inputs_received,
                                                                                T_ const *const ptr_array_parameters_received,
                                                                                T_ *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_input_index,
              tmp_connection_index;
    
    T_ const *tmp_ptr_array_derivative_inputs,
                  *tmp_ptr_array_parameters;
    T_ *tmp_ptr_array_derivatives,
         tmp_error;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_parameters = ptr_array_parameters_received;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index,
                                                                                                            tmp_ptr_array_parameters += derivative_size_received)
        {
            tmp_error = tmp_ptr_array_derivative_inputs[tmp_input_index];
            
            for(tmp_connection_index = 0_zu; tmp_connection_index != derivative_size_received; ++tmp_connection_index) { tmp_ptr_array_derivatives[tmp_connection_index] += tmp_error * tmp_ptr_array_parameters[tmp_connection_index]; }
        }
    }
}

void Neural_Network::Backward_Pass__Identity__OpenMP(size_t const time_step_index_received,
                                                                                      size_t const batch_size_received,
                                                                                      size_t const input_size_received,
                                                                                      T_ const *const ptr_array_derivative_inputs_received,
                                                                                      T_ *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received);
    size_t tmp_input_index;
    
    T_ const *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_derivatives[tmp_input_index] += tmp_ptr_array_derivative_inputs[tmp_input_index]; }
    }
}

void Neural_Network::Backward_Pass__Max_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                              size_t const batch_size_received,
                                                                                              size_t const input_size_received,
                                                                                              size_t const derivative_size_received,
                                                                                              size_t const padding_received,
                                                                                              size_t const *const ptr_array_indices_received,
                                                                                              T_ const *const ptr_array_derivative_inputs_received,
                                                                                              T_ *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const *tmp_ptr_array_indices,
                       tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received),
                       tmp_derivative_padded_half(derivative_size_received + padding_received);
    size_t tmp_indice,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives,
         tmp_error;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_indices = ptr_array_indices_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
        
        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_indice = tmp_ptr_array_indices[tmp_input_index];
            
            if(tmp_indice < padding_received || tmp_indice >= tmp_derivative_padded_half) { continue; }

            tmp_error = tmp_ptr_array_derivative_inputs[tmp_input_index];
            
            tmp_ptr_array_derivatives[tmp_indice - padding_received] += tmp_error;
        }
    }
}

void Neural_Network::Backward_Pass__Residual__OpenMP(size_t const time_step_index_received,
                                                                                        size_t const batch_size_received,
                                                                                        size_t const input_size_received,
                                                                                        size_t const derivative_size_received,
                                                                                        size_t const padding_received,
                                                                                        T_ const *const ptr_array_derivative_inputs_received,
                                                                                        T_ *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_input_index;
    
    T_ const *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives;
    
    if(input_size_received == derivative_size_received)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_derivatives[tmp_input_index] += tmp_ptr_array_derivative_inputs[tmp_input_index];
            }
        }
    }
    else if(input_size_received > derivative_size_received)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            
            for(tmp_input_index = 0_zu; tmp_input_index != derivative_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_derivatives[tmp_input_index] += tmp_ptr_array_derivative_inputs[tmp_input_index + padding_received];
            }
        }
    }
    else
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            
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

void Neural_Network::Backward_Pass__FC__DF__OpenMP(size_t const time_step_index_received,
                                                                                        size_t const batch_size_received,
                                                                                        size_t const input_size_received,
                                                                                        enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *const ptr_array_type_activations_functions_received,
                                                                                        T_ const *const ptr_array_activations_steepness_received,
                                                                                        T_ const *const ptr_array_pre_AFs_received,
                                                                                        T_ const *const ptr_array_AFs_received,
                                                                                        T_ const *const ptr_array_derivative_inputs_received,
                                                                                        T_ *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received);
    size_t tmp_input_index;
    
    T_ const *tmp_ptr_array_pre_AFs,
                 *tmp_ptr_array_AFs,
                 *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_pre_AFs = ptr_array_pre_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_ptr_array_AFs = ptr_array_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_ptr_array_derivatives[tmp_input_index] = tmp_ptr_array_derivative_inputs[tmp_input_index] * this->Activation_Function_Derive(ptr_array_type_activations_functions_received[tmp_input_index],
                                                                                                                                                                                                         tmp_ptr_array_pre_AFs[tmp_input_index],
                                                                                                                                                                                                         ptr_array_activations_steepness_received[tmp_input_index],
                                                                                                                                                                                                         tmp_ptr_array_AFs[tmp_input_index]);
        }
    }
}

void Neural_Network::Backward_Pass__FC__DF_Ind_RNN__OpenMP(size_t const time_step_index_received,
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
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_input_next_timed_batched_index(this->batch_size * input_size_received * (time_step_index_received + 1_zu));
    size_t tmp_input_index;
    
    T_ const *tmp_ptr_array_pre_AFs,
                 *tmp_ptr_array_AFs,
                 *tmp_ptr_array_next_timed_dAFs,
                 *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_dAFs,
         *tmp_ptr_array_derivatives;
    
    if(time_step_index_received + 1_zu != this->number_recurrent_depth)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_pre_AFs = ptr_array_pre_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_AFs = ptr_array_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

            tmp_ptr_array_dAFs = ptr_array_dAFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_next_timed_dAFs = ptr_array_dAFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_next_timed_batched_index;
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            
            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_dAFs[tmp_input_index] = this->Activation_Function_Derive(ptr_array_type_activations_functions_received[tmp_input_index],
                                                                                                                           tmp_ptr_array_pre_AFs[tmp_input_index],
                                                                                                                           ptr_array_activations_steepness_received[tmp_input_index],
                                                                                                                           tmp_ptr_array_AFs[tmp_input_index]);
                
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
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_pre_AFs = ptr_array_pre_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_AFs = ptr_array_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

            tmp_ptr_array_dAFs = ptr_array_dAFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            
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

void Neural_Network::Backward_Pass__Batch_Normalization__OpenMP(size_t const time_step_index_received,
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
    int const tmp_batch_size__int(static_cast<int>(batch_size_received)),
                  tmp_units_size__int(static_cast<int>(input_size_received)),
                  tmp_number_threads__int(MyEA::Math::Minimum<int>(tmp_batch_size__int, static_cast<int>(this->number_threads)));
    int tmp_thread_index__int,
        tmp_example_index__int,
        tmp_input_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index,
              tmp_input_data_timed_index,
              tmp_input_thread_timed_index,
              tmp_input_thread_parameter_index;
    
    T_ const tmp_batch_scale(1_T / static_cast<T_>(batch_size_received));
    T_ *tmp_ptr_array_derivative_mean,
         *tmp_ptr_array_derivative_variance,
         tmp_reduction_derivative_mean,
         tmp_reduction_derivative_variance,
         tmp_error,
         tmp_variance_b;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_input_thread_timed_index = static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth + tmp_input_timed_index;
        tmp_input_thread_parameter_index = static_cast<size_t>(tmp_thread_index__int) * this->total_parameters_allocated;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error * ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index];
                
            // Derivative shift.
            // dShift += dY
            ptr_array_derivatives_shifts_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error;

            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= ptr_array_scales_received[tmp_input_index];
                
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * ( -1_T / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) * ( -1_T / (tmp_variance_b * tmp_variance_b) );
                
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
        
    // Reduction.
    #pragma omp for schedule(static)
    for(tmp_input_index__int = 0; tmp_input_index__int < tmp_units_size__int; ++tmp_input_index__int)
    {
        tmp_reduction_derivative_mean = 0_T;
        tmp_reduction_derivative_variance = 0_T;
            
        tmp_ptr_array_derivative_mean = ptr_array_derivatives_means_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
        tmp_ptr_array_derivative_variance = ptr_array_derivatives_variances_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));

        // TODO: Transpose optimization.
        for(tmp_thread_index__int = 1; tmp_thread_index__int != tmp_number_threads__int; ++tmp_thread_index__int)
        {
            tmp_reduction_derivative_mean += tmp_ptr_array_derivative_mean[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth];
            tmp_reduction_derivative_variance += tmp_ptr_array_derivative_variance[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth];
        }

        *tmp_ptr_array_derivative_mean += tmp_reduction_derivative_mean;
        *tmp_ptr_array_derivative_variance += tmp_reduction_derivative_variance;
    }
        
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

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

void Neural_Network::Backward_Pass__Batch_Normalization__OpenMP(size_t const time_step_index_received,
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
    int const tmp_batch_size__int(static_cast<int>(batch_size_received)),
                  tmp_units_size__int(static_cast<int>(input_size_received)),
                  tmp_number_threads__int(MyEA::Math::Minimum<int>(tmp_batch_size__int, static_cast<int>(this->number_threads)));
    int tmp_thread_index__int,
        tmp_example_index__int,
        tmp_input_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index,
              tmp_input_data_timed_index,
              tmp_input_thread_timed_index,
              tmp_input_thread_parameter_index;
    
    T_ const tmp_batch_scale(1_T / static_cast<T_>(batch_size_received));
    T_ *tmp_ptr_array_derivative_mean,
         *tmp_ptr_array_derivative_variance,
         tmp_reduction_derivative_mean,
         tmp_reduction_derivative_variance,
         tmp_error,
         tmp_variance_b;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_input_thread_timed_index = static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth + tmp_input_timed_index;
        tmp_input_thread_parameter_index = static_cast<size_t>(tmp_thread_index__int) * this->total_parameters_allocated;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error * ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index];
                
            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= ptr_array_scales_received[tmp_input_index];
                
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * ( -1_T / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) * ( -1_T / (tmp_variance_b * tmp_variance_b) );
                
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
        
    // Reduction.
    #pragma omp for schedule(static)
    for(tmp_input_index__int = 0; tmp_input_index__int < tmp_units_size__int; ++tmp_input_index__int)
    {
        tmp_reduction_derivative_mean = 0_T;
        tmp_reduction_derivative_variance = 0_T;
            
        tmp_ptr_array_derivative_mean = ptr_array_derivatives_means_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
        tmp_ptr_array_derivative_variance = ptr_array_derivatives_variances_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));

        for(tmp_thread_index__int = 1; tmp_thread_index__int != tmp_number_threads__int; ++tmp_thread_index__int)
        {
            tmp_reduction_derivative_mean += tmp_ptr_array_derivative_mean[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth];
            tmp_reduction_derivative_variance += tmp_ptr_array_derivative_variance[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth];
        }

        *tmp_ptr_array_derivative_mean += tmp_reduction_derivative_mean;
        *tmp_ptr_array_derivative_variance += tmp_reduction_derivative_variance;
    }
        
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

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

void Neural_Network::Backward_Pass__Batch_Renormalization__OpenMP(size_t const time_step_index_received,
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
    int const tmp_batch_size__int(static_cast<int>(batch_size_received)),
                  tmp_units_size__int(static_cast<int>(input_size_received)),
                  tmp_number_threads__int(MyEA::Math::Minimum<int>(tmp_batch_size__int, static_cast<int>(this->number_threads)));
    int tmp_thread_index__int,
        tmp_example_index__int,
        tmp_input_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index,
              tmp_input_data_timed_index,
              tmp_input_thread_timed_index,
              tmp_input_thread_parameter_index;
    
    T_ const tmp_batch_scale(1_T / static_cast<T_>(batch_size_received));
    T_ *tmp_ptr_array_derivative_mean,
         *tmp_ptr_array_derivative_variance,
         tmp_reduction_derivative_mean,
         tmp_reduction_derivative_variance,
         tmp_error,
         tmp_variance_b,
         tmp_negate_r_correction;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_input_thread_timed_index = static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth + tmp_input_timed_index;
        tmp_input_thread_parameter_index = static_cast<size_t>(tmp_thread_index__int) * this->total_parameters_allocated;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];
            tmp_negate_r_correction = -ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index]; // Negate.

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error * ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index];
                
            // Derivative shift.
            // dShift += dY
            ptr_array_derivatives_shifts_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error;

            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= ptr_array_scales_received[tmp_input_index];
                
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * ( tmp_negate_r_correction / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) * ( tmp_negate_r_correction / (tmp_variance_b * tmp_variance_b) );
                
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
        
    // Reduction.
    #pragma omp for schedule(static)
    for(tmp_input_index__int = 0; tmp_input_index__int < tmp_units_size__int; ++tmp_input_index__int)
    {
        tmp_reduction_derivative_mean = 0_T;
        tmp_reduction_derivative_variance = 0_T;
            
        tmp_ptr_array_derivative_mean = ptr_array_derivatives_means_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
        tmp_ptr_array_derivative_variance = ptr_array_derivatives_variances_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));

        for(tmp_thread_index__int = 1; tmp_thread_index__int != tmp_number_threads__int; ++tmp_thread_index__int)
        {
            tmp_reduction_derivative_mean += tmp_ptr_array_derivative_mean[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth];
            tmp_reduction_derivative_variance += tmp_ptr_array_derivative_variance[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth];
        }

        *tmp_ptr_array_derivative_mean += tmp_reduction_derivative_mean;
        *tmp_ptr_array_derivative_variance += tmp_reduction_derivative_variance;
    }
        
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

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

void Neural_Network::Backward_Pass__Batch_Renormalization__OpenMP(size_t const time_step_index_received,
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
    int const tmp_batch_size__int(static_cast<int>(batch_size_received)),
                  tmp_units_size__int(static_cast<int>(input_size_received)),
                  tmp_number_threads__int(MyEA::Math::Minimum<int>(tmp_batch_size__int, static_cast<int>(this->number_threads)));
    int tmp_thread_index__int,
        tmp_example_index__int,
        tmp_input_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index,
              tmp_input_data_timed_index,
              tmp_input_thread_timed_index,
              tmp_input_thread_parameter_index;
    
    T_ const tmp_batch_scale(1_T / static_cast<T_>(batch_size_received));
    T_ *tmp_ptr_array_derivative_mean,
         *tmp_ptr_array_derivative_variance,
         tmp_reduction_derivative_mean,
         tmp_reduction_derivative_variance,
         tmp_error,
         tmp_variance_b,
         tmp_negate_r_correction;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_input_thread_timed_index = static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth + tmp_input_timed_index;
        tmp_input_thread_parameter_index = static_cast<size_t>(tmp_thread_index__int) * this->total_parameters_allocated;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];
            tmp_negate_r_correction = -ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index]; // Negate.

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error * ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index];

            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= ptr_array_scales_received[tmp_input_index];
                
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * ( tmp_negate_r_correction / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) * ( tmp_negate_r_correction / (tmp_variance_b * tmp_variance_b) );
                
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
        
    // Reduction.
    #pragma omp for schedule(static)
    for(tmp_input_index__int = 0; tmp_input_index__int < tmp_units_size__int; ++tmp_input_index__int)
    {
        tmp_reduction_derivative_mean = 0_T;
        tmp_reduction_derivative_variance = 0_T;
            
        tmp_ptr_array_derivative_mean = ptr_array_derivatives_means_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
        tmp_ptr_array_derivative_variance = ptr_array_derivatives_variances_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));

        for(tmp_thread_index__int = 1; tmp_thread_index__int != tmp_number_threads__int; ++tmp_thread_index__int)
        {
            tmp_reduction_derivative_mean += tmp_ptr_array_derivative_mean[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth];
            tmp_reduction_derivative_variance += tmp_ptr_array_derivative_variance[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->number_recurrent_depth];
        }

        *tmp_ptr_array_derivative_mean += tmp_reduction_derivative_mean;
        *tmp_ptr_array_derivative_variance += tmp_reduction_derivative_variance;
    }
        
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

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
