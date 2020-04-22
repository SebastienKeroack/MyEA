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

#include <Math/Math.hpp>

#include <Neural_Network/Neural_Network.hpp>

void Neural_Network::Forward_Pass(size_t const batch_size_received,
                                                     T_ const *const *const ptr_array_inputs_received,
                                                     long long int input_layer_index_received,
                                                     long long int output_layer_index_received)
{
    if(input_layer_index_received == -1ll) { input_layer_index_received = static_cast<long long int>(this->Get__Input_Layer() - this->ptr_array_layers); }

    if(output_layer_index_received == -1ll) { output_layer_index_received = static_cast<long long int>((this->Get__Output_Layer() + 1) - this->ptr_array_layers); }

    if(input_layer_index_received >= output_layer_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input layer index (%lld) can not be greater or equal to the output layer index(%lld). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_layer_index_received,
                                 output_layer_index_received,
                                 __LINE__);

        return;
    }

    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            if(this->pre_training_level != 0_zu && this->_initialized__weight)
            {
                this->Forward_Pass__Pre_Training(batch_size_received, ptr_array_inputs_received);

                break;
            }
        default:
            if(this->number_recurrent_depth > 1_zu)
            {
                if(this->use_OpenMP && this->is_OpenMP_initialized)
                {
                    this->RNN__Forward_Pass_Batch__OpenMP(batch_size_received,
                                                                                      ptr_array_inputs_received,
                                                                                      this->ptr_array_layers + input_layer_index_received,
                                                                                      this->ptr_array_layers + output_layer_index_received);
                }
                else
                {
                    this->RNN__Forward_Pass_Batch__Loop(batch_size_received,
                                                                                 ptr_array_inputs_received,
                                                                                 this->ptr_array_layers + input_layer_index_received,
                                                                                 this->ptr_array_layers + output_layer_index_received);
                }
            }
            else
            {
                if(this->use_OpenMP && this->is_OpenMP_initialized)
                {
                    this->FF__Forward_Pass_Batch__OpenMP(batch_size_received,
                                                                                    ptr_array_inputs_received,
                                                                                    this->ptr_array_layers + input_layer_index_received,
                                                                                    this->ptr_array_layers + output_layer_index_received);
                }
                else
                {
                    this->FF__Forward_Pass_Batch__Loop(batch_size_received,
                                                                              ptr_array_inputs_received,
                                                                              this->ptr_array_layers + input_layer_index_received,
                                                                              this->ptr_array_layers + output_layer_index_received);
                }
            }
                break;
    }
}

void Neural_Network::Forward_Pass__Pre_Training(size_t const batch_size_received, T_ const *const *const ptr_array_inputs_received)
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
        { this->RNN__Forward_Pass_Batch__Pre_Training__OpenMP(batch_size_received, ptr_array_inputs_received); }
        else
        { this->RNN__Forward_Pass_Batch__Pre_Training__Loop(batch_size_received, ptr_array_inputs_received); }
    }
    else
    {
        if(this->use_OpenMP && this->is_OpenMP_initialized)
        { this->FF__Forward_Pass_Batch__Pre_Training__OpenMP(batch_size_received, ptr_array_inputs_received); }
        else
        { this->FF__Forward_Pass_Batch__Pre_Training__Loop(batch_size_received, ptr_array_inputs_received); }
    }
}

void Neural_Network::FF__Forward_Pass_Batch__Loop(size_t const batch_size_received,
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
                        this->number_threads * this->total_normalized_units_allocated * sizeof(T_));
            // |END| Set all mean to zero. |END|

            // Set all variance to zero.
            MEMSET(this->ptr_array_normalized_batch_units_variances,
                        0,
                        this->number_threads * this->total_normalized_units_allocated * sizeof(T_));
            // |END| Set all variance to zero. |END|
        }
    }

    // Input layer.
    this->FF__Assign_Inputs__Loop(batch_size_received, ptr_array_inputs_received);
    // |END| Input layer. |END|
    
    // Loop through each layer and do a forward propagation.
    for(; tmp_ptr_layer_it != ptr_last_layer_received; ++tmp_ptr_layer_it)
    {
        tmp_ptr_previous_connected_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                this->Forward_Pass__Average_Pooling__Loop(0_zu,
                                                                                    batch_size_received,
                                                                                    *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                    tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                    tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                this->Forward_Pass__FC__Loop(0_zu,
                                                                batch_size_received,
                                                                *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                this->Forward_Pass__Max_Pooling__Loop(0_zu,
                                                                               batch_size_received,
                                                                               *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                               tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                               tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL: this->Forward_Pass__Residual__Loop(batch_size_received, tmp_ptr_layer_it); break;
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

void Neural_Network::FF__Forward_Pass_Batch__Pre_Training__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_inputs_received)
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
                           this->number_threads * this->total_normalized_units_allocated * sizeof(T_));
            // |END| Set all mean to zero. |END|

            // Set all variance to zero.
            MEMSET(this->ptr_array_normalized_batch_units_variances,
                           0,
                           this->number_threads * this->total_normalized_units_allocated * sizeof(T_));
            // |END| Set all variance to zero. |END|
        }
    }

    // Input layer.
    this->FF__Assign_Inputs__Pre_Training__Loop(batch_size_received, ptr_array_inputs_received);
    // |END| Input layer. |END|
    
    // Loop through each encoded layer and do a forward propagation.
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        tmp_ptr_previous_connected_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

        switch(tmp_ptr_layer_it->type_layer)
        {
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                this->Forward_Pass__Encode__FC__Loop(0_zu,
                                                                               batch_size_received,
                                                                               *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                               tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                               tmp_ptr_layer_it);
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

    // Code level part.
    tmp_ptr_previous_connected_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

    switch(tmp_ptr_layer_it->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            this->Forward_Pass__Code__FC__Loop(0_zu,
                                                                       batch_size_received,
                                                                       *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                       tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                       tmp_ptr_layer_it);
                    break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_layer_it->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                return;
    }
    // |END| Code level part. |END|

    // Decode level part.
    tmp_ptr_previous_connected_layer = tmp_ptr_layer_it;
    tmp_ptr_layer_it = this->ptr_last_layer - static_cast<size_t>(tmp_ptr_layer_it - this->ptr_array_layers);
    
    switch(tmp_ptr_layer_it->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            this->Forward_Pass__Decode__FC__Loop(0_zu,
                                                                           batch_size_received,
                                                                           *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                           tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                           tmp_ptr_layer_it);
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

void Neural_Network::Forward_Pass__Average_Pooling__Loop(size_t const time_step_index_received,
                                                                                           size_t const batch_size_received,
                                                                                           size_t const input_size_received,
                                                                                           T_ const *const ptr_array_inputs_received,
                                                                                           struct Layer *const ptr_layer_it_received)
{
    struct Basic_unit *const tmp_ptr_layer_first_basic_unit(ptr_layer_it_received->ptr_array_basic_units);
    
    this->Forward_Pass__Average_Pooling__Loop(time_step_index_received,
                                                                        batch_size_received,
                                                                        input_size_received,
                                                                        *ptr_layer_it_received->ptr_number_outputs,
                                                                        ptr_layer_it_received->pooling_values[0u],
                                                                        ptr_layer_it_received->pooling_values[1u],
                                                                        ptr_layer_it_received->pooling_values[2u],
                                                                        ptr_layer_it_received->pooling_values[3u],
                                                                        ptr_array_inputs_received,
                                                                        tmp_ptr_layer_first_basic_unit->ptr_array_values);
    
    ptr_layer_it_received->ptr_array_outputs = tmp_ptr_layer_first_basic_unit->ptr_array_values;
}

void Neural_Network::Forward_Pass__FC__Loop(size_t const time_step_index_received,
                                                                       size_t const batch_size_received,
                                                                       size_t const input_size_received,
                                                                       T_ const *const ptr_array_inputs_received,
                                                                       struct Layer *const ptr_layer_it_received)
{
    struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
    
    struct AF_unit *const tmp_ptr_layer_first_AF_unit(ptr_layer_it_received->ptr_array_AF_units);
    struct AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));

    T_ *tmp_ptr_array_inputs;

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // Weights.
        this->Forward_Pass__FC__Loop(time_step_index_received,
                                                        batch_size_received,
                                                        input_size_received,
                                                        tmp_output_size,
                                                        ptr_array_inputs_received,
                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                        tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(ptr_layer_it_received->Use__Bias())
        {
            this->Forward_Pass__Bias__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               tmp_output_size,
                                                               this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index,
                                                               tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }

        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;

        // Normalization before activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation)
        {
            switch(ptr_layer_it_received->type_normalization)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__Loop(time_step_index_received,
                                                                                                             batch_size_received,
                                                                                                             tmp_output_size,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(time_step_index_received,
                                                                                                                batch_size_received,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_normalization,
                                             MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__Loop(time_step_index_received,
                                                                           batch_size_received,
                                                                           tmp_output_size,
                                                                           this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                           tmp_ptr_array_inputs,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);
            
            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation == false)
        {
            switch(ptr_layer_it_received->type_normalization)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__Loop(time_step_index_received,
                                                                                                            batch_size_received,
                                                                                                            tmp_output_size,
                                                                                                            tmp_ptr_array_inputs,
                                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(time_step_index_received,
                                                                                                                batch_size_received,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_normalization,
                                             MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                        break;
            }
                
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }
    // Inference mode.
    else
    {
        // Weights.
        this->Forward_Pass__FC__Loop(time_step_index_received,
                                                        batch_size_received,
                                                        input_size_received,
                                                        tmp_output_size,
                                                        ptr_array_inputs_received,
                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                        tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(ptr_layer_it_received->Use__Bias())
        {
            this->Forward_Pass__Bias__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               tmp_output_size,
                                                               this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index,
                                                               tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
        // Normalization before activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation)
        {
            this->Forward_Pass__Batch_Normalization__Inference__Loop(time_step_index_received,
                                                                                                       batch_size_received,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__Loop(time_step_index_received,
                                                                           batch_size_received,
                                                                           tmp_output_size,
                                                                           this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                           tmp_ptr_array_inputs,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);
            
            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }
            
            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }
            
            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation == false)
        {
            this->Forward_Pass__Batch_Normalization__Inference__Loop(time_step_index_received,
                                                                                                       batch_size_received,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }
    
    // If the state of propagation is strictly at training.
    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // Dropout.
        switch(ptr_layer_it_received->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Training__Loop(ptr_layer_it_received->ptr_array__mask__dropout__bernoulli,
                                                                                                    time_step_index_received,
                                                                                                    batch_size_received,
                                                                                                    tmp_output_size,
                                                                                                    tmp_ptr_array_inputs);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(ptr_layer_it_received->ptr_array__mask__dropout__bernoulli,
                                                                                                      time_step_index_received,
                                                                                                      batch_size_received,
                                                                                                      tmp_output_size,
                                                                                                      ptr_layer_it_received->dropout_values[0u] == 0_T ? 0_T : 1_T / ptr_layer_it_received->dropout_values[0u],
                                                                                                      tmp_ptr_array_inputs);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                this->Forward_Pass__Dropout__Gaussian__Loop(time_step_index_received,
                                                                                       batch_size_received,
                                                                                       tmp_output_size,
                                                                                       ptr_layer_it_received->dropout_values[0u],
                                                                                       tmp_ptr_array_inputs);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                this->Forward_Pass__Dropout__Uout__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_output_size,
                                                                                ptr_layer_it_received->dropout_values[0u],
                                                                                tmp_ptr_array_inputs);
                    break;
            default: break;
        }

        // k-Sparse.
        if(ptr_layer_it_received->Use__K_Sparsity())
        {
            this->Sparse_K_Filter__Loop(time_step_index_received,
                                                       batch_size_received,
                                                       tmp_output_size,
                                                       ptr_layer_it_received->k_sparsity,
                                                       ptr_layer_it_received->ptr_array_k_sparse_activities,
                                                       tmp_ptr_array_inputs);
        }
    }
    // Inference mode.
    else
    {
        // Dropout.
        switch(ptr_layer_it_received->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(time_step_index_received,
                                                                                                        batch_size_received,
                                                                                                        tmp_output_size,
                                                                                                        ptr_layer_it_received->dropout_values[0u],
                                                                                                        tmp_ptr_array_inputs);
                    break;
            default: break;
        }

        // k-Sparse.
        if(ptr_layer_it_received->Use__K_Sparsity())
        {
            this->Sparse_K_Filter__Loop(time_step_index_received,
                                                       batch_size_received,
                                                       tmp_output_size,
                                                       static_cast<size_t>(ptr_layer_it_received->alpha_sparsity * static_cast<T_>(ptr_layer_it_received->k_sparsity)),
                                                       ptr_layer_it_received->ptr_array_k_sparse_activities,
                                                       tmp_ptr_array_inputs);
        }
    }
}

void Neural_Network::Forward_Pass__Encode__FC__Loop(size_t const time_step_index_received,
                                                                                      size_t const batch_size_received,
                                                                                      size_t const input_size_received,
                                                                                      T_ const *const ptr_array_inputs_received,
                                                                                      struct Layer *const ptr_layer_it_received)
{
    struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
    
    struct AF_unit *const tmp_ptr_layer_first_AF_unit(ptr_layer_it_received->ptr_array_AF_units);
    struct AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));
    
    T_ *tmp_ptr_array_inputs;

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // Weights.
    this->Forward_Pass__FC__Loop(time_step_index_received,
                                                    batch_size_received,
                                                    input_size_received,
                                                    tmp_output_size,
                                                    ptr_array_inputs_received,
                                                    this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                    tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
    // Bias.
    if(ptr_layer_it_received->Use__Bias())
    {
        this->Forward_Pass__Bias__Loop(time_step_index_received,
                                                            batch_size_received,
                                                            tmp_output_size,
                                                            this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index,
                                                            tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
    }
        
    // Store the new inputs (summation).
    tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
    // Normalization before activation.
    if(ptr_layer_it_received->Use__Normalization()
      &&
      ptr_layer_it_received->use_layer_normalization_before_activation)
    {
        this->Forward_Pass__Batch_Normalization__Inference__Loop(time_step_index_received,
                                                                                                    batch_size_received,
                                                                                                    tmp_output_size,
                                                                                                    tmp_ptr_array_inputs,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
        // Store the new inputs (value normalize).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
    }
        
    if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
    {
        // Recurrent activation function.
        this->Forward_Pass__FC_Ind_RNN__Loop(time_step_index_received,
                                                                        batch_size_received,
                                                                        tmp_output_size,
                                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                        tmp_ptr_array_inputs,
                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

        // Activation function.
        switch(ptr_layer_it_received->type_activation)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
            case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
            case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
            case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                        batch_size_received,
                                                                        tmp_output_size,
                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                        batch_size_received,
                                                                                        tmp_output_size,
                                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                            MyEA::Time::Date_Time_Now().c_str(),
                                            __FUNCTION__,
                                            ptr_layer_it_received->type_activation,
                                            MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                    break;
        }

        // Store the new inputs (value).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
    }
    else
    {
        // Activation function.
        switch(ptr_layer_it_received->type_activation)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
            case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
            case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
            case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                        batch_size_received,
                                                                        tmp_output_size,
                                                                        tmp_ptr_array_inputs,
                                                                        tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                        tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                        batch_size_received,
                                                                                        tmp_output_size,
                                                                                        tmp_ptr_array_inputs,
                                                                                        tmp_ptr_layer_first_AF_unit->ptr_array_values);
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                            MyEA::Time::Date_Time_Now().c_str(),
                                            __FUNCTION__,
                                            ptr_layer_it_received->type_activation,
                                            MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                    break;
        }

        // Store the new inputs (value).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
    }

    // Normalization after activation.
    if(ptr_layer_it_received->Use__Normalization()
      &&
      ptr_layer_it_received->use_layer_normalization_before_activation == false)
    {
        this->Forward_Pass__Batch_Normalization__Inference__Loop(time_step_index_received,
                                                                                                    batch_size_received,
                                                                                                    tmp_output_size,
                                                                                                    tmp_ptr_array_inputs,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
        
        // Store the new inputs (value normalize).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
    }
    
    // If the state of propagation is strictly at training && Input AE layer.
    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING
      &&
      ptr_layer_it_received == this->ptr_array_layers + (this->pre_training_level - 1_zu))
    {
        // Dropout.
        switch(ptr_layer_it_received->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Training__Loop(ptr_layer_it_received->ptr_array__mask__dropout__bernoulli,
                                                                                                      time_step_index_received,
                                                                                                      batch_size_received,
                                                                                                      tmp_output_size,
                                                                                                      tmp_ptr_array_inputs);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(ptr_layer_it_received->ptr_array__mask__dropout__bernoulli,
                                                                                                      time_step_index_received,
                                                                                                      batch_size_received,
                                                                                                      tmp_output_size,
                                                                                                      ptr_layer_it_received->dropout_values[0u] == 0_T ? 0_T : 1_T / ptr_layer_it_received->dropout_values[0u],
                                                                                                      tmp_ptr_array_inputs);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                this->Forward_Pass__Dropout__Gaussian__Loop(time_step_index_received,
                                                                                         batch_size_received,
                                                                                         tmp_output_size,
                                                                                         ptr_layer_it_received->dropout_values[0u],
                                                                                         tmp_ptr_array_inputs);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                this->Forward_Pass__Dropout__Uout__Loop(time_step_index_received,
                                                                                  batch_size_received,
                                                                                  tmp_output_size,
                                                                                  ptr_layer_it_received->dropout_values[0u],
                                                                                  tmp_ptr_array_inputs);
                    break;
            default: break;
        }
    }
    // Inference mode.
    else
    {
        // Dropout.
        switch(ptr_layer_it_received->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(time_step_index_received,
                                                                                                        batch_size_received,
                                                                                                        tmp_output_size,
                                                                                                        ptr_layer_it_received->dropout_values[0u],
                                                                                                        tmp_ptr_array_inputs);
                    break;
            default: break;
        }
    }

    // k-Sparse.
    if(ptr_layer_it_received->Use__K_Sparsity())
    {
        this->Sparse_K_Filter__Loop(time_step_index_received,
                                                   batch_size_received,
                                                   tmp_output_size,
                                                   static_cast<size_t>(ptr_layer_it_received->alpha_sparsity * static_cast<T_>(ptr_layer_it_received->k_sparsity)),
                                                   ptr_layer_it_received->ptr_array_k_sparse_activities,
                                                   tmp_ptr_array_inputs);
    }
}

void Neural_Network::Forward_Pass__Code__FC__Loop(size_t const time_step_index_received,
                                                                                  size_t const batch_size_received,
                                                                                  size_t const input_size_received,
                                                                                  T_ const *const ptr_array_inputs_received,
                                                                                  struct Layer *const ptr_layer_it_received)
{
    struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
    
    struct AF_unit *const tmp_ptr_layer_first_AF_unit(ptr_layer_it_received->ptr_array_AF_units);
    struct AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));
    
    T_ *tmp_ptr_array_inputs;

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // Weights.
        this->Forward_Pass__FC__Loop(time_step_index_received,
                                                        batch_size_received,
                                                        input_size_received,
                                                        tmp_output_size,
                                                        ptr_array_inputs_received,
                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                        tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(ptr_layer_it_received->Use__Bias())
        {
            this->Forward_Pass__Bias__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               tmp_output_size,
                                                               this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index,
                                                               tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
        // Normalization before activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation)
        {
            switch(ptr_layer_it_received->type_normalization)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__Loop(time_step_index_received,
                                                                                                             batch_size_received,
                                                                                                             tmp_output_size,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(time_step_index_received,
                                                                                                                batch_size_received,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_normalization,
                                             MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__Loop(time_step_index_received,
                                                                           batch_size_received,
                                                                           tmp_output_size,
                                                                           this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                           tmp_ptr_array_inputs,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation == false)
        {
            switch(ptr_layer_it_received->type_normalization)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__Loop(time_step_index_received,
                                                                                                             batch_size_received,
                                                                                                             tmp_output_size,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(time_step_index_received,
                                                                                                                batch_size_received,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_normalization,
                                             MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }
    // Inference mode.
    else
    {
        // Weights.
        this->Forward_Pass__FC__Loop(time_step_index_received,
                                                        batch_size_received,
                                                        input_size_received,
                                                        tmp_output_size,
                                                        ptr_array_inputs_received,
                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                        tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(ptr_layer_it_received->Use__Bias())
        {
            this->Forward_Pass__Bias__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               tmp_output_size,
                                                               this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index,
                                                               tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
        // Normalization before activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation)
        {
            this->Forward_Pass__Batch_Normalization__Inference__Loop(time_step_index_received,
                                                                                                       batch_size_received,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__Loop(time_step_index_received,
                                                                           batch_size_received,
                                                                           tmp_output_size,
                                                                           this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                           tmp_ptr_array_inputs,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation == false)
        {
            this->Forward_Pass__Batch_Normalization__Inference__Loop(time_step_index_received,
                                                                                                       batch_size_received,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }
    
    // If the state of propagation is strictly at training.
    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // Dropout.
        if(ptr_layer_it_received->Use__Coded_Dropout())
        {
            switch(ptr_layer_it_received->type_dropout)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                    this->Forward_Pass__Dropout__Bernoulli__Training__Loop(ptr_layer_it_received->ptr_array__mask__dropout__bernoulli,
                                                                                                          time_step_index_received,
                                                                                                          batch_size_received,
                                                                                                          tmp_output_size,
                                                                                                          tmp_ptr_array_inputs);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                    this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(ptr_layer_it_received->ptr_array__mask__dropout__bernoulli,
                                                                                                          time_step_index_received,
                                                                                                          batch_size_received,
                                                                                                          tmp_output_size,
                                                                                                          ptr_layer_it_received->dropout_values[0u] == 0_T ? 0_T : 1_T / ptr_layer_it_received->dropout_values[0u],
                                                                                                          tmp_ptr_array_inputs);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                    this->Forward_Pass__Dropout__Gaussian__Loop(time_step_index_received,
                                                                                             batch_size_received,
                                                                                             tmp_output_size,
                                                                                             ptr_layer_it_received->dropout_values[0u],
                                                                                             tmp_ptr_array_inputs);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                    this->Forward_Pass__Dropout__Uout__Loop(time_step_index_received,
                                                                                      batch_size_received,
                                                                                      tmp_output_size,
                                                                                      ptr_layer_it_received->dropout_values[0u],
                                                                                      tmp_ptr_array_inputs);
                        break;
                default: break;
            }
        }
        else
        {
            switch(ptr_layer_it_received->type_dropout)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                    this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(time_step_index_received,
                                                                                                            batch_size_received,
                                                                                                            tmp_output_size,
                                                                                                            ptr_layer_it_received->dropout_values[0u],
                                                                                                            tmp_ptr_array_inputs);
                        break;
                default: break;
            }
        }

        // k-Sparse.
        if(ptr_layer_it_received->Use__K_Sparsity())
        {
            this->Sparse_K_Filter__Loop(time_step_index_received,
                                                       batch_size_received,
                                                       tmp_output_size,
                                                       ptr_layer_it_received->k_sparsity,
                                                       ptr_layer_it_received->ptr_array_k_sparse_activities,
                                                       tmp_ptr_array_inputs);
        }
    }
    // Inference mode.
    else
    {
        // Dropout.
        switch(ptr_layer_it_received->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(time_step_index_received,
                                                                                                        batch_size_received,
                                                                                                        tmp_output_size,
                                                                                                        ptr_layer_it_received->dropout_values[0u],
                                                                                                        tmp_ptr_array_inputs);
                    break;
            default: break;
        }

        // k-Sparse.
        if(ptr_layer_it_received->Use__K_Sparsity())
        {
            this->Sparse_K_Filter__Loop(time_step_index_received,
                                                       batch_size_received,
                                                       tmp_output_size,
                                                       static_cast<size_t>(ptr_layer_it_received->alpha_sparsity * static_cast<T_>(ptr_layer_it_received->k_sparsity)),
                                                       ptr_layer_it_received->ptr_array_k_sparse_activities,
                                                       tmp_ptr_array_inputs);
        }
    }
}

void Neural_Network::Forward_Pass__Decode__FC__Loop(size_t const time_step_index_received,
                                                                                      size_t const batch_size_received,
                                                                                      size_t const input_size_received,
                                                                                      T_ const *const ptr_array_inputs_received,
                                                                                      struct Layer *const ptr_layer_it_received)
{
    struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
    
    struct AF_unit *const tmp_ptr_layer_first_AF_unit(ptr_layer_it_received->ptr_array_AF_units);
    struct AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));
    
    T_ *tmp_ptr_array_inputs;
    
    struct Layer const *const tmp_ptr_input_layer(this->ptr_array_layers + (this->pre_training_level - 1_zu));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // Weights.
        this->Forward_Pass__FC__Loop(time_step_index_received,
                                                        batch_size_received,
                                                        input_size_received,
                                                        tmp_output_size,
                                                        ptr_array_inputs_received,
                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                        tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(ptr_layer_it_received->Use__Bias())
        {
            this->Forward_Pass__Bias__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               tmp_output_size,
                                                               this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index,
                                                               tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
        // Normalization before activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation)
        {
            switch(ptr_layer_it_received->type_normalization)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__Loop(time_step_index_received,
                                                                                                             batch_size_received,
                                                                                                             tmp_output_size,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(time_step_index_received,
                                                                                                                batch_size_received,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_normalization,
                                             MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__Loop(time_step_index_received,
                                                                           batch_size_received,
                                                                           tmp_output_size,
                                                                           this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                           tmp_ptr_array_inputs,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation == false)
        {
            switch(ptr_layer_it_received->type_normalization)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__Loop(time_step_index_received,
                                                                                                             batch_size_received,
                                                                                                             tmp_output_size,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(time_step_index_received,
                                                                                                                batch_size_received,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_normalization,
                                             MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }
    // Inference mode.
    else
    {
        // Weights.
        this->Forward_Pass__FC__Loop(time_step_index_received,
                                                        batch_size_received,
                                                        input_size_received,
                                                        tmp_output_size,
                                                        ptr_array_inputs_received,
                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                        tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(ptr_layer_it_received->Use__Bias())
        {
            this->Forward_Pass__Bias__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               tmp_output_size,
                                                               this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index,
                                                               tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
        // Normalization before activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation)
        {
            this->Forward_Pass__Batch_Normalization__Inference__Loop(time_step_index_received,
                                                                                                       batch_size_received,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__Loop(time_step_index_received,
                                                                           batch_size_received,
                                                                           tmp_output_size,
                                                                           this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                           tmp_ptr_array_inputs,
                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(ptr_layer_it_received->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                          batch_size_received,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(ptr_layer_it_received->Use__Normalization()
          &&
          ptr_layer_it_received->use_layer_normalization_before_activation == false)
        {
            this->Forward_Pass__Batch_Normalization__Inference__Loop(time_step_index_received,
                                                                                                       batch_size_received,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }

    // Dropout.
    switch(ptr_layer_it_received->type_dropout)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
            this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(time_step_index_received,
                                                                                                    batch_size_received,
                                                                                                    tmp_output_size,
                                                                                                    ptr_layer_it_received->dropout_values[0u],
                                                                                                    tmp_ptr_array_inputs);
                break;
        default: break;
    }
}

void Neural_Network::Forward_Pass__LSTM__Loop(long long int const time_step_index_received,
                                                                            long long int const tmp_time_step_reverse_direction,
                                                                            long long int const tmp_time_step_start,
                                                                            size_t const batch_size_received,
                                                                            size_t const input_size_received,
                                                                            T_ const *const ptr_array_inputs_received,
                                                                            struct Layer *const ptr_layer_it_received)
{
    struct Block_unit *const tmp_ptr_layer_first_block_unit(ptr_layer_it_received->ptr_array_block_units);
    
    struct Cell_unit *const tmp_ptr_layer_first_cell_unit(ptr_layer_it_received->ptr_array_cell_units);
    
    size_t const tmp_number_block_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - tmp_ptr_layer_first_block_unit)),
                       tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - tmp_ptr_layer_first_cell_unit));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__Loop(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                ptr_layer_it_received);
            
        // Normalization.
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }

        // If the state of propagation is strictly at training.
        // Gates activation cell, input, forget and state.
        if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING
          &&
          ptr_layer_it_received->Use__Dropout__Zoneout())
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__Loop(time_step_index_received,
                                                                                                                    tmp_time_step_reverse_direction,
                                                                                                                    tmp_time_step_start,
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                                    ptr_layer_it_received);
        }
        else
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(time_step_index_received,
                                                                                                    tmp_time_step_reverse_direction,
                                                                                                    tmp_time_step_start,
                                                                                                    batch_size_received,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_number_cell_units,
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                    ptr_layer_it_received);
        }

        // Normalization.
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            default: break;
        }

        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                ptr_layer_it_received);
            
        // Normalization.
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
                }    
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }
        
        // If the state of propagation is strictly at training.
        // Gate activation, output.
        if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING
          &&
          ptr_layer_it_received->Use__Dropout__Zoneout())
        {
            this->Forward_Pass__LSTM__Output__Zoneout__Loop(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size_received,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received);
        }
        else
        {
            this->Forward_Pass__LSTM__Output__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                ptr_layer_it_received);
        }
    }
    // Inference mode.
    else
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__Loop(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                ptr_layer_it_received);

        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Block input, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                
            // Input gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Forget gate, input.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Block input, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Forget gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gates activation cell, input, forget and state.
        this->Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size_received,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received);

        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Cell state activate.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
        }
            
        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                ptr_layer_it_received);
            
        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Output gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Output gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gate activation, output.
        this->Forward_Pass__LSTM__Output__Loop(time_step_index_received,
                                                                            batch_size_received,
                                                                            tmp_number_block_units,
                                                                            tmp_number_cell_units,
                                                                            ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                            ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                            ptr_layer_it_received);
    }
}

void Neural_Network::Forward_Pass__Encode__LSTM__Loop(long long int const time_step_index_received,
                                                                                          long long int const tmp_time_step_reverse_direction,
                                                                                          long long int const tmp_time_step_start,
                                                                                          size_t const batch_size_received,
                                                                                          size_t const input_size_received,
                                                                                          T_ const *const ptr_array_inputs_received,
                                                                                          struct Layer *const ptr_layer_it_received)
{
    struct Block_unit *const tmp_ptr_layer_first_block_unit(ptr_layer_it_received->ptr_array_block_units);
    
    struct Cell_unit *const tmp_ptr_layer_first_cell_unit(ptr_layer_it_received->ptr_array_cell_units);
    
    size_t const tmp_number_block_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - tmp_ptr_layer_first_block_unit)),
                       tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - tmp_ptr_layer_first_cell_unit));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // Training mode && Input AE layer.
    if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING
      &&
      ptr_layer_it_received == this->ptr_array_layers + (this->pre_training_level - 1_zu))
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__Loop(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                ptr_layer_it_received);
            
        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Block input, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                
            // Input gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Forget gate, input.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Block input, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Forget gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
        
        // If the state of propagation is strictly at training.
        // Gates activation cell, input, forget and state.
        if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING
          &&
          ptr_layer_it_received->Use__Dropout__Zoneout())
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__Loop(time_step_index_received,
                                                                                                                    tmp_time_step_reverse_direction,
                                                                                                                    tmp_time_step_start,
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                                    ptr_layer_it_received);
        }
        else
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(time_step_index_received,
                                                                                                    tmp_time_step_reverse_direction,
                                                                                                    tmp_time_step_start,
                                                                                                    batch_size_received,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_number_cell_units,
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                    ptr_layer_it_received);
        }

        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Cell state activate.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
        }
        
        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                ptr_layer_it_received);
            
        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Output gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Output gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
        
        // If the state of propagation is strictly at training.
        // Gate activation, output.
        if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING
          &&
          ptr_layer_it_received->Use__Dropout__Zoneout())
        {
            this->Forward_Pass__LSTM__Output__Zoneout__Loop(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size_received,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received);
        }
        else
        {
            this->Forward_Pass__LSTM__Output__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                ptr_layer_it_received);
        }
    }
    // Inference mode.
    else
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__Loop(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                ptr_layer_it_received);

        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Block input, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                
            // Input gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Forget gate, input.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Block input, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Forget gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gates activation cell, input, forget and state.
        this->Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size_received,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received);

        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Cell state activate.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
        }
            
        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                ptr_layer_it_received);
            
        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Output gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Output gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gate activation, output.
        this->Forward_Pass__LSTM__Output__Loop(time_step_index_received,
                                                                            batch_size_received,
                                                                            tmp_number_block_units,
                                                                            tmp_number_cell_units,
                                                                            ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                            ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                            ptr_layer_it_received);
    }
}

void Neural_Network::Forward_Pass__Code__LSTM__Loop(long long int const time_step_index_received,
                                                                                       long long int const tmp_time_step_reverse_direction,
                                                                                       long long int const tmp_time_step_start,
                                                                                       size_t const batch_size_received,
                                                                                       size_t const input_size_received,
                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                       struct Layer *const ptr_layer_it_received)
{
    struct Block_unit *const tmp_ptr_layer_first_block_unit(ptr_layer_it_received->ptr_array_block_units);
    
    struct Cell_unit *const tmp_ptr_layer_first_cell_unit(ptr_layer_it_received->ptr_array_cell_units);
    
    size_t const tmp_number_block_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - tmp_ptr_layer_first_block_unit)),
                       tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - tmp_ptr_layer_first_cell_unit));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__Loop(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                ptr_layer_it_received);
            
        // Normalization.
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }
        
        // If the state of propagation is strictly at training.
        // Gates activation cell, input, forget and state.
        if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING
          &&
          ptr_layer_it_received->Use__Dropout__Zoneout() && ptr_layer_it_received->Use__Coded_Dropout())
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__Loop(time_step_index_received,
                                                                                                                    tmp_time_step_reverse_direction,
                                                                                                                    tmp_time_step_start,
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                                    ptr_layer_it_received);
        }
        else
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(time_step_index_received,
                                                                                                    tmp_time_step_reverse_direction,
                                                                                                    tmp_time_step_start,
                                                                                                    batch_size_received,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_number_cell_units,
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                    ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                    ptr_layer_it_received);
        }

        // Normalization.
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            default: break;
        }

        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                ptr_layer_it_received);
            
        // Normalization.
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
                }    
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }
        
        // If the state of propagation is strictly at training.
        // Gate activation, output.
        if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING
          &&
          ptr_layer_it_received->Use__Dropout__Zoneout() && ptr_layer_it_received->Use__Coded_Dropout())
        {
            this->Forward_Pass__LSTM__Output__Zoneout__Loop(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size_received,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received);
        }
        else
        {
            this->Forward_Pass__LSTM__Output__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                ptr_layer_it_received);
        }
    }
    // Inference mode.
    else
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__Loop(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                ptr_layer_it_received);

        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Block input, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                
            // Input gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Forget gate, input.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Block input, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Forget gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gates activation cell, input, forget and state.
        this->Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size_received,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received);

        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Cell state activate.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
        }
            
        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                ptr_layer_it_received);
            
        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Output gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Output gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gate activation, output.
        this->Forward_Pass__LSTM__Output__Loop(time_step_index_received,
                                                                            batch_size_received,
                                                                            tmp_number_block_units,
                                                                            tmp_number_cell_units,
                                                                            ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                            ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                            ptr_layer_it_received);
    }
}

void Neural_Network::Forward_Pass__Decode__LSTM__Loop(long long int const time_step_index_received,
                                                                                          long long int const tmp_time_step_reverse_direction,
                                                                                          long long int const tmp_time_step_start,
                                                                                          size_t const batch_size_received,
                                                                                          size_t const input_size_received,
                                                                                          T_ const *const ptr_array_inputs_received,
                                                                                          struct Layer *const ptr_layer_it_received)
{
    struct Block_unit *const tmp_ptr_layer_first_block_unit(ptr_layer_it_received->ptr_array_block_units);
    
    struct Cell_unit *const tmp_ptr_layer_first_cell_unit(ptr_layer_it_received->ptr_array_cell_units);
    
    size_t const tmp_number_block_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - tmp_ptr_layer_first_block_unit)),
                       tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - tmp_ptr_layer_first_cell_unit));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__Loop(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                ptr_layer_it_received);
            
        // Normalization.
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size_received,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }

        // Gates activation cell, input, forget and state.
        this->Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size_received,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received);

        // Normalization.
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            default: break;
        }

        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                ptr_layer_it_received);
            
        // Normalization.
        switch(ptr_layer_it_received->type_normalization)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
                }    
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size_received,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }

        // Gate activation, output.
        this->Forward_Pass__LSTM__Output__Loop(time_step_index_received,
                                                                            batch_size_received,
                                                                            tmp_number_block_units,
                                                                            tmp_number_cell_units,
                                                                            ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                            ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                            ptr_layer_it_received);
    }
    // Inference mode.
    else
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__Loop(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                ptr_layer_it_received);

        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Block input, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0u].normalized_batch_units.ptr_array_values_normalizes);
                
            // Input gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3u].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Forget gate, input.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Block input, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4u].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Forget gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6u].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gates activation cell, input, forget and state.
        this->Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size_received,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                ptr_layer_it_received->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                ptr_layer_it_received);

        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Cell state activate.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2u].normalized_batch_units.ptr_array_values_normalizes);
        }
            
        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__Loop(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                ptr_layer_it_received->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                ptr_layer_it_received);
            
        // Batch normalization.
        if(ptr_layer_it_received->Use__Normalization())
        {
            // Output gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size_received,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7u].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Output gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__Loop(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size_received,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8u].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gate activation, output.
        this->Forward_Pass__LSTM__Output__Loop(time_step_index_received,
                                                                            batch_size_received,
                                                                            tmp_number_block_units,
                                                                            tmp_number_cell_units,
                                                                            ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                            ptr_layer_it_received->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                            ptr_layer_it_received);
    }
}

void Neural_Network::Forward_Pass__Max_Pooling__Loop(size_t const time_step_index_received,
                                                                                      size_t const batch_size_received,
                                                                                      size_t const input_size_received,
                                                                                      T_ const *const ptr_array_inputs_received,
                                                                                      struct Layer *const ptr_layer_it_received)
{
    struct Basic_indice_unit *const tmp_ptr_layer_first_basic_indice_unit(ptr_layer_it_received->ptr_array_basic_indice_units);
    
    this->Forward_Pass__Max_Pooling__Loop(time_step_index_received,
                                                                   batch_size_received,
                                                                   input_size_received,
                                                                   *ptr_layer_it_received->ptr_number_outputs,
                                                                   ptr_layer_it_received->pooling_values[0u],
                                                                   ptr_layer_it_received->pooling_values[1u],
                                                                   ptr_layer_it_received->pooling_values[2u],
                                                                   ptr_layer_it_received->pooling_values[3u],
                                                                   tmp_ptr_layer_first_basic_indice_unit->ptr_array_indices,
                                                                   ptr_array_inputs_received,
                                                                   tmp_ptr_layer_first_basic_indice_unit->ptr_array_values);
    
    ptr_layer_it_received->ptr_array_outputs = tmp_ptr_layer_first_basic_indice_unit->ptr_array_values;
}

void Neural_Network::Forward_Pass__Residual__Loop(size_t const batch_size_received, struct Layer *&ptr_layer_it_received)
{
    T_ *tmp_ptr_array_inputs;
    
    struct Layer const *const tmp_ptr_end_block_layer(ptr_layer_it_received + ptr_layer_it_received->block_depth + 1),
                               *tmp_ptr_previous_connected_layer;
    struct Layer *const tmp_ptr_residual_layer(ptr_layer_it_received);
    
    union Normalized_unit *const tmp_ptr_residual_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // First block layer.
    this->Forward_Pass__Residual__Layer__Loop(true,
                                                                        batch_size_received,
                                                                        ++ptr_layer_it_received);
    // |END| First block layer. |END|

    // Remaining layer(s).
    for(++ptr_layer_it_received; ptr_layer_it_received != tmp_ptr_end_block_layer; ++ptr_layer_it_received)
    {
        this->Forward_Pass__Residual__Layer__Loop(false,
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
                    this->Forward_Pass__Batch_Normalization__Training__Loop(0_zu,
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
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(0_zu,
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
            this->Forward_Pass__Batch_Normalization__Inference__Loop(0_zu,
                                                                                                      batch_size_received,
                                                                                                      *ptr_layer_it_received->ptr_number_outputs,
                                                                                                      tmp_ptr_array_inputs,
                                                                                                      tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                      tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                      tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                      tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                      tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
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
            this->Forward_Pass__Dropout__ShakeDrop__Loop(0_zu,
                                                                                       batch_size_received,
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
            this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(0_zu,
                                                                                                    batch_size_received,
                                                                                                    *ptr_layer_it_received->ptr_number_outputs,
                                                                                                    1_T - tmp_ptr_residual_layer->dropout_values[0u],
                                                                                                    tmp_ptr_array_inputs);
        }
    }

    //  Zero-padded identity-mapping shortcut.
    this->Forward_Pass__Zero_Padded_Identity__Loop(0_zu,
                                                                               batch_size_received,
                                                                               *tmp_ptr_previous_connected_layer->ptr_number_outputs, // Shortcut.
                                                                               *ptr_layer_it_received->ptr_number_outputs, // Block, last layer.
                                                                               tmp_ptr_residual_layer->pooling_values[2u],
                                                                               tmp_ptr_previous_connected_layer->ptr_array_outputs, // Shortcut.
                                                                               tmp_ptr_array_inputs, // Block, last layer.
                                                                               tmp_ptr_residual_layer->ptr_array_basic_units->ptr_array_values);
    // |END| Shortcut. |END|
}

void Neural_Network::Forward_Pass__Residual__Layer__Loop(bool const is_block_input_layer_received,
                                                                                           size_t const batch_size_received,
                                                                                           struct Layer *&ptr_layer_it_received)
{
    struct Layer const *const tmp_ptr_previous_connected_layer(ptr_layer_it_received->previous_connected_layers[0u]);

    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            this->Forward_Pass__Average_Pooling__Loop(0_zu,
                                                                                batch_size_received,
                                                                                *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                ptr_layer_it_received);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            this->Forward_Pass__Residual__FC__Loop(is_block_input_layer_received,
                                                                            0_zu,
                                                                            batch_size_received,
                                                                            *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                            tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                            ptr_layer_it_received);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            this->Forward_Pass__Max_Pooling__Loop(0_zu,
                                                                           batch_size_received,
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

void Neural_Network::Forward_Pass__Residual__FC__Loop(bool const is_block_input_layer_received,
                                                                                       size_t const time_step_index_received,
                                                                                       size_t const batch_size_received,
                                                                                       size_t const input_size_received,
                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                       struct Layer *const ptr_layer_it_received)
{
    struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
    
    struct AF_unit *const tmp_ptr_layer_first_AF_unit(ptr_layer_it_received->ptr_array_AF_units);
    struct AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));

    T_ *tmp_ptr_array_inputs(const_cast<T_ *>(ptr_array_inputs_received));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        // Normalization before activation.
        if(ptr_layer_it_received->Use__Normalization())
        {
            switch(ptr_layer_it_received->type_normalization)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__Loop(time_step_index_received,
                                                                                                             batch_size_received,
                                                                                                             input_size_received,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__Loop(time_step_index_received,
                                                                                                                batch_size_received,
                                                                                                                input_size_received,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer normalization (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_layer_it_received->type_normalization,
                                             MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(is_block_input_layer_received == false)
        {
            if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
            {
                // Recurrent activation function.
                this->Forward_Pass__FC_Ind_RNN__Loop(time_step_index_received,
                                                                               batch_size_received,
                                                                               input_size_received,
                                                                               this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                               tmp_ptr_array_inputs,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

                // Activation function.
                switch(ptr_layer_it_received->type_activation)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                        this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                              batch_size_received,
                                                                              input_size_received,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                        this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                              batch_size_received,
                                                                                              input_size_received,
                                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 ptr_layer_it_received->type_activation,
                                                 MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                            break;
                }

                // Store the new inputs (value).
                tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
            }
            else
            {
                // Activation function.
                switch(ptr_layer_it_received->type_activation)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                        this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                              batch_size_received,
                                                                              input_size_received,
                                                                              tmp_ptr_array_inputs,
                                                                              tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                              tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                        this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                              batch_size_received,
                                                                                              input_size_received,
                                                                                              tmp_ptr_array_inputs,
                                                                                              tmp_ptr_layer_first_AF_unit->ptr_array_values);
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 ptr_layer_it_received->type_activation,
                                                 MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                            break;
                }

                // Store the new inputs (value).
                tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
            }
            
            // If the state of propagation is strictly at training.
            if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
            {
                // Dropout.
                switch(ptr_layer_it_received->type_dropout)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                        this->Forward_Pass__Dropout__Bernoulli__Training__Loop(ptr_layer_it_received->ptr_array__mask__dropout__bernoulli,
                                                                                                              time_step_index_received,
                                                                                                              batch_size_received,
                                                                                                              input_size_received,
                                                                                                              tmp_ptr_array_inputs);
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                        this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(ptr_layer_it_received->ptr_array__mask__dropout__bernoulli,
                                                                                                              time_step_index_received,
                                                                                                              batch_size_received,
                                                                                                              input_size_received,
                                                                                                              ptr_layer_it_received->dropout_values[0u] == 0_T ? 0_T : 1_T / ptr_layer_it_received->dropout_values[0u],
                                                                                                              tmp_ptr_array_inputs);
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                        this->Forward_Pass__Dropout__Gaussian__Loop(time_step_index_received,
                                                                                                 batch_size_received,
                                                                                                 input_size_received,
                                                                                                 ptr_layer_it_received->dropout_values[0u],
                                                                                                 tmp_ptr_array_inputs);
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                        this->Forward_Pass__Dropout__Uout__Loop(time_step_index_received,
                                                                                          batch_size_received,
                                                                                          input_size_received,
                                                                                          ptr_layer_it_received->dropout_values[0u],
                                                                                          tmp_ptr_array_inputs);
                            break;
                    default: break;
                }

                // k-Sparse.
                if(ptr_layer_it_received->Use__K_Sparsity())
                {
                    this->Sparse_K_Filter__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               input_size_received,
                                                               ptr_layer_it_received->k_sparsity,
                                                               ptr_layer_it_received->ptr_array_k_sparse_activities,
                                                               tmp_ptr_array_inputs);
                }
            }
            // Inference mode.
            else
            {
                // Dropout.
                switch(ptr_layer_it_received->type_dropout)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                        this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(time_step_index_received,
                                                                                                                batch_size_received,
                                                                                                                input_size_received,
                                                                                                                ptr_layer_it_received->dropout_values[0u],
                                                                                                                tmp_ptr_array_inputs);
                            break;
                    default: break;
                }

                // k-Sparse.
                if(ptr_layer_it_received->Use__K_Sparsity())
                {
                    this->Sparse_K_Filter__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               input_size_received,
                                                               static_cast<size_t>(ptr_layer_it_received->alpha_sparsity * static_cast<T_>(ptr_layer_it_received->k_sparsity)),
                                                               ptr_layer_it_received->ptr_array_k_sparse_activities,
                                                               tmp_ptr_array_inputs);
                }
            }
        }

        // Weights.
        this->Forward_Pass__FC__Loop(time_step_index_received,
                                                        batch_size_received,
                                                        input_size_received,
                                                        tmp_output_size,
                                                        tmp_ptr_array_inputs,
                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                        tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(ptr_layer_it_received->Use__Bias())
        {
            this->Forward_Pass__Bias__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               tmp_output_size,
                                                               this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index,
                                                               tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
    }
    // Inference mode.
    else
    {
        // Normalization before activation.
        if(ptr_layer_it_received->Use__Normalization())
        {
            this->Forward_Pass__Batch_Normalization__Inference__Loop(time_step_index_received,
                                                                                                       batch_size_received,
                                                                                                       input_size_received,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(is_block_input_layer_received == false)
        {
            if(ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
            {
                // Recurrent activation function.
                this->Forward_Pass__FC_Ind_RNN__Loop(time_step_index_received,
                                                                               batch_size_received,
                                                                               input_size_received,
                                                                               this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                               tmp_ptr_array_inputs,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

                // Activation function.
                switch(ptr_layer_it_received->type_activation)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                        this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                              batch_size_received,
                                                                              input_size_received,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                        this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                              batch_size_received,
                                                                                              input_size_received,
                                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 ptr_layer_it_received->type_activation,
                                                 MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                            break;
                }

                // Store the new inputs (value).
                tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
            }
            else
            {
                // Activation function.
                switch(ptr_layer_it_received->type_activation)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                        this->Forward_Pass__FC_AF__Loop(time_step_index_received,
                                                                              batch_size_received,
                                                                              input_size_received,
                                                                              tmp_ptr_array_inputs,
                                                                              tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                              tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: // Only output layer.
                        this->Forward_Pass__FC_AF__Softmax__Loop(time_step_index_received,
                                                                                              batch_size_received,
                                                                                              input_size_received,
                                                                                              tmp_ptr_array_inputs,
                                                                                              tmp_ptr_layer_first_AF_unit->ptr_array_values);
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer activation (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 ptr_layer_it_received->type_activation,
                                                 MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str());
                            break;
                }

                // Store the new inputs (value).
                tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
            }

            // Dropout.
            switch(ptr_layer_it_received->type_dropout)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                    this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(time_step_index_received,
                                                                                                            batch_size_received,
                                                                                                            input_size_received,
                                                                                                            ptr_layer_it_received->dropout_values[0u],
                                                                                                            tmp_ptr_array_inputs);
                        break;
                default: break;
            }

            // k-Sparse.
            if(ptr_layer_it_received->Use__K_Sparsity())
            {
                this->Sparse_K_Filter__Loop(time_step_index_received,
                                                           batch_size_received,
                                                           input_size_received,
                                                           static_cast<size_t>(ptr_layer_it_received->alpha_sparsity * static_cast<T_>(ptr_layer_it_received->k_sparsity)),
                                                           ptr_layer_it_received->ptr_array_k_sparse_activities,
                                                           tmp_ptr_array_inputs);
            }
        }

        // Weights.
        this->Forward_Pass__FC__Loop(time_step_index_received,
                                                        batch_size_received,
                                                        input_size_received,
                                                        tmp_output_size,
                                                        tmp_ptr_array_inputs,
                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                        tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(ptr_layer_it_received->Use__Bias())
        {
            this->Forward_Pass__Bias__Loop(time_step_index_received,
                                                               batch_size_received,
                                                               tmp_output_size,
                                                               this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index,
                                                               tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Forward_Pass__Average_Pooling__Loop(size_t const time_step_index_received,
                                                                                           size_t const batch_size_received,
                                                                                           size_t const input_size_received,
                                                                                           size_t const output_size_received,
                                                                                           size_t const kernel_size_received,
                                                                                           size_t const stride_received,
                                                                                           size_t const padding_received,
                                                                                           size_t const dilation_received,
                                                                                           T_ const *const ptr_array_inputs_received,
                                                                                           T_ *const ptr_array_outputs_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_output_timed_batched_index(this->batch_size * output_size_received * time_step_index_received),
                       tmp_input_padded_half(input_size_received + padding_received);
    size_t tmp_example_index,
              tmp_kernel_index,
              tmp_shift_index,
              tmp_output_index;
    
    T_ const tmp_scale(1_T / static_cast<T_>(kernel_size_received)),
                 *tmp_ptr_array_inputs;
    T_ *tmp_ptr_array_outputs,
         tmp_summation;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * output_size_received + tmp_output_timed_batched_index;
        
        for(tmp_output_index = 0_zu; tmp_output_index != output_size_received; ++tmp_output_index)
        {
            tmp_summation = 0_T;
            
            for(tmp_kernel_index = 0_zu; tmp_kernel_index != kernel_size_received; ++tmp_kernel_index)
            {
                tmp_shift_index = tmp_output_index * stride_received + tmp_kernel_index * dilation_received;

                if(tmp_shift_index < padding_received || tmp_shift_index >= tmp_input_padded_half) { continue; }

                tmp_summation += tmp_ptr_array_inputs[tmp_shift_index - padding_received];
            }
            
            tmp_ptr_array_outputs[tmp_output_index] = tmp_summation * tmp_scale;
        }
    }
}

void Neural_Network::Forward_Pass__Bias__Loop(size_t const time_step_index_received,
                                                                          size_t const batch_size_received,
                                                                          size_t const output_size_received,
                                                                          T_ const *const ptr_array_bias_received,
                                                                          T_ *const ptr_array_outputs_received)
{
    size_t const tmp_unit_timed_batched_index(this->batch_size * output_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_unit_index;
    
    T_ *tmp_ptr_array_layer_outputs;

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_layer_outputs = ptr_array_outputs_received + tmp_example_index * output_size_received + tmp_unit_timed_batched_index;
        
        for(tmp_unit_index = 0_zu; tmp_unit_index != output_size_received; ++tmp_unit_index) { tmp_ptr_array_layer_outputs[tmp_unit_index] += ptr_array_bias_received[tmp_unit_index]; }
    }
}

void Neural_Network::Forward_Pass__FC__Loop(size_t const time_step_index_received,
                                                                       size_t const batch_size_received,
                                                                       size_t const input_size_received,
                                                                       size_t const output_size_received,
                                                                       T_ const *const ptr_array_inputs_received,
                                                                       T_ const *const ptr_array_parameters_received,
                                                                       T_ *const ptr_array_outputs_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_output_timed_batched_index(this->batch_size * output_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_output_index;
    
    T_ const *tmp_ptr_array_inputs,
                  *tmp_ptr_array_parameters;
    T_ *tmp_ptr_array_outputs,
         tmp_summation;

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_parameters = ptr_array_parameters_received;

        tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * output_size_received + tmp_output_timed_batched_index;
        
        for(tmp_output_index = 0_zu; tmp_output_index != output_size_received; ++tmp_output_index,
                                                                                                                  tmp_ptr_array_parameters += input_size_received)
        {
            tmp_summation = 0_T;

            for(tmp_connection_index = 0_zu; tmp_connection_index != input_size_received; ++tmp_connection_index) { tmp_summation += tmp_ptr_array_inputs[tmp_connection_index] * tmp_ptr_array_parameters[tmp_connection_index]; }

            tmp_ptr_array_outputs[tmp_output_index] = tmp_summation;
        }
    }
}

void Neural_Network::Forward_Pass__Batch_Normalization__Inference__Loop(size_t const time_step_index_received,
                                                                                                                 size_t const batch_size_received,
                                                                                                                 size_t const input_size_received,
                                                                                                                 T_ const *const ptr_array_inputs_received,
                                                                                                                 T_ const *const ptr_array_scales_received,
                                                                                                                 T_ const *const ptr_array_shifts_received,
                                                                                                                 T_ const *const ptr_array_means_averages_received,
                                                                                                                 T_ const *const ptr_array_variances_averages_received,
                                                                                                                 T_ *const ptr_array_output_normalizes_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index,
              tmp_input_data_timed_index;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            // Normalize input, scale and shift.
            // value_normalize = scale * ( (summation - mean) / variance ) + shift
            ptr_array_output_normalizes_received[tmp_input_data_timed_index + tmp_input_index] = ptr_array_scales_received[tmp_input_index] * ( (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_averages_received[tmp_input_timed_index + tmp_input_index]) / ptr_array_variances_averages_received[tmp_input_timed_index + tmp_input_index] ) + ptr_array_shifts_received[tmp_input_index];
        }
    }
}

void Neural_Network::Forward_Pass__Batch_Normalization__Training__Loop(size_t const time_step_index_received,
                                                                                                                size_t const batch_size_received,
                                                                                                                size_t const input_size_received,
                                                                                                                T_ const *const ptr_array_inputs_received,
                                                                                                                T_ const *const ptr_array_scales_received,
                                                                                                                T_ const *const ptr_array_shifts_received,
                                                                                                                T_ *const ptr_array_means_received,
                                                                                                                T_ *const ptr_array_variances_received,
                                                                                                                T_ *const ptr_array_means_averages_received,
                                                                                                                T_ *const ptr_array_variances_averages_received,
                                                                                                                T_ *const ptr_array_output_hats_received,
                                                                                                                T_ *const ptr_array_output_normalizes_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index,
              tmp_input_data_timed_index;
    
    T_ const tmp_batch_scale(1_T / static_cast<T_>(batch_size_received)),
                  tmp_epsilon(this->normalization_epsilon);
    T_ tmp_summation;
    
    // Summation.
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_summation = ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index];

            // mean += summation
            ptr_array_means_received[tmp_input_timed_index + tmp_input_index] += tmp_summation;
            // variance += summation^2
            ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] += tmp_summation * tmp_summation;
        }
    }
    
    // Average.
    for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
    {
        // Average batch mean.
        // mean_b = sum(summation, N) / N
        ptr_array_means_received[tmp_input_timed_index + tmp_input_index] *= tmp_batch_scale;

        // Average exponentialy global mean.
        // mean += momentum * (mean_b - mean)
        ptr_array_means_averages_received[tmp_input_timed_index + tmp_input_index] += this->normalization_momentum_average * (ptr_array_means_received[tmp_input_timed_index + tmp_input_index] - ptr_array_means_averages_received[tmp_input_timed_index + tmp_input_index]); // Exponential moving average.
        
        // Average batch variance.
        // variance_b = sqrt( ((sum(summation^2, N) / N) - pow(mean_b, 2) + epsilon )
        ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] = sqrt(ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] * tmp_batch_scale - ptr_array_means_received[tmp_input_timed_index + tmp_input_index] * ptr_array_means_received[tmp_input_timed_index + tmp_input_index] + tmp_epsilon);
        
        // Average exponentialy global variance.
        // variance += momentum * (variance_b - variance)
        ptr_array_variances_averages_received[tmp_input_timed_index + tmp_input_index] += this->normalization_momentum_average * (ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] - ptr_array_variances_averages_received[tmp_input_timed_index + tmp_input_index]); // Exponential moving average.
    }

    // Activation function.
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_summation = ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index];

            // Normalize.
            // value_hat = (summation - mean_b) / variance_b
            ptr_array_output_hats_received[tmp_input_data_timed_index + tmp_input_index] = tmp_summation = (tmp_summation - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) / ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];
            
            // Scale and shift.
            // value_normalize = scale * value_hat + shift
            ptr_array_output_normalizes_received[tmp_input_data_timed_index + tmp_input_index] = ptr_array_scales_received[tmp_input_index] * tmp_summation + ptr_array_shifts_received[tmp_input_index];
        }
    }
}

void Neural_Network::Forward_Pass__Batch_Renormalization__Training__Loop(size_t const time_step_index_received,
                                                                                                                   size_t const batch_size_received,
                                                                                                                   size_t const input_size_received,
                                                                                                                   T_ const *const ptr_array_inputs_received,
                                                                                                                   T_ const *const ptr_array_scales_received,
                                                                                                                   T_ const *const ptr_array_shifts_received,
                                                                                                                   T_ *const ptr_array_means_received,
                                                                                                                   T_ *const ptr_array_variances_received,
                                                                                                                   T_ *const ptr_array_means_averages_received,
                                                                                                                   T_ *const ptr_array_variances_averages_received,
                                                                                                                   T_ *const ptr_array_r_corrections_received,
                                                                                                                   T_ *const ptr_array_d_corrections_received,
                                                                                                                   T_ *const ptr_array_output_hats_received,
                                                                                                                   T_ *const ptr_array_output_normalizes_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index,
              tmp_input_data_timed_index;
    
    T_ const tmp_batch_scale(1_T / static_cast<T_>(batch_size_received)),
                  tmp_r_correction_maximum(this->batch_renormalization_r_correction_maximum),
                  tmp_d_correction_maximum(this->batch_renormalization_d_correction_maximum),
                  tmp_epsilon(this->normalization_epsilon);
    T_ tmp_summation,
        tmp_gamma,
        tmp_r_correction,
        tmp_d_correction;
    
    // Summation.
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_summation = ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index];

            // mean += summation
            ptr_array_means_received[tmp_input_timed_index + tmp_input_index] += tmp_summation;
            // variance += summation^2
            ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] += tmp_summation * tmp_summation;
        }
    }
    
    // Average.
    for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
    {
        // Average batch mean.
        // mean_b = sum(summation, N) / N
        ptr_array_means_received[tmp_input_timed_index + tmp_input_index] *= tmp_batch_scale;

        // Average exponentialy global mean.
        // mean += momentum * (mean_b - mean)
        ptr_array_means_averages_received[tmp_input_timed_index + tmp_input_index] += this->normalization_momentum_average * (ptr_array_means_received[tmp_input_timed_index + tmp_input_index] - ptr_array_means_averages_received[tmp_input_timed_index + tmp_input_index]); // Exponential moving average.
        
        // Average batch variance.
        // variance_b = sqrt( ((sum(summation^2, N) / N) - mean_b^2 + epsilon )
        ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] = sqrt(ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] * tmp_batch_scale - ptr_array_means_received[tmp_input_timed_index + tmp_input_index] * ptr_array_means_received[tmp_input_timed_index + tmp_input_index] + tmp_epsilon);
        
        // Average exponentialy global variance.
        // variance += momentum * (variance_b - variance)
        ptr_array_variances_averages_received[tmp_input_timed_index + tmp_input_index] += this->normalization_momentum_average * (ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] - ptr_array_variances_averages_received[tmp_input_timed_index + tmp_input_index]); // Exponential moving average.

        // r correction.
        // value = variance_b / variance
        tmp_gamma = ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] / ptr_array_variances_averages_received[tmp_input_timed_index + tmp_input_index];
        // low = 1 / r_correction_max
        tmp_r_correction = 1_T / tmp_r_correction_maximum;
        // high = r_correction_max
        // r_correction = clip(value, low, high)
        ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index] = MyEA::Math::Clip<T_>(tmp_gamma, tmp_r_correction, tmp_r_correction_maximum);

        // d correction.
        // value = (mean_b - mean) / variance
        tmp_d_correction = (ptr_array_means_received[tmp_input_timed_index + tmp_input_index] - ptr_array_means_averages_received[tmp_input_timed_index + tmp_input_index]) / ptr_array_variances_averages_received[tmp_input_timed_index + tmp_input_index];
        // low = -d_correction_max
        // high = d_correction_max
        // d_correction = clip(value, low, high)
        ptr_array_d_corrections_received[tmp_input_timed_index + tmp_input_index] = MyEA::Math::Clip<T_>(tmp_d_correction, -tmp_d_correction_maximum, tmp_d_correction_maximum);
    }

    // Activation function.
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_input_data_timed_index = tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_summation = ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index];

            // Normalize.
            // value_hat = (summation - mean_b) / variance_b * r_correction + d_correction
            ptr_array_output_hats_received[tmp_input_data_timed_index + tmp_input_index] = tmp_summation = (tmp_summation - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) / ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] * ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index] + ptr_array_d_corrections_received[tmp_input_timed_index + tmp_input_index];
            
            // Scale and shift.
            // value_normalize = scale * value_hat + shift
            ptr_array_output_normalizes_received[tmp_input_data_timed_index + tmp_input_index] = ptr_array_scales_received[tmp_input_index] * tmp_summation + ptr_array_shifts_received[tmp_input_index];
        }
    }
}

void Neural_Network::Forward_Pass__FC_AF__Loop(size_t const time_step_index_received,
                                                                             size_t const batch_size_received,
                                                                             size_t const input_size_received,
                                                                             T_ const *const ptr_array_inputs_received,
                                                                             T_ *const ptr_array_outputs_received,
                                                                             enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *const ptr_array_type_activations_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_inputs;
    T_ *tmp_ptr_array_outputs;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            AF_FIRE(ptr_array_type_activations_received[tmp_input_index],
                          tmp_ptr_array_inputs[tmp_input_index],
                          tmp_ptr_array_outputs[tmp_input_index]);
        }
    }
}

void Neural_Network::Forward_Pass__FC_AF__Softmax__Loop(size_t const time_step_index_received,
                                                                                             size_t const batch_size_received,
                                                                                             size_t const input_size_received,
                                                                                             T_ const *const ptr_array_inputs_received,
                                                                                             T_ *const ptr_array_outputs_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ const *tmp_ptr_array_inputs;
    T_ *tmp_ptr_array_outputs,
        tmp_layer_maximum_summation,
        tmp_summation;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_summation = 0_T;

        tmp_layer_maximum_summation = -(std::numeric_limits<ST_>::max)();;
        
        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_layer_maximum_summation = MyEA::Math::Maximum<T_>(tmp_layer_maximum_summation, tmp_ptr_array_inputs[tmp_input_index]); }
        
        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_summation += tmp_ptr_array_outputs[tmp_input_index] = exp(tmp_ptr_array_inputs[tmp_input_index] - tmp_layer_maximum_summation); }

        tmp_summation = 1_T / tmp_summation;
        
        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_outputs[tmp_input_index] *= tmp_summation; }
    }
}

void Neural_Network::Forward_Pass__Dropout__Bernoulli__Inverted__Loop(bool const *const ptr_array__mask__dropout__bernoulli_received,
                                                                                                             size_t const time_step_index_received,
                                                                                                             size_t const batch_size_received,
                                                                                                             size_t const input_size_received,
                                                                                                             T_ const inverse_retention_probability_divided_received,
                                                                                                             T_ *const ptr_array_inputs_received)
{
    bool const *tmp_ptr_timed_mask_dropout_bernoulli;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index;

    T_ *tmp_ptr_array_inputs;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_timed_mask_dropout_bernoulli = ptr_array__mask__dropout__bernoulli_received + tmp_input_timed_index;

        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            if(tmp_ptr_timed_mask_dropout_bernoulli[tmp_input_index]) { tmp_ptr_array_inputs[tmp_input_index] *= inverse_retention_probability_divided_received; }
            else { tmp_ptr_array_inputs[tmp_input_index] = 0_T; }
        }
    }
}

void Neural_Network::Forward_Pass__Dropout__Bernoulli__Training__Loop(bool const *const ptr_array__mask__dropout__bernoulli_received,
                                                                                                             size_t const time_step_index_received,
                                                                                                             size_t const batch_size_received,
                                                                                                             size_t const input_size_received,
                                                                                                             T_ *const ptr_array_inputs_received)
{
    bool const *tmp_ptr_timed_mask_dropout_bernoulli;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ *tmp_ptr_array_inputs;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_timed_mask_dropout_bernoulli = ptr_array__mask__dropout__bernoulli_received + tmp_input_timed_index;
        
        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { if(tmp_ptr_timed_mask_dropout_bernoulli[tmp_input_index] == false) { tmp_ptr_array_inputs[tmp_input_index] = 0_T; } }
    }
}

void Neural_Network::Forward_Pass__Dropout__Bernoulli__Inference__Loop(size_t const time_step_index_received,
                                                                                                               size_t const batch_size_received,
                                                                                                               size_t const input_size_received,
                                                                                                               T_ const retention_probability_received,
                                                                                                               T_ *const ptr_array_inputs_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ *tmp_ptr_array_inputs;
    
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_inputs[tmp_input_index] *= retention_probability_received; }
    }
}

void Neural_Network::Forward_Pass__Dropout__Gaussian__Loop(size_t const time_step_index_received,
                                                                                                size_t const batch_size_received,
                                                                                                size_t const input_size_received,
                                                                                                T_ const variance_received,
                                                                                                T_ *const ptr_array_inputs_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ *tmp_ptr_array_inputs;
    
    this->ptr_array_Class_Generator_Real_Gaussian->Range(1_T, variance_received);

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_inputs[tmp_input_index] *= (*this->ptr_array_Class_Generator_Real_Gaussian)(); }
    }
}

void Neural_Network::Forward_Pass__Dropout__ShakeDrop__Loop(size_t const time_step_index_received,
                                                                                                  size_t const batch_size_received,
                                                                                                  size_t const input_size_received,
                                                                                                  bool *const ptr_array_mask_dopout_shakedrop_received,
                                                                                                  T_ const lower_bound_received,
                                                                                                  T_ const upper_bound_received,
                                                                                                  T_ const dropout_probability_received,
                                                                                                  T_ *const ptr_array_inputs_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index),
                       tmp_layer_timed_batched_index(this->batch_size * time_step_index_received);
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ *tmp_ptr_array_inputs;

    this->ptr_array_Class_Generator_Bernoulli_ShakeDrop->Probability(dropout_probability_received); 
    this->ptr_array_Class_Generator_Real_ShakeDrop->Range(lower_bound_received, upper_bound_received);

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        if((ptr_array_mask_dopout_shakedrop_received[tmp_layer_timed_batched_index + tmp_example_index] = (*this->ptr_array_Class_Generator_Bernoulli_ShakeDrop)()))
        {
            tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_inputs[tmp_input_index] *= (*this->ptr_array_Class_Generator_Real_ShakeDrop)(); }
        }
    }
}

void Neural_Network::Forward_Pass__Dropout__Uout__Loop(size_t const time_step_index_received,
                                                                                         size_t const batch_size_received,
                                                                                         size_t const input_size_received,
                                                                                         T_ const beta_received,
                                                                                         T_ *const ptr_array_inputs_received)
{
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_example_index,
              tmp_input_index;
    
    T_ *tmp_ptr_array_inputs;

    this->ptr_array_Class_Generator_Real_Uout->Range(-beta_received, beta_received);

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_zu; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_inputs[tmp_input_index] += tmp_ptr_array_inputs[tmp_input_index] * (*this->ptr_array_Class_Generator_Real_Uout)(); }
    }
}

void Neural_Network::Forward_Pass__Max_Pooling__Loop(size_t const time_step_index_received,
                                                                                      size_t const batch_size_received,
                                                                                      size_t const input_size_received,
                                                                                      size_t const output_size_received,
                                                                                      size_t const kernel_size_received,
                                                                                      size_t const stride_received,
                                                                                      size_t const padding_received,
                                                                                      size_t const dilation_received,
                                                                                      size_t *const ptr_array_indices_received,
                                                                                      T_ const *const ptr_array_inputs_received,
                                                                                      T_ *const ptr_array_outputs_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_output_timed_batched_index(this->batch_size * output_size_received * time_step_index_received),
                       tmp_input_padded_half(input_size_received + padding_received);
    size_t *tmp_ptr_array_indices,
              tmp_example_index,
              tmp_kernel_index,
              tmp_shift_index,
              tmp_indice,
              tmp_output_index;
    
    T_ const *tmp_ptr_array_inputs;
    T_ *tmp_ptr_array_outputs,
         tmp_max;

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_indices = ptr_array_indices_received + tmp_example_index * output_size_received + tmp_output_timed_batched_index;
        
        tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * output_size_received + tmp_output_timed_batched_index;
        
        for(tmp_output_index = 0_zu; tmp_output_index != output_size_received; ++tmp_output_index)
        {
            tmp_indice = 0;

            tmp_max = -(std::numeric_limits<ST_>::max)();;
            
            for(tmp_kernel_index = 0_zu; tmp_kernel_index != kernel_size_received; ++tmp_kernel_index)
            {
                tmp_shift_index = tmp_output_index * stride_received + tmp_kernel_index * dilation_received;

                if(tmp_shift_index < padding_received || tmp_shift_index >= tmp_input_padded_half)
                {
                    if(tmp_max < 0.0)
                    {
                        tmp_indice = tmp_shift_index;

                        tmp_max = 0.0;
                    }
                }
                else if(tmp_max < tmp_ptr_array_inputs[tmp_shift_index - padding_received])
                {
                    tmp_indice = tmp_shift_index;

                    tmp_max = tmp_ptr_array_inputs[tmp_shift_index - padding_received];
                }
            }
            
            tmp_ptr_array_indices[tmp_output_index] = tmp_indice;

            tmp_ptr_array_outputs[tmp_output_index] = tmp_max;
        }
    }
}

void Neural_Network::Forward_Pass__Zero_Padded_Identity__Loop(size_t const time_step_index_received,
                                                                                                  size_t const batch_size_received,
                                                                                                  size_t const A_unit_size_received,
                                                                                                  size_t const B_unit_size_received,
                                                                                                  size_t const padding_received,
                                                                                                  T_ const *const ptr_array_A_received,
                                                                                                  T_ const *const ptr_array_B_received,
                                                                                                  T_ *const ptr_array_outputs_received)
{
    size_t const tmp_A_unit_timed_index(A_unit_size_received * time_step_index_received),
                       tmp_B_unit_timed_index(B_unit_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_unit_index;
    
    T_ const *tmp_ptr_array_A_outputs,
                  *tmp_ptr_array_B_outputs;
    T_ *tmp_ptr_array_outputs;

    if(padding_received == 0_zu)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_A_outputs = ptr_array_A_received + tmp_example_index * A_unit_size_received + this->batch_size * tmp_A_unit_timed_index;
            tmp_ptr_array_B_outputs = ptr_array_B_received + tmp_example_index * B_unit_size_received + this->batch_size * tmp_B_unit_timed_index;
            tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * A_unit_size_received + this->batch_size * tmp_A_unit_timed_index;
            
        #if defined(COMPILE_DEBUG_PRINT)
            PRINT_FORMAT("FORWARD, TIME[%zu], DATA[%zu]" NEW_LINE, time_step_index_received, tmp_example_index);
            for(tmp_unit_index = 0_zu; tmp_unit_index != A_unit_size_received; ++tmp_unit_index) { PRINT_FORMAT("%+.2f ", tmp_ptr_array_B_outputs[tmp_unit_index]); }
            PRINT_FORMAT(NEW_LINE);
            for(tmp_unit_index = 0_zu; tmp_unit_index != A_unit_size_received; ++tmp_unit_index) { PRINT_FORMAT("%+.2f ", tmp_ptr_array_A_outputs[tmp_unit_index]); }
            PRINT_FORMAT(NEW_LINE "\t=" NEW_LINE);
        #endif

            for(tmp_unit_index = 0_zu; tmp_unit_index != A_unit_size_received; ++tmp_unit_index)
            {
                tmp_ptr_array_outputs[tmp_unit_index] = tmp_ptr_array_A_outputs[tmp_unit_index]
                                                                                    +
                                                                             tmp_ptr_array_B_outputs[tmp_unit_index];
            }
            
        #if defined(COMPILE_DEBUG_PRINT)
            for(tmp_unit_index = 0_zu; tmp_unit_index != A_unit_size_received; ++tmp_unit_index) { PRINT_FORMAT("%+.2f ", tmp_ptr_array_outputs[tmp_unit_index]); }
            PRINT_FORMAT(NEW_LINE);
        #endif
        }
    }
    else if(A_unit_size_received > B_unit_size_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_A_outputs = ptr_array_A_received + tmp_example_index * A_unit_size_received + this->batch_size * tmp_A_unit_timed_index;
            tmp_ptr_array_B_outputs = ptr_array_B_received + tmp_example_index * B_unit_size_received + this->batch_size * tmp_B_unit_timed_index;
            tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * A_unit_size_received + this->batch_size * tmp_A_unit_timed_index;
            
            for(tmp_unit_index = 0_zu; tmp_unit_index != A_unit_size_received; ++tmp_unit_index) { tmp_ptr_array_outputs[tmp_unit_index] = tmp_ptr_array_A_outputs[tmp_unit_index]; }

            for(tmp_unit_index = 0_zu; tmp_unit_index != B_unit_size_received; ++tmp_unit_index) { tmp_ptr_array_outputs[tmp_unit_index + padding_received] += tmp_ptr_array_B_outputs[tmp_unit_index]; }
        }
    }
    else // if(A_unit_size_received < B_unit_size_received)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_A_outputs = ptr_array_A_received + tmp_example_index * A_unit_size_received + this->batch_size * tmp_A_unit_timed_index;
            tmp_ptr_array_B_outputs = ptr_array_B_received + tmp_example_index * B_unit_size_received + this->batch_size * tmp_B_unit_timed_index;
            tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * B_unit_size_received + this->batch_size * tmp_B_unit_timed_index;
            
            for(tmp_unit_index = 0_zu; tmp_unit_index != B_unit_size_received; ++tmp_unit_index) { tmp_ptr_array_outputs[tmp_unit_index] = tmp_ptr_array_B_outputs[tmp_unit_index]; }

            for(tmp_unit_index = 0_zu; tmp_unit_index != A_unit_size_received; ++tmp_unit_index) { tmp_ptr_array_outputs[tmp_unit_index + padding_received] += tmp_ptr_array_A_outputs[tmp_unit_index]; }
        }
    }
}
