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

#include <fstream>

#include <Files/File.hpp>
#include <Strings/String.hpp>

bool Neural_Network::Save_General_Parameters(std::string const &ref_path_received)
{
    if(MyEA::File::Write_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Write_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    std::ofstream tmp_ofstream(ref_path_received, std::ios::out | std::ios::binary);

    if(tmp_ofstream.is_open())
    {
        std::string tmp_value_to_write;

        tmp_value_to_write += "|===| GRADIENT DESCENT PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "learning_rate " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->learning_rate) + NEW_LINE;
        tmp_value_to_write += "learning_rate_final " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->learning_rate_final) + NEW_LINE;
        tmp_value_to_write += "learning_momentum " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->learning_momentum) + NEW_LINE;
        tmp_value_to_write += "learning_gamma " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->learning_gamma) + NEW_LINE;
        tmp_value_to_write += "use_Nesterov " + std::to_string(this->use_Nesterov) + NEW_LINE;
        tmp_value_to_write += "|END| GRADIENT DESCENT PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;

        tmp_value_to_write += "|===| QUICKPROP PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "quickprop_decay " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->quickprop_decay) + NEW_LINE;
        tmp_value_to_write += "quickprop_mu " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->quickprop_mu) + NEW_LINE;
        tmp_value_to_write += "|END| QUICKPROP PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;

        tmp_value_to_write += "|===| RESILLENT PROPAGATION PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "rprop_increase_factor " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->rprop_increase_factor) + NEW_LINE;
        tmp_value_to_write += "rprop_decrease_factor " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->rprop_decrease_factor) + NEW_LINE;
        tmp_value_to_write += "rprop_delta_min " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->rprop_delta_min) + NEW_LINE;
        tmp_value_to_write += "rprop_delta_max " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->rprop_delta_max) + NEW_LINE;
        tmp_value_to_write += "rprop_delta_zero " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->rprop_delta_zero) + NEW_LINE;
        tmp_value_to_write += "|END| RESILLENT PROPAGATION PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;

        tmp_value_to_write += "|===| SARPROP PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "sarprop_weight_decay_shift " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->sarprop_weight_decay_shift) + NEW_LINE;
        tmp_value_to_write += "sarprop_step_error_threshold_factor " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->sarprop_step_error_threshold_factor) + NEW_LINE;
        tmp_value_to_write += "sarprop_step_error_shift " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->sarprop_step_error_shift) + NEW_LINE;
        tmp_value_to_write += "sarprop_temperature " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->sarprop_temperature) + NEW_LINE;
        tmp_value_to_write += "sarprop_epoch " + std::to_string(this->sarprop_epoch) + NEW_LINE;
        tmp_value_to_write += "|END| SARPROP PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;

        tmp_value_to_write += "|===| ADAM PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "adam_learning_rate " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->adam_learning_rate) + NEW_LINE;
        tmp_value_to_write += "adam_beta1 " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->adam_beta1) + NEW_LINE;
        tmp_value_to_write += "adam_beta2 " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->adam_beta2) + NEW_LINE;
        tmp_value_to_write += "adam_epsilon " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->adam_epsilon) + NEW_LINE;
        tmp_value_to_write += "adam_bias_correction " + std::to_string(this->use_adam_bias_correction) + NEW_LINE;
        tmp_value_to_write += "adam_gamma " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->adam_gamma) + NEW_LINE;
        tmp_value_to_write += "|END| ADAM PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;

        tmp_value_to_write += "|===| WARM RESTARTS PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "use_Warm_Restarts " + std::to_string(this->use_Warm_Restarts) + NEW_LINE;
        tmp_value_to_write += "warm_restarts_decay_learning_rate " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->warm_restarts_decay_learning_rate) + NEW_LINE;
        tmp_value_to_write += "warm_restarts_maximum_learning_rate " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->warm_restarts_initial_maximum_learning_rate) + NEW_LINE;
        tmp_value_to_write += "warm_restarts_minimum_learning_rate " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->warm_restarts_minimum_learning_rate) + NEW_LINE;
        tmp_value_to_write += "warm_restarts_initial_T_i " + std::to_string(static_cast<size_t>(this->warm_restarts_initial_T_i)) + NEW_LINE;
        tmp_value_to_write += "warm_restarts_multiplier " + std::to_string(static_cast<size_t>(this->warm_restarts_multiplier)) + NEW_LINE;
        tmp_value_to_write += "|END| WARM RESTARTS PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;

        tmp_value_to_write += "|===| TRAINING PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "type_optimizer_function " + std::to_string(this->type_optimizer_function) + NEW_LINE;
        tmp_value_to_write += "type_loss_function " + std::to_string(this->type_loss_function) + NEW_LINE;
        tmp_value_to_write += "type_accuracy_function " + std::to_string(this->type_accuracy_function) + NEW_LINE;
        tmp_value_to_write += "bit_fail_limit " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->bit_fail_limit) + NEW_LINE;
        tmp_value_to_write += "pre_training_level " + std::to_string(this->pre_training_level) + NEW_LINE;
        tmp_value_to_write += "use_clip_gradient " + std::to_string(this->use_clip_gradient) + NEW_LINE;
        tmp_value_to_write += "clip_gradient " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->clip_gradient) + NEW_LINE;
        tmp_value_to_write += "|END| TRAINING PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;

        tmp_value_to_write += "|===| REGULARIZATION PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "regularization__max_norm_constraints " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->regularization__max_norm_constraints) + NEW_LINE;
        tmp_value_to_write += "regularization__l1 " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->regularization__l1) + NEW_LINE;
        tmp_value_to_write += "regularization__l2 " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->regularization__l2) + NEW_LINE;
        tmp_value_to_write += "regularization__srip " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->regularization__srip) + NEW_LINE;
        tmp_value_to_write += "regularization__weight_decay " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->regularization__weight_decay) + NEW_LINE;
        tmp_value_to_write += "use_normalized_weight_decay " + std::to_string(this->use_normalized_weight_decay) + NEW_LINE;
        tmp_value_to_write += "|END| REGULARIZATION PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;
        
        tmp_value_to_write += "|===| NORMALIZATION PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "normalization_momentum_average " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->normalization_momentum_average) + NEW_LINE;
        tmp_value_to_write += "normalization_epsilon " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->normalization_epsilon) + NEW_LINE;
        tmp_value_to_write += "batch_renormalization_r_correction_maximum " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->batch_renormalization_r_correction_maximum) + NEW_LINE;
        tmp_value_to_write += "batch_renormalization_d_correction_maximum " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->batch_renormalization_d_correction_maximum) + NEW_LINE;
        tmp_value_to_write += "|END| NORMALIZATION PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;

        tmp_value_to_write += "|===| LOSS PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "loss_training " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->loss_training) + NEW_LINE;
        tmp_value_to_write += "loss_validating " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->loss_validating) + NEW_LINE;
        tmp_value_to_write += "loss_testing " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->loss_testing) + NEW_LINE;
        tmp_value_to_write += "|END| LOSS PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;

        tmp_value_to_write += "|===| ACCURANCY PARAMETERS |===|" NEW_LINE;
        tmp_value_to_write += "accuracy_variance " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->accuracy_variance) + NEW_LINE;
        tmp_value_to_write += "accuracy_training " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->accuracy_training) + NEW_LINE;
        tmp_value_to_write += "accuracy_validating " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->accuracy_validating) + NEW_LINE;
        tmp_value_to_write += "accuracy_testing " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->accuracy_testing) + NEW_LINE;
        tmp_value_to_write += "|END| ACCURANCY PARAMETERS |END|" NEW_LINE;
        tmp_value_to_write += NEW_LINE;

        tmp_value_to_write += "|===| COMPUTATION PARAMETERS |===|" NEW_LINE;
    #if defined(COMPILE_CUDA)
        if(this->use_CUDA)
        { tmp_value_to_write += "use_CUDA " + std::to_string(this->use_CUDA) + NEW_LINE; }
        else
    #endif
        { tmp_value_to_write += "use_CUDA 0" NEW_LINE; }
        tmp_value_to_write += "use_OpenMP " + std::to_string(this->use_OpenMP) + NEW_LINE;
        tmp_value_to_write += "percentage_maximum_thread_usage " + MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->percentage_maximum_thread_usage, 9u) + NEW_LINE;
        tmp_value_to_write += "maximum_batch_size " + std::to_string(this->maximum_batch_size) + NEW_LINE;
        tmp_value_to_write += "|END| COMPUTATION PARAMETERS |END|" NEW_LINE;

        tmp_ofstream.write(tmp_value_to_write.c_str(), static_cast<std::streamsize>(tmp_value_to_write.size()));
        
        if(tmp_ofstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"write(string, %zu)\" function. Logical error on i/o operation \"%s\". At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_value_to_write.size(),
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }

        tmp_ofstream.flush();

        if(tmp_ofstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"flush()\" function. Logical error on i/o operation \"%s\". At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }

        tmp_ofstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    if(MyEA::File::Delete_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Delete_Temporary_File(%s)\" function." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str());

        return(false);
    }

    return(true);
}

bool Neural_Network::Save_Dimension_Parameters(std::string const &ref_path_received)
{
    if(MyEA::File::Write_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Write_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    std::ofstream tmp_ofstream(ref_path_received, std::ios::out | std::ios::binary);

    if(tmp_ofstream.is_open())
    {
        std::string tmp_value_to_write;

        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1), // Subtract output layer.
                                   *tmp_ptr_previous_layer,
                                   *tmp_ptr_layer_it(this->ptr_array_layers);
        
    #if defined(COMPILE_CUDA)
        if(this->is_device_initialized
          &&
          this->is_update_from_device == false)
        {
            if(this->Copy__Parameters__Device_To_Host() == false)
            { PRINT_FORMAT("%s: ERROR: From \"Copy__Parameters__Device_To_Host\"." NEW_LINE, __FUNCTION__); }
            else if(this->Use__Normalization() && this->Copy__Batch_Normalization_Neurons__Device_To_Host() == false)
            { PRINT_FORMAT("%s: ERROR: From \"Copy__Batch_Normalization_Neurons__Device_To_Host\"." NEW_LINE, __FUNCTION__); }
        }
    #endif
        
        tmp_value_to_write = "|===| DIMENSION |===|" NEW_LINE;
        tmp_value_to_write += "type_network " + std::to_string(this->type_network) + NEW_LINE;
        tmp_value_to_write += "number_layers " + std::to_string(this->total_layers) + NEW_LINE;
        tmp_value_to_write += "number_recurrent_depth " + std::to_string(this->number_recurrent_depth) + NEW_LINE;
        tmp_value_to_write += "number_time_delays " + std::to_string(this->number_time_delays) + NEW_LINE;
        tmp_value_to_write += "use_first_layer_as_input " + std::to_string(this->use_first_layer_as_input) + NEW_LINE;
        tmp_value_to_write += "use_last_layer_as_output " + std::to_string(this->use_last_layer_as_output) + NEW_LINE;
        tmp_value_to_write += "total_basic_units " + std::to_string(this->total_basic_units) + NEW_LINE;
        tmp_value_to_write += "total_basic_indice_units " + std::to_string(this->total_basic_indice_units) + NEW_LINE;
        tmp_value_to_write += "total_neuron_units " + std::to_string(this->total_neuron_units) + NEW_LINE;
        tmp_value_to_write += "total_AF_units " + std::to_string(this->total_AF_units) + NEW_LINE;
        tmp_value_to_write += "total_AF_Ind_recurrent_units " + std::to_string(this->total_AF_Ind_recurrent_units) + NEW_LINE;
        tmp_value_to_write += "total_block_units " + std::to_string(this->total_block_units) + NEW_LINE;
        tmp_value_to_write += "total_cell_units " + std::to_string(this->total_cell_units) + NEW_LINE;
        tmp_value_to_write += "total_normalized_units " + std::to_string(this->total_normalized_units) + NEW_LINE;
        tmp_value_to_write += "total_parameters " + std::to_string(this->total_parameters) + NEW_LINE;
        tmp_value_to_write += "total_weights " + std::to_string(this->total_weights) + NEW_LINE;
        tmp_value_to_write += "total_bias " + std::to_string(this->total_bias) + NEW_LINE;
        
        auto tmp_Get__Dropout__Parameters([](struct Layer const *const ptr_layer_it_received, bool const is_hidden_layer_received = true) -> std::string
        {
            std::string tmp_string("");

            tmp_string += "    type_dropout " + std::to_string(ptr_layer_it_received->type_dropout) + NEW_LINE;
            
            if(is_hidden_layer_received) { tmp_string += "      use_coded_dropout " + std::to_string(ptr_layer_it_received->use_coded_dropout) + NEW_LINE; }
            
            switch(ptr_layer_it_received->type_dropout)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA:
                    tmp_string += "      dropout_values[0] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(1_T - ptr_layer_it_received->dropout_values[0u]) + NEW_LINE;
                    tmp_string += "      dropout_values[1] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_layer_it_received->dropout_values[1u]) + NEW_LINE;
                    tmp_string += "      dropout_values[2] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_layer_it_received->dropout_values[2u]) + NEW_LINE;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                    tmp_string += "      dropout_values[0] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_layer_it_received->dropout_values[0u] / (ptr_layer_it_received->dropout_values[0u] + 1_T)) + NEW_LINE;
                    tmp_string += "      dropout_values[1] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_layer_it_received->dropout_values[1u]) + NEW_LINE;
                    tmp_string += "      dropout_values[2] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_layer_it_received->dropout_values[2u]) + NEW_LINE;
                        break;
                default:
                    tmp_string += "      dropout_values[0] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_layer_it_received->dropout_values[0u]) + NEW_LINE;
                    tmp_string += "      dropout_values[1] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_layer_it_received->dropout_values[1u]) + NEW_LINE;
                    tmp_string += "      dropout_values[2] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_layer_it_received->dropout_values[2u]) + NEW_LINE;
                        break;
            }

            return(tmp_string);
        });

        // Input layer.
        tmp_value_to_write += "  Input layer:" NEW_LINE;
        tmp_value_to_write += "    type_layer " + std::to_string(tmp_ptr_layer_it->type_layer) + NEW_LINE;
        tmp_value_to_write += "    type_activation " + std::to_string(tmp_ptr_layer_it->type_activation) + NEW_LINE;
        tmp_value_to_write += tmp_Get__Dropout__Parameters(tmp_ptr_layer_it, false);
        tmp_value_to_write += "    number_inputs " + std::to_string(this->number_inputs) + NEW_LINE;
        // |END| Input layer. |END|

        // Hidden layer.
        for(++tmp_ptr_layer_it; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_value_to_write += "  Hidden layer " + std::to_string(static_cast<size_t>(tmp_ptr_layer_it - this->ptr_array_layers)) + ":" NEW_LINE;
            tmp_value_to_write += "    type_layer " + std::to_string(tmp_ptr_layer_it->type_layer) + NEW_LINE;
            tmp_value_to_write += "    use_bidirectional " + std::to_string(tmp_ptr_layer_it->use_bidirectional) + NEW_LINE;
            
            switch(tmp_ptr_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                    tmp_value_to_write += "    kernel_size " + std::to_string(tmp_ptr_layer_it->pooling_values[0u]) + NEW_LINE;
                    tmp_value_to_write += "    stride " + std::to_string(tmp_ptr_layer_it->pooling_values[1u]) + NEW_LINE;
                    tmp_value_to_write += "    padding " + std::to_string(tmp_ptr_layer_it->pooling_values[2u]) + NEW_LINE;
                    tmp_value_to_write += "    dilation " + std::to_string(tmp_ptr_layer_it->pooling_values[3u]) + NEW_LINE;
                    tmp_value_to_write += "    number_basic_units " + std::to_string(*tmp_ptr_layer_it->ptr_number_outputs) + NEW_LINE;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                    tmp_ptr_previous_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

                    tmp_value_to_write += "    type_activation " + std::to_string(tmp_ptr_layer_it->type_activation) + NEW_LINE;
                    tmp_value_to_write += tmp_Get__Dropout__Parameters(tmp_ptr_layer_it);
                    
                    // Normalization.
                    if(this->Information__Layer__Normalization(tmp_value_to_write, tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Normalization()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Normalization. |END|

                    tmp_value_to_write += "    use_tied_parameter " + std::to_string(tmp_ptr_layer_it->use_tied_parameter) + NEW_LINE;
                    tmp_value_to_write += "    k_sparsity " + std::to_string(tmp_ptr_layer_it->k_sparsity) + NEW_LINE;
                    tmp_value_to_write += "    alpha_sparsity " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_layer_it->alpha_sparsity) + NEW_LINE;
                    tmp_value_to_write += "    constraint_recurrent_weight_lower_bound " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_layer_it->constraint_recurrent_weight_lower_bound) + NEW_LINE;
                    tmp_value_to_write += "    constraint_recurrent_weight_upper_bound " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_layer_it->constraint_recurrent_weight_upper_bound) + NEW_LINE;
                    
                    // Neuron unit(s).
                    if(this->Information__Layer__FC(tmp_value_to_write,
                                                                   tmp_ptr_layer_it,
                                                                   tmp_ptr_previous_layer) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__FC()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Neuron unit(s). |END|
                    
                    // AF unit(s).
                    if(this->Information__Layer__AF(tmp_value_to_write, tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__AF()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| AF unit(s). |END|

                    // Bias parameter(s).
                    if(this->Information__Layer__Bias(tmp_value_to_write, tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Bias()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Bias parameter(s). |END|
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    tmp_ptr_previous_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

                    tmp_value_to_write += "    type_activation " + std::to_string(tmp_ptr_layer_it->type_activation) + NEW_LINE;
                    tmp_value_to_write += tmp_Get__Dropout__Parameters(tmp_ptr_layer_it);
                    
                    // Normalization.
                    if(this->Information__Layer__Normalization(tmp_value_to_write, tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Normalization()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Normalization. |END|

                    tmp_value_to_write += "    use_tied_parameter " + std::to_string(tmp_ptr_layer_it->use_tied_parameter) + NEW_LINE;
                    tmp_value_to_write += "    k_sparsity " + std::to_string(tmp_ptr_layer_it->k_sparsity) + NEW_LINE;
                    tmp_value_to_write += "    alpha_sparsity " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_layer_it->alpha_sparsity) + NEW_LINE;
                    tmp_value_to_write += "    constraint_recurrent_weight_lower_bound " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_layer_it->constraint_recurrent_weight_lower_bound) + NEW_LINE;
                    tmp_value_to_write += "    constraint_recurrent_weight_upper_bound " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_layer_it->constraint_recurrent_weight_upper_bound) + NEW_LINE;
                    
                    // Neuron unit(s).
                    if(this->Information__Layer__FC(tmp_value_to_write,
                                                                   tmp_ptr_layer_it,
                                                                   tmp_ptr_previous_layer) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__FC()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Neuron unit(s). |END|
                    
                    // AF Ind recurrent unit(s).
                    if(this->Information__Layer__AF_Ind_Recurrent(tmp_value_to_write, tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__AF_Ind_Recurrent()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| AF Ind recurrent unit(s). |END|

                    // Bias parameter(s).
                    if(this->Information__Layer__Bias(tmp_value_to_write, tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Bias()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Bias parameter(s). |END|
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    tmp_ptr_previous_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

                    tmp_value_to_write += "    type_activation " + std::to_string(tmp_ptr_layer_it->type_activation) + NEW_LINE;
                    tmp_value_to_write += tmp_Get__Dropout__Parameters(tmp_ptr_layer_it);
                    
                    // Normalization.
                    if(this->Information__Layer__Normalization(tmp_value_to_write, tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Normalization()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Normalization. |END|

                    tmp_value_to_write += "    use_tied_parameter " + std::to_string(tmp_ptr_layer_it->use_tied_parameter) + NEW_LINE;
                    tmp_value_to_write += "    k_sparsity " + std::to_string(tmp_ptr_layer_it->k_sparsity) + NEW_LINE;
                    tmp_value_to_write += "    alpha_sparsity " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_layer_it->alpha_sparsity) + NEW_LINE;
                    tmp_value_to_write += "    constraint_recurrent_weight_lower_bound " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_layer_it->constraint_recurrent_weight_lower_bound) + NEW_LINE;
                    tmp_value_to_write += "    constraint_recurrent_weight_upper_bound " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_layer_it->constraint_recurrent_weight_upper_bound) + NEW_LINE;
                    
                    // Blocks unit(s).
                    if(this->Information__Layer__LSTM(tmp_value_to_write,
                                                                        tmp_ptr_layer_it,
                                                                        tmp_ptr_previous_layer) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__LSTM()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Blocks unit(s). |END|
                    
                    // Bias parameter(s).
                    if(this->Information__Layer__Bias(tmp_value_to_write, tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Bias()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Bias parameter(s). |END|
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                    tmp_value_to_write += "    kernel_size " + std::to_string(tmp_ptr_layer_it->pooling_values[0u]) + NEW_LINE;
                    tmp_value_to_write += "    stride " + std::to_string(tmp_ptr_layer_it->pooling_values[1u]) + NEW_LINE;
                    tmp_value_to_write += "    padding " + std::to_string(tmp_ptr_layer_it->pooling_values[2u]) + NEW_LINE;
                    tmp_value_to_write += "    dilation " + std::to_string(tmp_ptr_layer_it->pooling_values[3u]) + NEW_LINE;
                    tmp_value_to_write += "    number_basic_indice_units " + std::to_string(*tmp_ptr_layer_it->ptr_number_outputs) + NEW_LINE;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    tmp_value_to_write += "    block_depth " + std::to_string(tmp_ptr_layer_it->block_depth) + NEW_LINE;
                    tmp_value_to_write += "    padding " + std::to_string(tmp_ptr_layer_it->pooling_values[2u]) + NEW_LINE;
                    tmp_value_to_write += tmp_Get__Dropout__Parameters(tmp_ptr_layer_it);
                    
                    // Normalization.
                    if(this->Information__Layer__Normalization(tmp_value_to_write, tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Normalization()\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Normalization. |END|

                    tmp_value_to_write += "    number_basic_units " + std::to_string(*tmp_ptr_layer_it->ptr_number_outputs) + NEW_LINE;
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_layer_it->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                        return(false);
            }
        }
        // |END| Hidden layer. |END|

        // Output layer.
        tmp_ptr_previous_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

        tmp_value_to_write += "  Output layer:" NEW_LINE;
        tmp_value_to_write += "    type_layer " + std::to_string(tmp_ptr_layer_it->type_layer) + NEW_LINE;
        tmp_value_to_write += "    type_activation " + std::to_string(tmp_ptr_layer_it->type_activation) + NEW_LINE;

        //  Neuron_unit.
        if(this->Information__Output_Layer(tmp_value_to_write,
                                                          tmp_ptr_layer_it,
                                                          tmp_ptr_previous_layer) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Output_Layer()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        //  |END| Neuron_unit. |END|
        
        //  AF unit(s).
        if(this->Information__Layer__AF(tmp_value_to_write, tmp_ptr_layer_it) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__AF()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        //  |END| AF unit(s). |END|
        
        //  Bias parameter(s).
        if(this->Information__Layer__Bias(tmp_value_to_write, tmp_ptr_layer_it) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Bias()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        //  |END| Bias parameter(s). |END|
        // |END| Output layer. |END|

        tmp_value_to_write += "|END| DIMENSION |END|" NEW_LINE;
        
        tmp_ofstream.write(tmp_value_to_write.c_str(), static_cast<std::streamsize>(tmp_value_to_write.size()));
        
        if(tmp_ofstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"write(string, %zu)\" function. Logical error on i/o operation \"%s\". At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_value_to_write.size(),
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }

        tmp_ofstream.flush();
        
        if(tmp_ofstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"flush()\" function. Logical error on i/o operation \"%s\". At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }

        tmp_ofstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    if(MyEA::File::Delete_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Delete_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::Information__Layer__Normalization(std::string &ref_output_received, struct Layer const *const ptr_layer_it_received)
{
    union Normalized_unit const *const tmp_ptr_layer_first_normalized_unit(ptr_layer_it_received->ptr_array_normalized_units),
                                              *const tmp_ptr_last_normalized_unit(ptr_layer_it_received->ptr_last_normalized_unit),
                                              *tmp_ptr_array_normalized_unit_it(tmp_ptr_layer_first_normalized_unit);
    
    size_t const tmp_number_normalized_units(static_cast<size_t>(tmp_ptr_last_normalized_unit - tmp_ptr_array_normalized_unit_it));

    ref_output_received += "    type_normalization " + std::to_string(ptr_layer_it_received->type_normalization) + NEW_LINE;

    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL: break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            ref_output_received += "      use_layer_normalization_before_activation " + std::to_string(ptr_layer_it_received->use_layer_normalization_before_activation) + NEW_LINE;
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str());
                return(false);
    }

    ref_output_received += "      number_normalized_units " + std::to_string(tmp_number_normalized_units) + NEW_LINE;
    
    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const tmp_type_layer_normalization(ptr_layer_it_received->type_normalization);

    for(; tmp_ptr_array_normalized_unit_it != tmp_ptr_last_normalized_unit; ++tmp_ptr_array_normalized_unit_it)
    {
        ref_output_received += "        NormU[" + std::to_string(static_cast<size_t>(tmp_ptr_array_normalized_unit_it - tmp_ptr_layer_first_normalized_unit)) + "]" NEW_LINE;
        
        if(this->Information__Normalized_Unit(tmp_number_normalized_units,
                                                               tmp_type_layer_normalization,
                                                               tmp_ptr_array_normalized_unit_it,
                                                               ref_output_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Normalized_Unit(%zu, ref, ref)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_number_normalized_units,
                                     __LINE__);
        }
    }

    return(true);
}

bool Neural_Network::Information__Normalized_Unit(size_t const number_units_received,
                                                                            enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_normalization_received,
                                                                            union Normalized_unit const *const ptr_normalized_unit_received,
                                                                            std::string &ref_output_received)
{
    size_t tmp_time_step_index,
              tmp_unit_timed_index;
    
    switch(type_normalization_received)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
            ref_output_received += "          scale " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(*ptr_normalized_unit_received->normalized_batch_units.ptr_scale) + NEW_LINE;
            ref_output_received += "          shift " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(*ptr_normalized_unit_received->normalized_batch_units.ptr_shift) + NEW_LINE;
            
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
            {
                tmp_unit_timed_index = number_units_received * tmp_time_step_index;

                ref_output_received += "          mean_average[" + std::to_string(tmp_time_step_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_normalized_unit_received->normalized_batch_units.ptr_mean_average[tmp_unit_timed_index]) + NEW_LINE;
                ref_output_received += "          variance_average[" + std::to_string(tmp_time_step_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_normalized_unit_received->normalized_batch_units.ptr_variance_average[tmp_unit_timed_index]) + NEW_LINE;
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer normalization type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     type_normalization_received,
                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[type_normalization_received].c_str());
                return(false);
    }

    return(true);
}

bool Neural_Network::Information__Layer__FC(std::string &ref_output_received,
                                                                    struct Layer const *const ptr_layer_it_received,
                                                                    struct Layer const *const ptr_previous_layer_received)
{
    struct Neuron_unit const *const tmp_ptr_layer_first_neuron(ptr_layer_it_received->ptr_array_neuron_units),
                                         *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit),
                                         *tmp_ptr_array_neuron_it;

    ref_output_received += "    number_neuron_units " + std::to_string(static_cast<size_t>(tmp_ptr_last_neuron_unit - tmp_ptr_layer_first_neuron)) + NEW_LINE;
    
    for(tmp_ptr_array_neuron_it = tmp_ptr_layer_first_neuron; tmp_ptr_array_neuron_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_array_neuron_it)
    {
        ref_output_received += "      Neuron_unit[" + std::to_string(static_cast<size_t>(tmp_ptr_array_neuron_it - tmp_ptr_layer_first_neuron)) + "]" NEW_LINE;
        ref_output_received += "        number_connections " + std::to_string(*tmp_ptr_array_neuron_it->ptr_number_connections) + NEW_LINE;
            
        if(*tmp_ptr_array_neuron_it->ptr_number_connections != 0_zu)
        {
            switch(ptr_previous_layer_received->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    this->Layer__Forward__Neuron_Information__Connection<struct Basic_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING>(ref_output_received,
                                                                                                                                                                                                                                                        tmp_ptr_array_neuron_it,
                                                                                                                                                                                                                                                        this->ptr_array_basic_units);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: 
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: 
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    this->Layer__Forward__Neuron_Information__Connection<struct Neuron_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED>(ref_output_received,
                                                                                                                                                                                                                                                          tmp_ptr_array_neuron_it,
                                                                                                                                                                                                                                                          this->ptr_array_neuron_units);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    this->Layer__Forward__Neuron_Information__Connection<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(ref_output_received,
                                                                                                                                                                                                                              tmp_ptr_array_neuron_it,
                                                                                                                                                                                                                              this->ptr_array_cell_units);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                    this->Layer__Forward__Neuron_Information__Connection<struct Basic_indice_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING>(ref_output_received,
                                                                                                                                                                                                                                                          tmp_ptr_array_neuron_it,
                                                                                                                                                                                                                                                          this->ptr_array_basic_indice_units);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_previous_layer_received->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_previous_layer_received->type_layer].c_str());
                        return(false);
            }
        }
    }

    return(true);
}

bool Neural_Network::Information__Output_Layer(std::string &ref_output_received,
                                                                       struct Layer const *const ptr_layer_it_received,
                                                                       struct Layer const *const ptr_previous_layer_received)
{
    struct Neuron_unit const *const tmp_ptr_layer_first_neuron(ptr_layer_it_received->ptr_array_neuron_units),
                                         *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit),
                                         *tmp_ptr_array_neuron_it;

    ref_output_received += "    number_outputs " + std::to_string(static_cast<size_t>(tmp_ptr_last_neuron_unit - tmp_ptr_layer_first_neuron)) + NEW_LINE;
    
    for(tmp_ptr_array_neuron_it = tmp_ptr_layer_first_neuron; tmp_ptr_array_neuron_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_array_neuron_it)
    {
        ref_output_received += "      Neuron_unit[" + std::to_string(static_cast<size_t>(tmp_ptr_array_neuron_it - tmp_ptr_layer_first_neuron)) + "]" NEW_LINE;
        ref_output_received += "        number_connections " + std::to_string(*tmp_ptr_array_neuron_it->ptr_number_connections) + NEW_LINE;
        
        if(*tmp_ptr_array_neuron_it->ptr_number_connections != 0_zu)
        {
            switch(ptr_previous_layer_received->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    this->Layer__Forward__Neuron_Information__Connection<struct Basic_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING>(ref_output_received,
                                                                                                                                                                                                                                                        tmp_ptr_array_neuron_it,
                                                                                                                                                                                                                                                        this->ptr_array_basic_units);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: 
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: 
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    this->Layer__Forward__Neuron_Information__Connection<struct Neuron_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED>(ref_output_received,
                                                                                                                                                                                                                                                          tmp_ptr_array_neuron_it,
                                                                                                                                                                                                                                                          this->ptr_array_neuron_units);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    this->Layer__Forward__Neuron_Information__Connection<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(ref_output_received,
                                                                                                                                                                                                                              tmp_ptr_array_neuron_it,
                                                                                                                                                                                                                              this->ptr_array_cell_units);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                    this->Layer__Forward__Neuron_Information__Connection<struct Basic_indice_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING>(ref_output_received,
                                                                                                                                                                                                                                                          tmp_ptr_array_neuron_it,
                                                                                                                                                                                                                                                          this->ptr_array_basic_indice_units);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             ptr_previous_layer_received->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_previous_layer_received->type_layer].c_str());
                        return(false);
            }
        }
    }
    
    return(true);
}

bool Neural_Network::Information__Layer__AF(std::string &ref_output_received, struct Layer const *const ptr_layer_it_received)
{
    struct AF_unit const *const tmp_ptr_layer_first_af(ptr_layer_it_received->ptr_array_AF_units),
                                  *const tmp_ptr_last_AF_unit(ptr_layer_it_received->ptr_last_AF_unit),
                                  *tmp_ptr_array_AF_it;

    ref_output_received += "    number_AF_units " + std::to_string(static_cast<size_t>(tmp_ptr_last_AF_unit - tmp_ptr_layer_first_af)) + NEW_LINE;
    
    for(tmp_ptr_array_AF_it = tmp_ptr_layer_first_af; tmp_ptr_array_AF_it != tmp_ptr_last_AF_unit; ++tmp_ptr_array_AF_it)
    {
        ref_output_received += "      AF[" + std::to_string(static_cast<size_t>(tmp_ptr_array_AF_it - tmp_ptr_layer_first_af)) + "]" NEW_LINE;
        ref_output_received += "        activation_steepness " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(*tmp_ptr_array_AF_it->ptr_activation_steepness) + NEW_LINE;
        ref_output_received += "        activation_function " + std::to_string(static_cast<size_t>(*tmp_ptr_array_AF_it->ptr_type_activation_function)) + NEW_LINE;
    }

    return(true);
}

bool Neural_Network::Information__Layer__AF_Ind_Recurrent(std::string &ref_output_received, struct Layer const *const ptr_layer_it_received)
{
    struct AF_Ind_recurrent_unit const *const *const tmp_ptr_array_ptr_connections_layer_AF_Ind_recurrent_units(reinterpret_cast<struct AF_Ind_recurrent_unit **>(this->ptr_array_ptr_connections)),
                                                       *const tmp_ptr_layer_ptr_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units),
                                                       *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit),
                                                       *tmp_ptr_array_AF_Ind_recurrent_it;

    ref_output_received += "    number_AF_Ind_recurrent_units " + std::to_string(static_cast<size_t>(tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_ptr_first_AF_Ind_recurrent_unit)) + NEW_LINE;
    
    for(tmp_ptr_array_AF_Ind_recurrent_it = tmp_ptr_layer_ptr_first_AF_Ind_recurrent_unit; tmp_ptr_array_AF_Ind_recurrent_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_array_AF_Ind_recurrent_it)
    {
        ref_output_received += "      AF_Ind_R[" + std::to_string(static_cast<size_t>(tmp_ptr_array_AF_Ind_recurrent_it - tmp_ptr_layer_ptr_first_AF_Ind_recurrent_unit)) + "]" NEW_LINE;
        ref_output_received += "        activation_steepness " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(*tmp_ptr_array_AF_Ind_recurrent_it->ptr_activation_steepness) + NEW_LINE;
        ref_output_received += "        activation_function " + std::to_string(static_cast<size_t>(*tmp_ptr_array_AF_Ind_recurrent_it->ptr_type_activation_function)) + NEW_LINE;
        ref_output_received += "        connected_to_AF_Ind_R " + std::to_string(tmp_ptr_array_ptr_connections_layer_AF_Ind_recurrent_units[*tmp_ptr_array_AF_Ind_recurrent_it->ptr_recurrent_connection_index] - this->ptr_array_AF_Ind_recurrent_units) + NEW_LINE;
        ref_output_received += "        weight[" + std::to_string(*tmp_ptr_array_AF_Ind_recurrent_it->ptr_recurrent_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->ptr_array_parameters[*tmp_ptr_array_AF_Ind_recurrent_it->ptr_recurrent_connection_index]) + NEW_LINE;
    }

    return(true);
}

bool Neural_Network::Information__Layer__Bias(std::string &ref_output_received, struct Layer const *const ptr_layer_it_received)
{
    size_t const tmp_number_connections(ptr_layer_it_received->last_bias_connection_index - ptr_layer_it_received->first_bias_connection_index);
    
    T_ const *const tmp_ptr_array_parameters(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index);

    ref_output_received += "    number_bias_parameters " + std::to_string(tmp_number_connections) + NEW_LINE;

    for(size_t tmp_connection_index(0_zu); tmp_connection_index != tmp_number_connections; ++tmp_connection_index)
    {
        ref_output_received += "      weight[" + std::to_string(ptr_layer_it_received->first_bias_connection_index + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }

    return(true);
}

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> void Neural_Network::Layer__Forward__Neuron_Information__Connection(std::string &ref_output_received,
                                                                                                                                                                                                                                 struct Neuron_unit const *const ptr_neuron_it_received,
                                                                                                                                                                                                                                 U const *const ptr_first_U_unit_received)
{
    size_t const tmp_number_connections(*ptr_neuron_it_received->ptr_number_connections);
    size_t tmp_connection_index;

    T_ const *const tmp_ptr_array_parameters(this->ptr_array_parameters + *ptr_neuron_it_received->ptr_first_connection_index);

    U **tmp_ptr_array_ptr_connections(reinterpret_cast<U **>(this->ptr_array_ptr_connections + *ptr_neuron_it_received->ptr_first_connection_index));

    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_connections; ++tmp_connection_index)
    {
        ref_output_received += "          " + MyEA::Common::ENUM_TYPE_LAYER_CONNECTION_NAME[E] + " " + std::to_string(tmp_ptr_array_ptr_connections[tmp_connection_index] - ptr_first_U_unit_received) + NEW_LINE;
        ref_output_received += "          weight[" + std::to_string(*ptr_neuron_it_received->ptr_first_connection_index + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }
}

bool Neural_Network::Information__Layer__LSTM(std::string &ref_output_received,
                                                                        struct Layer const *const ptr_layer_it_received,
                                                                        struct Layer const *const ptr_previous_layer_received)
{
    struct Block_unit const *const tmp_ptr_layer_first_block_unit(ptr_layer_it_received->ptr_array_block_units),
                                       *const tmp_ptr_layer_last_block_unit(ptr_layer_it_received->ptr_last_block_unit),
                                       *tmp_ptr_block_unit_it(tmp_ptr_layer_first_block_unit);
    
    size_t const tmp_number_block_units(static_cast<size_t>(tmp_ptr_layer_last_block_unit - tmp_ptr_block_unit_it)),
                       tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units));
    
    ref_output_received += "    number_block_units " + std::to_string(tmp_number_block_units) + NEW_LINE;
    ref_output_received += "    number_cell_units " + std::to_string(tmp_number_cell_units) + NEW_LINE;

    for(; tmp_ptr_block_unit_it != tmp_ptr_layer_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        ref_output_received += "      Block[" + std::to_string(static_cast<size_t>(tmp_ptr_block_unit_it - tmp_ptr_layer_first_block_unit)) + "]" NEW_LINE;
        ref_output_received += "        activation_function " + std::to_string(static_cast<size_t>(tmp_ptr_block_unit_it->activation_function_io)) + NEW_LINE;
        ref_output_received += "        activation_steepness 1" NEW_LINE;
        ref_output_received += "        number_connections " + std::to_string(tmp_ptr_block_unit_it->last_index_connection - tmp_ptr_block_unit_it->first_index_connection) + NEW_LINE;

        switch(ptr_previous_layer_received->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                this->Layer__LSTM_Information__Connection<struct Basic_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING>(ref_output_received,
                                                                                                                                                                                                                                    tmp_ptr_block_unit_it,
                                                                                                                                                                                                                                    this->ptr_array_basic_units);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: 
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: 
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Layer__LSTM_Information__Connection<struct Neuron_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED>(ref_output_received,
                                                                                                                                                                                                                                     tmp_ptr_block_unit_it,
                                                                                                                                                                                                                                     this->ptr_array_neuron_units);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                this->Layer__LSTM_Information__Connection<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(ref_output_received,
                                                                                                                                                                                                         tmp_ptr_block_unit_it,
                                                                                                                                                                                                         this->ptr_array_cell_units);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                this->Layer__LSTM_Information__Connection<struct Basic_indice_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING>(ref_output_received,
                                                                                                                                                                                                                                     tmp_ptr_block_unit_it,
                                                                                                                                                                                                                                     this->ptr_array_basic_indice_units);
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                            MyEA::Time::Date_Time_Now().c_str(),
                                            __FUNCTION__,
                                            ptr_previous_layer_received->type_layer,
                                            MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_previous_layer_received->type_layer].c_str());
                    return(false);
        }
    }

    return(true);
}

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> void Neural_Network::Layer__LSTM_Information__Connection(std::string &ref_output_received,
                                                                                                                                                                                                                struct Block_unit const *const ptr_block_unit_it_received,
                                                                                                                                                                                                                U const *const ptr_first_U_unit_received)
{
    size_t const tmp_number_peephole_connections(ptr_block_unit_it_received->last_index_peephole_input_gate - ptr_block_unit_it_received->first_index_peephole_input_gate),
                       tmp_number_inputs_connections(ptr_block_unit_it_received->last_index_feedforward_connection_input_gate - ptr_block_unit_it_received->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrents_connection(ptr_block_unit_it_received->last_index_recurrent_connection_input_gate - ptr_block_unit_it_received->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index;

    T_ const *tmp_ptr_array_parameters;

    U **tmp_ptr_array_ptr_connections(reinterpret_cast<U **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_feedforward_connection_input_gate));

    struct Cell_unit const *const *const tmp_ptr_array_ptr_connections_layer_cell_units(reinterpret_cast<struct Cell_unit **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_recurrent_connection_input_gate)),
                                    *const *const tmp_ptr_array_ptr_connections_block_cell_units(reinterpret_cast<struct Cell_unit **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_peephole_input_gate)),
                                    *const tmp_ptr_block_ptr_last_cell_unit(ptr_block_unit_it_received->ptr_last_cell_unit),
                                    *tmp_ptr_block_ptr_cell_unit_it(ptr_block_unit_it_received->ptr_array_cell_units);

    // [0] Cell input.
    for(; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        //    [1] Input, cell.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
        {
            ref_output_received += "          " + MyEA::Common::ENUM_TYPE_LAYER_CONNECTION_NAME[E] + " " + std::to_string(tmp_ptr_array_ptr_connections[tmp_connection_index] - ptr_first_U_unit_received) + NEW_LINE;
            ref_output_received += "          weight[" + std::to_string(tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
        }
        //    [1] |END| Input, cell. |END|

        //    [1] Recurrent, cell.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
        {
            ref_output_received += "          connected_to_cell " + std::to_string(tmp_ptr_array_ptr_connections_layer_cell_units[tmp_connection_index] - this->ptr_array_cell_units) + NEW_LINE;
            ref_output_received += "          weight[" + std::to_string(tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
        }
        //    [1] |END| Recurrent, cell. |END|
    }
    // [0] |END| Cell input. |END|

    // [0] Input, gates.
    //    [1] Input gate.
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_feedforward_connection_input_gate;

    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
    {
        ref_output_received += "          " + MyEA::Common::ENUM_TYPE_LAYER_CONNECTION_NAME[E] + " " + std::to_string(tmp_ptr_array_ptr_connections[tmp_connection_index] - ptr_first_U_unit_received) + NEW_LINE;
        ref_output_received += "          weight[" + std::to_string(ptr_block_unit_it_received->first_index_feedforward_connection_input_gate + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }
    //    [1] |END| Input gate. |END|
    
    //    [1] Forget gate.
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_feedforward_connection_forget_gate;

    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
    {
        ref_output_received += "          " + MyEA::Common::ENUM_TYPE_LAYER_CONNECTION_NAME[E] + " " + std::to_string(tmp_ptr_array_ptr_connections[tmp_connection_index] - ptr_first_U_unit_received) + NEW_LINE;
        ref_output_received += "          weight[" + std::to_string(ptr_block_unit_it_received->first_index_feedforward_connection_forget_gate + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }
    //    [1] |END| Forget gate. |END|

    //    [1] Output gate.
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_feedforward_connection_output_gate;

    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
    {
        ref_output_received += "          " + MyEA::Common::ENUM_TYPE_LAYER_CONNECTION_NAME[E] + " " + std::to_string(tmp_ptr_array_ptr_connections[tmp_connection_index] - ptr_first_U_unit_received) + NEW_LINE;
        ref_output_received += "          weight[" + std::to_string(ptr_block_unit_it_received->first_index_feedforward_connection_output_gate + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }
    //    [1] |END| Output gate. |END|
    // [0] |END| Input, gates. |END|
    
    // [0] Recurrent, gates.
    //    [1] Input gate.
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_recurrent_connection_input_gate;

    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
    {
        ref_output_received += "          connected_to_cell " + std::to_string(tmp_ptr_array_ptr_connections_layer_cell_units[tmp_connection_index] - this->ptr_array_cell_units) + NEW_LINE;
        ref_output_received += "          weight[" + std::to_string(ptr_block_unit_it_received->first_index_recurrent_connection_input_gate + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }
    //    [1] |END| Input gate. |END|

    //    [1] Forget gate.
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_recurrent_connection_forget_gate;

    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
    {
        ref_output_received += "          connected_to_cell " + std::to_string(tmp_ptr_array_ptr_connections_layer_cell_units[tmp_connection_index] - this->ptr_array_cell_units) + NEW_LINE;
        ref_output_received += "          weight[" + std::to_string(ptr_block_unit_it_received->first_index_recurrent_connection_forget_gate + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }
    //    [1] |END| Forget gate. |END|

    //    [1] Output gate.
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_recurrent_connection_output_gate;

    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
    {
        ref_output_received += "          connected_to_cell " + std::to_string(tmp_ptr_array_ptr_connections_layer_cell_units[tmp_connection_index] - this->ptr_array_cell_units) + NEW_LINE;
        ref_output_received += "          weight[" + std::to_string(ptr_block_unit_it_received->first_index_recurrent_connection_output_gate + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }
    //    [1] |END| Output gate. |END|
    // [0] |END| Recurrent, gates. |END|
    
#ifndef NO_PEEPHOLE
    // [0] Peepholes.
    //    [1] Input gate.
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_peephole_input_gate;

    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
    {
        ref_output_received += "          connected_to_cell " + std::to_string(tmp_ptr_array_ptr_connections_block_cell_units[tmp_connection_index] - this->ptr_array_cell_units) + NEW_LINE;
        ref_output_received += "          weight[" + std::to_string(ptr_block_unit_it_received->first_index_peephole_input_gate + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }
    //    [1] |END| Input gate. |END|

    //    [1] Forget gate.
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_peephole_forget_gate;

    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
    {
        ref_output_received += "          connected_to_cell " + std::to_string(tmp_ptr_array_ptr_connections_block_cell_units[tmp_connection_index] - this->ptr_array_cell_units) + NEW_LINE;
        ref_output_received += "          weight[" + std::to_string(ptr_block_unit_it_received->first_index_peephole_forget_gate + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }
    //    [1] |END| Forget gate. |END|

    //    [1] Output gate.
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_peephole_output_gate;

    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
    {
        ref_output_received += "          connected_to_cell " + std::to_string(tmp_ptr_array_ptr_connections_block_cell_units[tmp_connection_index] - this->ptr_array_cell_units) + NEW_LINE;
        ref_output_received += "          weight[" + std::to_string(ptr_block_unit_it_received->first_index_peephole_output_gate + tmp_connection_index) + "] " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_parameters[tmp_connection_index]) + NEW_LINE;
    }
    //    [1] |END| Output gate. |END|
    // [0] |END| Peepholes. |END|
#endif
}
