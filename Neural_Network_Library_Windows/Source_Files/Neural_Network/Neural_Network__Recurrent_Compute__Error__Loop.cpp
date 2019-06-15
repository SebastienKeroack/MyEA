#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

void Neural_Network::RNN__Compute__Error__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received)
{
    struct Layer const *const tmp_ptr_output_layer(this->Get__Output_Layer());

    size_t const tmp_output_size(static_cast<size_t>(tmp_ptr_output_layer->ptr_last_AF_unit - tmp_ptr_output_layer->ptr_array_AF_units));
    size_t tmp_output_index,
              tmp_output_desired_index,
              tmp_output_predicted_index(0_zu),
              tmp_output_data_timed_index,
              tmp_example_index,
              tmp_time_step_index;
    
    T_ const *tmp_ptr_array_desireds_outputs,
                  *tmp_ptr_array_previous_desireds_outputs,
                  *tmp_ptr_pre_activation_function_it;
    T_ tmp_target_difference,
        tmp_predicted_value,
        tmp_predicted_maximum;
    
    struct AF_unit const *const tmp_ptr_last_AF_unit(tmp_ptr_output_layer->ptr_last_AF_unit);
    struct AF_unit *tmp_ptr_AF_unit_it;

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        for(tmp_time_step_index = this->number_time_delays; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
        {
            // Desired output(s): If the NN is in pre-training mode use depending on the level, the dataset array or the hidden-input layer to the AE. Else take the dataset array.
            tmp_ptr_array_desireds_outputs = this->pre_training_level <= 1_zu ? ptr_array_desireds_outputs_received[tmp_example_index] + tmp_time_step_index * tmp_output_size : this->Get__Outputs(this->ptr_array_layers + (this->pre_training_level - 1_zu),
                                                                                                                                                                                                                                                                                                   tmp_example_index,
                                                                                                                                                                                                                                                                                                   tmp_time_step_index);
                
            // Previous desired output(s): If the NN is in pre-training mode use depending on the level, the dataset array or the hidden-input layer to the AE. Else take the dataset array.
            tmp_ptr_array_previous_desireds_outputs = this->pre_training_level <= 1_zu ? ptr_array_desireds_outputs_received[tmp_example_index] + (tmp_time_step_index - 1_zu) * tmp_output_size : this->Get__Outputs(this->ptr_array_layers + (this->pre_training_level - 1_zu),
                                                                                                                                                                                                                                                                                                                            tmp_example_index,
                                                                                                                                                                                                                                                                                                                            tmp_time_step_index - 1_zu);

            tmp_output_data_timed_index = tmp_example_index * tmp_output_size + this->batch_size * tmp_output_size * tmp_time_step_index;
            
            tmp_ptr_pre_activation_function_it = tmp_ptr_output_layer->ptr_array_pre_activation_functions + tmp_output_data_timed_index;

            tmp_ptr_AF_unit_it = tmp_ptr_output_layer->ptr_array_AF_units;
            
            if(this->type_accuracy_function == MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY)
            {
                // Loop through each desired output to find the target class.
                for(tmp_output_desired_index = 0_zu; tmp_output_desired_index != tmp_output_size; ++tmp_output_desired_index)
                { if(tmp_ptr_array_desireds_outputs[tmp_output_desired_index] == 1_T) { break; } }
                
                // Reset predicted maximum at -Inf.
                tmp_predicted_maximum = -(std::numeric_limits<ST_>::max)();

                // Loop through each predicted output to find the largest output.
                for(tmp_output_index = 0_zu; tmp_output_index != tmp_output_size; ++tmp_output_index)
                {
                    // If the oIdx is the largest output.
                    if(tmp_ptr_AF_unit_it->ptr_array_values[tmp_output_data_timed_index + tmp_output_index] >= tmp_predicted_maximum)
                    {
                        // Store the largest output index.
                        tmp_output_predicted_index = tmp_output_index;

                        // State the maximum predicted output.
                        tmp_predicted_maximum = tmp_ptr_AF_unit_it->ptr_array_values[tmp_output_data_timed_index + tmp_output_index];
                    }
                }

                // If the desired output index equal the largest predicted output index.
                this->ptr_array_accuracy_values[0u][0u] += static_cast<T_>(tmp_output_desired_index == tmp_output_predicted_index);
            }

            for(; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it,
                                                                                    ++tmp_ptr_array_desireds_outputs,
                                                                                    ++tmp_ptr_array_previous_desireds_outputs)
            {
                tmp_predicted_value = tmp_ptr_AF_unit_it->ptr_array_values[tmp_output_data_timed_index];
                
                /*
                PRINT_FORMAT("D[%zu], T[%zu], O[%zu]: %f / %f" NEW_LINE,
                                         tmp_example_index,
                                         tmp_time_step_index,
                                         static_cast<size_t>(tmp_ptr_AF_unit_it - tmp_ptr_output_layer->ptr_array_AF_units),
                                         Cast_T(tmp_ptr_output_layer->ptr_array_pre_activation_functions[tmp_output_data_timed_index + static_cast<size_t>(tmp_ptr_AF_unit_it - tmp_ptr_output_layer->ptr_array_AF_units)]),
                                         Cast_T(tmp_predicted_value));
                */

                tmp_target_difference = tmp_predicted_value - *tmp_ptr_array_desireds_outputs;

                this->Update_Error(tmp_ptr_AF_unit_it,
                                            tmp_predicted_value,
                                            *tmp_ptr_array_desireds_outputs,
                                            tmp_target_difference);
                
                switch(this->type_accuracy_function)
                {
                    case MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY: break;
                    case MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_DISTANCE: if(MyEA::Math::Absolute<T_>(tmp_target_difference) <= this->accuracy_variance) { this->ptr_array_accuracy_values[0u][0u] += 1_T; } break;
                    case MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_DIRECTIONAL:
                        if(tmp_time_step_index == 0_zu
                          ||
                          MyEA::Math::Sign<T_>(*tmp_ptr_array_desireds_outputs - *tmp_ptr_array_previous_desireds_outputs) == MyEA::Math::Sign<T_>(tmp_predicted_value - *tmp_ptr_array_previous_desireds_outputs))
                        { this->ptr_array_accuracy_values[0u][0u] += 1_T; }
                            break;
                    case MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_R:
                        this->ptr_array_accuracy_values[0u][0u] += *tmp_ptr_array_desireds_outputs;
                        this->ptr_array_accuracy_values[1u][0u] += tmp_predicted_value;
                            break;
                    case MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_SIGN:
                        if(MyEA::Math::Sign<T_>(*tmp_ptr_array_desireds_outputs) == MyEA::Math::Sign<T_>(tmp_predicted_value)
                          ||
                          (
                              MyEA::Math::Sign<T_>(*tmp_ptr_array_desireds_outputs) == 0_T
                              &&
                              tmp_predicted_value >= -FLT_EPSILON
                              &&
                              tmp_predicted_value <= FLT_EPSILON
                          ))
                        { this->ptr_array_accuracy_values[0u][0u] += 1_T; }
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Accuracy type (%u) is not managed in the switch." NEW_LINE,
                                                    MyEA::String::Get__Time().c_str(),
                                                    __FUNCTION__,
                                                    this->type_accuracy_function);
                            break;
                }
                
                tmp_ptr_AF_unit_it->ptr_array_errors[tmp_output_data_timed_index] = this->Activation_Function_Derive(*tmp_ptr_AF_unit_it->ptr_type_activation_function,
                                                                                                                                                                     *tmp_ptr_pre_activation_function_it,
                                                                                                                                                                     *tmp_ptr_AF_unit_it->ptr_activation_steepness,
                                                                                                                                                                     tmp_predicted_value) * tmp_target_difference;
            }
        }
    }

    // Copy derivative AFs to derivative neurons.
    MEMCPY(tmp_ptr_output_layer->ptr_array_neuron_units->ptr_array_errors + this->batch_size * tmp_output_size * this->number_time_delays,
                  tmp_ptr_output_layer->ptr_array_AF_units->ptr_array_errors + this->batch_size * tmp_output_size * this->number_time_delays,
                  this->batch_size * tmp_output_size * (this->number_recurrent_depth - this->number_time_delays) * sizeof(T_));
    // |END| Copy derivative AFs to derivative neurons. |END|
}
