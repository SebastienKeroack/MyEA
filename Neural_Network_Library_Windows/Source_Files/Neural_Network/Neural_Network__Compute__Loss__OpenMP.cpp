#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::FF__Compute__Loss__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received)
{
    struct Layer const *const tmp_ptr_output_layer(this->Get__Output_Layer());

    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int,
        tmp_thread_index__int;
    
    size_t const tmp_output_size(static_cast<size_t>(tmp_ptr_output_layer->ptr_last_AF_unit - tmp_ptr_output_layer->ptr_array_AF_units));
    size_t tmp_output_index,
              tmp_output_desired_index,
              tmp_output_predicted_index(0_zu),
              tmp_output_data_index;
    
    T_ const *tmp_ptr_array_desireds_outputs;
    T_ tmp_target_difference,
        tmp_predicted_value,
        tmp_predicted_maximum;
    
    struct AF_unit const *const tmp_ptr_last_AF_unit(tmp_ptr_output_layer->ptr_last_AF_unit); 
    struct AF_unit *tmp_ptr_AF_unit_it;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();
            
        // Desired output(s): If the NN is in pre-training mode use depending on the level, the dataset array or the hidden-input layer to the AE. Else take the dataset array.
        tmp_ptr_array_desireds_outputs = this->pre_training_level <= 1_zu ? ptr_array_desireds_outputs_received[tmp_example_index__int] : this->Get__Outputs(this->ptr_array_layers + (this->pre_training_level - 1_zu), static_cast<size_t>(tmp_example_index__int));

        tmp_output_data_index = static_cast<size_t>(tmp_example_index__int) * tmp_output_size;

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
                if(tmp_ptr_AF_unit_it->ptr_array_values[tmp_output_data_index + tmp_output_index] >= tmp_predicted_maximum)
                {
                    // Store the largest output index.
                    tmp_output_predicted_index = tmp_output_index;

                    // State the maximum predicted output.
                    tmp_predicted_maximum = tmp_ptr_AF_unit_it->ptr_array_values[tmp_output_data_index + tmp_output_index];
                }
            }

            // If the desired output index equal the largest predicted output index.
            this->ptr_array_accuracy_values[0u][tmp_thread_index__int] += static_cast<T_>(tmp_output_desired_index == tmp_output_predicted_index);
        }

        for(; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it,
                                                                            ++tmp_ptr_array_desireds_outputs)
        {
            tmp_predicted_value = tmp_ptr_AF_unit_it->ptr_array_values[tmp_output_data_index];

            tmp_target_difference = tmp_predicted_value - *tmp_ptr_array_desireds_outputs;
            
            this->Update_Error(tmp_ptr_AF_unit_it,
                                        tmp_predicted_value,
                                        *tmp_ptr_array_desireds_outputs,
                                        tmp_target_difference,
                                        tmp_thread_index__int);
            
            switch(this->type_accuracy_function)
            {
                case MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY: break;
                case MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_DISTANCE: if(MyEA::Math::Absolute<T_>(tmp_target_difference) <= this->accuracy_variance) { this->ptr_array_accuracy_values[0u][tmp_thread_index__int] += 1_T; } break;
                case MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_DIRECTIONAL: this->ptr_array_accuracy_values[0u][tmp_thread_index__int] += 1_T; break;
                case MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_R:
                    this->ptr_array_accuracy_values[0u][tmp_thread_index__int] += *tmp_ptr_array_desireds_outputs;
                    this->ptr_array_accuracy_values[1u][tmp_thread_index__int] += tmp_predicted_value;
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
                    { this->ptr_array_accuracy_values[0u][tmp_thread_index__int] += 1_T; }
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Accuracy type (%u) is not managed in the switch." NEW_LINE,
                                                MyEA::Time::Date_Time_Now().c_str(),
                                                __FUNCTION__,
                                                this->type_accuracy_function);
                        break;
            }
        }
    }
}
