#include "stdafx.hpp"

#include <Math/Mathematic.hpp>

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::FF__Update_Derivative_Weight_Batch__OpenMP(size_t const batch_size_received,
                                                                                                        struct Layer *ptr_layer_it_received,
                                                                                                        struct Layer const *const ptr_layer_end_received)
{
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
                this->Update_Derivative_Weight__FC__OpenMP(0_zu,
                                                                                       batch_size_received,
                                                                                       *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                       tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                       ptr_layer_it_received);
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

void Neural_Network::FF__Update_Derivative_Weight_Batch__Pre_Training__OpenMP(size_t const batch_size_received)
{
    struct Layer const *tmp_ptr_previous_connected_layer;
    struct Layer *tmp_ptr_layer_it;
    
    // Coded level part.
    tmp_ptr_layer_it = this->ptr_array_layers + this->pre_training_level;
    tmp_ptr_previous_connected_layer = tmp_ptr_layer_it->previous_connected_layers[0u];
    
    switch(tmp_ptr_layer_it->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            this->Update_Derivative_Weight__FC__OpenMP(0_zu,
                                                                                   batch_size_received,
                                                                                   *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                   tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                   tmp_ptr_layer_it);
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
            this->Update_Derivative_Weight__FC__OpenMP(0_zu,
                                                                                   batch_size_received,
                                                                                   *tmp_ptr_previous_connected_layer->ptr_number_outputs,
                                                                                   tmp_ptr_previous_connected_layer->ptr_array_outputs,
                                                                                   tmp_ptr_layer_it);
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

void Neural_Network::Update_Derivative_Weight__FC__OpenMP(size_t const time_step_index_received,
                                                                                              size_t const batch_size_received,
                                                                                              size_t const input_size_received,
                                                                                              T_ const *const ptr_array_inputs_received,
                                                                                              struct Layer *const ptr_layer_it_received)
{
    struct Neuron_unit *const tmp_ptr_layer_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);

    struct AF_unit *const tmp_ptr_layer_first_AF_unit(ptr_layer_it_received->ptr_array_AF_units);
    struct AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_AF_size(static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_unit - tmp_ptr_layer_first_AF_unit) + static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_first_AF_Ind_recurrent_unit)),
                       tmp_output_size(static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));

    // Weights.
    this->Update_Derivative_Weight__FC__OpenMP(time_step_index_received,
                                                                          batch_size_received,
                                                                          input_size_received,
                                                                          tmp_output_size,
                                                                          ptr_layer_it_received->ptr_array_pre_summations,
                                                                          tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                                          this->ptr_array_derivatives_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index);

    // Bias.
    if(ptr_layer_it_received->Use__Bias())
    {
        this->Update_Derivative_Weight__Bias__OpenMP(time_step_index_received,
                                                                                batch_size_received,
                                                                                tmp_output_size,
                                                                                tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                                                this->ptr_array_derivatives_parameters + ptr_layer_it_received->first_bias_connection_index);
    }

    // Recurrent connection(s).
    if(time_step_index_received != 0_zu
      &&
      ptr_layer_it_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT
      &&
      tmp_AF_size != 0_zu)
    {
        this->Update_Derivative_Weight__FC_Ind_RNN__OpenMP(time_step_index_received,
                                                                                             batch_size_received,
                                                                                             tmp_AF_size,
                                                                                             tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                                             tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_dAFs,
                                                                                             this->ptr_array_derivatives_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Neural_Network::Update_Derivative_Weight__FC__OpenMP(size_t const time_step_index_received,
                                                                                              size_t const batch_size_received,
                                                                                              size_t const input_size_received,
                                                                                              size_t const derivative_size_received,
                                                                                              T_ const *const ptr_array_inputs_received,
                                                                                              T_ const *const ptr_array_derivative_inputs_received,
                                                                                              T_ *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_connection_index,
              tmp_derivative_index;
    
    T_ const *tmp_ptr_array_inputs,
                  *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives_parameters,
         tmp_error;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;

        tmp_ptr_array_derivatives_parameters = ptr_array_derivatives_received + static_cast<size_t>(omp_get_thread_num()) * this->total_parameters_allocated;
        
        for(tmp_derivative_index = 0_zu; tmp_derivative_index != derivative_size_received; ++tmp_derivative_index,
                                                                                                                              tmp_ptr_array_derivatives_parameters += input_size_received)
        {
            tmp_error = tmp_ptr_array_derivative_inputs[tmp_derivative_index];
            
            for(tmp_connection_index = 0_zu; tmp_connection_index != input_size_received; ++tmp_connection_index) { tmp_ptr_array_derivatives_parameters[tmp_connection_index] += tmp_error * tmp_ptr_array_inputs[tmp_connection_index]; }
        }
    }
}

void Neural_Network::Update_Derivative_Weight__Bias__OpenMP(size_t const time_step_index_received,
                                                                                                size_t const batch_size_received,
                                                                                                size_t const derivative_size_received,
                                                                                                T_ const *const ptr_array_derivative_inputs_received,
                                                                                                T_ *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    size_t const tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_unit_index;
    
    T_ const *tmp_ptr_array_derivative_inputs;
    T_ *tmp_ptr_array_derivatives_bias;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
        
        tmp_ptr_array_derivatives_bias = ptr_array_derivatives_received + static_cast<size_t>(omp_get_thread_num()) * this->total_parameters_allocated;
        
        for(tmp_unit_index = 0_zu; tmp_unit_index != derivative_size_received; ++tmp_unit_index) { tmp_ptr_array_derivatives_bias[tmp_unit_index] += tmp_ptr_array_derivative_inputs[tmp_unit_index]; }
    }
}
