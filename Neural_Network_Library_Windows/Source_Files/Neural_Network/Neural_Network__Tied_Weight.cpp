#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::Tied__Transpose__Weight(struct Layer *const ptr_layer_received)
{
    struct Layer const *const tmp_ptr_previous_layer_connected(ptr_layer_received->previous_connected_layers[0u]);
    struct Layer *const tmp_ptr_mirror_layer_it(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1);
    
    // Recurrent tied.
    switch(tmp_ptr_previous_layer_connected->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL: break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Tied__Transpose__Weight__FC_Ind_RNN(tmp_ptr_previous_layer_connected, tmp_ptr_mirror_layer_it); 
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_previous_layer_connected->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_layer_connected->type_layer].c_str());
                break;
    }

    // Forward tied.
    switch(ptr_layer_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->Tied__Transpose__Weight__FC(ptr_layer_received, tmp_ptr_mirror_layer_it); break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_received->type_layer].c_str());
                break;
    }
}

void Neural_Network::Tied__Transpose__Weight__FC(struct Layer const *const ptr_layer_it_received, struct Layer const *const ptr_mirror_layer_it_received)
{
    struct Neuron_unit const *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units),
                                         *const tmp_ptr_mirror_layer_it_first_neuron(ptr_mirror_layer_it_received->ptr_array_neuron_units);
    
    size_t const tmp_number_coded_connections(*tmp_ptr_layer_it_first_neuron->ptr_number_connections),
                       tmp_number_mirror_connections(*tmp_ptr_mirror_layer_it_first_neuron->ptr_number_connections);
    size_t tmp_connection_index;
    
    T_ const *const tmp_ptr_mirror_parameter_end(this->ptr_array_parameters + *tmp_ptr_mirror_layer_it_first_neuron->ptr_last_connection_index),
                  *tmp_ptr_array_coded_parameters(this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_connection_index);
    T_ *tmp_ptr_mirror_parameter_it(this->ptr_array_parameters + *tmp_ptr_mirror_layer_it_first_neuron->ptr_first_connection_index);

    for(; tmp_ptr_mirror_parameter_it != tmp_ptr_mirror_parameter_end; ++tmp_ptr_mirror_parameter_it)
    {
        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_coded_connections; ++tmp_connection_index) { tmp_ptr_mirror_parameter_it[tmp_connection_index * tmp_number_mirror_connections] = tmp_ptr_array_coded_parameters[tmp_connection_index]; }

        tmp_ptr_array_coded_parameters += tmp_number_coded_connections;
    }
}

void Neural_Network::Tied__Transpose__Weight__FC_Ind_RNN(struct Layer const *const ptr_layer_it_received, struct Layer const *const ptr_mirror_layer_it_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_it_first_AF_ind(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units),
                                        *const tmp_ptr_mirror_layer_it_first_AF_ind(ptr_mirror_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_number_units(static_cast<size_t>(ptr_mirror_layer_it_received->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_mirror_layer_it_first_AF_ind));
    
    T_ *tmp_ptr_mirror_parameter_it(this->ptr_array_parameters + *tmp_ptr_mirror_layer_it_first_AF_ind->ptr_recurrent_connection_index);
    T_ const *const tmp_ptr_mirror_parameter_end(tmp_ptr_mirror_parameter_it + tmp_number_units),
                  *tmp_ptr_parameter_it(this->ptr_array_parameters + *tmp_ptr_layer_it_first_AF_ind->ptr_recurrent_connection_index);
    
    for(; tmp_ptr_mirror_parameter_it != tmp_ptr_mirror_parameter_end; ++tmp_ptr_mirror_parameter_it,
                                                                                                      ++tmp_ptr_parameter_it)
    { *tmp_ptr_mirror_parameter_it = *tmp_ptr_parameter_it; }
}
