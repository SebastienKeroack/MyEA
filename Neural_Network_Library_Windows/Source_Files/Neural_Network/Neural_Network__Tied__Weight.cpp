#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::Tied__Transpose__Weight(struct Layer *const ptr_layer_received)
{
    struct Layer *const tmp_ptr_mirror_layer_it(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1),
                      *tmp_ptr_next_layer_it(const_cast<struct Layer *>(tmp_ptr_mirror_layer_it->next_connected_layers[0u]));
    struct Layer const *const tmp_ptr_next_layer_end(tmp_ptr_mirror_layer_it->next_connected_layers[0u] + tmp_ptr_mirror_layer_it->next_connected_layers.size());

    // Recurrent tied weights.
    if(ptr_layer_received != tmp_ptr_mirror_layer_it)
    {
        switch(ptr_layer_received->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Tied__Transpose__Weight__FC_Ind_RNN(ptr_layer_received, tmp_ptr_mirror_layer_it); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         ptr_layer_received->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_received->type_layer].c_str());
                    break;
        }
    }

    // Forward tied weights.
    switch(ptr_layer_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            for(; tmp_ptr_next_layer_it != tmp_ptr_next_layer_end; ++tmp_ptr_next_layer_it)
            {
                switch(tmp_ptr_next_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->Tied__Transpose__Weight__FC(ptr_layer_received, tmp_ptr_next_layer_it); break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_next_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_next_layer_it->type_layer].c_str());
                            return;
                }
            }
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_received->type_layer].c_str());
                return;
    }
}

void Transpose(size_t const source_stride_received,
                      size_t const destination_stride_received,
                      T_ const *ptr_array_source_values_received,
                      T_ const *const ptr_array_source_value_end_received,
                      T_ *ptr_array_destination_values_received)
{
    size_t tmp_index;

    for(; ptr_array_source_values_received != ptr_array_source_value_end_received; ptr_array_source_values_received += source_stride_received,
                                                                                                                        ++ptr_array_destination_values_received)
    {
        for(tmp_index = 0_zu; tmp_index != source_stride_received; ++tmp_index) { ptr_array_destination_values_received[tmp_index * destination_stride_received] = ptr_array_source_values_received[tmp_index]; }
    }
}

void Neural_Network::Tied__Transpose__Weight__FC(struct Layer const *const ptr_coded_layer_it_received, struct Layer const *const ptr_mirror_layer_it_received)
{
    struct Neuron_unit const *const tmp_ptr_coded_layer_it_first_neuron(ptr_coded_layer_it_received->ptr_array_neuron_units),
                                         *const tmp_ptr_mirror_layer_it_first_neuron(ptr_mirror_layer_it_received->ptr_array_neuron_units);
    
    Transpose(*tmp_ptr_coded_layer_it_first_neuron->ptr_number_connections,
                    *tmp_ptr_mirror_layer_it_first_neuron->ptr_number_connections,
                    this->ptr_array_parameters + *tmp_ptr_coded_layer_it_first_neuron->ptr_first_connection_index,
                    this->ptr_array_parameters + *tmp_ptr_coded_layer_it_first_neuron->ptr_last_connection_index,
                    this->ptr_array_parameters + *tmp_ptr_mirror_layer_it_first_neuron->ptr_first_connection_index);
}

void Neural_Network::Tied__Transpose__Weight__FC_Ind_RNN(struct Layer const *const ptr_encoded_layer_it_received, struct Layer const *const ptr_mirror_layer_it_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_encoded_layer_it_first_AF_ind(ptr_encoded_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_number_units(static_cast<size_t>(ptr_encoded_layer_it_received->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_encoded_layer_it_first_AF_ind));
    
    MEMCPY(this->ptr_array_parameters + *ptr_mirror_layer_it_received->ptr_array_AF_Ind_recurrent_units->ptr_recurrent_connection_index,
                   this->ptr_array_parameters + *tmp_ptr_encoded_layer_it_first_AF_ind->ptr_recurrent_connection_index,
                   tmp_number_units * sizeof(T_));
}
