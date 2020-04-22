#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <array>

void Neural_Network::Initialization__Identity(T_ const bias_received)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1);

    // Loop though each layer.
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        // If the current layer is a pooling/residual layer, continue.
        if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
          ||
          tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING
          ||
          tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { continue; }
        
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                this->Initialize__Identity(*tmp_ptr_layer_it->ptr_array_neuron_units->ptr_number_connections,
                                                   static_cast<size_t>(tmp_ptr_layer_it->ptr_last_neuron_unit - tmp_ptr_layer_it->ptr_array_neuron_units),
                                                   this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_first_connection_index);

                this->Initialize__Constant__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Initialize__Identity(*tmp_ptr_layer_it->ptr_array_neuron_units->ptr_number_connections,
                                                   static_cast<size_t>(tmp_ptr_layer_it->ptr_last_neuron_unit - tmp_ptr_layer_it->ptr_array_neuron_units),
                                                   this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_first_connection_index);

                this->Initialize__Uniform__AF_Ind_Recurrent(tmp_ptr_layer_it);

                this->Initialize__Constant__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                // TODO: Intialize orthogonal LSTM.
                PRINT_FORMAT("%s: %s: ERROR: TODO: Intialize identity LSTM." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__);
                //this->Initialize__Identity(tmp_ptr_layer_it);

                this->Initialize__Constant__LSTM__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Can not initialize weights in the layer %zu with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         static_cast<size_t>(tmp_ptr_layer_it - this->ptr_array_layers),
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                         __LINE__);
                    break;
        }
    }

    // Independently recurrent neural network.
    if(this->number_recurrent_depth > 1_zu
      &&
      this->number_time_delays + 1_zu == this->number_recurrent_depth)
    { this->Initialize__Uniform__AF_Ind_Recurrent__Long_Term_Memory(); }

    if(this->ptr_array_derivatives_parameters != nullptr) { this->Clear_Training_Arrays(); }

    if(this->Use__Normalization()) { this->Clear__Parameter__Normalized_Unit(); }

    this->_initialized__weight = true;
    this->_type_weights_initializer = MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_IDENTITY;
}

void Neural_Network::Initialize__Identity(size_t const rows_received,
                                                          size_t const columns_received,
                                                          T_ *const ptr_array_weights_received)
{
    MEMSET(ptr_array_weights_received,
                   0,
                   rows_received * columns_received * sizeof(T_));

    size_t const tmp_squares(MyEA::Math::Minimum<size_t>(rows_received, columns_received));
    size_t tmp_square;

    for(tmp_square = 0_zu; tmp_square != tmp_squares; ++tmp_square) { ptr_array_weights_received[tmp_square * columns_received + tmp_square] = 1_T; }
}
