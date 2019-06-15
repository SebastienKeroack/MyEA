#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <array>

void Neural_Network::Initialization__Uniform(T_ const bias_received,
                                                                T_ const lower_bound_received,
                                                                T_ const upper_bound_received)
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
                this->Initialize__Uniform(this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_first_connection_index,
                                                    this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_last_connection_index,
                                                    lower_bound_received,
                                                    upper_bound_received);

                this->Initialize__Constant__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Initialize__Uniform(this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_first_connection_index,
                                                    this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_last_connection_index,
                                                    lower_bound_received,
                                                    upper_bound_received);

                this->Initialize__Uniform__AF_Ind_Recurrent(tmp_ptr_layer_it);

                this->Initialize__Constant__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                this->Initialize__Uniform__LSTM(std::array<T_, 5_zu>{lower_bound_received, lower_bound_received, lower_bound_received, lower_bound_received, lower_bound_received}.data(),
                                                               std::array<T_, 5_zu>{upper_bound_received, upper_bound_received, upper_bound_received, upper_bound_received, upper_bound_received}.data(),
                                                               tmp_ptr_layer_it);

                this->Initialize__Constant__LSTM__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Can not initialize weights in the layer %zu with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
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
    this->_type_weights_initializer = MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_UNIFORM;
}
