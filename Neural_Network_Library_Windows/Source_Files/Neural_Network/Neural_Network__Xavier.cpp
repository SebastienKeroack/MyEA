#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <array>

T_ Neural_Network::Xavier_Weights_Normal_Variance(size_t const fan_in_received,
                                                                              size_t const fan_out_received,
                                                                              enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION const type_layer_activation_received)
{
    switch(type_layer_activation_received)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC: 
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: return(static_cast<T_>(sqrt(2.0 / static_cast<double>(fan_in_received + fan_out_received)))); // Xavier Glorot & Yoshua Bengio.
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER: return(static_cast<T_>(sqrt(2.0 / static_cast<double>(fan_in_received)))); // Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION: return(static_cast<T_>(sqrt(1.0 / static_cast<double>(fan_in_received)))); // Self-Normalizing Neural Networks.
        default:
            PRINT_FORMAT("%s: %s: ERROR: Can not get variance with (%u | %s) as the type activation layer. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     type_layer_activation_received,
                                     MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[type_layer_activation_received].c_str(),
                                     __LINE__);
            return(1_T);
    }
}

T_ Neural_Network::Xavier_Weights_Uniform_Variance(size_t const fan_in_received,
                                                                               size_t const fan_out_received,
                                                                               enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION const type_layer_activation_received)
{
    switch(type_layer_activation_received)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: return(static_cast<T_>(sqrt(6.0 / static_cast<double>(fan_in_received + fan_out_received)))); // Xavier Glorot & Yoshua Bengio.
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER: 
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION: return(static_cast<T_>(sqrt(6.0 / static_cast<double>(fan_in_received)))); // Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
        default:
            PRINT_FORMAT("%s: %s: ERROR: Can not get variance with (%u | %s) as the type activation layer. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     type_layer_activation_received,
                                     MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[type_layer_activation_received].c_str(),
                                     __LINE__);
            return(1_T);
    }
}

void Neural_Network::Xavier_Weights_Normal(T_ const bias_received, T_ const scaling_factor_received)
{
    // Bias should always be zero at initialization.
    // Except the LSTM forget gate, bias should be to one at initialization.
    
        // fan_in: number of neurons feeding into it.
    size_t tmp_fan_in,
        // fan_out: number of neurons fed to.
              tmp_fan_out;
    
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                               *tmp_ptr_previous_layer,
                               *tmp_ptr_next_layer_end,
                               *tmp_ptr_next_layer_it;
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1);

    this->Class_Generator_Real_Weights.Range(0_T, 1_T);

    // Loop though each layer.
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        // If the current layer is a pooling/residual layer, continue.
        if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
          ||
          tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING
          ||
          tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { continue; }
        
        tmp_ptr_previous_layer = tmp_ptr_layer_it->previous_connected_layers[0u];
        
        if((tmp_fan_in = *tmp_ptr_previous_layer->ptr_number_outputs) == 0_zu)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not get \"fan_in\" with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_previous_layer->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str(),
                                     __LINE__);

            continue;
        }

        // Fan out.
        if(tmp_ptr_layer_it + 1 != this->ptr_last_layer)
        {
            tmp_fan_out = 0_zu;
            
            for(tmp_ptr_next_layer_it = tmp_ptr_layer_it->next_connected_layers[0u],
                tmp_ptr_next_layer_end = tmp_ptr_next_layer_it + 1; tmp_ptr_next_layer_it != tmp_ptr_next_layer_end; ++tmp_ptr_next_layer_it)
            {
                if((tmp_fan_out += *tmp_ptr_next_layer_it->ptr_number_outputs) == 0_zu)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not get \"fan_out\" with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_next_layer_it->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_next_layer_it->type_layer].c_str(),
                                             __LINE__);
                }
            }

            tmp_fan_out /= tmp_ptr_layer_it->next_connected_layers.size();
        }
        else { tmp_fan_out = tmp_fan_in; }
        // |END| Fan out. |END|

        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                this->Xavier_Weights_Normal__FC(scaling_factor_received * this->Xavier_Weights_Normal_Variance(tmp_fan_in,
                                                                                                                                                                 tmp_fan_out,
                                                                                                                                                                 tmp_ptr_layer_it->type_activation),
                                                                   tmp_ptr_layer_it);

                this->Initialize__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Xavier_Weights_Normal__FC(scaling_factor_received * this->Xavier_Weights_Normal_Variance(tmp_fan_in,
                                                                                                                                                                 tmp_fan_out,
                                                                                                                                                                 tmp_ptr_layer_it->type_activation),
                                                                   tmp_ptr_layer_it);
                
                
                this->Initialize__Weight__AF_Ind_Recurrent(tmp_ptr_layer_it);
                
                // Reset normal distribution variable. change by: "Initialize__Weight__AF_Ind_Recurrent".
                this->Class_Generator_Real_Weights.Range(0_T, 1_T);

                this->Initialize__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                this->Xavier_Weights_Normal__LSTM(scaling_factor_received * this->Xavier_Weights_Normal_Variance(tmp_fan_in,
                                                                                                                                                                      tmp_fan_out,
                                                                                                                                                                      tmp_ptr_layer_it->type_activation),
                                                                       scaling_factor_received * this->Xavier_Weights_Normal_Variance(tmp_fan_in,
                                                                                                                                                                      tmp_fan_out,
                                                                                                                                                                      MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC),
                                                                       scaling_factor_received,
                                                                       tmp_ptr_layer_it);

                this->Initialize__LSTM__Bias(bias_received, tmp_ptr_layer_it);
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

    if(this->number_recurrent_depth > 1_zu
      &&
      this->number_time_delays + 1_zu == this->number_recurrent_depth)
    { this->Initialize__Weight__AF_Ind_Recurrent__Long_Term_Memory(); }

    if(this->ptr_array_derivatives_parameters != nullptr) { this->Clear_Training_Arrays(); }

    if(this->Use__Normalization()) { this->Clear__Parameter__Normalized_Unit(); }
}

void Neural_Network::Xavier_Weights_Normal__FC(T_ const variance_received, struct Layer *const ptr_layer_it_received)
{
    struct Neuron_unit const *tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit);
    struct Neuron_unit *tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);
    
    size_t const tmp_number_connections(*tmp_ptr_neuron_unit_it->ptr_number_connections);
    size_t tmp_connection_index;

    T_ *tmp_ptr_array_weights(this->ptr_array_parameters + *tmp_ptr_neuron_unit_it->ptr_first_connection_index);
    
    // Loop through each neurons.
    for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                       tmp_ptr_array_weights += tmp_number_connections)
    {
        // Loop through each connections.
        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_weights[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, variance_received);

            /*
            if(tmp_ptr_array_weights[tmp_connection_index] < 0_T)
            {
                PRINT_FORMAT("xavW[%zu]: %f" NEW_LINE, *tmp_ptr_neuron_unit_it->ptr_first_connection_index + tmp_connection_index, tmp_ptr_array_weights[tmp_connection_index]);
            }
            */
        }
    }
}

void Neural_Network::Xavier_Weights_Normal__LSTM(T_ const input_cell_variance_received,
                                                                              T_ const input_gates_variance_received,
                                                                              T_ const scaling_factor_received,
                                                                              struct Layer *const ptr_layer_it_received)
{
    struct Block_unit const *tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;

    size_t const tmp_number_peephole_connections(tmp_ptr_block_unit_it->last_index_peephole_input_gate - tmp_ptr_block_unit_it->first_index_peephole_input_gate),
                       tmp_number_feedforward_connections(tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrent_connections(tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index;
    
    T_ const tmp_recurrent_cell_variance(scaling_factor_received * this->Xavier_Weights_Normal_Variance(tmp_number_recurrent_connections,
                                                                                                                                                         tmp_number_recurrent_connections,
                                                                                                                                                         ptr_layer_it_received->type_activation)),
                 tmp_recurrent_gates_variance(scaling_factor_received * this->Xavier_Weights_Normal_Variance(tmp_number_recurrent_connections,
                                                                                                                                                           tmp_number_recurrent_connections,
                                                                                                                                                           MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC));
    T_ *tmp_ptr_array_parameters;
    
#ifndef NO_PEEPHOLE
    T_ const tmp_peephole_variance(scaling_factor_received * this->Xavier_Weights_Normal_Variance(tmp_number_peephole_connections,
                                                                                                                                                  tmp_number_peephole_connections,
                                                                                                                                                  ptr_layer_it_received->type_activation));
#endif

    // Loop through each blocks.
    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        // Loop through each cells.
        for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
        {
            // Input, cell.
            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index)
            { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, input_cell_variance_received); }
            // |END| Input, cell. |END|
            
            // Recurrent, cell.
            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index)
            { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, tmp_recurrent_cell_variance); }
            // |END| Recurrent, cell. |END|
        }

        // Input, gates.
        //  Input gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, input_gates_variance_received); }
        //  |END| Input gate. |END|

        //  Forget gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, input_gates_variance_received); }
        //  |END| Forget gate. |END|

        //  Output gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, input_gates_variance_received); }
        //  |END| Output gate. |END|
        // |END| Input, gates. |END|
        
        // Recurrent, gates.
        //  Input gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, tmp_recurrent_gates_variance); }
        //  |END| Input gate. |END|

        //  Forget gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, tmp_recurrent_gates_variance); }
        //  |END| Forget gate. |END|

        //  Output gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, tmp_recurrent_gates_variance); }
        //  |END| Output gate. |END|
        // |END| Recurrent, gates. |END|
        
    #ifndef NO_PEEPHOLE
        // Peepholes.
        //  Input gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, tmp_peephole_variance); }
        //  |END| Input gate. |END|
        
        //  Forget gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
        {tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, tmp_peephole_variance); }
        //  |END| Forget gate. |END|
        
        //  Output gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real_Weights.Gaussian(0_T, tmp_peephole_variance); }
        //  |END| Output gate. |END|
        // |END| Peepholes. |END|
    #endif
    }
}

void Neural_Network::Xavier_Weights_Uniform(T_ const bias_received, T_ const scaling_factor_received)
{
        // fan_in: number of neurons feeding into it.
    size_t tmp_fan_in,
        // fan_out: number of neurons fed to.
              tmp_fan_out;
    
    T_ tmp_variance[5u];

    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                               *tmp_ptr_previous_layer,
                               *tmp_ptr_next_layer_end,
                               *tmp_ptr_next_layer_it;
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
        
        tmp_ptr_previous_layer = tmp_ptr_layer_it->previous_connected_layers[0u];
        
        if((tmp_fan_in = *tmp_ptr_previous_layer->ptr_number_outputs) == 0_zu)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not get \"fan_in\" with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_previous_layer->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str(),
                                     __LINE__);

            continue;
        }
        
        // Fan out.
        if(tmp_ptr_layer_it + 1 != this->ptr_last_layer)
        {
            tmp_fan_out = 0_zu;
            
            for(tmp_ptr_next_layer_it = tmp_ptr_layer_it->next_connected_layers[0u],
                tmp_ptr_next_layer_end = tmp_ptr_next_layer_it + tmp_ptr_layer_it->next_connected_layers.size(); tmp_ptr_next_layer_it != tmp_ptr_next_layer_end; ++tmp_ptr_next_layer_it)
            {
                if((tmp_fan_out += *tmp_ptr_next_layer_it->ptr_number_outputs) == 0_zu)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not get \"fan_out\" with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_next_layer_it->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_next_layer_it->type_layer].c_str(),
                                             __LINE__);
                }
            }

            tmp_fan_out /= tmp_ptr_layer_it->next_connected_layers.size();
        }
        else { tmp_fan_out = tmp_fan_in; }
        // |END| Fan out. |END|

        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                tmp_variance[0u] = scaling_factor_received * this->Xavier_Weights_Uniform_Variance(tmp_fan_in,
                                                                                                                                            tmp_fan_out,
                                                                                                                                            tmp_ptr_layer_it->type_activation);

                this->Initialize__Weight__FC(-tmp_variance[0u],
                                                           tmp_variance[0u],
                                                           tmp_ptr_layer_it);

                this->Initialize__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                tmp_variance[0u] = scaling_factor_received * this->Xavier_Weights_Uniform_Variance(tmp_fan_in,
                                                                                                                                            tmp_fan_out,
                                                                                                                                            tmp_ptr_layer_it->type_activation);
                
                this->Initialize__Weight__FC(-tmp_variance[0u],
                                                           tmp_variance[0u],
                                                           tmp_ptr_layer_it);

                this->Initialize__Weight__AF_Ind_Recurrent(tmp_ptr_layer_it);

                this->Initialize__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                tmp_variance[0u] = scaling_factor_received * this->Xavier_Weights_Uniform_Variance(tmp_fan_in,
                                                                                                                                            tmp_fan_out,
                                                                                                                                            tmp_ptr_layer_it->type_activation);
                
                tmp_variance[1u] = scaling_factor_received * this->Xavier_Weights_Uniform_Variance(tmp_fan_in,
                                                                                                                                            tmp_fan_out,
                                                                                                                                            MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC);
                
                tmp_variance[2u] = scaling_factor_received * this->Xavier_Weights_Uniform_Variance(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                            static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                            tmp_ptr_layer_it->type_activation);
                
                tmp_variance[3u] = scaling_factor_received * this->Xavier_Weights_Uniform_Variance(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                            static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                            MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC);
                
            #ifndef NO_PEEPHOLE
                tmp_variance[4u] = scaling_factor_received * this->Xavier_Weights_Uniform_Variance(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units) / static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units),
                                                                                                                                            static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units) / static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units),
                                                                                                                                            tmp_ptr_layer_it->type_activation);
            #endif
                
                this->Initialize__Weight__LSTM(std::array<T_, 5_zu>{-tmp_variance[0u], -tmp_variance[1u], -tmp_variance[2u], -tmp_variance[3u], -tmp_variance[4u]}.data(),
                                                               std::array<T_, 5_zu>{tmp_variance[0u], tmp_variance[1u], tmp_variance[2u], tmp_variance[3u], tmp_variance[4u]}.data(),
                                                               tmp_ptr_layer_it);

                this->Initialize__LSTM__Bias(bias_received, tmp_ptr_layer_it);
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
    
    if(this->number_recurrent_depth > 1_zu
      &&
      this->number_time_delays + 1_zu == this->number_recurrent_depth)
    { this->Initialize__Weight__AF_Ind_Recurrent__Long_Term_Memory(); }

    if(this->ptr_array_derivatives_parameters != nullptr) { this->Clear_Training_Arrays(); }

    if(this->Use__Normalization()) { this->Clear__Parameter__Normalized_Unit(); }
}
