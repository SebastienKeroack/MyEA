#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <array>

T_ Neural_Network::Initialization__Gain__Scale(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received)
{
    switch(type_activation_function_received)
    {
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_COSINE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_COSINE_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELU:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELLIOT:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELLIOT_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN_STEPWISE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ISRU:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ISRLU:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR_PIECE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR_PIECE_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SELU:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SINE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID_STEPWISE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SINE_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SOFTMAX:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_THRESHOLD:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_THRESHOLD_SYMMETRIC: return(1_T);
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LEAKY_RELU: return(static_cast<T_>(sqrt(2.0 / (1.0 + pow(AF_LRELU_ALPHA, 2.0) )) ));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_PARAMETRIC_RELU: return(static_cast<T_>(sqrt(2.0 / (1.0 + pow(AF_PRELU_ALPHA, 2.0) )) ));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_RELU: return(static_cast<T_>(sqrt(2.0)));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH_STEPWISE: return(5_T / 3_T);
        default:
            PRINT_FORMAT("%s: %s: ERROR: Activation function type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     type_activation_function_received,
                                     MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[type_activation_function_received].c_str(),
                                     __LINE__);
                return(1_T);
    }
}

T_ Neural_Network::Initialization__Gaussian__Variance(size_t const fan_in_received,
                                                                                size_t const fan_out_received,
                                                                                enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION const type_layer_activation_received)
{
    switch(type_layer_activation_received)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC: 
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX: return(static_cast<T_>(sqrt(2.0 / static_cast<double>(fan_in_received + fan_out_received)))); // Xavier Glorot & Yoshua Bengio.
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER: // Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION: return(static_cast<T_>(sqrt(1.0 / static_cast<double>(fan_in_received)))); // Self-Normalizing Neural Networks.
        default:
            PRINT_FORMAT("%s: %s: ERROR: Can not get variance with (%u | %s) as the type activation layer. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     type_layer_activation_received,
                                     MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[type_layer_activation_received].c_str(),
                                     __LINE__);
                return(1_T);
    }
}

T_ Neural_Network::Initialization__Uniform__Variance(size_t const fan_in_received,
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
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     type_layer_activation_received,
                                     MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[type_layer_activation_received].c_str(),
                                     __LINE__);
                return(1_T);
    }
}

void Neural_Network::Initialization__Glorot__Gaussian(T_ const bias_received)
{
        // fan_in: number of neurons feeding into it.
    size_t tmp_fan_in,
        // fan_out: number of neurons fed to.
              tmp_fan_out;
    
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                               *tmp_ptr_previous_connected_layer,
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
        
        tmp_ptr_previous_connected_layer = tmp_ptr_layer_it->previous_connected_layers[0u];
        
        if((tmp_fan_in = *tmp_ptr_previous_connected_layer->ptr_number_outputs) == 0_zu)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not get \"fan_in\" with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_previous_connected_layer->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_connected_layer->type_layer].c_str(),
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
                                             MyEA::Time::Date_Time_Now().c_str(),
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
                this->Initialize__Gaussian(this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_first_connection_index,
                                                      this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_last_connection_index,
                                                      this->Initialization__Gain__Scale(*tmp_ptr_layer_it->ptr_array_AF_units->ptr_type_activation_function) * this->Initialization__Gaussian__Variance(tmp_fan_in,
                                                                                                                                                                                                                                                                        tmp_fan_out,
                                                                                                                                                                                                                                                                        tmp_ptr_layer_it->type_activation));

                this->Initialize__Constant__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Initialize__Gaussian(this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_first_connection_index,
                                                      this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_last_connection_index,
                                                      this->Initialization__Gain__Scale(*tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units->ptr_type_activation_function) * this->Initialization__Gaussian__Variance(tmp_fan_in,
                                                                                                                                                                                                                                                                                             tmp_fan_out,
                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it->type_activation));
                
                this->Initialize__Uniform__AF_Ind_Recurrent(tmp_ptr_layer_it);
                
                this->Initialize__Constant__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                this->Initialize__Gaussian__LSTM(this->Initialization__Gain__Scale(tmp_ptr_layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Gaussian__Variance(tmp_fan_in,
                                                                                                                                                                                                                                                                             tmp_fan_out,
                                                                                                                                                                                                                                                                             tmp_ptr_layer_it->type_activation),
                                                                  this->Initialization__Gain__Scale(tmp_ptr_layer_it->ptr_array_block_units->activation_function_gate) * this->Initialization__Gaussian__Variance(tmp_fan_in,
                                                                                                                                                                                                                                                                                 tmp_fan_out,
                                                                                                                                                                                                                                                                                 this->Activation_Function__To__Class_Activation_Function(tmp_ptr_layer_it->ptr_array_block_units->activation_function_gate)),
                                                                  this->Initialization__Gain__Scale(tmp_ptr_layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Gaussian__Variance(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                                             static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                                             tmp_ptr_layer_it->type_activation),
                                                                 this->Initialization__Gain__Scale(tmp_ptr_layer_it->ptr_array_block_units->activation_function_gate) * this->Initialization__Gaussian__Variance(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                                                static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                                                this->Activation_Function__To__Class_Activation_Function(tmp_ptr_layer_it->ptr_array_block_units->activation_function_gate)),
                                                                 this->Initialization__Gain__Scale(tmp_ptr_layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Gaussian__Variance(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units) / static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units),
                                                                                                                                                                                                                                                                            static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units) / static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units),
                                                                                                                                                                                                                                                                            tmp_ptr_layer_it->type_activation),
                                                                  tmp_ptr_layer_it);

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
    this->_type_weights_initializer = MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_GLOROT_GAUSSIAN;
}

void Neural_Network::Initialization__Glorot__Uniform(T_ const bias_received)
{
        // fan_in: number of neurons feeding into it.
    size_t tmp_fan_in,
        // fan_out: number of neurons fed to.
              tmp_fan_out;
    
    T_ tmp_variance[5u];

    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                               *tmp_ptr_previous_connected_layer,
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
        
        tmp_ptr_previous_connected_layer = tmp_ptr_layer_it->previous_connected_layers[0u];
        
        if((tmp_fan_in = *tmp_ptr_previous_connected_layer->ptr_number_outputs) == 0_zu)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not get \"fan_in\" with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_previous_connected_layer->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_connected_layer->type_layer].c_str(),
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
                                             MyEA::Time::Date_Time_Now().c_str(),
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
                tmp_variance[0u] = this->Initialization__Gain__Scale(*tmp_ptr_layer_it->ptr_array_AF_units->ptr_type_activation_function) * this->Initialization__Uniform__Variance(tmp_fan_in,
                                                                                                                                                                                                                                                            tmp_fan_out,
                                                                                                                                                                                                                                                            tmp_ptr_layer_it->type_activation);
                
                this->Initialize__Uniform(this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_first_connection_index,
                                                    this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_last_connection_index,
                                                    -tmp_variance[0u],
                                                    tmp_variance[0u]);
                
                this->Initialize__Constant__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                tmp_variance[0u] = this->Initialization__Gain__Scale(*tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units->ptr_type_activation_function) * this->Initialization__Uniform__Variance(tmp_fan_in,
                                                                                                                                                                                                                                                                                tmp_fan_out,
                                                                                                                                                                                                                                                                                tmp_ptr_layer_it->type_activation);
                
                this->Initialize__Uniform(this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_first_connection_index,
                                                    this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_last_connection_index,
                                                    -tmp_variance[0u],
                                                    tmp_variance[0u]);

                this->Initialize__Uniform__AF_Ind_Recurrent(tmp_ptr_layer_it);

                this->Initialize__Constant__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                tmp_variance[0u] = this->Initialization__Gain__Scale(tmp_ptr_layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Uniform__Variance(tmp_fan_in,
                                                                                                                                                                                                                                                     tmp_fan_out,
                                                                                                                                                                                                                                                     tmp_ptr_layer_it->type_activation);
                
                tmp_variance[1u] = this->Initialization__Gain__Scale(tmp_ptr_layer_it->ptr_array_block_units->activation_function_gate) * this->Initialization__Uniform__Variance(tmp_fan_in,
                                                                                                                                                                                                                                                         tmp_fan_out,
                                                                                                                                                                                                                                                         this->Activation_Function__To__Class_Activation_Function(tmp_ptr_layer_it->ptr_array_block_units->activation_function_gate));
                
                tmp_variance[2u] = this->Initialization__Gain__Scale(tmp_ptr_layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Uniform__Variance(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                     static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                     tmp_ptr_layer_it->type_activation);
                
                tmp_variance[3u] = this->Initialization__Gain__Scale(tmp_ptr_layer_it->ptr_array_block_units->activation_function_gate) * this->Initialization__Uniform__Variance(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                         static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                         this->Activation_Function__To__Class_Activation_Function(tmp_ptr_layer_it->ptr_array_block_units->activation_function_gate));
                
                tmp_variance[4u] = this->Initialization__Gain__Scale(tmp_ptr_layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Uniform__Variance(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units) / static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units),
                                                                                                                                                                                                                                                     static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units) / static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units),
                                                                                                                                                                                                                                                     tmp_ptr_layer_it->type_activation);
                
                this->Initialize__Uniform__LSTM(std::array<T_, 5_zu>{-tmp_variance[0u], -tmp_variance[1u], -tmp_variance[2u], -tmp_variance[3u], -tmp_variance[4u]}.data(),
                                                               std::array<T_, 5_zu>{tmp_variance[0u], tmp_variance[1u], tmp_variance[2u], tmp_variance[3u], tmp_variance[4u]}.data(),
                                                               tmp_ptr_layer_it);

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
    this->_type_weights_initializer = MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_GLOROT_UNIFORM;
}
