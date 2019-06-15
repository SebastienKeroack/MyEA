#include "stdafx.hpp"

#include <iostream>

void Neural_Network::Organize__Previous_Layers_Connected(size_t &ref_state_layer_index_received,
                                                                                          struct Layer *const ptr_layer_received,
                                                                                          struct Layer const *&ptr_layer_state_received) const
{
    ptr_layer_received->previous_connected_layers.clear();

    if(ptr_layer_received <= this->ptr_array_layers) { return; }
    
    struct Layer const *tmp_ptr_previous_layer_connected;

    /* If the previous layer was inside a residual block.
       Connect the layer to the previous residual unit (identity-mapping shortcut). */
    if(ptr_layer_state_received != nullptr && ref_state_layer_index_received++ == ptr_layer_state_received->block_depth)
    {
        tmp_ptr_previous_layer_connected = ptr_layer_state_received;

        ptr_layer_received->previous_connected_layers.push_back(tmp_ptr_previous_layer_connected);

        ref_state_layer_index_received = 0_zu;

        ptr_layer_state_received = nullptr;
    }
    else
    {
        tmp_ptr_previous_layer_connected = ptr_layer_received - 1;

        /* If the previous layer is a residual block.
           Get the previously connected layer from the previous layer (residual block). */
        if(tmp_ptr_previous_layer_connected->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
        { tmp_ptr_previous_layer_connected = tmp_ptr_previous_layer_connected->previous_connected_layers[0u]; }
        
        ptr_layer_received->previous_connected_layers.push_back(tmp_ptr_previous_layer_connected);
    }

    /* If the layer is a residual block.
       Keep track the following layers are inside a residual block. */
    if(ptr_layer_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { ptr_layer_state_received = ptr_layer_received; }
}

void Neural_Network::Organize__Next_Layers_Connected(size_t &ref_state_layer_index_received,
                                                                                    struct Layer *const ptr_layer_received,
                                                                                    struct Layer const *&ptr_layer_state_received) const
{
    ptr_layer_received->next_connected_layers.clear();

    struct Layer const *tmp_ptr_next_layer_connected;

    // If the layer is a residual block. Add the next layer after the current residual block.
    if(ptr_layer_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
    { tmp_ptr_next_layer_connected = ptr_layer_received + ptr_layer_received->block_depth + 1_zu; }
    else
    { tmp_ptr_next_layer_connected = ptr_layer_received + 1; }
    
    /* If the layer are inside a residual block and is the last layer inside the block.
       Connect the layer to the residual unit (identity-mapping shortcut). */
    if(ptr_layer_state_received != nullptr && ++ref_state_layer_index_received == ptr_layer_state_received->block_depth)
    {
        tmp_ptr_next_layer_connected = ptr_layer_state_received;

        ptr_layer_received->next_connected_layers.push_back(tmp_ptr_next_layer_connected);

        ref_state_layer_index_received = 0_zu;

        ptr_layer_state_received = nullptr;
    }
    else if(tmp_ptr_next_layer_connected < this->ptr_last_layer)
    {
        // Push back the next connected layer.
        ptr_layer_received->next_connected_layers.push_back(tmp_ptr_next_layer_connected);

        // If the next layer is a residual block. Shift the next layer by plus one and push it back to the vector.
        if(tmp_ptr_next_layer_connected->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
        {
            ++tmp_ptr_next_layer_connected;

            // Push back the next connected layer.
            ptr_layer_received->next_connected_layers.push_back(tmp_ptr_next_layer_connected);
        }
    }

    // Keep track the following layers are inside a residual block.
    if(ptr_layer_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { ptr_layer_state_received = ptr_layer_received; }
}

void Neural_Network::Organize__Layer__Group(size_t &ref_state_layer_index_received,
                                                                     struct Layer *const ptr_layer_received,
                                                                     struct Layer const *&ptr_layer_state_received) const
{
    // If the layer are inside a residual block.
    if(ptr_layer_state_received != nullptr)
    {
        // If is the last layer inside the block.
        if(++ref_state_layer_index_received == ptr_layer_state_received->block_depth)
        {
            ref_state_layer_index_received = 0_zu;

            ptr_layer_state_received = nullptr;
        }

        ptr_layer_received->type_group = MyEA::Common::ENUM_TYPE_GROUP::TYPE_GROUP_RESIDUAL;
    }
    else { ptr_layer_received->type_group = MyEA::Common::ENUM_TYPE_GROUP::TYPE_GROUP_NONE; }

    // Keep track the following layers are inside a residual block.
    if(ptr_layer_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { ptr_layer_state_received = ptr_layer_received; }
}

bool Neural_Network::operator == (class Neural_Network const &ref_source_Neural_Network_received)
{
    if(&ref_source_Neural_Network_received == this) { return(true); }

    return(this->total_layers == ref_source_Neural_Network_received.total_layers
             &&
             this->total_weights == ref_source_Neural_Network_received.total_weights
             &&
             this->total_bias == ref_source_Neural_Network_received.total_bias
             &&
             this->total_parameters == ref_source_Neural_Network_received.total_parameters
             &&
             this->total_basic_units == ref_source_Neural_Network_received.total_basic_units
             &&
             this->total_basic_indice_units == ref_source_Neural_Network_received.total_basic_indice_units
             &&
             this->total_neuron_units == ref_source_Neural_Network_received.total_neuron_units
             &&
             this->total_AF_units == ref_source_Neural_Network_received.total_AF_units
             &&
             this->total_AF_Ind_recurrent_units == ref_source_Neural_Network_received.total_AF_Ind_recurrent_units
             &&
             this->total_cell_units == ref_source_Neural_Network_received.total_cell_units
             &&
             this->total_block_units == ref_source_Neural_Network_received.total_block_units
             &&
             this->total_normalized_units == ref_source_Neural_Network_received.total_normalized_units
             &&
             this->total_dropout_alpha_layers == ref_source_Neural_Network_received.total_dropout_alpha_layers
             &&
             this->total_dropout_bernoulli_layers == ref_source_Neural_Network_received.total_dropout_bernoulli_layers
             &&
             this->total_dropout_bernoulli_inverted_layers == ref_source_Neural_Network_received.total_dropout_bernoulli_inverted_layers
             &&
             this->total_dropout_gaussian_layers == ref_source_Neural_Network_received.total_dropout_gaussian_layers
             &&
             this->total_dropout_shakedrop_layers == ref_source_Neural_Network_received.total_dropout_shakedrop_layers
             &&
             this->total_dropout_uout_layers == ref_source_Neural_Network_received.total_dropout_uout_layers
             &&
             this->total_dropout_zoneout_layers == ref_source_Neural_Network_received.total_dropout_zoneout_layers
             &&
             this->total_batch_normalization_layers == ref_source_Neural_Network_received.total_batch_normalization_layers
             &&
             this->total_batch_renormalization_layers == ref_source_Neural_Network_received.total_batch_renormalization_layers
             &&
             this->total_ghost_batch_normalization_layers == ref_source_Neural_Network_received.total_ghost_batch_normalization_layers
             &&
             this->total_streaming_normalization_layers == ref_source_Neural_Network_received.total_streaming_normalization_layers
             &&
             this->total_k_sparse_layers == ref_source_Neural_Network_received.total_k_sparse_layers
             &&
             this->total_tied_parameter_layers == ref_source_Neural_Network_received.total_tied_parameter_layers
             &&
             this->total_constraint_recurrent_weight_layers == ref_source_Neural_Network_received.total_constraint_recurrent_weight_layers);
}

bool Neural_Network::operator != (class Neural_Network const &ref_source_Neural_Network_received) { return(!(*this == ref_source_Neural_Network_received)); }

bool Layer::Compare__Dimensions(struct Layer const &ref_source_Layer_received) const
{
    return(static_cast<size_t>(this->ptr_last_basic_unit - this->ptr_array_basic_units) == static_cast<size_t>(ref_source_Layer_received.ptr_last_basic_unit - ref_source_Layer_received.ptr_array_basic_units)
             &&
             static_cast<size_t>(this->ptr_last_basic_indice_unit - this->ptr_array_basic_indice_units) == static_cast<size_t>(ref_source_Layer_received.ptr_last_basic_indice_unit - ref_source_Layer_received.ptr_array_basic_indice_units)
             &&
             static_cast<size_t>(this->ptr_last_neuron_unit - this->ptr_array_neuron_units) == static_cast<size_t>(ref_source_Layer_received.ptr_last_neuron_unit - ref_source_Layer_received.ptr_array_neuron_units)
             &&
             static_cast<size_t>(this->ptr_last_AF_unit - this->ptr_array_AF_units) == static_cast<size_t>(ref_source_Layer_received.ptr_last_AF_unit - ref_source_Layer_received.ptr_array_AF_units)
             &&
             static_cast<size_t>(this->ptr_last_AF_Ind_recurrent_unit - this->ptr_array_AF_Ind_recurrent_units) == static_cast<size_t>(ref_source_Layer_received.ptr_last_AF_Ind_recurrent_unit - ref_source_Layer_received.ptr_array_AF_Ind_recurrent_units)
             &&
             static_cast<size_t>(this->ptr_last_block_unit - this->ptr_array_block_units) == static_cast<size_t>(ref_source_Layer_received.ptr_last_block_unit - ref_source_Layer_received.ptr_array_block_units)
             &&
             static_cast<size_t>(this->ptr_last_cell_unit - this->ptr_array_cell_units) == static_cast<size_t>(ref_source_Layer_received.ptr_last_cell_unit - ref_source_Layer_received.ptr_array_cell_units)
             &&
             static_cast<size_t>(this->ptr_last_normalized_unit - this->ptr_array_normalized_units) == static_cast<size_t>(ref_source_Layer_received.ptr_last_normalized_unit - ref_source_Layer_received.ptr_array_normalized_units));
}

bool Neural_Network_Initializer::Build__Layer__FC(struct Layer_Parameters &ref_Layer_Parameters_received)
{
    if(ref_Layer_Parameters_received.type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED
      ||
      ref_Layer_Parameters_received.type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT
      ||
      ref_Layer_Parameters_received.type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT)
    {
        ref_Layer_Parameters_received.unit_parameters[0u] = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tNeuron(s): ");
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not a fully connected layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 ref_Layer_Parameters_received.type_layer,
                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[ref_Layer_Parameters_received.type_layer].c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network_Initializer::Build__Layer__Pooling(struct Layer_Parameters &ref_Layer_Parameters_received)
{
    if(ref_Layer_Parameters_received.type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
      ||
      ref_Layer_Parameters_received.type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING)
    {
        ref_Layer_Parameters_received.unit_parameters[0u] = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tKernel size: ");
        
        ref_Layer_Parameters_received.unit_parameters[1u] = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tStride: ");
        
        ref_Layer_Parameters_received.unit_parameters[2u] = MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ":\tPadding: ");
        
        ref_Layer_Parameters_received.unit_parameters[3u] = MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ":\tDilation: ");

        ref_Layer_Parameters_received.unit_parameters[4u] = static_cast<size_t>(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ":\tCeil mode?"));
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not a pooling layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 ref_Layer_Parameters_received.type_layer,
                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[ref_Layer_Parameters_received.type_layer].c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network_Initializer::Build__Layer__LSTM(struct Layer_Parameters &ref_Layer_Parameters_received)
{
    if(ref_Layer_Parameters_received.type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM)
    {
        ref_Layer_Parameters_received.use_bidirectional = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ":\tBidirectional?");
        
        ref_Layer_Parameters_received.unit_parameters[0u] = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tBlock(s): ");
        
        ref_Layer_Parameters_received.unit_parameters[1u] = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tCells(s) per block: ");
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not a LSTM layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 ref_Layer_Parameters_received.type_layer,
                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[ref_Layer_Parameters_received.type_layer].c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network_Initializer::Build__Layer__Residual(void)
{
    unsigned int tmp_layer_type_index;

    size_t tmp_residual_unit_index,
              tmp_number_residual_units,
              tmp_layer_index,
              tmp_block_depth;

    struct Layer_Parameters tmp_Layer_Parameters;
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Number residual unit(s)." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[1, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
    tmp_number_residual_units = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tNumber residual unit(s): ");

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Block width." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[2, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=2.0." NEW_LINE, MyEA::String::Get__Time().c_str());
    tmp_block_depth = MyEA::String::Cin_Number<size_t>(2_zu, MyEA::String::Get__Time() + ":\tBlock width: ");
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use widening factor, alpha?"))
    {
        bool tmp_while(true);
        
        double tmp_widening_factor_alpha,
                  tmp_widening_factors[2u] = {0},
                  tmp_widening_factor_units[2u] = {0};

        struct Layer_Parameters tmp_widening_Layer_Parameters;
        
        // Residual unit type.
        do
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Layer type." NEW_LINE, MyEA::String::Get__Time().c_str());
            for(tmp_layer_type_index = 1u; tmp_layer_type_index != MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH; ++tmp_layer_type_index)
            {
                PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_layer_type_index,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[static_cast<enum MyEA::Common::ENUM_TYPE_LAYER>(tmp_layer_type_index)].c_str());
            }
            PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED].c_str());
            
            if((tmp_widening_Layer_Parameters.type_layer = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                                            MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH - 1u,
                                                                                                                                                                                                                                            MyEA::String::Get__Time() + ": Residual unit, type: "))) >= MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         1u,
                                         MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH - 1u,
                                         __LINE__);

                return(false);
            }

            switch(tmp_widening_Layer_Parameters.type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    if(this->Build__Layer__FC(tmp_widening_Layer_Parameters) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Build__Layer__FC()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    if(this->Build__Layer__LSTM(tmp_widening_Layer_Parameters) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Build__Layer__LSTM()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_widening_Layer_Parameters.type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_widening_Layer_Parameters.type_layer].c_str(),
                                             __LINE__);
                        continue;
            }
            
            tmp_widening_factor_units[0u] = static_cast<double>(tmp_widening_Layer_Parameters.unit_parameters[0u]);
            tmp_widening_factor_units[1u] = static_cast<double>(tmp_widening_Layer_Parameters.unit_parameters[1u]);
            
            // Widening factor #0.
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Widening factor, alpha[0]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[-%zu, inf] 0=Fixed." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_widening_Layer_Parameters.unit_parameters[0u] - 1_zu);
            tmp_widening_factor_alpha = static_cast<double>(MyEA::String::Cin_Number<long long int>(-static_cast<long long int>(tmp_widening_Layer_Parameters.unit_parameters[0u]) + 1ll, MyEA::String::Get__Time() + ":\tWidening factor, alpha[0]: "));
            tmp_widening_factors[0u] = tmp_widening_factor_alpha / static_cast<double>(tmp_number_residual_units);
            // |END| Widening factor #0. |END|

            // Widening factor #1.
            if(tmp_widening_factor_units[1u] != 0.0)
            {
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Widening factor, alpha[1]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[-%zu, inf] 0=Fixed." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_widening_Layer_Parameters.unit_parameters[1u] - 1_zu);
                tmp_widening_factor_alpha = static_cast<double>(MyEA::String::Cin_Number<long long int>(-static_cast<long long int>(tmp_widening_Layer_Parameters.unit_parameters[1u]) + 1ll, MyEA::String::Get__Time() + ":\tWidening factor, alpha[1]: "));
                tmp_widening_factors[1u] = tmp_widening_factor_alpha / static_cast<double>(tmp_number_residual_units);
            }
            // |END| Widening factor #1. |END|

            tmp_while = false;
        } while(tmp_while);
        
        // Loop through each remaining residual unit(s).
        for(tmp_residual_unit_index = 0_zu; tmp_residual_unit_index != tmp_number_residual_units; ++tmp_residual_unit_index)
        {
            // Residual unit.
            tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL;

            tmp_Layer_Parameters.unit_parameters[0u] = tmp_block_depth;
            
            this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
            // |END| Residual unit. |END|
            
            // Building block.
            tmp_Layer_Parameters.type_layer = tmp_widening_Layer_Parameters.type_layer;

            switch(tmp_Layer_Parameters.type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    tmp_widening_factor_units[0u] += tmp_widening_factors[0u];
                    tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    tmp_Layer_Parameters.use_bidirectional = tmp_widening_Layer_Parameters.use_bidirectional;
                    
                    tmp_widening_factor_units[0u] += tmp_widening_factors[0u];
                    tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);

                    tmp_widening_factor_units[1u] += tmp_widening_factors[1u];
                    tmp_Layer_Parameters.unit_parameters[1u] = static_cast<size_t>(tmp_widening_factor_units[1u]);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_Layer_Parameters.type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str(),
                                             __LINE__);
                        return(false);
            }

            for(tmp_layer_index = 0_zu; tmp_layer_index != tmp_block_depth; ++tmp_layer_index) { this->vector_layers_parameters.push_back(tmp_Layer_Parameters); }
            // |END| Building block. |END|
        }
    }
    else
    {
        // Loop through each remaining residual unit(s).
        for(tmp_residual_unit_index = 0_zu; tmp_residual_unit_index != tmp_number_residual_units; ++tmp_residual_unit_index)
        {
            // Residual unit.
            tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL;

            tmp_Layer_Parameters.unit_parameters[0u] = tmp_block_depth;
            
            this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
            // |END| Residual unit. |END|
            
            // Building block.
            for(tmp_layer_index = 0_zu; tmp_layer_index != tmp_block_depth; ++tmp_layer_index)
            {
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Layer type." NEW_LINE, MyEA::String::Get__Time().c_str());
                for(tmp_layer_type_index = 1u; tmp_layer_type_index != MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH; ++tmp_layer_type_index)
                {
                    PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             tmp_layer_type_index,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[static_cast<enum MyEA::Common::ENUM_TYPE_LAYER>(tmp_layer_type_index)].c_str());
                }
                PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED].c_str());
                
                if((tmp_Layer_Parameters.type_layer = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                               MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH - 1u,
                                                                                                                                                                                                                               MyEA::String::Get__Time() + ": Residual[" + std::to_string(tmp_residual_unit_index) + "], layer[" + std::to_string(tmp_layer_index) + "], type: "))) >= MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             1u,
                                             MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH - 1u,
                                             __LINE__);

                    return(false);
                }

                switch(tmp_Layer_Parameters.type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                        if(this->Build__Layer__Pooling(tmp_Layer_Parameters) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Build__Layer__Pooling()\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                        if(this->Build__Layer__FC(tmp_Layer_Parameters) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Build__Layer__FC()\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                        if(this->Build__Layer__LSTM(tmp_Layer_Parameters) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Build__Layer__LSTM()\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                            break;
                    default:
                            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_Layer_Parameters.type_layer,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str(),
                                                     __LINE__);
                                continue;
                }

                this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
            }
        }
    }

    return(true);
}

bool Neural_Network_Initializer::While__Push_Back__Layer(void)
{
    unsigned int tmp_layer_type_index;

    size_t tmp_layer_index(this->vector_layers_parameters.size());

    struct Layer_Parameters tmp_Layer_Parameters;

    while(this->vector_layers_parameters.size() < 2_zu || MyEA::String::NoOrYes(MyEA::String::Get__Time() + NEW_LINE + MyEA::String::Get__Time() + ": Add another layer (" + std::to_string(tmp_layer_index) + ")?"))
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Layer type." NEW_LINE, MyEA::String::Get__Time().c_str());
        for(tmp_layer_type_index = 1u; tmp_layer_type_index != MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH; ++tmp_layer_type_index)
        {
            PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_layer_type_index,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[static_cast<enum MyEA::Common::ENUM_TYPE_LAYER>(tmp_layer_type_index)].c_str());
        }
        PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED].c_str());
        
        if((tmp_Layer_Parameters.type_layer = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                         MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH - 1u,
                                                                                                                                                                                                                         MyEA::String::Get__Time() + ": Hidden layer " + std::to_string(tmp_layer_index) + " type: "))) >= MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        1u,
                                        MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH - 1u,
                                        __LINE__);

            return(false);
        }

        switch(tmp_Layer_Parameters.type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                if(this->Build__Layer__Pooling(tmp_Layer_Parameters) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Build__Layer__Pooling()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }

                this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                if(this->Build__Layer__FC(tmp_Layer_Parameters) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Build__Layer__FC()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }

                this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Build__Layer__LSTM(tmp_Layer_Parameters) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Build__Layer__LSTM()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }

                this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                if(tmp_layer_index <= 1_zu)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed as the first hidden/input layer. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_Layer_Parameters.type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str(),
                                             __LINE__);
                    
                    continue;
                }
                else if(this->type_neural_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER)
                {
                    PRINT_FORMAT("%s: %s: ERROR: The autoencoder network can not use residual layer. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    continue;
                }

                if(this->Build__Layer__Residual() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Build_Residual_Block()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_Layer_Parameters.type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str(),
                                             __LINE__);
                        continue;
        }

        tmp_layer_index = this->vector_layers_parameters.size();
    }

    return(true);
}

bool Neural_Network_Initializer::Input_Initialize(void)
{
    size_t tmp_layer_index,
              tmp_layer_length;

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Network type." NEW_LINE, MyEA::String::Get__Time().c_str());
    for(unsigned int tmp_network_type_index(1u); tmp_network_type_index != MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_LENGTH; ++tmp_network_type_index)
    {
        PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_network_type_index,
                                 MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_NETWORKS>(tmp_network_type_index)].c_str());
    }
    PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_FEEDFORWARD].c_str());
    
    if((this->type_neural_network = static_cast<enum MyEA::Common::ENUM_TYPE_NETWORKS>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                               MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_LENGTH - 1u,
                                                                                                                                                                                                               MyEA::String::Get__Time() + ": Type: "))) >= MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 1u,
                                 MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_LENGTH - 1u,
                                 __LINE__);

        return(false);
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    this->number_recurrent_depth = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": Recurrent depth: ");

    if(this->number_recurrent_depth > 1_zu)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Time delays." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0, %zu]." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->number_recurrent_depth - 1_zu);
        this->number_time_delays = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                       this->number_recurrent_depth - 1_zu,
                                                                                                       MyEA::String::Get__Time() + ": Time delays: ");
    }

    struct Layer_Parameters tmp_Layer_Parameters;

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Input layer:" NEW_LINE, MyEA::String::Get__Time().c_str());
    tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED;
    tmp_Layer_Parameters.unit_parameters[0u] = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tNumber of inputs: ");
    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
    
    switch(this->type_neural_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            if(this->While__Push_Back__Layer() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"While__Push_Back__Layer()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            tmp_layer_length = this->vector_layers_parameters.size() - 1_zu;

            for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_layer_length; ++tmp_layer_index)
            {
                tmp_Layer_Parameters.type_layer = this->vector_layers_parameters[tmp_layer_length - tmp_layer_index].type_layer;

                tmp_Layer_Parameters.use_bidirectional = this->vector_layers_parameters[tmp_layer_length - tmp_layer_index].use_bidirectional;

                tmp_Layer_Parameters.unit_parameters[0u] = this->vector_layers_parameters[tmp_layer_length - tmp_layer_index].unit_parameters[0u];
                tmp_Layer_Parameters.unit_parameters[1u] = this->vector_layers_parameters[tmp_layer_length - tmp_layer_index].unit_parameters[0u];

                this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
            }

            tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED;
            tmp_Layer_Parameters.unit_parameters[0u] = this->vector_layers_parameters[0u].unit_parameters[0u];
            this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
                break;
        default:
            if(this->While__Push_Back__Layer() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"While__Push_Back__Layer()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Output layer:" NEW_LINE, MyEA::String::Get__Time().c_str());
            tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED;
            tmp_Layer_Parameters.unit_parameters[0u] = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tNumber of output(s): ");
            this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
                break;
    }

    return(true);
}

bool Neural_Network_Initializer::Template_Initialize(void)
{
    this->type_neural_network = MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_RECURRENT;
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    this->number_recurrent_depth = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": Recurrent depth: ");

    if(this->number_recurrent_depth > 1_zu)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Time delays." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0, %zu]." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->number_recurrent_depth - 1_zu);
        this->number_time_delays = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                       this->number_recurrent_depth - 1_zu,
                                                                                                       MyEA::String::Get__Time() + ": Time delays: ");
    }
    
    bool tmp_use_pooling,
           tmp_use_bottleneck;
    
    double tmp_widening_factor_alpha,
              tmp_widening_factors[2u] = {0},
              tmp_widening_factor_units[2u] = {0};

    size_t tmp_residual_unit_index,
              tmp_number_residual_units,
              tmp_layer_index,
              tmp_block_depth,
              tmp_pooling_layer_mod;

    struct Layer_Parameters tmp_Layer_Parameters,
                                         tmp_pooling_layer_Parameters,
                                         tmp_widening_Layer_Parameters;
    
    // Input layer.
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Input layer:" NEW_LINE, MyEA::String::Get__Time().c_str());
    tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED;
    tmp_Layer_Parameters.unit_parameters[0u] = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tNumber of inputs: ");
    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
    // |END| Input layer. |END|
    
    // #0: Fully connected, independently recurrent.
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT;
    tmp_Layer_Parameters.unit_parameters[0u] = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": First hidden layer, number units: ");
    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

    PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            this->vector_layers_parameters.size() - 1_zu,
                            MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
    PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                            MyEA::String::Get__Time().c_str(),
                            this->vector_layers_parameters.size() - 1_zu,
                            tmp_Layer_Parameters.unit_parameters[0u]);
    // |END| #0: Fully connected, independently recurrent. |END|

    // #1: Residual, Fully connected, independently recurrent.
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Residual block." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[1, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
    tmp_number_residual_units = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tNumber residual units: ");

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Residual block, depth." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[2, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=3." NEW_LINE, MyEA::String::Get__Time().c_str());
    tmp_block_depth = MyEA::String::Cin_Number<size_t>(2_zu, MyEA::String::Get__Time() + ":\tBlock depth: ");
    
    //  Pooling.
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    if((tmp_use_pooling = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use pooling?")))
    {
        if(this->Build__Layer__Pooling(tmp_pooling_layer_Parameters) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Build__Layer__Pooling()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Pooling layer, frequency." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[1, %zu]." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_number_residual_units);
        PRINT_FORMAT("%s: default=%zu." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<size_t>(ceil(static_cast<double>(tmp_number_residual_units) / 3.0)));
        tmp_pooling_layer_mod = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tPooling layer, frequency: ");
    }

    //  Bottleneck.
    if(tmp_block_depth > 2_zu)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: default=true." NEW_LINE, MyEA::String::Get__Time().c_str());
        tmp_use_bottleneck = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use bottleneck?");
    }

    tmp_widening_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT;
    tmp_widening_Layer_Parameters.unit_parameters[0u] = tmp_Layer_Parameters.unit_parameters[0u];
    
    //  Widening factors.
    tmp_widening_factor_units[0u] = static_cast<double>(tmp_widening_Layer_Parameters.unit_parameters[0u]);
    tmp_widening_factor_units[1u] = static_cast<double>(tmp_widening_Layer_Parameters.unit_parameters[1u]);
    
    //      Widening factor #0.
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Widening factor, alpha[0]." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[-%zu, inf] 0=Fixed." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             tmp_widening_Layer_Parameters.unit_parameters[0u] - 1_zu);
    tmp_widening_factor_alpha = static_cast<double>(MyEA::String::Cin_Number<long long int>(-static_cast<long long int>(tmp_widening_Layer_Parameters.unit_parameters[0u]) + 1ll, MyEA::String::Get__Time() + ":\tWidening factor, alpha[0]: "));
    tmp_widening_factors[0u] = tmp_widening_factor_alpha / static_cast<double>(tmp_number_residual_units);
    //      |END| Widening factor #0. |END|

    //      Widening factor #1.
    if(tmp_widening_factor_units[1u] != 0.0)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Widening factor, alpha[1]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[-%zu, inf] 0=Fixed." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_widening_Layer_Parameters.unit_parameters[1u] - 1_zu);
        tmp_widening_factor_alpha = static_cast<double>(MyEA::String::Cin_Number<long long int>(-static_cast<long long int>(tmp_widening_Layer_Parameters.unit_parameters[1u]) + 1ll, MyEA::String::Get__Time() + ":\tWidening factor, alpha[1]: "));
        tmp_widening_factors[1u] = tmp_widening_factor_alpha / static_cast<double>(tmp_number_residual_units);
    }
    //      |END| Widening factor #1. |END|
    //  |END| Widening factors. |END|

    //  Loop through each remaining residual unit(s).
    for(tmp_residual_unit_index = 0_zu; tmp_residual_unit_index != tmp_number_residual_units; ++tmp_residual_unit_index)
    {
        // Residual unit.
        tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL;
        tmp_Layer_Parameters.unit_parameters[0u] = tmp_block_depth;
        this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

        PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->vector_layers_parameters.size() - 1_zu,
                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
        PRINT_FORMAT("%s: Layer[%zu]: Block depth: %zu." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->vector_layers_parameters.size() - 1_zu,
                                 tmp_Layer_Parameters.unit_parameters[0u]);
        // |END| Residual unit. |END|
        
        // Building block.
        tmp_Layer_Parameters.type_layer = tmp_widening_Layer_Parameters.type_layer;

        switch(tmp_Layer_Parameters.type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                tmp_layer_index = 0_zu;

                if(tmp_use_bottleneck)
                {
                    // First hidden layer inside the residual block.
                    tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);

                    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

                    PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                    PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             tmp_Layer_Parameters.unit_parameters[0u]);

                    ++tmp_layer_index;
                    // |END| First hidden layer inside the residual block. |END|

                    // Second hidden layer inside the residual block.
                    tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(MyEA::Math::Maximum<double>(tmp_widening_factor_units[0u], tmp_widening_factor_units[0u] + tmp_widening_factors[0u]) / 2.0);

                    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

                    PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                    PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             tmp_Layer_Parameters.unit_parameters[0u]);

                    ++tmp_layer_index;
                    // |END| Second hidden layer inside the residual block. |END|
                }

                tmp_widening_factor_units[0u] += tmp_widening_factors[0u];
                tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);

                for(; tmp_layer_index != tmp_block_depth; ++tmp_layer_index)
                {
                    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

                    PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                    PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             tmp_Layer_Parameters.unit_parameters[0u]);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                tmp_Layer_Parameters.use_bidirectional = tmp_widening_Layer_Parameters.use_bidirectional;
                
                tmp_layer_index = 0_zu;
                
                if(tmp_use_bottleneck)
                {
                    // First hidden layer inside the residual block.
                    tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);
                    tmp_Layer_Parameters.unit_parameters[1u] = static_cast<size_t>(tmp_widening_factor_units[1u]);

                    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

                    PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                    PRINT_FORMAT("%s: Layer[%zu]: Number block unit(s): %zu." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             tmp_Layer_Parameters.unit_parameters[0u]);
                    PRINT_FORMAT("%s: Layer[%zu]: Number cell unit(s) per block: %zu." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             tmp_Layer_Parameters.unit_parameters[1u]);

                    ++tmp_layer_index;
                    // |END| First hidden layer inside the residual block. |END|

                    // Second hidden layer inside the residual block.
                    tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(MyEA::Math::Maximum<double>(tmp_widening_factor_units[0u], tmp_widening_factor_units[0u] + tmp_widening_factors[0u]) / 2.0);
                    tmp_Layer_Parameters.unit_parameters[1u] = static_cast<size_t>(MyEA::Math::Maximum<double>(tmp_widening_factor_units[1u], tmp_widening_factor_units[1u] + tmp_widening_factors[1u]) / 2.0);

                    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

                    PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                    PRINT_FORMAT("%s: Layer[%zu]: Number block unit(s): %zu." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             tmp_Layer_Parameters.unit_parameters[0u]);
                    PRINT_FORMAT("%s: Layer[%zu]: Number cell unit(s) per block: %zu." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             tmp_Layer_Parameters.unit_parameters[1u]);

                    ++tmp_layer_index;
                    // |END| Second hidden layer inside the residual block. |END|
                }

                tmp_widening_factor_units[0u] += tmp_widening_factors[0u];
                tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);

                tmp_widening_factor_units[1u] += tmp_widening_factors[1u];
                tmp_Layer_Parameters.unit_parameters[1u] = static_cast<size_t>(tmp_widening_factor_units[1u]);

                for(; tmp_layer_index != tmp_block_depth; ++tmp_layer_index)
                {
                    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

                    PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
                    PRINT_FORMAT("%s: Layer[%zu]: Number block unit(s): %zu." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             tmp_Layer_Parameters.unit_parameters[0u]);
                    PRINT_FORMAT("%s: Layer[%zu]: Number cell unit(s) per block: %zu." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             this->vector_layers_parameters.size() - 1_zu,
                                             tmp_Layer_Parameters.unit_parameters[1u]);
                }
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_Layer_Parameters.type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str(),
                                         __LINE__);
                    return(false);
        }
        // |END| Building block. |END|
        
        //  Pooling layer.
        if(tmp_use_pooling
          &&
          (tmp_residual_unit_index + 1_zu) % tmp_pooling_layer_mod == 0_zu)
        {
            this->vector_layers_parameters.push_back(tmp_pooling_layer_Parameters);

            PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     this->vector_layers_parameters.size() - 1_zu,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_pooling_layer_Parameters.type_layer].c_str());
            PRINT_FORMAT("%s: Layer[%zu]: Kernel size: %zu." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     this->vector_layers_parameters.size() - 1_zu,
                                     tmp_pooling_layer_Parameters.unit_parameters[0u]);
            PRINT_FORMAT("%s: Layer[%zu]: Stride: %zu." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     this->vector_layers_parameters.size() - 1_zu,
                                     tmp_pooling_layer_Parameters.unit_parameters[1u]);
            PRINT_FORMAT("%s: Layer[%zu]: Padding: %zu." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     this->vector_layers_parameters.size() - 1_zu,
                                     tmp_pooling_layer_Parameters.unit_parameters[2u]);
            PRINT_FORMAT("%s: Layer[%zu]: Dilation: %zu." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     this->vector_layers_parameters.size() - 1_zu,
                                     tmp_pooling_layer_Parameters.unit_parameters[3u]);
            PRINT_FORMAT("%s: Layer[%zu]: Ceil mode: %s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     this->vector_layers_parameters.size() - 1_zu,
                                     tmp_pooling_layer_Parameters.unit_parameters[4u] != 0_zu ? "true" : "false");
        }
        //  |END| Pooling layer. |END|
    }
    // |END| #1: Residual, Fully connected, independently recurrent. |END|

    // #2: Fully connected, independently recurrent.
    tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT;
    tmp_Layer_Parameters.unit_parameters[0u] = static_cast<size_t>(tmp_widening_factor_units[0u]);
    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

    PRINT_FORMAT("%s: Layer[%zu]: Type: %s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             this->vector_layers_parameters.size() - 1_zu,
                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
    PRINT_FORMAT("%s: Layer[%zu]: Number neuron unit(s): %zu." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             this->vector_layers_parameters.size() - 1_zu,
                             tmp_Layer_Parameters.unit_parameters[0u]);
    // |END| #2: Fully connected, independently recurrent. |END|

    // Output layer.
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Output layer:" NEW_LINE, MyEA::String::Get__Time().c_str());
    tmp_Layer_Parameters.type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED;
    tmp_Layer_Parameters.unit_parameters[0u] = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ":\tNumber of output(s): ");
    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
    // |END| Output layer. |END|

    return(true);
}

class Neural_Network *Neural_Network_Initializer::Output_Initialize(size_t const maximum_allowable_memory_received) const
{
    if(this->vector_layers_parameters.empty())
    {
        PRINT_FORMAT("%s: %s: ERROR: The vector \"layers parameters\" is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(nullptr);
    }
    
    if(sizeof(class Neural_Network) > maximum_allowable_memory_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(class Neural_Network),
                                 __LINE__);

        return(nullptr);
    }

    class Neural_Network *tmp_ptr_Neural_Network(new class Neural_Network);
    if(tmp_ptr_Neural_Network == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(class Neural_Network),
                                 __LINE__);

        return(nullptr);
    }

    if(tmp_ptr_Neural_Network->Compile(this->vector_layers_parameters.size(),
                                                          this->number_recurrent_depth,
                                                          this->type_neural_network,
                                                          this->vector_layers_parameters.data(),
                                                          maximum_allowable_memory_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Compile(%zu, ptr, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 this->vector_layers_parameters.size(),
                                 maximum_allowable_memory_received,
                                 __LINE__);

        SAFE_DELETE(tmp_ptr_Neural_Network);
    }
    else if(tmp_ptr_Neural_Network->Set__Number_Time_Delays(this->number_time_delays) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Number_Time_Delays(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 this->number_time_delays,
                                 __LINE__);
                
        SAFE_DELETE(tmp_ptr_Neural_Network);
    }

    return(tmp_ptr_Neural_Network);
}

Neural_Network_Initializer::~Neural_Network_Initializer(void) { }

bool Activation_Steepness_Initializer::Input_Initialize(size_t const number_layers_received, enum MyEA::Common::ENUM_TYPE_NETWORKS const type_network_received)
{
    size_t tmp_layer_index;

    if(this->Allocate__Layers_Activation_Steepness(number_layers_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Layers_Activation_Steepness(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_layers_received,
                                 __LINE__);

        return(false);
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Activation steepness initializer:" NEW_LINE, MyEA::String::Get__Time().c_str());

    // Input layer.
    this->ptr_array_value_layers_activation_steepness[0u] = 1_T;
    
    switch(type_network_received)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            // Encoded layer(s).
            for(tmp_layer_index = 1_zu; tmp_layer_index != (number_layers_received - 3_zu) / 2_zu + 1_zu; ++tmp_layer_index)
            {
                this->ptr_array_value_layers_activation_steepness[tmp_layer_index] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                              1_T,
                                                                                                                                                                              MyEA::String::Get__Time() + ": Encoded layer " + std::to_string(tmp_layer_index) + ", activation steepness: ");
            }
            
            // Coded layer.
            this->ptr_array_value_layers_activation_steepness[tmp_layer_index] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                          1_T,
                                                                                                                                                                          MyEA::String::Get__Time() + ": Coded layer, activation steepness: ");
            
            // Decoded layer(s).
            for(++tmp_layer_index; tmp_layer_index != number_layers_received; ++tmp_layer_index)
            {
                this->ptr_array_value_layers_activation_steepness[tmp_layer_index] = this->ptr_array_value_layers_activation_steepness[number_layers_received - tmp_layer_index - 1_zu]; // Subtract coded layer.
            }
                break;
        default:
            // Hidden layer(s).
            for(tmp_layer_index = 1_zu; tmp_layer_index != number_layers_received - 1_zu; ++tmp_layer_index)
            {
                this->ptr_array_value_layers_activation_steepness[tmp_layer_index] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                              1_T,
                                                                                                                                                                              MyEA::String::Get__Time() + ": Hidden layer " + std::to_string(tmp_layer_index) + ", activation steepness: ");
            }
            
            // Output layer.
            this->ptr_array_value_layers_activation_steepness[tmp_layer_index] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                          1_T,
                                                                                                                                                                          MyEA::String::Get__Time() + ": Output layer, activation steepness: ");
                break;
    }
    
    return(true);
}

bool Activation_Steepness_Initializer::Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const
{
    if(this->ptr_array_value_layers_activation_steepness == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_array_value_layers_activation_steepness\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    size_t const tmp_number_layers(MyEA::Math::Minimum<size_t>(this->number_layers, ptr_Neural_Network_received->total_layers));
    size_t tmp_layer_index;

    for(tmp_layer_index = 0_zu; tmp_layer_index != tmp_number_layers; ++tmp_layer_index)
    {
        if(ptr_Neural_Network_received->Set__Layer_Activation_Steepness(tmp_layer_index, this->ptr_array_value_layers_activation_steepness[tmp_layer_index]) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Steepness(%zu, %f)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_layer_index,
                                     Cast_T(this->ptr_array_value_layers_activation_steepness[tmp_layer_index]),
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

void Activation_Steepness_Initializer::Deallocate_Layers_Activation_Steepness(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_value_layers_activation_steepness);
}

bool Activation_Steepness_Initializer::Allocate__Layers_Activation_Steepness(size_t const number_layers_received)
{
    if(this->number_layers == 0_zu)
    {
        if(this->ptr_array_value_layers_activation_steepness == nullptr)
        {
            T_ *tmp_ptr_array_value_layers_activation_steepness(new T_[number_layers_received]);
            if(tmp_ptr_array_value_layers_activation_steepness == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         number_layers_received * sizeof(T_),
                                         __LINE__);

                return(false);
            }
            MEMSET(tmp_ptr_array_value_layers_activation_steepness,
                           0,
                           number_layers_received * sizeof(T_));

            this->ptr_array_value_layers_activation_steepness = tmp_ptr_array_value_layers_activation_steepness;
        }

        this->number_layers = number_layers_received;
    }

    return(true);
}

Activation_Steepness_Initializer::~Activation_Steepness_Initializer(void) { this->Deallocate_Layers_Activation_Steepness(); }

bool Activation_Function_Initializer::Input_Initialize(size_t const number_layers_received, enum MyEA::Common::ENUM_TYPE_NETWORKS const type_network_received)
{
    size_t tmp_layer_index;

    if(this->Allocate__Layers_Activation_Function(number_layers_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Layers_Activation_Function(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_layers_received,
                                 __LINE__);

        return(false);
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Activation function initializer:" NEW_LINE, MyEA::String::Get__Time().c_str());

    PRINT_FORMAT("%s: Activation functions:" NEW_LINE, MyEA::String::Get__Time().c_str());
    for(unsigned int tmp_activation_function_index(1u); tmp_activation_function_index != MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH; ++tmp_activation_function_index)
    {
        PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_activation_function_index,
                                 MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(tmp_activation_function_index)].c_str());
    }
    PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LEAKY_RELU].c_str());
    
    // Input layer.
    this->ptr_array_type_layers_activation_function[0u] = MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR;

    switch(type_network_received)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            // Encoded layer(s).
            for(tmp_layer_index = 1_zu; tmp_layer_index != (number_layers_received - 3_zu) / 2_zu + 1_zu; ++tmp_layer_index)
            {
                if((this->ptr_array_type_layers_activation_function[tmp_layer_index] = static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                                                                                                      MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                                                                                                                                                                                                                                                                                      MyEA::String::Get__Time() + ": Encoded layer " + std::to_string(tmp_layer_index) + ", activation function: "))) >= MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             1u,
                                             MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                             __LINE__);

                    return(false);
                }
            }
            
            // Coded layer.
            if((this->ptr_array_type_layers_activation_function[tmp_layer_index] = static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                                                                                                   MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                                                                                                                                                                                                                                                                                   MyEA::String::Get__Time() + ": Coded layer, activation function: "))) >= MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         1u,
                                         MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                         __LINE__);

                return(false);
            }
            
            // Decoded layer(s).
            for(++tmp_layer_index; tmp_layer_index != number_layers_received; ++tmp_layer_index)
            {
                this->ptr_array_type_layers_activation_function[tmp_layer_index] = this->ptr_array_type_layers_activation_function[number_layers_received - tmp_layer_index - 1_zu]; // Subtract coded layer.
            }
                break;
        default:
            // Hidden layer(s).
            for(tmp_layer_index = 1_zu; tmp_layer_index != number_layers_received - 1_zu; ++tmp_layer_index)
            {
                if((this->ptr_array_type_layers_activation_function[tmp_layer_index] = static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                                                                                                       MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                                                                                                                                                                                                                                                                                       MyEA::String::Get__Time() + ": Hidden layer " + std::to_string(tmp_layer_index) + ", activation function: "))) >= MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             1u,
                                             MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                             __LINE__);

                    return(false);
                }
            }
            
            // Output layer.
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Output layer:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID].c_str());
            
            if((this->ptr_array_type_layers_activation_function[number_layers_received - 1_zu] = static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                                                                                                                        MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                                                                                                                                                                                                                                                                                                        MyEA::String::Get__Time() + ": Output layer, activation function: "))) >= MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         1u,
                                         MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                         __LINE__);

                return(false);
            }
            // |END| Output layer. |END|
                break;
    }
    
    return(true);
}

bool Activation_Function_Initializer::Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const
{
    if(this->ptr_array_type_layers_activation_function == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_array_type_layers_activation_function\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    size_t const tmp_number_layers(MyEA::Math::Minimum<size_t>(this->number_layers, ptr_Neural_Network_received->total_layers));
    size_t tmp_layer_index;

    for(tmp_layer_index = 0_zu; tmp_layer_index != tmp_number_layers; ++tmp_layer_index)
    {
        if(ptr_Neural_Network_received->Set__Layer_Activation_Function(tmp_layer_index, this->ptr_array_type_layers_activation_function[tmp_layer_index]) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function(%zu, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_layer_index,
                                     this->ptr_array_type_layers_activation_function[tmp_layer_index],
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

void Activation_Function_Initializer::Deallocate_Layers_Activation_Function(void) { SAFE_DELETE_ARRAY(this->ptr_array_type_layers_activation_function); }

bool Activation_Function_Initializer::Allocate__Layers_Activation_Function(size_t const number_layers_received)
{
    if(this->number_layers == 0_zu)
    {
        if(this->ptr_array_type_layers_activation_function == nullptr)
        {
            enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION *tmp_ptr_array_type_layers_activation_function(new enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION[number_layers_received]);
            if(tmp_ptr_array_type_layers_activation_function == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            number_layers_received * sizeof(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION),
                                            __LINE__);

                return(false);
            }
            memset(tmp_ptr_array_type_layers_activation_function,
                         0,
                         number_layers_received * sizeof(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION));

            this->ptr_array_type_layers_activation_function = tmp_ptr_array_type_layers_activation_function;
        }

        this->number_layers = number_layers_received;
    }

    return(true);
}

Activation_Function_Initializer::~Activation_Function_Initializer(void) { this->Deallocate_Layers_Activation_Function(); }

bool Loss_Function_Initializer::Input_Initialize(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Loss functions:" NEW_LINE, MyEA::String::Get__Time().c_str());
    for(unsigned int tmp_loss_function_index(1u); tmp_loss_function_index != MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_LENGTH; ++tmp_loss_function_index)
    {
        PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_loss_function_index,
                                 MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS>(tmp_loss_function_index)].c_str());
    }
    PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS_NAMES[MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_RMSE].c_str());
    
    if((this->type_loss_function = static_cast<enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                       MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_LENGTH - 1u,
                                                                                                                                                                                                                       MyEA::String::Get__Time() + ": Choose: "))) >= MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 1u,
                                 MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_LENGTH - 1u,
                                 __LINE__);

        return(false);
    }

    if(this->type_loss_function == MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_BIT)
    {
        PRINT_FORMAT("%s: Loss function, BIT." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0.0, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

        this->bit_fail_limit = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                             1_T,
                                                                                             MyEA::String::Get__Time() + ": Bit fail limit: ");
    }

    return(true);
}

void Loss_Function_Initializer::Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const
{
    ptr_Neural_Network_received->Set__Loss_Function(this->type_loss_function);

    if(this->type_loss_function == MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_BIT) { ptr_Neural_Network_received->Set__Bit_Fail_Limit(this->bit_fail_limit); }
}

bool Accuracy_Function_Initializer::Input_Initialize(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Accuracy functions:" NEW_LINE, MyEA::String::Get__Time().c_str());
    for(unsigned int tmp_type_accuracy_function_index(1u); tmp_type_accuracy_function_index != MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_LENGTH; ++tmp_type_accuracy_function_index)
    {
        PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_type_accuracy_function_index,
                                 MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS>(tmp_type_accuracy_function_index)].c_str());
    }
    PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS_NAMES[MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_DISTANCE].c_str());
    
    if((this->type_accuracy_function = static_cast<enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                                       MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_LENGTH - 1u,
                                                                                                                                                                                                                                       MyEA::String::Get__Time() + ": Choose: "))) >= MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 1u,
                                 MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_LENGTH - 1u,
                                 __LINE__);

        return(false);
    }

    return(true);
}

void Accuracy_Function_Initializer::Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const { ptr_Neural_Network_received->Set__Accuracy_Function(this->type_accuracy_function); }

bool Optimizer_Function_Initializer::Input_Initialize(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Optimizer functions:" NEW_LINE, MyEA::String::Get__Time().c_str());
    for(unsigned int tmp_optimizer_function_index(1u); tmp_optimizer_function_index != MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_LENGTH; ++tmp_optimizer_function_index)
    {
        PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_optimizer_function_index,
                                 MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS>(tmp_optimizer_function_index)].c_str());
    }
    PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad].c_str());
    
    if((this->type_optimizer_function = static_cast<enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                                      MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_LENGTH - 1u,
                                                                                                                                                                                                                                      MyEA::String::Get__Time() + ": Choose: "))) >= MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 1u,
                                 MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_LENGTH - 1u,
                                 __LINE__);

        return(false);
    }

    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Learning rate." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Range[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: default=0.01." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[0u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning rate: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Learning momentum." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Range[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: default=0.9" NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[1u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning momentum: ");

            if(this->values[1u] != 0_T)
            {
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Use Nesterov." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: default=Yes" NEW_LINE, MyEA::String::Get__Time().c_str());
                this->values[2u] = static_cast<T_>(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use Nesterov?")); break;
            }
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Increase factor." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Range[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: default=1.2." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[0u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Increase factor: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Decrease factor." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Range[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: default=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[1u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Decrease factor: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Delta maximum." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Range[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: default=50.0." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[2u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Delta maximum: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Delta minimum." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Range[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: default=1e-6." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[3u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Delta minimum: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Delta zero." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Range[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: default=0.1." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[4u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Delta zero: ");
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad:
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Learning rate." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.001." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[0u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning rate: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Beta1." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.9." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[1u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                               1_T - 1e-7_T,
                                                                                                MyEA::String::Get__Time() + ": Beta1: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Beta2." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.999." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[2u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                               1_T - 1e-7_T,
                                                                                               MyEA::String::Get__Time() + ": Beta2: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Epsilon." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=1e-8." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[3u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Epsilon: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Bias correction." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=true." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[4u] = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Bias correction: ");
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM:
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Learning rate." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.001." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[0u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning rate: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Beta1." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.9." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[1u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                               1_T - 1e-7_T,
                                                                                               MyEA::String::Get__Time() + ": Beta1: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Beta2." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.999." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[2u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                               1_T - 1e-7_T,
                                                                                               MyEA::String::Get__Time() + ": Beta2: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Epsilon." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=1e-8." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[3u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Epsilon: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Bias correction." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=true." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[4u] = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Bias correction: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Gamma." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[1e-7, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.1." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[5u] = MyEA::String::Cin_Real_Number<T_>(1e-7_T, MyEA::String::Get__Time() + ": Gamma: ");
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Learning rate." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.001." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[0u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning rate: ");
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Learning rate, final." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.1." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[1u] = MyEA::String::Cin_Real_Number<T_>(Cast_T(this->values[0u]), MyEA::String::Get__Time() + ": Learning rate, final: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Beta1." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.9." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[2u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                               1_T - 1e-7_T,
                                                                                               MyEA::String::Get__Time() + ": Beta1: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Beta2." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.999." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[3u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                               1_T - 1e-7_T,
                                                                                               MyEA::String::Get__Time() + ": Beta2: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Epsilon." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=1e-8." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[4u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Epsilon: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Bias correction." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=true." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[5u] = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Bias correction: ");
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Gamma." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=1e-3." NEW_LINE, MyEA::String::Get__Time().c_str());
            this->values[6u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                               1_T - 1e-7_T,
                                                                                               MyEA::String::Get__Time() + ": Gamma: ");
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type optimizer function (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->type_optimizer_function,
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str(),
                                     __LINE__);
                return(false);
    }

    if(this->type_optimizer_function != MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus
      &&
      this->type_optimizer_function != MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Weight decay:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0.0, 1.0]. Off = 0." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=1e-5." NEW_LINE, MyEA::String::Get__Time().c_str());
        this->weight_decay = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                1_T,
                                                                                                MyEA::String::Get__Time() + ": Weight decay: ");
    }

    return(true);
}

bool Optimizer_Function_Initializer::Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const
{
    ptr_Neural_Network_received->Set__Optimizer_Function(this->type_optimizer_function);
    
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
            ptr_Neural_Network_received->learning_rate = this->values[0u];

            ptr_Neural_Network_received->learning_momentum = this->values[1u];

            if(ptr_Neural_Network_received->learning_momentum != 0_T && ptr_Neural_Network_received->Allocate__Parameter__Gradient_Descent() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter__Gradient_Descent()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            
            ptr_Neural_Network_received->use_Nesterov = this->values[2u] != 0_T;
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
            ptr_Neural_Network_received->rprop_increase_factor = this->values[0u];

            ptr_Neural_Network_received->rprop_decrease_factor = this->values[1u];

            ptr_Neural_Network_received->rprop_delta_max = this->values[2u];

            ptr_Neural_Network_received->rprop_delta_min = this->values[3u];

            ptr_Neural_Network_received->rprop_delta_zero = this->values[4u];
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad:
            ptr_Neural_Network_received->adam_learning_rate = this->values[0u];

            ptr_Neural_Network_received->adam_beta1 = this->values[1u];

            ptr_Neural_Network_received->adam_beta2 = this->values[2u];

            ptr_Neural_Network_received->adam_epsilon = this->values[3u];

            ptr_Neural_Network_received->use_adam_bias_correction = this->values[4u] != 0_T;
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM:
            ptr_Neural_Network_received->adam_learning_rate = this->values[0u];

            ptr_Neural_Network_received->adam_beta1 = this->values[1u];

            ptr_Neural_Network_received->adam_beta2 = this->values[2u];

            ptr_Neural_Network_received->adam_epsilon = this->values[3u];

            ptr_Neural_Network_received->use_adam_bias_correction = this->values[4u] != 0_T;

            ptr_Neural_Network_received->adam_gamma = this->values[5u];
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
            ptr_Neural_Network_received->adam_learning_rate = this->values[0u];

            ptr_Neural_Network_received->learning_rate_final = this->values[1u];

            ptr_Neural_Network_received->adam_beta1 = this->values[2u];

            ptr_Neural_Network_received->adam_beta2 = this->values[3u];

            ptr_Neural_Network_received->adam_epsilon = this->values[4u];

            ptr_Neural_Network_received->use_adam_bias_correction = this->values[5u] != 0_T;

            ptr_Neural_Network_received->learning_gamma = this->values[6u];
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type optimizer function (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->type_optimizer_function,
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str(),
                                     __LINE__);
                return(false);
    }

    if(ptr_Neural_Network_received->Set__Regularization__Weight_Decay(this->weight_decay) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Weight_Decay(%f)\" function. At line %d." NEW_LINE,
                                MyEA::String::Get__Time().c_str(),
                                __FUNCTION__,
                                Cast_T(this->weight_decay),
                                __LINE__);

        return(false);
    }

#if defined(COMPILE_CUDA)
    if(ptr_Neural_Network_received->is_device_initialized)
    { ptr_Neural_Network_received->ptr_device_Neural_Network->Copy__Optimizer_Parameters(ptr_Neural_Network_received); }
#endif

    return(true);
}

void Warm_Restarts_Initializer::Input_Initialize(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Warm restarts:" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=Yes." NEW_LINE, MyEA::String::Get__Time().c_str());
    this->use_Warm_Restarts = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use warm restarts: ");

    if(this->use_Warm_Restarts)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Learning rate, decay:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[1e-5, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=0.95." NEW_LINE, MyEA::String::Get__Time().c_str());
        this->warm_restarts_decay_learning_rate = MyEA::String::Cin_Real_Number<T_>(1e-5_T,
                                                                                                                                1_T,
                                                                                                                                MyEA::String::Get__Time() + ": Learning rate, decay: ");
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Maximum learning rate:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0.0, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::String::Get__Time().c_str());
        this->warm_restarts_maximum_learning_rate = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                     1_T,
                                                                                                                                     MyEA::String::Get__Time() + ": Maximum learning rate: ");
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Minimum learning rate:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0.0, %f]." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->warm_restarts_maximum_learning_rate));
        PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());
        this->warm_restarts_minimum_learning_rate = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                    this->warm_restarts_maximum_learning_rate,
                                                                                                                                    MyEA::String::Get__Time() + ": Minimum learning rate: ");
        if(this->warm_restarts_minimum_learning_rate == 0_T) { this->warm_restarts_minimum_learning_rate = this->warm_restarts_maximum_learning_rate / 1e+7_T; }

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Initial Ti:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::String::Get__Time().c_str());
        this->warm_restarts_initial_T_i = static_cast<T_>(MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Initial Ti: "));
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Warm restarts multiplier:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=2." NEW_LINE, MyEA::String::Get__Time().c_str());
        this->warm_restarts_multiplier = static_cast<T_>(MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Warm restarts multiplier: "));
    }
}

bool Warm_Restarts_Initializer::Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const
{
    ptr_Neural_Network_received->use_Warm_Restarts = this->use_Warm_Restarts;

    if(this->use_Warm_Restarts)
    {
        ptr_Neural_Network_received->warm_restarts_decay_learning_rate = this->warm_restarts_decay_learning_rate;
        ptr_Neural_Network_received->warm_restarts_maximum_learning_rate = ptr_Neural_Network_received->warm_restarts_initial_maximum_learning_rate = this->warm_restarts_maximum_learning_rate;
        ptr_Neural_Network_received->warm_restarts_minimum_learning_rate = this->warm_restarts_minimum_learning_rate;
        ptr_Neural_Network_received->warm_restarts_T_i = ptr_Neural_Network_received->warm_restarts_initial_T_i = this->warm_restarts_initial_T_i;
        ptr_Neural_Network_received->warm_restarts_multiplier = this->warm_restarts_multiplier;
    }

#if defined(COMPILE_CUDA)
    if(ptr_Neural_Network_received->is_device_initialized)
    { ptr_Neural_Network_received->ptr_device_Neural_Network->Copy__Warm_Restarts_Parameters(ptr_Neural_Network_received); }
#endif

    return(true);
}

bool Weights_Initializer::Input_Initialize(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Weights initializer:" NEW_LINE, MyEA::String::Get__Time().c_str());
    for(unsigned int tmp_weights_initializer_type_index(1u); tmp_weights_initializer_type_index != MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_LENGTH; ++tmp_weights_initializer_type_index)
    {
        PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_weights_initializer_type_index,
                                 MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS>(tmp_weights_initializer_type_index)].c_str());
    }
    PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS_NAMES[MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_ORTHOGONAL].c_str());
    
    if((this->type_weights_initializer = static_cast<enum MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                                     MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_LENGTH - 1u,
                                                                                                                                                                                                                                     MyEA::String::Get__Time() + ": Type: "))) >= MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 1u,
                                 MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_LENGTH - 1u,
                                 __LINE__);

        return(false);
    }

    switch(this->type_weights_initializer)
    {
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_GLOROT_GAUSSIAN:
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_GLOROT_UNIFORM:
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_IDENTITY:
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_ORTHOGONAL:
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Initial bias:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.0." NEW_LINE, MyEA::String::Get__Time().c_str());
            
            this->initial_bias = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                1_T,
                                                                                                MyEA::String::Get__Time() + ": Initial bias: ");
                break;
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_LSUV:
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Initial bias:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.0." NEW_LINE, MyEA::String::Get__Time().c_str());
            
            this->initial_bias = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                1_T,
                                                                                                MyEA::String::Get__Time() + ": Initial bias: ");
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Maximum number of tials:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=10." NEW_LINE, MyEA::String::Get__Time().c_str());

            this->values[0u] = static_cast<T_>(MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Maximum number trials: "));
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Maximum batch size:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[1, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=32." NEW_LINE, MyEA::String::Get__Time().c_str());

            this->values[1u] = static_cast<T_>(MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": Maximum batch size: "));

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Variance target:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=1.0." NEW_LINE, MyEA::String::Get__Time().c_str());

            this->values[2u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Variance target: ");
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Variance tolerance:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.01." NEW_LINE, MyEA::String::Get__Time().c_str());

            this->values[3u] = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Variance tolerance: ");
                break;
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_UNIFORM:
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Initial bias:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0.0, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=0.0." NEW_LINE, MyEA::String::Get__Time().c_str());
            
            this->initial_bias = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                1_T,
                                                                                                MyEA::String::Get__Time() + ": Initial bias: ");

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Lower bound:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[-1.0, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=-1.0." NEW_LINE, MyEA::String::Get__Time().c_str());

            this->values[0u] = MyEA::String::Cin_Real_Number<T_>(-1_T,
                                                                                               1_T,
                                                                                               MyEA::String::Get__Time() + ": Lower bound: ");
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Upper bound:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[%f, 1.0]." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     Cast_T(this->values[0u]));
            PRINT_FORMAT("%s:\tdefault=1.0." NEW_LINE, MyEA::String::Get__Time().c_str());

            this->values[1u] = MyEA::String::Cin_Real_Number<T_>(this->values[0u],
                                                                                               1_T,
                                                                                               MyEA::String::Get__Time() + ": Upper bound: ");
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type weights initializer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->type_weights_initializer,
                                     MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS_NAMES[this->type_weights_initializer].c_str(),
                                     __LINE__);
                return(false);
    }

    return(true);
}

bool Weights_Initializer::Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const
{
    switch(this->type_weights_initializer)
    {
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_GLOROT_GAUSSIAN: ptr_Neural_Network_received->Initialization__Glorot__Gaussian(this->initial_bias); break;
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_GLOROT_UNIFORM: ptr_Neural_Network_received->Initialization__Glorot__Uniform(this->initial_bias); break;
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_IDENTITY: ptr_Neural_Network_received->Initialization__Identity(this->initial_bias); break;
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_LSUV:
            if(ptr_Neural_Network_received->Initialize__LSUV(static_cast<size_t>(this->values[0u]),
                                                                                    static_cast<size_t>(this->values[1u]),
                                                                                    this->initial_bias,
                                                                                    this->values[2u],
                                                                                    this->values[3u]) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__LSUV(%zu, %zu, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         static_cast<size_t>(this->values[0u]),
                                         static_cast<size_t>(this->values[1u]),
                                         Cast_T(this->initial_bias),
                                         Cast_T(this->values[2u]),
                                         Cast_T(this->values[3u]),
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_ORTHOGONAL: ptr_Neural_Network_received->Initialization__Orthogonal(false, this->initial_bias); break;
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_UNIFORM:
            ptr_Neural_Network_received->Initialization__Uniform(this->initial_bias,
                                                                                         this->values[0u],
                                                                                         this->values[1u]);
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type weights initializer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->type_weights_initializer,
                                     MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS_NAMES[this->type_weights_initializer].c_str(),
                                     __LINE__);
                return(false);
    }

#if defined(COMPILE_CUDA)
    if(ptr_Neural_Network_received->is_device_initialized)
    { ptr_Neural_Network_received->Copy__Parameters__Host_To_Device(); }
#endif

    return(true);
}

bool Dropout_Initializer::Input_Initialize(size_t const number_layers_received, enum MyEA::Common::ENUM_TYPE_NETWORKS const type_network_received)
{
    bool tmp_use_dropout(false);

    unsigned int tmp_type_dropout_layer_index;

    size_t const tmp_option_end(type_network_received == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
                                               ?
                                               number_layers_received / 2_zu + 1_zu
                                               :
                                               number_layers_received - 1_zu);
    size_t tmp_option,
              tmp_layer_index;
    
    std::string tmp_layer_name;
    
    if(this->Allocate__Layers_Using_Dropout(number_layers_received - 1_zu) == false) // Subtract output layer.
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Layers_Using_Dropout(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_layers_received - 1_zu,
                                 __LINE__);

        return(false);
    }
    
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Dropout initializer:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Input layer: (%f, %f), %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->ptr_array_layers_dropout_array_values[0u][0u]),
                                 Cast_T(this->ptr_array_layers_dropout_array_values[0u][1u]),
                                 MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[this->ptr_array_layers_type_dropout[0u]].c_str());
        for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_option_end; ++tmp_layer_index)
        {
            PRINT_FORMAT("%s:\t[%zu]: Hidden layer[%zu]: (%f, %f), %s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_layer_index,
                                     tmp_layer_index - 1_zu,
                                     Cast_T(this->ptr_array_layers_dropout_array_values[tmp_layer_index][0u]),
                                     Cast_T(this->ptr_array_layers_dropout_array_values[tmp_layer_index][1u]),
                                     MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[this->ptr_array_layers_type_dropout[tmp_layer_index]].c_str());
        }
        PRINT_FORMAT("%s:\t[%zu]: Quit." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_option_end);

        tmp_option = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                 tmp_option_end,
                                                                                 MyEA::String::Get__Time() + ": Option: ");

        if(tmp_option < tmp_option_end)
        {
            tmp_layer_name = tmp_option == 0_zu ? "Input" : "Hidden[" + std::to_string(tmp_option - 1_zu) + "]";
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Dropout layer:" NEW_LINE, MyEA::String::Get__Time().c_str());
            for(tmp_type_dropout_layer_index = 0u; tmp_type_dropout_layer_index != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_LENGTH; ++tmp_type_dropout_layer_index)
            {
                PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_type_dropout_layer_index,
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT>(tmp_type_dropout_layer_index)].c_str());
            }
            PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI].c_str());
            
            switch((this->ptr_array_layers_type_dropout[tmp_option] = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT>(MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                                                                                                                                                                                MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_LENGTH - 1u,
                                                                                                                                                                                                                                                                MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, type: "))))
            {
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE:
                    this->ptr_array_layers_dropout_array_values[tmp_option][0u] = 0_T;
                    this->ptr_array_layers_dropout_array_values[tmp_option][1u] = 0_T;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Alpha dropout: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    this->ptr_array_layers_dropout_array_values[tmp_option][0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                       1_T,
                                                                                                                                                                       MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability: ");

                    this->ptr_array_layers_dropout_array_values[tmp_option][1u] = 0_T;

                    if(this->ptr_array_layers_dropout_array_values[tmp_option][0u] != 0_T) { tmp_use_dropout = true; }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Dropout bernoulli: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    this->ptr_array_layers_dropout_array_values[tmp_option][0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                       1_T,
                                                                                                                                                                       MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, retention probability: ");

                    this->ptr_array_layers_dropout_array_values[tmp_option][1u] = 0_T;

                    if(this->ptr_array_layers_dropout_array_values[tmp_option][0u] != 1_T) { tmp_use_dropout = true; }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Dropout bernoulli inverted: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    this->ptr_array_layers_dropout_array_values[tmp_option][0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                       1_T,
                                                                                                                                                                       MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, retention probability: ");

                    this->ptr_array_layers_dropout_array_values[tmp_option][1u] = 0_T;

                    if(this->ptr_array_layers_dropout_array_values[tmp_option][0u] != 1_T) { tmp_use_dropout = true; }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Dropout gaussian: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    this->ptr_array_layers_dropout_array_values[tmp_option][0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                       1_T,
                                                                                                                                                                       MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability: ");

                    this->ptr_array_layers_dropout_array_values[tmp_option][1u] = 0_T;

                    if(this->ptr_array_layers_dropout_array_values[tmp_option][0u] != 0_T) { tmp_use_dropout = true; }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Dropout ShakeDrop: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    this->ptr_array_layers_dropout_array_values[tmp_option][0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                       1_T,
                                                                                                                                                                       MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability: ");

                    this->ptr_array_layers_dropout_array_values[tmp_option][1u] = 0_T;

                    if(this->ptr_array_layers_dropout_array_values[tmp_option][0u] != 0_T) { tmp_use_dropout = true; }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Dropout Uout: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    this->ptr_array_layers_dropout_array_values[tmp_option][0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                       1_T,
                                                                                                                                                                       MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability: ");

                    this->ptr_array_layers_dropout_array_values[tmp_option][1u] = 0_T;

                    if(this->ptr_array_layers_dropout_array_values[tmp_option][0u] != 0_T) { tmp_use_dropout = true; }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Zoneout cell: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    this->ptr_array_layers_dropout_array_values[tmp_option][0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                       1_T,
                                                                                                                                                                       MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, zoneout cell probability: ");
                    
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Zoneout hidden: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.05." NEW_LINE, MyEA::String::Get__Time().c_str());

                    this->ptr_array_layers_dropout_array_values[tmp_option][1u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                       1_T,
                                                                                                                                                                       MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, zoneout hidden probability: ");
                        
                    if(this->ptr_array_layers_dropout_array_values[tmp_option][0u] != 0_T || this->ptr_array_layers_dropout_array_values[tmp_option][1u] != 0_T) { tmp_use_dropout = true; }
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Type dropout layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             this->ptr_array_layers_type_dropout[tmp_option],
                                             MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[this->ptr_array_layers_type_dropout[tmp_option]].c_str(),
                                             __LINE__);
                        return(false);
            }

            
            if(type_network_received == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
              &&
              this->ptr_array_layers_type_dropout[tmp_option] != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE
              &&
              tmp_option != 0_zu)
            { this->ptr_array_layers_use_coded_dropout[tmp_option] = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Pre-training: Use dropout inside the coded layer?"); }
        }
        else if(tmp_option == tmp_option_end) { return(true); }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<size_t>(%zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     0_zu,
                                     tmp_option_end,
                                     __LINE__);
        }
    }

    if(tmp_use_dropout == false) { this->Deallocate__Layers_Using_Dropout(); }

    return(true);
}

bool Dropout_Initializer::Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const
{
    if(this->ptr_array_layers_dropout_array_values != nullptr && this->ptr_array_layers_type_dropout != nullptr)
    {
        size_t const tmp_number_layers(MyEA::Math::Minimum<size_t>(this->number_layers, ptr_Neural_Network_received->total_layers - 1_zu)); // Subtract output layer.
        size_t tmp_layer_index;

        for(tmp_layer_index = 0_zu; tmp_layer_index != tmp_number_layers; ++tmp_layer_index)
        {
            if(ptr_Neural_Network_received->Set__Dropout(tmp_layer_index,
                                                                                this->ptr_array_layers_type_dropout[tmp_layer_index],
                                                                                this->ptr_array_layers_dropout_array_values[tmp_layer_index]) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_layer_index,
                                         this->ptr_array_layers_type_dropout[tmp_layer_index],
                                         Cast_T(this->ptr_array_layers_dropout_array_values[tmp_layer_index][0u]),
                                         Cast_T(this->ptr_array_layers_dropout_array_values[tmp_layer_index][1u]),
                                         __LINE__);

                return(false);
            }

            ptr_Neural_Network_received->ptr_array_layers[tmp_layer_index].use_coded_dropout = this->ptr_array_layers_use_coded_dropout[tmp_layer_index];
        }
    }

    return(true);
}

void Dropout_Initializer::Deallocate__Layers_Using_Dropout(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_layers_use_coded_dropout);

    if(this->ptr_array_layers_dropout_array_values != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_layers_dropout_array_values[0u]);
        SAFE_DELETE_ARRAY(this->ptr_array_layers_dropout_array_values);
    }

    SAFE_DELETE_ARRAY(this->ptr_array_layers_type_dropout);
}

bool Dropout_Initializer::Allocate__Layers_Using_Dropout(size_t const number_layers_received)
{
    bool *tmp_ptr_array_layers_use_coded_dropout(new bool[number_layers_received]);
    if(tmp_ptr_array_layers_use_coded_dropout == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_layers_received * sizeof(bool),
                                 __LINE__);

        return(false);
    }
    Memory::Fill<bool>(tmp_ptr_array_layers_use_coded_dropout,
                                 tmp_ptr_array_layers_use_coded_dropout + number_layers_received,
                                 false);
    this->ptr_array_layers_use_coded_dropout = tmp_ptr_array_layers_use_coded_dropout;

    // Dropout value.
    T_ **tmp_ptr_array_layers_dropout_array_values(new T_*[number_layers_received]);
    if(tmp_ptr_array_layers_dropout_array_values == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_layers_received * sizeof(T_*),
                                 __LINE__);

        return(false);
    }
    Memory::Fill_Nullptr<T_*>(tmp_ptr_array_layers_dropout_array_values, tmp_ptr_array_layers_dropout_array_values + number_layers_received);

    this->ptr_array_layers_dropout_array_values = tmp_ptr_array_layers_dropout_array_values;

    T_ *tmp_ptr_array_values(new T_[number_layers_received * 2_zu]);
    if(tmp_ptr_array_values == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_layers_received * 2_zu * sizeof(T_),
                                 __LINE__);

        return(false);
    }
    MEMSET(tmp_ptr_array_values,
                  0,
                  number_layers_received * 2_zu * sizeof(T_));

    for(size_t tmp_index(0_zu); tmp_index != number_layers_received; ++tmp_index)
    { this->ptr_array_layers_dropout_array_values[tmp_index] = tmp_ptr_array_values + tmp_index * 2_zu; }
    // |END| Dropout value. |END|
    
    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT *tmp_ptr_array_layers_type_dropout(new enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT[number_layers_received]);
    if(tmp_ptr_array_layers_type_dropout == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_layers_received * sizeof(enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT),
                                 __LINE__);

        return(false);
    }
    Memory::Fill<MyEA::Common::ENUM_TYPE_LAYER_DROPOUT>(tmp_ptr_array_layers_type_dropout,
                                                                                                       tmp_ptr_array_layers_type_dropout + number_layers_received,
                                                                                                       MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE);
    this->ptr_array_layers_type_dropout = tmp_ptr_array_layers_type_dropout;

    this->number_layers = number_layers_received;

    return(true);
}

Dropout_Initializer::~Dropout_Initializer(void) { this->Deallocate__Layers_Using_Dropout(); }

bool Normalization_Initializer::Input_Initialize(size_t const number_layers_received,
                                                                 size_t const number_batch_received,
                                                                 enum MyEA::Common::ENUM_TYPE_NETWORKS const type_network_received)
{
    bool tmp_use_batch_normalization(false),
           tmp_use_batch_renormalization(false);

    unsigned int tmp_type_normalization_layer_index;
    
    size_t const tmp_option_end(type_network_received == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
                                               ?
                                               number_layers_received / 2_zu + 1_zu
                                               :
                                               number_layers_received - 1_zu);
    size_t tmp_option,
              tmp_layer_index;

    if(this->Allocate__Layers_Using_Normalization(number_layers_received - 1_zu) == false) // Subtract output layer.
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Layers_Using_Normalization(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_layers_received - 1_zu,
                                 __LINE__);

        return(false);
    }

    this->ptr_array_layers_using_normalization[0u] = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Normalization initializer:" NEW_LINE, MyEA::String::Get__Time().c_str());
        for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_option_end; ++tmp_layer_index)
        {
            PRINT_FORMAT("%s:\t[%zu] Hidden layer[%zu]: %s, %s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_layer_index - 1_zu,
                                     tmp_layer_index - 1_zu,
                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[this->ptr_array_layers_using_normalization[tmp_layer_index]].c_str(),
                                     this->ptr_array_layers_normalization_before_activation[tmp_layer_index] ? "true" : "false");
        }
        PRINT_FORMAT("%s:\t[%zu]: Quit." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_option_end - 1_zu);

        tmp_option = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                 tmp_option_end - 1_zu,
                                                                                 MyEA::String::Get__Time() + ": Option: ") + 1_zu;

        if(tmp_option < tmp_option_end)
        {
            tmp_layer_index = tmp_option;

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Layer normalization:" NEW_LINE, MyEA::String::Get__Time().c_str());
            for(tmp_type_normalization_layer_index = 0u; tmp_type_normalization_layer_index != MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_LENGTH; ++tmp_type_normalization_layer_index)
            {
                PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_type_normalization_layer_index,
                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION>(tmp_type_normalization_layer_index)].c_str());
            }
            PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION].c_str());
            
            if((this->ptr_array_layers_using_normalization[tmp_layer_index] = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION>(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                                                                                                                                                                                                                            MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_LENGTH - 1u,
                                                                                                                                                                                                                                                                                            MyEA::String::Get__Time() + ": Hidden layer " + std::to_string(tmp_layer_index) + ", type: "))) >= MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_LENGTH)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         1u,
                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_LENGTH - 1u,
                                         __LINE__);

                return(false);
            }
            
            if(this->ptr_array_layers_using_normalization[tmp_layer_index] != MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE)
            {
                this->ptr_array_layers_normalization_before_activation[tmp_layer_index] = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Hidden layer " + std::to_string(tmp_layer_index) + ", use normalization before activation?");
            }

            switch(this->ptr_array_layers_using_normalization[tmp_layer_index])
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION: tmp_use_batch_normalization = true; break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION: tmp_use_batch_renormalization = true; break;
                default: break;
            }
        }
        else if(tmp_option == tmp_option_end) { return(true); }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<size_t>(%zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     0_zu,
                                     tmp_option_end - 1_zu,
                                     __LINE__);
        }
    }

    if(tmp_use_batch_normalization || tmp_use_batch_renormalization)
    {
        // Normalization parameter.
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Momentum average:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0.0, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=%.9f." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(number_batch_received <= 1_zu ? 0.999_T : 1_T / static_cast<T_>(number_batch_received)));

        this->normalization_momentum_average = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                              1_T,
                                                                                                                              MyEA::String::Get__Time() + ": Momentum average: ");

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Epsilon:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=1e-5." NEW_LINE, MyEA::String::Get__Time().c_str());

        this->normalization_epsilon = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Epsilon: ");
        // |END| Normalization parameter. |END|
    }

    if(tmp_use_batch_renormalization)
    {
        // Batch renormalization parameter.
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: r correction maximum:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::String::Get__Time().c_str());

        this->batch_renormalization_r_correction_maximum = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": r correction maximum: ");

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: d correction maximum:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());

        this->batch_renormalization_d_correction_maximum = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": d correction maximum: ");
        // |END| Batch renormalization parameter. |END|
    }
    
    if(tmp_use_batch_normalization == false && tmp_use_batch_renormalization == false) { this->Deallocate__Layers_Using_Normalization(); }

    return(true);
}

bool Normalization_Initializer::Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const
{
    if(this->ptr_array_layers_using_normalization != nullptr)
    {
        bool tmp_use_normalization(false),
               tmp_use_renormalization(false);
        
        size_t const tmp_number_layers(MyEA::Math::Minimum<size_t>(this->number_layers, ptr_Neural_Network_received->total_layers - 1_zu)); // Subtract output layer.
        size_t tmp_layer_index;
        
        for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_number_layers; ++tmp_layer_index)
        {
            if(ptr_Neural_Network_received->Set__Layer_Normalization(tmp_layer_index, this->ptr_array_layers_using_normalization[tmp_layer_index]) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization(%zu, %u)\" function. At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_layer_index,
                                        this->ptr_array_layers_using_normalization[tmp_layer_index],
                                        __LINE__);

                return(false);
            }

            switch(this->ptr_array_layers_using_normalization[tmp_layer_index])
            {
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION: tmp_use_normalization = true; break;
                case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION: tmp_use_renormalization = true; break;
                default: break;
            }
            
            ptr_Neural_Network_received->ptr_array_layers[tmp_layer_index].use_layer_normalization_before_activation = this->ptr_array_layers_normalization_before_activation[tmp_layer_index];
        }

        if(tmp_use_normalization || tmp_use_renormalization)
        {
            // Normalization parameter.
            if(ptr_Neural_Network_received->Set__Normalization_Momentum_Average(this->normalization_momentum_average) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_Momentum_Average(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(this->normalization_momentum_average),
                                         __LINE__);

                return(false);
            }
            
            if(ptr_Neural_Network_received->Set__Normalization_Epsilon(this->normalization_epsilon) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_Epsilon(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(this->normalization_epsilon),
                                         __LINE__);

                return(false);
            }
            // |END| Normalization parameter. |END|
        }

        if(tmp_use_renormalization)
        {
            // Batch renormalization parameter.
            if(ptr_Neural_Network_received->Set__Batch_Renormalization_r_Correction_Maximum(this->batch_renormalization_r_correction_maximum) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Batch_Renormalization_r_Correction_Maximum(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(this->batch_renormalization_r_correction_maximum),
                                         __LINE__);

                return(false);
            }
            
            if(ptr_Neural_Network_received->Set__Batch_Renormalization_d_Correction_Maximum(this->batch_renormalization_d_correction_maximum) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Batch_Renormalization_d_Correction_Maximum(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(this->batch_renormalization_d_correction_maximum),
                                         __LINE__);

                return(false);
            }
            // |END| Batch renormalization parameter. |END|
        }
    }

    return(true);
}

void Normalization_Initializer::Deallocate__Layers_Using_Normalization(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_layers_normalization_before_activation);

    SAFE_DELETE_ARRAY(this->ptr_array_layers_using_normalization);
}

bool Normalization_Initializer::Allocate__Layers_Using_Normalization(size_t const number_layers_received)
{
    bool *tmp_ptr_array_layers_normalization_before_activation(new bool[number_layers_received]);
    if(tmp_ptr_array_layers_normalization_before_activation == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    number_layers_received * sizeof(bool),
                                    __LINE__);

        return(false);
    }
    memset(tmp_ptr_array_layers_normalization_before_activation,
                 0,
                 number_layers_received * sizeof(bool));
    this->ptr_array_layers_normalization_before_activation = tmp_ptr_array_layers_normalization_before_activation;

    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION *tmp_ptr_array_layers_using_normalization(new enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION[number_layers_received]);
    if(tmp_ptr_array_layers_using_normalization == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    number_layers_received * sizeof(enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION),
                                    __LINE__);

        return(false);
    }
    memset(tmp_ptr_array_layers_using_normalization,
                 0,
                 number_layers_received * sizeof(enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION));
    this->ptr_array_layers_using_normalization = tmp_ptr_array_layers_using_normalization;

    this->number_layers = number_layers_received;

    return(true);
}

Normalization_Initializer::~Normalization_Initializer(void) { this->Deallocate__Layers_Using_Normalization(); }

#if defined(COMPILE_CUDA)
bool Neural_Network::Set__CUDA(bool const use_cuda_received, size_t const maximum_allowable_memory_received)
{
    if((this->use_CUDA == false && use_cuda_received)
      ||
      (this->use_CUDA && use_cuda_received && this->is_device_initialized == false))
    {
        if(this->Initialize__CUDA(maximum_allowable_memory_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__CUDA(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     maximum_allowable_memory_received,
                                     __LINE__);

            return(false);
        }
    }
    else if((this->use_CUDA && use_cuda_received == false)
              ||
              (this->use_CUDA == false && use_cuda_received == false && this->is_device_initialized))
    {
        if(this->Deinitialize__CUDA() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Deinitialize__CUDA()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    this->use_CUDA = use_cuda_received;

    return(true);
}

bool Neural_Network::Initialize__CUDA(size_t const maximum_allowable_memory_received)
{
    if(this->is_device_initialized == false)
    {
        CUDA__Safe_Call(cudaMalloc((void**)&this->ptr_device_Neural_Network, sizeof(class CUDA_Neural_Network)));

        if(this->ptr_device_Neural_Network->Copy__Host_To_Device(this, maximum_allowable_memory_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Host_To_Device(ptr, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     maximum_allowable_memory_received,
                                     __LINE__);
            
            CUDA__Safe_Call(cudaFree(this->ptr_device_Neural_Network));
            
            return(false);
        }
        
        this->is_device_initialized = true;
    }

    return(true);
}

bool Neural_Network::Initialize__CUDA__Thread(class Dataset_Manager<T_> const *const ptr_Dataset_Manager_received)
{
    if(this->is_device_initialized == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: Device not initialized. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_number_examples_training(ptr_Dataset_Manager_received->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Number_Examples()),
                       tmp_number_examples_validation(ptr_Dataset_Manager_received->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Get__Number_Examples()),
                       tmp_number_examples_testing(ptr_Dataset_Manager_received->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Get__Number_Examples());
    size_t tmp_number_examples_max(0_zu);

    tmp_number_examples_max = MyEA::Math::Maximum<size_t>(tmp_number_examples_max, tmp_number_examples_training);
    tmp_number_examples_max = MyEA::Math::Maximum<size_t>(tmp_number_examples_max, tmp_number_examples_validation);
    tmp_number_examples_max = MyEA::Math::Maximum<size_t>(tmp_number_examples_max, tmp_number_examples_testing);

    PRINT_FORMAT("%s: GPU: Neural network: Update threads size" NEW_LINE, MyEA::String::Get__Time().c_str());
    if(this->ptr_device_Neural_Network->Update__Thread_Size(tmp_number_examples_max) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_number_examples_max,
                                 __LINE__);
        
        if(this->Deinitialize__CUDA() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Deinitialize__CUDA()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
        
    PRINT_FORMAT("%s: GPU: Neural network: Update batch size" NEW_LINE, MyEA::String::Get__Time().c_str());
    if(this->ptr_device_Neural_Network->Update__Batch_Size(tmp_number_examples_max) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_number_examples_max,
                                 __LINE__);
        
        if(this->Deinitialize__CUDA() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Deinitialize__CUDA()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }

    PRINT_FORMAT("%s: GPU: Neural network: Setting up limit device runtime pending launch count." NEW_LINE, MyEA::String::Get__Time().c_str());
    this->ptr_device_Neural_Network->Set__Limit_Device_Runtime_Pending_Launch_Count();

    return(true);
}

bool Neural_Network::Deinitialize__CUDA(void)
{
    if(this->is_device_initialized)
    {
        if(this->ptr_device_Neural_Network->Deallocate() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Deallocate()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);
            
            CUDA__Safe_Call(cudaFree(this->ptr_device_Neural_Network));
            
            return(false);
        }
        
        CUDA__Safe_Call(cudaFree(this->ptr_device_Neural_Network));
        
        this->is_device_initialized = false;
    }

    return(true);
}

bool Neural_Network::Use__CUDA(void) const { return(this->use_CUDA); }
#endif

bool Neural_Network::Use__Clip_Gradient(void) const { return(this->use_clip_gradient); }

bool Neural_Network::Use__Regularization_Parameter(void) const
{
    if(this->regularization__l1 != 0_T
      ||
      this->regularization__l2 != 0_T
      ||
      this->regularization__srip != 0_T
      ||
      this->regularization__weight_decay != 0_T)
    { return(true); }
    
    return(false);
}

bool Neural_Network::Use__Normalization(void) const { return(this->total_batch_normalization_layers + this->total_batch_renormalization_layers + this->total_ghost_batch_normalization_layers + this->total_streaming_normalization_layers != 0_zu); }

bool Neural_Network::Use__Batch_Normalization(void) const { return(this->total_batch_normalization_layers != 0_zu); }

bool Neural_Network::Use__Batch_Renormalization(void) const { return(this->total_batch_renormalization_layers != 0_zu); }

bool Neural_Network::Use__Ghost_Batch_Normalization(void) const { return(this->total_ghost_batch_normalization_layers != 0_zu); }

bool Neural_Network::Use__Streaming_Normalization(void) const { return(this->total_streaming_normalization_layers != 0_zu); }

bool Neural_Network::Use__Dropout__Alpha(void) const { return(this->total_dropout_alpha_layers != 0_zu); }

bool Neural_Network::Use__Dropout__Bernoulli(void) const { return(this->total_dropout_bernoulli_layers != 0_zu); }

bool Neural_Network::Use__Dropout__Bernoulli__Inverted(void) const { return(this->total_dropout_bernoulli_inverted_layers != 0_zu); }

bool Neural_Network::Use__Dropout__Gaussian(void) const { return(this->total_dropout_gaussian_layers != 0_zu); }

bool Neural_Network::Use__Dropout__ShakeDrop(void) const { return(this->total_dropout_shakedrop_layers != 0_zu); }

bool Neural_Network::Use__Dropout__Uout(void) const { return(this->total_dropout_uout_layers != 0_zu); }

bool Neural_Network::Use__Dropout__Zoneout(void) const { return(this->total_dropout_zoneout_layers != 0_zu); }

bool Neural_Network::Use__K_Sparse(void) const { return(this->total_k_sparse_layers != 0_zu); }

bool Neural_Network::Use__Tied_Parameter(void) const { return(this->total_tied_parameter_layers != 0_zu); }

bool Neural_Network::Use__Regularization__Constraint_Recurrent_Weight(void) const { return(this->total_constraint_recurrent_weight_layers != 0_zu); }

bool Neural_Network::Use__Multi_Label(void) const { return(this->use_multi_label); }

bool Neural_Network::Set__Multi_Label(bool const use_multi_label_received)
{
    if(this->use_multi_label == use_multi_label_received) { return(true); }
    else if(this->number_outputs == 1u && use_multi_label_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not use multi label with only one output. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    this->use_multi_label = use_multi_label_received;

    return(true);
}

bool Neural_Network::Set__Input_Mode(bool const use_first_layer_as_input_received)
{
    if(this->use_first_layer_as_input == use_first_layer_as_input_received) { return(true); }

    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            if(use_first_layer_as_input_received == false && this->use_last_layer_as_output == use_first_layer_as_input_received)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not use the decoded layer has input. The decoded layer is the output. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            this->use_first_layer_as_input = use_first_layer_as_input_received;
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Network type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->type_network,
                                     MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[this->type_network].c_str(),
                                     __LINE__);
                return(false);
    }

    return(true);
}

bool Neural_Network::Set__Output_Mode(bool const use_last_layer_as_output_received)
{
    if(this->use_last_layer_as_output == use_last_layer_as_output_received) { return(true); }

    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            if(use_last_layer_as_output_received == false && this->use_first_layer_as_input == use_last_layer_as_output_received)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not use the decoded layer has output. The decoded layer is the input. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            this->use_last_layer_as_output = use_last_layer_as_output_received;
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Network type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->type_network,
                                     MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[this->type_network].c_str(),
                                     __LINE__);

                return(false);
    }

    return(true);
}

bool Neural_Network::Usable_Warm_Restarts(void) const
{
    return(this->type_optimizer_function != MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus
             &&
             this->type_optimizer_function != MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus);
}

T_ Neural_Network::Activation_Function(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received,T_ summation_received)
{
    AF_FIRE(type_activation_function_received, summation_received, summation_received);

    return(summation_received);
}

T_ Neural_Network::Activation_Function_Derive(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received,
                                                                     T_ const summation_received,
                                                                     T_ const steepness_received,
                                                                     T_ value_received)
{
    switch(type_activation_function_received)
    {
    #if defined(COMPILE_ADEPT)
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_NONE:
    #endif
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_COSINE: return(AF_COS_derive(steepness_received, summation_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_COSINE_SYMMETRIC: return(AF_COS_SYMMETRIC_derive(steepness_received, summation_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELU: return(AF_ELU_derive(steepness_received, summation_received, value_received, 1_T));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELLIOT: return(AF_ELLIOT_derive(steepness_received, summation_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELLIOT_SYMMETRIC: return(AF_ELLIOT_SYMMETRIC_derive(steepness_received, summation_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN: return(AF_GAUSSIAN_derive(steepness_received, value_received, summation_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN_SYMMETRIC: return(AF_GAUSSIAN_SYMMETRIC_derive(steepness_received, value_received, summation_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ISRU: return(AF_ISRU_derive(steepness_received, summation_received, value_received, 1_T));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ISRLU: return(AF_ISRLU_derive(steepness_received, summation_received, value_received, 1_T));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR_PIECE:
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR_PIECE_SYMMETRIC: return(AF_LINEAR_derive(steepness_received, value_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LEAKY_RELU: return(AF_LRELU_derive(steepness_received, summation_received, AF_LRELU_ALPHA));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_PARAMETRIC_RELU: return(AF_PRELU_derive(steepness_received, summation_received, AF_PRELU_ALPHA));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_RELU: return(AF_RELU_derive(steepness_received, summation_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SELU: return(AF_SELU_derive(steepness_received, summation_received, value_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID: return(AF_SIGMOID_derive(steepness_received, value_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SINE: return(AF_SIN_derive(steepness_received, summation_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID_STEPWISE: return(AF_SIGMOID_derive(steepness_received, value_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SINE_SYMMETRIC: return(AF_SIN_SYMMETRIC_derive(steepness_received, summation_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH: return(AF_TANH_derive(steepness_received, value_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH_STEPWISE: return(AF_TANH_derive(steepness_received, value_received));
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SOFTMAX: return(steepness_received);
        default:
            PRINT_FORMAT("%s: %s: ERROR: Activation function type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     type_activation_function_received,
                                     MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[type_activation_function_received].c_str(),
                                     __LINE__);
                return(value_received);
    }

    return(value_received);
}

bool Neural_Network::Set__Maximum__Batch_Size(size_t const maximum_batch_size_received)
{
    if(maximum_batch_size_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Maximum batch size can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->maximum_batch_size != maximum_batch_size_received)
    {
        this->maximum_batch_size = maximum_batch_size_received;

        if(this->Update__Batch_Size(this->cache_batch_size, true) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu, true)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     maximum_batch_size_received,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

void Neural_Network::Clear_Outputs(void)
{
    struct Layer *tmp_ptr_output_layer;

    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            // Decode the encoded input as output.
            if(this->use_last_layer_as_output || this->pre_training_level != 0_zu) { tmp_ptr_output_layer = this->ptr_last_layer - (this->pre_training_level == 0_zu ? 1_zu : this->pre_training_level); }
            // Else it use the coded part as output.
            else { tmp_ptr_output_layer = this->ptr_last_layer - ((this->total_layers - 3_zu) / 2_zu + 2_zu); }
                break;
        default: tmp_ptr_output_layer = this->ptr_last_layer - 1; break;
    }
    
    size_t const tmp_number_outputs(*tmp_ptr_output_layer->ptr_number_outputs);

    MEMSET(tmp_ptr_output_layer->ptr_array_outputs,
                   0,
                   this->batch_size * tmp_number_outputs * this->number_recurrent_depth * sizeof(T_));
}

struct Layer *Neural_Network::Get__Input_Layer(void) const
{
    struct Layer *tmp_ptr_input_layer;

    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            // Use first layer or it is in pre-training mode.
            if(this->use_first_layer_as_input || this->pre_training_level != 0_zu) { tmp_ptr_input_layer = this->ptr_array_layers; }
            // Else it use the coded part as input.
            else { tmp_ptr_input_layer = this->ptr_last_layer - ((this->total_layers - 3_zu) / 2_zu + 2_zu); }
                break;
        default: tmp_ptr_input_layer = this->ptr_array_layers; break;
    }
    
    return(tmp_ptr_input_layer);
}

struct Layer *Neural_Network::Get__Output_Layer(void) const
{
    struct Layer *tmp_ptr_output_layer;

    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            // Decode the encoded input as output.
            if(this->use_last_layer_as_output || this->pre_training_level != 0_zu) { tmp_ptr_output_layer = this->ptr_last_layer - (this->pre_training_level == 0_zu ? 1_zu : this->pre_training_level); }
            // Else it use the coded part as output.
            else { tmp_ptr_output_layer = this->ptr_last_layer - ((this->total_layers - 3_zu) / 2_zu + 2_zu); }
                break;
        default: tmp_ptr_output_layer = this->ptr_last_layer - 1; break;
    }
    
    return(tmp_ptr_output_layer);
}

size_t Neural_Network::Get__Input_Size(void) const { return(this->number_inputs); }

size_t Neural_Network::Get__Output_Size(void) const
{
    struct Layer const *tmp_ptr_output_layer;

    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            // Decode the encoded input as output.
            if(this->use_last_layer_as_output || this->pre_training_level != 0_zu) { tmp_ptr_output_layer = this->ptr_last_layer - (this->pre_training_level == 0_zu ? 1_zu : this->pre_training_level); }
            // Else it use the coded part as output.
            else { tmp_ptr_output_layer = this->ptr_last_layer - ((this->total_layers - 3_zu) / 2_zu + 2_zu); }
                break;
        default: tmp_ptr_output_layer = this->ptr_last_layer - 1; break;
    }
    
    return(*tmp_ptr_output_layer->ptr_number_outputs);
}

T_ const *Neural_Network::Get__Outputs(size_t const data_index_received, size_t const time_step_index_received) const
{
    struct Layer const *tmp_ptr_output_layer;

    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER:
            // Decode the encoded input as output.
            if(this->use_last_layer_as_output || this->pre_training_level != 0_zu) { tmp_ptr_output_layer = this->ptr_last_layer - (this->pre_training_level == 0_zu ? 1_zu : this->pre_training_level); }
            // Else it use the coded part as output.
            else { tmp_ptr_output_layer = this->ptr_last_layer - ((this->total_layers - 3_zu) / 2_zu + 2_zu); }
                break;
        default: tmp_ptr_output_layer = this->ptr_last_layer - 1; break;
    }

    return(this->Get__Outputs(tmp_ptr_output_layer,
                                           data_index_received,
                                           time_step_index_received));
}

T_ const *Neural_Network::Get__Outputs(struct Layer const *const ptr_layer_it_received,
                                                            size_t const data_index_received,
                                                            size_t const time_step_index_received) const
{
    size_t const tmp_number_outputs(*ptr_layer_it_received->ptr_number_outputs);

    return(ptr_layer_it_received->ptr_array_outputs + data_index_received * tmp_number_outputs + this->batch_size * tmp_number_outputs * time_step_index_received);
}

T_ Neural_Network::Get__Outputs__Variance(size_t const layer_index_received, size_t const maximum_batch_size_received) const
{
    if(maximum_batch_size_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Maximum batch size can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(0_T);
    }

    return(this->Get__Outputs__Variance(this->ptr_array_layers + layer_index_received, maximum_batch_size_received));
}

T_ Neural_Network::Get__Outputs__Variance(struct Layer const *const ptr_layer_received, size_t const maximum_batch_size_received) const
{
    if(maximum_batch_size_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Maximum batch size can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(0_T);
    }

    size_t const tmp_output_size(*ptr_layer_received->ptr_number_outputs);
    size_t tmp_time_step_index,
              tmp_example_index,
              tmp_output_index;
    
    T_ const *const tmp_ptr_layer_ptr_array_outputs(ptr_layer_received->ptr_array_outputs),
                  *tmp_ptr_array_outputs,
                  tmp_batch_scaled(1_T / static_cast<T_>(this->number_recurrent_depth * maximum_batch_size_received * tmp_output_size));
    T_ tmp_output,
        tmp_mean(0_T),
        tmp_variance(0_T);
    
    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != maximum_batch_size_received; ++tmp_example_index)
        {
            tmp_ptr_array_outputs = tmp_ptr_layer_ptr_array_outputs + tmp_example_index * tmp_output_size + this->batch_size * tmp_output_size * tmp_time_step_index;

            for(tmp_output_index = 0_zu; tmp_output_index != tmp_output_size; ++tmp_output_index)
            {
                tmp_output = tmp_ptr_array_outputs[tmp_output_index];

                tmp_mean += tmp_output;
                tmp_variance += tmp_output * tmp_output;
            }
        }
    }

    // Variance = V / B - pow(M / B, 2)
    tmp_mean *= tmp_batch_scaled;
    tmp_variance = tmp_variance * tmp_batch_scaled - tmp_mean * tmp_mean;

    return(tmp_variance);
}

size_t Layer::Get__Number_Outputs(void) const { return(*this->ptr_number_outputs); }

size_t Layer::Get__First_Connection_Index(void) const { return(*this->ptr_first_connection_index); }

size_t Layer::Get__Last_Connection_Index(void) const { return(*this->ptr_last_connection_index); }

std::string Neural_Network::Get__Parameters(bool const full_description_received)
{
    std::string tmp_string("|===| GENERAL PARAMETERS |===|" NEW_LINE);
    tmp_string += "Network type: " + MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[this->type_network] + ", " + std::to_string(this->type_network) + NEW_LINE;
    tmp_string += "Number time prediction(s): " + std::to_string(this->number_recurrent_depth) + NEW_LINE;
    tmp_string += "Number time delay(s): " + std::to_string(this->number_time_delays) + NEW_LINE;
    tmp_string += "Use the first layer as input: " + std::string(this->use_first_layer_as_input ? "true" : "false") + NEW_LINE;
    tmp_string += "Use the last layer as output: " + std::string(this->use_last_layer_as_output ? "true" : "false") + NEW_LINE;
    tmp_string += "|END| GENERAL PARAMETERS |END|" NEW_LINE;
    tmp_string += NEW_LINE;
    
    if(this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD
      ||
      full_description_received)
    {
        tmp_string += "|===| GRADIENT DESCENT PARAMETERS |===|" NEW_LINE;
        tmp_string += "Learning rate: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->learning_rate) + NEW_LINE;
        tmp_string += "Learning momentum: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->learning_momentum) + NEW_LINE;
        tmp_string += "Use Nesterov: " + std::string(this->use_Nesterov ? "true" : "false") + NEW_LINE;
        tmp_string += "|END| GRADIENT DESCENT PARAMETERS |END|" NEW_LINE;
        tmp_string += NEW_LINE;
    }

    if(this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP
      ||
      full_description_received)
    {
        tmp_string += "|===| QUICKPROP PARAMETERS |===|" NEW_LINE;
        tmp_string += "Decay: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->quickprop_decay) + NEW_LINE;
        tmp_string += "Mu: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->quickprop_mu) + NEW_LINE;
        tmp_string += "|END| QUICKPROP PARAMETERS |END|" NEW_LINE;
        tmp_string += NEW_LINE;
    }

    if(this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus
      ||
      this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus
      ||
      full_description_received)
    {
        tmp_string += "|===| RESILLENT PROPAGATION PARAMETERS |===|" NEW_LINE;
        tmp_string += "Increase factor: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->rprop_increase_factor) + NEW_LINE;
        tmp_string += "Decrease factor: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->rprop_decrease_factor) + NEW_LINE;
        tmp_string += "Delta minimum: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->rprop_delta_min) + NEW_LINE;
        tmp_string += "Delta maximum: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->rprop_delta_max) + NEW_LINE;
        tmp_string += "Delta zero: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->rprop_delta_zero) + NEW_LINE;
        tmp_string += "|END| RESILLENT PROPAGATION PARAMETERS |END|" NEW_LINE;
        tmp_string += NEW_LINE;
    }
    
    if(this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP
      ||
      full_description_received)
    {
        tmp_string += "|===| SARPROP PARAMETERS |===|" NEW_LINE;
        tmp_string += "Weight decay shift: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->sarprop_weight_decay_shift) + NEW_LINE;
        tmp_string += "Step error threshold factor: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->sarprop_step_error_threshold_factor) + NEW_LINE;
        tmp_string += "Step error shift: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->sarprop_step_error_shift) + NEW_LINE;
        tmp_string += "Temperature: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->sarprop_temperature) + NEW_LINE;
        tmp_string += "Epoch(s): " + std::to_string(this->sarprop_epoch) + NEW_LINE;
        tmp_string += "|END| SARPROP PARAMETERS |END|" NEW_LINE;
        tmp_string += NEW_LINE;
    }
    
    if(this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM
      ||
      full_description_received)
    {
        tmp_string += "|===| " + MyEA::String::To_Upper(MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function]) + " PARAMETERS |===|" NEW_LINE;
        tmp_string += "Learning rate: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_learning_rate) + NEW_LINE;
        tmp_string += "Beta1: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_beta1) + NEW_LINE;
        tmp_string += "Beta2: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_beta2) + NEW_LINE;
        tmp_string += "Epsilon: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_epsilon) + NEW_LINE;
        tmp_string += "Bias correction: " + std::string(this->use_adam_bias_correction ? "true" : "false") + NEW_LINE;
        tmp_string += "Gamma: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_gamma) + NEW_LINE;
        tmp_string += "|END| " + MyEA::String::To_Upper(MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function]) + " PARAMETERS |END|" NEW_LINE;
        tmp_string += NEW_LINE;
    }
    else if(this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM
              ||
              this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX
              ||
              this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad
              ||
              full_description_received)
    {
        tmp_string += "|===| " + MyEA::String::To_Upper(MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function]) + " PARAMETERS |===|" NEW_LINE;
        tmp_string += "Learning rate: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_learning_rate) + NEW_LINE;
        tmp_string += "Beta1: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_beta1) + NEW_LINE;
        tmp_string += "Beta2: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_beta2) + NEW_LINE;
        tmp_string += "Epsilon: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_epsilon) + NEW_LINE;
        tmp_string += "Bias correction: " + std::string(this->use_adam_bias_correction ? "true" : "false") + NEW_LINE;
        tmp_string += "|END| " + MyEA::String::To_Upper(MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function]) + " PARAMETERS |END|" NEW_LINE;
        tmp_string += NEW_LINE;
    }
    else if(this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND
              ||
              this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND
              ||
              full_description_received)
    {
        tmp_string += "|===| " + MyEA::String::To_Upper(MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function]) + " PARAMETERS |===|" NEW_LINE;
        tmp_string += "Learning rate: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_learning_rate) + NEW_LINE;
        tmp_string += "Learning rate, final: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->learning_rate_final) + NEW_LINE;
        tmp_string += "Beta1: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_beta1) + NEW_LINE;
        tmp_string += "Beta2: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_beta2) + NEW_LINE;
        tmp_string += "Epsilon: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->adam_epsilon) + NEW_LINE;
        tmp_string += "Bias correction: " + std::string(this->use_adam_bias_correction ? "true" : "false") + NEW_LINE;
        tmp_string += "Gamma: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->learning_gamma) + NEW_LINE;
        tmp_string += "|END| " + MyEA::String::To_Upper(MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function]) + " PARAMETERS |END|" NEW_LINE;
        tmp_string += NEW_LINE;
    }
    
    tmp_string += "|===| WARM RESTARTS PARAMETERS |===|" NEW_LINE;
    tmp_string += "Use warm restarts: " + std::string(this->use_Warm_Restarts ? "true" : "false") + NEW_LINE;
    if(this->use_Warm_Restarts)
    {
        tmp_string += "Learning rate, decay: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->warm_restarts_decay_learning_rate) + NEW_LINE;
        tmp_string += "Maximum learning rate: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->warm_restarts_maximum_learning_rate) + " / " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->warm_restarts_initial_maximum_learning_rate) + NEW_LINE;
        tmp_string += "Minimum learning rate: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->warm_restarts_minimum_learning_rate) + NEW_LINE;
        tmp_string += "Ti: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->warm_restarts_T_i) + NEW_LINE;
        tmp_string += "Initial, Ti: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->warm_restarts_initial_T_i) + NEW_LINE;
        tmp_string += "Warm restart multiplier: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->warm_restarts_multiplier) + NEW_LINE;
    }
    tmp_string += "|END| WARM RESTARTS PARAMETERS |END|" NEW_LINE;
    tmp_string += NEW_LINE;

    tmp_string += "|===| TRAINING PARAMETERS |===|" NEW_LINE;
    tmp_string += "Training algorithm: " + MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function] + ", " + std::to_string(this->type_optimizer_function) + NEW_LINE;
    tmp_string += "Loss function: " + MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS_NAMES[this->type_loss_function] + ", " + std::to_string(this->type_loss_function) + NEW_LINE;
    tmp_string += "Accuracy function: " + MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS_NAMES[this->type_accuracy_function] + ", " + std::to_string(this->type_accuracy_function) + NEW_LINE;
    if(this->type_loss_function == MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_BIT
      ||
      full_description_received)
    { tmp_string += "Fail-limit: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->bit_fail_limit) + NEW_LINE; }
    tmp_string += "Optimizer time step: " + std::to_string(static_cast<size_t>(this->optimizer_time_step)) + NEW_LINE;
    tmp_string += "Epoch time step: " + std::to_string(static_cast<size_t>(this->epoch_time_step)) + NEW_LINE;
    tmp_string += "Pre-training level: " + std::to_string(this->pre_training_level) + NEW_LINE;
    tmp_string += "Use clip gradient: " + std::string(this->Use__Clip_Gradient() ? "true" : "false") + NEW_LINE;
    if(this->Use__Clip_Gradient()
      ||
      full_description_received)
    {
        tmp_string += "Clip gradient: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->clip_gradient) + NEW_LINE;
    }
    tmp_string += "|END| TRAINING PARAMETERS |END|" NEW_LINE;
    tmp_string += NEW_LINE;

    tmp_string += "|===| REGULARIZATION PARAMETERS |===|" NEW_LINE;
    tmp_string += "Use dropout, bernoulli: " + std::string(this->Use__Dropout__Bernoulli() ? "true" : "false") + NEW_LINE;
    tmp_string += "Use dropout, bernoulli inverted: " + std::string(this->Use__Dropout__Bernoulli__Inverted() ? "true" : "false") + NEW_LINE;
    tmp_string += "Use dropout, gaussian: " + std::string(this->Use__Dropout__Gaussian() ? "true" : "false") + NEW_LINE;
    tmp_string += "Use dropout, shakedrop: " + std::string(this->Use__Dropout__ShakeDrop() ? "true" : "false") + NEW_LINE;
    tmp_string += "Use dropout, uout: " + std::string(this->Use__Dropout__Uout() ? "true" : "false") + NEW_LINE;
    tmp_string += "Use dropout, zoneout: " + std::string(this->Use__Dropout__Zoneout() ? "true" : "false") + NEW_LINE;
    tmp_string += "Max-norm contraints: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->regularization__max_norm_constraints) + NEW_LINE;
    tmp_string += "L1 regularization: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->regularization__l1) + NEW_LINE;
    tmp_string += "L2 regularization: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->regularization__l2) + NEW_LINE;
    tmp_string += "SRIP regularization: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->regularization__srip) + NEW_LINE;
    tmp_string += "Weight decay: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->regularization__weight_decay) + NEW_LINE;
    tmp_string += "Use normalized weight decay: " + std::string(this->use_normalized_weight_decay ? "true" : "false") + NEW_LINE;
    tmp_string += "Use tied parameter: " + std::string(this->Use__Tied_Parameter() ? "true" : "false") + NEW_LINE;
    tmp_string += "Use k-Sparse: " + std::string(this->Use__K_Sparse() ? "true" : "false") + NEW_LINE;
    tmp_string += "Use constraint recurrent weight: " + std::string(this->Use__Regularization__Constraint_Recurrent_Weight() ? "true" : "false") + NEW_LINE;
    tmp_string += "|END| REGULARIZATION PARAMETERS |END|" NEW_LINE;
    tmp_string += NEW_LINE;
    
    tmp_string += "|===| NORMALIZATION PARAMETERS |===|" NEW_LINE;
    tmp_string += "Use batch normalization: " + std::string(this->Use__Batch_Normalization() ? "true" : "false") + NEW_LINE;
    tmp_string += "Use batch renormalization: " + std::string(this->Use__Batch_Renormalization() ? "true" : "false") + NEW_LINE;
    tmp_string += "momentum average: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->normalization_momentum_average) + NEW_LINE;
    tmp_string += "normalization epsilon: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->normalization_epsilon) + NEW_LINE;
    tmp_string += "r correction maximum: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->batch_renormalization_r_correction_maximum) + NEW_LINE;
    tmp_string += "d correction maximum: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->batch_renormalization_d_correction_maximum) + NEW_LINE;
    tmp_string += "|===| NORMALIZATION PARAMETERS |===|" NEW_LINE;
    tmp_string += NEW_LINE;

    tmp_string += "|===| LOSS PARAMETERS |===|" NEW_LINE;
    tmp_string += "Training: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->loss_training) + NEW_LINE;
    tmp_string += "Validating: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->loss_validating) + NEW_LINE;
    tmp_string += "Testing: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->loss_testing) + NEW_LINE;
    tmp_string += "|END| LOSS PARAMETERS |END|" NEW_LINE;
    tmp_string += NEW_LINE;
    
    tmp_string += "|===| ACCURANCY PARAMETERS |===|" NEW_LINE;
    tmp_string += "Variance: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->accuracy_variance) + NEW_LINE;
    tmp_string += "Training: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->accuracy_training) + NEW_LINE;
    tmp_string += "Validating: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->accuracy_validating) + NEW_LINE;
    tmp_string += "Testing: " + MyEA::String::To_string<T_,MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(this->accuracy_testing) + NEW_LINE;
    tmp_string += "|END| ACCURANCY PARAMETERS |END|" NEW_LINE;
    tmp_string += NEW_LINE;

    tmp_string += "|===| COMPUTATION PARAMETERS |===|" NEW_LINE;
#if defined(COMPILE_CUDA)
    tmp_string += "Use CUDA: " + std::string(this->use_CUDA ? "true" : "false") + NEW_LINE;
#endif
    tmp_string += "Use OpenMP: " + std::string(this->use_OpenMP ? "true" : "false") + NEW_LINE;
    tmp_string += "Maximum threads (percent): " + std::to_string(this->percentage_maximum_thread_usage) + "%" + NEW_LINE;
    tmp_string += "Number of threads: " + std::to_string(this->number_threads) + NEW_LINE;
    tmp_string += "Batch size: " + std::to_string(this->batch_size) + " / " + std::to_string(this->maximum_batch_size) + NEW_LINE;
    tmp_string += "Maximum allowable memory: " + std::to_string(this->maximum_allowable_memory_bytes) + " bytes | " + MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(static_cast<double>(this->maximum_allowable_memory_bytes) / 1024.0 / 1024.0, 4u) + " MBs" + NEW_LINE;
    tmp_string += "Size for one thread: " + std::to_string(this->Get__Threads_Sizeof(1u)) + " bytes | " + MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(static_cast<double>(this->Get__Threads_Sizeof(1u)) / 1024.0 / 1024.0, 4u) + " MBs" + NEW_LINE;
    tmp_string += "Size for a batch of size one: " + std::to_string(this->Get__Batch_Sizeof(1u)) + " bytes | " + MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(static_cast<double>(this->Get__Batch_Sizeof(1u)) / 1024.0 / 1024.0, 4u) + " MBs" + NEW_LINE;
    tmp_string += "Size neural network: " + std::to_string(this->Get__Sizeof()) + " bytes | " + MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_FIXED>(static_cast<double>(this->Get__Sizeof()) / 1024.0 / 1024.0, 4u) + " MBs" + NEW_LINE;
    tmp_string += "|END| COMPUTATION PARAMETERS |END|" NEW_LINE;
    tmp_string += NEW_LINE;
    
    tmp_string += "|===| DIMENSION |===|" NEW_LINE;
    tmp_string += "Total layer(s): " + std::to_string(this->total_layers) + NEW_LINE;
    tmp_string += "Total basic unit(s): " + std::to_string(this->total_basic_units) + "/" + std::to_string(this->total_basic_units_allocated) + NEW_LINE;
    tmp_string += "Total basic indice unit(s): " + std::to_string(this->total_basic_indice_units) + "/" + std::to_string(this->total_basic_indice_units_allocated) + NEW_LINE;
    tmp_string += "Total neuron unit(s): " + std::to_string(this->total_neuron_units) + "/" + std::to_string(this->total_neuron_units_allocated) + NEW_LINE;
    tmp_string += "Total AF unit(s): " + std::to_string(this->total_AF_units) + "/" + std::to_string(this->total_AF_units_allocated) + NEW_LINE;
    tmp_string += "Total AF Ind unit(s): " + std::to_string(this->total_AF_Ind_recurrent_units) + "/" + std::to_string(this->total_AF_Ind_recurrent_units_allocated) + NEW_LINE;
    tmp_string += "Total normalized unit(s): " + std::to_string(this->total_normalized_units) + "/" + std::to_string(this->total_normalized_units_allocated) + NEW_LINE;
    tmp_string += "Total block unit(s): " + std::to_string(this->total_block_units) + "/" + std::to_string(this->total_block_units_allocated) + NEW_LINE;
    tmp_string += "Total cell unit(s): " + std::to_string(this->total_cell_units) + "/" + std::to_string(this->total_cell_units_allocated) + NEW_LINE;
    tmp_string += "Total parameter(s): " + std::to_string(this->total_parameters) + "/" + std::to_string(this->total_parameters_allocated) + NEW_LINE;
    tmp_string += "Total weight(s): " + std::to_string(this->total_weights) + "/" + std::to_string(this->total_weights_allocated) + NEW_LINE;
    tmp_string += "Total bias(s): " + std::to_string(this->total_bias) + "/" + std::to_string(this->total_bias_allocated) + NEW_LINE;

    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1),
                               *tmp_ptr_previous_layer,
                               *tmp_ptr_layer_it(this->ptr_array_layers);

    // Input layer.
    tmp_string += "  Input layer:" NEW_LINE;
    tmp_string += "    Type: " + MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer] + ", " + std::to_string(tmp_ptr_layer_it->type_layer) + NEW_LINE;
    tmp_string += "    Type activation: " + MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[tmp_ptr_layer_it->type_activation] + ", " + std::to_string(tmp_ptr_layer_it->type_activation) + NEW_LINE;
    tmp_string += "    Type dropout: " + MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_ptr_layer_it->type_dropout] + ", " + std::to_string(tmp_ptr_layer_it->type_dropout) + NEW_LINE;
    tmp_string += "      Use coded dropout: " + std::to_string(tmp_ptr_layer_it->use_coded_dropout) + NEW_LINE;
    tmp_string += "      Dropout value[0]: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->dropout_values[0u]) + NEW_LINE;
    tmp_string += "      Dropout value[1]: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->dropout_values[1u]) + NEW_LINE;
    tmp_string += "      Dropout value[2]: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->dropout_values[2u]) + NEW_LINE;
    tmp_string += "    First connection index: " + std::to_string(*tmp_ptr_layer_it->ptr_first_connection_index) + NEW_LINE;
    tmp_string += "    Last connection index: " + std::to_string(*tmp_ptr_layer_it->ptr_last_connection_index) + NEW_LINE;
    tmp_string += "    Number input(s): " + std::to_string(this->number_inputs) + NEW_LINE;
    // |END| Input layer. |END|

    auto tmp_Information__Layer__Normalization([self = this](struct Layer const *const ptr_layer_it_received) -> std::string
    {
        std::string tmp_string("");

        if(static_cast<size_t>(ptr_layer_it_received->ptr_last_normalized_unit - ptr_layer_it_received->ptr_array_normalized_units) <= 12_zu)
        {
            // Normalization.
            if(self->Information__Layer__Normalization(tmp_string, ptr_layer_it_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Normalization()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return("");
            }
            // |END| Normalization. |END|
        }
        else
        {
            tmp_string += "    Type normalization: " + MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_it_received->type_normalization] + ", " + std::to_string(ptr_layer_it_received->type_normalization) + NEW_LINE;
            tmp_string += "      Use layer normalization before activation: " + std::string(ptr_layer_it_received->use_layer_normalization_before_activation ? "true" : "false") + NEW_LINE;
            tmp_string += "      Number normalized unit(s): " + std::to_string(static_cast<size_t>(ptr_layer_it_received->ptr_last_normalized_unit - ptr_layer_it_received->ptr_array_normalized_units)) + NEW_LINE;
        }

        return(tmp_string);
    });
    
    auto tmp_Information__Layer__FC([self = this](struct Layer const *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_connected_received) -> std::string
    {
        std::string tmp_string("");
        
        if(*ptr_previous_layer_connected_received->ptr_number_outputs <= 12_zu
          &&
          *ptr_layer_it_received->ptr_number_outputs <= 12_zu)
        {
            // Neuron(s).
            if(self->Information__Layer__FC(tmp_string,
                                                          ptr_layer_it_received,
                                                          ptr_previous_layer_connected_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__FC()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return("");
            }
            // |END| Neuron(s). |END|
            
            // Bias parameter(s).
            if(self->Information__Layer__Bias(tmp_string, ptr_layer_it_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Bias()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return("");
            }
            // |END| Bias parameter(s). |END|
        }
        else
        {
            tmp_string += "    Number neuron(s): " + std::to_string(static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - ptr_layer_it_received->ptr_array_neuron_units)) + NEW_LINE;
            tmp_string += "    Number bias: " + std::to_string(ptr_layer_it_received->last_bias_connection_index - ptr_layer_it_received->first_bias_connection_index) + NEW_LINE;
        }

        return(tmp_string);
    });
    
    auto tmp_Information__Layer__AF([self = this](struct Layer const *const ptr_layer_it_received) -> std::string
    {
        std::string tmp_string("");
        
        if(*ptr_layer_it_received->ptr_number_outputs <= 12_zu)
        {
            // AF(s).
            if(self->Information__Layer__AF(tmp_string, ptr_layer_it_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__AF()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return("");
            }
            // |END| AF(s). |END|
        }
        else
        {
            tmp_string += "    Number AF(s): " + std::to_string(static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_unit - ptr_layer_it_received->ptr_array_AF_units)) + NEW_LINE;
        }

        return(tmp_string);
    });
    
    auto tmp_Information__Layer__AF_Ind_Recurrent([self = this](struct Layer const *const ptr_layer_it_received) -> std::string
    {
        std::string tmp_string("");
        
        if(*ptr_layer_it_received->ptr_number_outputs <= 12_zu)
        {
            // AF(s) Ind recurrent.
            if(self->Information__Layer__AF_Ind_Recurrent(tmp_string, ptr_layer_it_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__AF()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return("");
            }
            // |END| AF(s) Ind recurrent. |END|
        }
        else
        {
            tmp_string += "    Number AF(s) Ind recurrent: " + std::to_string(static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit - ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units)) + NEW_LINE;
        }

        return(tmp_string);
    });
    
    auto tmp_Information__Layer__LSTM([self = this](struct Layer const *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_connected_received) -> std::string
    {
        std::string tmp_string("");
        
        if(*ptr_previous_layer_connected_received->ptr_number_outputs <= 12_zu
          &&
          *ptr_layer_it_received->ptr_number_outputs <= 12_zu)
        {
            // Blocks.
            if(self->Information__Layer__LSTM(tmp_string,
                                                               ptr_layer_it_received,
                                                               ptr_previous_layer_connected_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__LSTM()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return("");
            }
            // |END| Blocks. |END|
            
            // Bias parameter(s).
            if(self->Information__Layer__Bias(tmp_string, ptr_layer_it_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Bias()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return("");
            }
            // |END| Bias parameter(s). |END|
        }
        else
        {
            tmp_string += "    Number block unit(s): " + std::to_string(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units)) + NEW_LINE;
            tmp_string += "    Number cell unit(s): " + std::to_string(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units)) + NEW_LINE;
        }

        return(tmp_string);
    });
    
    // Hidden layer(s).
    for(++tmp_ptr_layer_it; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        tmp_ptr_previous_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

        tmp_string += "  Hidden layer [" + std::to_string(tmp_ptr_layer_it - this->ptr_array_layers) + "]" NEW_LINE;
        tmp_string += "    Type: " + MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer] + ", " + std::to_string(tmp_ptr_layer_it->type_layer) + NEW_LINE;
        tmp_string += "    Use bidirectional: " + std::to_string(tmp_ptr_layer_it->use_bidirectional) + NEW_LINE;
        
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                tmp_string += "    Kernel size: " + std::to_string(tmp_ptr_layer_it->pooling_values[0u]) + NEW_LINE;
                tmp_string += "    Stride: " + std::to_string(tmp_ptr_layer_it->pooling_values[1u]) + NEW_LINE;
                tmp_string += "    Padding: " + std::to_string(tmp_ptr_layer_it->pooling_values[2u]) + NEW_LINE;
                tmp_string += "    Dilation: " + std::to_string(tmp_ptr_layer_it->pooling_values[3u]) + NEW_LINE;
                tmp_string += "    Ceil mode: " + std::string(tmp_ptr_layer_it->pooling_values[4u] > 0_zu ? "true" : "false") + NEW_LINE;
                tmp_string += "    Number feature(s): " + std::to_string(*tmp_ptr_layer_it->ptr_number_outputs) + NEW_LINE;
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                tmp_string += "    Type activation: " + MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[tmp_ptr_layer_it->type_activation] + ", " + std::to_string(tmp_ptr_layer_it->type_activation) + NEW_LINE;
                tmp_string += "    Type dropout: " + MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_ptr_layer_it->type_dropout] + ", " + std::to_string(tmp_ptr_layer_it->type_dropout) + NEW_LINE;
                tmp_string += "      Use coded dropout: " + std::to_string(tmp_ptr_layer_it->use_coded_dropout) + NEW_LINE;
                tmp_string += "      Dropout value[0]: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->dropout_values[0u]) + NEW_LINE;
                tmp_string += "      Dropout value[1]: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->dropout_values[1u]) + NEW_LINE;
                tmp_string += "      Dropout value[2]: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->dropout_values[2u]) + NEW_LINE;
                tmp_string += tmp_Information__Layer__Normalization(tmp_ptr_layer_it);
                tmp_string += "    Use tied parameter: " + std::string(tmp_ptr_layer_it->use_tied_parameter ? "true" : "false") + NEW_LINE;
                tmp_string += "    k-Sparsity: " + std::to_string(tmp_ptr_layer_it->k_sparsity) + NEW_LINE;
                tmp_string += "    Alpha sparsity: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->alpha_sparsity) + NEW_LINE;
                tmp_string += "    Constraint recurrent weight lower bound: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->constraint_recurrent_weight_lower_bound) + NEW_LINE;
                tmp_string += "    Constraint recurrent weight upper bound: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->constraint_recurrent_weight_upper_bound) + NEW_LINE;
                tmp_string += "    First connection index: " + std::to_string(*tmp_ptr_layer_it->ptr_first_connection_index) + NEW_LINE;
                tmp_string += "    Last connection index: " + std::to_string(*tmp_ptr_layer_it->ptr_last_connection_index) + NEW_LINE;
                tmp_string += "    First bias connection index: " + std::to_string(tmp_ptr_layer_it->first_bias_connection_index) + NEW_LINE;
                tmp_string += "    Last bias connection index: " + std::to_string(tmp_ptr_layer_it->last_bias_connection_index) + NEW_LINE;
                tmp_string += "    Number feature(s): " + std::to_string(*tmp_ptr_layer_it->ptr_number_outputs) + NEW_LINE;
                tmp_string += tmp_Information__Layer__FC(tmp_ptr_layer_it, tmp_ptr_previous_layer);
                tmp_string += tmp_Information__Layer__AF(tmp_ptr_layer_it);
                tmp_string += tmp_Information__Layer__AF_Ind_Recurrent(tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                tmp_string += "    Type activation: " + MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[tmp_ptr_layer_it->type_activation] + ", " + std::to_string(tmp_ptr_layer_it->type_activation) + NEW_LINE;
                tmp_string += "    Type dropout: " + MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_ptr_layer_it->type_dropout] + ", " + std::to_string(tmp_ptr_layer_it->type_dropout) + NEW_LINE;
                tmp_string += "        Dropout value[0]: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->dropout_values[0u]) + NEW_LINE;
                tmp_string += "        Dropout value[1]: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->dropout_values[1u]) + NEW_LINE;
                tmp_string += "        Dropout value[2]: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->dropout_values[2u]) + NEW_LINE;
                tmp_string += tmp_Information__Layer__Normalization(tmp_ptr_layer_it);
                tmp_string += "    Use tied parameter: " + std::string(tmp_ptr_layer_it->use_tied_parameter ? "true" : "false") + NEW_LINE;
                tmp_string += "    k-Sparsity: " + std::to_string(tmp_ptr_layer_it->k_sparsity) + NEW_LINE;
                tmp_string += "    Alpha sparsity: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->alpha_sparsity) + NEW_LINE;
                tmp_string += "    Constraint recurrent weight lower bound: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->constraint_recurrent_weight_lower_bound) + NEW_LINE;
                tmp_string += "    Constraint recurrent weight upper bound: " + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->constraint_recurrent_weight_upper_bound) + NEW_LINE;
                tmp_string += "    First connection index: " + std::to_string(*tmp_ptr_layer_it->ptr_first_connection_index) + NEW_LINE;
                tmp_string += "    Last connection index: " + std::to_string(*tmp_ptr_layer_it->ptr_last_connection_index) + NEW_LINE;
                tmp_string += "    First bias connection index: " + std::to_string(tmp_ptr_layer_it->first_bias_connection_index) + NEW_LINE;
                tmp_string += "    Last bias connection index: " + std::to_string(tmp_ptr_layer_it->last_bias_connection_index) + NEW_LINE;
                tmp_string += "    Number feature(s): " + std::to_string(*tmp_ptr_layer_it->ptr_number_outputs) + NEW_LINE;
                tmp_string += tmp_Information__Layer__LSTM(tmp_ptr_layer_it, tmp_ptr_previous_layer);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                tmp_string += "    Block depth: " + std::to_string(tmp_ptr_layer_it->block_depth) + NEW_LINE;
                tmp_string += "    Padding: " + std::to_string(tmp_ptr_layer_it->pooling_values[2u]) + NEW_LINE;
                tmp_string += tmp_Information__Layer__Normalization(tmp_ptr_layer_it);
                tmp_string += "    Number feature(s): " + std::to_string(*tmp_ptr_layer_it->ptr_number_outputs) + NEW_LINE;
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    return("");
        }
    }
    // |END| Hidden layer(s). |END|

    // Output layer.
    tmp_ptr_previous_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

    tmp_string += "  Output layer:" NEW_LINE;
    tmp_string += "    Type: " + MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer] + ", " + std::to_string(tmp_ptr_layer_it->type_layer) + NEW_LINE;
    tmp_string += "    Type activation: " + MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[tmp_ptr_layer_it->type_activation] + ", " + std::to_string(tmp_ptr_layer_it->type_activation) + NEW_LINE;
    tmp_string += "    First connection index: " + std::to_string(*tmp_ptr_layer_it->ptr_first_connection_index) + NEW_LINE;
    tmp_string += "    Last connection index: " + std::to_string(*tmp_ptr_layer_it->ptr_last_connection_index) + NEW_LINE;
    tmp_string += "    First bias connection index: " + std::to_string(tmp_ptr_layer_it->first_bias_connection_index) + NEW_LINE;
    tmp_string += "    Last bias connection index: " + std::to_string(tmp_ptr_layer_it->last_bias_connection_index) + NEW_LINE;
    
    if(*tmp_ptr_previous_layer->ptr_number_outputs <= 12_zu
      &&
      *tmp_ptr_layer_it->ptr_number_outputs <= 12_zu)
    {
        // Neuron(s).
        if(this->Information__Output_Layer(tmp_string,
                                                          tmp_ptr_layer_it,
                                                          tmp_ptr_previous_layer) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Output_Layer()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return("");
        }
        // |END| Neuron(s). |END|
        
        // Bias parameter(s).
        if(this->Information__Layer__Bias(tmp_string, tmp_ptr_layer_it) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Information__Layer__Bias()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return("");
        }
        // |END| Bias parameter(s). |END|
    }
    else { tmp_string += "    Number output(s): " + std::to_string(this->number_outputs) + NEW_LINE; }
    // |END| Output layer. |END|
    tmp_string += "|END| DIMENSION |END|" NEW_LINE;

    return(tmp_string);
}

bool Neural_Network::Multi_Class_Classification(void) const
//{ return((this->ptr_last_layer - 1)->type_activation == MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX); }
{ return(this->number_outputs > 1_zu); }

size_t Neural_Network::Get__Total_Layers(void) const { return(this->total_layers); }

void Neural_Network::Reset__Parameter__Mask_Dropout(bool *ptr_array_units_mask_dropout_bernoulli_received)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

    this->ptr_array_units_mask_dropout_bernoulli = ptr_array_units_mask_dropout_bernoulli_received;

    for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        tmp_ptr_layer_it->ptr_array__mask__dropout__bernoulli = ptr_array_units_mask_dropout_bernoulli_received;

        ptr_array_units_mask_dropout_bernoulli_received += static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_unit - tmp_ptr_layer_it->ptr_array_AF_units) * this->number_recurrent_depth;
        ptr_array_units_mask_dropout_bernoulli_received += static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units) * this->number_recurrent_depth;
    }
}

void Neural_Network::Reset__Parameters__Cell_Unit__Mask_Dropout(bool *ptr_array_cell_units_mask_dropout_received)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

    struct Block_unit const *tmp_ptr_last_block_unit;
    struct Block_unit *tmp_ptr_block_unit_it;
    
    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    this->ptr_array_cell_units_mask_dropout_zoneout = ptr_array_cell_units_mask_dropout_received;

    for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        for(tmp_ptr_last_block_unit = tmp_ptr_layer_it->ptr_last_block_unit,
            tmp_ptr_block_unit_it = tmp_ptr_layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
        {
            tmp_ptr_block_unit_it->ptr_array_mask_dropout_zoneout = ptr_array_cell_units_mask_dropout_received;

            for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                   ++ptr_array_cell_units_mask_dropout_received)
            {
                tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state = ptr_array_cell_units_mask_dropout_received;
                tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output = ptr_array_cell_units_mask_dropout_received + this->number_recurrent_depth * this->total_cell_units_allocated;
            }
        }

        ptr_array_cell_units_mask_dropout_received += static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units) * (this->number_recurrent_depth - 1_zu);
    }
}

void Neural_Network::Reset__Parameter__Normalized_Unit(void)
{
    size_t tmp_number_units,
              tmp_index;
    
    void **tmp_ptr_array_ptr_connections(this->ptr_array_ptr_connections + this->total_weights_allocated + this->total_bias_allocated);
    
    T_ *tmp_ptr_array_parameters_scale_it(this->ptr_array_parameters + this->total_weights_allocated + this->total_bias_allocated),
         *tmp_ptr_array_parameters_shift_it(this->ptr_array_parameters + this->total_weights_allocated + this->total_bias_allocated + this->total_normalized_units);
    
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
    
    struct Block_unit const *tmp_ptr_last_block_unit;
    struct Block_unit *tmp_ptr_block_unit_it;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    union Normalized_unit const *tmp_ptr_last_normalized_unit;
    union Normalized_unit *tmp_ptr_normalized_unit_it;
    
    for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if((tmp_number_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_normalized_unit - tmp_ptr_layer_it->ptr_array_normalized_units)) != 0_zu)
        {
            switch(tmp_ptr_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    for(tmp_ptr_last_normalized_unit = tmp_ptr_layer_it->ptr_last_normalized_unit,
                        tmp_ptr_normalized_unit_it = tmp_ptr_layer_it->ptr_array_normalized_units; tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit; ++tmp_ptr_normalized_unit_it,
                                                                                                                                                                                                                               ++tmp_ptr_array_parameters_scale_it,
                                                                                                                                                                                                                               ++tmp_ptr_array_parameters_shift_it,
                                                                                                                                                                                                                               ++tmp_ptr_array_ptr_connections)
                    {
                        tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_scale = tmp_ptr_array_parameters_scale_it;
                        tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_shift = tmp_ptr_array_parameters_shift_it;

                        tmp_ptr_array_ptr_connections[0u] = tmp_ptr_normalized_unit_it;
                        tmp_ptr_array_ptr_connections[this->total_normalized_units] = tmp_ptr_normalized_unit_it;
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    if(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units) != 0_zu)
                    {
                        // [0]: Block input, input.
                        // [1]: Block input, recurrent.
                        // [2]: Cell state activate.

                        tmp_ptr_last_cell_unit = tmp_ptr_layer_it->ptr_last_cell_unit;
                    
                        tmp_number_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units);

                        for(tmp_index = 0_zu; tmp_index != 3_zu; ++tmp_index)
                        {
                            for(tmp_ptr_cell_unit_it = tmp_ptr_layer_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                            {
                                tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_scale = tmp_ptr_array_parameters_scale_it++;
                                tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_shift = tmp_ptr_array_parameters_shift_it++;
                            
                                tmp_ptr_array_ptr_connections[0u] = tmp_ptr_cell_unit_it;
                                tmp_ptr_array_ptr_connections[this->total_normalized_units] = tmp_ptr_cell_unit_it;
                                ++tmp_ptr_array_ptr_connections;
                            }
                        }
                    
                        // [3]: Input gate, input.
                        // [4]: Input gate, recurrent.
                        // [5]: Forget gate, input.
                        // [6]: Forget gate, recurrent.
                        // [7]: Output gate, input.
                        // [8]: Output gate, recurrent.

                        tmp_ptr_last_block_unit = tmp_ptr_layer_it->ptr_last_block_unit;
                    
                        tmp_number_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units);

                        for(tmp_index = 0_zu; tmp_index != 6_zu; ++tmp_index)
                        {
                            for(tmp_ptr_block_unit_it = tmp_ptr_layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
                            {
                                tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_scale = tmp_ptr_array_parameters_scale_it++;
                                tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_shift = tmp_ptr_array_parameters_shift_it++;

                                tmp_ptr_array_ptr_connections[0u] = tmp_ptr_block_unit_it;
                                tmp_ptr_array_ptr_connections[this->total_normalized_units] = tmp_ptr_block_unit_it;
                                ++tmp_ptr_array_ptr_connections;
                            }
                        }
                    } 
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_layer_it->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                             __LINE__);
                        return;
            }
        }
    }
}

void Neural_Network::Reset__Derivative_Parameter__Normalized_Unit(void)
{
    size_t tmp_index;
    
    T_ *tmp_ptr_array_derivatives_parameters_scale_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated + this->total_bias_allocated),
         *tmp_ptr_array_derivatives_parameters_shift_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated + this->total_bias_allocated + this->total_normalized_units);
    
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
    
    struct Block_unit const *tmp_ptr_last_block_unit;
    struct Block_unit *tmp_ptr_block_unit_it;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    union Normalized_unit const *tmp_ptr_last_normalized_unit;
    union Normalized_unit *tmp_ptr_normalized_unit_it;
    
    for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_normalized_unit - tmp_ptr_layer_it->ptr_array_normalized_units) != 0_zu)
        {
            switch(tmp_ptr_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    for(tmp_ptr_last_normalized_unit = tmp_ptr_layer_it->ptr_last_normalized_unit,
                        tmp_ptr_normalized_unit_it = tmp_ptr_layer_it->ptr_array_normalized_units; tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit; ++tmp_ptr_normalized_unit_it,
                                                                                                                                                                                                                               ++tmp_ptr_array_derivatives_parameters_scale_it,
                                                                                                                                                                                                                               ++tmp_ptr_array_derivatives_parameters_shift_it)
                    {
                        tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it;
                        tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it;
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    if(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units) != 0_zu)
                    {
                        // [0]: Block input, input.
                        // [1]: Block input, recurrent.
                        // [2]: Cell state activate.

                        tmp_ptr_last_cell_unit = tmp_ptr_layer_it->ptr_last_cell_unit;
                    
                        for(tmp_index = 0_zu; tmp_index != 3_zu; ++tmp_index)
                        {
                            for(tmp_ptr_cell_unit_it = tmp_ptr_layer_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                            {
                                tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it++;
                                tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it++;
                            }
                        }
                    
                        // [3]: Input gate, input.
                        // [4]: Input gate, recurrent.
                        // [5]: Forget gate, input.
                        // [6]: Forget gate, recurrent.
                        // [7]: Output gate, input.
                        // [8]: Output gate, recurrent.

                        tmp_ptr_last_block_unit = tmp_ptr_layer_it->ptr_last_block_unit;
                    
                        for(tmp_index = 0_zu; tmp_index != 6_zu; ++tmp_index)
                        {
                            for(tmp_ptr_block_unit_it = tmp_ptr_layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
                            {
                                tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it++;
                                tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it++;
                            }
                        }
                    }
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_layer_it->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                             __LINE__);
                        return;
            }
        }
    }
}

void Neural_Network::Clear__Parameter__Normalized_Unit(void)
{
    size_t tmp_number_units;
    
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
    
    for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if((tmp_number_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_normalized_unit - tmp_ptr_layer_it->ptr_array_normalized_units)) != 0_zu)
        {
            // Clear shift.
            MEMSET(tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_shift,
                           0,
                           tmp_number_units * sizeof(T_));
            
            // Clear scale.
            switch(tmp_ptr_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    Memory::Fill<T_>(tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale,
                                              tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale + tmp_number_units,
                                              1_T);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    Memory::Fill<T_>(tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale,
                                              tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale + tmp_number_units,
                                              0.1_T);
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    Memory::Fill<T_>(tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale,
                                              tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale + tmp_number_units,
                                              this->number_recurrent_depth == 1_zu ? 1_T : 0.1_T);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_ptr_layer_it->type_layer,
                                                MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                                __LINE__);
                        return;
            }

            // Clear average mean.
            MEMSET(tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_mean_average,
                           0,
                           tmp_number_units * this->number_recurrent_depth * sizeof(T_));

            // Clear average variance.
            Memory::Fill<T_>(tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_variance_average,
                                      tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_variance_average + tmp_number_units * this->number_recurrent_depth,
                                      1_T);
        }
    }
}

bool Neural_Network::Initialized__Weight(void) const { return(this->_initialized__weight); }

bool Neural_Network::Initialize__Weight(class Dataset<T_> const *const ptr_Dataset_received)
{
    switch(this->_type_weights_initializer)
    {
        case MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_LSUV:
            if(this->Initialization__LSUV(ptr_Dataset_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialization__LSUV(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type weights initializer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->_type_weights_initializer,
                                     MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS_NAMES[this->_type_weights_initializer].c_str(),
                                     __LINE__);
                return(false);
    }

    return(true);
}

size_t Neural_Network::Get__Batch_Sizeof(size_t batch_size_received) const
{
    size_t tmp_total_size_t(0_zu);

    if(batch_size_received == 0_zu) { batch_size_received = this->batch_size; }
    
    // Basic unit(s).
    if(this->ptr_array_basic_units_values != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_basic_units_allocated * sizeof(T_); }
    if(this->ptr_array_basic_units_errors != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_basic_units_allocated * sizeof(T_); }
    // |END| Basic unit(s). |END|
    
    // Basic indice unit(s).
    if(this->ptr_array_basic_indice_units_indices != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_basic_indice_units_allocated * sizeof(T_); }
    if(this->ptr_array_basic_indice_units_values != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_basic_indice_units_allocated * sizeof(T_); }
    if(this->ptr_array_basic_indice_units_errors != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_basic_indice_units_allocated * sizeof(T_); }
    // |END| Basic indice unit(s). |END|
    
    // Neuron unit(s).
    if(this->ptr_array_neuron_units_summations != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_neuron_units_errors != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    // |END| Neuron unit(s). |END|
    
    // AF unit(s).
    if(this->ptr_array_AF_units_values != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_AF_units_allocated * sizeof(T_); }
    if(this->ptr_array_AF_units_errors != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_AF_units_allocated * sizeof(T_); }
    // |END| AF unit(s). |END|
    
    // AF Ind recurrent unit(s).
    if(this->ptr_array_AF_Ind_recurrent_units_pre_AFs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_AF_Ind_recurrent_units_allocated * sizeof(T_); }
    if(this->ptr_array_AF_Ind_recurrent_units_errors != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_AF_Ind_recurrent_units_allocated * sizeof(T_); }
    if(this->ptr_array_AF_Ind_recurrent_units_dAFs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_AF_Ind_recurrent_units_allocated * sizeof(T_); }
    // |END| AF Ind recurrent unit(s). |END|
    
    // Normalized unit(s).
    if(this->ptr_array_normalized_batch_units_values_hats != nullptr) { tmp_total_size_t += batch_size_received * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_values_normalizes != nullptr) { tmp_total_size_t += batch_size_received * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_errors != nullptr) { tmp_total_size_t += batch_size_received * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    // |END| Normalized unit(s). |END|
    
    // LSTM.
    if(this->ptr_array_cells_summations_cells_inputs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_cells_summations_input_cells_inputs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_cells_summations_recurrent_cells_inputs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_blocks_summations_inputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_summations_input_inputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_summations_recurrent_inputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_summations_forgets_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_summations_input_forgets_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_summations_recurrent_forgets_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_summations_outputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_summations_input_outputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_summations_recurrent_outputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_cells_inputs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_cells_states != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_cells_states_activates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_cells_outputs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_blocks_inputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_forgets_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_outputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_cells_delta_inputs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_cells_delta_input_inputs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_cells_delta_recurrent_inputs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_cells_delta_states != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_cells_delta_outputs != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_cell_units * sizeof(T_); }
    if(this->ptr_array_blocks_delta_inputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_delta_input_inputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_delta_recurrent_inputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_delta_forgets_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_delta_input_forgets_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_delta_recurrent_forgets_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_delta_outputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_delta_input_outputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    if(this->ptr_array_blocks_delta_recurrent_outputs_gates != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_block_units * sizeof(T_); }
    // |END| LSTM. |END|
    
    return(tmp_total_size_t);
}

size_t Neural_Network::Get__Threads_Sizeof(size_t number_threads_received) const
{
    size_t tmp_total_size_t(0_zu);

    if(number_threads_received == 0_zu) { number_threads_received = this->number_threads; }
    
    if(this->ptr_array_k_sparse_activities != nullptr) { tmp_total_size_t += number_threads_received * (this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated + this->total_block_units_allocated) * sizeof(std::pair<size_t, T_>); }
    
    // Normalized unit(s).
    if(this->ptr_array_normalized_batch_units_means != nullptr) { tmp_total_size_t += number_threads_received * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_variances != nullptr) { tmp_total_size_t += number_threads_received * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_derivatives_means != nullptr) { tmp_total_size_t += number_threads_received * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_derivatives_variances != nullptr) { tmp_total_size_t += number_threads_received * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    // |END| Normalized unit(s). |END|

    // Cost.
    if(this->ptr_array_number_loss != nullptr) { tmp_total_size_t += number_threads_received * sizeof(size_t); }
    if(this->ptr_array_number_bit_fail != nullptr) { tmp_total_size_t += number_threads_received * sizeof(size_t); }
    if(this->ptr_array_loss_values != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    if(this->ptr_array_accuracy_values[0u] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    if(this->ptr_array_accuracy_values[1u] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    if(this->ptr_array_accuracy_values[2u] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    if(this->ptr_array_accuracy_values[3u] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    if(this->ptr_array_accuracy_values[4u] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    // |END| Cost. |END|

    // Parameters.
    if(this->ptr_array_derivatives_parameters != nullptr) { tmp_total_size_t += number_threads_received * this->total_parameters_allocated * sizeof(T_); }
    // |END| Parameters. |END|
    
    // Generator.
    if(this->ptr_array_Class_Generator_Bernoulli != nullptr) { tmp_total_size_t += number_threads_received * sizeof(class MyEA::Common::Class_Generator_Random_Bernoulli<T_>); }
    if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State != nullptr) { tmp_total_size_t += number_threads_received * sizeof(class MyEA::Common::Class_Generator_Random_Bernoulli<T_>); }
    if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden != nullptr) { tmp_total_size_t += number_threads_received * sizeof(class MyEA::Common::Class_Generator_Random_Bernoulli<T_>); }
    if(this->ptr_array_Class_Generator_Real_Uout != nullptr) { tmp_total_size_t += number_threads_received * sizeof(class MyEA::Common::Class_Generator_Random_Real<T_>); }
    if(this->ptr_array_Class_Generator_Real_Gaussian != nullptr) { tmp_total_size_t += number_threads_received * sizeof(class MyEA::Common::Class_Generator_Random_Gaussian<T_>); }
    // |END| Generator. |END|

    return(tmp_total_size_t);
}

size_t Neural_Network::Get__Sizeof(size_t number_threads_received, size_t batch_size_received) const
{
    size_t tmp_total_size_t(0_zu);

    tmp_total_size_t += sizeof(class Neural_Network); // this
    
    tmp_total_size_t += this->Get__Threads_Sizeof(number_threads_received == 0_zu ? this->number_threads : number_threads_received);
    
    tmp_total_size_t += this->Get__Batch_Sizeof(batch_size_received == 0_zu ? this->batch_size : batch_size_received);

    // Parameters.
    if(this->ptr_array_ptr_connections != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(void*); }

    if(this->ptr_array_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    if(this->ptr_array_mask_regularized_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }

    //    Optimizer iRPROP.
    if(this->ptr_array_previous_steps != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    if(this->ptr_array_previous_delta_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    if(this->ptr_array_previous_derivatives_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    //    |END| Optimizer iRPROP. |END|

    //    Optimizer Adam.
    if(this->ptr_array_previous_biased_first_moment != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    if(this->ptr_array_previous_biased_second_moment != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    //    |END| Optimizer Adam. |END|

    //    Optimizer AMSGrad.
    if(this->ptr_array_previous_biased_second_moment_hat != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    //    |END| Optimizer AMSGrad. |END|
    // |END| Parameters. |END|
    
    // Dropout variable.
    if(this->ptr_array_units_mask_dropout_bernoulli != nullptr) { tmp_total_size_t += (this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated) * this->number_recurrent_depth * sizeof(bool); }
    if(this->ptr_array_layers_mask_dropout_shakedrop != nullptr) { tmp_total_size_t += this->total_layers * this->batch_size * this->number_recurrent_depth * sizeof(bool); }
    if(this->ptr_array_cell_units_mask_dropout_zoneout != nullptr) { tmp_total_size_t += 2_zu * this->total_cell_units_allocated * this->number_recurrent_depth * sizeof(bool); }
    // |END| Dropout variable. |END|
    
    // Layer(s).
    if(this->ptr_array_layers != nullptr) { tmp_total_size_t += this->total_layers * sizeof(struct Layer); }
    if(this->ptr_array_layers_number_outputs != nullptr) { tmp_total_size_t += this->total_layers * sizeof(size_t); }
    if(this->ptr_array_layers_first_connection_index != nullptr) { tmp_total_size_t += this->total_layers * sizeof(size_t); }
    if(this->ptr_array_layers_last_connection_index != nullptr) { tmp_total_size_t += this->total_layers * sizeof(size_t); }
    
    if(this->ptr_array_number_neurons_by_layer != nullptr) { tmp_total_size_t += this->total_layers * sizeof(size_t); }
    if(this->ptr_array_number_connections_by_layer != nullptr) { tmp_total_size_t += this->total_layers * sizeof(size_t); }
    // |END| Layer(s). |END|

    // Neuron unit(s).
    if(this->ptr_array_neuron_units != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(struct Neuron_unit); }

    if(this->ptr_array_neuron_units_first_forward_connection_index != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_last_forward_connection_index != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_number_forward_connections != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    // |END| Neuron unit(s). |END|
    
    // AF unit(s).
    if(this->ptr_array_AF_units != nullptr) { tmp_total_size_t += this->total_AF_units_allocated * sizeof(struct AF_unit); }

    if(this->ptr_array_AF_units_activation_steepness != nullptr) { tmp_total_size_t += this->total_AF_units_allocated * sizeof(T_); }

    if(this->ptr_array_AF_units_type_activation_function != nullptr) { tmp_total_size_t += this->total_AF_units_allocated * sizeof(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION); }
    // |END| AF unit(s). |END|

    // AF Ind recurrent unit(s).
    if(this->ptr_array_AF_Ind_recurrent_units != nullptr) { tmp_total_size_t += this->total_AF_Ind_recurrent_units_allocated * sizeof(struct AF_Ind_recurrent_unit); }
    
    if(this->ptr_array_AF_Ind_recurrent_units_recurrent_connection_index != nullptr) { tmp_total_size_t += this->total_AF_Ind_recurrent_units_allocated * sizeof(size_t); }
    
    if(this->ptr_array_AF_Ind_recurrent_units_activation_steepness != nullptr) { tmp_total_size_t += this->total_AF_Ind_recurrent_units_allocated * sizeof(T_); }

    if(this->ptr_array_AF_Ind_recurrent_units_type_activation_function != nullptr) { tmp_total_size_t += this->total_AF_Ind_recurrent_units_allocated * sizeof(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION); }
    // |END| AF Ind recurrent unit(s). |END|
    
    // Cell unit(s).
    if(this->ptr_array_cell_units != nullptr) { tmp_total_size_t += this->total_cell_units_allocated * sizeof(struct Cell_unit); }

    // Block unit(s).
    if(this->ptr_array_block_units != nullptr) { tmp_total_size_t += this->total_block_units_allocated * sizeof(struct Block_unit); }

    // Normalized unit(s).
    if(this->ptr_array_normalized_units != nullptr) { tmp_total_size_t += this->total_normalized_units_allocated * sizeof(union Normalized_unit); }

    if(this->ptr_array_normalized_batch_units_r_corrections != nullptr) { tmp_total_size_t += this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_d_corrections != nullptr) { tmp_total_size_t += this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_means_averages != nullptr) { tmp_total_size_t += this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_variances_averages != nullptr) { tmp_total_size_t += this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_); }
    // |END| Normalized unit(s). |END|
    
    // CUDA
#if defined(COMPILE_CUDA)
    if(this->ptr_device_Neural_Network != NULL) { tmp_total_size_t += this->ptr_device_Neural_Network->Get__Sizeof(); }
#endif
    // |END| CUDA |END|
    
    return(tmp_total_size_t);
}

T_ const *Layer::Get__Array_Summations__Cell__Block_Input__Input__Activation(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_normalized_units[0u].normalized_batch_units.ptr_array_values_normalizes); }
                else
                { return(this->ptr_array_cell_units->ptr_summation_input_cell_input); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_normalized_units[1u].normalized_batch_units.ptr_array_values_normalizes); }
                else
                { return(this->ptr_array_cell_units->ptr_summation_recurrent_cell_input); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Summations__Cell__Cell_State__Activation(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_normalized_units[2u].normalized_batch_units.ptr_array_values_normalizes); }
                else
                { return(this->ptr_array_cell_units->ptr_cell_state); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Summations__Block__Input_Gate__Input__Activation(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_normalized_units[3u].normalized_batch_units.ptr_array_values_normalizes); }
                else
                { return(this->ptr_array_block_units->ptr_summation_input_inputs_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_normalized_units[4u].normalized_batch_units.ptr_array_values_normalizes); }
                else
                { return(this->ptr_array_block_units->ptr_summation_recurrent_inputs_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Summations__Block__Forget_Gate__Input__Activation(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_normalized_units[5u].normalized_batch_units.ptr_array_values_normalizes); }
                else
                { return(this->ptr_array_block_units->ptr_summation_input_forgets_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_normalized_units[6u].normalized_batch_units.ptr_array_values_normalizes); }
                else
                { return(this->ptr_array_block_units->ptr_summation_recurrent_forgets_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Summations__Block__Output_Gate__Input__Activation(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_normalized_units[7u].normalized_batch_units.ptr_array_values_normalizes); }
                else
                { return(this->ptr_array_block_units->ptr_summation_input_outputs_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_normalized_units[8u].normalized_batch_units.ptr_array_values_normalizes); }
                else
                { return(this->ptr_array_block_units->ptr_summation_recurrent_outputs_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Deltas__Cell__Block_Input__Input(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_cell_units->ptr_delta_cell_input_input); }
                else
                { return(this->ptr_array_cell_units->ptr_delta_cell_input); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Deltas__Cell__Block_Input__Recurrent(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_cell_units->ptr_delta_cell_recurrent_input); }
                else
                { return(this->ptr_array_cell_units->ptr_delta_cell_input); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Deltas__Block__Input_Gate__Input(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_block_units->ptr_delta_input_inputs_gates); }
                else
                { return(this->ptr_array_block_units->ptr_delta_inputs_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Deltas__Block__Input_Gate__Recurrent(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_block_units->ptr_delta_recurrent_inputs_gates); }
                else
                { return(this->ptr_array_block_units->ptr_delta_inputs_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Deltas__Block__Forget_Gate__Input(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_block_units->ptr_delta_input_forgets_gates); }
                else
                { return(this->ptr_array_block_units->ptr_delta_forgets_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Deltas__Block__Forget_Gate__Recurrent(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_block_units->ptr_delta_recurrent_forgets_gates); }
                else
                { return(this->ptr_array_block_units->ptr_delta_forgets_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Deltas__Block__Output_Gate__Input(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_block_units->ptr_delta_input_outputs_gates); }
                else
                { return(this->ptr_array_block_units->ptr_delta_outputs_gates); }
        default: return(nullptr);
    }
}

T_ const *Layer::Get__Array_Deltas__Block__Output_Gate__Recurrent(void) const
{
    switch(this->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                if(this->Use__Normalization())
                { return(this->ptr_array_block_units->ptr_delta_recurrent_outputs_gates); }
                else
                { return(this->ptr_array_block_units->ptr_delta_outputs_gates); }
        default: return(nullptr);
    }
}

void Neural_Network::Update_Parameter(size_t const batch_size_received, size_t const training_size_received)
{
    if(this->use_OpenMP && this->is_OpenMP_initialized)
    { this->Update_Parameter__OpenMP(batch_size_received, training_size_received); }
    else
    { this->Update_Parameter__Loop(batch_size_received, training_size_received); }
}

void Neural_Network::Update_Parameter__OpenMP(size_t const batch_size_received, size_t const training_size_received)
{
    if(this->Get__Regularization__L1() != 0_T)
    {
        this->Update_Derivative_Weight__Regularization__L1__OpenMP(batch_size_received,
                                                                                                    0_zu,
                                                                                                    this->total_weights);
    }

    if(this->Get__Regularization__L2() != 0_T)
    {
        this->Update_Derivative_Weight__Regularization__L2__OpenMP(batch_size_received,
                                                                                                    0_zu,
                                                                                                    this->total_weights);
    }
    
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND: 
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: 
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Merge_Derivatives_Parameters(0_zu, this->total_parameters); break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Optimizer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->type_optimizer_function,
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str(),
                                     __LINE__);
                break;
    }
    
    this->Plot__Gradient();

    if(this->Use__Clip_Gradient()) { this->Clip_Gradient__OpenMP(0_zu, this->total_parameters); }

    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
            this->Update_Parameter__Gradient_Descent(batch_size_received,
                                                                              training_size_received,
                                                                              0_zu,
                                                                              this->total_parameters);
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: this->Update_Parameter__iRPROP_plus(0_zu, this->total_parameters); break;
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP: Update_Weight_QuickProp_Parallel(this, this->Get__Number_Examples(), 0_zu, this->total_parameters); break;
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP: Update_Weight_SARProp_Parallel(this, this->sarprop_epoch, 0_zu, this->total_parameters); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
            this->Update_Parameters__AdaBound(batch_size_received,
                                                                      training_size_received,
                                                                      0_zu,
                                                                      this->total_parameters);
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
            this->Update_Parameters__Adam(batch_size_received,
                                                              training_size_received,
                                                              0_zu,
                                                              this->total_parameters);
                break;
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SADAMAX: this->Update_Weight_AdaMax(0_zu, this->total_parameters); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
            this->Update_Parameters__AMSBound(batch_size_received,
                                                                      training_size_received,
                                                                      0_zu,
                                                                      this->total_parameters);
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad:
            this->Update_Parameters__AMSGrad(batch_size_received,
                                                                    training_size_received,
                                                                    0_zu,
                                                                    this->total_parameters);
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM:
            this->Update_Parameters__NosAdam(batch_size_received,
                                                                    training_size_received,
                                                                    0_zu,
                                                                    this->total_parameters);
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Optimizer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->type_optimizer_function,
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str(),
                                     __LINE__);
                break;
    }

    if(this->Get__Regularization__Max_Norm_Constraints() != 0_T) { this->Update_Weight_Regularization__Max_Norm_Constraints__OpenMP(0_zu, this->total_weights); }
    
    if(this->Use__Regularization__Constraint_Recurrent_Weight()) { this->Update_Weight_Regularization__Constraint_Recurrent_Weight(0_zu, this->total_weights); }

    if(this->Use__Tied_Parameter()) { this->Tied__Transpose(); }
}

void Neural_Network::Update_Parameter__Loop(size_t const batch_size_received, size_t const training_size_received)
{
    if(this->Get__Regularization__L1() != 0_T)
    {
        this->Update_Derivative_Weight__Regularization__L1__Loop(batch_size_received,
                                                                                                0_zu,
                                                                                                this->total_weights);
    }

    if(this->Get__Regularization__L2() != 0_T)
    {
        this->Update_Derivative_Weight__Regularization__L2__Loop(batch_size_received,
                                                                                                0_zu,
                                                                                                this->total_weights);
    }
    
    this->Plot__Gradient();
    
    if(this->Use__Clip_Gradient()) { this->Clip_Gradient__Loop(0_zu, this->total_parameters); }

    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
            this->Update_Parameter__Gradient_Descent(batch_size_received,
                                                                              training_size_received,
                                                                              0_zu,
                                                                              this->total_parameters);
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: this->Update_Parameter__iRPROP_plus(0_zu, this->total_parameters); break;
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP: Update_Weight_QuickProp(this, this->Get__Number_Examples(), 0_zu, this->total_parameters); break;
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP: Update_Weight_SARProp(this, this->sarprop_epoch, 0_zu, this->total_parameters); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
            this->Update_Parameters__AdaBound(batch_size_received,
                                                                      training_size_received,
                                                                      0_zu,
                                                                      this->total_parameters);
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
            this->Update_Parameters__Adam(batch_size_received,
                                                              training_size_received,
                                                              0_zu,
                                                              this->total_parameters);
                break;
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        //case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SADAMAX: this->Update_Weight_AdaMax(0_zu, this->total_parameters); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
            this->Update_Parameters__AMSBound(batch_size_received,
                                                                      training_size_received,
                                                                      0_zu,
                                                                      this->total_parameters);
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad:
            this->Update_Parameters__AMSGrad(batch_size_received,
                                                                    training_size_received,
                                                                    0_zu,
                                                                    this->total_parameters);
                break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM:
            this->Update_Parameters__NosAdam(batch_size_received,
                                                                    training_size_received,
                                                                    0_zu,
                                                                    this->total_parameters);
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Optimizer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->type_optimizer_function,
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str(),
                                     __LINE__);
                break;
    }

    if(this->Get__Regularization__Max_Norm_Constraints() != 0_T) { this->Update_Weight_Regularization__Max_Norm_Constraints__Loop(0_zu, this->total_weights); }
    
    if(this->Use__Regularization__Constraint_Recurrent_Weight()) { this->Update_Weight_Regularization__Constraint_Recurrent_Weight(0_zu, this->total_weights); }

    if(this->Use__Tied_Parameter()) { this->Tied__Transpose(); }
}

struct Layer const *Neural_Network::Get__Layer(size_t const index_received) const { return(this->ptr_array_layers + index_received); }

struct Layer const *Neural_Network::Get__End_Layer__Active(void) const
{
    struct Layer *tmp_ptr_last_layer_active;

    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER: tmp_ptr_last_layer_active = this->ptr_last_layer - ((this->total_layers - 3_zu) / 2_zu + 1_zu); break;
        default: tmp_ptr_last_layer_active = this->ptr_last_layer; break;
    }
    
    return(tmp_ptr_last_layer_active);
}

/* Strategy comparison index:
    [0], default: tr_validating < td_validating && tr_testing <= td_testing || tr_validating <= td_validating && tr_testing < td_testing.
    [1]: tr_testing <= global_testing.
    [2]: tr_validating <= td_testing && tr_testing < td_testing. */
bool Neural_Network::Strategy_Comparison__Loss(unsigned int const strategy_index_received,
                                                                           enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_in_received,
                                                                           enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_out_received,
                                                                           class Neural_Network const *const ptr_source_Neural_Network_received) const
{
    switch(strategy_index_received)
    {
        case 0u:
        default:
            if((ptr_source_Neural_Network_received->Get__Loss(type_dataset_in_received) < this->Get__Loss(type_dataset_in_received)
               &&
               ptr_source_Neural_Network_received->Get__Loss(type_dataset_out_received) <= this->Get__Loss(type_dataset_out_received))
               ||
               (ptr_source_Neural_Network_received->Get__Loss(type_dataset_in_received) <= this->Get__Loss(type_dataset_in_received)
               &&
               ptr_source_Neural_Network_received->Get__Loss(type_dataset_out_received) < this->Get__Loss(type_dataset_out_received)))
            { return(true); }
                break;
        case 1u:
            if(ptr_source_Neural_Network_received->Get__Loss(type_dataset_in_received) <= this->Get__Loss(type_dataset_out_received)
               &&
               ptr_source_Neural_Network_received->Get__Loss(type_dataset_out_received) < this->Get__Loss(type_dataset_out_received))
            { return(true); }
                break;
    }

    return(false);
}

bool Neural_Network::Strategy_Comparison__Accuracy(unsigned int const strategy_index_received,
                                                                                 enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_in_received,
                                                                                 enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_out_received,
                                                                                 class Neural_Network const *const ptr_source_Neural_Network_received) const
{
    switch(strategy_index_received)
    {
        case 0u:
        default:
            if((ptr_source_Neural_Network_received->Get__Accuracy(type_dataset_in_received) > this->Get__Accuracy(type_dataset_in_received)
               &&
               ptr_source_Neural_Network_received->Get__Accuracy(type_dataset_out_received) >= this->Get__Accuracy(type_dataset_out_received))
               ||
               (ptr_source_Neural_Network_received->Get__Accuracy(type_dataset_in_received) >= this->Get__Accuracy(type_dataset_in_received)
               &&
               ptr_source_Neural_Network_received->Get__Accuracy(type_dataset_out_received) > this->Get__Accuracy(type_dataset_out_received)))
            { return(true); }
                break;
        case 1u:
            if(ptr_source_Neural_Network_received->Get__Accuracy(type_dataset_in_received) >= this->Get__Accuracy(type_dataset_out_received)
               &&
               ptr_source_Neural_Network_received->Get__Accuracy(type_dataset_out_received) > this->Get__Accuracy(type_dataset_out_received))
            { return(true); }
                break;
    }

    return(false);
}

bool Neural_Network::Compare(bool const use_metric_loss_received,
                                              bool const dataset_in_equal_less_dataset_out_accepted_received,
                                              enum MyEA::Common::ENUM_TYPE_DATASET const type_holdout_dataset_received,
                                              T_ const minimum_loss_holdout_dataset_accepted_received,
                                              class Neural_Network const *const ptr_source_Neural_Network_received) const
{
    enum MyEA::Common::ENUM_TYPE_DATASET tmp_type_dataset_in,
                                                                            tmp_type_dataset_out;
    
    switch(type_holdout_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
            tmp_type_dataset_in = MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING;
            tmp_type_dataset_out = MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING;
                break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
            tmp_type_dataset_in = MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING;
            tmp_type_dataset_out = MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION;
                break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING:
            tmp_type_dataset_in = MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION;
            tmp_type_dataset_out = MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING;
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Evaluation type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     type_holdout_dataset_received,
                                     MyEA::Common::ENUM_TYPE_DATASET_NAMES[type_holdout_dataset_received].c_str(),
                                     __LINE__);
                return(false);
    }

    if(use_metric_loss_received)
    {
        if(ptr_source_Neural_Network_received->Get__Loss(tmp_type_dataset_out) <= minimum_loss_holdout_dataset_accepted_received
           &&
           (
                (
                    dataset_in_equal_less_dataset_out_accepted_received == false
                    &&
                    this->Strategy_Comparison__Loss(0u, tmp_type_dataset_in, tmp_type_dataset_out, ptr_source_Neural_Network_received)
                )
                ||
                (
                    dataset_in_equal_less_dataset_out_accepted_received
                    &&
                    (
                        this->Strategy_Comparison__Loss(1u, tmp_type_dataset_in, tmp_type_dataset_out, ptr_source_Neural_Network_received)
                        ||
                        (
                            this->Get__Loss(tmp_type_dataset_in) > this->Get__Loss(tmp_type_dataset_out)
                            &&
                            this->Strategy_Comparison__Loss(0u, tmp_type_dataset_in, tmp_type_dataset_out, ptr_source_Neural_Network_received)
                        )
                    )
                )
           )
           )
        { return(true); }
        else
        { return(false); }
    }
    else
    {
        if(ptr_source_Neural_Network_received->Get__Accuracy(tmp_type_dataset_out) >= minimum_loss_holdout_dataset_accepted_received
           &&
           (
                (
                    dataset_in_equal_less_dataset_out_accepted_received == false
                    &&
                    this->Strategy_Comparison__Accuracy(0u, tmp_type_dataset_in, tmp_type_dataset_out, ptr_source_Neural_Network_received)
                )
                ||
                (
                    dataset_in_equal_less_dataset_out_accepted_received
                    &&
                    (
                        this->Strategy_Comparison__Accuracy(1u, tmp_type_dataset_in, tmp_type_dataset_out, ptr_source_Neural_Network_received)
                        ||
                        (
                            this->Get__Accuracy(tmp_type_dataset_in) < this->Get__Accuracy(tmp_type_dataset_out)
                            &&
                            this->Strategy_Comparison__Accuracy(0u, tmp_type_dataset_in, tmp_type_dataset_out, ptr_source_Neural_Network_received)
                        )
                    )
                )
           )
           )
        { return(true); }
        else
        { return(false); }
    }
}

void Neural_Network::Initialize__Constant__Bias(T_ const bias_received, struct Layer const *const ptr_layer_it_received)
{
    T_ const *const tmp_ptr_parameter_end(this->ptr_array_parameters + ptr_layer_it_received->last_bias_connection_index);
    T_ *tmp_ptr_parameter_it(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index);

    for(; tmp_ptr_parameter_it != tmp_ptr_parameter_end; ++tmp_ptr_parameter_it) { *tmp_ptr_parameter_it = bias_received; }
}

void Neural_Network::Initialize__Constant__LSTM__Bias(T_ const bias_received, struct Layer const *const ptr_layer_it_received)
{
    size_t const tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units)),
                        tmp_number_block_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units));

    if(tmp_number_cell_units * tmp_number_block_units != 0_zu)
    {
        T_ const *tmp_ptr_parameter_end;
        T_ *tmp_ptr_parameter_it(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index);

        // Cell input && Input gate.
        for(tmp_ptr_parameter_end = tmp_ptr_parameter_it + tmp_number_cell_units + tmp_number_block_units; tmp_ptr_parameter_it != tmp_ptr_parameter_end; ++tmp_ptr_parameter_it) { *tmp_ptr_parameter_it = bias_received; }
        // |END| Cell input && Input gate. |END|
        
        // Forget gate.
        for(tmp_ptr_parameter_end = tmp_ptr_parameter_it + tmp_number_block_units; tmp_ptr_parameter_it != tmp_ptr_parameter_end; ++tmp_ptr_parameter_it) { *tmp_ptr_parameter_it = 1_T; }
        // |END| Forget gate. |END|

        // Output gate.
        for(tmp_ptr_parameter_end = tmp_ptr_parameter_it + tmp_number_block_units + tmp_number_block_units; tmp_ptr_parameter_it != tmp_ptr_parameter_end; ++tmp_ptr_parameter_it) { *tmp_ptr_parameter_it = bias_received; }
        // |END| Output gate. |END|
    }
}

void Neural_Network::Initialize__Uniform(T_ *ptr_array_weights_received,
                                                          T_ const *const ptr_last_weight_received,
                                                          T_ const lower_bound_received,
                                                          T_ const upper_bound_received)
{
    this->Class_Generator_Real.Range(lower_bound_received, upper_bound_received);
    
    for(; ptr_array_weights_received != ptr_last_weight_received; ++ptr_array_weights_received)  { *ptr_array_weights_received = this->Class_Generator_Real.Generate_Real(); }
}

void Neural_Network::Initialize__Uniform__LSTM(T_ const lower_bound_received[5u],
                                                                      T_ const upper_bound_received[5u],
                                                                      struct Layer const *const ptr_layer_it_received)
{
    struct Block_unit const *tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;

    size_t const tmp_number_peephole_connections(tmp_ptr_block_unit_it->last_index_peephole_input_gate - tmp_ptr_block_unit_it->first_index_peephole_input_gate),
                       tmp_number_feedforward_connections(tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrent_connections(tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index;
    
    T_ *tmp_ptr_array_parameters;
    
    // Loop through each blocks.
    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        // Loop through each cells.
        for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
        {
            // Input, cell.
            this->Class_Generator_Real.Range(lower_bound_received[0u], upper_bound_received[0u]);
            
            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index)
            { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
            // |END| Input, cell. |END|
            
            // Recurrent, cell.
            this->Class_Generator_Real.Range(lower_bound_received[2u], upper_bound_received[2u]);
            
            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index)
            { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
            // |END| Recurrent, cell. |END|
        }

        // Input, gates.
        this->Class_Generator_Real.Range(lower_bound_received[1u], upper_bound_received[1u]);
        
        //  Input gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
        //  |END| Input gate. |END|

        //  Forget gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
        //  |END| Forget gate. |END|

        //  Output gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
        //  |END| Output gate. |END|
        // |END| Input, gates. |END|
        
        // Recurrent, gates.
        this->Class_Generator_Real.Range(lower_bound_received[3u], upper_bound_received[3u]);
        
        //  Input gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
        //  |END| Input gate. |END|

        //  Forget gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
        //  |END| Forget gate. |END|

        //  Output gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
        //  |END| Output gate. |END|
        // |END| Recurrent, gates. |END|

    #ifndef NO_PEEPHOLE
        this->Class_Generator_Real.Range(lower_bound_received[4u], upper_bound_received[4u]);
        
        // Peepholes.
        //  Input gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
        //  |END| Input gate. |END|
        
        //  Forget gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
        {tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
        //  |END| Forget gate. |END|
        
        //  Output gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
        { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real.Generate_Real(); }
        //  |END| Output gate. |END|
        // |END| Peepholes. |END|
    #endif
    }
}

void Neural_Network::Initialize__Uniform__AF_Ind_Recurrent(struct Layer const *const ptr_layer_it_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);

    size_t const tmp_number_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit));
    
    T_ *tmp_ptr_weight_it(this->ptr_array_parameters + *tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index),
         tmp_lower_bound,
         tmp_upper_bound;
    T_ const *const tmp_ptr_weight_end(tmp_ptr_weight_it + tmp_number_units),
                  tmp_MAG(MyEA::Math::Clip<T_>(this->clip_gradient, 2_T, 10_T));

    switch(ptr_layer_it_received->type_activation)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
            tmp_lower_bound = -pow(tmp_MAG / pow(0.9_T, static_cast<T_>(this->number_recurrent_depth) / 10_T), 1_T / static_cast<T_>(this->number_recurrent_depth));
            tmp_upper_bound = pow(tmp_MAG / pow(0.9_T, static_cast<T_>(this->number_recurrent_depth) / 10_T), 1_T / static_cast<T_>(this->number_recurrent_depth));
                break;
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
            tmp_lower_bound = 0_T;
            tmp_upper_bound = pow(tmp_MAG, 1_T / static_cast<T_>(this->number_recurrent_depth));
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer activation type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_activation,
                                     MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_it_received->type_activation].c_str(),
                                     __LINE__);
                return;
    }

    this->Class_Generator_Real.Range(-tmp_lower_bound, tmp_upper_bound);
    
    for(; tmp_ptr_weight_it != tmp_ptr_weight_end; ++tmp_ptr_weight_it) { *tmp_ptr_weight_it = this->Class_Generator_Real.Generate_Real(); }
}

void Neural_Network::Initialize__Uniform__AF_Ind_Recurrent__Long_Term_Memory(void)
{
    T_ const tmp_MAG(MyEA::Math::Clip<T_>(this->clip_gradient, 2_T, 10_T));
    T_ tmp_lower_bound,
         tmp_upper_bound;

    struct Layer *tmp_ptr_layer_it;

    // Loop though each layer (backward).
    for(tmp_ptr_layer_it = this->ptr_last_layer - 2; tmp_ptr_layer_it > this->ptr_array_layers; --tmp_ptr_layer_it)
    {
        if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            switch(tmp_ptr_layer_it->type_activation)
            {
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
                    tmp_lower_bound = pow(1_T / tmp_MAG / pow(0.9_T, static_cast<T_>(this->number_recurrent_depth) / 10_T), 1_T / static_cast<T_>(this->number_recurrent_depth));
                    tmp_upper_bound = pow(tmp_MAG / pow(0.9_T, static_cast<T_>(this->number_recurrent_depth) / 10_T), 1_T / static_cast<T_>(this->number_recurrent_depth));
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
                case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
                    tmp_lower_bound = pow(1_T / tmp_MAG, 1_T / static_cast<T_>(this->number_recurrent_depth));
                    tmp_upper_bound = pow(tmp_MAG, 1_T / static_cast<T_>(this->number_recurrent_depth));
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer activation type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_layer_it->type_activation,
                                             MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[tmp_ptr_layer_it->type_activation].c_str(),
                                             __LINE__);
                        return;
            }
            
            struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit(tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units);

            size_t const tmp_number_units(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit));
            
            T_ *tmp_ptr_array_weights(this->ptr_array_parameters + *tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index);
            T_ const *const tmp_ptr_last_weight(tmp_ptr_array_weights + tmp_number_units);
            
            this->Class_Generator_Real.Range(tmp_lower_bound, tmp_upper_bound);
            
            // Recurrent connection(s).
            do { *tmp_ptr_array_weights = this->Class_Generator_Real.Generate_Real();
            } while(++tmp_ptr_array_weights != tmp_ptr_last_weight);

            break;
        }
    }
}

void Neural_Network::Initialize__Gaussian(T_ *ptr_array_weights_received,
                                                             T_ const *const ptr_last_weight_received,
                                                             T_ const variance_received)
{
    this->Class_Generator_Gaussian.Range(0_T, variance_received);
    
    for(; ptr_array_weights_received != ptr_last_weight_received; ++ptr_array_weights_received) { *ptr_array_weights_received = this->Class_Generator_Gaussian.Generate_Gaussian(); }
}

void Neural_Network::Initialize__Gaussian__LSTM(T_ const feedforward_cell_variance_received,
                                                                         T_ const feedforward_gates_variance_received,
                                                                         T_ const recurrent_cell_variance_received,
                                                                         T_ const recurrent_gates_variance_received,
                                                                         T_ const peephole_variance_received,
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
    
    T_ *tmp_ptr_array_parameters;
    
    // Loop through each blocks.
    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        // Loop through each cells.
        for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
        {
            // Input, cell.
            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;
            
            this->Class_Generator_Gaussian.Range(0_T, feedforward_cell_variance_received);
            
            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index) { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
            // |END| Input, cell. |END|
            
            // Recurrent, cell.
            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;
            
            this->Class_Generator_Gaussian.Range(0_T, recurrent_cell_variance_received);
            
            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index) { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
            // |END| Recurrent, cell. |END|
        }

        // Input, gates.
        this->Class_Generator_Gaussian.Range(0_T, feedforward_gates_variance_received);
        
        //  Input gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index) { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
        //  |END| Input gate. |END|

        //  Forget gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index) { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
        //  |END| Forget gate. |END|

        //  Output gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index) { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
        //  |END| Output gate. |END|
        // |END| Input, gates. |END|
        
        // Recurrent, gates.
        this->Class_Generator_Gaussian.Range(0_T, recurrent_gates_variance_received);
        
        //  Input gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index) { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
        //  |END| Input gate. |END|

        //  Forget gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index) { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
        //  |END| Forget gate. |END|

        //  Output gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index) { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
        //  |END| Output gate. |END|
        // |END| Recurrent, gates. |END|
        
    #ifndef NO_PEEPHOLE
        // Peepholes.
        this->Class_Generator_Gaussian.Range(0_T, peephole_variance_received);
        
        //  Input gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index) { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
        //  |END| Input gate. |END|
        
        //  Forget gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index) {tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
        //  |END| Forget gate. |END|
        
        //  Output gate.
        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index) { tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Gaussian.Generate_Gaussian(); }
        //  |END| Output gate. |END|
        // |END| Peepholes. |END|
    #endif
    }
}
