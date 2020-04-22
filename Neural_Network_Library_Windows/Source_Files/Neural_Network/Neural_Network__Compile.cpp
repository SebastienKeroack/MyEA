#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

bool Neural_Network::Assign__Layers(struct Layer_Parameters const *const ptr_array_layers_received)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
    
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if(this->Assign__Layer(tmp_ptr_layer_it, ptr_array_layers_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Layer()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

bool Neural_Network::Assign__Layer(struct Layer *&ptr_layer_it_received, struct Layer_Parameters const *const ptr_array_layers_received)
{
    size_t const tmp_layer_index(static_cast<size_t>(ptr_layer_it_received - this->ptr_array_layers));

    ptr_layer_it_received->type_layer = ptr_array_layers_received[tmp_layer_index].type_layer;
    
    ptr_layer_it_received->use_bidirectional = ptr_array_layers_received[tmp_layer_index].use_bidirectional;

    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            ptr_layer_it_received->pooling_values[0u] = ptr_array_layers_received[tmp_layer_index].unit_parameters[0u]; // Kernel size.
            ptr_layer_it_received->pooling_values[1u] = ptr_array_layers_received[tmp_layer_index].unit_parameters[1u]; // Stride.
            ptr_layer_it_received->pooling_values[2u] = ptr_array_layers_received[tmp_layer_index].unit_parameters[2u]; // Padding.
            ptr_layer_it_received->pooling_values[3u] = ptr_array_layers_received[tmp_layer_index].unit_parameters[3u]; // Dilation.
            ptr_layer_it_received->pooling_values[4u] = ptr_array_layers_received[tmp_layer_index].unit_parameters[4u]; // Ceil mode.
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            *ptr_layer_it_received->ptr_number_outputs = ptr_array_layers_received[tmp_layer_index].unit_parameters[0u];

            ptr_layer_it_received->ptr_last_neuron_unit = ptr_layer_it_received->ptr_array_neuron_units + *ptr_layer_it_received->ptr_number_outputs;
            ptr_layer_it_received->ptr_last_AF_unit = ptr_layer_it_received->ptr_array_AF_units + *ptr_layer_it_received->ptr_number_outputs;

            this->total_neuron_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - ptr_layer_it_received->ptr_array_neuron_units);
            this->total_AF_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_unit - ptr_layer_it_received->ptr_array_AF_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            *ptr_layer_it_received->ptr_number_outputs = ptr_array_layers_received[tmp_layer_index].unit_parameters[0u];

            ptr_layer_it_received->ptr_last_neuron_unit = ptr_layer_it_received->ptr_array_neuron_units + *ptr_layer_it_received->ptr_number_outputs;
            ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit = ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units + *ptr_layer_it_received->ptr_number_outputs;

            this->total_neuron_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - ptr_layer_it_received->ptr_array_neuron_units);
            this->total_AF_Ind_recurrent_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit - ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            *ptr_layer_it_received->ptr_number_outputs = ptr_array_layers_received[tmp_layer_index].unit_parameters[0u] * ptr_array_layers_received[tmp_layer_index].unit_parameters[1u];

            if(ptr_array_layers_received[tmp_layer_index].use_bidirectional)
            {
                ptr_layer_it_received->ptr_last_block_unit = ptr_layer_it_received->ptr_array_block_units + ptr_array_layers_received[tmp_layer_index].unit_parameters[0u] * 2u;
                ptr_layer_it_received->ptr_last_cell_unit = ptr_layer_it_received->ptr_array_cell_units + ptr_array_layers_received[tmp_layer_index].unit_parameters[0u] * ptr_array_layers_received[tmp_layer_index].unit_parameters[1u] * 2u;
            }
            else
            {
                ptr_layer_it_received->ptr_last_block_unit = ptr_layer_it_received->ptr_array_block_units + ptr_array_layers_received[tmp_layer_index].unit_parameters[0u];
                ptr_layer_it_received->ptr_last_cell_unit = ptr_layer_it_received->ptr_array_cell_units + ptr_array_layers_received[tmp_layer_index].unit_parameters[0u] * ptr_array_layers_received[tmp_layer_index].unit_parameters[1u];
            }
            
            this->total_block_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units);
            this->total_cell_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
            ptr_layer_it_received->block_depth = ptr_array_layers_received[tmp_layer_index].unit_parameters[0u];
            
            if(this->Assign__Residual_Block(ptr_layer_it_received, ptr_array_layers_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Residual_Block()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str());
                return(false);
    }

    return(true);
}

bool Neural_Network::Assign__Residual_Block(struct Layer *&ptr_layer_it_received, struct Layer_Parameters const *const ptr_array_layers_received)
{
    if(ptr_layer_it_received->type_layer != MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is not a residual unit. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    struct Layer const *const tmp_ptr_residual_block_end(ptr_layer_it_received + ptr_layer_it_received->block_depth + 1);
    
    for(++ptr_layer_it_received; ptr_layer_it_received != tmp_ptr_residual_block_end; ++ptr_layer_it_received)
    {
        if(this->Assign__Residual_Layer(ptr_layer_it_received, ptr_array_layers_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Residual_Layer()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    
    // Assign layer iterator to the last layer inside the block.
    --ptr_layer_it_received;

    return(true);
}

bool Neural_Network::Assign__Residual_Layer(struct Layer *&ptr_layer_it_received, struct Layer_Parameters const *const ptr_array_layers_received)
{
    size_t const tmp_layer_index(static_cast<size_t>(ptr_layer_it_received - this->ptr_array_layers));

    ptr_layer_it_received->type_layer = ptr_array_layers_received[tmp_layer_index].type_layer;
    
    ptr_layer_it_received->use_bidirectional = ptr_array_layers_received[tmp_layer_index].use_bidirectional;

    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            ptr_layer_it_received->pooling_values[0u] = ptr_array_layers_received[tmp_layer_index].unit_parameters[0u]; // Kernel size.
            ptr_layer_it_received->pooling_values[1u] = ptr_array_layers_received[tmp_layer_index].unit_parameters[1u]; // Stride.
            ptr_layer_it_received->pooling_values[2u] = ptr_array_layers_received[tmp_layer_index].unit_parameters[2u]; // Padding.
            ptr_layer_it_received->pooling_values[3u] = ptr_array_layers_received[tmp_layer_index].unit_parameters[3u]; // Dilation.
            ptr_layer_it_received->pooling_values[4u] = ptr_array_layers_received[tmp_layer_index].unit_parameters[4u]; // Ceil mode.
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            *ptr_layer_it_received->ptr_number_outputs = ptr_array_layers_received[tmp_layer_index].unit_parameters[0u];

            ptr_layer_it_received->ptr_last_neuron_unit = ptr_layer_it_received->ptr_array_neuron_units + *ptr_layer_it_received->ptr_number_outputs;

            this->total_neuron_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_neuron_unit - ptr_layer_it_received->ptr_array_neuron_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            *ptr_layer_it_received->ptr_number_outputs = ptr_array_layers_received[tmp_layer_index].unit_parameters[0u] * ptr_array_layers_received[tmp_layer_index].unit_parameters[1u];

            if(ptr_array_layers_received[tmp_layer_index].use_bidirectional)
            {
                ptr_layer_it_received->ptr_last_block_unit = ptr_layer_it_received->ptr_array_block_units + ptr_array_layers_received[tmp_layer_index].unit_parameters[0u] * 2u;
                ptr_layer_it_received->ptr_last_cell_unit = ptr_layer_it_received->ptr_array_cell_units + ptr_array_layers_received[tmp_layer_index].unit_parameters[0u] * ptr_array_layers_received[tmp_layer_index].unit_parameters[1u] * 2u;
            }
            else
            {
                ptr_layer_it_received->ptr_last_block_unit = ptr_layer_it_received->ptr_array_block_units + ptr_array_layers_received[tmp_layer_index].unit_parameters[0u];
                ptr_layer_it_received->ptr_last_cell_unit = ptr_layer_it_received->ptr_array_cell_units + ptr_array_layers_received[tmp_layer_index].unit_parameters[0u] * ptr_array_layers_received[tmp_layer_index].unit_parameters[1u];
            }
            
            this->total_block_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units);
            this->total_cell_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
            ptr_layer_it_received->block_depth = ptr_array_layers_received[tmp_layer_index].unit_parameters[0u];
            
            if(this->Assign__Residual_Block(ptr_layer_it_received, ptr_array_layers_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Residual_Block()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str());
                return(false);
    }

    return(true);
}

bool Neural_Network::Assign__Post__Layers(void)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1); // Subtract output layer.
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1); // Skip input layer.
    
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if(this->Assign__Post__Layer(tmp_ptr_layer_it) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Post__Layer()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

bool Neural_Network::Assign__Post__Layer(struct Layer *&ptr_layer_it_received)
{
    struct Layer const *const tmp_ptr_previous_layer_connected(ptr_layer_it_received->previous_connected_layers[0u]),
                               *tmp_ptr_residual_block_last_layer;

    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            // Output_Size = floor((Input_Size + 2 * Padding - Kernel) / Stride + 1)
            if(ptr_layer_it_received->pooling_values[4u] == 0_zu)
            { *ptr_layer_it_received->ptr_number_outputs = static_cast<size_t>(floor(static_cast<double>(*tmp_ptr_previous_layer_connected->ptr_number_outputs + 2_zu * ptr_layer_it_received->pooling_values[2u] - ptr_layer_it_received->pooling_values[0u]) / static_cast<double>(ptr_layer_it_received->pooling_values[1u]) + 1.0)); }
            else
            { *ptr_layer_it_received->ptr_number_outputs = static_cast<size_t>(ceil(static_cast<double>(*tmp_ptr_previous_layer_connected->ptr_number_outputs + 2_zu * ptr_layer_it_received->pooling_values[2u] - ptr_layer_it_received->pooling_values[0u]) / static_cast<double>(ptr_layer_it_received->pooling_values[1u]) + 1.0)); }

            ptr_layer_it_received->ptr_last_basic_unit = ptr_layer_it_received->ptr_array_basic_units + *ptr_layer_it_received->ptr_number_outputs;

            this->total_basic_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_basic_unit - ptr_layer_it_received->ptr_array_basic_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            // Output_Size = floor((Input_Size + 2 * Padding - Dilation * (Kernel - 1) - 1) / Stride + 1)
            if(ptr_layer_it_received->pooling_values[4u] == 0_zu)
            { *ptr_layer_it_received->ptr_number_outputs = static_cast<size_t>(floor(static_cast<double>(*tmp_ptr_previous_layer_connected->ptr_number_outputs + 2_zu * ptr_layer_it_received->pooling_values[2u] - ptr_layer_it_received->pooling_values[3u] * (ptr_layer_it_received->pooling_values[0u] - 1_zu) - 1_zu) / static_cast<double>(ptr_layer_it_received->pooling_values[1u]) + 1.0)); }
            else
            { *ptr_layer_it_received->ptr_number_outputs = static_cast<size_t>(ceil(static_cast<double>(*tmp_ptr_previous_layer_connected->ptr_number_outputs + 2_zu * ptr_layer_it_received->pooling_values[2u] - ptr_layer_it_received->pooling_values[3u] * (ptr_layer_it_received->pooling_values[0u] - 1_zu) - 1_zu) / static_cast<double>(ptr_layer_it_received->pooling_values[1u]) + 1.0)); }

            ptr_layer_it_received->ptr_last_basic_indice_unit = ptr_layer_it_received->ptr_array_basic_indice_units + *ptr_layer_it_received->ptr_number_outputs;

            this->total_basic_indice_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_basic_indice_unit - ptr_layer_it_received->ptr_array_basic_indice_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
            tmp_ptr_residual_block_last_layer = ptr_layer_it_received + ptr_layer_it_received->block_depth;

            *ptr_layer_it_received->ptr_number_outputs = MyEA::Math::Maximum<size_t>(*tmp_ptr_previous_layer_connected->ptr_number_outputs, *tmp_ptr_residual_block_last_layer->ptr_number_outputs);
            
            ptr_layer_it_received->pooling_values[2u] = (static_cast<size_t>(MyEA::Math::Absolute<long long int>(static_cast<long long int>(*tmp_ptr_previous_layer_connected->ptr_number_outputs) - static_cast<long long int>(*tmp_ptr_residual_block_last_layer->ptr_number_outputs))) + 1_zu) / 2_zu; // Padding.

            ptr_layer_it_received->ptr_last_basic_unit = ptr_layer_it_received->ptr_array_basic_units + *ptr_layer_it_received->ptr_number_outputs;

            this->total_basic_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_basic_unit - ptr_layer_it_received->ptr_array_basic_units);
            
            if(this->Assign__Post__Residual_Block(ptr_layer_it_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Post__Residual_Block()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default: break;
    }

    return(true);
}

bool Neural_Network::Assign__Post__Residual_Block(struct Layer *&ptr_layer_it_received)
{
    if(ptr_layer_it_received->type_layer != MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is not a residual unit. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    struct Layer const *const tmp_ptr_residual_block_end(ptr_layer_it_received + ptr_layer_it_received->block_depth + 1);
    
    // First block layer.
    if(this->Assign__Post__Residual_Layer(true, ++ptr_layer_it_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Post__Residual_Layer(true)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| First block layer. |END|

    // Remaining layer(s).
    for(++ptr_layer_it_received; ptr_layer_it_received != tmp_ptr_residual_block_end; ++ptr_layer_it_received)
    {
        if(this->Assign__Post__Residual_Layer(false, ptr_layer_it_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Post__Residual_Layer(false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    // |END| Remaining layer(s). |END|
    
    // Assign layer iterator to the last layer inside the block.
    --ptr_layer_it_received;

    return(true);
}

bool Neural_Network::Assign__Post__Residual_Layer(bool const is_block_input_layer_received, struct Layer *&ptr_layer_it_received)
{
    struct Layer const *const tmp_ptr_previous_layer_connected(ptr_layer_it_received->previous_connected_layers[0u]),
                               *tmp_ptr_residual_block_last_layer;

    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            // Output_Size = floor((Input_Size + 2 * Padding - Kernel) / Stride + 1)
            if(ptr_layer_it_received->pooling_values[4u] == 0_zu)
            { *ptr_layer_it_received->ptr_number_outputs = static_cast<size_t>(floor(static_cast<double>(*tmp_ptr_previous_layer_connected->ptr_number_outputs + 2_zu * ptr_layer_it_received->pooling_values[2u] - ptr_layer_it_received->pooling_values[0u]) / static_cast<double>(ptr_layer_it_received->pooling_values[1u]) + 1.0)); }
            else
            { *ptr_layer_it_received->ptr_number_outputs = static_cast<size_t>(ceil(static_cast<double>(*tmp_ptr_previous_layer_connected->ptr_number_outputs + 2_zu * ptr_layer_it_received->pooling_values[2u] - ptr_layer_it_received->pooling_values[0u]) / static_cast<double>(ptr_layer_it_received->pooling_values[1u]) + 1.0)); }

            ptr_layer_it_received->ptr_last_basic_unit = ptr_layer_it_received->ptr_array_basic_units + *ptr_layer_it_received->ptr_number_outputs;

            this->total_basic_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_basic_unit - ptr_layer_it_received->ptr_array_basic_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            // Output_Size = floor((Input_Size + 2 * Padding - Dilation * (Kernel - 1) - 1) / Stride + 1)
            if(ptr_layer_it_received->pooling_values[4u] == 0_zu)
            { *ptr_layer_it_received->ptr_number_outputs = static_cast<size_t>(floor(static_cast<double>(*tmp_ptr_previous_layer_connected->ptr_number_outputs + 2_zu * ptr_layer_it_received->pooling_values[2u] - ptr_layer_it_received->pooling_values[3u] * (ptr_layer_it_received->pooling_values[0u] - 1_zu) - 1_zu) / static_cast<double>(ptr_layer_it_received->pooling_values[1u]) + 1.0)); }
            else
            { *ptr_layer_it_received->ptr_number_outputs = static_cast<size_t>(ceil(static_cast<double>(*tmp_ptr_previous_layer_connected->ptr_number_outputs + 2_zu * ptr_layer_it_received->pooling_values[2u] - ptr_layer_it_received->pooling_values[3u] * (ptr_layer_it_received->pooling_values[0u] - 1_zu) - 1_zu) / static_cast<double>(ptr_layer_it_received->pooling_values[1u]) + 1.0)); }

            ptr_layer_it_received->ptr_last_basic_indice_unit = ptr_layer_it_received->ptr_array_basic_indice_units + *ptr_layer_it_received->ptr_number_outputs;

            this->total_basic_indice_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_basic_indice_unit - ptr_layer_it_received->ptr_array_basic_indice_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            if(is_block_input_layer_received == false)
            {
                ptr_layer_it_received->ptr_last_AF_unit = ptr_layer_it_received->ptr_array_AF_units + *tmp_ptr_previous_layer_connected->ptr_number_outputs;

                this->total_AF_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_unit - ptr_layer_it_received->ptr_array_AF_units);
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            if(is_block_input_layer_received == false)
            {
                ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit = ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units + *tmp_ptr_previous_layer_connected->ptr_number_outputs;

                this->total_AF_Ind_recurrent_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit - ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
            }
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
            tmp_ptr_residual_block_last_layer = ptr_layer_it_received + ptr_layer_it_received->block_depth;

            *ptr_layer_it_received->ptr_number_outputs = MyEA::Math::Maximum<size_t>(*tmp_ptr_previous_layer_connected->ptr_number_outputs, *tmp_ptr_residual_block_last_layer->ptr_number_outputs);
            
            ptr_layer_it_received->pooling_values[2u] = (static_cast<size_t>(MyEA::Math::Absolute<long long int>(static_cast<long long int>(*tmp_ptr_previous_layer_connected->ptr_number_outputs) - static_cast<long long int>(*tmp_ptr_residual_block_last_layer->ptr_number_outputs))) + 1_zu) / 2_zu; // Padding.

            ptr_layer_it_received->ptr_last_basic_unit = ptr_layer_it_received->ptr_array_basic_units + *ptr_layer_it_received->ptr_number_outputs;

            this->total_basic_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_basic_unit - ptr_layer_it_received->ptr_array_basic_units);
            
            if(this->Assign__Post__Residual_Block(ptr_layer_it_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Post__Residual_Block()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str());
                return(false);
    }

    return(true);
}

bool Neural_Network::Compile(size_t const number_layers_received,
                                             size_t const number_recurrent_depth_received,
                                             MyEA::Common::ENUM_TYPE_NETWORKS const type_network_received,
                                             struct Layer_Parameters const *const ptr_array_layers_received,
                                             size_t const maximum_allowable_memory_received)
{
    if(number_recurrent_depth_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else { this->number_recurrent_depth = number_recurrent_depth_received; }
    
    if(this->Allocate__Structure(number_layers_received, maximum_allowable_memory_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Structure(%zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 number_layers_received,
                                 maximum_allowable_memory_received,
                                 __LINE__);

        return(false);
    }
    else { this->type_network = type_network_received; }

    size_t tmp_fan_in;
    
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                               *tmp_ptr_previous_layer_connected;
    struct Layer *tmp_ptr_layer_it;
    
    if(this->Assign__Layers(ptr_array_layers_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Layers()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    // Layers, connections.
    this->Order__Layers__Connection();
    
    if(this->Assign__Post__Layers() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Assign__Post__Layers()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    this->number_inputs = ptr_array_layers_received[0u].unit_parameters[0u];
    this->number_outputs = ptr_array_layers_received[number_layers_received - 1_zu].unit_parameters[0u];
    
    if(this->Allocate__Basic_Units() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Basic_Units()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->Allocate__Basic_Indice_Units() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Basic_Indice_Units()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->Allocate__Neuron_Units() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Neuron_Units()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->Allocate__AF_Units() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__AF_Units()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->Allocate__AF_Ind_Recurrent_Units() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__AF_Ind_Recurrent_Units()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->Allocate__LSTM_Layers() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__LSTM_Layers()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->Allocate__Bidirectional__Layers() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Bidirectional__Layers()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    // Layers, outputs pointers.
    this->Order__Layers__Output();

    // Initialize weight dimension.
    for(tmp_ptr_layer_it = this->ptr_array_layers + 1; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        // If the current layer is a pooling/residual layer, continue.
        if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
          ||
          tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING
          ||
          tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { continue; }
        
        tmp_fan_in = *tmp_ptr_layer_it->previous_connected_layers[0u]->ptr_number_outputs;
        
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                this->total_weights += this->Prepare__Connections__FC(tmp_fan_in, tmp_ptr_layer_it);
                
                if(this->Set__Layer_Activation_Function(tmp_ptr_layer_it, MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function(%u)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID,
                                             __LINE__);
                    
                    return(false);
                }
                
                if(this->Set__Layer_Activation_Steepness(tmp_ptr_layer_it, 1_T) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function(1)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->total_weights += this->Prepare__Connections__FC_Ind_RNN(tmp_fan_in, tmp_ptr_layer_it);
                
                if(this->Set__Layer_Activation_Function(tmp_ptr_layer_it, MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_RELU) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function(%u)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_RELU,
                                             __LINE__);
                    
                    return(false);
                }
                
                if(this->Set__Layer_Activation_Steepness(tmp_ptr_layer_it, 1_T) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function(1)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    return(false);
                }
                
                // Regularization on recurrent connection(s) (Independently RNN).
                if(this->Set__Regularization__Constraint_Recurrent_Weight__Default(tmp_ptr_layer_it) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight__Default(ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->total_weights += this->Prepare__Connections__LSTM(tmp_fan_in, tmp_ptr_layer_it); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    return(false);
        }
    }
    
    // Initialize bias dimension.
    for(tmp_ptr_layer_it = this->ptr_array_layers + 1; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
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
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->total_bias += this->Prepare__Bias__FC(this->total_weights, tmp_ptr_layer_it); break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->total_bias += this->Prepare__Bias__LSTM(this->total_weights, tmp_ptr_layer_it); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                    return(false);
        }
    }

    this->total_parameters = this->total_weights + this->total_bias;

    if(this->Allocate__Parameter() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    // Initialize connection(s).
    for(tmp_ptr_layer_it = this->ptr_array_layers + 1; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        // If the current layer is a pooling/residual layer, continue.
        if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
          ||
          tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING
          ||
          tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { continue; }
        
        tmp_ptr_previous_layer_connected = tmp_ptr_layer_it->previous_connected_layers[0u];

        switch(tmp_ptr_previous_layer_connected->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                switch(tmp_ptr_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: this->Initialize_Connections__Basic_unit_to_FC(tmp_ptr_layer_it, tmp_ptr_previous_layer_connected); break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Initialize_Connections__Basic_unit_to_LSTM(tmp_ptr_layer_it, tmp_ptr_previous_layer_connected); break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                                 __LINE__);
                            return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                switch(tmp_ptr_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: this->Initialize_Connections__FC_to_FC(tmp_ptr_layer_it, tmp_ptr_previous_layer_connected); break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Initialize_Connections__FC_to_LSTM(tmp_ptr_layer_it, tmp_ptr_previous_layer_connected); break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                                 __LINE__);
                            return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                switch(tmp_ptr_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: this->Initialize_Connections__LSTM_to_FC(tmp_ptr_layer_it, tmp_ptr_previous_layer_connected); break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Initialize_Connections__LSTM_to_LSTM(tmp_ptr_layer_it, tmp_ptr_previous_layer_connected); break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                                 __LINE__);
                            return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                switch(tmp_ptr_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: this->Initialize_Connections__Basic_indice_unit_to_FC(tmp_ptr_layer_it, tmp_ptr_previous_layer_connected); break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Initialize_Connections__Basic_indice_unit_to_LSTM(tmp_ptr_layer_it, tmp_ptr_previous_layer_connected); break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                                 __LINE__);
                            return(false);
                }
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_previous_layer_connected->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_layer_connected->type_layer].c_str(),
                                         __LINE__);
                    return(false);
        }
        
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Initialize_Connections__AF_Ind_Recurrent(tmp_ptr_layer_it);
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->Initialize_Connections__Bias(tmp_ptr_layer_it); break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Initialize_Connections__LSTM__Bias(tmp_ptr_layer_it); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                            MyEA::Time::Date_Time_Now().c_str(),
                                            __FUNCTION__,
                                            tmp_ptr_layer_it->type_layer,
                                            MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                            __LINE__);
                    return(false);
        }
    }

    return(true);
}