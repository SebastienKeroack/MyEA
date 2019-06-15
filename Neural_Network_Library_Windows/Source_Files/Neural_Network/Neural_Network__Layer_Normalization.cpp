#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

bool Neural_Network::Prepare__Normalized__Layers(void)
{
    this->total_normalized_units = 0_zu;

    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
    
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if(this->Prepare__Normalized__Layer(tmp_ptr_layer_it) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Prepare__Normalized__Layer()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

bool Neural_Network::Prepare__Normalized__Layer(struct Layer *&ptr_layer_it_received)
{
    struct Layer const *tmp_ptr_residual_block_last_layer;

    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING: break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            ptr_layer_it_received->ptr_array_normalized_units = nullptr;
            ptr_layer_it_received->ptr_last_normalized_unit = ptr_layer_it_received->ptr_array_normalized_units + *ptr_layer_it_received->ptr_number_outputs;

            this->total_normalized_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_normalized_unit - ptr_layer_it_received->ptr_array_normalized_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            ptr_layer_it_received->ptr_array_normalized_units = nullptr;
            ptr_layer_it_received->ptr_last_normalized_unit = ptr_layer_it_received->ptr_array_normalized_units + 3_zu * static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units)
                                                                                                                                                                  +
                                                                                                                                                               6_zu * static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units);

            this->total_normalized_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_normalized_unit - ptr_layer_it_received->ptr_array_normalized_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
            tmp_ptr_residual_block_last_layer = ptr_layer_it_received + ptr_layer_it_received->block_depth;

            ptr_layer_it_received->ptr_array_normalized_units = nullptr;
            ptr_layer_it_received->ptr_last_normalized_unit = ptr_layer_it_received->ptr_array_normalized_units + *tmp_ptr_residual_block_last_layer->ptr_number_outputs;

            this->total_normalized_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_normalized_unit - ptr_layer_it_received->ptr_array_normalized_units);

            if(this->Prepare__Normalized__Residual_Block(ptr_layer_it_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Prepare__Normalized__Residual_Block()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str());
                return(false);
    }

    return(true);
}

bool Neural_Network::Prepare__Normalized__Residual_Block(struct Layer *&ptr_layer_it_received)
{
    if(ptr_layer_it_received->type_layer != MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is not a residual unit. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    struct Layer const *const tmp_ptr_residual_block_end(ptr_layer_it_received + ptr_layer_it_received->block_depth + 1);
    
    for(++ptr_layer_it_received; ptr_layer_it_received != tmp_ptr_residual_block_end; ++ptr_layer_it_received)
    {
        if(this->Prepare__Normalized__Residual_Layer(ptr_layer_it_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Prepare__Normalized__Residual_Layer()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    
    // Assign layer iterator to the last layer inside the block.
    --ptr_layer_it_received;

    return(true);
}

bool Neural_Network::Prepare__Normalized__Residual_Layer(struct Layer *&ptr_layer_it_received)
{
    struct Layer const *const tmp_ptr_previous_layer_connected(ptr_layer_it_received->previous_connected_layers[0u]),
                               *tmp_ptr_residual_block_last_layer;

    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING: break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
            ptr_layer_it_received->ptr_array_normalized_units = nullptr;
            ptr_layer_it_received->ptr_last_normalized_unit = ptr_layer_it_received->ptr_array_normalized_units + *tmp_ptr_previous_layer_connected->ptr_number_outputs;

            this->total_normalized_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_normalized_unit - ptr_layer_it_received->ptr_array_normalized_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            ptr_layer_it_received->ptr_array_normalized_units = nullptr;
            ptr_layer_it_received->ptr_last_normalized_unit = ptr_layer_it_received->ptr_array_normalized_units + 3_zu * static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units)
                                                                                                                                                                  +
                                                                                                                                                               6_zu * static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units);

            this->total_normalized_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_normalized_unit - ptr_layer_it_received->ptr_array_normalized_units);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
            tmp_ptr_residual_block_last_layer = ptr_layer_it_received + ptr_layer_it_received->block_depth;

            ptr_layer_it_received->ptr_array_normalized_units = nullptr;
            ptr_layer_it_received->ptr_last_normalized_unit = ptr_layer_it_received->ptr_array_normalized_units + *tmp_ptr_residual_block_last_layer->ptr_number_outputs;

            this->total_normalized_units += static_cast<size_t>(ptr_layer_it_received->ptr_last_normalized_unit - ptr_layer_it_received->ptr_array_normalized_units);

            if(this->Prepare__Normalized__Residual_Block(ptr_layer_it_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Prepare__Normalized__Residual_Block()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str());
                return(false);
    }

    return(true);
}

bool Neural_Network::Set__Layer_Normalization(size_t const index_layer_received,
                                                                       enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_layer_normalization_received,
                                                                       bool const reallocate_dimension_parameters_received,
                                                                       bool const organize_pointers_received)
{
    if(index_layer_received >= this->total_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received (%zu) as argument overflow the number of layers (%zu) in the neural network. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 index_layer_received,
                                 this->total_layers,
                                 __LINE__);

        return(false);
    }
    else if(this->ptr_array_layers == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(this->Set__Layer_Normalization(this->ptr_array_layers + index_layer_received,
                                                            type_layer_normalization_received,
                                                            reallocate_dimension_parameters_received,
                                                            organize_pointers_received));
}

bool Neural_Network::Set__Layer_Normalization(struct Layer *const ptr_layer_received,
                                                                       enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_layer_normalization_received,
                                                                       bool const reallocate_dimension_parameters_received,
                                                                       bool const organize_pointers_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the input layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the output layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
             &&
             ptr_layer_received >= this->ptr_last_layer - (this->total_layers - 3_zu) / 2_zu + 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is a decoded layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    switch(type_layer_normalization_received)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE: return(this->Set__Normalization_None(ptr_layer_received, organize_pointers_received));
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
            return(this->Set__Batch_Normalization(ptr_layer_received,
                                                                     true,
                                                                     reallocate_dimension_parameters_received,
                                                                     organize_pointers_received));
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
            return(this->Set__Batch_Renormalization(ptr_layer_received,
                                                                        true,
                                                                        reallocate_dimension_parameters_received,
                                                                        organize_pointers_received));
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
            return(this->Set__Ghost_Batch_Normalization(ptr_layer_received,
                                                                               true,
                                                                               reallocate_dimension_parameters_received,
                                                                               organize_pointers_received));
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type normalization layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_received->type_normalization,
                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_received->type_normalization].c_str(),
                                     __LINE__);
                return(false);
    }
}

bool Neural_Network::Set__Normalization_None(struct Layer *const ptr_layer_received, bool const organize_pointers_received)
{
    switch(ptr_layer_received->type_normalization)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE:
            if(organize_pointers_received) { this->Order__Layer__Output(false, ptr_layer_received); }
                return(true);
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
            return(this->Set__Batch_Normalization(ptr_layer_received,
                                                                   false,
                                                                   false,
                                                                   false));
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
            return(this->Set__Batch_Renormalization(ptr_layer_received,
                                                                      false,
                                                                      false,
                                                                      false));
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
            return(this->Set__Ghost_Batch_Normalization(ptr_layer_received,
                                                                                false,
                                                                                false,
                                                                                false));
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type normalization layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_received->type_normalization,
                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_layer_received->type_normalization].c_str(),
                                     __LINE__);
                return(false);
    }
}

bool Neural_Network::Set__Batch_Normalization(struct Layer *const ptr_layer_received,
                                                                       bool const use_batch_normalization_received,
                                                                       bool const reallocate_dimension_parameters_received,
                                                                       bool const organize_pointers_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the input layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the output layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(ptr_layer_received->type_normalization != MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION)
    {
        if(this->Set__Normalization_None(ptr_layer_received, organize_pointers_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_None(%s)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     organize_pointers_received ? "true" : "false",
                                     __LINE__);

            return(false);
        }
    }

    if(use_batch_normalization_received && ptr_layer_received->type_normalization == MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE)
    {
        ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION;

        bool const tmp_normalization_initialized(this->Use__Normalization());

        if(++this->total_batch_normalization_layers == 1_zu)
        {
            if(this->Allocate__Normalized_Unit(organize_pointers_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Normalized_Unit(%s)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         organize_pointers_received ? "true" : "false",
                                         __LINE__);
                
                ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

                --this->total_batch_normalization_layers;

                return(false);
            }
            else if(tmp_normalization_initialized == false
                     &&
                     reallocate_dimension_parameters_received
                     &&
                     this->Allocate__Parameter__Normalization() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter__Normalization()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
                
                ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

                --this->total_batch_normalization_layers;

                return(false);
            }
            else if(this->Allocate__Normalized_Unit__Batch_Normalization() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Normalized_Unit__Batch_Normalization()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
                
                ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

                --this->total_batch_normalization_layers;

                return(false);
            }
        }
    }
    else if(use_batch_normalization_received == false && ptr_layer_received->type_normalization == MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION)
    {
        ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;
        
        if(this->total_batch_normalization_layers != 0_zu
           &&
           --this->total_batch_normalization_layers == 0u
           &&
           this->Use__Normalization() == false)
        {
            this->Deallocate__Parameter__Batch_Normalization();
            
            this->Deallocate__Normalized_Unit();

            this->Deallocate__Normalized_Unit__Batch_Normalization();
        }
    }
    
    if(organize_pointers_received) { this->Order__Layer__Output(false, ptr_layer_received); }
    
    // Mirror layer.
    if(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
      &&
      ptr_layer_received < this->Get__End_Layer__Active() - 1 // Get last active layer.
      &&
      this->Set__Batch_Normalization(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1,
                                                      use_batch_normalization_received,
                                                      reallocate_dimension_parameters_received,
                                                      organize_pointers_received))
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Batch_Normalization(ptr, %s, %s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 use_batch_normalization_received ? "true" : "false",
                                 reallocate_dimension_parameters_received ? "true" : "false",
                                 organize_pointers_received ? "true" : "false",
                                 __LINE__);

        return(false);
    }
    // |END| Mirror layer. |END|

    return(true);
}

bool Neural_Network::Set__Batch_Renormalization(struct Layer *const ptr_layer_received,
                                                                           bool const use_batch_renormalization_received,
                                                                           bool const reallocate_dimension_parameters_received,
                                                                           bool const organize_pointers_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the input layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the output layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(ptr_layer_received->type_normalization != MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION)
    {
        if(this->Set__Normalization_None(ptr_layer_received, organize_pointers_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_None(%s)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     organize_pointers_received ? "true" : "false",
                                     __LINE__);

            return(false);
        }
    }

    if(use_batch_renormalization_received && ptr_layer_received->type_normalization == MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE)
    {
        ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION;
        
        bool const tmp_normalization_initialized(this->Use__Normalization());

        if(++this->total_batch_renormalization_layers == 1_zu)
        {
            if(this->Allocate__Normalized_Unit(organize_pointers_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Normalized_Unit(%s)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         organize_pointers_received ? "true" : "false",
                                         __LINE__);
                
                ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

                --this->total_batch_renormalization_layers;

                return(false);
            }
            else if(tmp_normalization_initialized == false
                     &&
                     reallocate_dimension_parameters_received
                     &&
                     this->Allocate__Parameter__Normalization() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter__Normalization()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
                
                ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

                --this->total_batch_renormalization_layers;

                return(false);
            }
            else if(this->Allocate__Normalized_Unit__Batch_Normalization() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Normalized_Unit__Batch_Normalization()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
                
                ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

                --this->total_batch_renormalization_layers;

                return(false);
            }
            else if(this->Allocate__Normalized_Unit__Batch_Renormalization() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Normalized_Unit__Batch_Renormalization()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
                
                ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

                --this->total_batch_renormalization_layers;

                return(false);
            }
        }
    }
    else if(use_batch_renormalization_received == false && ptr_layer_received->type_normalization == MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION)
    {
        ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;
        
        if(this->total_batch_renormalization_layers != 0_zu && --this->total_batch_renormalization_layers == 0_zu)
        {
            this->Deallocate__Normalized_Unit__Batch_Renormalization();
            
            if(this->Use__Normalization() == false)
            {
                this->Deallocate__Parameter__Batch_Normalization();

                this->Deallocate__Normalized_Unit();

                this->Deallocate__Normalized_Unit__Batch_Normalization();
            }
        }
    }
    
    if(organize_pointers_received) { this->Order__Layer__Output(false, ptr_layer_received); }
    
    // Mirror layer.
    if(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
      &&
      ptr_layer_received < this->Get__End_Layer__Active() - 1 // Get last active layer.
      &&
      this->Set__Batch_Renormalization(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1,
                                                         use_batch_renormalization_received,
                                                         reallocate_dimension_parameters_received,
                                                         organize_pointers_received))
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Batch_Renormalization(ptr, %s, %s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 use_batch_renormalization_received ? "true" : "false",
                                 reallocate_dimension_parameters_received ? "true" : "false",
                                 organize_pointers_received ? "true" : "false",
                                 __LINE__);

        return(false);
    }
    // |END| Mirror layer. |END|

    return(true);
}

bool Neural_Network::Set__Ghost_Batch_Normalization(struct Layer *const ptr_layer_received,
                                                                                  bool const use_ghost_batch_normalization_received,
                                                                                  bool const reallocate_dimension_parameters_received,
                                                                                  bool const organize_pointers_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the input layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the output layer. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(ptr_layer_received->type_normalization != MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION)
    {
        if(this->Set__Normalization_None(ptr_layer_received, organize_pointers_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_None(%s)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     organize_pointers_received ? "true" : "false",
                                     __LINE__);

            return(false);
        }
    }

    if(use_ghost_batch_normalization_received && ptr_layer_received->type_normalization == MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE)
    {
        ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION;
        
        bool const tmp_normalization_initialized(this->Use__Normalization());

        if(++this->total_ghost_batch_normalization_layers == 1_zu)
        {
            if(this->Allocate__Normalized_Unit(organize_pointers_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Normalized_Unit(%s)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         organize_pointers_received ? "true" : "false",
                                         __LINE__);
                
                ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

                --this->total_ghost_batch_normalization_layers;

                return(false);
            }
            else if(tmp_normalization_initialized == false
                     &&
                     reallocate_dimension_parameters_received
                     &&
                     this->Allocate__Parameter__Normalization() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter__Normalization()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
                
                ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

                --this->total_ghost_batch_normalization_layers;

                return(false);
            }
            else if(this->Allocate__Normalized_Unit__Batch_Normalization() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Normalized_Unit__Batch_Normalization()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
                
                ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;

                --this->total_ghost_batch_normalization_layers;

                return(false);
            }
        }
    }
    else if(use_ghost_batch_normalization_received == false && ptr_layer_received->type_normalization == MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION)
    {
        ptr_layer_received->type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;
        
        if(this->total_ghost_batch_normalization_layers != 0_zu
           &&
           --this->total_ghost_batch_normalization_layers == 0u
           &&
           this->Use__Normalization() == false)
        {
            this->Deallocate__Parameter__Batch_Normalization();
            
            this->Deallocate__Normalized_Unit();

            this->Deallocate__Normalized_Unit__Batch_Normalization();
        }
    }
    
    if(organize_pointers_received) { this->Order__Layer__Output(false, ptr_layer_received); }
    
    // Mirror layer.
    if(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
      &&
      ptr_layer_received < this->Get__End_Layer__Active() - 1 // Get last active layer.
      &&
      this->Set__Ghost_Batch_Normalization(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1,
                                                                use_ghost_batch_normalization_received,
                                                                reallocate_dimension_parameters_received,
                                                                organize_pointers_received))
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Ghost_Batch_Normalization(ptr, %s, %s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 use_ghost_batch_normalization_received ? "true" : "false",
                                 reallocate_dimension_parameters_received ? "true" : "false",
                                 organize_pointers_received ? "true" : "false",
                                 __LINE__);

        return(false);
    }
    // |END| Mirror layer. |END|

    return(true);
}