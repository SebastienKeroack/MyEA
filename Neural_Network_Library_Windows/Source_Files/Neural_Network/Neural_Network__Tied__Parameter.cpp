#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

bool Layer::Use__Tied_Parameter(void) const { return(this->use_tied_parameter); }

bool Neural_Network::Set__Tied_Parameter(size_t const index_layer_received,
                                                                bool const use_tied_parameter_received,
                                                                bool const transpose_received)
{
    if(index_layer_received >= this->total_layers)
    {
        PRINT_FORMAT("%s: ERROR: Layer received (%zu) overflow the number of layers (%zu) in the neural network." NEW_LINE,
                                 __FUNCTION__,
                                 index_layer_received,
                                 this->total_layers);

        return(false);
    }
    else if(this->ptr_array_layers == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_array_layers\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(this->Set__Tied_Parameter(this->ptr_array_layers + index_layer_received,
                                                      use_tied_parameter_received,
                                                      transpose_received));
}

bool Neural_Network::Set__Tied_Parameter(struct Layer *const ptr_layer_received,
                                                                bool const use_tied_parameter_received,
                                                                bool const transpose_received)
{
    auto tmp_Valid_Layer([](struct Layer const *const ptr_layer_received) -> bool
    {
        if(ptr_layer_received->type_group == MyEA::Common::ENUM_TYPE_GROUP::TYPE_GROUP_RESIDUAL)
        {
            PRINT_FORMAT("%s: %s: ERROR: Group type (%u | %s) is not managed in the function." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        ptr_layer_received->type_group,
                                        MyEA::Common::ENUM_TYPE_GROUP_NAME[ptr_layer_received->type_group].c_str());
            
            return(false);
        }

        switch(ptr_layer_received->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the function." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         ptr_layer_received->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_received->type_layer].c_str());
                    return(false);
        }

        return(true);
    });

    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
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
    else if(tmp_Valid_Layer(ptr_layer_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Valid_Layer(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received->use_tied_parameter == use_tied_parameter_received) { return(true); }

    // Mirror.
    if(ptr_layer_received < this->Get__End_Layer__Active() - 1) // Get last active layer.
    {
        struct Layer *const tmp_ptr_mirror_layer(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1);
        struct Layer const *const tmp_ptr_previous_layer_it(ptr_layer_received->previous_connected_layers[0u]),
                                   *tmp_ptr_next_layer_it(tmp_ptr_mirror_layer->next_connected_layers[0u]),
                                   *const tmp_ptr_next_layer_end(tmp_ptr_next_layer_it + tmp_ptr_mirror_layer->next_connected_layers.size());
        
        if(tmp_Valid_Layer(tmp_ptr_previous_layer_it) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Valid_Layer(ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        else if(ptr_layer_received->type_layer != tmp_ptr_mirror_layer->type_layer)
        {
            PRINT_FORMAT("%s: %s: ERROR: The layer type (%u | %s) differ from the mirror layer type (%u | %s). At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_received->type_layer].c_str(),
                                     tmp_ptr_mirror_layer->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_mirror_layer->type_layer].c_str(),
                                     __LINE__);

            return(false);
        }
        else if(*ptr_layer_received->ptr_number_outputs != *tmp_ptr_mirror_layer->ptr_number_outputs)
        {
            PRINT_FORMAT("%s: %s: ERROR: The layer size (%zu) differ from the mirror layer size (%zu). At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     *ptr_layer_received->ptr_number_outputs,
                                     *tmp_ptr_mirror_layer->ptr_number_outputs,
                                     __LINE__);

            return(false);
        }

        for(; tmp_ptr_next_layer_it != tmp_ptr_next_layer_end; ++tmp_ptr_next_layer_it)
        {
            if(tmp_Valid_Layer(tmp_ptr_next_layer_it) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Valid_Layer(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            else if(tmp_ptr_previous_layer_it->type_layer != tmp_ptr_next_layer_it->type_layer)
            {
                PRINT_FORMAT("%s: %s: ERROR: The previous connected layer type (%u | %s) differ from the next connected layer type (%u | %s). At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_previous_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_layer_it->type_layer].c_str(),
                                         tmp_ptr_next_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_next_layer_it->type_layer].c_str(),
                                         __LINE__);

                return(false);
            }
            else if(*tmp_ptr_previous_layer_it->ptr_number_outputs != *tmp_ptr_next_layer_it->ptr_number_outputs)
            {
                PRINT_FORMAT("%s: %s: ERROR: The previous connected layer size (%zu) differ from the next connected layer size (%zu). At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         *tmp_ptr_previous_layer_it->ptr_number_outputs,
                                         *tmp_ptr_next_layer_it->ptr_number_outputs,
                                         __LINE__);

                return(false);
            }
        }

        if(this->Set__Tied_Parameter(tmp_ptr_mirror_layer,
                                                   use_tied_parameter_received,
                                                   false))
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Tied_Parameter(ptr, %s, false)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     use_tied_parameter_received ? "true" : "false",
                                     __LINE__);

            return(false);
        }
    }
    // |END| Mirror. |END|
    
    if(ptr_layer_received->use_tied_parameter == false && use_tied_parameter_received)
    {
        ++this->total_tied_parameter_layers;

        if(transpose_received) { this->Tied__Transpose(ptr_layer_received); }
    }
    else if(ptr_layer_received->use_tied_parameter && use_tied_parameter_received == false) { --this->total_tied_parameter_layers; }

    ptr_layer_received->use_tied_parameter = use_tied_parameter_received;
    
    return(true);
}

void Neural_Network::Tied__Transpose(void)
{
    struct Layer const *const tmp_ptr_end_layer(this->ptr_array_layers + (this->total_layers - 3_zu) / 2_zu + 1_zu);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
    for(; tmp_ptr_layer_it != tmp_ptr_end_layer; ++tmp_ptr_layer_it)
    {
        if(tmp_ptr_layer_it->Use__Tied_Parameter())
        {
            this->Tied__Transpose(tmp_ptr_layer_it);
        }
    }
}

void Neural_Network::Tied__Transpose(struct Layer *const ptr_layer_received)
{
    this->Tied__Transpose__Weight(ptr_layer_received);

    if(ptr_layer_received->Use__Normalization()) { this->Tied__Transpose__Normalization(ptr_layer_received); }
}