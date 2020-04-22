#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

Neural_Network& Neural_Network::operator = (class Neural_Network const &ref_source_Neural_Network_received)
{
    if(&ref_source_Neural_Network_received != this) { this->Copy(ref_source_Neural_Network_received); }

    return(*this);
}

bool Neural_Network::Copy(class Neural_Network const &ref_source_Neural_Network_received,
                                        bool const initialize_parallel_computation_received,
                                        bool const copy_delta_optimizer_received,
                                        size_t const maximum_allowable_memory_received)
{
    /*
#if defined(COMPILE_CUDA)
    if(ref_source_Neural_Network_received.is_device_initialized
       &&
       ref_source_Neural_Network_received.is_update_from_device == false)
    {
        if(ref_source_Neural_Network_received.Copy__Parameters__Device_To_Host() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Parameters__Device_To_Host()\" function. At line %d." NEW_LINE,
                                        MyEA::Time::Date_Time_Now().c_str(),
                                        __FUNCTION__,
                                        __LINE__);

            return(false);
        }
        else if(ref_source_Neural_Network_received.Use__Normalization() && ref_source_Neural_Network_received.Copy__Batch_Normalization_Neurons__Device_To_Host() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Batch_Normalization_Neurons__Device_To_Host()\" function. At line %d." NEW_LINE,
                                        MyEA::Time::Date_Time_Now().c_str(),
                                        __FUNCTION__,
                                        __LINE__);

            return(false);
        }
    }
#endif
    */

    this->Clear();

    if(this->Allocate__Structure(ref_source_Neural_Network_received.total_layers, maximum_allowable_memory_received != 0_zu ? maximum_allowable_memory_received : ref_source_Neural_Network_received.maximum_allowable_memory_bytes) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Structure(%zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_source_Neural_Network_received.total_layers,
                                 maximum_allowable_memory_received != 0_zu ? maximum_allowable_memory_received : ref_source_Neural_Network_received.maximum_allowable_memory_bytes,
                                 __LINE__);

        return(false);
    }
    
    // General parameters.
    this->type_network = ref_source_Neural_Network_received.type_network;
    this->number_recurrent_depth = ref_source_Neural_Network_received.number_recurrent_depth;
    // |END| General parameters. |END|

    // Loss parameters.
    *this->ptr_array_number_bit_fail = *ref_source_Neural_Network_received.ptr_array_number_bit_fail;
    *this->ptr_array_number_loss = *ref_source_Neural_Network_received.ptr_array_number_loss;
    *this->ptr_array_loss_values = *ref_source_Neural_Network_received.ptr_array_loss_values;

    this->Copy__Loss(&ref_source_Neural_Network_received);
    // |END| Loss parameters. |END|

    // Accuracy parameters.
    this->number_accuracy_trial = ref_source_Neural_Network_received.number_accuracy_trial;
    this->ptr_array_accuracy_values[0u][0u] = ref_source_Neural_Network_received.ptr_array_accuracy_values[0u][0u];
    this->ptr_array_accuracy_values[1u][0u] = ref_source_Neural_Network_received.ptr_array_accuracy_values[1u][0u];
    this->ptr_array_accuracy_values[2u][0u] = ref_source_Neural_Network_received.ptr_array_accuracy_values[2u][0u];
    this->ptr_array_accuracy_values[3u][0u] = ref_source_Neural_Network_received.ptr_array_accuracy_values[3u][0u];
    this->ptr_array_accuracy_values[4u][0u] = ref_source_Neural_Network_received.ptr_array_accuracy_values[4u][0u];

    this->Copy__Accuracy(&ref_source_Neural_Network_received);
    // |END| Accuracy parameters. |END|

    // Dimension.
    this->total_layers = ref_source_Neural_Network_received.total_layers;
    this->number_inputs = ref_source_Neural_Network_received.number_inputs;
    this->number_outputs = ref_source_Neural_Network_received.number_outputs;
    this->total_basic_units = ref_source_Neural_Network_received.total_basic_units;
    this->total_basic_indice_units = ref_source_Neural_Network_received.total_basic_indice_units;
    this->total_neuron_units = ref_source_Neural_Network_received.total_neuron_units;
    this->total_AF_units = ref_source_Neural_Network_received.total_AF_units;
    this->total_AF_Ind_recurrent_units = ref_source_Neural_Network_received.total_AF_Ind_recurrent_units;
    this->total_block_units = ref_source_Neural_Network_received.total_block_units;
    this->total_cell_units = ref_source_Neural_Network_received.total_cell_units;
    this->total_parameters = ref_source_Neural_Network_received.total_weights + ref_source_Neural_Network_received.total_bias;
    this->total_weights = ref_source_Neural_Network_received.total_weights;
    this->total_bias = ref_source_Neural_Network_received.total_bias;

    if(this->Set__Input_Mode(ref_source_Neural_Network_received.use_first_layer_as_input) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Input_Mode(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_source_Neural_Network_received.use_first_layer_as_input ? "true" : "false",
                                 __LINE__);

        return(false);
    }
    else if(this->Set__Output_Mode(ref_source_Neural_Network_received.use_last_layer_as_output) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Output_Mode(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_source_Neural_Network_received.use_last_layer_as_output ? "true" : "false",
                                 __LINE__);

        return(false);
    }
    
    struct Layer const *const tmp_ptr_destination_last_layer(this->ptr_last_layer),
                               *tmp_ptr_destination_previous_layer_it,
                               *tmp_ptr_source_layer_it;
    struct Layer *tmp_ptr_destination_layer_it;
    
    for(tmp_ptr_source_layer_it = ref_source_Neural_Network_received.ptr_array_layers,
        tmp_ptr_destination_layer_it = this->ptr_array_layers; tmp_ptr_destination_layer_it != tmp_ptr_destination_last_layer; ++tmp_ptr_destination_layer_it,
                                                                                                                                                                                  ++tmp_ptr_source_layer_it)
    {
        tmp_ptr_destination_layer_it->type_layer = tmp_ptr_source_layer_it->type_layer;

        // Pooling.
        tmp_ptr_destination_layer_it->pooling_values[0u] = tmp_ptr_source_layer_it->pooling_values[0u];
        tmp_ptr_destination_layer_it->pooling_values[1u] = tmp_ptr_source_layer_it->pooling_values[1u];
        tmp_ptr_destination_layer_it->pooling_values[2u] = tmp_ptr_source_layer_it->pooling_values[2u];
        tmp_ptr_destination_layer_it->pooling_values[3u] = tmp_ptr_source_layer_it->pooling_values[3u];
        tmp_ptr_destination_layer_it->pooling_values[4u] = tmp_ptr_source_layer_it->pooling_values[4u];
        // |END Pooling. |END|

        tmp_ptr_destination_layer_it->type_activation = tmp_ptr_source_layer_it->type_activation;

        tmp_ptr_destination_layer_it->block_depth = tmp_ptr_source_layer_it->block_depth;

        *tmp_ptr_destination_layer_it->ptr_number_outputs = *tmp_ptr_source_layer_it->ptr_number_outputs;

        *tmp_ptr_destination_layer_it->ptr_first_connection_index = *tmp_ptr_source_layer_it->ptr_first_connection_index;
        *tmp_ptr_destination_layer_it->ptr_last_connection_index = *tmp_ptr_source_layer_it->ptr_last_connection_index;

        tmp_ptr_destination_layer_it->first_bias_connection_index = tmp_ptr_source_layer_it->first_bias_connection_index;
        tmp_ptr_destination_layer_it->last_bias_connection_index = tmp_ptr_source_layer_it->last_bias_connection_index;
        
        // Basic unit(s).
        tmp_ptr_destination_layer_it->ptr_last_basic_unit = tmp_ptr_destination_layer_it->ptr_array_basic_units + static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_basic_unit - tmp_ptr_source_layer_it->ptr_array_basic_units);
        // |END| Basic unit(s). |END|
        
        // Basic indice unit(s).
        tmp_ptr_destination_layer_it->ptr_last_basic_indice_unit = tmp_ptr_destination_layer_it->ptr_array_basic_indice_units + static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_basic_indice_unit - tmp_ptr_source_layer_it->ptr_array_basic_indice_units);
        // |END| Basic indice unit(s). |END|

        // Neuron unit(s).
        tmp_ptr_destination_layer_it->ptr_last_neuron_unit = tmp_ptr_destination_layer_it->ptr_array_neuron_units + static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_neuron_unit - tmp_ptr_source_layer_it->ptr_array_neuron_units);
        // |END| Neuron unit(s). |END|
        
        // AF unit(s).
        tmp_ptr_destination_layer_it->ptr_last_AF_unit = tmp_ptr_destination_layer_it->ptr_array_AF_units + static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_AF_unit - tmp_ptr_source_layer_it->ptr_array_AF_units);
        // |END| AF unit(s). |END|
        
        // AF Ind recurrent unit(s).
        tmp_ptr_destination_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_destination_layer_it->ptr_array_AF_Ind_recurrent_units + static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_source_layer_it->ptr_array_AF_Ind_recurrent_units);
        // |END| AF Ind recurrent unit(s). |END|

        // Block unit(s).
        tmp_ptr_destination_layer_it->ptr_last_block_unit = tmp_ptr_destination_layer_it->ptr_array_block_units + static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_block_unit - tmp_ptr_source_layer_it->ptr_array_block_units);
        // |END| Block unit(s). |END|
        
        // Cell unit(s).
        tmp_ptr_destination_layer_it->ptr_last_cell_unit = tmp_ptr_destination_layer_it->ptr_array_cell_units + static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_cell_unit - tmp_ptr_source_layer_it->ptr_array_cell_units);
        // |END| Cell unit(s). |END|
    }
    // |END| Dimension. |END|
    
    // Layers, connections.
    this->Order__Layers__Connection();

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
    else if(this->Allocate__Parameter() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    this->Copy__Neuron_Units(0_zu,
                                            this->total_neuron_units,
                                            ref_source_Neural_Network_received.ptr_array_neuron_units);
    
    this->Copy__AF_Units(0_zu,
                                      this->total_AF_units,
                                      ref_source_Neural_Network_received.ptr_array_AF_units);

    this->Copy__AF_Ind_Recurrent_Units(0_zu,
                                                            this->total_AF_Ind_recurrent_units,
                                                            ref_source_Neural_Network_received.ptr_array_AF_Ind_recurrent_units);
    
    this->Order__Layers__Output();

    // Copy connections.
    for(tmp_ptr_source_layer_it = ref_source_Neural_Network_received.ptr_array_layers + 1,
        tmp_ptr_destination_layer_it = this->ptr_array_layers + 1; tmp_ptr_destination_layer_it != tmp_ptr_destination_last_layer; ++tmp_ptr_destination_layer_it,
                                                                                                                                                                                        ++tmp_ptr_source_layer_it)
    {
        // If the current layer is a pooling/residual layer, continue.
        if(tmp_ptr_destination_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
          ||
          tmp_ptr_destination_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING
          ||
          tmp_ptr_destination_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { continue; }
        
        tmp_ptr_destination_previous_layer_it = tmp_ptr_destination_layer_it->previous_connected_layers[0u];

        if(tmp_ptr_destination_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            this->Copy__Layer__AF_Ind_Recurrent(tmp_ptr_source_layer_it,
                                                                      ref_source_Neural_Network_received.ptr_array_AF_Ind_recurrent_units,
                                                                      this->ptr_array_AF_Ind_recurrent_units,
                                                                      reinterpret_cast<struct AF_Ind_recurrent_unit**>(ref_source_Neural_Network_received.ptr_array_ptr_connections),
                                                                      reinterpret_cast<struct AF_Ind_recurrent_unit**>(this->ptr_array_ptr_connections));
        }

        switch(tmp_ptr_destination_previous_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                switch(tmp_ptr_destination_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                        this->Copy__Layer__FC<struct Basic_unit>(tmp_ptr_source_layer_it,
                                                                                         tmp_ptr_destination_layer_it,
                                                                                         ref_source_Neural_Network_received.ptr_array_basic_units,
                                                                                         this->ptr_array_basic_units,
                                                                                         reinterpret_cast<struct Basic_unit**>(ref_source_Neural_Network_received.ptr_array_ptr_connections),
                                                                                         reinterpret_cast<struct Basic_unit**>(this->ptr_array_ptr_connections));
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                        this->Copy__Layer__LSTM<struct Basic_unit>(tmp_ptr_source_layer_it,
                                                                                            tmp_ptr_destination_layer_it,
                                                                                            ref_source_Neural_Network_received.ptr_array_cell_units,
                                                                                            ref_source_Neural_Network_received.ptr_array_basic_units,
                                                                                            this->ptr_array_basic_units,
                                                                                            ref_source_Neural_Network_received.ptr_array_ptr_connections,
                                                                                            this->ptr_array_ptr_connections);
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_destination_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_destination_layer_it->type_layer].c_str());
                            return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                switch(tmp_ptr_destination_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                        this->Copy__Layer__FC<struct Neuron_unit>(tmp_ptr_source_layer_it,
                                                                                           tmp_ptr_destination_layer_it,
                                                                                           ref_source_Neural_Network_received.ptr_array_neuron_units,
                                                                                           this->ptr_array_neuron_units,
                                                                                           reinterpret_cast<struct Neuron_unit**>(ref_source_Neural_Network_received.ptr_array_ptr_connections),
                                                                                           reinterpret_cast<struct Neuron_unit**>(this->ptr_array_ptr_connections));
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                        this->Copy__Layer__LSTM<struct Neuron_unit>(tmp_ptr_source_layer_it,
                                                                                        tmp_ptr_destination_layer_it,
                                                                                        ref_source_Neural_Network_received.ptr_array_cell_units,
                                                                                        ref_source_Neural_Network_received.ptr_array_neuron_units,
                                                                                        this->ptr_array_neuron_units,
                                                                                        ref_source_Neural_Network_received.ptr_array_ptr_connections,
                                                                                        this->ptr_array_ptr_connections);
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_destination_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_destination_layer_it->type_layer].c_str());
                            return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                switch(tmp_ptr_destination_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                        this->Copy__Layer__FC<struct Cell_unit>(tmp_ptr_source_layer_it,
                                                                                      tmp_ptr_destination_layer_it,
                                                                                      ref_source_Neural_Network_received.ptr_array_cell_units,
                                                                                      this->ptr_array_cell_units,
                                                                                      reinterpret_cast<struct Cell_unit**>(ref_source_Neural_Network_received.ptr_array_ptr_connections),
                                                                                      reinterpret_cast<struct Cell_unit**>(this->ptr_array_ptr_connections));
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                        this->Copy__Layer__LSTM<struct Cell_unit>(tmp_ptr_source_layer_it,
                                                                                          tmp_ptr_destination_layer_it,
                                                                                          ref_source_Neural_Network_received.ptr_array_cell_units,
                                                                                          ref_source_Neural_Network_received.ptr_array_cell_units,
                                                                                          this->ptr_array_cell_units,
                                                                                          ref_source_Neural_Network_received.ptr_array_ptr_connections,
                                                                                          this->ptr_array_ptr_connections);
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_destination_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_destination_layer_it->type_layer].c_str());
                            return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                switch(tmp_ptr_destination_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                        this->Copy__Layer__FC<struct Basic_indice_unit>(tmp_ptr_source_layer_it,
                                                                                                   tmp_ptr_destination_layer_it,
                                                                                                   ref_source_Neural_Network_received.ptr_array_basic_indice_units,
                                                                                                   this->ptr_array_basic_indice_units,
                                                                                                   reinterpret_cast<struct Basic_indice_unit**>(ref_source_Neural_Network_received.ptr_array_ptr_connections),
                                                                                                   reinterpret_cast<struct Basic_indice_unit**>(this->ptr_array_ptr_connections));
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                        this->Copy__Layer__LSTM<struct Basic_indice_unit>(tmp_ptr_source_layer_it,
                                                                                                       tmp_ptr_destination_layer_it,
                                                                                                       ref_source_Neural_Network_received.ptr_array_cell_units,
                                                                                                       ref_source_Neural_Network_received.ptr_array_basic_indice_units,
                                                                                                       this->ptr_array_basic_indice_units,
                                                                                                       ref_source_Neural_Network_received.ptr_array_ptr_connections,
                                                                                                       this->ptr_array_ptr_connections);
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_destination_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_destination_layer_it->type_layer].c_str());
                            return(false);
                }
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_destination_previous_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_destination_previous_layer_it->type_layer].c_str());
                    return(false);
        }
    }
    // |END| Copy connections. |END|
    
    // Dropout.
    this->Copy__Dropout(ref_source_Neural_Network_received.ptr_array_layers,
                                   ref_source_Neural_Network_received.Get__End_Layer__Active() - 1, // Get last active layer.
                                   this->ptr_array_layers);
    // |END| Dropout. |END|
    
    // Normalization.
    this->Copy__Normalization(ref_source_Neural_Network_received.ptr_array_layers + 1, // Skip input layer.
                                            ref_source_Neural_Network_received.Get__End_Layer__Active() == ref_source_Neural_Network_received.ptr_last_layer ? (ref_source_Neural_Network_received.ptr_last_layer - 1) : ref_source_Neural_Network_received.Get__End_Layer__Active(),
                                            this->ptr_array_layers + 1); // Skip input layer.

    this->Copy__Normalization(&ref_source_Neural_Network_received);

    this->Copy__Normalized_Units(0_zu,
                                                  ref_source_Neural_Network_received.total_normalized_units_allocated,
                                                  ref_source_Neural_Network_received.ptr_array_normalized_units);
    // |END| Normalization. |END|

    // Parameters.
    MEMCPY(this->ptr_array_parameters,
                   ref_source_Neural_Network_received.ptr_array_parameters,
                   ref_source_Neural_Network_received.total_parameters * sizeof(T_));
    // |END| Parameters. |END|
    
    // Initializer weight parameters.
    this->Copy__Initializer__Weight_Parameter(ref_source_Neural_Network_received);
    // |END| Initializer weight parameters. |END|
    
    // Training parameters.
    this->Copy__Training_Parameters(&ref_source_Neural_Network_received);
    // |END| Training parameters. |END|
    
    // Optimizer parameters.
    if(this->Copy__Optimizer_Parameters(&ref_source_Neural_Network_received, copy_delta_optimizer_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Optimizer_Parameters(ref, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 copy_delta_optimizer_received ? "true" : "false",
                                 __LINE__);

        return(false);
    }
    // |END| Optimizer parameters. |END|

    // Regularization parameters.
    this->Copy__Regularization(&ref_source_Neural_Network_received);
    
    if(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER)
    {
        this->Copy__Tied_Weight(ref_source_Neural_Network_received.ptr_array_layers + 1, // Skip input layer.
                                              ref_source_Neural_Network_received.Get__End_Layer__Active(),
                                              this->ptr_array_layers + 1); // Skip input layer.
    }

    this->Copy__Sparse_K_Filters(ref_source_Neural_Network_received.ptr_array_layers + 1, // Skip input layer.
                                                 ref_source_Neural_Network_received.Get__End_Layer__Active() == ref_source_Neural_Network_received.ptr_last_layer ? (ref_source_Neural_Network_received.ptr_last_layer - 1) : ref_source_Neural_Network_received.Get__End_Layer__Active(),
                                                 this->ptr_array_layers + 1); // Skip input layer.
    
    this->Copy__Constraint_Recurrent_Weight(ref_source_Neural_Network_received.ptr_array_layers + 1, // Skip input layer.
                                                                   ref_source_Neural_Network_received.Get__End_Layer__Active() == ref_source_Neural_Network_received.ptr_last_layer ? (ref_source_Neural_Network_received.ptr_last_layer - 1) : ref_source_Neural_Network_received.Get__End_Layer__Active(),
                                                                   this->ptr_array_layers + 1); // Skip input layer.
    // |END| Regularization parameters. |END|
    
    // Compute parameters.
    this->maximum_allowable_memory_bytes = ref_source_Neural_Network_received.maximum_allowable_memory_bytes;
    this->percentage_maximum_thread_usage = ref_source_Neural_Network_received.percentage_maximum_thread_usage;

    this->maximum_batch_size = ref_source_Neural_Network_received.maximum_batch_size;
    
    if(initialize_parallel_computation_received)
    {
        this->Set__OpenMP(ref_source_Neural_Network_received.use_OpenMP);

    #if defined(COMPILE_CUDA)
        this->Set__CUDA(ref_source_Neural_Network_received.use_CUDA, ref_source_Neural_Network_received.is_device_initialized ? ref_source_Neural_Network_received.ptr_device_Neural_Network->Get__Maximum_Allowable_Memory() : 0_zu);
    #endif
    }
    else
    {
        this->use_OpenMP = ref_source_Neural_Network_received.use_OpenMP;

    #if defined(COMPILE_CUDA)
        this->use_CUDA = ref_source_Neural_Network_received.use_CUDA;
    #endif
    }
    // |END| Compute parameters. |END|

    return(true);
}

bool Neural_Network::Update(class Neural_Network const &ref_source_Neural_Network_received,
                                           bool const initialize_parallel_computation_received,
                                           bool const update_delta_optimizer_received)
{
    // Lambda: Redirect to copy.
    auto tmp_Redirect_To_Copy([self = this,
                                               &tmp_source_Neural_Network = ref_source_Neural_Network_received,
                                               tmp_initialize_parallel_computation = initialize_parallel_computation_received,
                                               tmp_update_delta_optimizer = update_delta_optimizer_received]() -> bool
    {
        if(self->Copy(tmp_source_Neural_Network,
                           tmp_initialize_parallel_computation,
                           tmp_update_delta_optimizer) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy(ref, %s, %s)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_initialize_parallel_computation ? "true" : "false",
                                     tmp_update_delta_optimizer ? "true" : "false",
                                     __LINE__);

            return(false);
        }

        return(true);
    });

    // Compare network topology. If different redirect to "Copy" function.
    if(this->total_layers != ref_source_Neural_Network_received.total_layers
       ||
       this->total_weights != ref_source_Neural_Network_received.total_weights
       ||
       this->total_bias != ref_source_Neural_Network_received.total_bias
       ||
       this->total_neuron_units != ref_source_Neural_Network_received.total_neuron_units
       ||
       this->total_AF_units != ref_source_Neural_Network_received.total_AF_units
       ||
       this->total_AF_Ind_recurrent_units != ref_source_Neural_Network_received.total_AF_Ind_recurrent_units
       ||
       this->total_cell_units != ref_source_Neural_Network_received.total_cell_units
       ||
       this->total_block_units != ref_source_Neural_Network_received.total_block_units) { return(tmp_Redirect_To_Copy()); }

    /*
#if defined(COMPILE_CUDA)
    if(ref_source_Neural_Network_received.is_device_initialized
      &&
      ref_source_Neural_Network_received.is_update_from_device == false)
    {
        if(ref_source_Neural_Network_received.Copy__Parameters__Device_To_Host() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Parameters__Device_To_Host()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        else if(ref_source_Neural_Network_received.Use__Normalization() && ref_source_Neural_Network_received.Copy__Batch_Normalization_Neurons__Device_To_Host() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Batch_Normalization_Neurons__Device_To_Host()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
#endif
    */

    // Dropout.
    this->Copy__Dropout(ref_source_Neural_Network_received.ptr_array_layers,
                                   ref_source_Neural_Network_received.Get__End_Layer__Active() - 1, // Get last active layer.
                                   this->ptr_array_layers);
    // |END| Dropout. |END|
    
    // Normalization.
    this->Copy__Normalization(ref_source_Neural_Network_received.ptr_array_layers + 1, // Skip input layer.
                                            ref_source_Neural_Network_received.Get__End_Layer__Active() == ref_source_Neural_Network_received.ptr_last_layer ? (ref_source_Neural_Network_received.ptr_last_layer - 1) : ref_source_Neural_Network_received.Get__End_Layer__Active(),
                                            this->ptr_array_layers + 1); // Skip input layer.
    
    this->Copy__Normalization(&ref_source_Neural_Network_received);

    this->Copy__Normalized_Units(0_zu,
                                                  ref_source_Neural_Network_received.total_normalized_units_allocated,
                                                  ref_source_Neural_Network_received.ptr_array_normalized_units);
    // |END| Normalization. |END|

    // Loss parameters.
    this->Copy__Loss(&ref_source_Neural_Network_received);
    // |END| Loss parameters. |END|

    // Accuracy parameters.
    this->Copy__Accuracy(&ref_source_Neural_Network_received);
    // |END| Accuracy parameters. |END|
    
    // Compare total parameters (Can be modified by normalization).
    if(this->total_parameters != ref_source_Neural_Network_received.total_parameters) { return(tmp_Redirect_To_Copy()); }
    
    // Parameters.
    MEMCPY(this->ptr_array_parameters,
                   ref_source_Neural_Network_received.ptr_array_parameters,
                   ref_source_Neural_Network_received.total_parameters * sizeof(T_));
    // |END| Parameters. |END|
    
    // Initializer weight parameters.
    this->Copy__Initializer__Weight_Parameter(ref_source_Neural_Network_received);
    // |END| Initializer weight parameters. |END|
    
    // Training parameters.
    this->Copy__Training_Parameters(&ref_source_Neural_Network_received);
    // |END| Training parameters. |END|
    
    // Optimizer parameters.
    if(this->Copy__Optimizer_Parameters(&ref_source_Neural_Network_received, update_delta_optimizer_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Optimizer_Parameters(ref, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 update_delta_optimizer_received ? "true" : "false",
                                 __LINE__);

        return(false);
    }
    // |END| Optimizer parameters. |END|
    
    // Regularization parameters.
    this->Copy__Regularization(&ref_source_Neural_Network_received);
    
    if(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER)
    {
        this->Copy__Tied_Weight(ref_source_Neural_Network_received.ptr_array_layers + 1, // Skip input layer.
                                              ref_source_Neural_Network_received.Get__End_Layer__Active(),
                                              this->ptr_array_layers + 1); // Skip input layer.
    }

    this->Copy__Sparse_K_Filters(ref_source_Neural_Network_received.ptr_array_layers + 1, // Skip input layer.
                                                 ref_source_Neural_Network_received.Get__End_Layer__Active() == ref_source_Neural_Network_received.ptr_last_layer ? (ref_source_Neural_Network_received.ptr_last_layer - 1) : ref_source_Neural_Network_received.Get__End_Layer__Active(),
                                                 this->ptr_array_layers + 1); // Skip input layer.
    
    this->Copy__Constraint_Recurrent_Weight(ref_source_Neural_Network_received.ptr_array_layers + 1, // Skip input layer.
                                                                   ref_source_Neural_Network_received.Get__End_Layer__Active() == ref_source_Neural_Network_received.ptr_last_layer ? (ref_source_Neural_Network_received.ptr_last_layer - 1) : ref_source_Neural_Network_received.Get__End_Layer__Active(),
                                                                   this->ptr_array_layers + 1); // Skip input layer.
    // |END| Regularization parameters. |END|
    
    // Compute parameters.
    this->maximum_allowable_memory_bytes = ref_source_Neural_Network_received.maximum_allowable_memory_bytes;
    
    if(this->Set__Maximum_Thread_Usage(ref_source_Neural_Network_received.percentage_maximum_thread_usage) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Maximum_Thread_Usage(%f)\" function. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    ref_source_Neural_Network_received.percentage_maximum_thread_usage,
                                    __LINE__);

        return(false);
    }
    else if(this->Set__Maximum__Batch_Size(ref_source_Neural_Network_received.maximum_batch_size) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Maximum__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_source_Neural_Network_received.maximum_batch_size,
                                 __LINE__);

        return(false);
    }

    if(initialize_parallel_computation_received)
    {
        this->Set__OpenMP(ref_source_Neural_Network_received.use_OpenMP);

    #if defined(COMPILE_CUDA)
        this->Set__CUDA(ref_source_Neural_Network_received.use_CUDA, ref_source_Neural_Network_received.is_device_initialized ? ref_source_Neural_Network_received.ptr_device_Neural_Network->Get__Maximum_Allowable_Memory() : 0_zu);
    #endif
    }
    else
    {
        this->use_OpenMP = ref_source_Neural_Network_received.use_OpenMP;

    #if defined(COMPILE_CUDA)
        this->use_CUDA = ref_source_Neural_Network_received.use_CUDA;
    #endif
    }
    // |END| Compute parameters. |END|

    return(true);
}

void Neural_Network::Copy__Warm_Restarts_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
    this->use_Warm_Restarts = ptr_Neural_Network_received->use_Warm_Restarts;
    this->warm_restarts_decay_learning_rate = ptr_Neural_Network_received->warm_restarts_decay_learning_rate;
    this->warm_restarts_initial_maximum_learning_rate = ptr_Neural_Network_received->warm_restarts_initial_maximum_learning_rate;
    this->warm_restarts_maximum_learning_rate = ptr_Neural_Network_received->warm_restarts_maximum_learning_rate;
    this->warm_restarts_minimum_learning_rate = ptr_Neural_Network_received->warm_restarts_minimum_learning_rate;
    this->warm_restarts_initial_T_i = ptr_Neural_Network_received->warm_restarts_initial_T_i;
    this->warm_restarts_T_i = ptr_Neural_Network_received->warm_restarts_T_i;
    this->warm_restarts_multiplier = ptr_Neural_Network_received->warm_restarts_multiplier;
}

bool Neural_Network::Copy__Optimizer_Parameters(class Neural_Network const *const ptr_Neural_Network_received, bool const copy_delta_optimizer_received)
{
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD: this->Copy__Gradient_Descent_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus: this->Copy__RPROP_minus_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: this->Copy__RPROP_plus_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP: this->Copy__SARProp_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP: this->Copy__QuickProp_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: this->Copy__Adam_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Copy__NosAdam_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND: this->Copy__AdaBound_Parameters(ptr_Neural_Network_received); break;
        default:
            PRINT_FORMAT("%s: ERROR: Can not copy parameters of the optimizer (%u | %s)." NEW_LINE,
                        __FUNCTION__,
                        this->type_optimizer_function,
                        MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
                return(false);
    }

    if(copy_delta_optimizer_received)
    {
        switch(this->type_optimizer_function)
        {
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
                if(this->Copy__Delta__Gradient_Descent(ptr_Neural_Network_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Delta__Gradient_Descent(ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
                if(this->Copy__Delta__iRPROP_minus(ptr_Neural_Network_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Delta__iRPROP_minus(ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
                if(this->Copy__Delta__iRPROP_plus(ptr_Neural_Network_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Delta__iRPROP_plus(ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM:
                if(this->Copy__Delta__Adam(ptr_Neural_Network_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Delta__Adam(ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad:
            case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
                if(this->Copy__Delta__AMSGrad(ptr_Neural_Network_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Delta__AMSGrad(ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            default:
                PRINT_FORMAT("%s: ERROR: Can not allocate parameters of the optimizer (%u | %s)." NEW_LINE,
                                         __FUNCTION__,
                                         this->type_optimizer_function,
                                         MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
                    return(false);
        }
    }

    this->Copy__Warm_Restarts_Parameters(ptr_Neural_Network_received);

    this->optimizer_time_step = ptr_Neural_Network_received->optimizer_time_step;
    this->epoch_time_step = ptr_Neural_Network_received->epoch_time_step;

#if defined(COMPILE_CUDA)
    if(this->is_device_initialized)
    { this->ptr_device_Neural_Network->Copy__Optimizer_Parameters(this); }
#endif

    return(true);
}

void Neural_Network::Copy__Gradient_Descent_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
    // Gradient descent parameters.
    T_ const tmp_learning_momentum(this->learning_momentum);

    this->learning_rate = ptr_Neural_Network_received->learning_rate;
    this->learning_momentum = ptr_Neural_Network_received->learning_momentum;
    this->use_Nesterov = ptr_Neural_Network_received->use_Nesterov;
    
    if(tmp_learning_momentum == 0_T)
    { this->Allocate__Parameter__Gradient_Descent(); }
    else if(this->learning_momentum == 0_T)
    { this->Deallocate__Parameter__Gradient_Descent(); }
    // |END| Gradient descent parameters. |END|
}

void Neural_Network::Copy__QuickProp_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
    // Quickprop parameters.
    this->quickprop_decay = ptr_Neural_Network_received->quickprop_decay;
    this->quickprop_mu = ptr_Neural_Network_received->quickprop_mu;
    // |END| Quickprop parameters. |END|
}

void Neural_Network::Copy__RPROP_minus_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
    // Resillent propagation minus parameters.
    this->rprop_increase_factor = ptr_Neural_Network_received->rprop_increase_factor;
    this->rprop_decrease_factor = ptr_Neural_Network_received->rprop_decrease_factor;
    this->rprop_delta_min = ptr_Neural_Network_received->rprop_delta_min;
    this->rprop_delta_max = ptr_Neural_Network_received->rprop_delta_max;
    this->rprop_delta_zero = ptr_Neural_Network_received->rprop_delta_zero;
    // |END| Resillent propagation minus parameters. |END|
}

void Neural_Network::Copy__RPROP_plus_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
    // Resillent propagation plus parameters.
    this->Copy__RPROP_minus_Parameters(ptr_Neural_Network_received);

    this->loss_rprop = ptr_Neural_Network_received->loss_rprop;
    this->previous_loss_rprop = ptr_Neural_Network_received->previous_loss_rprop;
    // |END| Resillent propagation plus parameters. |END|
}

void Neural_Network::Copy__SARProp_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
    // SARProp parameters.
    this->sarprop_weight_decay_shift = ptr_Neural_Network_received->sarprop_weight_decay_shift;
    this->sarprop_step_error_threshold_factor = ptr_Neural_Network_received->sarprop_step_error_threshold_factor;
    this->sarprop_step_error_shift = ptr_Neural_Network_received->sarprop_step_error_shift;
    this->sarprop_temperature = ptr_Neural_Network_received->sarprop_temperature;
    this->sarprop_epoch = ptr_Neural_Network_received->sarprop_epoch;
    // |END| SARProp parameters. |END|
}

void Neural_Network::Copy__Adam_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
    // Adam parameters.
     this->adam_learning_rate = ptr_Neural_Network_received->adam_learning_rate;
     this->adam_beta1 = ptr_Neural_Network_received->adam_beta1;
     this->adam_beta2 = ptr_Neural_Network_received->adam_beta2;
     this->adam_epsilon = ptr_Neural_Network_received->adam_epsilon;
     this->use_adam_bias_correction = ptr_Neural_Network_received->use_adam_bias_correction;
    // |END| Adam parameters. |END|
}

void Neural_Network::Copy__NosAdam_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
    // Adam parameters.
    this->Copy__Adam_Parameters(ptr_Neural_Network_received);

     this->adam_gamma = ptr_Neural_Network_received->adam_gamma;
    // |END| Adam parameters. |END|
}

void Neural_Network::Copy__AdaBound_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
    // Adam parameters.
    this->Copy__Adam_Parameters(ptr_Neural_Network_received);

     this->learning_rate_final = ptr_Neural_Network_received->learning_rate_final;

     this->learning_gamma = ptr_Neural_Network_received->learning_gamma;
    // |END| Adam parameters. |END|
}

bool Neural_Network::Copy__Delta__Gradient_Descent(class Neural_Network const *const ptr_Neural_Network_received)
{
    if(this->learning_momentum == 0_T) { return(true); }
    else if(ptr_Neural_Network_received->ptr_array_previous_delta_parameters == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Source array \"ptr_array_previous_delta_parameters\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->ptr_array_previous_delta_parameters == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Destination array \"ptr_array_previous_delta_parameters\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    MEMCPY(this->ptr_array_previous_delta_parameters,
                 ptr_Neural_Network_received->ptr_array_previous_delta_parameters,
                 this->total_parameters * sizeof(T_));

    return(true);
}

bool Neural_Network::Copy__Delta__iRPROP_minus(class Neural_Network const *const ptr_Neural_Network_received)
{
    if(ptr_Neural_Network_received->ptr_array_previous_steps == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Source array \"ptr_array_previous_steps\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->ptr_array_previous_steps == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Destination array \"ptr_array_previous_steps\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else
    {
        MEMCPY(this->ptr_array_previous_steps,
                     ptr_Neural_Network_received->ptr_array_previous_steps,
                     this->total_parameters * sizeof(T_));
    }
    
    if(ptr_Neural_Network_received->ptr_array_previous_derivatives_parameters == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Source array \"ptr_array_previous_derivatives_parameters\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->ptr_array_previous_derivatives_parameters == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Destination array \"ptr_array_previous_derivatives_parameters\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else
    {
        MEMCPY(this->ptr_array_previous_derivatives_parameters,
                     ptr_Neural_Network_received->ptr_array_previous_derivatives_parameters,
                     this->total_parameters * sizeof(T_));
    }

    return(true);
}

bool Neural_Network::Copy__Delta__iRPROP_plus(class Neural_Network const *const ptr_Neural_Network_received)
{
    if(this->Copy__Delta__iRPROP_minus(ptr_Neural_Network_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Delta__iRPROP_minus()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(ptr_Neural_Network_received->ptr_array_previous_delta_parameters == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Source array \"ptr_array_previous_delta_parameters\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->ptr_array_previous_delta_parameters == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Destination array \"ptr_array_previous_delta_parameters\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else
    {
        MEMCPY(this->ptr_array_previous_delta_parameters,
                     ptr_Neural_Network_received->ptr_array_previous_delta_parameters,
                     this->total_parameters * sizeof(T_));
    }

    return(true);
}

bool Neural_Network::Copy__Delta__Adam(class Neural_Network const *const ptr_Neural_Network_received)
{
    if(ptr_Neural_Network_received->ptr_array_previous_biased_first_moment == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Source array \"ptr_array_previous_biased_first_moment\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->ptr_array_previous_biased_first_moment == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Destination array \"ptr_array_previous_biased_first_moment\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else
    {
        MEMCPY(this->ptr_array_previous_biased_first_moment,
                     ptr_Neural_Network_received->ptr_array_previous_biased_first_moment,
                     this->total_parameters * sizeof(T_));
    }
    
    if(ptr_Neural_Network_received->ptr_array_previous_biased_second_moment == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Source array \"ptr_array_previous_biased_second_moment\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->ptr_array_previous_biased_second_moment == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Destination array \"ptr_array_previous_biased_second_moment\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else
    {
        MEMCPY(this->ptr_array_previous_biased_second_moment,
                     ptr_Neural_Network_received->ptr_array_previous_biased_second_moment,
                     this->total_parameters * sizeof(T_));
    }

    return(true);
}

bool Neural_Network::Copy__Delta__AMSGrad(class Neural_Network const *const ptr_Neural_Network_received)
{
    if(this->Copy__Delta__Adam(ptr_Neural_Network_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Delta__Adam()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(ptr_Neural_Network_received->ptr_array_previous_biased_second_moment_hat == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Source array \"ptr_array_previous_biased_second_moment_hat\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->ptr_array_previous_biased_second_moment_hat == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Destination array \"ptr_array_previous_biased_second_moment_hat\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else
    {
        MEMCPY(this->ptr_array_previous_biased_second_moment_hat,
                     ptr_Neural_Network_received->ptr_array_previous_biased_second_moment_hat,
                     this->total_parameters * sizeof(T_));
    }

    return(true);
}

void Neural_Network::Copy__Training_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
    this->Set__Optimizer_Function(ptr_Neural_Network_received->type_optimizer_function);
    this->Set__Loss_Function(ptr_Neural_Network_received->type_loss_function);
    this->Set__Accuracy_Function(ptr_Neural_Network_received->type_accuracy_function);
    this->Set__Bit_Fail_Limit(ptr_Neural_Network_received->bit_fail_limit);
    
    if(this->Set__Clip_Gradient(ptr_Neural_Network_received->clip_gradient) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Clip_Gradient(%f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(ptr_Neural_Network_received->clip_gradient),
                                 __LINE__);
    }
    else if(this->Set__Pre_Training_Level(ptr_Neural_Network_received->pre_training_level) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Pre_Training_Level(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_Neural_Network_received->pre_training_level,
                                 __LINE__);
    }
    else if(this->Set__Number_Time_Delays(ptr_Neural_Network_received->number_time_delays) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Number_Time_Delays(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_Neural_Network_received->number_time_delays,
                                 __LINE__);
    }
}

void Neural_Network::Copy__Initializer__Weight_Parameter(class Neural_Network const &ref_source_Neural_Network_received)
{
    this->_initialized__weight = ref_source_Neural_Network_received._initialized__weight;
    
    this->_type_weights_initializer = ref_source_Neural_Network_received._type_weights_initializer;

    this->_LSUV_Parameters = ref_source_Neural_Network_received._LSUV_Parameters;
}

void Neural_Network::Copy__Regularization(class Neural_Network const *const ptr_Neural_Network_received)
{
    // Regularization parameters.
    if(this->Set__Regularization__Max_Norm_Constraints(ptr_Neural_Network_received->regularization__max_norm_constraints) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Max_Norm_Constraints(%f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(ptr_Neural_Network_received->regularization__max_norm_constraints),
                                 __LINE__);

        return;
    }
    else if(this->Set__Regularization__L1(ptr_Neural_Network_received->regularization__l1) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__L1(%f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(ptr_Neural_Network_received->regularization__l1),
                                 __LINE__);

        return;
    }
    else if(this->Set__Regularization__L2(ptr_Neural_Network_received->regularization__l2) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__L2(%f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(ptr_Neural_Network_received->regularization__l2),
                                 __LINE__);

        return;
    }
    else if(this->Set__Regularization__SRIP(ptr_Neural_Network_received->regularization__srip) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__SRIP(%f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(ptr_Neural_Network_received->regularization__srip),
                                 __LINE__);

        return;
    }
    else if(this->Set__Regularization__Weight_Decay(ptr_Neural_Network_received->regularization__weight_decay) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Weight_Decay(%f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(ptr_Neural_Network_received->regularization__weight_decay),
                                 __LINE__);

        return;
    }
    // |END| Regularization parameters. |END|
}

void Neural_Network::Copy__Tied_Weight(struct Layer const *ptr_array_source_layers_received,
                                                             struct Layer const *const ptr_last_source_layer_received,
                                                             struct Layer *ptr_array_destination_layers_received)
{
    for(; ptr_array_source_layers_received != ptr_last_source_layer_received; ++ptr_array_source_layers_received,
                                                                                                              ++ptr_array_destination_layers_received)
    {
        if(this->Set__Tied_Parameter(ptr_array_destination_layers_received, ptr_array_source_layers_received->use_tied_parameter) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Tied_Parameter(ptr, %s)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_array_source_layers_received->use_tied_parameter ? "true" : "false",
                                     __LINE__);
        }
    }
}

void Neural_Network::Copy__Sparse_K_Filters(struct Layer const *ptr_array_source_layers_received,
                                                                    struct Layer const *const ptr_last_source_layer_received,
                                                                    struct Layer *ptr_array_destination_layers_received)
{
    for(; ptr_array_source_layers_received != ptr_last_source_layer_received; ++ptr_array_source_layers_received,
                                                                                                              ++ptr_array_destination_layers_received)
    {
        if(this->Set__K_Sparsity(ptr_array_destination_layers_received, ptr_array_source_layers_received->k_sparsity) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__K_Sparsity(ptr, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_array_source_layers_received->k_sparsity,
                                     __LINE__);

            return;
        }
        else if(this->Set__Alpha_Sparsity(ptr_array_destination_layers_received, ptr_array_source_layers_received->alpha_sparsity) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Alpha_Sparsity(%f)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     Cast_T(ptr_array_source_layers_received->alpha_sparsity),
                                     __LINE__);

            return;
        }
    }
}

void Neural_Network::Copy__Constraint_Recurrent_Weight(struct Layer const *ptr_array_source_layers_received,
                                                                                      struct Layer const *const ptr_last_source_layer_received,
                                                                                      struct Layer *ptr_array_destination_layers_received)
{
    for(; ptr_array_source_layers_received != ptr_last_source_layer_received; ++ptr_array_source_layers_received,
                                                                                                              ++ptr_array_destination_layers_received)
    {
        if(this->Set__Regularization__Constraint_Recurrent_Weight(ptr_array_destination_layers_received,
                                                                                               ptr_array_source_layers_received->constraint_recurrent_weight_lower_bound,
                                                                                               ptr_array_source_layers_received->constraint_recurrent_weight_upper_bound) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight(ptr, %f, %f)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     Cast_T(ptr_array_source_layers_received->constraint_recurrent_weight_lower_bound),
                                     Cast_T(ptr_array_source_layers_received->constraint_recurrent_weight_upper_bound),
                                     __LINE__);

            return;
        }
    }
}

void Neural_Network::Copy__Loss(class Neural_Network const *const ptr_Neural_Network_received)
{
    // Loss parameters.
    this->loss_training = ptr_Neural_Network_received->loss_training;
    this->loss_validating = ptr_Neural_Network_received->loss_validating;
    this->loss_testing = ptr_Neural_Network_received->loss_testing;
    // |END| Loss parameters. |END|
}

void Neural_Network::Copy__Accuracy(class Neural_Network const *const ptr_Neural_Network_received)
{
    // Accuracy parameters.
    this->accuracy_variance = ptr_Neural_Network_received->accuracy_variance;
    this->accuracy_training = ptr_Neural_Network_received->accuracy_training;
    this->accuracy_validating = ptr_Neural_Network_received->accuracy_validating;
    this->accuracy_testing = ptr_Neural_Network_received->accuracy_testing;
    // |END| Accuracy parameters. |END|
}

void Neural_Network::Copy__Dropout(struct Layer const *ptr_array_source_layers_received,
                                                       struct Layer const *const ptr_last_source_layer_received,
                                                       struct Layer *ptr_array_destination_layers_received)
{
#if defined(COMPILE_CUDA)
    bool tmp_parameters_has_change(false);
#endif
    
    for(; ptr_array_source_layers_received != ptr_last_source_layer_received; ++ptr_array_source_layers_received,
                                                                                                              ++ptr_array_destination_layers_received)
    {
    #if defined(COMPILE_CUDA)
        if(ptr_array_source_layers_received->dropout_values[0u] != ptr_array_destination_layers_received->dropout_values[0u]
          ||
          ptr_array_source_layers_received->type_dropout != ptr_array_destination_layers_received->type_dropout)
        { tmp_parameters_has_change = true; }
    #endif
        
        if(this->Set__Dropout(ptr_array_destination_layers_received,
                                        ptr_array_source_layers_received->type_dropout,
                                        ptr_array_source_layers_received->dropout_values,
                                        false) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization(ptr, %s, %f, %f, %f, false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[ptr_array_source_layers_received->type_dropout].c_str(),
                                     Cast_T(ptr_array_source_layers_received->dropout_values[0u]),
                                     Cast_T(ptr_array_source_layers_received->dropout_values[1u]),
                                     Cast_T(ptr_array_source_layers_received->dropout_values[2u]),
                                     __LINE__);

            return;
        }
    }

#if defined(COMPILE_CUDA)
    if(this->is_device_initialized && tmp_parameters_has_change)
    { this->ptr_device_Neural_Network->Copy__Dropout(this); }
#endif
}

void Neural_Network::Copy__Normalization(class Neural_Network const *const ptr_source_Neural_Network_received)
{
    if(this->Set__Normalization_Momentum_Average(ptr_source_Neural_Network_received->normalization_momentum_average) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_Momentum_Average(%f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(ptr_source_Neural_Network_received->normalization_momentum_average),
                                 __LINE__);

        return;
    }
    else if(this->Set__Normalization_Epsilon(ptr_source_Neural_Network_received->normalization_epsilon) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_Epsilon(%f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(ptr_source_Neural_Network_received->normalization_epsilon),
                                 __LINE__);

        return;
    }
    else if(this->Set__Batch_Renormalization_r_Correction_Maximum(ptr_source_Neural_Network_received->batch_renormalization_r_correction_maximum) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Batch_Renormalization_r_Correction_Maximum(%f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(ptr_source_Neural_Network_received->batch_renormalization_r_correction_maximum),
                                 __LINE__);

        return;
    }
    else if(this->Set__Batch_Renormalization_d_Correction_Maximum(ptr_source_Neural_Network_received->batch_renormalization_d_correction_maximum) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Batch_Renormalization_d_Correction_Maximum(%f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(ptr_source_Neural_Network_received->batch_renormalization_d_correction_maximum),
                                 __LINE__);

        return;
    }
}

void Neural_Network::Copy__Normalization(struct Layer const *ptr_array_source_layers_received,
                                                               struct Layer const *const ptr_last_source_layer_received,
                                                               struct Layer *ptr_array_destination_layers_received)
{
#if defined(COMPILE_CUDA)
    bool tmp_parameters_has_change(false);
#endif
    
    // Hidden layer(s).
    for(; ptr_array_source_layers_received != ptr_last_source_layer_received; ++ptr_array_source_layers_received,
                                                                                                              ++ptr_array_destination_layers_received)
    {
    #if defined(COMPILE_CUDA)
        if(ptr_array_source_layers_received->type_normalization != ptr_array_destination_layers_received->type_normalization) { tmp_parameters_has_change = true; }
    #endif
        
        if(this->Set__Layer_Normalization(ptr_array_destination_layers_received, ptr_array_source_layers_received->type_normalization) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization(ptr, %s)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[ptr_array_source_layers_received->type_normalization].c_str(),
                                     __LINE__);

            return;
        }
    }
    // |END| Hidden layer(s). |END|

#if defined(COMPILE_CUDA)
    if(this->is_device_initialized && (tmp_parameters_has_change || this->Use__Normalization()))
    { this->ptr_device_Neural_Network->Copy__Normalization(this); }
#endif
}

void Neural_Network::Copy__Block(struct Block_unit const *const ptr_source_block_unit_received, struct Block_unit *const ptr_destination_block_unit_received)
{
    struct Cell_unit const *tmp_ptr_source_cell_unit_it,
                                    *tmp_ptr_destination_block_ptr_last_unit;
    struct Cell_unit *tmp_ptr_destination_block_ptr_cell_unit_it;
    
    ptr_destination_block_unit_received->first_index_connection = ptr_source_block_unit_received->first_index_connection;
    ptr_destination_block_unit_received->last_index_connection = ptr_source_block_unit_received->last_index_connection;
    ptr_destination_block_unit_received->first_index_feedforward_connection_input_gate = ptr_source_block_unit_received->first_index_feedforward_connection_input_gate;
    ptr_destination_block_unit_received->last_index_feedforward_connection_input_gate = ptr_source_block_unit_received->last_index_feedforward_connection_input_gate;
    ptr_destination_block_unit_received->first_index_feedforward_connection_forget_gate = ptr_source_block_unit_received->first_index_feedforward_connection_forget_gate;
    ptr_destination_block_unit_received->last_index_feedforward_connection_forget_gate = ptr_source_block_unit_received->last_index_feedforward_connection_forget_gate;
    ptr_destination_block_unit_received->first_index_feedforward_connection_output_gate = ptr_source_block_unit_received->first_index_feedforward_connection_output_gate;
    ptr_destination_block_unit_received->last_index_feedforward_connection_output_gate = ptr_source_block_unit_received->last_index_feedforward_connection_output_gate;
    ptr_destination_block_unit_received->first_index_recurrent_connection_input_gate = ptr_source_block_unit_received->first_index_recurrent_connection_input_gate;
    ptr_destination_block_unit_received->last_index_recurrent_connection_input_gate = ptr_source_block_unit_received->last_index_recurrent_connection_input_gate;
    ptr_destination_block_unit_received->first_index_recurrent_connection_forget_gate = ptr_source_block_unit_received->first_index_recurrent_connection_forget_gate;
    ptr_destination_block_unit_received->last_index_recurrent_connection_forget_gate = ptr_source_block_unit_received->last_index_recurrent_connection_forget_gate;
    ptr_destination_block_unit_received->first_index_recurrent_connection_output_gate = ptr_source_block_unit_received->first_index_recurrent_connection_output_gate;
    ptr_destination_block_unit_received->last_index_recurrent_connection_output_gate = ptr_source_block_unit_received->last_index_recurrent_connection_output_gate;

#ifndef NO_PEEPHOLE
    ptr_destination_block_unit_received->first_index_peephole_input_gate = ptr_source_block_unit_received->first_index_peephole_input_gate;
    ptr_destination_block_unit_received->last_index_peephole_input_gate = ptr_source_block_unit_received->last_index_peephole_input_gate;
    ptr_destination_block_unit_received->first_index_peephole_forget_gate = ptr_source_block_unit_received->first_index_peephole_forget_gate;
    ptr_destination_block_unit_received->last_index_peephole_forget_gate = ptr_source_block_unit_received->last_index_peephole_forget_gate;
    ptr_destination_block_unit_received->first_index_peephole_output_gate = ptr_source_block_unit_received->first_index_peephole_output_gate;
    ptr_destination_block_unit_received->last_index_peephole_output_gate = ptr_source_block_unit_received->last_index_peephole_output_gate;
#endif

    this->Copy__Block__AF(ptr_source_block_unit_received, ptr_destination_block_unit_received);

    for(tmp_ptr_source_cell_unit_it = ptr_source_block_unit_received->ptr_array_cell_units,
        tmp_ptr_destination_block_ptr_last_unit = ptr_destination_block_unit_received->ptr_last_cell_unit,
        tmp_ptr_destination_block_ptr_cell_unit_it = ptr_destination_block_unit_received->ptr_array_cell_units; tmp_ptr_destination_block_ptr_cell_unit_it != tmp_ptr_destination_block_ptr_last_unit; ++tmp_ptr_destination_block_ptr_cell_unit_it,
                                                                                                                                                                                                                                            ++tmp_ptr_source_cell_unit_it)
    {
        tmp_ptr_destination_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input = tmp_ptr_source_cell_unit_it->first_index_feedforward_connection_cell_input;
        tmp_ptr_destination_block_ptr_cell_unit_it->last_index_feedforward_connection_cell_input = tmp_ptr_source_cell_unit_it->last_index_feedforward_connection_cell_input;
        tmp_ptr_destination_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input = tmp_ptr_source_cell_unit_it->first_index_recurrent_connection_cell_input;
        tmp_ptr_destination_block_ptr_cell_unit_it->last_index_recurrent_connection_cell_input = tmp_ptr_source_cell_unit_it->last_index_recurrent_connection_cell_input;
        
    #ifndef NO_PEEPHOLE
        tmp_ptr_destination_block_ptr_cell_unit_it->index_peephole_input_gate = tmp_ptr_source_cell_unit_it->index_peephole_input_gate;
        tmp_ptr_destination_block_ptr_cell_unit_it->index_peephole_forget_gate = tmp_ptr_source_cell_unit_it->index_peephole_forget_gate;
        tmp_ptr_destination_block_ptr_cell_unit_it->index_peephole_output_gate = tmp_ptr_source_cell_unit_it->index_peephole_output_gate;
    #endif
    }
}

void Neural_Network::Copy__Block__AF(struct Block_unit const *const ptr_source_block_unit_received, struct Block_unit *const ptr_destination_block_unit_received)
{
    ptr_destination_block_unit_received->activation_function_gate = ptr_source_block_unit_received->activation_function_gate;
    ptr_destination_block_unit_received->activation_function_io = ptr_source_block_unit_received->activation_function_io;
}

void Neural_Network::Copy__Blocks(size_t const start_index_received,
                                                     size_t const end_index_received,
                                                     struct Block_unit const *ptr_array_source_block_units_received,
                                                     bool const copy_connections_received)
{
    if(start_index_received + end_index_received == 0_zu) { return; }
    else if(start_index_received > end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 start_index_received,
                                 end_index_received,
                                 __LINE__);
        
        return;
    }

    struct Block_unit *tmp_ptr_destination_block_it(this->ptr_array_block_units + start_index_received);

    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        int const tmp_total_units__int(static_cast<int>(end_index_received - start_index_received));
        int tmp_unit_index__int;
        
        if(copy_connections_received)
        {
            #pragma omp parallel for schedule(static)
            for(tmp_unit_index__int = static_cast<int>(start_index_received); tmp_unit_index__int < tmp_total_units__int; ++tmp_unit_index__int)
            { this->Copy__Block(ptr_array_source_block_units_received + tmp_unit_index__int, tmp_ptr_destination_block_it + tmp_unit_index__int); }
        }
        else
        {
            #pragma omp parallel for schedule(static)
            for(tmp_unit_index__int = static_cast<int>(start_index_received); tmp_unit_index__int < tmp_total_units__int; ++tmp_unit_index__int)
            { this->Copy__Block__AF(ptr_array_source_block_units_received + tmp_unit_index__int, tmp_ptr_destination_block_it + tmp_unit_index__int); }
        }
    }
    else
    {
        struct Block_unit const *const tmp_ptr_destination_last_block_unit(tmp_ptr_destination_block_it + (end_index_received - start_index_received));
        
        if(copy_connections_received)
        {
            for(; tmp_ptr_destination_block_it != tmp_ptr_destination_last_block_unit; ++tmp_ptr_destination_block_it,
                                                                                                                       ++ptr_array_source_block_units_received)
            { this->Copy__Block(ptr_array_source_block_units_received, tmp_ptr_destination_block_it); }
        }
        else
        {
            for(; tmp_ptr_destination_block_it != tmp_ptr_destination_last_block_unit; ++tmp_ptr_destination_block_it,
                                                                                                                       ++ptr_array_source_block_units_received)
            { this->Copy__Block__AF(ptr_array_source_block_units_received, tmp_ptr_destination_block_it); }
        }
    }
}

void Neural_Network::Copy__Blocks__AF(size_t const start_index_received,
                                                             size_t const end_index_received,
                                                             struct Block_unit const *ptr_array_source_block_units_received)
{
    if(start_index_received + end_index_received == 0_zu) { return; }
    else if(start_index_received > end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 start_index_received,
                                 end_index_received,
                                 __LINE__);
        
        return;
    }

    struct Block_unit *tmp_ptr_destination_block_it(this->ptr_array_block_units + start_index_received);

    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        int const tmp_total_units__int(static_cast<int>(end_index_received - start_index_received));
        int tmp_unit_index__int;
        
        #pragma omp parallel for schedule(static)
        for(tmp_unit_index__int = static_cast<int>(start_index_received); tmp_unit_index__int < tmp_total_units__int; ++tmp_unit_index__int)
        { this->Copy__Block__AF(ptr_array_source_block_units_received + tmp_unit_index__int, tmp_ptr_destination_block_it + tmp_unit_index__int); }
    }
    else
    {
        struct Block_unit const *const tmp_ptr_destination_last_block_unit(tmp_ptr_destination_block_it + (end_index_received - start_index_received));

        for(; tmp_ptr_destination_block_it != tmp_ptr_destination_last_block_unit; ++tmp_ptr_destination_block_it,
                                                                                                                   ++ptr_array_source_block_units_received)
        { this->Copy__Block__AF(ptr_array_source_block_units_received, tmp_ptr_destination_block_it); }
    }
}

void Neural_Network::Copy__Neuron_Units(size_t const start_index_received,
                                                               size_t const end_index_received,
                                                               struct Neuron_unit const *ptr_array_source_neuron_units_received)
{
    if(start_index_received + end_index_received == 0_zu) { return; }
    else if(start_index_received > end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 start_index_received,
                                 end_index_received,
                                 __LINE__);
        
        return;
    }

    struct Neuron_unit *tmp_ptr_destination_neuron_it(this->ptr_array_neuron_units + start_index_received);

    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        int const tmp_total_units__int(static_cast<int>(end_index_received - start_index_received));
        int tmp_unit_index__int;
        
        #pragma omp parallel for schedule(static)
        for(tmp_unit_index__int = 0; tmp_unit_index__int < tmp_total_units__int; ++tmp_unit_index__int)
        { this->Copy__Neuron_Unit(ptr_array_source_neuron_units_received + tmp_unit_index__int, tmp_ptr_destination_neuron_it + tmp_unit_index__int); }
    }
    else
    {
        struct Neuron_unit const *const tmp_ptr_destination_last_neuron(tmp_ptr_destination_neuron_it + (end_index_received - start_index_received));

        for(; tmp_ptr_destination_neuron_it != tmp_ptr_destination_last_neuron; ++tmp_ptr_destination_neuron_it,
                                                                                                                ++ptr_array_source_neuron_units_received)
        { this->Copy__Neuron_Unit(ptr_array_source_neuron_units_received, tmp_ptr_destination_neuron_it); }
    }
}

void Neural_Network::Copy__Neuron_Unit(struct Neuron_unit const *const ptr_source_neuron_unit_received, struct Neuron_unit *const ptr_destination_neuron_unit_received)
{
    *ptr_destination_neuron_unit_received->ptr_first_connection_index = *ptr_source_neuron_unit_received->ptr_first_connection_index;
    *ptr_destination_neuron_unit_received->ptr_last_connection_index = *ptr_source_neuron_unit_received->ptr_last_connection_index;
    *ptr_destination_neuron_unit_received->ptr_number_connections = *ptr_source_neuron_unit_received->ptr_number_connections;
}

void Neural_Network::Copy__AF_Units(size_t const start_index_received,
                                                         size_t const end_index_received,
                                                         struct AF_unit const *ptr_array_source_AF_units_received)
{
    if(start_index_received + end_index_received == 0_zu) { return; }
    else if(start_index_received > end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 start_index_received,
                                 end_index_received,
                                 __LINE__);
        
        return;
    }

    struct AF_unit *tmp_ptr_destination_AF_it(this->ptr_array_AF_units + start_index_received);

    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        int const tmp_total_units__int(static_cast<int>(end_index_received - start_index_received));
        int tmp_unit_index__int;
        
        #pragma omp parallel for schedule(static)
        for(tmp_unit_index__int = 0; tmp_unit_index__int < tmp_total_units__int; ++tmp_unit_index__int)
        { this->Copy__AF_Unit(ptr_array_source_AF_units_received + tmp_unit_index__int, tmp_ptr_destination_AF_it + tmp_unit_index__int); }
    }
    else
    {
        struct AF_unit const *const tmp_ptr_destination_last_af(tmp_ptr_destination_AF_it + (end_index_received - start_index_received));

        for(; tmp_ptr_destination_AF_it != tmp_ptr_destination_last_af; ++tmp_ptr_destination_AF_it,
                                                                                                 ++ptr_array_source_AF_units_received)
        { this->Copy__AF_Unit(ptr_array_source_AF_units_received, tmp_ptr_destination_AF_it); }
    }
}

void Neural_Network::Copy__AF_Unit(struct AF_unit const *const ptr_source_AF_unit_received, struct AF_unit *const ptr_destination_AF_unit_received)
{
    *ptr_destination_AF_unit_received->ptr_activation_steepness = *ptr_source_AF_unit_received->ptr_activation_steepness;

    *ptr_destination_AF_unit_received->ptr_type_activation_function = *ptr_source_AF_unit_received->ptr_type_activation_function;
}

void Neural_Network::Copy__AF_Ind_Recurrent_Units(size_t const start_index_received,
                                                                               size_t const end_index_received,
                                                                               struct AF_Ind_recurrent_unit const *ptr_array_source_AF_Ind_recurrent_units_received,
                                                                               bool const copy_connections_received)
{
    if(start_index_received + end_index_received == 0_zu) { return; }
    else if(start_index_received > end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 start_index_received,
                                 end_index_received,
                                 __LINE__);
        
        return;
    }

    struct AF_Ind_recurrent_unit *tmp_ptr_destination_AF_Ind_it(this->ptr_array_AF_Ind_recurrent_units + start_index_received);

    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        int const tmp_total_units__int(static_cast<int>(end_index_received - start_index_received));
        int tmp_unit_index__int;
        
        #pragma omp parallel for schedule(static)
        for(tmp_unit_index__int = 0; tmp_unit_index__int < tmp_total_units__int; ++tmp_unit_index__int)
        {
            this->Copy__AF_Ind_Recurrent_Unit(ptr_array_source_AF_Ind_recurrent_units_received + tmp_unit_index__int,
                                                                  tmp_ptr_destination_AF_Ind_it + tmp_unit_index__int,
                                                                  copy_connections_received);
        }
    }
    else
    {
        struct AF_Ind_recurrent_unit const *const tmp_ptr_destination_last_AF_ind(tmp_ptr_destination_AF_Ind_it + (end_index_received - start_index_received));

        for(; tmp_ptr_destination_AF_Ind_it != tmp_ptr_destination_last_AF_ind; ++tmp_ptr_destination_AF_Ind_it,
                                                                                                                 ++ptr_array_source_AF_Ind_recurrent_units_received)
        {
            this->Copy__AF_Ind_Recurrent_Unit(ptr_array_source_AF_Ind_recurrent_units_received,
                                                                  tmp_ptr_destination_AF_Ind_it,
                                                                  copy_connections_received);
        }
    }
}

void Neural_Network::Copy__AF_Ind_Recurrent_Unit(struct AF_Ind_recurrent_unit const *const ptr_source_AF_Ind_recurrent_unit_received,
                                                                             struct AF_Ind_recurrent_unit *const ptr_destination_AF_Ind_recurrent_unit_received,
                                                                             bool const copy_connections_received)
{
    if(copy_connections_received) { *ptr_destination_AF_Ind_recurrent_unit_received->ptr_recurrent_connection_index = *ptr_source_AF_Ind_recurrent_unit_received->ptr_recurrent_connection_index; }
    
    *ptr_destination_AF_Ind_recurrent_unit_received->ptr_activation_steepness = *ptr_source_AF_Ind_recurrent_unit_received->ptr_activation_steepness;

    *ptr_destination_AF_Ind_recurrent_unit_received->ptr_type_activation_function = *ptr_source_AF_Ind_recurrent_unit_received->ptr_type_activation_function;
}

void Neural_Network::Copy__Normalized_Units(size_t const start_index_received,
                                                                     size_t const end_index_received,
                                                                     union Normalized_unit const *ptr_array_source_normalized_units_received)
{
    if(start_index_received + end_index_received == 0_zu) { return; }
    else if(start_index_received > end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 start_index_received,
                                 end_index_received,
                                 __LINE__);
        
        return;
    }

    if(this->Use__Normalization())
    {
        size_t tmp_number_units[2u] = {0};
        
        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
        
        struct Block_unit const *tmp_ptr_last_block_unit;
        struct Block_unit *tmp_ptr_block_unit_it;
        
        struct Cell_unit const *tmp_ptr_last_cell_unit;
        struct Cell_unit *tmp_ptr_cell_unit_it;
        
        union Normalized_unit const *tmp_ptr_destination_last_normalized_unit;
        union Normalized_unit *tmp_ptr_destination_normalized_unit_it(nullptr);
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            if(static_cast<size_t>(tmp_ptr_layer_it->ptr_array_normalized_units - this->ptr_array_normalized_units) < start_index_received) { continue; }
            else if(static_cast<size_t>(tmp_ptr_layer_it->ptr_array_normalized_units - this->ptr_array_normalized_units) >= end_index_received) { break; }

            if((tmp_number_units[0u] = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_normalized_unit - tmp_ptr_layer_it->ptr_array_normalized_units)) != 0_zu)
            {
                switch(tmp_ptr_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                        switch(tmp_ptr_layer_it->type_normalization)
                        {
                            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
                                for(tmp_ptr_destination_last_normalized_unit = this->ptr_array_normalized_units + MyEA::Math::Minimum<size_t>(static_cast<size_t>(tmp_ptr_layer_it->ptr_last_normalized_unit - this->ptr_array_normalized_units), end_index_received),
                                    tmp_ptr_destination_normalized_unit_it = tmp_ptr_layer_it->ptr_array_normalized_units; tmp_ptr_destination_normalized_unit_it < tmp_ptr_destination_last_normalized_unit; ++tmp_ptr_destination_normalized_unit_it,
                                                                                                                                                                                                                                                                                              ++ptr_array_source_normalized_units_received)
                                {
                                    this->Copy__Normalized_Batch_Unit(tmp_number_units[0u],
                                                                                          ptr_array_source_normalized_units_received->normalized_batch_units,
                                                                                          tmp_ptr_destination_normalized_unit_it->normalized_batch_units);
                                }
                                    break;
                            default: ptr_array_source_normalized_units_received += tmp_number_units[0u]; break;
                        }
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                        switch(tmp_ptr_layer_it->type_normalization)
                        {
                            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
                            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
                            case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
                                // Number block unit(s) in layer.
                                tmp_number_units[0u] = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units);
                            
                                // Number cell unit(s) in layer.
                                tmp_number_units[1u] = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units);
                            
                                // Loop through each block unit(s) in the layer.
                                for(tmp_ptr_last_block_unit = tmp_ptr_layer_it->ptr_last_block_unit,
                                    tmp_ptr_block_unit_it = tmp_ptr_layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
                                {
                                    // Loop through each cell unit(s) in the block.
                                    for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                                        tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                                    {
                                        // Loop through each normalized unit(s) in the cell.
                                        for(tmp_ptr_destination_last_normalized_unit = this->ptr_array_normalized_units + MyEA::Math::Minimum<size_t>(static_cast<size_t>(tmp_ptr_cell_unit_it->ptr_last_normalized_unit - this->ptr_array_normalized_units), end_index_received),
                                            tmp_ptr_destination_normalized_unit_it = tmp_ptr_cell_unit_it->ptr_array_normalized_units; tmp_ptr_destination_normalized_unit_it < tmp_ptr_destination_last_normalized_unit; ++tmp_ptr_destination_normalized_unit_it,
                                                                                                                                                                                                                                                                                                           ++ptr_array_source_normalized_units_received)
                                        {
                                            this->Copy__Normalized_Batch_Unit(tmp_number_units[1u],
                                                                                                  ptr_array_source_normalized_units_received->normalized_batch_units,
                                                                                                  tmp_ptr_destination_normalized_unit_it->normalized_batch_units);
                                        }
                                    }
                                    
                                    // Loop through each normalized unit(s) in the block.
                                    for(tmp_ptr_destination_last_normalized_unit = this->ptr_array_normalized_units + MyEA::Math::Minimum<size_t>(static_cast<size_t>(tmp_ptr_block_unit_it->ptr_last_normalized_unit - this->ptr_array_normalized_units), end_index_received),
                                        tmp_ptr_destination_normalized_unit_it = tmp_ptr_block_unit_it->ptr_array_normalized_units; tmp_ptr_destination_normalized_unit_it < tmp_ptr_destination_last_normalized_unit; ++tmp_ptr_destination_normalized_unit_it,
                                                                                                                                                                                                                                                                                                          ++ptr_array_source_normalized_units_received)
                                    {
                                        this->Copy__Normalized_Batch_Unit(tmp_number_units[0u],
                                                                                              ptr_array_source_normalized_units_received->normalized_batch_units,
                                                                                              tmp_ptr_destination_normalized_unit_it->normalized_batch_units);
                                    }
                                }
                                    break;
                            default: ptr_array_source_normalized_units_received += 6_zu * static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units) + 3_zu * static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units); break;
                        }
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                            break;
                }
            }
        }
    }
}

void Neural_Network::Copy__Normalized_Batch_Unit(size_t const number_units_received,
                                                                             struct Normalized_batch_unit const &ref_source_normalized_batch_unit_received,
                                                                             struct Normalized_batch_unit &ref_destination_normalized_batch_unit_received)
{
    size_t tmp_time_step_index,
              tmp_unit_timed_index;

    *ref_destination_normalized_batch_unit_received.ptr_scale = *ref_source_normalized_batch_unit_received.ptr_scale;
    *ref_destination_normalized_batch_unit_received.ptr_shift = *ref_source_normalized_batch_unit_received.ptr_shift;

    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        tmp_unit_timed_index = number_units_received * tmp_time_step_index;

        ref_destination_normalized_batch_unit_received.ptr_mean_average[tmp_unit_timed_index] = ref_source_normalized_batch_unit_received.ptr_mean_average[tmp_unit_timed_index];
        ref_destination_normalized_batch_unit_received.ptr_variance_average[tmp_unit_timed_index] = ref_source_normalized_batch_unit_received.ptr_variance_average[tmp_unit_timed_index];
    }
}

template<typename U> void Neural_Network::Copy__Layer__FC(struct Layer const *const ptr_source_layer_received,
                                                                                              struct Layer *const ptr_destination_layer_received,
                                                                                              U *const ptr_source_first_U_received,
                                                                                              U *const ptr_destination_first_U_received,
                                                                                              U *const *ptr_source_array_ptr_connections_received,
                                                                                              U **ptr_destination_array_ptr_connections_received)
{
    struct Neuron_unit const *const tmp_ptr_source_last_neuron_unit(ptr_source_layer_received->ptr_last_neuron_unit),
                                        *tmp_ptr_source_neuron_unit_it(ptr_source_layer_received->ptr_array_neuron_units);
    struct Neuron_unit *tmp_ptr_destination_neuron_unit_it(ptr_destination_layer_received->ptr_array_neuron_units);

    size_t const tmp_number_forward_connections(*tmp_ptr_source_neuron_unit_it->ptr_number_connections);
    size_t tmp_connection_index;

    U *const *tmp_ptr_source_array_ptr_connection_U,
       **tmp_ptr_destination_array_ptr_connection_U;

    for(; tmp_ptr_source_neuron_unit_it != tmp_ptr_source_last_neuron_unit; ++tmp_ptr_source_neuron_unit_it,
                                                                                                              ++tmp_ptr_destination_neuron_unit_it)
    {
        tmp_ptr_source_array_ptr_connection_U = ptr_source_array_ptr_connections_received + *tmp_ptr_source_neuron_unit_it->ptr_first_connection_index;
        tmp_ptr_destination_array_ptr_connection_U = ptr_destination_array_ptr_connections_received + *tmp_ptr_source_neuron_unit_it->ptr_first_connection_index;
        
        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_forward_connections; ++tmp_connection_index)
        { tmp_ptr_destination_array_ptr_connection_U[tmp_connection_index] = ptr_destination_first_U_received + static_cast<size_t>(tmp_ptr_source_array_ptr_connection_U[tmp_connection_index] - ptr_source_first_U_received); }
    }
}

void Neural_Network::Copy__Layer__AF_Ind_Recurrent(struct Layer const *const ptr_source_layer_received,
                                                                          struct AF_Ind_recurrent_unit *const ptr_source_first_AF_Ind_recurrent_unit_received,
                                                                          struct AF_Ind_recurrent_unit *const ptr_destination_first_AF_Ind_recurrent_unit_received,
                                                                          struct AF_Ind_recurrent_unit *const *ptr_source_array_ptr_connections_received,
                                                                          struct AF_Ind_recurrent_unit **ptr_destination_array_ptr_connections_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_source_last_AF_Ind_recurrent_unit(ptr_source_layer_received->ptr_last_AF_Ind_recurrent_unit),
                                        *tmp_ptr_source_AF_Ind_recurrent_unit_it(ptr_source_layer_received->ptr_array_AF_Ind_recurrent_units);

    for(; tmp_ptr_source_AF_Ind_recurrent_unit_it != tmp_ptr_source_last_AF_Ind_recurrent_unit; ++tmp_ptr_source_AF_Ind_recurrent_unit_it)
    { ptr_destination_array_ptr_connections_received[*tmp_ptr_source_AF_Ind_recurrent_unit_it->ptr_recurrent_connection_index] = ptr_destination_first_AF_Ind_recurrent_unit_received + static_cast<size_t>(ptr_source_array_ptr_connections_received[*tmp_ptr_source_AF_Ind_recurrent_unit_it->ptr_recurrent_connection_index] - ptr_source_first_AF_Ind_recurrent_unit_received); }
}

template<typename U> void Neural_Network::Copy__Layer__LSTM(struct Layer const *const ptr_source_layer_received,
                                                                                                  struct Layer *const ptr_destination_layer_received,
                                                                                                  struct Cell_unit *const ptr_source_first_cell_unit_received,
                                                                                                  U *const ptr_source_first_U_received,
                                                                                                  U *const ptr_destination_first_U_received,
                                                                                                  void *const *ptr_source_array_ptr_connections_received,
                                                                                                  void **ptr_destination_array_ptr_connections_received)
{
    struct Block_unit const *const tmp_ptr_source_last_block_unit(ptr_source_layer_received->ptr_last_block_unit),
                                       *tmp_ptr_source_block_unit_it(ptr_source_layer_received->ptr_array_block_units);
    struct Block_unit *tmp_ptr_destination_block_unit_it(ptr_destination_layer_received->ptr_array_block_units);
    
    size_t const tmp_number_inputs_connections(tmp_ptr_source_block_unit_it->last_index_feedforward_connection_input_gate - tmp_ptr_source_block_unit_it->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrents_connection(tmp_ptr_source_block_unit_it->last_index_recurrent_connection_input_gate - tmp_ptr_source_block_unit_it->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index;

    U *const *tmp_ptr_source_cell_input_array_ptr_connection_U,
        *const *tmp_ptr_source_input_gate_array_ptr_connection_U,
        *const *tmp_ptr_source_forget_gate_array_ptr_connection_U,
        *const *tmp_ptr_source_output_gate_array_ptr_connection_U,
        **tmp_ptr_destination_cell_input_array_ptr_connection_U,
        **tmp_ptr_destination_input_gate_array_ptr_connection_U,
        **tmp_ptr_destination_forget_gate_array_ptr_connection_U,
        **tmp_ptr_destination_output_gate_array_ptr_connection_U;
    
    struct Cell_unit const *tmp_ptr_source_block_ptr_last_unit,
                                    *tmp_ptr_source_block_ptr_cell_unit_it;
    struct Cell_unit *const *tmp_ptr_source_array_ptr_connection_cell_units,
                           *const *tmp_ptr_source_cell_input_array_ptr_connection_cell_units,
                           *const *tmp_ptr_source_input_gate_array_ptr_connection_cell_units,
                           *const *tmp_ptr_source_forget_gate_array_ptr_connection_cell_units,
                           *const *tmp_ptr_source_output_gate_array_ptr_connection_cell_units,
                           **tmp_ptr_destination_array_ptr_connection_cell_units,
                           **tmp_ptr_destination_cell_input_array_ptr_connection_cell_units,
                           **tmp_ptr_destination_input_gate_array_ptr_connection_cell_units,
                           **tmp_ptr_destination_forget_gate_array_ptr_connection_cell_units,
                           **tmp_ptr_destination_output_gate_array_ptr_connection_cell_units,
                           *tmp_ptr_destination_block_ptr_cell_unit_it;

    for(; tmp_ptr_source_block_unit_it != tmp_ptr_source_last_block_unit; ++tmp_ptr_source_block_unit_it,
                                                                                                          ++tmp_ptr_destination_block_unit_it)
    {
        this->Copy__Block(tmp_ptr_source_block_unit_it, tmp_ptr_destination_block_unit_it);
        
        // [0] Cell input.
        tmp_ptr_source_block_ptr_last_unit = tmp_ptr_source_block_unit_it->ptr_last_cell_unit;

        for(tmp_ptr_destination_block_ptr_cell_unit_it = tmp_ptr_destination_block_unit_it->ptr_array_cell_units,
            tmp_ptr_source_block_ptr_cell_unit_it = tmp_ptr_source_block_unit_it->ptr_array_cell_units; tmp_ptr_source_block_ptr_cell_unit_it != tmp_ptr_source_block_ptr_last_unit; ++tmp_ptr_source_block_ptr_cell_unit_it,
                                                                                                                                                                                                                                                                    ++tmp_ptr_destination_block_ptr_cell_unit_it)
        {
            //    [1] Input, cell input.
            tmp_ptr_source_cell_input_array_ptr_connection_U = reinterpret_cast<U *const *>(ptr_source_array_ptr_connections_received + tmp_ptr_source_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input);
            
            tmp_ptr_destination_cell_input_array_ptr_connection_U = reinterpret_cast<U **>(ptr_destination_array_ptr_connections_received + tmp_ptr_destination_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input);
            
            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
            { tmp_ptr_destination_cell_input_array_ptr_connection_U[tmp_connection_index] = ptr_destination_first_U_received + static_cast<size_t>(tmp_ptr_source_cell_input_array_ptr_connection_U[tmp_connection_index] - ptr_source_first_U_received); }
            //    [1] |END| Input, cell input. |END|

            //    [1] Recurrent, cell input.
            tmp_ptr_source_cell_input_array_ptr_connection_cell_units = reinterpret_cast<struct Cell_unit *const *>(ptr_source_array_ptr_connections_received + tmp_ptr_source_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input);
            
            tmp_ptr_destination_cell_input_array_ptr_connection_cell_units = reinterpret_cast<struct Cell_unit **>(ptr_destination_array_ptr_connections_received + tmp_ptr_destination_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input);
            
            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
            { tmp_ptr_destination_cell_input_array_ptr_connection_cell_units[tmp_connection_index] = this->ptr_array_cell_units + static_cast<size_t>(tmp_ptr_source_cell_input_array_ptr_connection_cell_units[tmp_connection_index] - ptr_source_first_cell_unit_received); }
            //    [1] |END| Recurrent, cell input. |END|
        }
        // [0] |END| Cell input. |END|
        
        // [0] Input, gates.
        tmp_ptr_source_input_gate_array_ptr_connection_U = reinterpret_cast<U *const *>(ptr_source_array_ptr_connections_received + tmp_ptr_source_block_unit_it->first_index_feedforward_connection_input_gate);
        tmp_ptr_source_forget_gate_array_ptr_connection_U = reinterpret_cast<U *const *>(ptr_source_array_ptr_connections_received + tmp_ptr_source_block_unit_it->first_index_feedforward_connection_forget_gate);
        tmp_ptr_source_output_gate_array_ptr_connection_U = reinterpret_cast<U *const *>(ptr_source_array_ptr_connections_received + tmp_ptr_source_block_unit_it->first_index_feedforward_connection_output_gate);
        
        tmp_ptr_destination_input_gate_array_ptr_connection_U = reinterpret_cast<U **>(ptr_destination_array_ptr_connections_received + tmp_ptr_destination_block_unit_it->first_index_feedforward_connection_input_gate);
        tmp_ptr_destination_forget_gate_array_ptr_connection_U = reinterpret_cast<U **>(ptr_destination_array_ptr_connections_received + tmp_ptr_destination_block_unit_it->first_index_feedforward_connection_forget_gate);
        tmp_ptr_destination_output_gate_array_ptr_connection_U = reinterpret_cast<U **>(ptr_destination_array_ptr_connections_received + tmp_ptr_destination_block_unit_it->first_index_feedforward_connection_output_gate);
        
        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
        {
            tmp_ptr_destination_input_gate_array_ptr_connection_U[tmp_connection_index] = ptr_destination_first_U_received + static_cast<size_t>(tmp_ptr_source_input_gate_array_ptr_connection_U[tmp_connection_index] - ptr_source_first_U_received);
            tmp_ptr_destination_forget_gate_array_ptr_connection_U[tmp_connection_index] = ptr_destination_first_U_received + static_cast<size_t>(tmp_ptr_source_forget_gate_array_ptr_connection_U[tmp_connection_index] - ptr_source_first_U_received);
            tmp_ptr_destination_output_gate_array_ptr_connection_U[tmp_connection_index] = ptr_destination_first_U_received + static_cast<size_t>(tmp_ptr_source_output_gate_array_ptr_connection_U[tmp_connection_index] - ptr_source_first_U_received);
        }
        // [0] |END| Input, gates. |END|
        
        // [0] Recurrent, gates.
        tmp_ptr_source_input_gate_array_ptr_connection_cell_units = reinterpret_cast<struct Cell_unit *const *>(ptr_source_array_ptr_connections_received + tmp_ptr_source_block_unit_it->first_index_recurrent_connection_input_gate);
        tmp_ptr_source_forget_gate_array_ptr_connection_cell_units = reinterpret_cast<struct Cell_unit *const *>(ptr_source_array_ptr_connections_received + tmp_ptr_source_block_unit_it->first_index_recurrent_connection_forget_gate);
        tmp_ptr_source_output_gate_array_ptr_connection_cell_units = reinterpret_cast<struct Cell_unit *const *>(ptr_source_array_ptr_connections_received + tmp_ptr_source_block_unit_it->first_index_recurrent_connection_output_gate);
        
        tmp_ptr_destination_input_gate_array_ptr_connection_cell_units = reinterpret_cast<struct Cell_unit **>(ptr_destination_array_ptr_connections_received + tmp_ptr_destination_block_unit_it->first_index_recurrent_connection_input_gate);
        tmp_ptr_destination_forget_gate_array_ptr_connection_cell_units = reinterpret_cast<struct Cell_unit **>(ptr_destination_array_ptr_connections_received + tmp_ptr_destination_block_unit_it->first_index_recurrent_connection_forget_gate);
        tmp_ptr_destination_output_gate_array_ptr_connection_cell_units = reinterpret_cast<struct Cell_unit **>(ptr_destination_array_ptr_connections_received + tmp_ptr_destination_block_unit_it->first_index_recurrent_connection_output_gate);
        
        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
        {
            tmp_ptr_destination_input_gate_array_ptr_connection_cell_units[tmp_connection_index] = this->ptr_array_cell_units + static_cast<size_t>(tmp_ptr_source_input_gate_array_ptr_connection_cell_units[tmp_connection_index] - ptr_source_first_cell_unit_received);
            tmp_ptr_destination_forget_gate_array_ptr_connection_cell_units[tmp_connection_index] = this->ptr_array_cell_units + static_cast<size_t>(tmp_ptr_source_forget_gate_array_ptr_connection_cell_units[tmp_connection_index] - ptr_source_first_cell_unit_received);
            tmp_ptr_destination_output_gate_array_ptr_connection_cell_units[tmp_connection_index] = this->ptr_array_cell_units + static_cast<size_t>(tmp_ptr_source_output_gate_array_ptr_connection_cell_units[tmp_connection_index] - ptr_source_first_cell_unit_received);
        }
        // [0] |END| Recurrent, gates. |END|

    #ifndef NO_PEEPHOLE
        //    [1] Peepholes.
        tmp_ptr_source_array_ptr_connection_cell_units = reinterpret_cast<struct Cell_unit *const *>(ptr_source_array_ptr_connections_received);
        
        tmp_ptr_destination_array_ptr_connection_cell_units = reinterpret_cast<struct Cell_unit **>(ptr_destination_array_ptr_connections_received);
        
        for(tmp_ptr_destination_block_ptr_cell_unit_it = tmp_ptr_destination_block_unit_it->ptr_array_cell_units,
            tmp_ptr_source_block_ptr_cell_unit_it = tmp_ptr_source_block_unit_it->ptr_array_cell_units; tmp_ptr_source_block_ptr_cell_unit_it != tmp_ptr_source_block_ptr_last_unit; ++tmp_ptr_source_block_ptr_cell_unit_it,
                                                                                                                                                                                                                                                                    ++tmp_ptr_destination_block_ptr_cell_unit_it)
        {
            tmp_ptr_destination_array_ptr_connection_cell_units[tmp_ptr_destination_block_ptr_cell_unit_it->index_peephole_input_gate] = this->ptr_array_cell_units + static_cast<size_t>(tmp_ptr_source_array_ptr_connection_cell_units[tmp_ptr_source_block_ptr_cell_unit_it->index_peephole_input_gate] - ptr_source_first_cell_unit_received);
            tmp_ptr_destination_array_ptr_connection_cell_units[tmp_ptr_destination_block_ptr_cell_unit_it->index_peephole_forget_gate] = this->ptr_array_cell_units + static_cast<size_t>(tmp_ptr_source_array_ptr_connection_cell_units[tmp_ptr_source_block_ptr_cell_unit_it->index_peephole_forget_gate] - ptr_source_first_cell_unit_received);
            tmp_ptr_destination_array_ptr_connection_cell_units[tmp_ptr_destination_block_ptr_cell_unit_it->index_peephole_output_gate] = this->ptr_array_cell_units + static_cast<size_t>(tmp_ptr_source_array_ptr_connection_cell_units[tmp_ptr_source_block_ptr_cell_unit_it->index_peephole_output_gate] - ptr_source_first_cell_unit_received);
        }
        //    [1] |END| Peepholes. |END|
    #endif
    }
}
