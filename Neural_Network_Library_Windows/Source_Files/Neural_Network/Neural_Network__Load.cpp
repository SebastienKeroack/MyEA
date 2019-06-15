#include "stdafx.hpp"

#include <Files/File.hpp>
#include <Strings/String.hpp>

#include <fstream>
#include <iostream>

#if defined(COMPILE_ADEPT)
bool Neural_Network::Load_Parameters(std::string const &ref_path_received) { return(true); }

bool Neural_Network::Load(std::string const &ref_path_dimension_received,
                                        std::string const &ref_path_parameters_received,
                                        size_t const maximum_allowable_memory_received) { return(true); }

bool Neural_Network::Load_Dimension__Neuron(struct Neuron_unit *const ptr_neuron_received, std::ifstream &ref_ifstream_received) { return(true); }

bool Neural_Network::Load_Dimension__AF(struct AF_unit *const ptr_AF_received, std::ifstream &ref_ifstream_received) { return(true); }

bool Neural_Network::Load_Dimension__AF_Ind_Recurrent(struct AF_Ind_recurrent_unit *const ptr_AF_received, std::ifstream &ref_ifstream_received) { return(true); }

bool Neural_Network::Load_Dimension__Normalized_Unit(size_t const number_units_received,
                                                                                    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_normalization_received,
                                                                                    union Normalized_unit *const ptr_normalized_unit_received,
                                                                                    std::ifstream &ref_ifstream_received) { return(true); }

bool Neural_Network::Load_Dimension__Bias(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received) { return(true); }

bool Neural_Network::Load_Dimension__Block(size_t const layer_number_block_units_received,
                                                                     size_t const layer_number_cell_units_received,
                                                                     enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_normalization_received,
                                                                     struct Block_unit *const ptr_block_unit_it_received,
                                                                     std::ifstream &ref_ifstream_received) { return(true); }

bool Neural_Network::Load_Dimension__Cell_Units(struct Layer *const ptr_layer_it_received,
                                                                           struct Cell_unit *&ptr_reference_array_cells_received,
                                                                           std::ifstream &ref_ifstream_received) { return(true); }

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> inline bool Neural_Network::Load_Dimension__Connection(size_t index_received,
                                                                                                                                                                                                            T_ *const ptr_array_parameters_received,
                                                                                                                                                                                                            U *const ptr_first_U_unit_received,
                                                                                                                                                                                                            U **ptr_array_ptr_U_unit_connection_received,
                                                                                                                                                                                                            std::ifstream &ref_ifstream_received) { return(true); }

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Neural_Network::Load_Dimension__Neuron__Forward__Connection(struct Neuron_unit *const ptr_neuron_received,
                                                                                                                                                                                                                                U *const ptr_first_U_unit_received,
                                                                                                                                                                                                                                std::ifstream &ref_ifstream_received) { return(true); }

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Neural_Network::Load_Dimension__Block__Connection(struct Block_unit *const ptr_block_unit_it_received,
                                                                                                                                                                                                               U *const ptr_first_U_unit_received,
                                                                                                                                                                                                               std::ifstream &ref_ifstream_received) { return(true); }

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Neural_Network::Load_Dimension__FC(struct Layer *const ptr_layer_it_received,
                                                                                                                                                                                       U *const ptr_first_U_unit_received,
                                                                                                                                                                                       std::ifstream &ref_ifstream_received) { return(true); }

bool Neural_Network::Load_Dimension__AF(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received) { return(true); }

bool Neural_Network::Load_Dimension__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received) { return(true); }

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Neural_Network::Load_Dimension__LSTM(struct Layer *const ptr_layer_it_received,
                                                                                                                                                                                           U *const ptr_first_U_unit_received,
                                                                                                                                                                                           std::ifstream &ref_ifstream_received) { return(true); }

bool Neural_Network::Load_Dimension__Normalization(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received) { return(true); }
#else
bool Neural_Network::Load_Parameters(std::string const &ref_path_received)
{
    if(MyEA::File::Path_Exist(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: Could not find the following path \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(MyEA::File::Retrieve_Tempory_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Retrieve_Tempory_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    std::ifstream tmp_ifstream(ref_path_received, std::ios::in | std::ios::binary);

    if(tmp_ifstream.is_open())
    {
        if(tmp_ifstream.eof())
        {
            PRINT_FORMAT("%s: %s: ERROR: File \"%s\" is empty. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }

        size_t tmp_input_integer;

        T_ tmp_input_real;

        std::string tmp_line;

        getline(tmp_ifstream, tmp_line); // "|===| GRADIENT DESCENT PARAMETERS |===|"

        if((tmp_ifstream >> tmp_line) && tmp_line.find("learning_rate") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"learning_rate\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->learning_rate >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("learning_rate_final") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"learning_rate_final\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->learning_rate_final >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("learning_momentum") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"learning_momentum\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->learning_momentum >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("learning_gamma") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"learning_gamma\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->learning_gamma >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("use_Nesterov") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_Nesterov\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->use_Nesterov >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| GRADIENT DESCENT PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE

        getline(tmp_ifstream, tmp_line); // "|===| QUICKPROP PARAMETERS |===|"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("quickprop_decay") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"quickprop_decay\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->quickprop_decay >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("quickprop_mu") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"quickprop_mu\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->quickprop_mu >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| QUICKPROP PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE

        getline(tmp_ifstream, tmp_line); // "|===| RESILLENT PROPAGATION PARAMETERS |===|"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("rprop_increase_factor") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"rprop_increase_factor\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->rprop_increase_factor >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("rprop_decrease_factor") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"rprop_decrease_factor\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->rprop_decrease_factor >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("rprop_delta_min") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"rprop_delta_min\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->rprop_delta_min >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("rprop_delta_max") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"rprop_delta_max\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->rprop_delta_max >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("rprop_delta_zero") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"rprop_delta_zero\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->rprop_delta_zero >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| RESILLENT PROPAGATION PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE

        getline(tmp_ifstream, tmp_line); // "|===| SARPROP PARAMETERS |===|"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("sarprop_weight_decay_shift") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"sarprop_weight_decay_shift\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->sarprop_weight_decay_shift >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("sarprop_step_error_threshold_factor") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"sarprop_step_error_threshold_factor\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->sarprop_step_error_threshold_factor >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("sarprop_step_error_shift") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"sarprop_step_error_shift\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->sarprop_step_error_shift >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("sarprop_temperature") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"sarprop_temperature\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->sarprop_temperature >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("sarprop_epoch") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"sarprop_epoch\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->sarprop_epoch >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| SARPROP PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE

        getline(tmp_ifstream, tmp_line); // "|===| ADAM PARAMETERS |===|"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("adam_learning_rate") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"adam_learning_rate\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->adam_learning_rate >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("adam_beta1") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"adam_beta1\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->adam_beta1 >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("adam_beta2") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"adam_beta2\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->adam_beta2 >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("adam_epsilon") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"adam_epsilon\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->adam_epsilon >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("adam_bias_correction") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"adam_bias_correction\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->use_adam_bias_correction >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("adam_gamma") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"adam_gamma\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->adam_gamma >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| ADAM PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE

        getline(tmp_ifstream, tmp_line); // "|===| WARM RESTARTS PARAMETERS |===|"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("use_Warm_Restarts") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_Warm_Restarts\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->use_Warm_Restarts >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("warm_restarts_decay_learning_rate") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"warm_restarts_decay_learning_rate\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->warm_restarts_decay_learning_rate >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("warm_restarts_maximum_learning_rate") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"warm_restarts_maximum_learning_rate\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->warm_restarts_initial_maximum_learning_rate >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("warm_restarts_minimum_learning_rate") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"warm_restarts_minimum_learning_rate\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->warm_restarts_minimum_learning_rate >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("warm_restarts_initial_T_i") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"warm_restarts_initial_T_i\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->warm_restarts_initial_T_i >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("warm_restarts_multiplier") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"warm_restarts_multiplier\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->warm_restarts_multiplier >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| WARM RESTARTS PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE

        getline(tmp_ifstream, tmp_line); // "|===| TRAINING PARAMETERS |===|"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("type_optimizer_function") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_optimizer_function\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_LENGTH))
            {
                PRINT_FORMAT("%s: %s: ERROR: Undefined optimization type %zu. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_input_integer,
                                         __LINE__);

                return(false);
            }

            this->Set__Optimizer_Function(static_cast<enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS>(tmp_input_integer));
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("type_loss_function") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_loss_function\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_LENGTH))
            {
                PRINT_FORMAT("%s: %s: ERROR: Undefined loss function type %zu. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_input_integer,
                                         __LINE__);

                return(false);
            }
            
            this->Set__Loss_Function(static_cast<enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS>(tmp_input_integer));
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("type_accuracy_function") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_accuracy_function\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_LENGTH))
            {
                PRINT_FORMAT("%s: %s: ERROR: Undefined loss function type %zu. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_input_integer,
                                         __LINE__);

                return(false);
            }
            
            this->Set__Accuracy_Function(static_cast<enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS>(tmp_input_integer));
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("bit_fail_limit") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"bit_fail_limit\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Bit_Fail_Limit(tmp_input_real);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("pre_training_level") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"pre_training_level\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->pre_training_level >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("use_clip_gradient") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_clip_gradient\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->use_clip_gradient >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("clip_gradient") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"clip_gradient\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->clip_gradient >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| TRAINING PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE

        getline(tmp_ifstream, tmp_line); // "|===| REGULARIZATION PARAMETERS |===|"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("regularization__max_norm_constraints") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"regularization__max_norm_constraints\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Regularization__Max_Norm_Constraints(tmp_input_real);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("regularization__l1") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"regularization__l1\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Regularization__L1(tmp_input_real);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("regularization__l2") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"regularization__l2\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Regularization__L2(tmp_input_real);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("regularization__srip") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"regularization__srip\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Regularization__SRIP(tmp_input_real);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("regularization__weight_decay") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"regularization__weight_decay\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Regularization__Weight_Decay(tmp_input_real);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("use_normalized_weight_decay") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_normalized_weight_decay\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->use_normalized_weight_decay >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| REGULARIZATION PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE
        
        getline(tmp_ifstream, tmp_line); // "|===| NORMALIZATION PARAMETERS |===|"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("normalization_momentum_average") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"normalization_momentum_average\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Normalization_Momentum_Average(tmp_input_real);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("normalization_epsilon") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"normalization_epsilon\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Normalization_Epsilon(tmp_input_real);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("batch_renormalization_r_correction_maximum") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"batch_renormalization_r_correction_maximum\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Batch_Renormalization_r_Correction_Maximum(tmp_input_real);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("batch_renormalization_d_correction_maximum") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"batch_renormalization_d_correction_maximum\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Batch_Renormalization_d_Correction_Maximum(tmp_input_real);
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| NORMALIZATION PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE

        getline(tmp_ifstream, tmp_line); // "|===| LOSS PARAMETERS |===|"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("loss_training") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"loss_training\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->loss_training >> std::ws;

            if(tmp_ifstream.fail())
            {
                tmp_ifstream.clear();
                
                tmp_ifstream >> std::ws;

                this->loss_training = (std::numeric_limits<ST_>::max)();
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("loss_validating") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"loss_validating\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->loss_validating >> std::ws;

            if(tmp_ifstream.fail())
            {
                tmp_ifstream.clear();
                
                tmp_ifstream >> std::ws;

                this->loss_validating = (std::numeric_limits<ST_>::max)();
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("loss_testing") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"loss_testing\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->loss_testing >> std::ws;

            if(tmp_ifstream.fail())
            {
                tmp_ifstream.clear();
                
                tmp_ifstream >> std::ws;

                this->loss_testing = (std::numeric_limits<ST_>::max)();
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| LOSS PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE

        getline(tmp_ifstream, tmp_line); // "|===| ACCURANCY PARAMETERS |===|"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("accuracy_variance") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"accuracy_variance\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_real >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            this->Set__Accurancy_Variance(tmp_input_real);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("accuracy_training") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"accuracy_training\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->accuracy_training >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("accuracy_validating") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"accuracy_validating\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->accuracy_validating >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("accuracy_testing") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"accuracy_testing\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->accuracy_testing >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| ACCURANCY PARAMETERS |END|"
        getline(tmp_ifstream, tmp_line); // NEW_LINE

        getline(tmp_ifstream, tmp_line); // "|===| COMPUTATION PARAMETERS |===|"
        
    #if defined(COMPILE_CUDA)
        if((tmp_ifstream >> tmp_line) && tmp_line.find("use_CUDA") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_CUDA\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->use_CUDA >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
    #else
        getline(tmp_ifstream, tmp_line); // use_CUDA 0
    #endif
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("use_OpenMP") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_OpenMP\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->use_OpenMP >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("percentage_maximum_thread_usage") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"percentage_maximum_thread_usage\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->percentage_maximum_thread_usage >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("maximum_batch_size") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"maximum_batch_size\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> this->maximum_batch_size >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
        }
        
        getline(tmp_ifstream, tmp_line); // "|END| COMPUTATION PARAMETERS |END|"

        if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Logical error on i/o operation \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }

        tmp_ifstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::Load(std::string const &ref_path_dimension_received,
                                        std::string const &ref_path_parameters_received,
                                        size_t const maximum_allowable_memory_received)
{
    if(MyEA::File::Path_Exist(ref_path_parameters_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: Could not find the following path \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 ref_path_parameters_received.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(MyEA::File::Path_Exist(ref_path_dimension_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: Could not find the following path \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 ref_path_dimension_received.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(MyEA::File::Retrieve_Tempory_File(ref_path_dimension_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Retrieve_Tempory_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 ref_path_dimension_received.c_str(),
                                 __LINE__);

        return(false);
    }

    std::ifstream tmp_ifstream(ref_path_dimension_received, std::ios::in | std::ios::binary);

    if(tmp_ifstream.is_open())
    {
        if(tmp_ifstream.eof())
        {
            PRINT_FORMAT("%s: %s: ERROR: File \"%s\" is empty. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ref_path_dimension_received.c_str(),
                                     __LINE__);

            tmp_ifstream.close();

            return(false);
        }
        
        this->Deallocate();

        bool tmp_input_boolean;

        size_t tmp_state_layer_index(0_zu),
                  tmp_input_integer;
        
        T_ tmp_input_T[2u] = {0};

        std::string tmp_line;
        
        auto tmp_Load__Dropout__Parameters([self = this, &tmp_ifstream = tmp_ifstream](struct Layer *const ptr_layer_it_received, bool const is_hidden_layer_received = true) -> bool
        {
            size_t tmp_input_integer;

            T_ tmp_dropout_values[3u] = {0};

            enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT tmp_type_layer_dropout;
            
            std::string tmp_line;
            
            if((tmp_ifstream >> tmp_line) && tmp_line.find("type_dropout") == std::string::npos)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_dropout\" inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
            else if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
            else
            {
                tmp_ifstream >> tmp_input_integer;

                if(tmp_ifstream.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_line.c_str(),
                                                __LINE__);

                    return(false);
                }

                if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_LENGTH))
                {
                    PRINT_FORMAT("%s: %s: ERROR: Undefined layer dropout type %zu. At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_input_integer,
                                                __LINE__);

                    return(false);
                }

                tmp_type_layer_dropout = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT>(tmp_input_integer);
            }
            
            if(is_hidden_layer_received)
            {
                if((tmp_ifstream >> tmp_line) && tmp_line.find("use_coded_dropout") == std::string::npos)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_coded_dropout\" inside \"%s\". At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_line.c_str(),
                                                __LINE__);

                    return(false);
                }
                else if(tmp_ifstream.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_line.c_str(),
                                                __LINE__);
            
                    return(false);
                }
                else
                {
                    tmp_ifstream >> ptr_layer_it_received->use_coded_dropout >> std::ws;

                    if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                    MyEA::String::Get__Time().c_str(),
                                                    __FUNCTION__,
                                                    tmp_line.c_str(),
                                                    __LINE__);
                
                        return(false);
                    }
                }
            }

            if((tmp_ifstream >> tmp_line) && tmp_line.find("dropout_values[0]") == std::string::npos)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not find \"dropout_values[0]\" inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
            else if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);
            
                return(false);
            }
            else
            {
                tmp_ifstream >> tmp_dropout_values[0u] >> std::ws;

                if(tmp_ifstream.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_line.c_str(),
                                                __LINE__);
                
                    return(false);
                }
            }
            
            if((tmp_ifstream >> tmp_line) && tmp_line.find("dropout_values[1]") == std::string::npos)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not find \"dropout_values[1]\" inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
            else if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);
            
                return(false);
            }
            else
            {
                tmp_ifstream >> tmp_dropout_values[1u] >> std::ws;

                if(tmp_ifstream.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_line.c_str(),
                                                __LINE__);
                
                    return(false);
                }
            }
            
            if((tmp_ifstream >> tmp_line) && tmp_line.find("dropout_values[2]") == std::string::npos)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not find \"dropout_values[2]\" inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
            else if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);
            
                return(false);
            }
            else
            {
                tmp_ifstream >> tmp_dropout_values[2u] >> std::ws;

                if(tmp_ifstream.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_line.c_str(),
                                                __LINE__);
                
                    return(false);
                }
            }
            
            if(self->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
              &&
               (ptr_layer_it_received == self->ptr_last_layer - (self->total_layers - 3_zu) / 2_zu + 2_zu
                    ||
                ptr_layer_it_received >= self->ptr_last_layer - (self->total_layers - 3_zu) / 2_zu + 1_zu))
            { return(true); }

            if(self->Set__Dropout(ptr_layer_it_received,
                                           tmp_type_layer_dropout,
                                           tmp_dropout_values,
                                           false) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(ptr, %u, %f, %f)\" function. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_type_layer_dropout,
                                            tmp_dropout_values[0u],
                                            tmp_dropout_values[1u],
                                            __LINE__);

                return(false);
            }

            return(true);
        });
        
        auto tmp_Valid__Layer__Normalization([self = this](struct Layer *const ptr_layer_it_received) -> bool
        {
            if(self->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
              &&
              ptr_layer_it_received >= self->ptr_last_layer - (self->total_layers - 3_zu) / 2_zu + 1_zu)
            { return(false); }

            return(true);
        });

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Load dimension %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_path_dimension_received.c_str());

        getline(tmp_ifstream, tmp_line); // "|===| DIMENSION |===|"

        if((tmp_ifstream >> tmp_line) && tmp_line.find("type_network") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_network\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_LENGTH))
            {
                PRINT_FORMAT("%s: %s: ERROR: Undefined network type %zu. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_input_integer,
                                         __LINE__);

                return(false);
            }

            this->type_network = static_cast<enum MyEA::Common::ENUM_TYPE_NETWORKS>(tmp_input_integer);
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("number_layers") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_layers\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
            else if(tmp_input_integer < 2)
            {
                PRINT_FORMAT("%s: %s: ERROR: The number of layers is set too small. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }
        
        // Allocate structure.
        PRINT_FORMAT("%s: Allocate %zu layer(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_input_integer);
        if(this->Allocate__Structure(tmp_input_integer, maximum_allowable_memory_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Structure(%zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_input_integer,
                                     maximum_allowable_memory_received,
                                     __LINE__);

            return(false);
        }

        if((tmp_ifstream >> tmp_line) && tmp_line.find("number_recurrent_depth") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_recurrent_depth\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->number_recurrent_depth >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("number_time_delays") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_time_delays\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->number_time_delays >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("use_first_layer_as_input") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_first_layer_as_input\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_boolean >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }

            if(this->Set__Input_Mode(tmp_input_boolean) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Input_Mode(%s)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_input_boolean ? "true" : "false",
                                         __LINE__);

                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("use_last_layer_as_output") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_last_layer_as_output\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_boolean >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }

            if(this->Set__Output_Mode(tmp_input_boolean) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Output_Mode(%s)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_input_boolean ? "true" : "false",
                                         __LINE__);

                return(false);
            }
        }
        
        struct Layer const *const tmp_ptr_first_layer(this->ptr_array_layers),
                                   *const tmp_ptr_last_layer(this->ptr_last_layer - 1), // Subtract output layer.
                                   *tmp_ptr_previous_layer,
                                   *tmp_ptr_layer_state(nullptr);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
        // |END| Allocate structure. |END|
        
        // Allocate basic unit(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_basic_units") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_basic_units\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_basic_units >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        tmp_ptr_layer_it->ptr_array_basic_units = nullptr;
        tmp_ptr_layer_it->ptr_last_basic_unit = tmp_ptr_layer_it->ptr_array_basic_units + this->total_basic_units;
        
        PRINT_FORMAT("%s: Allocate %zu basic unit(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->total_basic_units);
        if(this->Allocate__Basic_Units() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Basic_Units()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        // |END| Allocate basic unit(s). |END|
        
        // Allocate basic indice unit(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_basic_indice_units") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_basic_indice_units\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_basic_indice_units >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        tmp_ptr_layer_it->ptr_array_basic_indice_units = nullptr;
        tmp_ptr_layer_it->ptr_last_basic_indice_unit = tmp_ptr_layer_it->ptr_array_basic_indice_units + this->total_basic_indice_units;
        
        PRINT_FORMAT("%s: Allocate %zu basic indice unit(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->total_basic_indice_units);
        if(this->Allocate__Basic_Indice_Units() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Basic_Indice_Units()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        // |END| Allocate basic indice unit(s). |END|

        // Allocate neuron unit(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_neuron_units") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_neuron_units\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_neuron_units >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        tmp_ptr_layer_it->ptr_array_neuron_units = nullptr;
        tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_array_neuron_units + this->total_neuron_units;
        
        PRINT_FORMAT("%s: Allocate %zu neuron unit(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->total_neuron_units);
        if(this->Allocate__Neuron_Units() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Neuron_Units()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        // |END| Allocate neuron unit(s). |END|
        
        // Allocate AF unit(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_AF_units") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_AF_units\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_AF_units >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        tmp_ptr_layer_it->ptr_array_AF_units = nullptr;
        tmp_ptr_layer_it->ptr_last_AF_unit = tmp_ptr_layer_it->ptr_array_AF_units + this->total_AF_units;
        
        PRINT_FORMAT("%s: Allocate %zu AF unit(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->total_AF_units);
        if(this->Allocate__AF_Units() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__AF_Units()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        // |END| Allocate AF unit(s). |END|
        
        // Allocate af_ind unit(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_AF_Ind_recurrent_units") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_AF_Ind_recurrent_units\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_AF_Ind_recurrent_units >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units = nullptr;
        tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units + this->total_AF_Ind_recurrent_units;
        
        PRINT_FORMAT("%s: Allocate %zu AF Ind recurrent unit(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->total_AF_Ind_recurrent_units);
        if(this->Allocate__AF_Ind_Recurrent_Units() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__AF_Ind_Recurrent_Units()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        // |END| Allocate af_ind unit(s). |END|

        // Allocate block/cell unit(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_block_units") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_block_units\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_block_units >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        tmp_ptr_layer_it->ptr_array_block_units = nullptr;
        tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_layer_it->ptr_array_block_units + this->total_block_units;
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_cell_units") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_cell_units\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_cell_units >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        tmp_ptr_layer_it->ptr_array_cell_units = nullptr;
        tmp_ptr_layer_it->ptr_last_cell_unit = tmp_ptr_layer_it->ptr_array_cell_units + this->total_cell_units;

        if(this->total_block_units != 0_zu && this->total_cell_units != 0_zu)
        {
            PRINT_FORMAT("%s: Allocate %zu block unit(s)." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     this->total_block_units);
            PRINT_FORMAT("%s: Allocate %zu cell unit(s)." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     this->total_cell_units);
            if(this->Allocate__LSTM_Layers() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__LSTM_Layers()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }
        // |END| Allocate block/cell unit(s). |END|
        
        // Allocate normalized unit(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_normalized_units") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_normalized_units\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_normalized_units >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        PRINT_FORMAT("%s: Allocate %zu normalized unit(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->total_normalized_units);
        if(this->Allocate__Normalized_Unit(false) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Normalized_Unit(false)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        // |END| Allocate normalized unit(s). |END|

        // Allocate parameter(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_parameters") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_parameters\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_parameters >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_weights") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_weights\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_weights >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("total_bias") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"total_bias\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> this->total_bias >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }
        }
        
        PRINT_FORMAT("%s: Allocate %zu parameter(s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->total_parameters);
        if(this->Allocate__Parameter() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        // |END| Allocate parameter(s). |END|

        // Initialize layer(s).
        // Reset number of weights to zero. Increment the variable inside the loading layer.
        this->total_weights = 0_zu;
        this->total_bias = 0_zu;
        
        struct Basic_unit *tmp_ptr_array_basic_units(this->ptr_array_basic_units);
        
        struct Basic_indice_unit *tmp_ptr_array_basic_indice_units(this->ptr_array_basic_indice_units);

        struct Neuron_unit *tmp_ptr_array_neuron_units(this->ptr_array_neuron_units);
        
        struct AF_unit *tmp_ptr_array_AF_units(this->ptr_array_AF_units);

        struct AF_Ind_recurrent_unit *tmp_ptr_array_AF_Ind_recurrent_units(this->ptr_array_AF_Ind_recurrent_units);

        struct Block_unit *tmp_ptr_array_block_units(this->ptr_array_block_units);

        struct Cell_unit *tmp_ptr_array_cell_units(this->ptr_array_cell_units);
        
        union Normalized_unit *tmp_ptr_array_normalized_units(this->ptr_array_normalized_units);

        // Input layer.
        //  Type layer.
        PRINT_FORMAT("%s: Load input layer." NEW_LINE, MyEA::String::Get__Time().c_str());
        getline(tmp_ifstream, tmp_line); // "Input layer:"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("type_layer") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_layer\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH))
            {
                PRINT_FORMAT("%s: %s: ERROR: Undefined layer type %zu. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_input_integer,
                                         __LINE__);

                return(false);
            }

            tmp_ptr_layer_it->type_layer = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER>(tmp_input_integer);
        }
        //  |END| Type layer. |END|
        
        //  Type activation.
        if((tmp_ifstream >> tmp_line) && tmp_line.find("type_activation") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_activation\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }

            if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_LENGTH))
            {
                PRINT_FORMAT("%s: %s: ERROR: Undefined layer activation type %zu. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_input_integer,
                                         __LINE__);

                return(false);
            }

            tmp_ptr_layer_it->type_activation = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION>(tmp_input_integer);
        }
        //  |END| Type activation. |END|
        
        //  Dropout.
        if(tmp_Load__Dropout__Parameters(tmp_ptr_layer_it, false) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Load__Dropout__Parameters(false)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        //  |END| Dropout. |END|
        
        //  Initialize input(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("number_inputs") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_inputs\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                return(false);
            }

            *tmp_ptr_layer_it->ptr_number_outputs = this->number_inputs = tmp_input_integer;

            tmp_ptr_layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
            tmp_ptr_array_neuron_units += tmp_input_integer;
            tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
            
            this->Order__Layer__Neuron(tmp_ptr_layer_it);
        }
        //  |END| Initialize input(s). |END|
        
        //  Initialize normalized unit(s).
        tmp_ptr_layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
        if(this->total_normalized_units_allocated != 0_zu) { tmp_ptr_array_normalized_units += *tmp_ptr_layer_it->ptr_number_outputs; } // If use normalization.
        tmp_ptr_layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
        //  |END| Initialize normalized unit(s). |END|
        
        // Initialize AF unit(s).
        tmp_ptr_layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
        tmp_ptr_array_AF_units += *tmp_ptr_layer_it->ptr_number_outputs;
        tmp_ptr_layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
        // |END| Initialize AF unit(s). |END|
        
        // Initialize AF Ind recurrent unit(s).
        tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
        tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
        // |END| Initialize AF Ind recurrent unit(s). |END|
        
        //  Initialize basic unit(s).
        tmp_ptr_layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
        tmp_ptr_layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
        //  |END| Initialize basic unit(s). |END|
        
        //  Initialize basic indice unit(s).
        tmp_ptr_layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
        tmp_ptr_layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
        //  |END| Initialize basic indice unit(s). |END|
        
        //  Initialize block/cell unit(s).
        tmp_ptr_layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
        tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

        tmp_ptr_layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
        tmp_ptr_layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
        //  |END| Initialize block/cell unit(s). |END|
        // |END| Input layer. |END|

        // Hidden layer.
        for(++tmp_ptr_layer_it; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            // Type layer.
            getline(tmp_ifstream, tmp_line); // "Hidden layer %u:"
            
            if((tmp_ifstream >> tmp_line) && tmp_line.find("type_layer") == std::string::npos)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_layer\" inside \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
            else if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
            else
            {
                tmp_ifstream >> tmp_input_integer;

                if(tmp_ifstream.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_line.c_str(),
                                             __LINE__);

                    return(false);
                }

                if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH))
                {
                    PRINT_FORMAT("%s: %s: ERROR: Undefined layer type %zu. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_input_integer,
                                             __LINE__);

                    return(false);
                }

                tmp_ptr_layer_it->type_layer = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER>(tmp_input_integer);
            }
            
            PRINT_FORMAT("%s: Load hidden layer %zu (%s | %u)." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    static_cast<size_t>(tmp_ptr_layer_it - tmp_ptr_first_layer),
                                    MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                    tmp_ptr_layer_it->type_layer);
            // |END| Type layer. |END|
            
            this->Organize__Previous_Layers_Connected(tmp_state_layer_index,
                                                                               tmp_ptr_layer_it,
                                                                               tmp_ptr_layer_state);

            tmp_ptr_previous_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

            // Use bidirectional.
            if((tmp_ifstream >> tmp_line) && tmp_line.find("use_bidirectional") == std::string::npos)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_bidirectional\" inside \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
            else if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);

                return(false);
            }
            else
            {
                tmp_ifstream >> tmp_ptr_layer_it->use_bidirectional;

                if(tmp_ifstream.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_line.c_str(),
                                             __LINE__);

                    return(false);
                }
            }
            // |END| Use bidirectional. |END|
            
            switch(tmp_ptr_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                    // Pooling.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("kernel_size") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"kernel_size\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->pooling_values[0u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("stride") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"stride\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->pooling_values[1u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("padding") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"padding\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->pooling_values[2u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("dilation") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"dilation\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->pooling_values[3u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    // |END| Pooling. |END|

                    //  Initialize normalized unit(s).
                    tmp_ptr_layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                    tmp_ptr_layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                    //  |END| Initialize normalized unit(s). |END|
                    
                    // Initialize basic unit(s).
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_basic_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_basic_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> *tmp_ptr_layer_it->ptr_number_outputs >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }

                        tmp_ptr_layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                        tmp_ptr_array_basic_units += *tmp_ptr_layer_it->ptr_number_outputs;
                        tmp_ptr_layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;

                        this->Order__Layer__Basic(tmp_ptr_layer_it);
                    }
                    // |END| Initialize basic unit(s). |END|
                    
                    //  Initialize basic indice unit(s).
                    tmp_ptr_layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
                    tmp_ptr_layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
                    //  |END| Initialize basic indice unit(s). |END|
                    
                    // Initialize neuron unit(s).
                    tmp_ptr_layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
                    tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
                    // |END| Initialize neuron unit(s). |END|
                    
                    // Initialize AF unit(s).
                    tmp_ptr_layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
                    tmp_ptr_layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
                    // |END| Initialize AF unit(s). |END|
                    
                    // Initialize AF Ind recurrent unit(s).
                    tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                    tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
                    // |END| Initialize AF Ind recurrent unit(s). |END|
                    
                    //  Initialize block/cell unit(s).
                    tmp_ptr_layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
                    tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

                    tmp_ptr_layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
                    tmp_ptr_layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
                    //  |END| Initialize block/cell unit(s). |END|
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                    // Type activation.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("type_activation") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_activation\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);

                            return(false);
                        }

                        if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_LENGTH))
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Undefined layer activation type %zu. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);

                            return(false);
                        }

                        tmp_ptr_layer_it->type_activation = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION>(tmp_input_integer);
                    }
                    // |END| Type activation. |END|

                    // Dropout.
                    if(tmp_Load__Dropout__Parameters(tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Load__Dropout__Parameters()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Dropout. |END|
                    
                    // Normalization.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("type_normalization") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_normalization\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);

                            return(false);
                        }

                        if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_LENGTH))
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Undefined layer normalization type %zu. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);

                            return(false);
                        }
                        
                        if(tmp_Valid__Layer__Normalization(tmp_ptr_layer_it)
                          &&
                          this->Set__Layer_Normalization(tmp_ptr_layer_it,
                                                                         static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION>(tmp_input_integer),
                                                                         false,
                                                                         false) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization(%zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);
                            
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("use_layer_normalization_before_activation") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_layer_normalization_before_activation\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->use_layer_normalization_before_activation >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_normalized_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_normalized_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }
                        
                        tmp_ptr_layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                        tmp_ptr_array_normalized_units += tmp_input_integer;
                        tmp_ptr_layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                        
                        this->Order__Layer__Normalization(tmp_ptr_layer_it);

                        if(this->Load_Dimension__Normalization(tmp_ptr_layer_it, tmp_ifstream) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Normalization()\" function. At line %d." NEW_LINE,
                                                        MyEA::String::Get__Time().c_str(),
                                                        __FUNCTION__,
                                                        __LINE__);

                            return(false);
                        }
                    }
                    // |END| Normalization. |END|

                    if((tmp_ifstream >> tmp_line) && tmp_line.find("use_tied_parameter") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_tied_parameter\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_boolean >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__Tied_Parameter(tmp_ptr_layer_it,
                                                              tmp_input_boolean,
                                                              false) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Tied_Parameter(ptr, %s, false)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_boolean ? "true" : "false",
                                                     __LINE__);

                            return(false);
                        }
                    }
                    
                    // k-Sparse filters.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("k_sparsity") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"k_sparsity\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__K_Sparsity(tmp_ptr_layer_it, tmp_input_integer) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__K_Sparsity(ptr, %zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);

                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("alpha_sparsity") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"alpha_sparsity\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_T[0u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__Alpha_Sparsity(tmp_ptr_layer_it, tmp_input_T[0u]) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Alpha_Sparsity(ptr, %f)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_T[0u],
                                                     __LINE__);

                            return(false);
                        }
                    }
                    // |END| k-Sparse filters. |END|
                    
                    // Constraint.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("constraint_recurrent_weight_lower_bound") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"constraint_recurrent_weight_lower_bound\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_T[0u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("constraint_recurrent_weight_upper_bound") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"constraint_recurrent_weight_upper_bound\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_T[1u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__Regularization__Constraint_Recurrent_Weight(tmp_ptr_layer_it,
                                                                                       tmp_input_T[0u],
                                                                                       tmp_input_T[1u]) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight(ptr, %f, %f)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_T[0u],
                                                     tmp_input_T[1u],
                                                     __LINE__);

                            return(false);
                        }
                    }
                    // |END| Constraint. |END|
                    
                    // Initialize basic unit(s).
                    tmp_ptr_layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                    tmp_ptr_layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
                    // |END| Initialize basic unit(s). |END|
                    
                    // Initialize basic indice unit(s).
                    tmp_ptr_layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
                    tmp_ptr_layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
                    // |END| Initialize basic indice unit(s). |END|
                    
                    // Initialize neuron unit(s).
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_neuron_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_neuron_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> *tmp_ptr_layer_it->ptr_number_outputs >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }
                        
                        tmp_ptr_layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
                        tmp_ptr_array_neuron_units += *tmp_ptr_layer_it->ptr_number_outputs;
                        tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;

                        this->Order__Layer__Neuron(tmp_ptr_layer_it);

                        switch(tmp_ptr_previous_layer->type_layer)
                        {
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                                if(this->Load_Dimension__FC<struct Basic_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING>(tmp_ptr_layer_it,
                                                                                                                                                                                                                            this->ptr_array_basic_units,
                                                                                                                                                                                                                            tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                                if(this->Load_Dimension__FC<struct Neuron_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED>(tmp_ptr_layer_it,
                                                                                                                                                                                                                              this->ptr_array_neuron_units,
                                                                                                                                                                                                                              tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                                if(this->Load_Dimension__FC<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_ptr_layer_it,
                                                                                                                                                                                                  this->ptr_array_cell_units,
                                                                                                                                                                                                  tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                                if(this->Load_Dimension__FC<struct Basic_indice_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING>(tmp_ptr_layer_it,
                                                                                                                                                                                                                              this->ptr_array_basic_indice_units,
                                                                                                                                                                                                                              tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            default:
                                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         tmp_ptr_previous_layer->type_layer,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str());
                                    return(false);
                        }
                    }
                    // |END| Initialize neuron unit(s). |END|
                    
                    // Initialize AF unit(s).
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_AF_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_AF_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }
                        
                        tmp_ptr_layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
                        tmp_ptr_array_AF_units += tmp_input_integer;
                        tmp_ptr_layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
                        
                        this->Order__Layer__AF(tmp_ptr_layer_it);

                        if(this->Load_Dimension__AF(tmp_ptr_layer_it, tmp_ifstream) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__AF()\" function. At line %d." NEW_LINE,
                                                        MyEA::String::Get__Time().c_str(),
                                                        __FUNCTION__,
                                                        __LINE__);

                            return(false);
                        }
                    }
                    // |END| Initialize AF unit(s). |END|
                    
                    // Initialize AF Ind recurrent unit(s).
                    tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                    tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
                    // |END| Initialize AF Ind recurrent unit(s). |END|
                    
                    //  Initialize block/cell unit(s).
                    tmp_ptr_layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
                    tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

                    tmp_ptr_layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
                    tmp_ptr_layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
                    //  |END| Initialize block/cell unit(s). |END|
                    
                    // Initialize bias parameter(s).
                    if(this->Load_Dimension__Bias(tmp_ptr_layer_it, tmp_ifstream) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Bias()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Initialize bias parameter(s). |END|
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    // Type activation.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("type_activation") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_activation\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);

                            return(false);
                        }

                        if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_LENGTH))
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Undefined layer activation type %zu. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);

                            return(false);
                        }

                        tmp_ptr_layer_it->type_activation = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION>(tmp_input_integer);
                    }
                    // |END| Type activation. |END|

                    // Dropout.
                    if(tmp_Load__Dropout__Parameters(tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Load__Dropout__Parameters()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Dropout. |END|
                    
                    // Normalization.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("type_normalization") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_normalization\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);

                            return(false);
                        }

                        if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_LENGTH))
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Undefined layer normalization type %zu. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);

                            return(false);
                        }
                        
                        if(this->Set__Layer_Normalization(tmp_ptr_layer_it,
                                                                          static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION>(tmp_input_integer),
                                                                          false,
                                                                          false) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization(%zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);
                            
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("use_layer_normalization_before_activation") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_layer_normalization_before_activation\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->use_layer_normalization_before_activation >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_normalized_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_normalized_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }
                        
                        tmp_ptr_layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                        tmp_ptr_array_normalized_units += tmp_input_integer;
                        tmp_ptr_layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                        
                        this->Order__Layer__Normalization(tmp_ptr_layer_it);

                        if(this->Load_Dimension__Normalization(tmp_ptr_layer_it, tmp_ifstream) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Normalization()\" function. At line %d." NEW_LINE,
                                                        MyEA::String::Get__Time().c_str(),
                                                        __FUNCTION__,
                                                        __LINE__);

                            return(false);
                        }
                    }
                    // |END| Normalization. |END|

                    if((tmp_ifstream >> tmp_line) && tmp_line.find("use_tied_parameter") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_tied_parameter\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_boolean >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__Tied_Parameter(tmp_ptr_layer_it,
                                                              tmp_input_boolean,
                                                              false) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Tied_Parameter(ptr, %s, false)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_boolean ? "true" : "false",
                                                     __LINE__);

                            return(false);
                        }
                    }
                    
                    // k-Sparse filters.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("k_sparsity") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"k_sparsity\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__K_Sparsity(tmp_ptr_layer_it, tmp_input_integer) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__K_Sparsity(ptr, %zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);

                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("alpha_sparsity") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"alpha_sparsity\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_T[0u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__Alpha_Sparsity(tmp_ptr_layer_it, tmp_input_T[0u]) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Alpha_Sparsity(ptr, %f)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_T[0u],
                                                     __LINE__);

                            return(false);
                        }
                    }
                    // |END| k-Sparse filters. |END|
                    
                    // Constraint.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("constraint_recurrent_weight_lower_bound") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"constraint_recurrent_weight_lower_bound\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_T[0u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("constraint_recurrent_weight_upper_bound") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"constraint_recurrent_weight_upper_bound\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_T[1u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__Regularization__Constraint_Recurrent_Weight(tmp_ptr_layer_it,
                                                                                       tmp_input_T[0u],
                                                                                       tmp_input_T[1u]) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight(ptr, %f, %f)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_T[0u],
                                                     tmp_input_T[1u],
                                                     __LINE__);

                            return(false);
                        }
                    }
                    // |END| Constraint. |END|
                    
                    // Initialize basic unit(s).
                    tmp_ptr_layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                    tmp_ptr_layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
                    // |END| Initialize basic unit(s). |END|
                    
                    // Initialize basic indice unit(s).
                    tmp_ptr_layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
                    tmp_ptr_layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
                    // |END| Initialize basic indice unit(s). |END|
                    
                    // Initialize neuron unit(s).
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_neuron_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_neuron_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> *tmp_ptr_layer_it->ptr_number_outputs >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }
                        
                        tmp_ptr_layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
                        tmp_ptr_array_neuron_units += *tmp_ptr_layer_it->ptr_number_outputs;
                        tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;

                        this->Order__Layer__Neuron(tmp_ptr_layer_it);

                        switch(tmp_ptr_previous_layer->type_layer)
                        {
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                                if(this->Load_Dimension__FC<struct Basic_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING>(tmp_ptr_layer_it,
                                                                                                                                                                                                                             this->ptr_array_basic_units,
                                                                                                                                                                                                                             tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                                if(this->Load_Dimension__FC<struct Neuron_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED>(tmp_ptr_layer_it,
                                                                                                                                                                                                                              this->ptr_array_neuron_units,
                                                                                                                                                                                                                              tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                                if(this->Load_Dimension__FC<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_ptr_layer_it,
                                                                                                                                                                                                  this->ptr_array_cell_units,
                                                                                                                                                                                                  tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                                if(this->Load_Dimension__FC<struct Basic_indice_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING>(tmp_ptr_layer_it,
                                                                                                                                                                                                                              this->ptr_array_basic_indice_units,
                                                                                                                                                                                                                              tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            default:
                                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         tmp_ptr_previous_layer->type_layer,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str());
                                    return(false);
                        }
                    }
                    // |END| Initialize neuron unit(s). |END|
                    
                    // Initialize AF unit(s).
                    tmp_ptr_layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
                    tmp_ptr_layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
                    // |END| Initialize AF unit(s). |END|
                    
                    // Initialize AF Ind recurrent unit(s).
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_AF_Ind_recurrent_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_AF_Ind_recurrent_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }
                        
                        tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                        tmp_ptr_array_AF_Ind_recurrent_units += tmp_input_integer;
                        tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
                        
                        this->Order__Layer__AF_Ind_Recurrent(tmp_ptr_layer_it);

                        if(this->Load_Dimension__AF_Ind_Recurrent(tmp_ptr_layer_it, tmp_ifstream) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__AF_Ind_Recurrent()\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                    }
                    // |END| Initialize AF Ind recurrent unit(s). |END|

                    // Initialize block/cell unit(s).
                    tmp_ptr_layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
                    tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

                    tmp_ptr_layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
                    tmp_ptr_layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
                    // |END| Initialize block/cell unit(s). |END|
                    
                    // Initialize bias parameter(s).
                    if(this->Load_Dimension__Bias(tmp_ptr_layer_it, tmp_ifstream) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Bias()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Initialize bias parameter(s). |END|
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    // Type activation.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("type_activation") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_activation\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);

                            return(false);
                        }

                        if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_LENGTH))
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Undefined layer activation type %zu. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);

                            return(false);
                        }

                        tmp_ptr_layer_it->type_activation = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION>(tmp_input_integer);
                    }
                    // |END| Type activation. |END|

                    // Dropout.
                    if(tmp_Load__Dropout__Parameters(tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Load__Dropout__Parameters()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Dropout. |END|
                    
                    // Normalization.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("type_normalization") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_normalization\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);

                            return(false);
                        }

                        if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_LENGTH))
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Undefined layer normalization type %zu. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);

                            return(false);
                        }
                        
                        if(this->Set__Layer_Normalization(tmp_ptr_layer_it,
                                                                          static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION>(tmp_input_integer),
                                                                          false,
                                                                          false) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization(%zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);
                            
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("use_layer_normalization_before_activation") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_layer_normalization_before_activation\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->use_layer_normalization_before_activation >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_normalized_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_normalized_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }
                        
                        tmp_ptr_layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                        tmp_ptr_array_normalized_units += tmp_input_integer;
                        tmp_ptr_layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                        
                        this->Order__Layer__Normalization(tmp_ptr_layer_it);

                        if(this->Load_Dimension__Normalization(tmp_ptr_layer_it, tmp_ifstream) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Normalization()\" function. At line %d." NEW_LINE,
                                                        MyEA::String::Get__Time().c_str(),
                                                        __FUNCTION__,
                                                        __LINE__);

                            return(false);
                        }
                    }
                    // |END| Normalization. |END|

                    if((tmp_ifstream >> tmp_line) && tmp_line.find("use_tied_parameter") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"use_tied_parameter\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_boolean >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__Tied_Parameter(tmp_ptr_layer_it,
                                                              tmp_input_boolean,
                                                              false) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Tied_Parameter(ptr, %s, false)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_boolean ? "true" : "false",
                                                     __LINE__);

                            return(false);
                        }
                    }
                    
                    // k-Sparse filters.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("k_sparsity") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"k_sparsity\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__K_Sparsity(tmp_ptr_layer_it, tmp_input_integer) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__K_Sparsity(ptr, %zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);

                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("alpha_sparsity") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"alpha_sparsity\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_T[0u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__Alpha_Sparsity(tmp_ptr_layer_it, tmp_input_T[0u]) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Alpha_Sparsity(ptr, %f)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_T[0u],
                                                     __LINE__);

                            return(false);
                        }
                    }
                    // |END| k-Sparse filters. |END|
                    
                    // Constraint.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("constraint_recurrent_weight_lower_bound") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"constraint_recurrent_weight_lower_bound\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_T[0u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("constraint_recurrent_weight_upper_bound") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"constraint_recurrent_weight_upper_bound\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_T[1u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }

                        if(this->Set__Regularization__Constraint_Recurrent_Weight(tmp_ptr_layer_it,
                                                                                       tmp_input_T[0u],
                                                                                       tmp_input_T[1u]) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight(ptr, %f, %f)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_T[0u],
                                                     tmp_input_T[1u],
                                                     __LINE__);

                            return(false);
                        }
                    }
                    // |END| Constraint. |END|
                    
                    // Initialize basic unit(s).
                    tmp_ptr_layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                    tmp_ptr_layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
                    // |END| Initialize basic unit(s). |END|
                    
                    // Initialize basic indice unit(s).
                    tmp_ptr_layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
                    tmp_ptr_layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
                    // |END| Initialize basic indice unit(s). |END|
                    
                    // Initialize neuron unit(s).
                    tmp_ptr_layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
                    tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
                    // |END| Initialize neuron unit(s). |END|
                    
                    // Initialize AF unit(s).
                    tmp_ptr_layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
                    tmp_ptr_layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
                    // |END| Initialize AF unit(s). |END|
                    
                    // Initialize AF Ind recurrent unit(s).
                    tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                    tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
                    // |END| Initialize AF Ind recurrent unit(s). |END|
                    
                    // Initialize block/cell unit(s).
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_block_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_block_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }
                            
                        tmp_ptr_layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
                        tmp_ptr_array_block_units += tmp_input_integer;
                        tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

                        if(this->Load_Dimension__Cell_Units(tmp_ptr_layer_it,
                                                                                tmp_ptr_array_cell_units,
                                                                                tmp_ifstream) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Cell_Units()\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                    
                        *tmp_ptr_layer_it->ptr_number_outputs = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units);

                        this->Order__Layer__LSTM(tmp_ptr_layer_it);

                        switch(tmp_ptr_previous_layer->type_layer)
                        {
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                                if(this->Load_Dimension__LSTM<struct Basic_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING>(tmp_ptr_layer_it,
                                                                                                                                                                                                                                 this->ptr_array_basic_units,
                                                                                                                                                                                                                                 tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__LSTM()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                                if(this->Load_Dimension__LSTM<struct Neuron_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED>(tmp_ptr_layer_it,
                                                                                                                                                                                                                                  this->ptr_array_neuron_units,
                                                                                                                                                                                                                                  tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__LSTM()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                                if(this->Load_Dimension__LSTM<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_ptr_layer_it,
                                                                                                                                                                                                       this->ptr_array_cell_units,
                                                                                                                                                                                                       tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__LSTM()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                    
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                                if(this->Load_Dimension__LSTM<struct Basic_indice_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING>(tmp_ptr_layer_it,
                                                                                                                                                                                                                                  this->ptr_array_basic_indice_units,
                                                                                                                                                                                                                                  tmp_ifstream) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__LSTM()\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);
                                        
                                    tmp_ifstream.close();

                                    return(false);
                                }
                                    break;
                            default:
                                PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         tmp_ptr_previous_layer->type_layer,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str());
                                    return(false);
                        }
                    }
                    // |END| Initialize block/cell unit(s). |END|
                    
                    // Initialize bias parameter(s).
                    if(this->Load_Dimension__Bias(tmp_ptr_layer_it, tmp_ifstream) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Bias()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Initialize bias parameter(s). |END|
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                    // Pooling.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("kernel_size") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"kernel_size\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->pooling_values[0u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("stride") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"stride\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->pooling_values[1u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("padding") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"padding\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->pooling_values[2u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("dilation") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"dilation\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->pooling_values[3u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    // |END| Pooling. |END|

                    //  Initialize normalized unit(s).
                    tmp_ptr_layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                    tmp_ptr_layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                    //  |END| Initialize normalized unit(s). |END|
                    
                    //  Initialize basic unit(s).
                    tmp_ptr_layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                    tmp_ptr_layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
                    //  |END| Initialize basic unit(s). |END|
                    
                    // Initialize basic indice unit(s).
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_basic_indice_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_basic_indice_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> *tmp_ptr_layer_it->ptr_number_outputs >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }

                        tmp_ptr_layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
                        tmp_ptr_array_basic_indice_units += *tmp_ptr_layer_it->ptr_number_outputs;
                        tmp_ptr_layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;

                        this->Order__Layer__Basic_indice(tmp_ptr_layer_it);
                    }
                    // |END| Initialize basic indice unit(s). |END|
                    
                    // Initialize neuron unit(s).
                    tmp_ptr_layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
                    tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
                    // |END| Initialize neuron unit(s). |END|
                    
                    // Initialize AF unit(s).
                    tmp_ptr_layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
                    tmp_ptr_layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
                    // |END| Initialize AF unit(s). |END|
                    
                    // Initialize AF Ind recurrent unit(s).
                    tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                    tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
                    // |END| Initialize AF Ind recurrent unit(s). |END|
                    
                    //  Initialize block/cell unit(s).
                    tmp_ptr_layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
                    tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

                    tmp_ptr_layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
                    tmp_ptr_layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
                    //  |END| Initialize block/cell unit(s). |END|
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    // Initialize block depth.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("block_depth") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"block_depth\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->block_depth >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }
                    }
                    // |END| Initialize block depth. |END|
                    
                    // Initialize padding.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("padding") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"padding\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
            
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_ptr_layer_it->pooling_values[2u] >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                
                            return(false);
                        }
                    }
                    // |END| Initialize padding. |END|
                    
                    // Dropout.
                    if(tmp_Load__Dropout__Parameters(tmp_ptr_layer_it) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Load__Dropout__Parameters()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                    // |END| Dropout. |END|
                    
                    // Normalization.
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("type_normalization") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_normalization\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);

                            return(false);
                        }

                        if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_LENGTH))
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Undefined layer normalization type %zu. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);

                            return(false);
                        }
                        
                        if(this->Set__Layer_Normalization(tmp_ptr_layer_it,
                                                                          static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION>(tmp_input_integer),
                                                                          false,
                                                                          false) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization(%zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_integer,
                                                     __LINE__);
                            
                            return(false);
                        }
                    }
                    
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_normalized_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_normalized_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> tmp_input_integer >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }
                        
                        tmp_ptr_layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                        tmp_ptr_array_normalized_units += tmp_input_integer;
                        tmp_ptr_layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                        
                        this->Order__Layer__Normalization(tmp_ptr_layer_it);

                        if(this->Load_Dimension__Normalization(tmp_ptr_layer_it, tmp_ifstream) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Normalization()\" function. At line %d." NEW_LINE,
                                                        MyEA::String::Get__Time().c_str(),
                                                        __FUNCTION__,
                                                        __LINE__);

                            return(false);
                        }
                    }
                    // |END| Normalization. |END|
                    
                    // Initialize basic unit(s).
                    if((tmp_ifstream >> tmp_line) && tmp_line.find("number_basic_units") == std::string::npos)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_basic_units\" inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                    else if(tmp_ifstream.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);
                
                        return(false);
                    }
                    else
                    {
                        tmp_ifstream >> *tmp_ptr_layer_it->ptr_number_outputs >> std::ws;

                        if(tmp_ifstream.fail())
                        {
                            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_line.c_str(),
                                                     __LINE__);
                    
                            return(false);
                        }

                        tmp_ptr_layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                        tmp_ptr_array_basic_units += *tmp_ptr_layer_it->ptr_number_outputs;
                        tmp_ptr_layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;

                        this->Order__Layer__Basic(tmp_ptr_layer_it);
                    }
                    // |END| Initialize basic unit(s). |END|
                    
                    //  Initialize basic indice unit(s).
                    tmp_ptr_layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
                    tmp_ptr_layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
                    //  |END| Initialize basic indice unit(s). |END|
                    
                    // Initialize neuron unit(s).
                    tmp_ptr_layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
                    tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
                    // |END| Initialize neuron unit(s). |END|
                    
                    // Initialize AF unit(s).
                    tmp_ptr_layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
                    tmp_ptr_layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
                    // |END| Initialize AF unit(s). |END|
                    
                    // Initialize AF Ind recurrent unit(s).
                    tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                    tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
                    // |END| Initialize AF Ind recurrent unit(s). |END|
                    
                    //  Initialize block/cell unit(s).
                    tmp_ptr_layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
                    tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

                    tmp_ptr_layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
                    tmp_ptr_layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
                    //  |END| Initialize block/cell unit(s). |END|
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_layer_it->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                        return(false);
            }
        }
        // |END| Hidden layer. |END|

        // Allocate bidirectional layer(s).
        if(this->Allocate__Bidirectional__Layers() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Bidirectional__Layers()\" function. At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        __LINE__);

            tmp_ifstream.close();

            return(false);
        }
        // |END| Allocate bidirectional layer(s). |END|

        // Output layer.
        //  Type layer.
        PRINT_FORMAT("%s: Load output layer." NEW_LINE, MyEA::String::Get__Time().c_str());
        getline(tmp_ifstream, tmp_line); // "Output layer:"
        
        if((tmp_ifstream >> tmp_line) && tmp_line.find("type_layer") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_layer\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            tmp_ifstream.close();

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            tmp_ifstream.close();

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                tmp_ifstream.close();

                return(false);
            }

            if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LENGTH))
            {
                PRINT_FORMAT("%s: %s: ERROR: Undefined layer type %zu. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_input_integer,
                                         __LINE__);
                
                tmp_ifstream.close();

                return(false);
            }

            tmp_ptr_layer_it->type_layer = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER>(tmp_input_integer);
        }
        //  |END| Type layer. |END|
        
        this->Organize__Previous_Layers_Connected(tmp_state_layer_index,
                                                                           tmp_ptr_layer_it,
                                                                           tmp_ptr_layer_state);
        
        tmp_ptr_previous_layer = tmp_ptr_layer_it->previous_connected_layers[0u];

        //  Type activation.
        if((tmp_ifstream >> tmp_line) && tmp_line.find("type_activation") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"type_activation\" inside \"%s\". At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_line.c_str(),
                                        __LINE__);
            
            tmp_ifstream.close();

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_line.c_str(),
                                        __LINE__);
            
            tmp_ifstream.close();

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);
                
                tmp_ifstream.close();

                return(false);
            }

            if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_LENGTH))
            {
                PRINT_FORMAT("%s: %s: ERROR: Undefined layer activation type %zu. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_input_integer,
                                            __LINE__);
                
                tmp_ifstream.close();

                return(false);
            }

            tmp_ptr_layer_it->type_activation = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION>(tmp_input_integer);
        }
        //  |END| Type activation. |END|
        
        //  Initialize output unit(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("number_outputs") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_outputs\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            tmp_ifstream.close();

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
            
            tmp_ifstream.close();

            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                
                tmp_ifstream.close();

                return(false);
            }

            *tmp_ptr_layer_it->ptr_number_outputs = this->number_outputs = tmp_input_integer;
            
            tmp_ptr_layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
            tmp_ptr_array_neuron_units += tmp_input_integer;
            tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
            
            this->Order__Layer__Neuron(tmp_ptr_layer_it);

            switch(tmp_ptr_previous_layer->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                    if(this->Load_Dimension__FC<struct Basic_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING>(tmp_ptr_layer_it,
                                                                                                                                                                                                                 this->ptr_array_basic_units,
                                                                                                                                                                                                                 tmp_ifstream) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);
                                    
                        tmp_ifstream.close();

                        return(false);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    if(this->Load_Dimension__FC<struct Neuron_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED>(tmp_ptr_layer_it,
                                                                                                                                                                                                                  this->ptr_array_neuron_units,
                                                                                                                                                                                                                  tmp_ifstream) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);
                        
                        tmp_ifstream.close();

                        return(false);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                    if(this->Load_Dimension__FC<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_ptr_layer_it,
                                                                                                                                                                                      this->ptr_array_cell_units,
                                                                                                                                                                                      tmp_ifstream) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);
                        
                        tmp_ifstream.close();

                        return(false);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
                    if(this->Load_Dimension__FC<struct Basic_indice_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING>(tmp_ptr_layer_it,
                                                                                                                                                                                                                  this->ptr_array_basic_indice_units,
                                                                                                                                                                                                                  tmp_ifstream) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__FC()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);
                                        
                        tmp_ifstream.close();

                        return(false);
                    }
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_previous_layer->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str());
                    tmp_ifstream.close();
                        return(false);
            }
        }
        //  |END| Initialize output unit(s). |END|
        
        // Initialize AF unit(s).
        if((tmp_ifstream >> tmp_line) && tmp_line.find("number_AF_units") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_AF_units\" inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
        else if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
                
            return(false);
        }
        else
        {
            tmp_ifstream >> tmp_input_integer >> std::ws;

            if(tmp_ifstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_line.c_str(),
                                         __LINE__);
                    
                return(false);
            }
                        
            tmp_ptr_layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
            tmp_ptr_array_AF_units += tmp_input_integer;
            tmp_ptr_layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
                        
            this->Order__Layer__AF(tmp_ptr_layer_it);

            if(this->Load_Dimension__AF(tmp_ptr_layer_it, tmp_ifstream) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__AF()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }
        // |END| Initialize AF unit(s). |END|
        
        // Initialize AF Ind recurrent unit(s).
        tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
        tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
        // |END| Initialize AF Ind recurrent unit(s). |END|
        
        //  Initialize basic unit(s).
        tmp_ptr_layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
        tmp_ptr_layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
        //  |END| Initialize basic unit(s). |END|
        
        //  Initialize basic indice unit(s).
        tmp_ptr_layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
        tmp_ptr_layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
        //  |END| Initialize basic indice unit(s). |END|
        
        //  Initialize block/cell unit(s).
        tmp_ptr_layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
        tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

        tmp_ptr_layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
        tmp_ptr_layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
        //  |END| Initialize block/cell unit(s). |END|
                    
        //  Initialize bias parameter(s).
        if(this->Load_Dimension__Bias(tmp_ptr_layer_it, tmp_ifstream) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Bias()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        //  |END| Initialize bias parameter(s). |END|
        // |END| Output layer. |END|
        
        if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Logical error on i/o operation \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ref_path_dimension_received.c_str(),
                                     __LINE__);
            
            tmp_ifstream.close();

            return(false);
        }

        tmp_ifstream.close();
        
        if(this->total_weights != this->total_weights_allocated)
        {
            PRINT_FORMAT("%s: %s: ERROR: Total weights prepared (%zu) differ from the total weights allocated (%zu). At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->total_weights,
                                     this->total_weights_allocated,
                                     __LINE__);

            return(false);
        }
        else if(this->total_bias != this->total_bias_allocated)
        {
            PRINT_FORMAT("%s: %s: ERROR: Total bias prepared (%zu) differ from the total bias allocated (%zu). At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->total_bias,
                                     this->total_bias_allocated,
                                     __LINE__);

            return(false);
        }
        // |END| Initialize layer(s). |END|
        
        // Layers, connections.
        this->Order__Layers__Connection();
        
        // Layers, outputs pointers.
        this->Order__Layers__Output();

        if(this->Load_Parameters(ref_path_parameters_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Parameters(%s)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     ref_path_parameters_received.c_str(),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 ref_path_dimension_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::Load_Dimension__Neuron(struct Neuron_unit *const ptr_neuron_received, std::ifstream &ref_ifstream_received)
{
    std::string tmp_line;

    getline(ref_ifstream_received, tmp_line); // "Neuron_unit[%zu]"
    
    // Number connection(s).
    if((ref_ifstream_received >> tmp_line) && tmp_line.find("number_connections") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_connections\" inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else
    {
        ref_ifstream_received >> *ptr_neuron_received->ptr_number_connections >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
    }

    *ptr_neuron_received->ptr_first_connection_index = this->total_weights;
    this->total_weights += *ptr_neuron_received->ptr_number_connections;
    *ptr_neuron_received->ptr_last_connection_index = this->total_weights;
    // |END| Number connection(s). |END|

    return(true);
}

bool Neural_Network::Load_Dimension__AF(struct AF_unit *const ptr_AF_received, std::ifstream &ref_ifstream_received)
{
    size_t tmp_input_integer;

    std::string tmp_line;

    getline(ref_ifstream_received, tmp_line); // "AF[%zu]"
    
    // Activation steepness.
    if((ref_ifstream_received >> tmp_line) && tmp_line.find("activation_steepness") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"activation_steepness\" inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else
    {
        ref_ifstream_received >> *ptr_AF_received->ptr_activation_steepness >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
    }
    // |END| Activation steepness. |END|

    // Activation function.
    if((ref_ifstream_received >> tmp_line) && tmp_line.find("activation_function") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"activation_function\" inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else
    {
        ref_ifstream_received >> tmp_input_integer >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }

        if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH))
        {
            PRINT_FORMAT("%s: %s: ERROR: Undefined activation function type %zu. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_input_integer,
                                     __LINE__);

            return(false);
        }

        *ptr_AF_received->ptr_type_activation_function = static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(tmp_input_integer);
    }
    // |END| Activation function. |END|

    return(true);
}

bool Neural_Network::Load_Dimension__AF_Ind_Recurrent(struct AF_Ind_recurrent_unit *const ptr_AF_Ind_received, std::ifstream &ref_ifstream_received)
{
    size_t tmp_input_integer;

    std::string tmp_line;

    getline(ref_ifstream_received, tmp_line); // "AF_Ind_R[%zu]"
    
    // Activation steepness.
    if((ref_ifstream_received >> tmp_line) && tmp_line.find("activation_steepness") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"activation_steepness\" inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else
    {
        ref_ifstream_received >> *ptr_AF_Ind_received->ptr_activation_steepness >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
    }
    // |END| Activation steepness. |END|

    // Activation function.
    if((ref_ifstream_received >> tmp_line) && tmp_line.find("activation_function") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"activation_function\" inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else
    {
        ref_ifstream_received >> tmp_input_integer >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }

        if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH))
        {
            PRINT_FORMAT("%s: %s: ERROR: Undefined activation function type %zu. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_input_integer,
                                     __LINE__);

            return(false);
        }

        *ptr_AF_Ind_received->ptr_type_activation_function = static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(tmp_input_integer);
    }
    // |END| Activation function. |END|
    
    *ptr_AF_Ind_received->ptr_recurrent_connection_index = this->total_weights++;
    
    AF_Ind_recurrent_unit **tmp_ptr_array_U_ptr_connections(reinterpret_cast<AF_Ind_recurrent_unit **>(this->ptr_array_ptr_connections));
    
    if(this->Load_Dimension__Connection<struct AF_Ind_recurrent_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT>(*ptr_AF_Ind_received->ptr_recurrent_connection_index,
                                                                                                                                                                                                                                                                this->ptr_array_parameters,
                                                                                                                                                                                                                                                                this->ptr_array_AF_Ind_recurrent_units,
                                                                                                                                                                                                                                                                tmp_ptr_array_U_ptr_connections,
                                                                                                                                                                                                                                                                ref_ifstream_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 *ptr_AF_Ind_received->ptr_recurrent_connection_index,
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::Load_Dimension__Normalized_Unit(size_t const number_units_received,
                                                                                    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_normalization_received,
                                                                                    union Normalized_unit *const ptr_normalized_unit_received,
                                                                                    std::ifstream &ref_ifstream_received)
{
    size_t tmp_time_step_index,
              tmp_unit_timed_index;

    std::string tmp_line;
    
    getline(ref_ifstream_received, tmp_line); // "NormU[%zu]"
    
    switch(type_normalization_received)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION:
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION:
        case MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION:
            // Scale.
            if((ref_ifstream_received >> tmp_line) && tmp_line.find("scale") == std::string::npos)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not find \"scale\" inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
            else if(ref_ifstream_received.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
            else
            {
                ref_ifstream_received >> *ptr_normalized_unit_received->normalized_batch_units.ptr_scale >> std::ws;

                if(ref_ifstream_received.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_line.c_str(),
                                                __LINE__);

                    return(false);
                }
            }
            // |END| Scale. |END|
                
            // Shift.
            if((ref_ifstream_received >> tmp_line) && tmp_line.find("shift") == std::string::npos)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not find \"scale\" inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
            else if(ref_ifstream_received.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
            else
            {
                ref_ifstream_received >> *ptr_normalized_unit_received->normalized_batch_units.ptr_shift >> std::ws;

                if(ref_ifstream_received.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                tmp_line.c_str(),
                                                __LINE__);

                    return(false);
                }
            }
            // |END| Shift. |END|
                
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
            {
                tmp_unit_timed_index = number_units_received * tmp_time_step_index;
                
                // Mean average.
                if((ref_ifstream_received >> tmp_line) && tmp_line.find("mean_average[" + std::to_string(tmp_time_step_index) + "]") == std::string::npos)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not find \"mean_average[%zu]\" inside \"%s\". At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_time_step_index,
                                             tmp_line.c_str(),
                                             __LINE__);

                    return(false);
                }
                else if(ref_ifstream_received.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_line.c_str(),
                                             __LINE__);

                    return(false);
                }
                else
                {
                    ref_ifstream_received >> ptr_normalized_unit_received->normalized_batch_units.ptr_mean_average[tmp_unit_timed_index] >> std::ws;

                    if(ref_ifstream_received.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                }
                // |END| Mean average. |END|
                
                // Variance average.
                if((ref_ifstream_received >> tmp_line) && tmp_line.find("variance_average[" + std::to_string(tmp_time_step_index) + "]") == std::string::npos)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not find \"variance_average[%zu]\" inside \"%s\". At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_time_step_index,
                                             tmp_line.c_str(),
                                             __LINE__);

                    return(false);
                }
                else if(ref_ifstream_received.fail())
                {
                    PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_line.c_str(),
                                             __LINE__);

                    return(false);
                }
                else
                {
                    ref_ifstream_received >> ptr_normalized_unit_received->normalized_batch_units.ptr_variance_average[tmp_unit_timed_index] >> std::ws;

                    if(ref_ifstream_received.fail())
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_line.c_str(),
                                                 __LINE__);

                        return(false);
                    }
                }
                // |END| Variance average. |END|
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer normalization type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     type_normalization_received,
                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[type_normalization_received].c_str());
                break;
    }

    return(true);
}

bool Neural_Network::Load_Dimension__Bias(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received)
{
    size_t tmp_input_integer;

    std::string tmp_line;

    if((ref_ifstream_received >> tmp_line) && tmp_line.find("number_bias_parameters") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_bias_parameters\" inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);
                
        return(false);
    }
    else
    {
        ref_ifstream_received >> tmp_input_integer >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);
                    
            return(false);
        }
    }

    T_ *const tmp_ptr_array_parameters(this->ptr_array_parameters + this->total_weights_allocated + this->total_bias);

    ptr_layer_it_received->first_bias_connection_index = this->total_weights_allocated + this->total_bias;

    for(size_t tmp_connection_index(0_zu); tmp_connection_index != tmp_input_integer; ++tmp_connection_index)
    {
        if((ref_ifstream_received >> tmp_line) && tmp_line.find("weight") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"weight\" inside \"%s\". At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_line.c_str(),
                                        __LINE__);

            return(false);
        }
        else if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_line.c_str(),
                                        __LINE__);

            return(false);
        }
        else
        {
            ref_ifstream_received >> tmp_ptr_array_parameters[tmp_connection_index] >> std::ws;

            if(ref_ifstream_received.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
        }
    }

    this->total_bias += tmp_input_integer;

    ptr_layer_it_received->last_bias_connection_index = this->total_weights_allocated + this->total_bias;

    return(true);
}

bool Neural_Network::Load_Dimension__Block(size_t const layer_number_block_units_received,
                                                                     size_t const layer_number_cell_units_received,
                                                                     enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_normalization_received,
                                                                     struct Block_unit *const ptr_block_unit_it_received,
                                                                     std::ifstream &ref_ifstream_received)
{
    struct Cell_unit const *const tmp_ptr_block_ptr_cell_unit(ptr_block_unit_it_received->ptr_array_cell_units),
                                    *const tmp_ptr_block_ptr_last_cell_unit(ptr_block_unit_it_received->ptr_last_cell_unit);
    struct Cell_unit *tmp_ptr_block_ptr_cell_unit_it;
    
    size_t const tmp_block_number_cell_units(static_cast<size_t>(tmp_ptr_block_ptr_last_cell_unit - tmp_ptr_block_ptr_cell_unit));
    size_t tmp_input_integer;
    
    std::string tmp_line;

    getline(ref_ifstream_received, tmp_line); // "Block[%zu]"

    // Activation function.
    if((ref_ifstream_received >> tmp_line) && tmp_line.find("activation_function") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"activation_function\" inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else
    {
        ref_ifstream_received >> tmp_input_integer >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }

        if(tmp_input_integer >= static_cast<size_t>(MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH))
        {
            PRINT_FORMAT("%s: %s: ERROR: Undefined activation function type %zu. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_input_integer,
                                     __LINE__);

            return(false);
        }

        ptr_block_unit_it_received->activation_function_io = static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(tmp_input_integer);
    }
    // |END| Activation function. |END|

    // Activation steepness.
    if((ref_ifstream_received >> tmp_line) && tmp_line.find("activation_steepness") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"activation_steepness\" inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else
    {
        // TODO: Block unit activation steepness.
        ref_ifstream_received >> tmp_input_integer >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
    }
    // |END| Activation steepness. |END|
    
    // Number connection(s).
    if((ref_ifstream_received >> tmp_line) && tmp_line.find("number_connections") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_connections\" inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_line.c_str(),
                                 __LINE__);

        return(false);
    }
    else
    {
        ref_ifstream_received >> tmp_input_integer >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_line.c_str(),
                                     __LINE__);

            return(false);
        }
    }

#ifndef NO_PEEPHOLE
    size_t const tmp_number_inputs((tmp_input_integer - layer_number_cell_units_received * 4_zu - tmp_block_number_cell_units * 3_zu) / 4_zu);
#else
    size_t const tmp_number_inputs((tmp_input_integer - layer_number_cell_units_received * 4_zu) / 4_zu);
#endif

    ptr_block_unit_it_received->first_index_connection = this->total_weights;

    // [0] Cell input.
    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        //    [1] Input, cell.
        tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input = this->total_weights;
        this->total_weights += tmp_number_inputs;
        tmp_ptr_block_ptr_cell_unit_it->last_index_feedforward_connection_cell_input = this->total_weights;
        //    [1] |END| Input, cell. |END|

        //    [1] Recurrent, cell.
        tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input = this->total_weights;
        this->total_weights += layer_number_cell_units_received;
        tmp_ptr_block_ptr_cell_unit_it->last_index_recurrent_connection_cell_input = this->total_weights;
        //    [1] |END| Recurrent, cell. |END|
    }
    // [0] |END| Cell input. |END|
    
    // [0] Input, gates.
    //    [1] Input gate.
    ptr_block_unit_it_received->first_index_feedforward_connection_input_gate = this->total_weights;
    this->total_weights += tmp_number_inputs;
    ptr_block_unit_it_received->last_index_feedforward_connection_input_gate = this->total_weights;
    //    [1] |END| Input gate. |END|
    
    //    [1] Forget gate.
    ptr_block_unit_it_received->first_index_feedforward_connection_forget_gate = this->total_weights;
    this->total_weights += tmp_number_inputs;
    ptr_block_unit_it_received->last_index_feedforward_connection_forget_gate = this->total_weights;
    //    [1] |END| Forget gate. |END|

    //    [1] Output gate.
    ptr_block_unit_it_received->first_index_feedforward_connection_output_gate = this->total_weights;
    this->total_weights += tmp_number_inputs;
    ptr_block_unit_it_received->last_index_feedforward_connection_output_gate = this->total_weights;
    //    [1] |END| Output gate. |END|
    // [0] |END| Input, gates. |END|
    
    // [0] Recurrent, gates.
    //    [1] Input gate.
    ptr_block_unit_it_received->first_index_recurrent_connection_input_gate = this->total_weights;
    this->total_weights += layer_number_cell_units_received;
    ptr_block_unit_it_received->last_index_recurrent_connection_input_gate = this->total_weights;
    //    [1] |END| Input gate. |END|

    //    [1] Forget gate.
    ptr_block_unit_it_received->first_index_recurrent_connection_forget_gate = this->total_weights;
    this->total_weights += layer_number_cell_units_received;
    ptr_block_unit_it_received->last_index_recurrent_connection_forget_gate = this->total_weights;
    //    [1] |END| Forget gate. |END|

    //    [1] Output gate.
    ptr_block_unit_it_received->first_index_recurrent_connection_output_gate = this->total_weights;
    this->total_weights += layer_number_cell_units_received;
    ptr_block_unit_it_received->last_index_recurrent_connection_output_gate = this->total_weights;
    //    [1] |END| Output gate. |END|
    // [0] |END| Recurrent, gates. |END|

#ifndef NO_PEEPHOLE
    // [0] Peepholes.
    //    [1] Input gate.
    ptr_block_unit_it_received->first_index_peephole_input_gate = this->total_weights;
    this->total_weights += tmp_block_number_cell_units;
    ptr_block_unit_it_received->last_index_peephole_input_gate = this->total_weights;
    //    [1] |END| Input gate. |END|

    //    [1] Forget gate.
    ptr_block_unit_it_received->first_index_peephole_forget_gate = this->total_weights;
    this->total_weights += tmp_block_number_cell_units;
    ptr_block_unit_it_received->last_index_peephole_forget_gate = this->total_weights;
    //    [1] |END| Forget gate. |END|

    //    [1] Output gate.
    ptr_block_unit_it_received->first_index_peephole_output_gate = this->total_weights;
    this->total_weights += tmp_block_number_cell_units;
    ptr_block_unit_it_received->last_index_peephole_output_gate = this->total_weights;
    //    [1] |END| Output gate. |END|

    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        tmp_ptr_block_ptr_cell_unit_it->index_peephole_input_gate = ptr_block_unit_it_received->first_index_peephole_input_gate + static_cast<size_t>(tmp_ptr_block_ptr_cell_unit_it - tmp_ptr_block_ptr_cell_unit);
        tmp_ptr_block_ptr_cell_unit_it->index_peephole_forget_gate = ptr_block_unit_it_received->first_index_peephole_forget_gate + static_cast<size_t>(tmp_ptr_block_ptr_cell_unit_it - tmp_ptr_block_ptr_cell_unit);
        tmp_ptr_block_ptr_cell_unit_it->index_peephole_output_gate = ptr_block_unit_it_received->first_index_peephole_output_gate + static_cast<size_t>(tmp_ptr_block_ptr_cell_unit_it - tmp_ptr_block_ptr_cell_unit);
    }
    // [0] |END| Peepholes. |END|
#endif

    ptr_block_unit_it_received->last_index_connection = this->total_weights;
    // |END| Number connection(s). |END|

    return(true);
}

bool Neural_Network::Load_Dimension__Cell_Units(struct Layer *const ptr_layer_it_received,
                                                                           struct Cell_unit *&ptr_reference_array_cells_received,
                                                                           std::ifstream &ref_ifstream_received)
{
    size_t tmp_input_integer;

    std::string tmp_line;

    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);
    
    // Number cell unit(s).
    if((ref_ifstream_received >> tmp_line) && tmp_line.find("number_cell_units") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"number_cell_units\" inside \"%s\". At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    tmp_line.c_str(),
                                    __LINE__);

        return(false);
    }
    else if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    tmp_line.c_str(),
                                    __LINE__);

        return(false);
    }
    else
    {
        ref_ifstream_received >> tmp_input_integer >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_line.c_str(),
                                        __LINE__);

            return(false);
        }
    }

    size_t const tmp_layer_number_block_units(static_cast<size_t>(tmp_ptr_last_block_unit - tmp_ptr_block_unit_it)),
                       tmp_block_number_cell_units(tmp_input_integer / tmp_layer_number_block_units);
    // |END| Number cell unit(s). |END|

    ptr_layer_it_received->ptr_array_cell_units = ptr_reference_array_cells_received;

    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        tmp_ptr_block_unit_it->ptr_array_cell_units = ptr_reference_array_cells_received;
        ptr_reference_array_cells_received += tmp_block_number_cell_units;
        tmp_ptr_block_unit_it->ptr_last_cell_unit = ptr_reference_array_cells_received;
    }

    ptr_layer_it_received->ptr_last_cell_unit = ptr_reference_array_cells_received;

    return(true);
}

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> inline bool Neural_Network::Load_Dimension__Connection(size_t index_received,
                                                                                                                                                                                                         T_ *const ptr_array_parameters_received,
                                                                                                                                                                                                         U *const ptr_first_U_unit_received,
                                                                                                                                                                                                         U **ptr_array_ptr_U_unit_connection_received,
                                                                                                                                                                                                         std::ifstream &ref_ifstream_received)
{
    size_t tmp_input_integer;

    std::string tmp_line;

    ref_ifstream_received >> tmp_line; // "connected_to_%s=%u"

    if(ref_ifstream_received.fail())
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    tmp_line.c_str(),
                                    __LINE__);

        return(false);
    }

    if(tmp_line.find(MyEA::Common::ENUM_TYPE_LAYER_CONNECTION_NAME[E]) != std::string::npos) // If is "connected_to_neuron".
    {
        ref_ifstream_received >> tmp_input_integer >> std::ws;

        if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_line.c_str(),
                                        __LINE__);

            return(false);
        }

        ptr_array_ptr_U_unit_connection_received[index_received] = ptr_first_U_unit_received + tmp_input_integer;
            
        if((ref_ifstream_received >> tmp_line) && tmp_line.find("weight") == std::string::npos)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not find \"weight\" inside \"%s\". At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_line.c_str(),
                                        __LINE__);

            return(false);
        }
        else if(ref_ifstream_received.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read properly inside \"%s\". At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_line.c_str(),
                                        __LINE__);

            return(false);
        }
        else
        {
            ref_ifstream_received >> ptr_array_parameters_received[index_received] >> std::ws;

            if(ref_ifstream_received.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not read input of \"%s\". At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            tmp_line.c_str(),
                                            __LINE__);

                return(false);
            }
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \"connected_to_neuron\" inside \"%s\". At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    tmp_line.c_str(),
                                    __LINE__);

        return(false);
    }

    return(true);
}

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Neural_Network::Load_Dimension__Neuron__Forward__Connection(struct Neuron_unit *const ptr_neuron_received,
                                                                                                                                                                                                                                U *const ptr_first_U_unit_received,
                                                                                                                                                                                                                                std::ifstream &ref_ifstream_received)
{
    size_t const tmp_number_connections(*ptr_neuron_received->ptr_number_connections);
    size_t tmp_connection_index;

    T_ *const tmp_ptr_array_parameters(this->ptr_array_parameters + *ptr_neuron_received->ptr_first_connection_index);
    
    U **tmp_ptr_array_U_ptr_connections(reinterpret_cast<U **>(this->ptr_array_ptr_connections + *ptr_neuron_received->ptr_first_connection_index));
    
    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_connections; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<U, E>(tmp_connection_index,
                                                                          tmp_ptr_array_parameters,
                                                                          ptr_first_U_unit_received,
                                                                          tmp_ptr_array_U_ptr_connections,
                                                                          ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_connection_index,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Neural_Network::Load_Dimension__Block__Connection(struct Block_unit *const ptr_block_unit_it_received,
                                                                                                                                                                                                             U *const ptr_first_U_unit_received,
                                                                                                                                                                                                             std::ifstream &ref_ifstream_received)
{
    size_t const tmp_number_inputs_connections(ptr_block_unit_it_received->last_index_feedforward_connection_input_gate - ptr_block_unit_it_received->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrents_connection(ptr_block_unit_it_received->last_index_recurrent_connection_input_gate - ptr_block_unit_it_received->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index;
    
    T_ *tmp_ptr_array_parameters;

    struct Cell_unit const *const tmp_ptr_block_ptr_last_cell_unit(ptr_block_unit_it_received->ptr_last_cell_unit);
    struct Cell_unit *tmp_ptr_block_ptr_cell_unit_it,
                           **tmp_ptr_array_ptr_connections_layer_cell_units;
    
#ifndef NO_PEEPHOLE
    struct Cell_unit **tmp_ptr_array_ptr_connections_peepholes_cell_units(reinterpret_cast<struct Cell_unit **>(this->ptr_array_ptr_connections));
#endif

    U **tmp_ptr_array_U_ptr_connections;
    
    // [0] Cell input.
    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        //    [1] Input, cell input.
        tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(this->ptr_array_ptr_connections + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input);

        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
        {
            if(this->Load_Dimension__Connection<U, E>(tmp_connection_index,
                                                                              tmp_ptr_array_parameters,
                                                                              ptr_first_U_unit_received,
                                                                              tmp_ptr_array_U_ptr_connections,
                                                                              ref_ifstream_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_connection_index,
                                         __LINE__);

                return(false);
            }
        }
        //    [1] |END| Input, cell input. |END|
        
        //    [1] Recurrent, cell input.
        tmp_ptr_array_ptr_connections_layer_cell_units = reinterpret_cast<struct Cell_unit **>(this->ptr_array_ptr_connections + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input);

        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;
        
        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
        {
            if(this->Load_Dimension__Connection<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_connection_index,
                                                                                                                                                                                             tmp_ptr_array_parameters,
                                                                                                                                                                                             this->ptr_array_cell_units,
                                                                                                                                                                                             tmp_ptr_array_ptr_connections_layer_cell_units,
                                                                                                                                                                                             ref_ifstream_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_connection_index,
                                         __LINE__);

                return(false);
            }
        }
        //    [1] |END| Recurrent, cell input. |END|
    }
    // [0] |END| Cell input. |END|

    // [0] Input, Gates.
    //  [1] Input gate.
    tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_feedforward_connection_input_gate);

    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_feedforward_connection_input_gate;
    
    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<U, E>(tmp_connection_index,
                                                                          tmp_ptr_array_parameters,
                                                                          ptr_first_U_unit_received,
                                                                          tmp_ptr_array_U_ptr_connections,
                                                                          ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_connection_index,
                                     __LINE__);

            return(false);
        }
    }
    //  [1] |END| Input gate. |END|

    //  [1] Forget gate.
    tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_feedforward_connection_forget_gate);
    
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_feedforward_connection_forget_gate;
    
    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<U, E>(tmp_connection_index,
                                                                          tmp_ptr_array_parameters,
                                                                          ptr_first_U_unit_received,
                                                                          tmp_ptr_array_U_ptr_connections,
                                                                          ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_connection_index,
                                     __LINE__);

            return(false);
        }
    }
    //  [1] |END| Forget gate. |END|

    //  [1] Output gate.
    tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_feedforward_connection_output_gate);
    
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_feedforward_connection_output_gate;
    
    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<U, E>(tmp_connection_index,
                                                                          tmp_ptr_array_parameters,
                                                                          ptr_first_U_unit_received,
                                                                          tmp_ptr_array_U_ptr_connections,
                                                                          ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_connection_index,
                                     __LINE__);

            return(false);
        }
    }
    //  [1] |END| Output gate. |END|
    // [0] |END| Input, Gates. |END|
    
    // [0] Recurrent, Gates.
    //  [1] Input gate.
    tmp_ptr_array_ptr_connections_layer_cell_units = reinterpret_cast<struct Cell_unit **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_recurrent_connection_input_gate);

    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_recurrent_connection_input_gate;
    
    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_connection_index,
                                                                                                                                                                                         tmp_ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_layer_cell_units,
                                                                                                                                                                                         ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_connection_index,
                                     __LINE__);

            return(false);
        }
    }
    //  [1] |END| Input gate. |END|
    
    //  [1] Forget gate.
    tmp_ptr_array_ptr_connections_layer_cell_units = reinterpret_cast<struct Cell_unit **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_recurrent_connection_forget_gate);
    
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_recurrent_connection_forget_gate;
    
    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_connection_index,
                                                                                                                                                                                         tmp_ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_layer_cell_units,
                                                                                                                                                                                         ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_connection_index,
                                     __LINE__);

            return(false);
        }
    }
    //  [1] |END| Forget gate. |END|
    
    //  [1] Output gate.
    tmp_ptr_array_ptr_connections_layer_cell_units = reinterpret_cast<struct Cell_unit **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_recurrent_connection_output_gate);
    
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_recurrent_connection_output_gate;
    
    for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_connection_index,
                                                                                                                                                                                         tmp_ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_layer_cell_units,
                                                                                                                                                                                         ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_connection_index,
                                     __LINE__);

            return(false);
        }
    }
    //  [1] |END| Output gate. |END|
    // [0] |END| Recurrent, Gates. |END|

#ifndef NO_PEEPHOLE
    // [0] Peepholes.
    //  [1] Input gate.
    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        if(this->Load_Dimension__Connection<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_ptr_block_ptr_cell_unit_it->index_peephole_input_gate,
                                                                                                                                                                                         this->ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_peepholes_cell_units,
                                                                                                                                                                                         ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_block_ptr_cell_unit_it->index_peephole_input_gate,
                                     __LINE__);

            return(false);
        }
    }

    //  [1] Forget gate.
    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        if(this->Load_Dimension__Connection<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_ptr_block_ptr_cell_unit_it->index_peephole_forget_gate,
                                                                                                                                                                                         this->ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_peepholes_cell_units,
                                                                                                                                                                                         ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_block_ptr_cell_unit_it->index_peephole_forget_gate,
                                     __LINE__);

            return(false);
        }
    }

    //  [1] Output gate.
    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        if(this->Load_Dimension__Connection<struct Cell_unit, MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM>(tmp_ptr_block_ptr_cell_unit_it->index_peephole_output_gate,
                                                                                                                                                                                         this->ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_peepholes_cell_units,
                                                                                                                                                                                         ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_ptr_block_ptr_cell_unit_it->index_peephole_output_gate,
                                     __LINE__);

            return(false);
        }
    }
    // [0] |END| Peepholes. |END|
#endif

    return(true);
}

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Neural_Network::Load_Dimension__FC(struct Layer *const ptr_layer_it_received,
                                                                                                                                                                                     U *const ptr_first_U_unit_received,
                                                                                                                                                                                     std::ifstream &ref_ifstream_received)
{
    struct Neuron_unit const *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit);
    struct Neuron_unit *tmp_ptr_neuron_unit_it;
    
    *ptr_layer_it_received->ptr_first_connection_index = this->total_weights;

    // Forward connection.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_received->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
    {
        if(this->Load_Dimension__Neuron(tmp_ptr_neuron_unit_it, ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Neuron()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        else if(this->Load_Dimension__Neuron__Forward__Connection<U, E>(tmp_ptr_neuron_unit_it,
                                                                                                              ptr_first_U_unit_received,
                                                                                                              ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Neuron__Forward__Connection()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    *ptr_layer_it_received->ptr_last_connection_index = this->total_weights;

    return(true);
}

bool Neural_Network::Load_Dimension__AF(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received)
{
    struct AF_unit const *const tmp_ptr_last_AF_unit(ptr_layer_it_received->ptr_last_AF_unit);
    struct AF_unit *tmp_ptr_AF_unit_it(ptr_layer_it_received->ptr_array_AF_units);
    
    for(; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it)
    {
        if(this->Load_Dimension__AF(tmp_ptr_AF_unit_it, ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__AF_Unit(\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

bool Neural_Network::Load_Dimension__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit);
    struct AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    for(; tmp_ptr_AF_Ind_recurrent_unit_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_AF_Ind_recurrent_unit_it)
    {
        if(this->Load_Dimension__AF_Ind_Recurrent(tmp_ptr_AF_Ind_recurrent_unit_it, ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__AF_Ind_Recurrent(\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Neural_Network::Load_Dimension__LSTM(struct Layer *const ptr_layer_it_received,
                                                                                                                                                                                         U *const ptr_first_U_unit_received,
                                                                                                                                                                                         std::ifstream &ref_ifstream_received)
{
    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);
    
    size_t const tmp_layer_number_block_units(static_cast<size_t>(tmp_ptr_last_block_unit - tmp_ptr_block_unit_it)),
                       tmp_layer_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units));
    
    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const tmp_type_layer_normalization(ptr_layer_it_received->type_normalization);
    
    *ptr_layer_it_received->ptr_first_connection_index = this->total_weights;

    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        if(this->Load_Dimension__Block(tmp_layer_number_block_units,
                                                        tmp_layer_number_cell_units >> static_cast<size_t>(ptr_layer_it_received->Use__Bidirectional()),
                                                        tmp_type_layer_normalization,
                                                        tmp_ptr_block_unit_it,
                                                        ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Block(%zu, %zu, %u, ptr, ref)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_layer_number_block_units,
                                     tmp_layer_number_cell_units >> static_cast<size_t>(ptr_layer_it_received->Use__Bidirectional()),
                                     tmp_type_layer_normalization,
                                     __LINE__);

            return(false);
        }
        else if(this->Load_Dimension__Block__Connection<U, E>(tmp_ptr_block_unit_it,
                                                                                             ptr_first_U_unit_received,
                                                                                             ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__LSTM__Connection()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    *ptr_layer_it_received->ptr_last_connection_index = this->total_weights;

    return(true);
}

bool Neural_Network::Load_Dimension__Normalization(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received)
{
    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const tmp_type_layer_normalization(ptr_layer_it_received->type_normalization);

    union Normalized_unit const *const tmp_ptr_last_normalized_unit(ptr_layer_it_received->ptr_last_normalized_unit);
    union Normalized_unit *tmp_ptr_normalized_unit_it(ptr_layer_it_received->ptr_array_normalized_units);
    
    size_t const tmp_number_normalized_units(static_cast<size_t>(tmp_ptr_last_normalized_unit - tmp_ptr_normalized_unit_it));
    
    for(; tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit; ++tmp_ptr_normalized_unit_it)
    {
        if(this->Load_Dimension__Normalized_Unit(tmp_number_normalized_units,
                                                                       tmp_type_layer_normalization,
                                                                       tmp_ptr_normalized_unit_it,
                                                                       ref_ifstream_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Load_Dimension__Normalized_Unit(%zu, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_number_normalized_units,
                                     tmp_type_layer_normalization,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}
#endif
