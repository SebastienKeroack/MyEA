#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::Dropout_Zoneout(void)
{
    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        #pragma omp parallel
        this->Dropout_Zoneout__OpenMP();
    }
    else
    { this->Dropout_Zoneout__Loop(); }
}

void Neural_Network::Dropout_Zoneout__Loop(void)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT)
        {
            switch(tmp_ptr_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Dropout_Zoneout__Block_Units__Loop(tmp_ptr_layer_it); break;
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

void Neural_Network::Dropout_Zoneout__Block_Units__Loop(struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units));
    size_t tmp_time_step_index,
              tmp_timed_mask_index;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    this->ptr_array_Class_Generator_Bernoulli_Zoneout_State->Probability(ptr_layer_it_received->dropout_values[0u]);
    this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden->Probability(ptr_layer_it_received->dropout_values[1u]);

    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
    {
        tmp_timed_mask_index = tmp_time_step_index * tmp_number_cell_units;

        for(tmp_ptr_last_cell_unit = ptr_layer_it_received->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = ptr_layer_it_received->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
        {
            if((*this->ptr_array_Class_Generator_Bernoulli_Zoneout_State)()) // Zoneout cell state.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state[tmp_timed_mask_index] = false; }
            else // Keep cell state.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state[tmp_timed_mask_index] = true; }

            if((*this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden)()) // Zoneout cell output.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output[tmp_timed_mask_index] = false; }
            else // Keep cell output.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output[tmp_timed_mask_index] = true; }
        }
    }
}
