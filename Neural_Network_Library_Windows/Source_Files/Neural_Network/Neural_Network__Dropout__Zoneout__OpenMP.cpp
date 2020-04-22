#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::Dropout_Zoneout__OpenMP(void)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT)
        {
            switch(tmp_ptr_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Dropout_Zoneout__Block_Units__OpenMP(tmp_ptr_layer_it); break;
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

void Neural_Network::Dropout_Zoneout__Block_Units__OpenMP(struct Layer *const ptr_layer_it_received)
{
    int const tmp_number_recurrent_depth__int(static_cast<int>(this->number_recurrent_depth));
    int tmp_time_step__int,
        tmp_thread_index__int;

    size_t const tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units));
    size_t tmp_timed_mask_index;

    struct Cell_unit const *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;
    
    this->ptr_array_Class_Generator_Bernoulli_Zoneout_State[omp_get_thread_num()].Probability(ptr_layer_it_received->dropout_values[0u]);
    this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden[omp_get_thread_num()].Probability(ptr_layer_it_received->dropout_values[1u]);
    
    #pragma omp for schedule(static)
    for(tmp_time_step__int = 0; tmp_time_step__int < tmp_number_recurrent_depth__int; ++tmp_time_step__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        tmp_timed_mask_index = static_cast<size_t>(tmp_time_step__int) * tmp_number_cell_units;

        for(tmp_ptr_last_cell_unit = ptr_layer_it_received->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = ptr_layer_it_received->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
        {
            if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State[tmp_thread_index__int]()) // Zoneout cell state.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state[tmp_timed_mask_index] = false; }
            else // Keep cell state.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state[tmp_timed_mask_index] = true; }

            if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden[tmp_thread_index__int]()) // Zoneout cell output.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output[tmp_timed_mask_index] = false; }
            else // Keep cell output.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output[tmp_timed_mask_index] = true; }
        }
    }
}
