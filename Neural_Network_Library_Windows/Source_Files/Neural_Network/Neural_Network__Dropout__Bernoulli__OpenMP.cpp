#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::Dropout_Bernoulli__OpenMP(void)
{
    size_t tmp_number_outputs;

    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

    // Input layer.
    if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI) { this->Dropout_Bernoulli__Layer__OpenMP(this->number_inputs, tmp_ptr_layer_it); }
    
    for(++tmp_ptr_layer_it; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI)
        {
            switch(tmp_ptr_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: tmp_number_outputs = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_unit - tmp_ptr_layer_it->ptr_array_AF_units); break;
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: tmp_number_outputs = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units); break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_layer_it->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str());
                        return;
            }

            this->Dropout_Bernoulli__Layer__OpenMP(tmp_number_outputs, tmp_ptr_layer_it);
        }
    }
}

void Neural_Network::Dropout_Bernoulli__Layer__OpenMP(size_t const number_outputs_received, struct Layer *const ptr_layer_it_received)
{
    T_ const tmp_retained_probability(ptr_layer_it_received->dropout_values[0u]);

    if(tmp_retained_probability != 0_T)
    {
        int const tmp_number_recurrent_depth__int(static_cast<int>(this->number_recurrent_depth));
        int tmp_time_step__int,
            tmp_thread_index__int;

        size_t tmp_unit_index,
                  tmp_timed_mask_index;
        
        this->ptr_array_Class_Generator_Bernoulli[omp_get_thread_num()].Probability(tmp_retained_probability);

        #pragma omp for schedule(static)
        for(tmp_time_step__int = 0; tmp_time_step__int < tmp_number_recurrent_depth__int; ++tmp_time_step__int)
        {
            tmp_thread_index__int = omp_get_thread_num();

            tmp_timed_mask_index = static_cast<size_t>(tmp_time_step__int) * number_outputs_received;

            for(tmp_unit_index = 0_zu; tmp_unit_index != number_outputs_received; ++tmp_unit_index)
            {
                if(this->ptr_array_Class_Generator_Bernoulli[tmp_thread_index__int]()) // Keep unit.
                { ptr_layer_it_received->ptr_array__mask__dropout__bernoulli[tmp_timed_mask_index + tmp_unit_index] = true; }
                else // Drop unit.
                { ptr_layer_it_received->ptr_array__mask__dropout__bernoulli[tmp_timed_mask_index + tmp_unit_index] = false; }
            }
        }
    }
    else
    {
        #pragma omp single
        MyEA::Memory::Fill<bool>(ptr_layer_it_received->ptr_array__mask__dropout__bernoulli,
                                     ptr_layer_it_received->ptr_array__mask__dropout__bernoulli + number_outputs_received * this->number_recurrent_depth,
                                     false);
    }
}