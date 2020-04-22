#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

bool Neural_Network::Reallocate__Batch(size_t const batch_size_received)
{
    if(this->Reallocate__Batch__Basic_Unit(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Batch__Basic_Unit(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 batch_size_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Reallocate__Batch__Basic_Indice_Unit(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Batch__Basic_Indice_Unit(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 batch_size_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Reallocate__Batch__Neuron_Unit(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Batch__Neuron_Unit(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 batch_size_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Reallocate__Batch__AF_Unit(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Batch__AF_Unit(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 batch_size_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Reallocate__Batch__AF_Ind_Recurrent_Unit(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Batch__AF_Ind_Recurrent_Unit(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 batch_size_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Reallocate__Normalized_Unit__Batch_Normalization(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Normalized_Unit__Batch_Normalization(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 batch_size_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Reallocate__Batch__LSTM(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Batch__LSTM(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 batch_size_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Use__Dropout__ShakeDrop() && this->Reallocate__Batch__Dropout__ShakeDrop(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Batch__Dropout__ShakeDrop(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 batch_size_received,
                                 __LINE__);

        return(false);
    }
    
    this->Order__Layers__Output();

    return(true);
}

bool Neural_Network::Reallocate__Batch__Basic_Unit(size_t const batch_size_received)
{
    if(this->total_basic_units_allocated != 0_zu)
    {
        size_t tmp_number_basic_units;

        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct Basic_unit const *tmp_ptr_last_basic_unit;
        struct Basic_unit *tmp_ptr_basic_unit_it;

        // Allocating basic unit(s) value.
        T_ *tmp_ptr_array_basic_units_values(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_basic_units_values,
                                                                                                           batch_size_received * this->total_basic_units_allocated * this->number_recurrent_depth,
                                                                                                           this->batch_size * this->total_basic_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_basic_units_values == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_basic_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_basic_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_units_values = tmp_ptr_array_basic_units_values;
        // |END| Allocating basic unit(s) value. |END|

        // Allocating basic unit(s) error.
        T_ *tmp_ptr_array_basic_units_errors(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_basic_units_errors,
                                                                                                          batch_size_received * this->total_basic_units_allocated * this->number_recurrent_depth,
                                                                                                          this->batch_size * this->total_basic_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_basic_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_basic_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_basic_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_units_errors = tmp_ptr_array_basic_units_errors;
        // |END| Allocating basic unit(s) error. |END|

        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_basic_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_basic_unit - tmp_ptr_layer_it->ptr_array_basic_units);

            if(tmp_number_basic_units != 0_zu)
            {
                for(tmp_ptr_last_basic_unit = tmp_ptr_layer_it->ptr_last_basic_unit,
                    tmp_ptr_basic_unit_it = tmp_ptr_layer_it->ptr_array_basic_units; tmp_ptr_basic_unit_it != tmp_ptr_last_basic_unit; ++tmp_ptr_basic_unit_it,
                                                                                                                                                                                           ++tmp_ptr_array_basic_units_values,
                                                                                                                                                                                           ++tmp_ptr_array_basic_units_errors)
                {
                    tmp_ptr_basic_unit_it->ptr_array_values = tmp_ptr_array_basic_units_values;
                    tmp_ptr_basic_unit_it->ptr_array_errors = tmp_ptr_array_basic_units_errors;
                }
                
                tmp_ptr_array_basic_units_values += (batch_size_received - 1_zu) * tmp_number_basic_units * this->number_recurrent_depth + tmp_number_basic_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_basic_units_errors += (batch_size_received - 1_zu) * tmp_number_basic_units * this->number_recurrent_depth + tmp_number_basic_units * (this->number_recurrent_depth - 1_zu);
            }
        }
    }

    return(true);
}

bool Neural_Network::Reallocate__Batch__Basic_Indice_Unit(size_t const batch_size_received)
{
    if(this->total_basic_indice_units_allocated != 0_zu)
    {
        size_t tmp_number_basic_indice_units;

        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct Basic_indice_unit const *tmp_ptr_last_basic_indice_unit;
        struct Basic_indice_unit *tmp_ptr_basic_indice_unit_it;
        
        // Allocating basic unit(s) indice.
        size_t *tmp_ptr_array_basic_indice_units_indices(MyEA::Memory::Cpp::Reallocate<size_t, false>(this->ptr_array_basic_indice_units_indices,
                                                                                                                                 batch_size_received * this->total_basic_indice_units_allocated * this->number_recurrent_depth,
                                                                                                                                 this->batch_size * this->total_basic_indice_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_basic_indice_units_indices == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(size_t),
                                     batch_size_received * this->total_basic_indice_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_basic_indice_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_indice_units_indices = tmp_ptr_array_basic_indice_units_indices;
        // |END| Allocating basic unit(s) indice. |END|

        // Allocating basic unit(s) value.
        T_ *tmp_ptr_array_basic_indice_units_values(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_basic_indice_units_values,
                                                                                                                     batch_size_received * this->total_basic_indice_units_allocated * this->number_recurrent_depth,
                                                                                                                     this->batch_size * this->total_basic_indice_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_basic_indice_units_values == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_basic_indice_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_basic_indice_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_indice_units_values = tmp_ptr_array_basic_indice_units_values;
        // |END| Allocating basic unit(s) value. |END|

        // Allocating basic unit(s) error.
        T_ *tmp_ptr_array_basic_indice_units_errors(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_basic_indice_units_errors,
                                                                                                                    batch_size_received * this->total_basic_indice_units_allocated * this->number_recurrent_depth,
                                                                                                                    this->batch_size * this->total_basic_indice_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_basic_indice_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_basic_indice_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_basic_indice_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_indice_units_errors = tmp_ptr_array_basic_indice_units_errors;
        // |END| Allocating basic unit(s) error. |END|

        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_basic_indice_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_basic_indice_unit - tmp_ptr_layer_it->ptr_array_basic_indice_units);

            if(tmp_number_basic_indice_units != 0_zu)
            {
                for(tmp_ptr_last_basic_indice_unit = tmp_ptr_layer_it->ptr_last_basic_indice_unit,
                    tmp_ptr_basic_indice_unit_it = tmp_ptr_layer_it->ptr_array_basic_indice_units; tmp_ptr_basic_indice_unit_it != tmp_ptr_last_basic_indice_unit; ++tmp_ptr_basic_indice_unit_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_basic_indice_units_indices,
                                                                                                                                                                                                                                   ++tmp_ptr_array_basic_indice_units_values,
                                                                                                                                                                                                                                   ++tmp_ptr_array_basic_indice_units_errors)
                {
                    tmp_ptr_basic_indice_unit_it->ptr_array_indices = tmp_ptr_array_basic_indice_units_indices;

                    tmp_ptr_basic_indice_unit_it->ptr_array_values = tmp_ptr_array_basic_indice_units_values;
                    tmp_ptr_basic_indice_unit_it->ptr_array_errors = tmp_ptr_array_basic_indice_units_errors;
                }
                
                tmp_ptr_array_basic_indice_units_indices += (batch_size_received - 1_zu) * tmp_number_basic_indice_units * this->number_recurrent_depth + tmp_number_basic_indice_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_basic_indice_units_values += (batch_size_received - 1_zu) * tmp_number_basic_indice_units * this->number_recurrent_depth + tmp_number_basic_indice_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_basic_indice_units_errors += (batch_size_received - 1_zu) * tmp_number_basic_indice_units * this->number_recurrent_depth + tmp_number_basic_indice_units * (this->number_recurrent_depth - 1_zu);
            }
        }
    }

    return(true);
}

bool Neural_Network::Reallocate__Batch__Neuron_Unit(size_t const batch_size_received)
{
    if(this->total_neuron_units_allocated != 0_zu)
    {
        size_t tmp_number_neuron_units;

        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct Neuron_unit const *tmp_ptr_last_neuron_unit;
        struct Neuron_unit *tmp_ptr_neuron_unit_it;

        // Allocating neuron unit(s) summation(s).
        T_ *tmp_ptr_array_neuron_units_summations(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_neuron_units_summations,
                                                                                                                      batch_size_received * this->total_neuron_units_allocated * this->number_recurrent_depth,
                                                                                                                      this->batch_size * this->total_neuron_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_neuron_units_summations == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_neuron_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_neuron_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_neuron_units_summations = tmp_ptr_array_neuron_units_summations;
        // |END| Allocating neuron unit(s) summation(s). |END|

        // Allocating neuron unit(s) error(s).
        T_ *tmp_ptr_array_neuron_units_errors(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_neuron_units_errors,
                                                                                                            batch_size_received * this->total_neuron_units_allocated * this->number_recurrent_depth,
                                                                                                            this->batch_size * this->total_neuron_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_neuron_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_neuron_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_neuron_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_neuron_units_errors = tmp_ptr_array_neuron_units_errors;
        // |END| Allocating neuron unit(s) error(s). |END|
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_neuron_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_neuron_unit - tmp_ptr_layer_it->ptr_array_neuron_units);

            if(tmp_number_neuron_units != 0_zu)
            {
                for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit,
                    tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                                                  ++tmp_ptr_array_neuron_units_summations,
                                                                                                                                                                                                  ++tmp_ptr_array_neuron_units_errors)
                {
                    tmp_ptr_neuron_unit_it->ptr_array_summations = tmp_ptr_array_neuron_units_summations;
                    tmp_ptr_neuron_unit_it->ptr_array_errors = tmp_ptr_array_neuron_units_errors;
                }
                
                tmp_ptr_array_neuron_units_summations += (batch_size_received - 1_zu) * tmp_number_neuron_units * this->number_recurrent_depth + tmp_number_neuron_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_neuron_units_errors += (batch_size_received - 1_zu) * tmp_number_neuron_units * this->number_recurrent_depth + tmp_number_neuron_units * (this->number_recurrent_depth - 1_zu);
            }
        }
    }

    return(true);
}

bool Neural_Network::Reallocate__Batch__AF_Unit(size_t const batch_size_received)
{
    if(this->total_AF_units_allocated != 0_zu)
    {
        size_t tmp_number_AF_units;

        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct AF_unit const *tmp_ptr_last_AF_unit;
        struct AF_unit *tmp_ptr_AF_unit_it;

        // Allocating AF unit(s) value(s).
        T_ *tmp_ptr_array_AF_units_values(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_AF_units_values,
                                                                                                      batch_size_received * this->total_AF_units_allocated * this->number_recurrent_depth,
                                                                                                      this->batch_size * this->total_AF_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_AF_units_values == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_AF_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_AF_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_units_values = tmp_ptr_array_AF_units_values;
        // |END| Allocating AF unit(s) value(s). |END|

        // Allocating AF unit(s) error(s).
        T_ *tmp_ptr_array_AF_units_errors(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_AF_units_errors,
                                                                                                     batch_size_received * this->total_AF_units_allocated * this->number_recurrent_depth,
                                                                                                     this->batch_size * this->total_AF_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_AF_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_AF_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_AF_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_units_errors = tmp_ptr_array_AF_units_errors;
        // |END| Allocating AF unit(s) error(s). |END|
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_AF_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_unit - tmp_ptr_layer_it->ptr_array_AF_units);

            if(tmp_number_AF_units != 0_zu)
            {
                for(tmp_ptr_last_AF_unit = tmp_ptr_layer_it->ptr_last_AF_unit,
                    tmp_ptr_AF_unit_it = tmp_ptr_layer_it->ptr_array_AF_units; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it,
                                                                                                                                                                      ++tmp_ptr_array_AF_units_values,
                                                                                                                                                                      ++tmp_ptr_array_AF_units_errors)
                {
                    tmp_ptr_AF_unit_it->ptr_array_values = tmp_ptr_array_AF_units_values;
                    tmp_ptr_AF_unit_it->ptr_array_errors = tmp_ptr_array_AF_units_errors;
                }
                
                tmp_ptr_array_AF_units_values += (batch_size_received - 1_zu) * tmp_number_AF_units * this->number_recurrent_depth + tmp_number_AF_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_AF_units_errors += (batch_size_received - 1_zu) * tmp_number_AF_units * this->number_recurrent_depth + tmp_number_AF_units * (this->number_recurrent_depth - 1_zu);
            }
        }
    }

    return(true);
}

bool Neural_Network::Reallocate__Batch__AF_Ind_Recurrent_Unit(size_t const batch_size_received)
{
    if(this->total_AF_Ind_recurrent_units_allocated != 0_zu)
    {
        size_t tmp_number_AF_Ind_recurrent_units;

        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct AF_Ind_recurrent_unit const *tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit;
        struct AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it;

        // Allocating af_ind unit(s) value(s).
        T_ *tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_AF_Ind_recurrent_units_pre_AFs,
                                                                                                            batch_size_received * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                                                                                            this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_pre_AFs = tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs;
        // |END| Allocating af_ind unit(s) value(s). |END|
        
        // Allocating af_ind unit(s) value(s).
        T_ *tmp_ptr_array_AF_Ind_recurrent_units_AFs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_AF_Ind_recurrent_units_AFs,
                                                                                                            batch_size_received * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                                                                                            this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_AF_Ind_recurrent_units_AFs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_AFs = tmp_ptr_array_AF_Ind_recurrent_units_AFs;
        // |END| Allocating af_ind unit(s) value(s). |END|

        // Allocating af_ind unit(s) error(s).
        T_ *tmp_ptr_array_AF_Ind_recurrent_units_errors(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_AF_Ind_recurrent_units_errors,
                                                                                                           batch_size_received * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                                                                                           this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_AF_Ind_recurrent_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_errors = tmp_ptr_array_AF_Ind_recurrent_units_errors;
        // |END| Allocating af_ind unit(s) error(s). |END|
        
        // Allocating af_ind unit(s) dAF_Ind_Recurrent(s).
        T_ *tmp_ptr_array_AF_Ind_recurrent_units_dAFs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_AF_Ind_recurrent_units_dAFs,
                                                                                                                 batch_size_received * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                                                                                                 this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_AF_Ind_recurrent_units_dAFs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_dAFs = tmp_ptr_array_AF_Ind_recurrent_units_dAFs;
        // |END| Allocating af_ind unit(s) dAF_Ind_Recurrent(s). |END|
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_AF_Ind_recurrent_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units);

            if(tmp_number_AF_Ind_recurrent_units != 0_zu)
            {
                for(tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit = tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit,
                    tmp_ptr_AF_Ind_recurrent_unit_it = tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units; tmp_ptr_AF_Ind_recurrent_unit_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_AF_Ind_recurrent_unit_it,
                                                                                                                                                                                              ++tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs,
                                                                                                                                                                                              ++tmp_ptr_array_AF_Ind_recurrent_units_AFs,
                                                                                                                                                                                              ++tmp_ptr_array_AF_Ind_recurrent_units_errors,
                                                                                                                                                                                              ++tmp_ptr_array_AF_Ind_recurrent_units_dAFs)
                {
                    tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_pre_AFs = tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs;
                    tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_AFs = tmp_ptr_array_AF_Ind_recurrent_units_AFs;
                    tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_errors = tmp_ptr_array_AF_Ind_recurrent_units_errors;
                    tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_dAFs = tmp_ptr_array_AF_Ind_recurrent_units_dAFs;
                }
                
                tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs += (batch_size_received - 1_zu) * tmp_number_AF_Ind_recurrent_units * this->number_recurrent_depth + tmp_number_AF_Ind_recurrent_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_AF_Ind_recurrent_units_AFs += (batch_size_received - 1_zu) * tmp_number_AF_Ind_recurrent_units * this->number_recurrent_depth + tmp_number_AF_Ind_recurrent_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_AF_Ind_recurrent_units_errors += (batch_size_received - 1_zu) * tmp_number_AF_Ind_recurrent_units * this->number_recurrent_depth + tmp_number_AF_Ind_recurrent_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_AF_Ind_recurrent_units_dAFs += (batch_size_received - 1_zu) * tmp_number_AF_Ind_recurrent_units * this->number_recurrent_depth + tmp_number_AF_Ind_recurrent_units * (this->number_recurrent_depth - 1_zu);
            }
        }
    }

    return(true);
}

bool Neural_Network::Reallocate__Normalized_Unit__Batch_Normalization(size_t const batch_size_received)
{
    if(this->Use__Normalization()
       &&
       this->ptr_array_normalized_batch_units_values_hats != nullptr
       &&
       this->ptr_array_normalized_batch_units_values_normalizes != nullptr
       &&
       this->ptr_array_normalized_batch_units_errors != nullptr)
    {
        size_t tmp_number_units,
                  tmp_index;

        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
        
        struct Block_unit const *tmp_ptr_last_block_unit;
        struct Block_unit *tmp_ptr_block_unit_it;
        
        struct Cell_unit const *tmp_ptr_last_cell_unit;
        struct Cell_unit *tmp_ptr_cell_unit_it;
        
        union Normalized_unit const *tmp_ptr_last_normalized_unit;
        union Normalized_unit *tmp_ptr_normalized_unit_it;
        
        // Allocating normalized unit(s) value(s) hat.
        T_ *tmp_ptr_array_normalized_units_values_hat(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_normalized_batch_units_values_hats,
                                                                                                                         batch_size_received * this->total_normalized_units_allocated * this->number_recurrent_depth,
                                                                                                                         this->batch_size * this->total_normalized_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_normalized_units_values_hat == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_normalized_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_normalized_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_normalized_batch_units_values_hats = tmp_ptr_array_normalized_units_values_hat;
        // |END| Allocating normalized unit(s) value(s) hat. |END|

        // Allocating normalized unit(s) value(s) normalize.
        T_ *tmp_ptr_array_normalized_units_values_normalize(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_normalized_batch_units_values_normalizes,
                                                                                                                                   batch_size_received * this->total_normalized_units_allocated * this->number_recurrent_depth,
                                                                                                                                   this->batch_size * this->total_normalized_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_normalized_units_values_normalize == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_normalized_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_normalized_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_normalized_batch_units_values_normalizes = tmp_ptr_array_normalized_units_values_normalize;
        // |END| Allocating normalized unit(s) value(s) normalize. |END|
        
        // Allocating normalized unit(s) error(s).
        T_ *tmp_ptr_array_normalized_units_errors(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_normalized_batch_units_errors,
                                                                                                                  batch_size_received * this->total_normalized_units_allocated * this->number_recurrent_depth,
                                                                                                                  this->batch_size * this->total_normalized_units_allocated * this->number_recurrent_depth));
        if(tmp_ptr_array_normalized_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_normalized_units_allocated * this->number_recurrent_depth,
                                     this->batch_size * this->total_normalized_units_allocated * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_normalized_batch_units_errors = tmp_ptr_array_normalized_units_errors;
        // |END| Allocating normalized unit(s) error(s). |END|

        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
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
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_values_hat,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_values_normalize,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_errors)
                        {
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_values_hats = tmp_ptr_array_normalized_units_values_hat;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_values_normalizes = tmp_ptr_array_normalized_units_values_normalize;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors;
                        }
                        
                        tmp_ptr_array_normalized_units_values_hat += (batch_size_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        tmp_ptr_array_normalized_units_values_normalize += (batch_size_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        tmp_ptr_array_normalized_units_errors += (batch_size_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
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
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_values_hats = tmp_ptr_array_normalized_units_values_hat++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_values_normalizes = tmp_ptr_array_normalized_units_values_normalize++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors++;
                                }
                            
                                tmp_ptr_array_normalized_units_values_hat += (batch_size_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_values_normalize += (batch_size_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_errors += (batch_size_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
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
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_values_hats = tmp_ptr_array_normalized_units_values_hat++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_values_normalizes = tmp_ptr_array_normalized_units_values_normalize++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors++;
                                }
                            
                                tmp_ptr_array_normalized_units_values_hat += (batch_size_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_values_normalize += (batch_size_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_errors += (batch_size_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                            }
                        } 
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                                 __LINE__);
                            return(false);
                }
            }
        }
    }

    return(true);
}

bool Neural_Network::Reallocate__Batch__LSTM(size_t const batch_size_received)
{
    if(this->total_block_units_allocated * this->total_cell_units_allocated != 0_zu)
    {
        size_t tmp_number_block_units,
                  tmp_number_cell_units;
        
        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1); // Subtract output layer.
        struct Layer *tmp_ptr_layer_it;

        struct Block_unit const *tmp_ptr_last_block_unit;
        struct Block_unit *tmp_ptr_block_unit_it;

        struct Cell_unit const *tmp_ptr_last_cell_unit;
        struct Cell_unit *tmp_ptr_cell_unit_it;
        
        // Allocating summation cell input.
        T_ *tmp_ptr_array_summation_cells_inputs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_summations_cells_inputs,
                                                                                                                   batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                                   this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_cells_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_summations_cells_inputs = tmp_ptr_array_summation_cells_inputs;
        // |END| Allocating summation cell input. |END|
        
        // Allocating summation input cell input.
        T_ *tmp_ptr_array_summation_input_cells_inputs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_summations_input_cells_inputs,
                                                                                                                           batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                                           this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_input_cells_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_summations_input_cells_inputs = tmp_ptr_array_summation_input_cells_inputs;
        // |END| Allocating summation input cell input. |END|
        
        // Allocating summation recurrent cell input.
        T_ *tmp_ptr_array_summation_recurrent_cells_inputs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_summations_recurrent_cells_inputs,
                                                                                                                                 batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                                                 this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_recurrent_cells_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_summations_recurrent_cells_inputs = tmp_ptr_array_summation_recurrent_cells_inputs;
        // |END| Allocating summation recurrent cell input. |END|
        
        // Allocating LSTM summation input gate.
        T_ *tmp_ptr_array_summation_inputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_summations_inputs_gates,
                                                                                                                    batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                    this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_summations_inputs_gates = tmp_ptr_array_summation_inputs_gates;
        // |END| Allocating LSTM summation input gate. |END|
        
        // Allocating LSTM summation input input gate.
        T_ *tmp_ptr_array_summation_input_inputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_summations_input_inputs_gates,
                                                                                                                             batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                             this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_input_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_summations_input_inputs_gates = tmp_ptr_array_summation_input_inputs_gates;
        // |END| Allocating LSTM summation input input gate. |END|
        
        // Allocating LSTM summation recurrent input gate.
        T_ *tmp_ptr_array_summation_recurrent_inputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_summations_recurrent_inputs_gates,
                                                                                                                                   batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                                   this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_recurrent_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_summations_recurrent_inputs_gates = tmp_ptr_array_summation_recurrent_inputs_gates;
        // |END| Allocating LSTM summation recurrent input gate. |END|
        
        // Allocating LSTM summation forget gate.
        T_ *tmp_ptr_array_summation_forgets_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_summations_forgets_gates,
                                                                                                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                     this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_summations_forgets_gates = tmp_ptr_array_summation_forgets_gates;
        // |END| Allocating LSTM summation forget gate. |END|
        
        // Allocating LSTM summation input forget gate.
        T_ *tmp_ptr_array_summation_input_forgets_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_summations_input_forgets_gates,
                                                                                                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                     this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_input_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_summations_input_forgets_gates = tmp_ptr_array_summation_input_forgets_gates;
        // |END| Allocating LSTM summation input forget gate. |END|
        
        // Allocating LSTM summation recurrent forget gate.
        T_ *tmp_ptr_array_summation_recurrent_forgets_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_summations_recurrent_forgets_gates,
                                                                                                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                     this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_recurrent_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_summations_recurrent_forgets_gates = tmp_ptr_array_summation_recurrent_forgets_gates;
        // |END| Allocating LSTM summation recurrent forget gate. |END|
        
        // Allocating LSTM summation outputs gate.
        T_ *tmp_ptr_array_summation_outputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_summations_outputs_gates,
                                                                                                                      batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                      this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_summations_outputs_gates = tmp_ptr_array_summation_outputs_gates;
        // |END| Allocating LSTM summation outputs gate. |END|
        
        // Allocating LSTM summation input outputs gate.
        T_ *tmp_ptr_array_summation_input_outputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_summations_input_outputs_gates,
                                                                                                                      batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                      this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_input_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_summations_input_outputs_gates = tmp_ptr_array_summation_input_outputs_gates;
        // |END| Allocating LSTM summation input outputs gate. |END|
        
        // Allocating LSTM summation recurrent outputs gate.
        T_ *tmp_ptr_array_summation_recurrent_outputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_summations_recurrent_outputs_gates,
                                                                                                                      batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                      this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_summation_recurrent_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_summations_recurrent_outputs_gates = tmp_ptr_array_summation_recurrent_outputs_gates;
        // |END| Allocating LSTM summation recurrent outputs gate. |END|
        
        // Allocating cell input.
        T_ *tmp_ptr_array_cells_inputs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_inputs,
                                                                                                 batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                 this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_cells_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_inputs = tmp_ptr_array_cells_inputs;
        // |END| Allocating cell input. |END|
        
        // Allocating LSTM cell state.
        T_ *tmp_ptr_array_cells_states(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_states,
                                                                                                 batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                 this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_cells_states == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_states = tmp_ptr_array_cells_states;
        // |END| Allocating LSTM cell state. |END|
        
        // Allocating LSTM cell state activate.
        T_ *tmp_ptr_array_cells_states_activates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_states_activates,
                                                                                                 batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                 this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_cells_states_activates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_states_activates = tmp_ptr_array_cells_states_activates;
        // |END| Allocating LSTM cell state activate. |END|
        
        // Allocating LSTM cell outputs.
        T_ *tmp_ptr_array_cells_outputs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_outputs,
                                                                                                   batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                   this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_cells_outputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_outputs = tmp_ptr_array_cells_outputs;
        // |END| Allocating LSTM cell outputs. |END|
        
        // Allocating LSTM input gate.
        T_ *tmp_ptr_array_inputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_inputs_gates,
                                                                                                  batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                  this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_inputs_gates = tmp_ptr_array_inputs_gates;
        // |END| Allocating LSTM input gate. |END|
        
        // Allocating LSTM forget gate.
        T_ *tmp_ptr_array_forgets_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_forgets_gates,
                                                                                                   batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                   this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_forgets_gates = tmp_ptr_array_forgets_gates;
        // |END| Allocating LSTM forget gate. |END|
        
        // Allocating LSTM outputs gate.
        T_ *tmp_ptr_array_outputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_outputs_gates,
                                                                                                    batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                    this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_outputs_gates = tmp_ptr_array_outputs_gates;
        // |END| Allocating LSTM outputs gate. |END|
        
        // Allocating LSTM delta cell inputs.
        T_ *tmp_ptr_array_delta_cells_inputs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_delta_inputs,
                                                                                                          batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                          this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_cells_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_delta_inputs = tmp_ptr_array_delta_cells_inputs;
        // |END| Allocating LSTM delta cell inputs. |END|
        
        // Allocating LSTM delta cell input inputs.
        T_ *tmp_ptr_array_delta_cells_input_inputs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_delta_input_inputs,
                                                                                                                  batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                                  this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_cells_input_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_delta_input_inputs = tmp_ptr_array_delta_cells_input_inputs;
        // |END| Allocating LSTM delta cell input inputs. |END|
        
        // Allocating LSTM delta cell recurrent inputs.
        T_ *tmp_ptr_array_delta_cells_recurrent_inputs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_delta_recurrent_inputs,
                                                                                                                      batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                                      this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_cells_recurrent_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_delta_recurrent_inputs = tmp_ptr_array_delta_cells_recurrent_inputs;
        // |END| Allocating LSTM delta cell recurrent inputs. |END|
        
        // Allocating LSTM delta cell state.
        T_ *tmp_ptr_array_delta_cells_states(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_delta_states,
                                                                                                          batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                          this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_cells_states == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_delta_states = tmp_ptr_array_delta_cells_states;
        // |END| Allocating LSTM delta cell state. |END|
        
        // Allocating LSTM delta cell outputs.
        T_ *tmp_ptr_array_delta_cells_outputs(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_cells_delta_outputs,
                                                                                                            batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                                                                                            this->batch_size * this->total_cell_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_cells_outputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_cell_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_cells_delta_outputs = tmp_ptr_array_delta_cells_outputs;
        // |END| Allocating LSTM delta cell outputs. |END|
        
        // Allocating LSTM delta input gate.
        T_ *tmp_ptr_array_delta_inputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_delta_inputs_gates,
                                                                                                           batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                           this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_delta_inputs_gates = tmp_ptr_array_delta_inputs_gates;
        // |END| Allocating LSTM delta input gate. |END|
        
        // Allocating LSTM delta input input gate.
        T_ *tmp_ptr_array_delta_input_inputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_delta_input_inputs_gates,
                                                                                                                    batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                    this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_input_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_delta_input_inputs_gates = tmp_ptr_array_delta_input_inputs_gates;
        // |END| Allocating LSTM delta input input gate. |END|
        
        // Allocating LSTM delta recurrent input gate.
        T_ *tmp_ptr_array_delta_recurrent_inputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_delta_recurrent_inputs_gates,
                                                                                                                        batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                        this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_recurrent_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_delta_recurrent_inputs_gates = tmp_ptr_array_delta_recurrent_inputs_gates;
        // |END| Allocating LSTM delta recurrent input gate. |END|
        
        // Allocating LSTM delta forget gate.
        T_ *tmp_ptr_array_delta_forgets_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_delta_forgets_gates,
                                                                                                            batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                            this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_delta_forgets_gates = tmp_ptr_array_delta_forgets_gates;
        // |END| Allocating LSTM delta forget gate. |END|
        
        // Allocating LSTM delta input forget gate.
        T_ *tmp_ptr_array_delta_input_forgets_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_delta_input_forgets_gates,
                                                                                                                    batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                    this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_input_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_delta_input_forgets_gates = tmp_ptr_array_delta_input_forgets_gates;
        // |END| Allocating LSTM delta input forget gate. |END|
        
        // Allocating LSTM delta recurrent forget gate.
        T_ *tmp_ptr_array_delta_recurrent_forgets_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_delta_recurrent_forgets_gates,
                                                                                                                            batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                            this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_recurrent_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_delta_recurrent_forgets_gates = tmp_ptr_array_delta_recurrent_forgets_gates;
        // |END| Allocating LSTM delta recurrent forget gate. |END|
        
        // Allocating LSTM delta outputs gate.
        T_ *tmp_ptr_array_delta_outputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_delta_outputs_gates,
                                                                                                             batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                             this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_delta_outputs_gates = tmp_ptr_array_delta_outputs_gates;
        // |END| Allocating LSTM delta outputs gate. |END|
        
        // Allocating LSTM delta input outputs gate.
        T_ *tmp_ptr_array_delta_input_outputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_delta_input_outputs_gates,
                                                                                                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                     this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_input_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_delta_input_outputs_gates = tmp_ptr_array_delta_input_outputs_gates;
        // |END| Allocating LSTM delta input outputs gate. |END|
        
        // Allocating LSTM delta recurrent outputs gate.
        T_ *tmp_ptr_array_delta_recurrent_outputs_gates(MyEA::Memory::Cpp::Reallocate<T_, false>(this->ptr_array_blocks_delta_recurrent_outputs_gates,
                                                                                                                            batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                                                                                                            this->batch_size * this->total_block_units * this->number_recurrent_depth));
        if(tmp_ptr_array_delta_recurrent_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     batch_size_received * this->total_block_units * this->number_recurrent_depth,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_blocks_delta_recurrent_outputs_gates = tmp_ptr_array_delta_recurrent_outputs_gates;
        // |END| Allocating LSTM delta recurrent outputs gate. |END|
        
        for(tmp_ptr_layer_it = this->ptr_array_layers + 1; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_block_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units);
            tmp_number_cell_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units);
            
            if(tmp_number_block_units * tmp_number_cell_units != 0_zu)
            {
                for(tmp_ptr_last_block_unit = tmp_ptr_layer_it->ptr_last_block_unit,
                    tmp_ptr_block_unit_it = tmp_ptr_layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                                                                                          ++tmp_ptr_array_summation_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_input_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_recurrent_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_input_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_recurrent_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_input_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_recurrent_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_input_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_recurrent_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_input_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_recurrent_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_input_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_recurrent_outputs_gates)
                {
                    tmp_ptr_block_unit_it->ptr_array_summation_cells_inputs = tmp_ptr_array_summation_cells_inputs;
                    tmp_ptr_block_unit_it->ptr_array_summation_input_cells_inputs = tmp_ptr_array_summation_input_cells_inputs;
                    tmp_ptr_block_unit_it->ptr_array_summation_recurrent_cells_inputs = tmp_ptr_array_summation_recurrent_cells_inputs;
                    tmp_ptr_block_unit_it->ptr_summation_inputs_gates = tmp_ptr_array_summation_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_summation_input_inputs_gates = tmp_ptr_array_summation_input_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_summation_recurrent_inputs_gates = tmp_ptr_array_summation_recurrent_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_summation_forgets_gates = tmp_ptr_array_summation_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_summation_input_forgets_gates = tmp_ptr_array_summation_input_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_summation_recurrent_forgets_gates = tmp_ptr_array_summation_recurrent_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_summation_outputs_gates = tmp_ptr_array_summation_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_summation_input_outputs_gates = tmp_ptr_array_summation_input_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_summation_recurrent_outputs_gates = tmp_ptr_array_summation_recurrent_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_array_cells_inputs = tmp_ptr_array_cells_inputs;
                    tmp_ptr_block_unit_it->ptr_array_cells_states = tmp_ptr_array_cells_states;
                    tmp_ptr_block_unit_it->ptr_array_cells_states_activates = tmp_ptr_array_cells_states_activates;
                    tmp_ptr_block_unit_it->ptr_array_cells_outputs = tmp_ptr_array_cells_outputs;
                    tmp_ptr_block_unit_it->ptr_inputs_gates = tmp_ptr_array_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_forgets_gates = tmp_ptr_array_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_outputs_gates = tmp_ptr_array_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_array_delta_cells_inputs = tmp_ptr_array_delta_cells_inputs;
                    tmp_ptr_block_unit_it->ptr_array_delta_cells_input_inputs = tmp_ptr_array_delta_cells_input_inputs;
                    tmp_ptr_block_unit_it->ptr_array_delta_cells_recurrent_inputs = tmp_ptr_array_delta_cells_recurrent_inputs;
                    tmp_ptr_block_unit_it->ptr_array_delta_cells_states = tmp_ptr_array_delta_cells_states;
                    tmp_ptr_block_unit_it->ptr_array_delta_cells_outputs = tmp_ptr_array_delta_cells_outputs;
                    tmp_ptr_block_unit_it->ptr_delta_inputs_gates = tmp_ptr_array_delta_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_delta_input_inputs_gates = tmp_ptr_array_delta_input_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_delta_recurrent_inputs_gates = tmp_ptr_array_delta_recurrent_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_delta_forgets_gates = tmp_ptr_array_delta_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_delta_input_forgets_gates = tmp_ptr_array_delta_input_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_delta_recurrent_forgets_gates = tmp_ptr_array_delta_recurrent_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_delta_outputs_gates = tmp_ptr_array_delta_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_delta_input_outputs_gates = tmp_ptr_array_delta_input_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_delta_recurrent_outputs_gates = tmp_ptr_array_delta_recurrent_outputs_gates;
                    
                    for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                        tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                           ++tmp_ptr_array_summation_cells_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_summation_input_cells_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_summation_recurrent_cells_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_cells_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_cells_states,
                                                                                                                                                                                           ++tmp_ptr_array_cells_states_activates,
                                                                                                                                                                                           ++tmp_ptr_array_cells_outputs,
                                                                                                                                                                                           ++tmp_ptr_array_delta_cells_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_delta_cells_input_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_delta_cells_recurrent_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_delta_cells_states,
                                                                                                                                                                                           ++tmp_ptr_array_delta_cells_outputs)
                    {
                            tmp_ptr_cell_unit_it->ptr_summation_cell_input = tmp_ptr_array_summation_cells_inputs;
                            tmp_ptr_cell_unit_it->ptr_summation_input_cell_input = tmp_ptr_array_summation_input_cells_inputs;
                            tmp_ptr_cell_unit_it->ptr_summation_recurrent_cell_input = tmp_ptr_array_summation_recurrent_cells_inputs;
                            tmp_ptr_cell_unit_it->ptr_cell_input = tmp_ptr_array_cells_inputs;
                            tmp_ptr_cell_unit_it->ptr_cell_state = tmp_ptr_array_cells_states;
                            tmp_ptr_cell_unit_it->ptr_cell_state_activate = tmp_ptr_array_cells_states_activates;
                            tmp_ptr_cell_unit_it->ptr_cell_output = tmp_ptr_array_cells_outputs;
                            tmp_ptr_cell_unit_it->ptr_delta_cell_input = tmp_ptr_array_delta_cells_inputs;
                            tmp_ptr_cell_unit_it->ptr_delta_cell_input_input = tmp_ptr_array_delta_cells_input_inputs;
                            tmp_ptr_cell_unit_it->ptr_delta_cell_recurrent_input = tmp_ptr_array_delta_cells_recurrent_inputs;
                            tmp_ptr_cell_unit_it->ptr_delta_cell_state = tmp_ptr_array_delta_cells_states;
                            tmp_ptr_cell_unit_it->ptr_delta_cell_output = tmp_ptr_array_delta_cells_outputs;
                    }
                }

                tmp_ptr_array_summation_cells_inputs += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_input_cells_inputs += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_recurrent_cells_inputs += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_inputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_input_inputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_recurrent_inputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_forgets_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_input_forgets_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_recurrent_forgets_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_outputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_input_outputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_recurrent_outputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_cells_inputs += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_cells_states += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_cells_states_activates += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_cells_outputs += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_inputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_forgets_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_outputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_cells_inputs += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_cells_input_inputs += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_cells_recurrent_inputs += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_cells_states += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_cells_outputs += (batch_size_received - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_inputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_input_inputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_recurrent_inputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_forgets_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_input_forgets_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_recurrent_forgets_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_outputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_input_outputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_recurrent_outputs_gates += (batch_size_received - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
            }
        }
    }

    return(true);
}

bool Neural_Network::Reallocate__Batch__Dropout__ShakeDrop(size_t const batch_size_received)
{
    if(this->total_layers != 0_zu)
    {
        bool *tmp_ptr_array_layers_mask_dropout_shakedrop(MyEA::Memory::Cpp::Reallocate_PtOfPt<bool, false>(this->ptr_array_layers_mask_dropout_shakedrop,
                                                                                                                                     this->total_layers * this->number_recurrent_depth * batch_size_received,
                                                                                                                                     this->total_layers * this->number_recurrent_depth * this->batch_size));
        if(tmp_ptr_array_layers_mask_dropout_shakedrop == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(bool),
                                     this->total_layers * this->number_recurrent_depth * batch_size_received,
                                     this->total_layers * this->number_recurrent_depth * this->batch_size,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_layers_mask_dropout_shakedrop = tmp_ptr_array_layers_mask_dropout_shakedrop;
        
        for(struct Layer *tmp_ptr_layer_it(this->ptr_array_layers); tmp_ptr_layer_it != this->ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_ptr_layer_it->ptr_array__mask__dropout__shakedrop = tmp_ptr_array_layers_mask_dropout_shakedrop;
            
            tmp_ptr_array_layers_mask_dropout_shakedrop += this->number_recurrent_depth * batch_size_received;
        }
    }

    return(true);
}
