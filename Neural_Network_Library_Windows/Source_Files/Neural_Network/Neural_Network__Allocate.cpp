#include "stdafx.hpp"

#include <chrono>

#include <Neural_Network/Neural_Network.hpp>

bool Neural_Network::Allocate__Structure(size_t const number_layers_received, size_t const maximum_allowable_memory_received)
{
    // Computation parameters.
    this->maximum_allowable_memory_bytes = maximum_allowable_memory_received;
    // |END| Computation parameters. |END|

    // Allocate layers.
    struct Layer *tmp_ptr_layer_it(new struct Layer[number_layers_received]);
    if(tmp_ptr_layer_it == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 number_layers_received * sizeof(struct Layer),
                                 __LINE__);

        return(false);
    }
    this->ptr_array_layers = tmp_ptr_layer_it;
    
    this->ptr_last_layer = this->ptr_array_layers + number_layers_received;
    
    this->total_layers = number_layers_received;
    
    size_t *tmp_ptr_array_layers_number_outputs_it(new size_t[number_layers_received]);
    if(tmp_ptr_array_layers_number_outputs_it == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 number_layers_received * sizeof(size_t),
                                 __LINE__);

        return(false);
    }
    memset(tmp_ptr_array_layers_number_outputs_it,
                 0,
                 number_layers_received * sizeof(size_t));
    this->ptr_array_layers_number_outputs = tmp_ptr_array_layers_number_outputs_it;
    
    size_t *tmp_ptr_array_layers_first_connection_index_it(new size_t[number_layers_received]);
    if(tmp_ptr_array_layers_first_connection_index_it == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 number_layers_received * sizeof(size_t),
                                 __LINE__);

        return(false);
    }
    memset(tmp_ptr_array_layers_first_connection_index_it,
                 0,
                 number_layers_received * sizeof(size_t));
    this->ptr_array_layers_first_connection_index = tmp_ptr_array_layers_first_connection_index_it;
    
    size_t *tmp_ptr_array_layers_last_connection_index_it(new size_t[number_layers_received]);
    if(tmp_ptr_array_layers_last_connection_index_it == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 number_layers_received * sizeof(size_t),
                                 __LINE__);

        return(false);
    }
    memset(tmp_ptr_array_layers_last_connection_index_it,
                 0,
                 number_layers_received * sizeof(size_t));
    this->ptr_array_layers_last_connection_index = tmp_ptr_array_layers_last_connection_index_it;
    
    for(; tmp_ptr_layer_it != this->ptr_last_layer; ++tmp_ptr_layer_it,
                                                                    ++tmp_ptr_array_layers_number_outputs_it,
                                                                    ++tmp_ptr_array_layers_first_connection_index_it,
                                                                    ++tmp_ptr_array_layers_last_connection_index_it)
    {
        tmp_ptr_layer_it->ptr_number_outputs = tmp_ptr_array_layers_number_outputs_it;

        tmp_ptr_layer_it->ptr_first_connection_index = tmp_ptr_array_layers_first_connection_index_it;
        tmp_ptr_layer_it->ptr_last_connection_index = tmp_ptr_array_layers_last_connection_index_it;
    }
    // |END| Allocate layers. |END|

    // Loss parameters.
    if((this->ptr_array_number_loss = new size_t[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sizeof(size_t),
                                 __LINE__);

        return(false);
    }
    else { *this->ptr_array_number_loss = 0_zu; }

    if((this->ptr_array_number_bit_fail = new size_t[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sizeof(size_t),
                                 __LINE__);

        return(false);
    }
    else { *this->ptr_array_number_bit_fail = 0_zu; }
    
    if((this->ptr_array_loss_values = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sizeof(size_t),
                                 __LINE__);

        return(false);
    }
    else { *this->ptr_array_loss_values = 0_T; }
    // |END| Loss parameters. |END|
    
    // Accuracy parameters.
    if((this->ptr_array_accuracy_values[0u] = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 __LINE__);

        return(false);
    }
    else { this->ptr_array_accuracy_values[0u][0u] = 0_T; }

    if((this->ptr_array_accuracy_values[1u] = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 __LINE__);

        return(false);
    }
    else { this->ptr_array_accuracy_values[1u][0u] = 0_T; }

    if((this->ptr_array_accuracy_values[2u] = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 __LINE__);

        return(false);
    }
    else { this->ptr_array_accuracy_values[2u][0u] = 0_T; }

    if((this->ptr_array_accuracy_values[3u] = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 __LINE__);

        return(false);
    }
    else { this->ptr_array_accuracy_values[3u][0u] = 0_T; }

    if((this->ptr_array_accuracy_values[4u] = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 __LINE__);

        return(false);
    }
    else { this->ptr_array_accuracy_values[4u][0u] = 0_T; }
    // |END| Accuracy parameters. |END|
    
    this->Class_Generator_Real.Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    this->Class_Generator_Real.Range(0_T, 1_T);

    this->Class_Generator_Gaussian.Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    this->Class_Generator_Gaussian.Range(0_T, 1_T);

    return(true);
}

bool Neural_Network::Allocate__Sparse_K_Filter(void)
{
    if((this->ptr_array_k_sparse_activities = new std::pair<size_t, T_>[this->number_threads * (this->total_AF_units + this->total_AF_Ind_recurrent_units + this->total_block_units)]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->number_threads * (this->total_AF_units + this->total_AF_Ind_recurrent_units + this->total_block_units) * sizeof(std::pair<size_t, T_>),
                                 __LINE__);

        return(false);
    }

    this->Assign__Sparsity_Activities(this->number_threads);

    return(true);
}

bool Neural_Network::Allocate__Generator__Dropout_Bernoulli(void)
{
    if(this->ptr_array_Class_Generator_Bernoulli == nullptr)
    {
        if((this->ptr_array_Class_Generator_Bernoulli = new class MyEA::Random::Bernoulli<T_>[this->number_threads]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * sizeof(class MyEA::Random::Bernoulli<T_>),
                                     __LINE__);

            return(false);
        }

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        { this->ptr_array_Class_Generator_Bernoulli[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    return(true);
}

bool Neural_Network::Allocate__Generator__Dropout_Gaussian(void)
{
    if(this->ptr_array_Class_Generator_Real_Gaussian == nullptr)
    {
        if((this->ptr_array_Class_Generator_Real_Gaussian = new class MyEA::Random::Gaussian<T_>[this->number_threads]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * sizeof(class MyEA::Random::Gaussian<T_>),
                                     __LINE__);

            return(false);
        }

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        {
            this->ptr_array_Class_Generator_Real_Gaussian[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            this->ptr_array_Class_Generator_Real_Gaussian[tmp_generator_index].Range(0_T, 1_T);
        }
    }

    return(true);
}

bool Neural_Network::Allocate__Generator__Dropout_ShakeDrop(void)
{
    if(this->ptr_array_Class_Generator_Bernoulli_ShakeDrop == nullptr)
    {
        if((this->ptr_array_Class_Generator_Bernoulli_ShakeDrop = new class MyEA::Random::Bernoulli<T_>[this->number_threads]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * sizeof(class MyEA::Random::Bernoulli<T_>),
                                     __LINE__);

            return(false);
        }

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        { this->ptr_array_Class_Generator_Bernoulli_ShakeDrop[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    if(this->ptr_array_Class_Generator_Real_ShakeDrop == nullptr)
    {
        if((this->ptr_array_Class_Generator_Real_ShakeDrop = new class MyEA::Random::Floating<T_>[this->number_threads]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * sizeof(class MyEA::Random::Floating<T_>),
                                     __LINE__);

            return(false);
        }

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        {
            this->ptr_array_Class_Generator_Real_ShakeDrop[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            this->ptr_array_Class_Generator_Real_ShakeDrop[tmp_generator_index].Range(0_T, 1_T);
        }
    }

    return(true);
}

bool Neural_Network::Allocate__Generator__Dropout_Uout(void)
{
    if(this->ptr_array_Class_Generator_Real_Uout == nullptr)
    {
        if((this->ptr_array_Class_Generator_Real_Uout = new class MyEA::Random::Floating<T_>[this->number_threads]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * sizeof(class MyEA::Random::Floating<T_>),
                                     __LINE__);

            return(false);
        }

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        {
            this->ptr_array_Class_Generator_Real_Uout[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            this->ptr_array_Class_Generator_Real_Uout[tmp_generator_index].Range(0_T, 1_T);
        }
    }

    return(true);
}

bool Neural_Network::Allocate__Generator__Dropout_Zoneout(void)
{
    if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State == nullptr)
    {
        if((this->ptr_array_Class_Generator_Bernoulli_Zoneout_State = new class MyEA::Random::Bernoulli<T_>[this->number_threads]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * sizeof(class MyEA::Random::Bernoulli<T_>),
                                     __LINE__);

            return(false);
        }

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        { this->ptr_array_Class_Generator_Bernoulli_Zoneout_State[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }
    
    if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden == nullptr)
    {
        if((this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden = new class MyEA::Random::Bernoulli<T_>[this->number_threads]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * sizeof(class MyEA::Random::Bernoulli<T_>),
                                     __LINE__);

            return(false);
        }

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        { this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    return(true);
}

bool Neural_Network::Allocate__Neuron__Mask_Dropout_Bernoulli(void)
{
    if(this->ptr_array_units_mask_dropout_bernoulli == nullptr)
    {
        bool *tmp_ptr_array_units_mask_dropout_bernoulli(new bool[(this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated) * this->number_recurrent_depth]);

        if(tmp_ptr_array_units_mask_dropout_bernoulli == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     (this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated) * this->number_recurrent_depth * sizeof(bool),
                                     __LINE__);

            return(false);
        }
        
        MyEA::Memory::Fill<bool>(tmp_ptr_array_units_mask_dropout_bernoulli,
                                     tmp_ptr_array_units_mask_dropout_bernoulli + (this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated) * this->number_recurrent_depth,
                                     true);
        
        this->Reset__Parameter__Mask_Dropout(tmp_ptr_array_units_mask_dropout_bernoulli);
    }

    return(true);
}

bool Neural_Network::Allocate__Layer__Mask__Dropout__ShakeDrop(void)
{
    if(this->ptr_array_layers_mask_dropout_shakedrop == nullptr)
    {
        bool *tmp_ptr_array_layers_mask_dropout_shakedrop(new bool[this->total_layers * this->number_recurrent_depth * this->batch_size]);
        if(tmp_ptr_array_layers_mask_dropout_shakedrop == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_layers * this->number_recurrent_depth * this->batch_size * sizeof(bool),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_layers_mask_dropout_shakedrop = tmp_ptr_array_layers_mask_dropout_shakedrop;
        MyEA::Memory::Fill<bool>(tmp_ptr_array_layers_mask_dropout_shakedrop,
                                     tmp_ptr_array_layers_mask_dropout_shakedrop + this->total_layers * this->number_recurrent_depth * this->batch_size,
                                     true);
        
        for(struct Layer *tmp_ptr_layer_it(this->ptr_array_layers); tmp_ptr_layer_it != this->ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_ptr_layer_it->ptr_array__mask__dropout__shakedrop = tmp_ptr_array_layers_mask_dropout_shakedrop;
            
            tmp_ptr_array_layers_mask_dropout_shakedrop += this->number_recurrent_depth * this->batch_size;
        }
    }

    return(true);
}

bool Neural_Network::Allocate__Basic_Units(void)
{
    size_t tmp_number_basic_units,
              tmp_basic_unit_index;
    
    if(this->ptr_array_basic_units == nullptr && this->total_basic_units != 0_zu)
    {
        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct Basic_unit *tmp_ptr_array_basic_units(new struct Basic_unit[this->total_basic_units]);
        if(tmp_ptr_array_basic_units == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_basic_units * sizeof(struct Basic_unit),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_units = tmp_ptr_array_basic_units;
        
        // Allocating basic unit(s) value.
        T_ *tmp_ptr_array_basic_units_values(new T_[this->batch_size * this->total_basic_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_basic_units_values == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_basic_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_units_values = tmp_ptr_array_basic_units_values;
        MEMSET(tmp_ptr_array_basic_units_values,
                       0,
                       this->batch_size * this->total_basic_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating basic unit(s) value. |END|
        
        // Allocating basic unit(s) error.
        T_ *tmp_ptr_array_basic_units_errors(new T_[this->batch_size * this->total_basic_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_basic_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_basic_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_units_errors = tmp_ptr_array_basic_units_errors;
        MEMSET(tmp_ptr_array_basic_units_errors,
                       0,
                       this->batch_size * this->total_basic_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating basic unit(s) error. |END|

        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_basic_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_basic_unit - tmp_ptr_layer_it->ptr_array_basic_units);

            tmp_ptr_layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
            
            if(tmp_number_basic_units != 0_zu)
            {
                // Assign basic unit variables.
                for(tmp_basic_unit_index = 0_zu; tmp_basic_unit_index != tmp_number_basic_units; ++tmp_basic_unit_index,
                                                                                                                                            ++tmp_ptr_array_basic_units_values,
                                                                                                                                            ++tmp_ptr_array_basic_units_errors)
                {
                    tmp_ptr_array_basic_units[tmp_basic_unit_index].ptr_array_values = tmp_ptr_array_basic_units_values;
                    tmp_ptr_array_basic_units[tmp_basic_unit_index].ptr_array_errors = tmp_ptr_array_basic_units_errors;
                }

                tmp_ptr_array_basic_units_values += (this->batch_size - 1_zu) * tmp_number_basic_units * this->number_recurrent_depth + tmp_number_basic_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_basic_units_errors += (this->batch_size - 1_zu) * tmp_number_basic_units * this->number_recurrent_depth + tmp_number_basic_units * (this->number_recurrent_depth - 1_zu);
                // |END| Assign basic unit variables. |END|
                
                tmp_ptr_array_basic_units += tmp_number_basic_units;
            }

            tmp_ptr_layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
        }
        
        this->ptr_last_basic_unit = tmp_ptr_array_basic_units;

        this->total_basic_units_allocated = this->total_basic_units;
    }

    return(true);
}

bool Neural_Network::Allocate__Basic_Indice_Units(void)
{
    size_t tmp_number_basic_indice_units,
              tmp_basic_indice_unit_index;
    
    if(this->ptr_array_basic_indice_units == nullptr && this->total_basic_indice_units != 0_zu)
    {
        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct Basic_indice_unit *tmp_ptr_array_basic_indice_units(new struct Basic_indice_unit[this->total_basic_indice_units]);
        if(tmp_ptr_array_basic_indice_units == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_basic_indice_units * sizeof(struct Basic_indice_unit),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
        
        // Allocating basic indice unit(s) indice.
        size_t *tmp_ptr_array_basic_indice_units_indices(new size_t[this->batch_size * this->total_basic_indice_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_basic_indice_units_indices == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_basic_indice_units * this->number_recurrent_depth * sizeof(size_t),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_indice_units_indices = tmp_ptr_array_basic_indice_units_indices;
        memset(tmp_ptr_array_basic_indice_units_indices,
                     0,
                     this->batch_size * this->total_basic_indice_units * this->number_recurrent_depth * sizeof(size_t));
        // |END| Allocating basic indice unit(s) indice. |END|
        
        // Allocating basic indice unit(s) value.
        T_ *tmp_ptr_array_basic_indice_units_values(new T_[this->batch_size * this->total_basic_indice_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_basic_indice_units_values == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_basic_indice_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_indice_units_values = tmp_ptr_array_basic_indice_units_values;
        MEMSET(tmp_ptr_array_basic_indice_units_values,
                       0,
                       this->batch_size * this->total_basic_indice_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating basic indice unit(s) value. |END|
        
        // Allocating basic indice unit(s) errors.
        T_ *tmp_ptr_array_basic_indice_units_errors(new T_[this->batch_size * this->total_basic_indice_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_basic_indice_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_basic_indice_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_basic_indice_units_errors = tmp_ptr_array_basic_indice_units_errors;
        MEMSET(tmp_ptr_array_basic_indice_units_errors,
                       0,
                       this->batch_size * this->total_basic_indice_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating basic indice unit(s) error. |END|

        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_basic_indice_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_basic_indice_unit - tmp_ptr_layer_it->ptr_array_basic_indice_units);

            tmp_ptr_layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
            
            if(tmp_number_basic_indice_units != 0_zu)
            {
                // Assign basic unit variables.
                for(tmp_basic_indice_unit_index = 0_zu; tmp_basic_indice_unit_index != tmp_number_basic_indice_units; ++tmp_basic_indice_unit_index,
                                                                                                                                                                          ++tmp_ptr_array_basic_indice_units_indices,
                                                                                                                                                                          ++tmp_ptr_array_basic_indice_units_values,
                                                                                                                                                                          ++tmp_ptr_array_basic_indice_units_errors)
                {
                    tmp_ptr_array_basic_indice_units[tmp_basic_indice_unit_index].ptr_array_indices = tmp_ptr_array_basic_indice_units_indices;

                    tmp_ptr_array_basic_indice_units[tmp_basic_indice_unit_index].ptr_array_values = tmp_ptr_array_basic_indice_units_values;
                    tmp_ptr_array_basic_indice_units[tmp_basic_indice_unit_index].ptr_array_errors = tmp_ptr_array_basic_indice_units_errors;
                }

                tmp_ptr_array_basic_indice_units_indices += (this->batch_size - 1_zu) * tmp_number_basic_indice_units * this->number_recurrent_depth + tmp_number_basic_indice_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_basic_indice_units_values += (this->batch_size - 1_zu) * tmp_number_basic_indice_units * this->number_recurrent_depth + tmp_number_basic_indice_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_basic_indice_units_errors += (this->batch_size - 1_zu) * tmp_number_basic_indice_units * this->number_recurrent_depth + tmp_number_basic_indice_units * (this->number_recurrent_depth - 1_zu);
                // |END| Assign basic unit variables. |END|
                
                tmp_ptr_array_basic_indice_units += tmp_number_basic_indice_units;
            }

            tmp_ptr_layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
        }
        
        this->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;

        this->total_basic_indice_units_allocated = this->total_basic_indice_units;
    }

    return(true);
}

bool Neural_Network::Allocate__Neuron_Units(void)
{
    size_t tmp_number_neuron_units,
              tmp_neuron_index;
    
    if(this->ptr_array_neuron_units == nullptr && this->total_neuron_units != 0_zu)
    {
        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct Neuron_unit *tmp_ptr_array_neuron_units(new struct Neuron_unit[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_neuron_units * sizeof(struct Neuron_unit),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
        
        // Allocating neurons first forward connection index.
        size_t *tmp_ptr_array_neuron_units_first_forward_connection_index(new size_t[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_first_forward_connection_index == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_neuron_units * sizeof(size_t),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_neuron_units_first_forward_connection_index = tmp_ptr_array_neuron_units_first_forward_connection_index;
        memset(tmp_ptr_array_neuron_units_first_forward_connection_index,
                    0,
                    this->total_neuron_units * sizeof(size_t));
        // |END| Allocating neurons first forward connection index. |END|
        
        // Allocating neurons last forward connection index.
        size_t *tmp_ptr_array_neuron_units_last_forward_connection_index(new size_t[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_last_forward_connection_index == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_neuron_units * sizeof(size_t),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_neuron_units_last_forward_connection_index = tmp_ptr_array_neuron_units_last_forward_connection_index;
        memset(tmp_ptr_array_neuron_units_last_forward_connection_index,
                    0,
                    this->total_neuron_units * sizeof(size_t));
        // |END| Allocating neurons last forward connection index. |END|
        
        // Allocating neurons number forward connections.
        size_t *tmp_ptr_array_neuron_units_number_forward_connections(new size_t[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_number_forward_connections == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_neuron_units * sizeof(size_t),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_neuron_units_number_forward_connections = tmp_ptr_array_neuron_units_number_forward_connections;
        memset(tmp_ptr_array_neuron_units_number_forward_connections,
                    0,
                    this->total_neuron_units * sizeof(size_t));
        // |END| Allocating neurons number forward connections. |END|
        
        // Allocating neuron unit(s) summation(s).
        T_ *tmp_ptr_array_neuron_units_summations(new T_[this->batch_size * this->total_neuron_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_neuron_units_summations == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_neuron_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_neuron_units_summations = tmp_ptr_array_neuron_units_summations;
        MEMSET(tmp_ptr_array_neuron_units_summations,
                        0,
                        this->batch_size * this->total_neuron_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating neuron unit(s) summation(s). |END|
        
        // Allocating neuron unit(s) dAF(s).
        T_ *tmp_ptr_array_neuron_units_errors(new T_[this->batch_size * this->total_neuron_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_neuron_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_neuron_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_neuron_units_errors = tmp_ptr_array_neuron_units_errors;
        MEMSET(tmp_ptr_array_neuron_units_errors,
                     0,
                     this->batch_size * this->total_neuron_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating neuron unit(s) dAF(s). |END|
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_neuron_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_neuron_unit - tmp_ptr_layer_it->ptr_array_neuron_units);

            tmp_ptr_layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
            
            if(tmp_number_neuron_units != 0_zu)
            {
                // Assign neurons variable.
                for(tmp_neuron_index = 0_zu; tmp_neuron_index != tmp_number_neuron_units; ++tmp_neuron_index,
                                                                                                                                    ++tmp_ptr_array_neuron_units_first_forward_connection_index,
                                                                                                                                    ++tmp_ptr_array_neuron_units_last_forward_connection_index,
                                                                                                                                    ++tmp_ptr_array_neuron_units_number_forward_connections,
                                                                                                                                    ++tmp_ptr_array_neuron_units_summations,
                                                                                                                                    ++tmp_ptr_array_neuron_units_errors)
                {
                    tmp_ptr_array_neuron_units[tmp_neuron_index].ptr_first_connection_index = tmp_ptr_array_neuron_units_first_forward_connection_index;
                    tmp_ptr_array_neuron_units[tmp_neuron_index].ptr_last_connection_index = tmp_ptr_array_neuron_units_last_forward_connection_index;
                    tmp_ptr_array_neuron_units[tmp_neuron_index].ptr_number_connections = tmp_ptr_array_neuron_units_number_forward_connections;

                    tmp_ptr_array_neuron_units[tmp_neuron_index].ptr_array_summations = tmp_ptr_array_neuron_units_summations;
                    tmp_ptr_array_neuron_units[tmp_neuron_index].ptr_array_errors = tmp_ptr_array_neuron_units_errors;
                }

                tmp_ptr_array_neuron_units_summations += (this->batch_size - 1_zu) * tmp_number_neuron_units * this->number_recurrent_depth + tmp_number_neuron_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_neuron_units_errors += (this->batch_size - 1_zu) * tmp_number_neuron_units * this->number_recurrent_depth + tmp_number_neuron_units * (this->number_recurrent_depth - 1_zu);
                // |END| Assign neurons variable. |END|
                
                tmp_ptr_array_neuron_units += tmp_number_neuron_units;
            }

            tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
        }
        
        this->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;

        this->total_neuron_units_allocated = this->total_neuron_units;
    }

    return(true);
}

bool Neural_Network::Allocate__AF_Units(void)
{
    size_t tmp_number_AF_units,
              tmp_AF_index;
    
    if(this->ptr_array_AF_units == nullptr && this->total_AF_units != 0_zu)
    {
        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct AF_unit *tmp_ptr_array_AF_units(new struct AF_unit[this->total_AF_units]);
        if(tmp_ptr_array_AF_units == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_AF_units * sizeof(struct AF_unit),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_units = tmp_ptr_array_AF_units;
        
        // Allocating AF unit(s) activation steepness.
        T_ *tmp_ptr_array_AF_units_activation_steepness(new T_[this->total_AF_units]);
        if(tmp_ptr_array_AF_units_activation_steepness == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_AF_units * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_units_activation_steepness = tmp_ptr_array_AF_units_activation_steepness;
        MyEA::Memory::Fill<T_>(tmp_ptr_array_AF_units_activation_steepness,
                                   tmp_ptr_array_AF_units_activation_steepness + this->total_AF_units,
                                   1_T);
        // |END| Allocating AF unit(s) activation steepness. |END|
        
        // Allocating AF unit(s) value(s).
        T_ *tmp_ptr_array_AF_units_values(new T_[this->batch_size * this->total_AF_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_AF_units_values == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_AF_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_units_values = tmp_ptr_array_AF_units_values;
        MEMSET(tmp_ptr_array_AF_units_values,
                        0,
                        this->batch_size * this->total_AF_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating AF unit(s) value(s). |END|
        
        // Allocating AF unit(s) error(s).
        T_ *tmp_ptr_array_AF_units_errors(new T_[this->batch_size * this->total_AF_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_AF_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_AF_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_units_errors = tmp_ptr_array_AF_units_errors;
        MEMSET(tmp_ptr_array_AF_units_errors,
                     0,
                     this->batch_size * this->total_AF_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating AF unit(s) error(s). |END|
        
        // Allocating AF unit(s) type activation function(s).
        enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION *tmp_ptr_array_AF_units_type_activations_functions(new enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION[this->total_AF_units]);
        if(tmp_ptr_array_AF_units_type_activations_functions == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_AF_units * sizeof(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_units_type_activation_function = tmp_ptr_array_AF_units_type_activations_functions;
        MyEA::Memory::Fill<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(tmp_ptr_array_AF_units_type_activations_functions,
                                                                                                                              tmp_ptr_array_AF_units_type_activations_functions + this->total_AF_units,
                                                                                                                              MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_NONE);
        // |END| Allocating AF unit(s) type activation function(s). |END|
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_AF_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_unit - tmp_ptr_layer_it->ptr_array_AF_units);

            tmp_ptr_layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
            
            if(tmp_number_AF_units != 0_zu)
            {
                // Assign AF unit(s) variable.
                for(tmp_AF_index = 0_zu; tmp_AF_index != tmp_number_AF_units; ++tmp_AF_index,
                                                                                                              ++tmp_ptr_array_AF_units_activation_steepness,
                                                                                                              ++tmp_ptr_array_AF_units_values,
                                                                                                              ++tmp_ptr_array_AF_units_errors,
                                                                                                              ++tmp_ptr_array_AF_units_type_activations_functions)
                {
                    tmp_ptr_array_AF_units[tmp_AF_index].ptr_activation_steepness = tmp_ptr_array_AF_units_activation_steepness;
                    tmp_ptr_array_AF_units[tmp_AF_index].ptr_array_values = tmp_ptr_array_AF_units_values;
                    tmp_ptr_array_AF_units[tmp_AF_index].ptr_array_errors = tmp_ptr_array_AF_units_errors;

                    tmp_ptr_array_AF_units[tmp_AF_index].ptr_type_activation_function = tmp_ptr_array_AF_units_type_activations_functions;
                }

                tmp_ptr_array_AF_units_values += (this->batch_size - 1_zu) * tmp_number_AF_units * this->number_recurrent_depth + tmp_number_AF_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_AF_units_errors += (this->batch_size - 1_zu) * tmp_number_AF_units * this->number_recurrent_depth + tmp_number_AF_units * (this->number_recurrent_depth - 1_zu);
                // |END| Assign AF unit(s) variable. |END|
                
                tmp_ptr_array_AF_units += tmp_number_AF_units;
            }

            tmp_ptr_layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
        }
        
        this->ptr_last_AF_unit = tmp_ptr_array_AF_units;

        this->total_AF_units_allocated = this->total_AF_units;
    }

    return(true);
}

bool Neural_Network::Allocate__AF_Ind_Recurrent_Units(void)
{
    size_t tmp_number_AF_Ind_recurrent_units,
              tmp_AF_Ind_index;
    
    if(this->ptr_array_AF_Ind_recurrent_units == nullptr && this->total_AF_Ind_recurrent_units != 0_zu)
    {
        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct AF_Ind_recurrent_unit *tmp_ptr_array_AF_Ind_recurrent_units(new struct AF_Ind_recurrent_unit[this->total_AF_Ind_recurrent_units]);
        if(tmp_ptr_array_AF_Ind_recurrent_units == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_AF_Ind_recurrent_units * sizeof(struct AF_Ind_recurrent_unit),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
        
        // Allocating AF Ind recurrent unit(s) first recurrent connection index.
        size_t *tmp_ptr_array_AF_Ind_recurrent_units_first_recurrent_connection_index(new size_t[this->total_AF_Ind_recurrent_units]);
        if(tmp_ptr_array_AF_Ind_recurrent_units_first_recurrent_connection_index == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_AF_Ind_recurrent_units * sizeof(size_t),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_recurrent_connection_index = tmp_ptr_array_AF_Ind_recurrent_units_first_recurrent_connection_index;
        memset(tmp_ptr_array_AF_Ind_recurrent_units_first_recurrent_connection_index,
                    0,
                    this->total_AF_Ind_recurrent_units * sizeof(size_t));
        // |END| Allocating AF Ind recurrent unit(s) first recurrent connection index. |END|
        
        // Allocating AF Ind recurrent unit(s) activation steepness.
        T_ *tmp_ptr_array_AF_Ind_recurrent_units_activation_steepness(new T_[this->total_AF_Ind_recurrent_units]);
        if(tmp_ptr_array_AF_Ind_recurrent_units_activation_steepness == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_AF_Ind_recurrent_units * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_activation_steepness = tmp_ptr_array_AF_Ind_recurrent_units_activation_steepness;
        MyEA::Memory::Fill<T_>(tmp_ptr_array_AF_Ind_recurrent_units_activation_steepness,
                                   tmp_ptr_array_AF_Ind_recurrent_units_activation_steepness + this->total_AF_Ind_recurrent_units,
                                   1_T);
        // |END| Allocating AF Ind recurrent unit(s) activation steepness. |END|
        
        // Allocating AF Ind recurrent unit(s) value(s).
        T_ *tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs(new T_[this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_pre_AFs = tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs;
        MEMSET(tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs,
                       0,
                       this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating AF Ind recurrent unit(s) value(s). |END|
        
        // Allocating AF Ind recurrent unit(s) value(s).
        T_ *tmp_ptr_array_AF_Ind_recurrent_units_AFs(new T_[this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_AF_Ind_recurrent_units_AFs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_AFs = tmp_ptr_array_AF_Ind_recurrent_units_AFs;
        MEMSET(tmp_ptr_array_AF_Ind_recurrent_units_AFs,
                       0,
                       this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating AF Ind recurrent unit(s) value(s). |END|
        
        // Allocating AF Ind recurrent unit(s) error(s).
        T_ *tmp_ptr_array_AF_Ind_recurrent_units_errors(new T_[this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_AF_Ind_recurrent_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_errors = tmp_ptr_array_AF_Ind_recurrent_units_errors;
        MEMSET(tmp_ptr_array_AF_Ind_recurrent_units_errors,
                     0,
                     this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating AF Ind recurrent unit(s) error(s). |END|
        
        // Allocating AF Ind recurrent unit(s) dAF(s).
        T_ *tmp_ptr_array_AF_Ind_recurrent_units_dAFs(new T_[this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_AF_Ind_recurrent_units_dAFs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_dAFs = tmp_ptr_array_AF_Ind_recurrent_units_dAFs;
        MEMSET(tmp_ptr_array_AF_Ind_recurrent_units_dAFs,
                      0,
                      this->batch_size * this->total_AF_Ind_recurrent_units * this->number_recurrent_depth * sizeof(T_));
        // |END| Allocating AF Ind recurrent unit(s) dAF(s). |END|
        
        // Allocating AF Ind recurrent unit(s) type activation function(s).
        enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION *tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions(new enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION[this->total_AF_Ind_recurrent_units]);
        if(tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_AF_Ind_recurrent_units * sizeof(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION),
                                     __LINE__);

            return(false);
        }
        this->ptr_array_AF_Ind_recurrent_units_type_activation_function = tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions;
        MyEA::Memory::Fill<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions,
                                                                                                                              tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions + this->total_AF_Ind_recurrent_units,
                                                                                                                              MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_NONE);
        // |END| Allocating AF Ind recurrent unit(s) type activation function(s). |END|
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_number_AF_Ind_recurrent_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units);

            tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
            
            if(tmp_number_AF_Ind_recurrent_units != 0_zu)
            {
                // Assign AF Ind recurrent unit(s) variable.
                for(tmp_AF_Ind_index = 0_zu; tmp_AF_Ind_index != tmp_number_AF_Ind_recurrent_units; ++tmp_AF_Ind_index,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_first_recurrent_connection_index,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_activation_steepness,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_AFs,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_errors,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_dAFs,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions)
                {
                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_recurrent_connection_index = tmp_ptr_array_AF_Ind_recurrent_units_first_recurrent_connection_index;

                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_activation_steepness = tmp_ptr_array_AF_Ind_recurrent_units_activation_steepness;
                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_array_pre_AFs = tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs;
                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_array_AFs = tmp_ptr_array_AF_Ind_recurrent_units_AFs;
                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_array_errors = tmp_ptr_array_AF_Ind_recurrent_units_errors;
                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_array_dAFs = tmp_ptr_array_AF_Ind_recurrent_units_dAFs;

                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_type_activation_function = tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions;
                }

                tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs += (this->batch_size - 1_zu) * tmp_number_AF_Ind_recurrent_units * this->number_recurrent_depth + tmp_number_AF_Ind_recurrent_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_AF_Ind_recurrent_units_AFs += (this->batch_size - 1_zu) * tmp_number_AF_Ind_recurrent_units * this->number_recurrent_depth + tmp_number_AF_Ind_recurrent_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_AF_Ind_recurrent_units_errors += (this->batch_size - 1_zu) * tmp_number_AF_Ind_recurrent_units * this->number_recurrent_depth + tmp_number_AF_Ind_recurrent_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_AF_Ind_recurrent_units_dAFs += (this->batch_size - 1_zu) * tmp_number_AF_Ind_recurrent_units * this->number_recurrent_depth + tmp_number_AF_Ind_recurrent_units * (this->number_recurrent_depth - 1_zu);
                // |END| Assign AF Ind recurrent unit(s) variable. |END|
                
                tmp_ptr_array_AF_Ind_recurrent_units += tmp_number_AF_Ind_recurrent_units;
            }

            tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
        }
        
        this->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;

        this->total_AF_Ind_recurrent_units_allocated = this->total_AF_Ind_recurrent_units;
    }

    return(true);
}

bool Neural_Network::Allocate__Normalized_Unit(bool const organize_pointers_received)
{
    if(this->ptr_array_normalized_units == nullptr)
    {
        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it;
        
        // TODO: Allocate the exact number of necessary normalized unit(s) per layer.
        if(organize_pointers_received) { this->Prepare__Normalized__Layers(); }

        if(this->total_normalized_units != 0_zu)
        {
            // Normalized unit.
            union Normalized_unit *tmp_ptr_array_normalized_units(new union Normalized_unit[this->total_normalized_units]);
            if(tmp_ptr_array_normalized_units == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->total_normalized_units * sizeof(union Normalized_unit),
                                         __LINE__);

                return(false);
            }
            this->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
            // |END| Normalized unit. |END|

            if(organize_pointers_received)
            {
                size_t tmp_number_normalized_units;

                for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
                {
                    tmp_number_normalized_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_normalized_unit - tmp_ptr_layer_it->ptr_array_normalized_units);

                    tmp_ptr_layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                    tmp_ptr_array_normalized_units += tmp_number_normalized_units;
                    tmp_ptr_layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                }
            }
            
            this->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;

            this->total_normalized_units_allocated = this->total_normalized_units;
        }
    }

    return(true);
}

bool Neural_Network::Allocate__Block_Unit__Mask_Dropout_Zoneout(void)
{
    if(this->ptr_array_cell_units_mask_dropout_zoneout == nullptr)
    {
        bool *tmp_ptr_array_cell_units_mask_dropout_zoneout(new bool[2_zu * this->total_cell_units_allocated * this->number_recurrent_depth]);

        if(tmp_ptr_array_cell_units_mask_dropout_zoneout == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     2_zu * this->total_cell_units_allocated * this->number_recurrent_depth * sizeof(bool),
                                     __LINE__);

            return(false);
        }
        
        MyEA::Memory::Fill<bool>(tmp_ptr_array_cell_units_mask_dropout_zoneout,
                                     tmp_ptr_array_cell_units_mask_dropout_zoneout + 2_zu * this->total_cell_units_allocated * this->number_recurrent_depth,
                                     true);
        
        this->Reset__Parameters__Cell_Unit__Mask_Dropout(tmp_ptr_array_cell_units_mask_dropout_zoneout);
    }

    return(true);
}

bool Neural_Network::Allocate__LSTM_Layers(void)
{
    size_t tmp_number_block_units,
              tmp_number_cell_units,
              tmp_number_cell_units_per_block,
              tmp_block_index,
              tmp_block_index_cell_index;
    
    if(this->total_block_units * this->total_cell_units != 0_zu)
    {
        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it;

        // Allocating block units.
        struct Block_unit *tmp_ptr_array_block_units(new struct Block_unit[this->total_block_units]);
        if(tmp_ptr_array_block_units == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_block_units * sizeof(struct Block_unit),
                                     __LINE__);

            return(false);
        }
        // |END| Allocating block units. |END|
        
        // Allocating cell units.
        struct Cell_unit *tmp_ptr_array_cell_units(new struct Cell_unit[this->total_cell_units]);
        if(tmp_ptr_array_cell_units == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_cell_units * sizeof(struct Cell_unit),
                                     __LINE__);

            return(false);
        }
        // |END| Allocating cell units. |END|
        
        // Allocating summation cell input.
        T_ *tmp_ptr_array_summation_cells_inputs(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_cells_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_cells_inputs,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_summations_cells_inputs = tmp_ptr_array_summation_cells_inputs;
        // |END| Allocating summation cell input. |END|
        
        // Allocating summation input cell input.
        T_ *tmp_ptr_array_summation_input_cells_inputs(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_input_cells_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_input_cells_inputs,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_summations_input_cells_inputs = tmp_ptr_array_summation_input_cells_inputs;
        // |END| Allocating summation input cell input. |END|
        
        // Allocating summation recurrent cell input.
        T_ *tmp_ptr_array_summation_recurrent_cells_inputs(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_recurrent_cells_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_recurrent_cells_inputs,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_summations_recurrent_cells_inputs = tmp_ptr_array_summation_recurrent_cells_inputs;
        // |END| Allocating summation recurrent cell input. |END|
        
        // Allocating block summation input gate.
        T_ *tmp_ptr_array_summation_inputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_inputs_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_summations_inputs_gates = tmp_ptr_array_summation_inputs_gates;
        // |END| Allocating block summation input gate. |END|
        
        // Allocating block summation input input gate.
        T_ *tmp_ptr_array_summation_input_inputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_input_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_input_inputs_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_summations_input_inputs_gates = tmp_ptr_array_summation_input_inputs_gates;
        // |END| Allocating block summation input input gate. |END|
        
        // Allocating block summation recurrent input gate.
        T_ *tmp_ptr_array_summation_recurrent_inputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_recurrent_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_recurrent_inputs_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_summations_recurrent_inputs_gates = tmp_ptr_array_summation_recurrent_inputs_gates;
        // |END| Allocating block summation recurrent input gate. |END|
        
        // Allocating block summation forget gate.
        T_ *tmp_ptr_array_summation_forgets_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_forgets_gates,
                        0,
                        this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_summations_forgets_gates = tmp_ptr_array_summation_forgets_gates;
        // |END| Allocating block summation forget gate. |END|
        
        // Allocating block summation input forget gate.
        T_ *tmp_ptr_array_summation_input_forgets_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_input_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_input_forgets_gates,
                        0,
                        this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_summations_input_forgets_gates = tmp_ptr_array_summation_input_forgets_gates;
        // |END| Allocating block summation input forget gate. |END|
        
        // Allocating block summation recurrent forget gate.
        T_ *tmp_ptr_array_summation_recurrent_forgets_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_recurrent_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_recurrent_forgets_gates,
                        0,
                        this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_summations_recurrent_forgets_gates = tmp_ptr_array_summation_recurrent_forgets_gates;
        // |END| Allocating block summation recurrent forget gate. |END|
        
        // Allocating block summation outputs gate.
        T_ *tmp_ptr_array_summation_outputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_outputs_gates,
                        0,
                        this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_summations_outputs_gates = tmp_ptr_array_summation_outputs_gates;
        // |END| Allocating block summation outputs gate. |END|
        
        // Allocating block summation input outputs gate.
        T_ *tmp_ptr_array_summation_input_outputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_input_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_input_outputs_gates,
                        0,
                        this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_summations_input_outputs_gates = tmp_ptr_array_summation_input_outputs_gates;
        // |END| Allocating block summation input outputs gate. |END|
        
        // Allocating block summation recurrent outputs gate.
        T_ *tmp_ptr_array_summation_recurrent_outputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_summation_recurrent_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_summation_recurrent_outputs_gates,
                        0,
                        this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_summations_recurrent_outputs_gates = tmp_ptr_array_summation_recurrent_outputs_gates;
        // |END| Allocating block summation recurrent outputs gate. |END|
        
        // Allocating cell input.
        T_ *tmp_ptr_array_cells_inputs(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_cells_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_cells_inputs,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_inputs = tmp_ptr_array_cells_inputs;
        // |END| Allocating cell input. |END|
        
        // Allocating cell state.
        T_ *tmp_ptr_array_cells_states(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_cells_states == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_cells_states,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_states = tmp_ptr_array_cells_states;
        // |END| Allocating cell state. |END|
        
        // Allocating cell state activate.
        T_ *tmp_ptr_array_cells_states_activates(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_cells_states_activates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_cells_states_activates,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_states_activates = tmp_ptr_array_cells_states_activates;
        // |END| Allocating cell state activate. |END|
        
        // Allocating cell outputs.
        T_ *tmp_ptr_array_cells_outputs(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_cells_outputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_cells_outputs,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_outputs = tmp_ptr_array_cells_outputs;
        // |END| Allocating cell outputs. |END|
        
        // Allocating block input gate.
        T_ *tmp_ptr_array_inputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_inputs_gates,
                        0,
                        this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_inputs_gates = tmp_ptr_array_inputs_gates;
        // |END| Allocating block input gate. |END|
        
        // Allocating block forget gate.
        T_ *tmp_ptr_array_forgets_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_forgets_gates,
                        0,
                        this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_forgets_gates = tmp_ptr_array_forgets_gates;
        // |END| Allocating block forget gate. |END|
        
        // Allocating block outputs gate.
        T_ *tmp_ptr_array_outputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_outputs_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_outputs_gates = tmp_ptr_array_outputs_gates;
        // |END| Allocating block outputs gate. |END|
        
        // Allocating delta cell inputs.
        T_ *tmp_ptr_array_delta_cells_inputs(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_cells_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_cells_inputs,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_delta_inputs = tmp_ptr_array_delta_cells_inputs;
        // |END| Allocating delta cell inputs. |END|
        
        // Allocating delta cell input inputs.
        T_ *tmp_ptr_array_delta_cells_input_inputs(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_cells_input_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_cells_input_inputs,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_delta_input_inputs = tmp_ptr_array_delta_cells_input_inputs;
        // |END| Allocating delta cell input inputs. |END|
        
        // Allocating delta cell recurrent inputs.
        T_ *tmp_ptr_array_delta_cells_recurrent_inputs(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_cells_recurrent_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_cells_recurrent_inputs,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_delta_recurrent_inputs = tmp_ptr_array_delta_cells_recurrent_inputs;
        // |END| Allocating delta cell recurrent inputs. |END|
        
        // Allocating delta cell state.
        T_ *tmp_ptr_array_delta_cells_states(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_cells_states == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_cells_states,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_delta_states = tmp_ptr_array_delta_cells_states;
        // |END| Allocating delta cell state. |END|
        
        // Allocating delta cell outputs.
        T_ *tmp_ptr_array_delta_cells_outputs(new T_[this->batch_size * this->total_cell_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_cells_outputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_cells_outputs,
                    0,
                    this->batch_size * this->total_cell_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_cells_delta_outputs = tmp_ptr_array_delta_cells_outputs;
        // |END| Allocating delta cell outputs. |END|
        
        // Allocating block delta input gate.
        T_ *tmp_ptr_array_delta_inputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_inputs_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_delta_inputs_gates = tmp_ptr_array_delta_inputs_gates;
        // |END| Allocating block delta input gate. |END|
        
        // Allocating block delta input input gate.
        T_ *tmp_ptr_array_delta_input_inputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_input_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_input_inputs_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_delta_input_inputs_gates = tmp_ptr_array_delta_input_inputs_gates;
        // |END| Allocating block delta input input gate. |END|
        
        // Allocating block delta recurrent input gate.
        T_ *tmp_ptr_array_delta_recurrent_inputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_recurrent_inputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_recurrent_inputs_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_delta_recurrent_inputs_gates = tmp_ptr_array_delta_recurrent_inputs_gates;
        // |END| Allocating block delta recurrent input gate. |END|
        
        // Allocating block delta forget gate.
        T_ *tmp_ptr_array_delta_forgets_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_forgets_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_delta_forgets_gates = tmp_ptr_array_delta_forgets_gates;
        // |END| Allocating block delta forget gate. |END|
        
        // Allocating block delta input forget gate.
        T_ *tmp_ptr_array_delta_input_forgets_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_input_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_input_forgets_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_delta_input_forgets_gates = tmp_ptr_array_delta_input_forgets_gates;
        // |END| Allocating block delta input forget gate. |END|
        
        // Allocating block delta recurrent forget gate.
        T_ *tmp_ptr_array_delta_recurrent_forgets_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_recurrent_forgets_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_recurrent_forgets_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_delta_recurrent_forgets_gates = tmp_ptr_array_delta_recurrent_forgets_gates;
        // |END| Allocating block delta recurrent forget gate. |END|
        
        // Allocating block delta outputs gate.
        T_ *tmp_ptr_array_delta_outputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_outputs_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_delta_outputs_gates = tmp_ptr_array_delta_outputs_gates;
        // |END| Allocating block delta outputs gate. |END|
        
        // Allocating block delta input outputs gate.
        T_ *tmp_ptr_array_delta_input_outputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_input_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_input_outputs_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_delta_input_outputs_gates = tmp_ptr_array_delta_input_outputs_gates;
        // |END| Allocating block delta input outputs gate. |END|
        
        // Allocating block delta recurrent outputs gate.
        T_ *tmp_ptr_array_delta_recurrent_outputs_gates(new T_[this->batch_size * this->total_block_units * this->number_recurrent_depth]);
        if(tmp_ptr_array_delta_recurrent_outputs_gates == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_delta_recurrent_outputs_gates,
                    0,
                    this->batch_size * this->total_block_units * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_blocks_delta_recurrent_outputs_gates = tmp_ptr_array_delta_recurrent_outputs_gates;
        // |END| Allocating block delta recurrent outputs gate. |END|
        
        this->ptr_array_block_units = tmp_ptr_array_block_units;
        this->ptr_array_cell_units = tmp_ptr_array_cell_units;

        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            // [0] Assign block units.
            tmp_number_block_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units);
            tmp_number_cell_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units);
            
            tmp_ptr_layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
            tmp_ptr_layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
            
            if(tmp_number_block_units * tmp_number_cell_units != 0_zu)
            {
                tmp_number_cell_units_per_block = tmp_number_cell_units / tmp_number_block_units;

                //    [1] Assign time step.
                for(tmp_block_index = 0_zu; tmp_block_index != tmp_number_block_units; ++tmp_block_index,
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
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_summation_cells_inputs = tmp_ptr_array_summation_cells_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_summation_input_cells_inputs = tmp_ptr_array_summation_input_cells_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_summation_recurrent_cells_inputs = tmp_ptr_array_summation_recurrent_cells_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_inputs_gates = tmp_ptr_array_summation_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_input_inputs_gates = tmp_ptr_array_summation_input_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_recurrent_inputs_gates = tmp_ptr_array_summation_recurrent_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_forgets_gates = tmp_ptr_array_summation_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_input_forgets_gates = tmp_ptr_array_summation_input_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_recurrent_forgets_gates = tmp_ptr_array_summation_recurrent_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_outputs_gates = tmp_ptr_array_summation_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_input_outputs_gates = tmp_ptr_array_summation_input_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_recurrent_outputs_gates = tmp_ptr_array_summation_recurrent_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_cells_inputs = tmp_ptr_array_cells_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_cells_states = tmp_ptr_array_cells_states;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_cells_states_activates = tmp_ptr_array_cells_states_activates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_cells_outputs = tmp_ptr_array_cells_outputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_inputs_gates = tmp_ptr_array_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_forgets_gates = tmp_ptr_array_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_outputs_gates = tmp_ptr_array_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_delta_cells_inputs = tmp_ptr_array_delta_cells_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_delta_cells_input_inputs = tmp_ptr_array_delta_cells_input_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_delta_cells_recurrent_inputs = tmp_ptr_array_delta_cells_recurrent_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_delta_cells_states = tmp_ptr_array_delta_cells_states;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_delta_cells_outputs = tmp_ptr_array_delta_cells_outputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_inputs_gates = tmp_ptr_array_delta_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_input_inputs_gates = tmp_ptr_array_delta_input_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_recurrent_inputs_gates = tmp_ptr_array_delta_recurrent_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_forgets_gates = tmp_ptr_array_delta_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_input_forgets_gates = tmp_ptr_array_delta_input_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_recurrent_forgets_gates = tmp_ptr_array_delta_recurrent_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_outputs_gates = tmp_ptr_array_delta_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_input_outputs_gates = tmp_ptr_array_delta_input_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_recurrent_outputs_gates = tmp_ptr_array_delta_recurrent_outputs_gates;
                    
                    //        [2] Assign LSTM cells.
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_cell_units = tmp_ptr_array_cell_units;
                    
                    for(tmp_block_index_cell_index = 0_zu; tmp_block_index_cell_index != tmp_number_cell_units_per_block; ++tmp_block_index_cell_index,
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
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_summation_cell_input = tmp_ptr_array_summation_cells_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_summation_input_cell_input = tmp_ptr_array_summation_input_cells_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_summation_recurrent_cell_input = tmp_ptr_array_summation_recurrent_cells_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_cell_input = tmp_ptr_array_cells_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_cell_state = tmp_ptr_array_cells_states;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_cell_state_activate = tmp_ptr_array_cells_states_activates;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_cell_output = tmp_ptr_array_cells_outputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_delta_cell_input = tmp_ptr_array_delta_cells_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_delta_cell_input_input = tmp_ptr_array_delta_cells_input_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_delta_cell_recurrent_input = tmp_ptr_array_delta_cells_recurrent_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_delta_cell_state = tmp_ptr_array_delta_cells_states;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_delta_cell_output = tmp_ptr_array_delta_cells_outputs;
                    }

                    tmp_ptr_array_cell_units += tmp_number_cell_units_per_block;

                    tmp_ptr_array_block_units[tmp_block_index].ptr_last_cell_unit = tmp_ptr_array_cell_units;
                    //        [2] |END| Assign LSTM cells. |END|
                }
                
                tmp_ptr_array_summation_cells_inputs += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_input_cells_inputs += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_recurrent_cells_inputs += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_inputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_input_inputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_recurrent_inputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_forgets_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_input_forgets_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_recurrent_forgets_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_outputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_input_outputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_summation_recurrent_outputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_cells_inputs += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_cells_states += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_cells_states_activates += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_cells_outputs += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_inputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_forgets_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_outputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_cells_inputs += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_cells_input_inputs += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_cells_recurrent_inputs += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_cells_states += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_cells_outputs += (this->batch_size - 1_zu) * tmp_number_cell_units * this->number_recurrent_depth + tmp_number_cell_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_inputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_input_inputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_recurrent_inputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_forgets_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_input_forgets_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_recurrent_forgets_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_outputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_input_outputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                tmp_ptr_array_delta_recurrent_outputs_gates += (this->batch_size - 1_zu) * tmp_number_block_units * this->number_recurrent_depth + tmp_number_block_units * (this->number_recurrent_depth - 1_zu);
                //    [1] |END| Assign time step. |END|

                tmp_ptr_array_block_units += tmp_number_block_units;
            }

            tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;
            tmp_ptr_layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
            // [0] |END| Assign block units. |END|
        }
        
        this->ptr_last_block_unit = tmp_ptr_array_block_units;
        this->ptr_last_cell_unit = tmp_ptr_array_cell_units;

        this->total_block_units_allocated = this->total_block_units;
        this->total_cell_units_allocated = this->total_cell_units;
    }

    return(true);
}

bool Neural_Network::Allocate__Bidirectional__Layers(void)
{
    size_t tmp_number_block_units_per_layer,
              tmp_number_cell_units_per_layer,
              tmp_number_bidirectional_layers(0_zu);
    
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it;
        
    struct Block_unit *tmp_ptr_array_block_units;

    struct Cell_unit *tmp_ptr_array_cell_units;

    for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    { if(tmp_ptr_layer_it->Use__Bidirectional()) { ++tmp_number_bidirectional_layers; } }

    if(tmp_number_bidirectional_layers != 0_zu)
    {
        struct Bidirectional_Layer *tmp_ptr_array_bidirectional_layers(new struct Bidirectional_Layer[tmp_number_bidirectional_layers]);
        if(tmp_ptr_array_bidirectional_layers == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     static_cast<size_t>(tmp_number_bidirectional_layers) * sizeof(struct Bidirectional_Layer),
                                     __LINE__);

            return(false);
        }

        this->ptr_array_bidirectional_layers = tmp_ptr_array_bidirectional_layers;
            
        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_ptr_layer_it->ptr_Bidirectional_Layer = tmp_ptr_array_bidirectional_layers;

            if(tmp_ptr_layer_it->Use__Bidirectional())
            {
                //    [1] Forward layer.
                //        [2] Assign parameters.
                tmp_ptr_array_bidirectional_layers->forward_layer.type_layer = tmp_ptr_layer_it->type_layer;
                tmp_ptr_array_bidirectional_layers->forward_layer.type_activation = tmp_ptr_layer_it->type_activation;
                tmp_ptr_array_bidirectional_layers->forward_layer.type_dropout = tmp_ptr_layer_it->type_dropout;
                tmp_ptr_array_bidirectional_layers->forward_layer.type_normalization = tmp_ptr_layer_it->type_normalization;

                tmp_ptr_array_bidirectional_layers->forward_layer.ptr_number_outputs = tmp_ptr_layer_it->ptr_number_outputs;
                tmp_ptr_array_bidirectional_layers->forward_layer.ptr_first_connection_index = tmp_ptr_layer_it->ptr_first_connection_index;
                tmp_ptr_array_bidirectional_layers->forward_layer.ptr_last_connection_index = tmp_ptr_layer_it->ptr_last_connection_index;
                tmp_ptr_array_bidirectional_layers->forward_layer.first_bias_connection_index = tmp_ptr_layer_it->first_bias_connection_index;
                tmp_ptr_array_bidirectional_layers->forward_layer.last_bias_connection_index = tmp_ptr_layer_it->first_bias_connection_index + (tmp_ptr_layer_it->last_bias_connection_index - tmp_ptr_layer_it->first_bias_connection_index) / 2_zu;

                tmp_ptr_array_bidirectional_layers->forward_layer.dropout_values[0u] = tmp_ptr_layer_it->dropout_values[0u];
                tmp_ptr_array_bidirectional_layers->forward_layer.dropout_values[1u] = tmp_ptr_layer_it->dropout_values[1u];
                tmp_ptr_array_bidirectional_layers->forward_layer.dropout_values[2u] = tmp_ptr_layer_it->dropout_values[2u];
                //        [2] |END| Assign parameters. |END|
                //    [1] |END| Forward layer. |END|
                    
                //    [1] Backward layer.
                //        [2] Assign parameters.
                tmp_ptr_array_bidirectional_layers->backward_layer.type_layer = tmp_ptr_layer_it->type_layer;
                tmp_ptr_array_bidirectional_layers->backward_layer.type_activation = tmp_ptr_layer_it->type_activation;
                tmp_ptr_array_bidirectional_layers->backward_layer.type_dropout = tmp_ptr_layer_it->type_dropout;
                tmp_ptr_array_bidirectional_layers->backward_layer.type_normalization = tmp_ptr_layer_it->type_normalization;

                tmp_ptr_array_bidirectional_layers->backward_layer.ptr_number_outputs = tmp_ptr_layer_it->ptr_number_outputs;
                tmp_ptr_array_bidirectional_layers->backward_layer.ptr_first_connection_index = tmp_ptr_layer_it->ptr_first_connection_index;
                tmp_ptr_array_bidirectional_layers->backward_layer.ptr_last_connection_index = tmp_ptr_layer_it->ptr_last_connection_index;
                tmp_ptr_array_bidirectional_layers->backward_layer.first_bias_connection_index = tmp_ptr_layer_it->first_bias_connection_index + (tmp_ptr_layer_it->last_bias_connection_index - tmp_ptr_layer_it->first_bias_connection_index) / 2_zu;
                tmp_ptr_array_bidirectional_layers->backward_layer.last_bias_connection_index = tmp_ptr_layer_it->last_bias_connection_index;

                tmp_ptr_array_bidirectional_layers->backward_layer.dropout_values[0u] = tmp_ptr_layer_it->dropout_values[0u];
                tmp_ptr_array_bidirectional_layers->backward_layer.dropout_values[1u] = tmp_ptr_layer_it->dropout_values[1u];
                tmp_ptr_array_bidirectional_layers->backward_layer.dropout_values[2u] = tmp_ptr_layer_it->dropout_values[2u];
                //        [2] |END| Assign parameters. |END|
                //    [1] |END| Backward layer. |END|

                switch(tmp_ptr_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                        //    [1] Forward layer.
                        //        [2] Assign block units.
                        tmp_number_block_units_per_layer = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_block_unit - tmp_ptr_layer_it->ptr_array_block_units) >> 1;
                        tmp_number_cell_units_per_layer = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_cell_unit - tmp_ptr_layer_it->ptr_array_cell_units) >> 1;

                        tmp_ptr_array_block_units = tmp_ptr_layer_it->ptr_array_block_units;
                        tmp_ptr_array_bidirectional_layers->forward_layer.ptr_array_block_units = tmp_ptr_array_block_units;
                        tmp_ptr_array_block_units += tmp_number_block_units_per_layer;
                        tmp_ptr_array_bidirectional_layers->forward_layer.ptr_last_block_unit = tmp_ptr_array_block_units;

                        tmp_ptr_array_cell_units = tmp_ptr_layer_it->ptr_array_cell_units;
                        tmp_ptr_array_bidirectional_layers->forward_layer.ptr_array_cell_units = tmp_ptr_array_cell_units;
                        tmp_ptr_array_cell_units += tmp_number_cell_units_per_layer;
                        tmp_ptr_array_bidirectional_layers->forward_layer.ptr_last_cell_unit = tmp_ptr_array_cell_units;
                        //        [2] |END| Assign block units. |END|
                        //    [1] |END| Forward layer. |END|

                        //    [1] Backward layer.
                        //        [2] Assign block units.
                        tmp_ptr_array_bidirectional_layers->backward_layer.ptr_array_block_units = tmp_ptr_array_block_units;
                        tmp_ptr_array_block_units += tmp_number_block_units_per_layer;
                        tmp_ptr_array_bidirectional_layers->backward_layer.ptr_last_block_unit = tmp_ptr_array_block_units;

                        tmp_ptr_array_bidirectional_layers->backward_layer.ptr_array_cell_units = tmp_ptr_array_cell_units;
                        tmp_ptr_array_cell_units += tmp_number_cell_units_per_layer;
                        tmp_ptr_array_bidirectional_layers->backward_layer.ptr_last_cell_unit = tmp_ptr_array_cell_units;
                        //        [2] |END| Assign block units. |END|
                        //    [1] |END| Backward layer. |END|
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_ptr_layer_it->type_layer,
                                                 MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                                 __LINE__);
                        return(false);
                }
                    
                ++tmp_ptr_array_bidirectional_layers;
            }
        }
            
        this->ptr_last_bidirectional_layer = tmp_ptr_array_bidirectional_layers;
    }

    return(true);
}

bool Neural_Network::Allocate__Parameter(void)
{
    // Parameters.
    if((this->ptr_array_parameters = new T_[this->total_parameters]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->total_parameters * sizeof(T_),
                                 __LINE__);

        return(false);
    }
    MEMSET(this->ptr_array_parameters,
                0,
                this->total_parameters * sizeof(T_));
    // |END| Parameters. |END|
    
    // Connections.
    if((this->ptr_array_ptr_connections = new void*[this->total_parameters]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->total_parameters * sizeof(void*),
                                 __LINE__);

        return(false);
    }
    memset(this->ptr_array_ptr_connections,
                0,
                this->total_parameters * sizeof(void*));
    // |END| Connections. |END|

    this->total_parameters_allocated = this->total_parameters;

    this->total_weights_allocated = this->total_weights;

    this->total_bias_allocated = this->total_bias;

    return(true);
}

bool Neural_Network::Allocate__Parameter__Optimizer(void)
{
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD: return(this->Allocate__Parameter__Gradient_Descent());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus: return(this->Allocate__Parameter__iRPROP_minus());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: return(this->Allocate__Parameter__iRPROP_plus());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: return(this->Allocate__Parameter__Adam());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: return(this->Allocate__Parameter__AMSGrad());
        default:
            PRINT_FORMAT("%s: ERROR: Can not allocate parameters of the optimizer (%u | %s)." NEW_LINE,
                                     __FUNCTION__,
                                     this->type_optimizer_function,
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
                return(false);
    }
}

bool Neural_Network::Allocate__Parameter__Gradient_Descent(void)
{
    if(this->learning_momentum != 0_T
      &&
      this->ptr_array_previous_delta_parameters == nullptr)
    {
        if((this->ptr_array_previous_delta_parameters = new T_[this->total_parameters]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_parameters * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(this->ptr_array_previous_delta_parameters,
                    0,
                    this->total_parameters * sizeof(T_));
    }

    return(true);
}

bool Neural_Network::Allocate__Parameter__iRPROP_minus(void)
{
    if(this->ptr_array_previous_steps == nullptr)
    {
        if((this->ptr_array_previous_steps = new T_[this->total_parameters]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_parameters * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        
        MyEA::Memory::Fill<T_>(this->ptr_array_previous_steps,
                                   this->ptr_array_previous_steps + this->total_parameters,
                                   this->rprop_delta_zero);
    }
    
    if(this->ptr_array_previous_derivatives_parameters == nullptr)
    {
        if((this->ptr_array_previous_derivatives_parameters = new T_[this->total_parameters]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_parameters * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(this->ptr_array_previous_derivatives_parameters,
                    0,
                    this->total_parameters * sizeof(T_));
    }

    return(true);
}

bool Neural_Network::Allocate__Parameter__iRPROP_plus(void)
{
    if(this->Allocate__Parameter__iRPROP_minus() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter__iRPROP_minus()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->ptr_array_previous_delta_parameters == nullptr)
    {
        if((this->ptr_array_previous_delta_parameters = new T_[this->total_parameters]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_parameters * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(this->ptr_array_previous_delta_parameters,
                    0,
                    this->total_parameters * sizeof(T_));
    }

    return(true);
}

bool Neural_Network::Allocate__Parameter__Adam(void)
{
    if(this->ptr_array_previous_biased_first_moment == nullptr)
    {
        if((this->ptr_array_previous_biased_first_moment = new T_[this->total_parameters]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_parameters * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(this->ptr_array_previous_biased_first_moment,
                    0,
                    this->total_parameters * sizeof(T_));
    }
    
    if(this->ptr_array_previous_biased_second_moment == nullptr)
    {
        if((this->ptr_array_previous_biased_second_moment = new T_[this->total_parameters]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_parameters * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(this->ptr_array_previous_biased_second_moment,
                    0,
                    this->total_parameters * sizeof(T_));
    }

    return(true);
}

bool Neural_Network::Allocate__Parameter__AMSGrad(void)
{
    if(this->Allocate__Parameter__Adam() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter__Adam()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->ptr_array_previous_biased_second_moment_hat == nullptr)
    {
        if((this->ptr_array_previous_biased_second_moment_hat = new T_[this->total_parameters]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_parameters * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(this->ptr_array_previous_biased_second_moment_hat,
                    0,
                    this->total_parameters * sizeof(T_));
    }

    return(true);
}

bool Neural_Network::Allocate__Parameter__Normalization(void)
{
    // TODO: Reorganasition of the array. [------Weights-----][----Bias----][----Normalized unit----]. Allocating with the size of each layer. No waste of memory.
    if(this->ptr_array_parameters != nullptr)
    {
        // Parameters + ((Scale + Shift)=2) * NormUnits.
        size_t const tmp_new_dimension_parameters(this->total_parameters_allocated + 2_zu * this->total_normalized_units);
        
        if(this->Reallocate__Parameter(tmp_new_dimension_parameters) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Parameter(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_new_dimension_parameters,
                                     __LINE__);

            return(false);
        }

        // Clear shift array.
        MEMSET(this->ptr_array_parameters + this->total_weights_allocated + this->total_bias_allocated + this->total_normalized_units,
                       0,
                       this->total_normalized_units * sizeof(T_));
    }
    else { return(false); }

    return(true);
}

bool Neural_Network::Allocate__Parameter__Regularization(void)
{
    if(this->ptr_array_mask_regularized_parameters == nullptr)
    {
        if((this->ptr_array_mask_regularized_parameters = new T_[this->total_parameters_allocated]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_parameters_allocated * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        
        if(this->total_weights_allocated + this->total_bias_allocated < this->total_parameters_allocated)
        {
            // Do not regularize parameters than is not a weight
            MEMSET(this->ptr_array_mask_regularized_parameters + this->total_weights_allocated + this->total_bias_allocated,
                          0,
                          (this->total_parameters_allocated - this->total_weights_allocated - this->total_bias_allocated) * sizeof(T_));
        }
    }

    return(true);
}

bool Neural_Network::Allocate__Normalized_Unit__Batch_Normalization(void)
{
    if(this->ptr_array_normalized_batch_units_values_hats == nullptr
      &&
      this->ptr_array_normalized_batch_units_values_normalizes == nullptr
      &&
      this->ptr_array_normalized_batch_units_means == nullptr
      &&
      this->ptr_array_normalized_batch_units_variances == nullptr
      &&
      this->ptr_array_normalized_batch_units_derivatives_means == nullptr
      &&
      this->ptr_array_normalized_batch_units_derivatives_variances == nullptr
      &&
      this->ptr_array_normalized_batch_units_means_averages == nullptr
      &&
      this->ptr_array_normalized_batch_units_variances_averages == nullptr
      &&
      this->ptr_array_normalized_batch_units_errors == nullptr)
    {
        size_t tmp_number_units,
                  tmp_index;
        
        void **tmp_ptr_array_ptr_connections(this->ptr_array_ptr_connections + this->total_weights_allocated + this->total_bias_allocated);
        
        T_ *tmp_ptr_array_parameters_scale_it(this->ptr_array_parameters + this->total_weights_allocated + this->total_bias_allocated),
             *tmp_ptr_array_parameters_shift_it(this->ptr_array_parameters + this->total_weights_allocated + this->total_bias_allocated + this->total_normalized_units_allocated),
             *tmp_ptr_array_derivatives_parameters_scale_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated + this->total_bias_allocated),
             *tmp_ptr_array_derivatives_parameters_shift_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated + this->total_bias_allocated + this->total_normalized_units_allocated);
        
        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
        
        struct Block_unit const *tmp_ptr_last_block_unit;
        struct Block_unit *tmp_ptr_block_unit_it;
        
        struct Cell_unit const *tmp_ptr_last_cell_unit;
        struct Cell_unit *tmp_ptr_cell_unit_it;
        
        union Normalized_unit const *tmp_ptr_last_normalized_unit;
        union Normalized_unit *tmp_ptr_normalized_unit_it;
        
        // Allocating normalized unit value hat.
        T_ *tmp_ptr_array_normalized_units_values_hat(new T_[this->batch_size * this->number_recurrent_depth * this->total_normalized_units_allocated]);
        if(tmp_ptr_array_normalized_units_values_hat == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->number_recurrent_depth * this->total_normalized_units_allocated * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_normalized_units_values_hat,
                     0,
                     this->batch_size * this->number_recurrent_depth * this->total_normalized_units_allocated * sizeof(T_));
        this->ptr_array_normalized_batch_units_values_hats = tmp_ptr_array_normalized_units_values_hat;
        // |END| Allocating normalized unit value hat. |END|
        
        // Allocating normalized unit value normalize.
        T_ *tmp_ptr_array_normalized_units_values_normalize(new T_[this->batch_size * this->number_recurrent_depth * this->total_normalized_units_allocated]);
        if(tmp_ptr_array_normalized_units_values_normalize == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->number_recurrent_depth * this->total_normalized_units_allocated * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_normalized_units_values_normalize,
                     0,
                     this->batch_size * this->number_recurrent_depth * this->total_normalized_units_allocated * sizeof(T_));
        this->ptr_array_normalized_batch_units_values_normalizes = tmp_ptr_array_normalized_units_values_normalize;
        // |END| Allocating normalized unit value normalize. |END|
        
        // Allocating normalized unit mean.
        T_ *tmp_ptr_array_normalized_units_mean_it(new T_[this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth]);
        if(tmp_ptr_array_normalized_units_mean_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_normalized_units_mean_it,
                     0,
                     this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_normalized_batch_units_means = tmp_ptr_array_normalized_units_mean_it;
        // |END| Allocating normalized unit mean. |END|
        
        // Allocating normalized unit variance.
        T_ *tmp_ptr_array_normalized_units_variance_it(new T_[this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth]);
        if(tmp_ptr_array_normalized_units_variance_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_normalized_units_variance_it,
                     0,
                     this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_normalized_batch_units_variances = tmp_ptr_array_normalized_units_variance_it;
        // |END| Allocating normalized unit variance. |END|
        
        // Allocating normalized unit derivative mean.
        T_ *tmp_ptr_array_normalized_units_derivative_mean_it(new T_[this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth]);
        if(tmp_ptr_array_normalized_units_derivative_mean_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_normalized_units_derivative_mean_it,
                     0,
                     this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_normalized_batch_units_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it;
        // |END| Allocating normalized unit derivative mean. |END|
        
        // Allocating normalized unit derivative variance.
        T_ *tmp_ptr_array_normalized_units_derivative_variance_it(new T_[this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth]);
        if(tmp_ptr_array_normalized_units_derivative_variance_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_normalized_units_derivative_variance_it,
                     0,
                     this->number_threads * this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_normalized_batch_units_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it;
        // |END| Allocating normalized unit derivative variance. |END|
        
        // Allocating normalized unit mean average.
        T_ *tmp_ptr_array_normalized_units_mean_average_it(new T_[this->total_normalized_units_allocated * this->number_recurrent_depth]);
        if(tmp_ptr_array_normalized_units_mean_average_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_normalized_units_allocated * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_normalized_units_mean_average_it,
                     0,
                     this->total_normalized_units_allocated * this->number_recurrent_depth * sizeof(T_));
        this->ptr_array_normalized_batch_units_means_averages = tmp_ptr_array_normalized_units_mean_average_it;
        // |END| Allocating normalized unit mean average. |END|
        
        // Allocating normalized unit variance average.
        T_ *tmp_ptr_array_normalized_units_variance_average_it(new T_[this->total_normalized_units_allocated * this->number_recurrent_depth]);
        if(tmp_ptr_array_normalized_units_variance_average_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->total_normalized_units_allocated * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MyEA::Memory::Fill<T_>(tmp_ptr_array_normalized_units_variance_average_it,
                                   tmp_ptr_array_normalized_units_variance_average_it + this->total_normalized_units_allocated * this->number_recurrent_depth,
                                   1_T);
        this->ptr_array_normalized_batch_units_variances_averages = tmp_ptr_array_normalized_units_variance_average_it;
        // |END| Allocating normalized unit variance average. |END|
        
        // Allocating normalized unit error(s).
        T_ *tmp_ptr_array_normalized_units_errors(new T_[this->batch_size * this->number_recurrent_depth * this->total_normalized_units_allocated]);
        if(tmp_ptr_array_normalized_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->batch_size * this->number_recurrent_depth * this->total_normalized_units_allocated * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_normalized_units_errors,
                     0,
                     this->batch_size * this->number_recurrent_depth * this->total_normalized_units_allocated * sizeof(T_));
        this->ptr_array_normalized_batch_units_errors = tmp_ptr_array_normalized_units_errors;
        // |END| Allocating normalized unit error(s). |END|
        
        this->ptr_array_normalized_batch_units_scales = tmp_ptr_array_parameters_scale_it;
        this->ptr_array_normalized_batch_units_shifts = tmp_ptr_array_parameters_shift_it;
        
        this->ptr_array_normalized_batch_units_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it;
        this->ptr_array_normalized_batch_units_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it;
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            if((tmp_number_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_normalized_unit - tmp_ptr_layer_it->ptr_array_normalized_units)) != 0_zu)
            {
                // Initialize values.
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
                                                                                                                                                                                                                                   ++tmp_ptr_array_derivatives_parameters_scale_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_derivatives_parameters_shift_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_values_hat,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_values_normalize,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_mean_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_variance_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_derivative_mean_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_derivative_variance_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_mean_average_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_variance_average_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_errors,
                                                                                                                                                                                                                                   ++tmp_ptr_array_ptr_connections)
                        {
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_scale = tmp_ptr_array_parameters_scale_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_shift = tmp_ptr_array_parameters_shift_it;

                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it;
                            
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_values_hats = tmp_ptr_array_normalized_units_values_hat;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_values_normalizes = tmp_ptr_array_normalized_units_values_normalize;
                            
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it;
                            
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it;
                            
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_mean_average = tmp_ptr_array_normalized_units_mean_average_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_variance_average = tmp_ptr_array_normalized_units_variance_average_it;

                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors;

                            tmp_ptr_array_ptr_connections[0u] = tmp_ptr_normalized_unit_it;
                            tmp_ptr_array_ptr_connections[this->total_normalized_units_allocated] = tmp_ptr_normalized_unit_it;
                        }
                        
                        tmp_ptr_array_normalized_units_values_hat += (this->batch_size - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        tmp_ptr_array_normalized_units_values_normalize += (this->batch_size - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        
                        tmp_ptr_array_normalized_units_mean_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        tmp_ptr_array_normalized_units_variance_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        
                        tmp_ptr_array_normalized_units_derivative_mean_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        tmp_ptr_array_normalized_units_derivative_variance_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        
                        tmp_ptr_array_normalized_units_mean_average_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        tmp_ptr_array_normalized_units_variance_average_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);

                        tmp_ptr_array_normalized_units_errors += (this->batch_size - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
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
                                
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_scale = tmp_ptr_array_parameters_scale_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_shift = tmp_ptr_array_parameters_shift_it++;

                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it++;
                                
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it++;
                                
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it++;
                                
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_mean_average = tmp_ptr_array_normalized_units_mean_average_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_variance_average = tmp_ptr_array_normalized_units_variance_average_it++;
                                    
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors++;
                                
                                    tmp_ptr_array_ptr_connections[0u] = tmp_ptr_cell_unit_it;
                                    tmp_ptr_array_ptr_connections[this->total_normalized_units_allocated] = tmp_ptr_cell_unit_it;
                                    ++tmp_ptr_array_ptr_connections;
                                }
                            
                                tmp_ptr_array_normalized_units_values_hat += (this->batch_size - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_values_normalize += (this->batch_size - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                            
                                tmp_ptr_array_normalized_units_mean_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_variance_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                            
                                tmp_ptr_array_normalized_units_derivative_mean_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_derivative_variance_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                            
                                tmp_ptr_array_normalized_units_mean_average_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_variance_average_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);

                                tmp_ptr_array_normalized_units_errors += (this->batch_size - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
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
                                
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_scale = tmp_ptr_array_parameters_scale_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_shift = tmp_ptr_array_parameters_shift_it++;

                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it++;
                                
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it++;
                                
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it++;
                                
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_mean_average = tmp_ptr_array_normalized_units_mean_average_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_variance_average = tmp_ptr_array_normalized_units_variance_average_it++;
                                    
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors++;

                                    tmp_ptr_array_ptr_connections[0u] = tmp_ptr_block_unit_it;
                                    tmp_ptr_array_ptr_connections[this->total_normalized_units_allocated] = tmp_ptr_block_unit_it;
                                    ++tmp_ptr_array_ptr_connections;
                                }
                            
                                tmp_ptr_array_normalized_units_values_hat += (this->batch_size - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_values_normalize += (this->batch_size - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                            
                                tmp_ptr_array_normalized_units_mean_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_variance_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                            
                                tmp_ptr_array_normalized_units_derivative_mean_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_derivative_variance_it += (this->number_threads - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                            
                                tmp_ptr_array_normalized_units_mean_average_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_variance_average_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                
                                tmp_ptr_array_normalized_units_errors += (this->batch_size - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
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
                
                tmp_number_units = static_cast<size_t>(tmp_ptr_layer_it->ptr_last_normalized_unit - tmp_ptr_layer_it->ptr_array_normalized_units);

                // Initialize scale.
                switch(tmp_ptr_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                        MyEA::Memory::Fill<T_>(tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale,
                                                  tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale + tmp_number_units,
                                                  1_T);
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                        MyEA::Memory::Fill<T_>(tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale,
                                                  tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale + tmp_number_units,
                                                  0.1_T);
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                        MyEA::Memory::Fill<T_>(tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale,
                                                  tmp_ptr_layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale + tmp_number_units,
                                                  this->number_recurrent_depth == 1_zu ? 1_T : 0.1_T);
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
    else { return(false); }

    return(true);
}

bool Neural_Network::Allocate__Normalized_Unit__Batch_Renormalization(void)
{
    if(this->ptr_array_normalized_batch_units_r_corrections == nullptr && this->ptr_array_normalized_batch_units_d_corrections == nullptr)
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
        
        // Allocating normalized unit(s) r correction.
        T_ *tmp_ptr_array_normalized_units_r_correction_it(new T_[this->number_recurrent_depth * this->total_normalized_units_allocated]);
        if(tmp_ptr_array_normalized_units_r_correction_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_recurrent_depth * this->total_normalized_units_allocated * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_normalized_units_r_correction_it,
                     0,
                     this->number_recurrent_depth * this->total_normalized_units_allocated * sizeof(T_));
        // |END| Allocating normalized unit(s) r correction. |END|
        
        // Allocating normalized unit(s) d correction.
        T_ *tmp_ptr_array_normalized_units_d_correction_it(new T_[this->number_recurrent_depth * this->total_normalized_units_allocated]);
        if(tmp_ptr_array_normalized_units_d_correction_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_recurrent_depth * this->total_normalized_units_allocated * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_normalized_units_d_correction_it,
                     0,
                     this->number_recurrent_depth * this->total_normalized_units_allocated * sizeof(T_));
        // |END| Allocating normalized unit(s) d correction. |END|
        
        this->ptr_array_normalized_batch_units_r_corrections = tmp_ptr_array_normalized_units_r_correction_it;
        this->ptr_array_normalized_batch_units_d_corrections = tmp_ptr_array_normalized_units_d_correction_it;
        
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
                                                                                                                                                                                                                                  ++tmp_ptr_array_normalized_units_r_correction_it,
                                                                                                                                                                                                                                  ++tmp_ptr_array_normalized_units_d_correction_it)
                        {
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_r_correction = tmp_ptr_array_normalized_units_r_correction_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_d_correction = tmp_ptr_array_normalized_units_d_correction_it;
                        }

                        tmp_ptr_array_normalized_units_r_correction_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        tmp_ptr_array_normalized_units_d_correction_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);
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
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_r_correction = tmp_ptr_array_normalized_units_r_correction_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_d_correction = tmp_ptr_array_normalized_units_d_correction_it++;
                                }

                                tmp_ptr_array_normalized_units_r_correction_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_d_correction_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);
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
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_r_correction = tmp_ptr_array_normalized_units_r_correction_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_d_correction = tmp_ptr_array_normalized_units_d_correction_it++;
                                }

                                tmp_ptr_array_normalized_units_r_correction_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_d_correction_it += tmp_number_units * (this->number_recurrent_depth - 1_zu);
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
    else { return(false); }

    return(true);
}
