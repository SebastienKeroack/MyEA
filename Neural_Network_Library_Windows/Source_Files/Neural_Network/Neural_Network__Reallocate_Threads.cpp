#include "stdafx.hpp"

#include <chrono>

#include <Neural_Network/Neural_Network.hpp>

bool Neural_Network::Reallocate__Thread(size_t const number_threads_received)
{
    if(this->Reallocate__Thread__Cost(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread__Cost(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_threads_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Reallocate__Thread__Normalized_Unit__Batch_Normalization(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread__Normalized_Unit__Batch_Normalization(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_threads_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Reallocate__Thread__Parameter(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread__Parameter(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_threads_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Use__K_Sparse() && this->Reallocate__Thread__Sparse_K_Filter(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread__Sparse_K_Filter(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_threads_received,
                                 __LINE__);

        return(false);
    }
    else if((this->Use__Dropout__Bernoulli() || this->Use__Dropout__Bernoulli__Inverted()) && this->Reallocate__Thread__Generator__Dropout__Bernoulli(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread__Generator__Dropout__Bernoulli(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_threads_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Use__Dropout__Gaussian() && this->Reallocate__Thread__Generator__Dropout__Gaussian(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread__Generator__Dropout__Gaussian(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_threads_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Use__Dropout__ShakeDrop() && this->Reallocate__Thread__Generator__Dropout__ShakeDrop(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread__Generator__Dropout__ShakeDrop(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_threads_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Use__Dropout__Uout() && this->Reallocate__Thread__Generator__Dropout__Uout(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread__Generator__Dropout__Uout(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_threads_received,
                                 __LINE__);

        return(false);
    }
    else if(this->Use__Dropout__Zoneout() && this->Reallocate__Thread__Generator__Dropout__Zoneout(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread__Generator__Dropout__Zoneout(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_threads_received,
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::Reallocate__Thread__Sparse_K_Filter(size_t const number_threads_received)
{
    this->ptr_array_k_sparse_activities = Memory::reallocate_objects_cpp<std::pair<size_t, T_>>(this->ptr_array_k_sparse_activities,
                                                                                                                                            number_threads_received * (this->total_basic_units_allocated + this->total_basic_indice_units_allocated + this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated + this->total_cell_units_allocated),
                                                                                                                                            this->number_threads * (this->total_basic_units_allocated + this->total_basic_indice_units_allocated + this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated + this->total_cell_units_allocated),
                                                                                                                                            false);
    if(this->ptr_array_k_sparse_activities == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(std::pair<size_t, T_>),
                                 number_threads_received * (this->total_basic_units_allocated + this->total_basic_indice_units_allocated + this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated + this->total_cell_units_allocated),
                                 this->number_threads * (this->total_basic_units_allocated + this->total_basic_indice_units_allocated + this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated + this->total_cell_units_allocated),
                                 __LINE__);

        return(false);
    }
    
    this->Assign__Sparsity_Activities(number_threads_received);

    return(true);
}

bool Neural_Network::Reallocate__Thread__Cost(size_t const number_threads_received)
{        
    // Reallocate number loss.
    size_t *tmp_ptr_array_number_loss(Memory::reallocate_cpp<size_t>(this->ptr_array_number_loss,
                                                                                                         number_threads_received,
                                                                                                         this->number_threads,
                                                                                                         false));
    if(tmp_ptr_array_number_loss == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(size_t),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_number_loss = tmp_ptr_array_number_loss;
    // |END| Reallocate number loss. |END|
        
    // Reallocate number loss.
    size_t *tmp_ptr_array_bit_fail_values(Memory::reallocate_cpp<size_t>(this->ptr_array_number_bit_fail,
                                                                                                          number_threads_received,
                                                                                                          this->number_threads,
                                                                                                          false));
    if(tmp_ptr_array_bit_fail_values == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(size_t),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_number_bit_fail = tmp_ptr_array_bit_fail_values;
    // |END| Reallocate number loss. |END|
    
    // Reallocate loss values.
    T_ *tmp_ptr_array_loss_values(Memory::reallocate_cpp<T_>(this->ptr_array_loss_values,
                                                                                             number_threads_received,
                                                                                             this->number_threads,
                                                                                             false));
    if(tmp_ptr_array_loss_values == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_loss_values = tmp_ptr_array_loss_values;
    // |END| Reallocate loss values. |END|

    // Reallocate number accuracy value.
    T_ *tmp_ptr_array_number_accuracy_value(Memory::reallocate_cpp<T_>(this->ptr_array_accuracy_values[0u],
                                                                                                               number_threads_received,
                                                                                                               this->number_threads,
                                                                                                               false));
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_accuracy_values[0u] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<T_>(this->ptr_array_accuracy_values[1u],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_accuracy_values[1u] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<T_>(this->ptr_array_accuracy_values[2u],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_accuracy_values[2u] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<T_>(this->ptr_array_accuracy_values[3u],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_accuracy_values[3u] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<T_>(this->ptr_array_accuracy_values[4u],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_accuracy_values[4u] = tmp_ptr_array_number_accuracy_value;
    // |END| Reallocate number accuracy value. |END|

    return(true);
}

bool Neural_Network::Reallocate__Thread__Normalized_Unit__Batch_Normalization(size_t const number_threads_received)
{
    if(this->Use__Normalization()
      &&
      this->ptr_array_normalized_batch_units_means != nullptr
      &&
      this->ptr_array_normalized_batch_units_variances != nullptr
      &&
      this->ptr_array_normalized_batch_units_derivatives_means != nullptr
      &&
      this->ptr_array_normalized_batch_units_derivatives_variances != nullptr)
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
        
        // Allocating normalized unit(s) mean.
        T_ *tmp_ptr_array_normalized_units_mean_it(Memory::reallocate_cpp<T_>(this->ptr_array_normalized_batch_units_means,
                                                                                                                     number_threads_received * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                                                                                                     this->number_threads * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                                                                                                     false));
        if(tmp_ptr_array_normalized_units_mean_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_threads_received * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                     this->number_threads * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                     __LINE__);

            return(false);
        }
        // |END| Allocating normalized unit(s) mean. |END|
        
        // Allocating normalized unit(s) variance.
        T_ *tmp_ptr_array_normalized_units_variance_it(Memory::reallocate_cpp<T_>(this->ptr_array_normalized_batch_units_variances,
                                                                                                                         number_threads_received * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                                                                                                         this->number_threads * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                                                                                                         false));
        if(tmp_ptr_array_normalized_units_variance_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_threads_received * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                     this->number_threads * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                     __LINE__);

            return(false);
        }
        // |END| Allocating normalized unit(s) variance. |END|
        
        // Allocating normalized unit(s) derivative mean.
        T_ *tmp_ptr_array_normalized_units_derivative_mean_it(Memory::reallocate_cpp<T_>(this->ptr_array_normalized_batch_units_derivatives_means,
                                                                                                                                    number_threads_received * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                                                                                                                    this->number_threads * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                                                                                                                    false));
        if(tmp_ptr_array_normalized_units_derivative_mean_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_threads_received * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                     this->number_threads * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                     __LINE__);

            return(false);
        }
        // |END| Allocating normalized unit(s) derivative mean. |END|
        
        // Allocating normalized unit(s) derivative variance.
        T_ *tmp_ptr_array_normalized_units_derivative_variance_it(Memory::reallocate_cpp<T_>(this->ptr_array_normalized_batch_units_derivatives_variances,
                                                                                                                                        number_threads_received * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                                                                                                                        this->number_threads * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                                                                                                                        false));
        if(tmp_ptr_array_normalized_units_derivative_variance_it == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_threads_received * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                     this->number_threads * this->number_recurrent_depth * this->total_normalized_units_allocated,
                                     __LINE__);

            return(false);
        }
        // |END| Allocating normalized unit(s) derivative variance. |END|
        
        this->ptr_array_normalized_batch_units_means = tmp_ptr_array_normalized_units_mean_it;
        this->ptr_array_normalized_batch_units_variances = tmp_ptr_array_normalized_units_variance_it;
        this->ptr_array_normalized_batch_units_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it;
        this->ptr_array_normalized_batch_units_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it;
        
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
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_mean_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_variance_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_derivative_mean_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_derivative_variance_it)
                        {
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it;
                            
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it;
                        }

                        tmp_ptr_array_normalized_units_mean_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        tmp_ptr_array_normalized_units_variance_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        
                        tmp_ptr_array_normalized_units_derivative_mean_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                        tmp_ptr_array_normalized_units_derivative_variance_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
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
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it++;
                                
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it++;
                                }
                            
                                tmp_ptr_array_normalized_units_mean_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_variance_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                            
                                tmp_ptr_array_normalized_units_derivative_mean_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_derivative_variance_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
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
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it++;
                                    
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it++;
                                }
                            
                                tmp_ptr_array_normalized_units_mean_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_variance_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                            
                                tmp_ptr_array_normalized_units_derivative_mean_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                                tmp_ptr_array_normalized_units_derivative_variance_it += (number_threads_received - 1_zu) * tmp_number_units * this->number_recurrent_depth + tmp_number_units * (this->number_recurrent_depth - 1_zu);
                            }
                        } 
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
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

bool Neural_Network::Reallocate__Thread__Parameter(size_t const number_threads_received)
{
    if(this->total_parameters_allocated != 0_zu)
    {
        // Derivates parameters.
        if(this->ptr_array_derivatives_parameters != nullptr)
        {
            T_ *tmp_ptr_array_derivatives_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_derivatives_parameters,
                                                                                                                    number_threads_received * this->total_parameters_allocated,
                                                                                                                    this->number_threads * this->total_parameters_allocated));
            if(tmp_ptr_array_derivatives_parameters == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         sizeof(T_),
                                         number_threads_received * this->total_parameters_allocated,
                                         this->number_threads * this->total_parameters_allocated,
                                         __LINE__);

                return(false);
            }
            this->ptr_array_derivatives_parameters = tmp_ptr_array_derivatives_parameters;

            if(this->Use__Normalization()) { this->Reset__Derivative_Parameter__Normalized_Unit(); }
        }
        // |END| Derivates parameters. |END|
    }

    return(true);
}

bool Neural_Network::Reallocate__Thread__Generator__Dropout__Bernoulli(size_t const number_threads_received)
{
    if(this->ptr_array_Class_Generator_Bernoulli != nullptr)
    {
        class MyEA::Common::Class_Generator_Random_Bernoulli<T_> *tmp_ptr_array_Class_Generator_Random_Bernoulli(Memory::reallocate_objects_cpp<class MyEA::Common::Class_Generator_Random_Bernoulli<T_>>(this->ptr_array_Class_Generator_Bernoulli,
                                                                                                                                                                                                                                                                                                                                      number_threads_received,
                                                                                                                                                                                                                                                                                                                                      this->number_threads));
        if(tmp_ptr_array_Class_Generator_Random_Bernoulli == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(class MyEA::Common::Class_Generator_Random_Bernoulli<T_>),
                                     number_threads_received,
                                     this->number_threads,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_Class_Generator_Bernoulli = tmp_ptr_array_Class_Generator_Random_Bernoulli;

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        { tmp_ptr_array_Class_Generator_Random_Bernoulli[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    return(true);
}

bool Neural_Network::Reallocate__Thread__Generator__Dropout__Zoneout(size_t const number_threads_received)
{
    if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State != nullptr)
    {
        class MyEA::Common::Class_Generator_Random_Bernoulli<T_> *tmp_ptr_array_Class_Generator_Random_Zoneout(Memory::reallocate_objects_cpp<class MyEA::Common::Class_Generator_Random_Bernoulli<T_>>(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State,
                                                                                                                                                                                                                                                                                                                                     number_threads_received,
                                                                                                                                                                                                                                                                                                                                     this->number_threads));
        if(tmp_ptr_array_Class_Generator_Random_Zoneout == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(class MyEA::Common::Class_Generator_Random_Bernoulli<T_>),
                                     number_threads_received,
                                     this->number_threads,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_Class_Generator_Bernoulli_Zoneout_State = tmp_ptr_array_Class_Generator_Random_Zoneout;

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        { tmp_ptr_array_Class_Generator_Random_Zoneout[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }
    
    if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden != nullptr)
    {
        class MyEA::Common::Class_Generator_Random_Bernoulli<T_> *tmp_ptr_array_Class_Generator_Random_Hidden(Memory::reallocate_objects_cpp<class MyEA::Common::Class_Generator_Random_Bernoulli<T_>>(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden,
                                                                                                                                                                                                                                                                                                                                    number_threads_received,
                                                                                                                                                                                                                                                                                                                                    this->number_threads));
        if(tmp_ptr_array_Class_Generator_Random_Hidden == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(class MyEA::Common::Class_Generator_Random_Bernoulli<T_>),
                                     number_threads_received,
                                     this->number_threads,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden = tmp_ptr_array_Class_Generator_Random_Hidden;

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        { tmp_ptr_array_Class_Generator_Random_Hidden[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    return(true);
}

bool Neural_Network::Reallocate__Thread__Generator__Dropout__Gaussian(size_t const number_threads_received)
{
    if(this->ptr_array_Class_Generator_Real_Gaussian != nullptr)
    {
        class MyEA::Common::Class_Generator_Random_Gaussian<T_> *tmp_ptr_array_Class_Generator_Random_Gaussian(Memory::reallocate_objects_cpp<class MyEA::Common::Class_Generator_Random_Gaussian<T_>>(this->ptr_array_Class_Generator_Real_Gaussian,
                                                                                                                                                                                                                                                                                                                                          number_threads_received,
                                                                                                                                                                                                                                                                                                                                          this->number_threads));
        if(tmp_ptr_array_Class_Generator_Random_Gaussian == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(class MyEA::Common::Class_Generator_Random_Gaussian<T_>),
                                     number_threads_received,
                                     this->number_threads,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_Class_Generator_Real_Gaussian = tmp_ptr_array_Class_Generator_Random_Gaussian;

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        {
            tmp_ptr_array_Class_Generator_Random_Gaussian[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            tmp_ptr_array_Class_Generator_Random_Gaussian[tmp_generator_index].Range(0_T, 1_T);
        }
    }

    return(true);
}

bool Neural_Network::Reallocate__Thread__Generator__Dropout__ShakeDrop(size_t const number_threads_received)
{
    if(this->ptr_array_Class_Generator_Bernoulli_ShakeDrop != nullptr)
    {
        class MyEA::Common::Class_Generator_Random_Bernoulli<T_> *tmp_ptr_array_Class_Generator_Random_Bernoulli_ShakeDrop(Memory::reallocate_objects_cpp<class MyEA::Common::Class_Generator_Random_Bernoulli<T_>>(this->ptr_array_Class_Generator_Bernoulli_ShakeDrop,
                                                                                                                                                                                                                                                                                                                                                        number_threads_received,
                                                                                                                                                                                                                                                                                                                                                        this->number_threads));
        if(tmp_ptr_array_Class_Generator_Random_Bernoulli_ShakeDrop == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(class MyEA::Common::Class_Generator_Random_Bernoulli<T_>),
                                     number_threads_received,
                                     this->number_threads,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_Class_Generator_Bernoulli_ShakeDrop = tmp_ptr_array_Class_Generator_Random_Bernoulli_ShakeDrop;

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        { tmp_ptr_array_Class_Generator_Random_Bernoulli_ShakeDrop[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    if(this->ptr_array_Class_Generator_Real_ShakeDrop != nullptr)
    {
        class MyEA::Common::Class_Generator_Random_Real<T_> *tmp_ptr_array_Class_Generator_Random_ShakeDrop(Memory::reallocate_objects_cpp<class MyEA::Common::Class_Generator_Random_Real<T_>>(this->ptr_array_Class_Generator_Real_ShakeDrop,
                                                                                                                                                                                                                                                                                                                              number_threads_received,
                                                                                                                                                                                                                                                                                                                              this->number_threads));
        if(tmp_ptr_array_Class_Generator_Random_ShakeDrop == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(class MyEA::Common::Class_Generator_Random_Real<T_>),
                                     number_threads_received,
                                     this->number_threads,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_Class_Generator_Real_ShakeDrop = tmp_ptr_array_Class_Generator_Random_ShakeDrop;

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        {
            tmp_ptr_array_Class_Generator_Random_ShakeDrop[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            tmp_ptr_array_Class_Generator_Random_ShakeDrop[tmp_generator_index].Range(0_T, 1_T);
        }
    }

    return(true);
}

bool Neural_Network::Reallocate__Thread__Generator__Dropout__Uout(size_t const number_threads_received)
{
    if(this->ptr_array_Class_Generator_Real_Uout != nullptr)
    {
        class MyEA::Common::Class_Generator_Random_Real<T_> *tmp_ptr_array_Class_Generator_Random_Uout(Memory::reallocate_objects_cpp<class MyEA::Common::Class_Generator_Random_Real<T_>>(this->ptr_array_Class_Generator_Real_Uout,
                                                                                                                                                                                                                                                                                                                    number_threads_received,
                                                                                                                                                                                                                                                                                                                    this->number_threads));
        if(tmp_ptr_array_Class_Generator_Random_Uout == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(class MyEA::Common::Class_Generator_Random_Real<T_>),
                                     number_threads_received,
                                     this->number_threads,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_Class_Generator_Real_Uout = tmp_ptr_array_Class_Generator_Random_Uout;

        for(size_t tmp_generator_index(0_zu); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        {
            tmp_ptr_array_Class_Generator_Random_Uout[tmp_generator_index].Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            tmp_ptr_array_Class_Generator_Random_Uout[tmp_generator_index].Range(0_T, 1_T);
        }
    }

    return(true);
}
