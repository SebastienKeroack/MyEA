#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

void Neural_Network::Merge_Derivatives_Parameters(size_t const start_index_received, size_t const end_index_received)
{
    if(this->use_OpenMP
      &&
      this->is_OpenMP_initialized
      &&
      this->number_threads > 1_zu)
    {
        int const tmp_end_index__int(static_cast<int>(end_index_received));
        int tmp_connection_index__int;

        size_t tmp_thread_index;
        
        T_ const *tmp_ptr_array_derivatives_parameters;
        T_ *const tmp_ptr_array_derivatives_parameters_summation(this->ptr_array_derivatives_parameters);
        
        for(tmp_thread_index = 1_zu; tmp_thread_index != this->number_threads; ++tmp_thread_index)
        {
            tmp_ptr_array_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_thread_index * this->total_parameters_allocated;
            
            #pragma omp parallel for schedule(static)
            for(tmp_connection_index__int = static_cast<int>(start_index_received); tmp_connection_index__int < tmp_end_index__int; ++tmp_connection_index__int)
            { tmp_ptr_array_derivatives_parameters_summation[tmp_connection_index__int] += tmp_ptr_array_derivatives_parameters[tmp_connection_index__int]; }
        }

        // Clear the whole threaded gradients.
        MEMSET(this->ptr_array_derivatives_parameters + this->total_parameters_allocated,
                      0,
                      (this->number_threads - 1_zu) * this->total_parameters_allocated * sizeof(T_));
    }
}

void Neural_Network::Merge__Post__Training(void)
{
    if(this->use_OpenMP
      &&
      this->is_OpenMP_initialized
      &&
      this->number_threads > 1_zu)
    {
        size_t tmp_thread_index;
        
        if(this->ptr_array_number_loss != nullptr)
        {
            for(tmp_thread_index = 1_zu; tmp_thread_index != this->number_threads; ++tmp_thread_index)
            {
                this->ptr_array_number_loss[0u] += this->ptr_array_number_loss[tmp_thread_index];
            }
        }
        
        if(this->ptr_array_loss_values != nullptr)
        {
            for(tmp_thread_index = 1_zu; tmp_thread_index != this->number_threads; ++tmp_thread_index)
            {
                this->ptr_array_loss_values[0u] += this->ptr_array_loss_values[tmp_thread_index];
            }
        }
        
        if(this->type_loss_function == MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_BIT && this->ptr_array_number_bit_fail != nullptr)
        {
            for(tmp_thread_index = 1_zu; tmp_thread_index != this->number_threads; ++tmp_thread_index)
            {
                this->ptr_array_number_bit_fail[0u] += this->ptr_array_number_bit_fail[tmp_thread_index];
            }
        }
        
        if(this->ptr_array_accuracy_values != nullptr)
        {
            for(tmp_thread_index = 1_zu; tmp_thread_index != this->number_threads; ++tmp_thread_index)
            {
                this->ptr_array_accuracy_values[0u][0u] += this->ptr_array_accuracy_values[0u][tmp_thread_index];
                this->ptr_array_accuracy_values[1u][0u] += this->ptr_array_accuracy_values[1u][tmp_thread_index];
                this->ptr_array_accuracy_values[2u][0u] += this->ptr_array_accuracy_values[2u][tmp_thread_index];
                this->ptr_array_accuracy_values[3u][0u] += this->ptr_array_accuracy_values[3u][tmp_thread_index];
                this->ptr_array_accuracy_values[4u][0u] += this->ptr_array_accuracy_values[4u][tmp_thread_index];
            }
        }
    }
}

void Neural_Network::Merge__Accuracy__R(void)
{
    if(this->use_OpenMP
      &&
      this->is_OpenMP_initialized
      &&
      this->number_threads > 1_zu)
    {
        size_t tmp_thread_index;
        
        for(tmp_thread_index = 1_zu; tmp_thread_index != this->number_threads; ++tmp_thread_index)
        {
            this->ptr_array_accuracy_values[2u][0u] += this->ptr_array_accuracy_values[2u][tmp_thread_index];
            this->ptr_array_accuracy_values[3u][0u] += this->ptr_array_accuracy_values[3u][tmp_thread_index];
            this->ptr_array_accuracy_values[4u][0u] += this->ptr_array_accuracy_values[4u][tmp_thread_index];
        }
    }
}