/* Copyright 2020 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

bool Layer::Use__K_Sparsity(void) const { return(this->k_sparsity != 0_zu); }

size_t Layer::Get__K_Sparsity(void) const { return(this->k_sparsity); }

T_ Layer::Get__Alpha_Sparsity(void) const { return(this->alpha_sparsity); }

bool Neural_Network::Set__K_Sparsity(size_t const index_layer_received, size_t const k_sparsity_received)
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
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(this->Set__K_Sparsity(this->ptr_array_layers + index_layer_received, k_sparsity_received));
}

bool Neural_Network::Set__K_Sparsity(struct Layer *const ptr_layer_received, size_t const k_sparsity_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the input layer. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the output layer. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received->k_sparsity == k_sparsity_received) { return(true); }
    else if(static_cast<size_t>(static_cast<T_>(k_sparsity_received) * ptr_layer_received->alpha_sparsity) > *ptr_layer_received->ptr_number_outputs)
    {
        PRINT_FORMAT("%s: %s: ERROR: k-sparse cannot be %zu because an overflow (%zu * %f = %zu) occur (limit=%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 k_sparsity_received,
                                 k_sparsity_received,
                                 Cast_T(ptr_layer_received->alpha_sparsity),
                                 static_cast<size_t>(static_cast<T_>(k_sparsity_received) * ptr_layer_received->alpha_sparsity),
                                 *ptr_layer_received->ptr_number_outputs,
                                 __LINE__);

        return(false);
    }

    if(ptr_layer_received->k_sparsity == 0_zu && k_sparsity_received != 0_zu)
    {
        // TODO: Allocation based on the number of k-sparse filters and not the total units.
        if(++this->total_k_sparse_layers == 1_zu && this->Allocate__Sparse_K_Filter() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Sparse_K_Filter()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);
        
            --this->total_k_sparse_layers;

            return(false);
        }
    }
    else if(ptr_layer_received->k_sparsity != 0_zu && k_sparsity_received == 0_zu)
    {
        if(this->total_k_sparse_layers != 0_zu
          &&
          --this->total_k_sparse_layers == 0_zu)
        { this->Deallocate__Sparse_K_Filter(); }
    }

    ptr_layer_received->k_sparsity = k_sparsity_received;

    return(true);
}

bool Neural_Network::Set__Alpha_Sparsity(size_t const index_layer_received, T_ const alpha_sparsity_received)
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
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(this->Set__Alpha_Sparsity(this->ptr_array_layers + index_layer_received, alpha_sparsity_received));
}

bool Neural_Network::Set__Alpha_Sparsity(struct Layer *const ptr_layer_received,T_ const alpha_sparsity_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the input layer. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the output layer. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(alpha_sparsity_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: alpha k-sparse (%f) cannot be less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(alpha_sparsity_received),
                                 __LINE__);

        return(false);
    }
    else if(static_cast<size_t>(static_cast<T_>(ptr_layer_received->k_sparsity) * alpha_sparsity_received) > *ptr_layer_received->ptr_number_outputs)
    {
        PRINT_FORMAT("%s: %s: ERROR: alpha k-sparse cannot be %f because an overflow (%zu * %f = %zu) occur (limit=%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(alpha_sparsity_received),
                                 ptr_layer_received->k_sparsity,
                                 Cast_T(alpha_sparsity_received),
                                 static_cast<size_t>(static_cast<T_>(ptr_layer_received->k_sparsity) * alpha_sparsity_received),
                                 *ptr_layer_received->ptr_number_outputs,
                                 __LINE__);

        return(false);
    }

    ptr_layer_received->alpha_sparsity = alpha_sparsity_received;

    return(true);
}

void Neural_Network::Assign__Sparsity_Activities(size_t const number_threads_received)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers);
    
    std::pair<size_t, T_> *tmp_ptr_array_k_sparses(this->ptr_array_k_sparse_activities);

    // Assign array position to each layers.
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        tmp_ptr_layer_it->ptr_array_k_sparse_activities = tmp_ptr_array_k_sparses;
        
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL: tmp_ptr_array_k_sparses += number_threads_received * *tmp_ptr_layer_it->ptr_number_outputs; break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: tmp_ptr_array_k_sparses += number_threads_received * static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_unit - tmp_ptr_layer_it->ptr_array_AF_units); break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: tmp_ptr_array_k_sparses += number_threads_received * static_cast<size_t>(tmp_ptr_layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units); break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                         __LINE__);
                    return;
        }
    }
}

void Neural_Network::Sparse_K_Filter(size_t const time_step_index_received,
                                                        size_t const batch_size_received,
                                                        size_t const input_unit_size_received,
                                                        size_t const k_sparsity_received,
                                                        std::pair<size_t, T_> *const ptr_array_k_sparses_received,
                                                        T_ *const ptr_array_inputs_received)
{
    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        this->Sparse_K_Filter__Loop(time_step_index_received,
                                                   batch_size_received,
                                                   input_unit_size_received,
                                                   k_sparsity_received,
                                                   ptr_array_k_sparses_received,
                                                   ptr_array_inputs_received);
    }
    else
    {
        this->Sparse_K_Filter__OpenMP(time_step_index_received,
                                                        batch_size_received,
                                                        input_unit_size_received,
                                                        k_sparsity_received,
                                                        ptr_array_k_sparses_received,
                                                        ptr_array_inputs_received);
    }
}

void Neural_Network::Sparse_K_Filter__Loop(size_t const time_step_index_received,
                                                                  size_t const batch_size_received,
                                                                  size_t const input_unit_size_received,
                                                                  size_t const k_sparsity_received,
                                                                  std::pair<size_t, T_> *const ptr_array_k_sparses_received,
                                                                  T_ *const ptr_array_inputs_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(input_unit_size_received < k_sparsity_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Number of input(s) (%zu) cannot be less than the number of k-sparse filters (%zu).  At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_unit_size_received,
                                 k_sparsity_received,
                                 __LINE__);

        return;
    }
#endif

    if(k_sparsity_received == input_unit_size_received) { return; }

    size_t const tmp_unit_timed_index(input_unit_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_unit_index,
              tmp_unit_data_timed_index;
    
    // Custom sorting.
    auto tmp_Sort_Pair([](std::pair<size_t, T_> &a_received, std::pair<size_t, T_> &b_received) -> bool { return a_received.second < b_received.second; });

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_unit_data_timed_index = tmp_example_index * input_unit_size_received + this->batch_size * tmp_unit_timed_index;
        
        // Initialize array of pairs.
        for(tmp_unit_index = 0_zu; tmp_unit_index != k_sparsity_received; ++tmp_unit_index)
        {
            ptr_array_k_sparses_received[tmp_unit_index].first = tmp_unit_index;

            ptr_array_k_sparses_received[tmp_unit_index].second = ptr_array_inputs_received[tmp_unit_data_timed_index + tmp_unit_index];
        }

        // Sort the array of pairs.
        std::sort(ptr_array_k_sparses_received,
                     ptr_array_k_sparses_received + k_sparsity_received,
                     tmp_Sort_Pair);

        // Compute the highest input into the array of pair.
        for(tmp_unit_index = k_sparsity_received; tmp_unit_index != input_unit_size_received; ++tmp_unit_index)
        {
            if(ptr_array_k_sparses_received[0u].second < ptr_array_inputs_received[tmp_unit_data_timed_index + tmp_unit_index])
            {
                ptr_array_k_sparses_received[0u].first = tmp_unit_index;

                ptr_array_k_sparses_received[0u].second = ptr_array_inputs_received[tmp_unit_data_timed_index + tmp_unit_index];
                
                std::sort(ptr_array_k_sparses_received,
                             ptr_array_k_sparses_received + k_sparsity_received,
                             tmp_Sort_Pair);
            }
        }

        // Zero out array of inputs.
        MEMSET(ptr_array_inputs_received + tmp_unit_data_timed_index,
                     0,
                     input_unit_size_received * sizeof(T_));

        // Keep the k-sparses input(s).
        for(tmp_unit_index = 0_zu; tmp_unit_index != k_sparsity_received; ++tmp_unit_index)
        { ptr_array_inputs_received[tmp_unit_data_timed_index + ptr_array_k_sparses_received[tmp_unit_index].first] = ptr_array_k_sparses_received[tmp_unit_index].second; }
    }
}

void Neural_Network::Sparse_K_Filter__OpenMP(size_t const time_step_index_received,
                                                                        size_t const batch_size_received,
                                                                        size_t const input_unit_size_received,
                                                                        size_t const k_sparsity_received,
                                                                        std::pair<size_t, T_> *const ptr_array_k_sparses_received,
                                                                        T_ *const ptr_array_inputs_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(input_unit_size_received < k_sparsity_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Number of input(s) (%zu) cannot be less than the number of k-sparse filters (%zu).  At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_unit_size_received,
                                 k_sparsity_received,
                                 __LINE__);

        return;
    }
#endif
    
    if(k_sparsity_received == input_unit_size_received) { return; }

    int batch_size_received__int(static_cast<int>(batch_size_received)),
        tmp_example_index__int;
    
    size_t const tmp_unit_timed_index(input_unit_size_received * time_step_index_received);
    size_t tmp_unit_index(0_zu),
              tmp_unit_data_timed_index(0_zu);
    
    std::pair<size_t, T_> *tmp_ptr_array_k_sparses(nullptr);

    // Custom sorting.
    auto tmp_Sort_Pair([](std::pair<size_t, T_> &a_received, std::pair<size_t, T_> &b_received) -> bool { return a_received.second < b_received.second; });
    
    #pragma omp parallel for schedule(static) private(tmp_unit_index, \
                                                                           tmp_unit_data_timed_index, \
                                                                           tmp_ptr_array_k_sparses)
    for(tmp_example_index__int = 0; tmp_example_index__int < batch_size_received__int; ++tmp_example_index__int)
    {
        tmp_unit_data_timed_index = tmp_example_index__int * input_unit_size_received + this->batch_size * tmp_unit_timed_index;
        
        tmp_ptr_array_k_sparses = ptr_array_k_sparses_received + static_cast<size_t>(omp_get_thread_num()) * k_sparsity_received;

        // Initialize array of pairs.
        for(tmp_unit_index = 0_zu; tmp_unit_index != k_sparsity_received; ++tmp_unit_index)
        {
            tmp_ptr_array_k_sparses[tmp_unit_index].first = tmp_unit_index;

            tmp_ptr_array_k_sparses[tmp_unit_index].second = ptr_array_inputs_received[tmp_unit_data_timed_index + tmp_unit_index];
        }

        // Sort the array of pairs.
        std::sort(tmp_ptr_array_k_sparses,
                     tmp_ptr_array_k_sparses + k_sparsity_received,
                     tmp_Sort_Pair);

        // Compute the highest input into the array of pair.
        for(tmp_unit_index = k_sparsity_received; tmp_unit_index != input_unit_size_received; ++tmp_unit_index)
        {
            if(tmp_ptr_array_k_sparses[0u].second < ptr_array_inputs_received[tmp_unit_data_timed_index + tmp_unit_index])
            {
                tmp_ptr_array_k_sparses[0u].first = tmp_unit_index;

                tmp_ptr_array_k_sparses[0u].second = ptr_array_inputs_received[tmp_unit_data_timed_index + tmp_unit_index];
                
                std::sort(tmp_ptr_array_k_sparses,
                             tmp_ptr_array_k_sparses + k_sparsity_received,
                             tmp_Sort_Pair);
            }
        }

        // Zero out array of inputs.
        MEMSET(ptr_array_inputs_received + tmp_unit_data_timed_index,
                     0,
                     input_unit_size_received * sizeof(T_));

        // Keep the k-sparses input(s).
        for(tmp_unit_index = 0_zu; tmp_unit_index != k_sparsity_received; ++tmp_unit_index)
        { ptr_array_inputs_received[tmp_unit_data_timed_index + tmp_ptr_array_k_sparses[tmp_unit_index].first] = tmp_ptr_array_k_sparses[tmp_unit_index].second; }
    }
}