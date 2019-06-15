#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

bool Neural_Network::Allouable__Batch_Size(size_t const desired_batch_size_received,
                                                                  size_t &ref_batch_size_allouable_received,
                                                                  size_t &ref_number_threads_allouable_received)
{
    // Size of a thread.
    size_t const tmp_size_thread(this->Get__Threads_Sizeof(1_zu)),
    // Size of a batch.
                       tmp_size_batch(this->Get__Batch_Sizeof(1_zu)),
    // Size of a neural network with no batch.
                       tmp_size_neural_network(this->Get__Sizeof(1_zu, 1_zu) - (tmp_size_thread + tmp_size_batch)),
    // Available memory substraction size of the neural network without batch.
                       tmp_available_memory_mbs(this->maximum_allowable_memory_bytes - tmp_size_neural_network);
    
    // If the neural network overflow the maximum allowable memory.
    if(this->maximum_allowable_memory_bytes < tmp_size_neural_network)
    {
        PRINT_FORMAT("%s: %s: ERROR: Maximum allowable memory (%zu) is less than the memory allocate (%zu). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 this->maximum_allowable_memory_bytes,
                                 tmp_size_neural_network + tmp_size_thread + tmp_size_batch,
                                 __LINE__);

        ref_batch_size_allouable_received = 0_zu;
        ref_number_threads_allouable_received = 0_zu;

        return(false);
    }
    // If one the size of one thread overflow the available memory.
    else if(tmp_available_memory_mbs < tmp_size_thread)
    {
        PRINT_FORMAT("%s: %s: ERROR: No enought available memory (%zu) for allocating %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_available_memory_mbs,
                                 tmp_size_thread,
                                 __LINE__);

        ref_batch_size_allouable_received = 0_zu;
        ref_number_threads_allouable_received = 0_zu;

        return(false);
    }
    // If one the size of one batch overflow the available memory.
    else if(tmp_available_memory_mbs - tmp_size_thread < tmp_size_batch)
    {
        PRINT_FORMAT("%s: %s: ERROR: No enought available memory (%zu) for allocating %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_available_memory_mbs - tmp_size_thread,
                                 tmp_size_batch,
                                 __LINE__);

        ref_batch_size_allouable_received = 0_zu;
        ref_number_threads_allouable_received = 0_zu;

        return(false);
    }

    size_t const tmp_maximum_threads((this->use_OpenMP || this->is_OpenMP_initialized) ? (this->percentage_maximum_thread_usage == 0.0f ? 1_zu : MyEA::Math::Minimum<size_t>(static_cast<size_t>(ceil(static_cast<double>(this->percentage_maximum_thread_usage) * static_cast<double>(omp_get_num_procs()) / 100.0)), desired_batch_size_received)) : 1_zu);
    size_t tmp_maximum_batch_size_allocatable((tmp_available_memory_mbs - tmp_size_thread) / tmp_size_batch),
              tmp_batch_size_allocate(MyEA::Math::Minimum<size_t>(desired_batch_size_received, this->maximum_batch_size)),
              tmp_threads_allocate(1_zu);

    if(tmp_batch_size_allocate > tmp_maximum_batch_size_allocatable)
    {
        PRINT_FORMAT("%s: %s: WARNING: Can not allocate the optimal batch size (%zu). The batch size allocated will be reduced to %zu." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_batch_size_allocate,
                                 tmp_maximum_batch_size_allocatable);

        // Batch size equal maximum batch size allocatables.
        tmp_batch_size_allocate = tmp_maximum_batch_size_allocatable;
    }
    else
    {
        for(; tmp_threads_allocate != tmp_maximum_threads; ++tmp_threads_allocate)
        {
            // Maximum batch size equal available memory minus allocates threads, then divide by one batch size.
            tmp_maximum_batch_size_allocatable = static_cast<size_t>((tmp_available_memory_mbs - tmp_threads_allocate * tmp_size_thread) / tmp_size_batch);

            // If batch size is greater than maximum batch size allocatables.
            if(tmp_batch_size_allocate > tmp_maximum_batch_size_allocatable)
            {
                PRINT_FORMAT("%s: %s: WARNING: Can not allocate the optimal number of threads (%zu). The number of threads allocated will be reduced to %zu." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_threads_allocate,
                                         tmp_threads_allocate - 1_zu);

                // Batch size equal available memory minus past allocates threads, then divide by one batch size.
                tmp_batch_size_allocate = static_cast<size_t>((tmp_available_memory_mbs - (tmp_threads_allocate - 1_zu) * tmp_size_thread) / tmp_size_batch);

                break;
            }
        }
    }

    ref_batch_size_allouable_received = tmp_batch_size_allocate;
    ref_number_threads_allouable_received = tmp_threads_allocate;

    return(true);
}

bool Neural_Network::Update__Thread_Size(size_t const desired_number_threads_received)
{
    if(this->is_OpenMP_initialized == false) { return(true); }
    else if(desired_number_threads_received <= this->cache_number_threads && this->percentage_maximum_thread_usage == this->cache_maximum_threads_percent) { return(true); }
    
    size_t tmp_batch_size_allocate(desired_number_threads_received),
              tmp_number_threads_allocate(desired_number_threads_received);
    
    if(this->Allouable__Batch_Size(desired_number_threads_received,
                                                 tmp_batch_size_allocate,
                                                 tmp_number_threads_allocate) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allouable__Thread_Size(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 desired_number_threads_received,
                                 tmp_batch_size_allocate,
                                 tmp_number_threads_allocate,
                                 __LINE__);

        return(false);
    }

    // If number of threads differ from the new desired.
    if(this->number_threads != tmp_number_threads_allocate)
    {
        if(this->Reallocate__Thread(tmp_number_threads_allocate) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_number_threads_allocate,
                                     __LINE__);

            return(false);
        }

        if(this->Update__Batch_Size(this->cache_batch_size, true) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu, true)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->cache_batch_size,
                                     __LINE__);

            return(false);
        }
        
        this->number_threads = tmp_number_threads_allocate;
    }

    // If number of threads is greater than the cached.
    if(desired_number_threads_received > this->cache_number_threads) { this->cache_number_threads = desired_number_threads_received; }

    // Cache the maximum threads in percent.
    this->cache_maximum_threads_percent = this->percentage_maximum_thread_usage;

    return(true);
}

bool Neural_Network::Update__Batch_Size(size_t const desired_batch_size_received, bool const force_update_received)
{
    if(force_update_received == false && desired_batch_size_received <= this->cache_batch_size) { return(true); }
    
    size_t tmp_batch_size_allocate(desired_batch_size_received),
              tmp_number_threads_allocate(desired_batch_size_received);
    
    if(this->Allouable__Batch_Size(desired_batch_size_received,
                                                 tmp_batch_size_allocate,
                                                 tmp_number_threads_allocate) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allouable__Thread_Size(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 desired_batch_size_received,
                                 tmp_batch_size_allocate,
                                 tmp_number_threads_allocate,
                                 __LINE__);

        return(false);
    }

    // If total data batch differ from the new desired
    if(this->batch_size != tmp_batch_size_allocate)
    {
        // Reallocate batch size with the new batch size meet.
        if(this->Reallocate__Batch(tmp_batch_size_allocate) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Batch(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_batch_size_allocate,
                                     __LINE__);

            return(false);
        }

        this->batch_size = tmp_batch_size_allocate;
    }

    // Cache total data batch.
    this->cache_batch_size = desired_batch_size_received;

    return(true);
}
