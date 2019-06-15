#include "stdafx.hpp"

#if defined(COMPILE_CUDA)
    #include <CUDA/CUDA_Dataset_Manager.cuh>
#endif

#include <Neural_Network/Dataset_Manager.hpp>

#include <iostream>
#include <array>
#include <omp.h>

template<typename T>
Gaussian_Search<T>::Gaussian_Search(void)
{
}

template<typename T>
bool Gaussian_Search<T>::Initialize__OpenMP(void)
{
    if(this->_is_OpenMP_initialized == false)
    {
        this->_is_OpenMP_initialized = true;

        if(this->Update__Thread_Size(this->_population_size) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->_population_size,
                                     __LINE__);

            return(false);
        }

        omp_set_dynamic(0);
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Set__OpenMP(bool const use_openmp_received)
{
    if((this->_use_OpenMP == false && use_openmp_received)
      ||
      (this->_use_OpenMP && use_openmp_received && this->_is_OpenMP_initialized == false))
    {
        if(this->Initialize__OpenMP() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__OpenMP()\" function. At line %d." NEW_LINE,
                MyEA::String::Get__Time().c_str(),
                __FUNCTION__,
                __LINE__);

            return(false);
        }
    }
    else if((this->_use_OpenMP && use_openmp_received == false)
             ||
             (this->_use_OpenMP == false && use_openmp_received == false && this->_is_OpenMP_initialized))
    {
        if(this->Deinitialize__OpenMP() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Deinitialize__OpenMP()\" function. At line %d." NEW_LINE,
                MyEA::String::Get__Time().c_str(),
                __FUNCTION__,
                __LINE__);

            return(false);
        }
    }

    this->_use_OpenMP = use_openmp_received;

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Set__Population_Size(size_t const population_size_received)
{
    if(population_size_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The population size can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->_population_size != population_size_received)
    {
        if(this->_population_size == 0_zu)
        {
            if(this->Allocate__Population(population_size_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Population(%zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         population_size_received,
                                         __LINE__);

                return(false);
            }
        }
        else
        {
            if(this->Reallocate__Population(population_size_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Population(%zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         population_size_received,
                                         __LINE__);

                return(false);
            }
        }

        this->_population_size = population_size_received;
        
        if(this->Update__Thread_Size(population_size_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     population_size_received,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Set__Population_Gaussian(double const population_gaussian_percent_received)
{
    if(population_gaussian_percent_received <= 1.0)
    {
        PRINT_FORMAT("%s: %s: ERROR: The population gaussian in percent can not be equal or less than one percent. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    this->_population_gaussian_percent = population_gaussian_percent_received;

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Set__Maximum_Thread_Usage(double const percentage_maximum_thread_usage_received)
{
    if(this->_percentage_maximum_thread_usage == percentage_maximum_thread_usage_received) { return(true); }

    this->_percentage_maximum_thread_usage = percentage_maximum_thread_usage_received;

    if(this->Update__Thread_Size(this->_population_size) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 this->_population_size,
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Allocate__Population(size_t const population_size_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(population_size_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The population size can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    if(this->p_ptr_array_ptr_individuals == nullptr && this->p_ptr_array_individuals == nullptr)
    {
        if((this->p_ptr_array_ptr_individuals = new class Neural_Network*[population_size_received]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     population_size_received * sizeof(class Neural_Network*),
                                     __LINE__);

            return(false);
        }
            
        if(population_size_received > 1_zu)
        {
            if((this->p_ptr_array_individuals = new class Neural_Network[population_size_received - 1_zu]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         (population_size_received - 1_zu) * sizeof(class Neural_Network),
                                         __LINE__);

                return(false);
            }

            for(size_t tmp_individual_index(1_zu); tmp_individual_index != population_size_received; ++tmp_individual_index)
            { this->p_ptr_array_ptr_individuals[tmp_individual_index] = this->p_ptr_array_individuals + (tmp_individual_index - 1_zu); }
        }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Allocate__Thread(size_t const number_threads_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(number_threads_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The population size can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    if(this->p_ptr_array_ptr_dataset_manager == nullptr && this->p_ptr_array_dataset_manager == nullptr)
    {
        if((this->p_ptr_array_ptr_dataset_manager = new class Dataset_Manager<T>*[number_threads_received]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     number_threads_received * sizeof(class Dataset_Manager<T>*),
                                     __LINE__);

            return(false);
        }
            
        if(number_threads_received > 1_zu)
        {
            if((this->p_ptr_array_dataset_manager = new class Dataset_Manager<T>[number_threads_received - 1_zu]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         (number_threads_received - 1_zu) * sizeof(class Dataset_Manager<T>),
                                         __LINE__);

                return(false);
            }

            for(size_t tmp_thread_index(1_zu); tmp_thread_index != number_threads_received; ++tmp_thread_index)
            { this->p_ptr_array_ptr_dataset_manager[tmp_thread_index] = this->p_ptr_array_dataset_manager + (tmp_thread_index - 1_zu); }
        }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Reallocate__Population(size_t const population_size_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(population_size_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The population size can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    if(this->p_ptr_array_ptr_individuals != nullptr && this->p_ptr_array_individuals != nullptr)
    {
        this->p_ptr_array_ptr_individuals = Memory::reallocate_pointers_array_cpp<class Neural_Network*>(this->p_ptr_array_ptr_individuals,
                                                                                                                                                         population_size_received,
                                                                                                                                                         this->_population_size,
                                                                                                                                                         false);
        if(this->p_ptr_array_ptr_individuals == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     population_size_received * sizeof(class Neural_Network*),
                                     __LINE__);

            return(false);
        }

        if(population_size_received > 1_zu)
        {
            this->p_ptr_array_individuals = Memory::reallocate_objects_cpp<class Neural_Network>(this->p_ptr_array_individuals,
                                                                                                                                            population_size_received - 1_zu,
                                                                                                                                            this->_population_size - 1_zu,
                                                                                                                                            false);
            if(this->p_ptr_array_individuals == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         (population_size_received - 1_zu) * sizeof(class Neural_Network),
                                         __LINE__);

                return(false);
            }

            for(size_t tmp_individual_index(1_zu); tmp_individual_index != population_size_received; ++tmp_individual_index)
            { this->p_ptr_array_ptr_individuals[tmp_individual_index] = this->p_ptr_array_individuals + (tmp_individual_index - 1_zu); }
        }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Reallocate__Thread(size_t const number_threads_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(number_threads_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The population size can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    if(this->p_ptr_array_ptr_dataset_manager != nullptr && this->p_ptr_array_dataset_manager != nullptr)
    {
        this->p_ptr_array_ptr_dataset_manager = Memory::reallocate_pointers_array_cpp<class Dataset_Manager<T>*>(this->p_ptr_array_ptr_dataset_manager,
                                                                                                                                                                            number_threads_received,
                                                                                                                                                                            this->_number_threads,
                                                                                                                                                                            false);
        if(this->p_ptr_array_ptr_dataset_manager == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     number_threads_received * sizeof(class Dataset_Manager<T>*),
                                     __LINE__);

            return(false);
        }

        if(number_threads_received > 1_zu)
        {
            this->p_ptr_array_dataset_manager = Memory::reallocate_objects_cpp<class Dataset_Manager<T>>(this->p_ptr_array_dataset_manager,
                                                                                                                                                               number_threads_received - 1_zu,
                                                                                                                                                               this->_number_threads - 1_zu,
                                                                                                                                                               false);
            if(this->p_ptr_array_dataset_manager == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         (number_threads_received - 1_zu) * sizeof(class Dataset_Manager<T>),
                                         __LINE__);

                return(false);
            }

            for(size_t tmp_thread_index(1_zu); tmp_thread_index != number_threads_received; ++tmp_thread_index)
            { this->p_ptr_array_ptr_dataset_manager[tmp_thread_index] = this->p_ptr_array_dataset_manager + (tmp_thread_index - 1_zu); }
        }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Allouable__Thread_Size(size_t const desired_number_threads_received, size_t &ref_number_threads_allouable_received)
{
    // TODO: Available memory.
    size_t const tmp_maximum_threads((this->_use_OpenMP || this->_is_OpenMP_initialized) ? (this->_percentage_maximum_thread_usage == 0.0f ? 1_zu : MyEA::Math::Minimum<size_t>(static_cast<size_t>(ceil(static_cast<double>(this->_percentage_maximum_thread_usage) * static_cast<double>(omp_get_num_procs()) / 100.0)), desired_number_threads_received)) : 1_zu);
    size_t tmp_threads_allocate(MyEA::Math::Minimum<size_t>(desired_number_threads_received, tmp_maximum_threads));

    ref_number_threads_allouable_received = tmp_threads_allocate;

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Update__Thread_Size(size_t const desired_number_threads_received)
{
    if(this->_is_OpenMP_initialized == false) { return(true); }
    else if(desired_number_threads_received <= this->_cache_number_threads && this->_percentage_maximum_thread_usage == this->_cache_maximum_threads_percent) { return(true); }
    
    size_t tmp_number_threads_allocate(desired_number_threads_received);
    
    if(this->Allouable__Thread_Size(desired_number_threads_received, tmp_number_threads_allocate) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allouable__Thread_Size(%zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 desired_number_threads_received,
                                 tmp_number_threads_allocate,
                                 __LINE__);

        return(false);
    }

    // If number of threads differ from the new desired.
    if(this->_number_threads != tmp_number_threads_allocate)
    {
        if(this->_number_threads == 0_zu)
        {
            if(this->Allocate__Thread(tmp_number_threads_allocate) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Thread(%zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_number_threads_allocate,
                                         __LINE__);

                return(false);
            }
        }
        else
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
        }

        this->_number_threads = tmp_number_threads_allocate;
    }

    // If number of threads is greater than the cached.
    if(desired_number_threads_received > this->_cache_number_threads) { this->_cache_number_threads = desired_number_threads_received; }

    // Cache the maximum threads in percent.
    this->_cache_maximum_threads_percent = this->_percentage_maximum_thread_usage;

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Update__Thread_Size__Population(size_t const desired_number_threads_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    if(this->p_ptr_array_ptr_individuals[0u]->use_OpenMP && this->p_ptr_array_ptr_individuals[0u]->is_OpenMP_initialized)
    {
        class Neural_Network *tmp_ptr_Neural_Network;
        
        for(size_t tmp_individual_index(1_zu); tmp_individual_index != this->_population_size; ++tmp_individual_index)
        {
            tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
            
        #if defined(_DEBUG) || defined(COMPILE_DEBUG)
            if(tmp_ptr_Neural_Network == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_individual_index,
                                         __LINE__);

                return(false);
            }
        #endif
            
            if(tmp_ptr_Neural_Network->Update__Thread_Size(desired_number_threads_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         desired_number_threads_received,
                                         __LINE__);

                return(false);
            }
        }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Update__Batch_Size__Population(size_t const desired_batch_size_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    class Neural_Network *tmp_ptr_Neural_Network;
    
    for(size_t tmp_individual_index(1_zu); tmp_individual_index != this->_population_size; ++tmp_individual_index)
    {
        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_individual_index,
                                     __LINE__);

            return(false);
        }
    #endif
        
        if(tmp_ptr_Neural_Network->Update__Batch_Size(desired_batch_size_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     desired_batch_size_received,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Update__Population(class Neural_Network *const ptr_source_Dataset_Manager_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    this->p_ptr_array_ptr_individuals[0u] = ptr_source_Dataset_Manager_received;
    
    class Neural_Network *tmp_ptr_Neural_Network;
    
    for(size_t tmp_individual_index(1_zu); tmp_individual_index != this->_population_size; ++tmp_individual_index)
    {
        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_individual_index,
                                     __LINE__);

            return(false);
        }
    #endif

        if(tmp_ptr_Neural_Network->type_network != ptr_source_Dataset_Manager_received->type_network
          ||
          tmp_ptr_Neural_Network->total_basic_units != ptr_source_Dataset_Manager_received->total_basic_units
          ||
          tmp_ptr_Neural_Network->total_basic_indice_units != ptr_source_Dataset_Manager_received->total_basic_indice_units
          ||
          tmp_ptr_Neural_Network->total_neuron_units != ptr_source_Dataset_Manager_received->total_neuron_units
          ||
          tmp_ptr_Neural_Network->total_AF_units != ptr_source_Dataset_Manager_received->total_AF_units
          ||
          tmp_ptr_Neural_Network->total_AF_Ind_recurrent_units != ptr_source_Dataset_Manager_received->total_AF_Ind_recurrent_units
          ||
          tmp_ptr_Neural_Network->total_block_units != ptr_source_Dataset_Manager_received->total_block_units
          ||
          tmp_ptr_Neural_Network->total_cell_units != ptr_source_Dataset_Manager_received->total_cell_units
          ||
          tmp_ptr_Neural_Network->total_weights != ptr_source_Dataset_Manager_received->total_weights
          ||
          tmp_ptr_Neural_Network->total_bias != ptr_source_Dataset_Manager_received->total_bias)
        {
            if(tmp_ptr_Neural_Network->Copy(*ptr_source_Dataset_Manager_received,
                                                              true,
                                                              true) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy(ptr, true, true)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }
        else if(tmp_ptr_Neural_Network->Update(*ptr_source_Dataset_Manager_received,
                                                                   true,
                                                                   true) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update(ptr, true, true)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Update__Dataset_Manager(class Dataset_Manager<T> *const ptr_source_Dataset_Manager_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->_number_threads == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of dataset manager is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_dataset_manager == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of dataset manager is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    this->p_ptr_array_ptr_dataset_manager[0u] = ptr_source_Dataset_Manager_received;
    
    class Dataset_Manager<T> *tmp_ptr_Dataset_Manager;
    
    for(size_t tmp_thread_index(1_zu); tmp_thread_index != this->_number_threads; ++tmp_thread_index)
    {
        tmp_ptr_Dataset_Manager = this->p_ptr_array_ptr_dataset_manager[tmp_thread_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Dataset_Manager == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Dataset manager #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_thread_index,
                                     __LINE__);

            return(false);
        }
    #endif

        if(tmp_ptr_Dataset_Manager->Reference(ptr_source_Dataset_Manager_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reference(ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Enable__OpenMP__Population(void)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    class Neural_Network *tmp_ptr_Neural_Network;
    
    for(size_t tmp_individual_index(0_zu); tmp_individual_index != this->_population_size; ++tmp_individual_index)
    {
        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_individual_index,
                                     __LINE__);

            return(false);
        }
    #endif
        
        tmp_ptr_Neural_Network->use_OpenMP = true;
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Disable__OpenMP__Population(void)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    class Neural_Network *tmp_ptr_Neural_Network;
    
    for(size_t tmp_individual_index(0_zu); tmp_individual_index != this->_population_size; ++tmp_individual_index)
    {
        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_individual_index,
                                     __LINE__);

            return(false);
        }
    #endif
        
        tmp_ptr_Neural_Network->use_OpenMP = false;
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Optimize(size_t const number_iterations_received,
                                                      class Dataset_Manager<T> *const ptr_Dataset_Manager_received,
                                                      class Neural_Network *const ptr_Neural_Network_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Dataset_Manager_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Dataset manager is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_Neural_Network_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Neural network is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    if(this->_use_OpenMP && this->_is_OpenMP_initialized)
    {
        return(this->Optimize__OpenMP(number_iterations_received,
                                                        ptr_Dataset_Manager_received,
                                                        ptr_Neural_Network_received));
    }
    else
    {
        return(this->Optimize__Loop(number_iterations_received,
                                                  ptr_Dataset_Manager_received,
                                                  ptr_Neural_Network_received));
    }
}

template<typename T>
bool Gaussian_Search<T>::Optimize__Loop(size_t const number_iterations_received,
                                                                 class Dataset_Manager<T> *const ptr_Dataset_Manager_received,
                                                                 class Neural_Network *const ptr_Neural_Network_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Dataset_Manager_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Dataset manager is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_Neural_Network_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Neural network is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    size_t tmp_iterations,
              tmp_individual_index;

    class Neural_Network *tmp_ptr_Neural_Network;

    if(this->Update__Population(ptr_Neural_Network_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Population(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Initialize__Hyper_Parameters(ptr_Neural_Network_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__Hyper_Parameters(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Shuffle__Hyper_Parameter() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shuffle__Hyper_Parameter()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Feed__Hyper_Parameter() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Feed__Hyper_Parameter()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    for(tmp_iterations = 0_zu; tmp_iterations != number_iterations_received; ++tmp_iterations)
    {
        for(tmp_individual_index = 0_zu; tmp_individual_index != this->_population_size; ++tmp_individual_index)
        {
            tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
            
        #if defined(_DEBUG) || defined(COMPILE_DEBUG)
            if(tmp_ptr_Neural_Network == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_individual_index,
                                         __LINE__);

                return(false);
            }
        #endif
            
        #if defined(COMPILE_CUDA)
            if(tmp_ptr_Neural_Network->Use__CUDA())
            { ptr_Dataset_Manager_received->Get__CUDA()->Training(tmp_ptr_Neural_Network); }
            else
        #endif
            { ptr_Dataset_Manager_received->Training(tmp_ptr_Neural_Network); }
        }
    }
    
    for(tmp_individual_index = 0_zu; tmp_individual_index != this->_population_size; ++tmp_individual_index)
    {
        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
            
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_individual_index,
                                     __LINE__);

            return(false);
        }
    #endif
            
    #if defined(COMPILE_CUDA)
        if(tmp_ptr_Neural_Network->Use__CUDA())
        { ptr_Dataset_Manager_received->Get__CUDA()->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_ptr_Neural_Network); }
        else
    #endif
        { ptr_Dataset_Manager_received->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_ptr_Neural_Network); }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Optimize__OpenMP(size_t const number_iterations_received,
                                                                       class Dataset_Manager<T> *const ptr_Dataset_Manager_received,
                                                                       class Neural_Network *const ptr_Neural_Network_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Dataset_Manager_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Dataset manager is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_Neural_Network_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Neural network is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_number_threads == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of dataset manager is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_dataset_manager == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of dataset manager is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    size_t tmp_iterations(0_zu),
              tmp_error_count(0_zu);

    int const tmp_population_size__int(static_cast<int>(this->_population_size));
    int tmp_individual_index__int(0);
    
    class Neural_Network *tmp_ptr_Neural_Network(nullptr);
    
    class Dataset_Manager<T> *tmp_ptr_Dataset_Manager(nullptr);
    
    if(this->Update__Population(ptr_Neural_Network_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Population(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Update__Dataset_Manager(ptr_Dataset_Manager_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Dataset_Manager(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Initialize__Hyper_Parameters(ptr_Neural_Network_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__Hyper_Parameters(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Shuffle__Hyper_Parameter() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shuffle__Hyper_Parameter()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Feed__Hyper_Parameter() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Feed__Hyper_Parameter()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    // If the neural network use OpenMP, disable it.
    if(this->p_ptr_array_ptr_individuals[0u]->is_OpenMP_initialized && this->Disable__OpenMP__Population() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Disable__OpenMP__Population()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    omp_set_num_threads(static_cast<int>(this->_number_threads));

    #pragma omp parallel private(tmp_iterations, \
                                              tmp_individual_index__int, \
                                              tmp_ptr_Neural_Network, \
                                              tmp_ptr_Dataset_Manager)
    {
        for(tmp_iterations = 0_zu; tmp_iterations != number_iterations_received; ++tmp_iterations)
        {
            #pragma omp for schedule(dynamic)
            for(tmp_individual_index__int = 0; tmp_individual_index__int < tmp_population_size__int; ++tmp_individual_index__int)
            {
                tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index__int];
                
            #if defined(_DEBUG) || defined(COMPILE_DEBUG)
                if(tmp_ptr_Neural_Network == nullptr)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Neural network #%d is a nullptr. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_individual_index__int,
                                             __LINE__);

                    #pragma omp atomic
                    ++tmp_error_count;
                }
            #endif
                
                tmp_ptr_Dataset_Manager = this->p_ptr_array_ptr_dataset_manager[omp_get_thread_num()];
                
            #if defined(_DEBUG) || defined(COMPILE_DEBUG)
                if(tmp_ptr_Dataset_Manager == nullptr)
                {
                    PRINT_FORMAT("%s: %s: ERROR: Dataset manager #%d is a nullptr. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             omp_get_thread_num(),
                                             __LINE__);
                
                    #pragma omp atomic
                    ++tmp_error_count;
                }
            #endif
                
            #if defined(COMPILE_CUDA)
                if(tmp_ptr_Neural_Network->Use__CUDA())
                { tmp_ptr_Dataset_Manager->Get__CUDA()->Training(tmp_ptr_Neural_Network); }
                else
            #endif
                { tmp_ptr_Dataset_Manager->Training(tmp_ptr_Neural_Network); }
            }
        }
        
        #pragma omp for schedule(dynamic)
        for(tmp_individual_index__int = 0; tmp_individual_index__int < tmp_population_size__int; ++tmp_individual_index__int)
        {
            tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index__int];
            
        #if defined(_DEBUG) || defined(COMPILE_DEBUG)
            if(tmp_ptr_Neural_Network == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Neural network #%d is a nullptr. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_individual_index__int,
                                         __LINE__);

                #pragma omp atomic
                ++tmp_error_count;
            }
        #endif
            
            tmp_ptr_Dataset_Manager = this->p_ptr_array_ptr_dataset_manager[omp_get_thread_num()];
            
        #if defined(_DEBUG) || defined(COMPILE_DEBUG)
            if(tmp_ptr_Dataset_Manager == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Dataset manager #%d is a nullptr. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         omp_get_thread_num(),
                                         __LINE__);
            
                #pragma omp atomic
                ++tmp_error_count;
            }
        #endif
            
        #if defined(COMPILE_CUDA)
            if(tmp_ptr_Neural_Network->Use__CUDA())
            { tmp_ptr_Dataset_Manager->Get__CUDA()->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_ptr_Neural_Network); }
            else
        #endif
            { tmp_ptr_Dataset_Manager->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_ptr_Neural_Network); }
        }
    }
    
    // If the neural network was using OpenMP, enable it.
    if(this->p_ptr_array_ptr_individuals[0u]->is_OpenMP_initialized && this->Enable__OpenMP__Population() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Disable__OpenMP__Population()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(tmp_error_count > 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered inside the parallel region. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Evaluation(void)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    T tmp_loss(this->p_ptr_array_ptr_individuals[0u]->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));

    std::pair<size_t, T> tmp_best_model(0_zu, tmp_loss);

    class Neural_Network *tmp_ptr_Neural_Network;
    
    for(size_t tmp_individual_index(1_zu); tmp_individual_index != this->_population_size; ++tmp_individual_index)
    {
        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_individual_index,
                                     __LINE__);

            return(false);
        }
    #endif

        if(tmp_best_model.second > (tmp_loss = tmp_ptr_Neural_Network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE)))
        {
            tmp_best_model.first = tmp_individual_index;

            tmp_best_model.second = tmp_loss;
        }
    }
    
    if(tmp_best_model.first != 0_zu && this->p_ptr_array_ptr_individuals[0u]->Update(*this->p_ptr_array_ptr_individuals[tmp_best_model.first],
                                                                                                                         true,
                                                                                                                         true) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update(ptr[%zu], true, true)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_best_model.first,
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Evaluation(class Dataset_Manager<T> *const ptr_Dataset_Manager_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Dataset_Manager_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Dataset manager is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    if(this->_use_OpenMP && this->_is_OpenMP_initialized)
    { return(this->Evaluation__OpenMP(ptr_Dataset_Manager_received)); }
    else
    { return(this->Evaluation__Loop(ptr_Dataset_Manager_received)); }
}

template<typename T>
bool Gaussian_Search<T>::Evaluation__Loop(class Dataset_Manager<T> *const ptr_Dataset_Manager_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Dataset_Manager_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Dataset manager is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    size_t tmp_individual_index,
              tmp_best_individual_index(0_zu);

    class Neural_Network *tmp_ptr_Neural_Network(this->p_ptr_array_ptr_individuals[0u]);
    
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(tmp_ptr_Neural_Network == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Neural network #0 is a nullptr. At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    __LINE__);

        return(false);
    }
#endif
    
#if defined(COMPILE_CUDA)
    if(tmp_ptr_Neural_Network->Use__CUDA())
    { ptr_Dataset_Manager_received->Get__CUDA()->Type_Testing(ptr_Dataset_Manager_received->Get__Type_Dataset_Evaluation(), tmp_ptr_Neural_Network); }
    else
#endif
    { ptr_Dataset_Manager_received->Type_Testing(ptr_Dataset_Manager_received->Get__Type_Dataset_Evaluation(), tmp_ptr_Neural_Network); }

    for(tmp_individual_index = 1_zu; tmp_individual_index != this->_population_size; ++tmp_individual_index)
    {
        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_individual_index,
                                     __LINE__);

            return(false);
        }
    #endif
        
    #if defined(COMPILE_CUDA)
        if(tmp_ptr_Neural_Network->Use__CUDA())
        { ptr_Dataset_Manager_received->Get__CUDA()->Type_Testing(ptr_Dataset_Manager_received->Get__Type_Dataset_Evaluation(), tmp_ptr_Neural_Network); }
        else
    #endif
        { ptr_Dataset_Manager_received->Type_Testing(ptr_Dataset_Manager_received->Get__Type_Dataset_Evaluation(), tmp_ptr_Neural_Network); }
        
        if(this->p_ptr_array_ptr_individuals[tmp_best_individual_index]->Compare(ptr_Dataset_Manager_received->Use__Metric_Loss(),
                                                                                                                  ptr_Dataset_Manager_received->Get__Dataset_In_Equal_Less_Holdout_Accepted(),
                                                                                                                  ptr_Dataset_Manager_received->Get__Type_Dataset_Evaluation(),
                                                                                                                  ptr_Dataset_Manager_received->Get__Minimum_Loss_Holdout_Accepted(),
                                                                                                                  tmp_ptr_Neural_Network))
        { tmp_best_individual_index = tmp_individual_index; }
    }
    
    if(tmp_best_individual_index != 0_zu && this->p_ptr_array_ptr_individuals[0u]->Update(*this->p_ptr_array_ptr_individuals[tmp_best_individual_index],
                                                                                                                                 true,
                                                                                                                                 true) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update(ptr[%zu], true, true)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_best_individual_index,
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Evaluation__OpenMP(class Dataset_Manager<T> *const ptr_Dataset_Manager_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Dataset_Manager_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Dataset manager is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_number_threads == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of dataset manager is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_dataset_manager == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of dataset manager is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    size_t tmp_error_count(0_zu);

    int const tmp_population_size__int(static_cast<int>(this->_population_size));
    int tmp_individual_index__int(0),
         tmp_best_individual_index__int(0);
    
    class Neural_Network *tmp_ptr_Neural_Network(nullptr);
    
    class Dataset_Manager<T> *tmp_ptr_Dataset_Manager(nullptr);
    
    if(this->Update__Dataset_Manager(ptr_Dataset_Manager_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Dataset_Manager(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    // If the neural network use OpenMP, disable it.
    if(this->p_ptr_array_ptr_individuals[0u]->is_OpenMP_initialized && this->Disable__OpenMP__Population() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Disable__OpenMP__Population()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    omp_set_num_threads(static_cast<int>(this->_number_threads));

    #pragma omp parallel private(tmp_ptr_Neural_Network, tmp_ptr_Dataset_Manager)
    {
        #pragma omp single nowait
        {
            tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[0u];
            
        #if defined(_DEBUG) || defined(COMPILE_DEBUG)
            if(tmp_ptr_Neural_Network == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Neural network #0 is a nullptr. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                #pragma omp atomic
                ++tmp_error_count;
            }
        #endif
            
            tmp_ptr_Dataset_Manager = this->p_ptr_array_ptr_dataset_manager[omp_get_thread_num()];
            
        #if defined(_DEBUG) || defined(COMPILE_DEBUG)
            if(tmp_ptr_Dataset_Manager == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Dataset manager #%d is a nullptr. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         omp_get_thread_num(),
                                         __LINE__);
                
                #pragma omp atomic
                ++tmp_error_count;
            }
        #endif

        #if defined(COMPILE_CUDA)
            if(tmp_ptr_Neural_Network->Use__CUDA())
            { tmp_ptr_Dataset_Manager->Get__CUDA()->Type_Testing(tmp_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(), tmp_ptr_Neural_Network); }
            else
        #endif
            { tmp_ptr_Dataset_Manager->Type_Testing(tmp_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(), tmp_ptr_Neural_Network); }
        }
        
        #pragma omp for schedule(dynamic)
        for(tmp_individual_index__int = 1; tmp_individual_index__int < tmp_population_size__int; ++tmp_individual_index__int)
        {
            tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index__int];
            
        #if defined(_DEBUG) || defined(COMPILE_DEBUG)
            if(tmp_ptr_Neural_Network == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Neural network #%d is a nullptr. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_individual_index__int,
                                         __LINE__);
                
                #pragma omp atomic
                ++tmp_error_count;
            }
        #endif
            
            tmp_ptr_Dataset_Manager = this->p_ptr_array_ptr_dataset_manager[omp_get_thread_num()];
            
        #if defined(_DEBUG) || defined(COMPILE_DEBUG)
            if(tmp_ptr_Dataset_Manager == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Dataset manager #%d is a nullptr. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         omp_get_thread_num(),
                                         __LINE__);
                
                #pragma omp atomic
                ++tmp_error_count;
            }
        #endif
            
        #if defined(COMPILE_CUDA)
            if(tmp_ptr_Neural_Network->Use__CUDA())
            { tmp_ptr_Dataset_Manager->Get__CUDA()->Type_Testing(tmp_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(), tmp_ptr_Neural_Network); }
            else
        #endif
            { tmp_ptr_Dataset_Manager->Type_Testing(tmp_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(), tmp_ptr_Neural_Network); }
            
            #pragma omp critical
            if(this->p_ptr_array_ptr_individuals[tmp_best_individual_index__int]->Compare(tmp_ptr_Dataset_Manager->Use__Metric_Loss(),
                                                                                                                             tmp_ptr_Dataset_Manager->Get__Dataset_In_Equal_Less_Holdout_Accepted(),
                                                                                                                             tmp_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(),
                                                                                                                             tmp_ptr_Dataset_Manager->Get__Minimum_Loss_Holdout_Accepted(),
                                                                                                                             tmp_ptr_Neural_Network))
            { tmp_best_individual_index__int = tmp_individual_index__int; }
        }
    }
    
    // If the neural network was using OpenMP, enable it.
    if(this->p_ptr_array_ptr_individuals[0u]->is_OpenMP_initialized && this->Enable__OpenMP__Population() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Disable__OpenMP__Population()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(tmp_error_count > 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered inside the parallel region. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(tmp_best_individual_index__int != 0 && this->p_ptr_array_ptr_individuals[0u]->Update(*this->p_ptr_array_ptr_individuals[tmp_best_individual_index__int],
                                                                                                                                   true,
                                                                                                                                   true) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update(ptr[%d], true, true)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_best_individual_index__int,
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::User_Controls(void)
{
#if defined(COMPILE_UINPUT)
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Population size (%zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_population_size);
        PRINT_FORMAT("%s:\t[1]: Population gaussian (%f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_population_gaussian_percent);
        PRINT_FORMAT("%s:\t[2]: Add Hyperparameter." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[3]: Modify Hyperparameter." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[4]: OpenMP." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[5]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());
        
        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                5u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Population size:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[1, 8]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=60." NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__Population_Size(MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Population size: ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Population_Size()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Population gaussian in percent:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[2.0, 100.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=75.0%%." NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__Population_Gaussian(MyEA::String::Cin_Real_Number<double>(2.0,
                                                                                                                                 100.0,
                                                                                                                                 MyEA::String::Get__Time() + ": Population gaussian (percent): ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Population_Gaussian()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 2u:
                if(this->User_Controls__Push_Back() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Push_Back()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 3u:
                if(this->User_Controls__Hyperparameter_Manager() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Hyperparameter()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 4u:
                if(this->User_Controls__OpenMP() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__OpenMP()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 5u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         5u,
                                         __LINE__);
                    return(false);
        }
    }
#endif

    return(false);
}

template<typename T>
bool Gaussian_Search<T>::User_Controls__Push_Back(void)
{
#if defined(COMPILE_UINPUT)
    int tmp_option;
    
    size_t tmp_layer_index;

    T tmp_minimum_value,
       tmp_maximum_value,
       tmp_variance;

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, Hyper parameter push back:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Regularization, weight decay." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[1]: Regularization, L1." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[2]: Regularization, L2." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[3]: Regularization, max-norm constraints." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[4]: Normalization, average momentum." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[5]: Dropout, alpha, dropout probability." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[6]: Dropout, alpha, a." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[7]: Dropout, alpha, b." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[8]: Dropout, bernoulli, keep probability." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[9]: Dropout, bernoulli-inverted, keep probability." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[10]: Dropout, gaussian, dropout probability." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[11]: Dropout, uout, dropout probability." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[12]: Dropout, zoneout, cell zoneout probability." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[13]: Dropout, zoneout, hidden zoneout probability." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[14]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());
        
        if((tmp_option = MyEA::String::Cin_Number<int>(0,
                                                                               14,
                                                                               MyEA::String::Get__Time() + ": Option: ")) == 14) { return(true); }

        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Variance." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[1e-7, 8]." NEW_LINE, MyEA::String::Get__Time().c_str());
        tmp_variance = MyEA::String::Cin_Real_Number<T>(T(1.0e-7), MyEA::String::Get__Time() + ": Variance: ");

        switch(tmp_option)
        {
            case 0: // Regularization, Weight decay.
            case 1: // Regularization, L1.
            case 2: // Regularization, L2.
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Minimum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0, 1]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_minimum_value = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                        T(1),
                                                                                                        MyEA::String::Get__Time() + ": Minimum value: ");

                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Maximum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[%f, 1]." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(tmp_minimum_value));
                PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_maximum_value = MyEA::String::Cin_Real_Number<T>(tmp_minimum_value,
                                                                                                         T(1),
                                                                                                         MyEA::String::Get__Time() + ": Maximum value: ");

                if(this->Push_Back(tmp_option,
                                            0_zu,
                                            -(std::numeric_limits<ST_>::max)(),
                                            tmp_minimum_value,
                                            tmp_maximum_value,
                                            tmp_variance) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 3: // Regularization, Max-norm constraints.
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Minimum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0, 8]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_minimum_value = MyEA::String::Cin_Real_Number<T>(T(0), MyEA::String::Get__Time() + ": Minimum value: ");
                
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Maximum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[%f, 8]." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(tmp_minimum_value));
                PRINT_FORMAT("%s:\tdefault=16." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_maximum_value = MyEA::String::Cin_Real_Number<T>(tmp_minimum_value, MyEA::String::Get__Time() + ": Maximum value: ");

                if(this->Push_Back(3,
                                            0_zu,
                                            -(std::numeric_limits<ST_>::max)(),
                                            tmp_minimum_value,
                                            tmp_maximum_value,
                                            tmp_variance) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 4: // Normalization, average momentum.
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Minimum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0, %f]." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(T(1) - T(1.0e-7)));
                PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_minimum_value = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                        T(1) - T(1.0e-7),
                                                                                                        MyEA::String::Get__Time() + ": Minimum value: ");
                
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Maximum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[%f, %f]." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(tmp_minimum_value),
                                         Cast_T(T(1) - T(1.0e-7)));
                PRINT_FORMAT("%s:\tdefault=0.999." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_maximum_value = MyEA::String::Cin_Real_Number<T>(tmp_minimum_value,
                                                                                                         T(1) - T(1.0e-7),
                                                                                                         MyEA::String::Get__Time() + ": Maximum value: ");

                if(this->Push_Back(4,
                                            0_zu,
                                            -(std::numeric_limits<ST_>::max)(),
                                            tmp_minimum_value,
                                            tmp_maximum_value,
                                            tmp_variance) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 5: // Dropout, alpha, dropout probability.
            case 6: // Dropout, alpha, a.
            case 7: // Dropout, alpha, b.
            case 8: // Dropout, bernoulli, keep probability.
            case 9: // Dropout, bernoulli-inverted, keep probability.
            case 10: // Dropout, gaussian, dropout probability.
            case 11: // Dropout, uout, dropout probability.
            case 12: // Dropout, zoneout, cell zoneout probability.
            case 13: // Dropout, zoneout, hidden zoneout probability.
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_layer_index = MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Layer index: ");

                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Minimum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0, 1]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_minimum_value = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                        T(1),
                                                                                                        MyEA::String::Get__Time() + ": Minimum value: ");
                
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Maximum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[%f, 1]." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(tmp_minimum_value));
                PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_maximum_value = MyEA::String::Cin_Real_Number<T>(tmp_minimum_value,
                                                                                                         T(1),
                                                                                                         MyEA::String::Get__Time() + ": Maximum value: ");

                if(this->Push_Back(tmp_option,
                                            tmp_layer_index,
                                            -(std::numeric_limits<ST_>::max)(),
                                            tmp_minimum_value,
                                            tmp_maximum_value,
                                            tmp_variance) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<int>(%d, %d)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0,
                                         14,
                                         __LINE__);
                    return(false);
        }
    }
#endif

    return(false);
}

template<typename T>
std::string Gaussian_Search<T>::Get__ID_To_String(int const hyperparameter_id_received) const
{
    switch(hyperparameter_id_received)
    {
        case 0: return("Regularization, weight decay.");
        case 1: return("Regularization, L1.");
        case 2: return("Regularization, L2.");
        case 3: return("Regularization, max-norm constraints.");
        case 4: return("Normalization, average momentum.");
        case 5: return("Dropout, alpha, dropout probability.");
        case 6: return("Dropout, alpha, a.");
        case 7: return("Dropout, alpha, b.");
        case 8: return("Dropout, bernoulli, keep probability.");
        case 9: return("Dropout, bernoulli-inverted, keep probability.");
        case 10: return("Dropout, gaussian, dropout probability.");
        case 11: return("Dropout, uout, dropout probability.");
        case 12: return("Dropout, zoneout, cell zoneout probability.");
        case 13: return("Dropout, zoneout, hidden zoneout probability.");
        default:
            PRINT_FORMAT("%s: %s: ERROR: Hyper parameter id (%d) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     hyperparameter_id_received,
                                     __LINE__);
                return("");
    }
}

template<typename T>
bool Gaussian_Search<T>::User_Controls__Hyperparameter_Manager(void)
{
#if defined(COMPILE_UINPUT)
    size_t tmp_option,
              tmp_layer_index;

    T tmp_minimum_value,
       tmp_maximum_value,
       tmp_variance;

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, hyperparameter manager." NEW_LINE, MyEA::String::Get__Time().c_str());
        for(size_t tmp_hyperparameter_index(0); tmp_hyperparameter_index != this->_vector_hyperparameters.size(); ++tmp_hyperparameter_index)
        {
            PRINT_FORMAT("%s:\t[%zu]: %s (%d, %zu, %f, %f, %f, %f)." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_hyperparameter_index,
                                     this->Get__ID_To_String(std::get<0>(this->_vector_hyperparameters[tmp_hyperparameter_index])).c_str(),
                                     std::get<0>(this->_vector_hyperparameters[tmp_hyperparameter_index]),
                                     std::get<1>(this->_vector_hyperparameters[tmp_hyperparameter_index]),
                                     Cast_T(std::get<2>(this->_vector_hyperparameters[tmp_hyperparameter_index])),
                                     Cast_T(std::get<3>(this->_vector_hyperparameters[tmp_hyperparameter_index])),
                                     Cast_T(std::get<4>(this->_vector_hyperparameters[tmp_hyperparameter_index])),
                                     Cast_T(std::get<5>(this->_vector_hyperparameters[tmp_hyperparameter_index])));
        }

        PRINT_FORMAT("%s:\t[%zu]: Quit." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_vector_hyperparameters.size());
        
        if((tmp_option = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                     this->_vector_hyperparameters.size(),
                                                                                     MyEA::String::Get__Time() + ": Option: ")) <= this->_vector_hyperparameters.size())
        {
            if(tmp_option == this->_vector_hyperparameters.size()) { return(true); }

            tmp_layer_index = 0_zu;

            tmp_minimum_value = T(0);
            tmp_maximum_value = T(1);

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Variance." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[1e-7, 8]." NEW_LINE, MyEA::String::Get__Time().c_str());
            tmp_variance = MyEA::String::Cin_Real_Number<T>(T(1.0e-7), MyEA::String::Get__Time() + ": Variance: ");

            switch(std::get<0>(this->_vector_hyperparameters[tmp_option]))
            {
                case 0: // Regularization, Weight decay.
                case 1: // Regularization, L1.
                case 2: // Regularization, L2.
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Minimum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0, 1]." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());
                    tmp_minimum_value = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                            T(1),
                                                                                                            MyEA::String::Get__Time() + ": Minimum value: ");

                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Maximum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[%f, 1]." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             Cast_T(tmp_minimum_value));
                    PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::String::Get__Time().c_str());
                    tmp_maximum_value = MyEA::String::Cin_Real_Number<T>(tmp_minimum_value,
                                                                                                             T(1),
                                                                                                             MyEA::String::Get__Time() + ": Maximum value: ");
                        break;
                case 3: // Regularization, Max-norm constraints.
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Minimum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0, 8]." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());
                    tmp_minimum_value = MyEA::String::Cin_Real_Number<T>(T(0), MyEA::String::Get__Time() + ": Minimum value: ");
                
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Maximum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[%f, 8]." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             Cast_T(tmp_minimum_value));
                    PRINT_FORMAT("%s:\tdefault=16." NEW_LINE, MyEA::String::Get__Time().c_str());
                    tmp_maximum_value = MyEA::String::Cin_Real_Number<T>(tmp_minimum_value, MyEA::String::Get__Time() + ": Maximum value: ");
                        break;
                case 4: // Normalization, average momentum.
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Minimum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0, %f]." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             Cast_T(T(1) - T(1.0e-7)));
                    PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());
                    tmp_minimum_value = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                            T(1) - T(1.0e-7),
                                                                                                            MyEA::String::Get__Time() + ": Minimum value: ");
                
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Maximum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[%f, %f]." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             Cast_T(tmp_minimum_value),
                                             Cast_T(T(1) - T(1.0e-7)));
                    PRINT_FORMAT("%s:\tdefault=0.999." NEW_LINE, MyEA::String::Get__Time().c_str());
                    tmp_maximum_value = MyEA::String::Cin_Real_Number<T>(tmp_minimum_value,
                                                                                                             T(1) - T(1.0e-7),
                                                                                                             MyEA::String::Get__Time() + ": Maximum value: ");
                        break;
                case 5: // Dropout, alpha, dropout probability.
                case 6: // Dropout, alpha, a.
                case 7: // Dropout, alpha, b.
                case 8: // Dropout, bernoulli, keep probability.
                case 9: // Dropout, bernoulli-inverted, keep probability.
                case 10: // Dropout, gaussian, dropout probability.
                case 11: // Dropout, uout, dropout probability.
                case 12: // Dropout, zoneout, cell zoneout probability.
                case 13: // Dropout, zoneout, hidden zoneout probability.
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    tmp_layer_index = MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Layer index: ");

                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Minimum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0, 1]." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());
                    tmp_minimum_value = MyEA::String::Cin_Real_Number<T>(T(0),
                                                                                                            T(1),
                                                                                                            MyEA::String::Get__Time() + ": Minimum value: ");
                
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Maximum value." NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[%f, 1]." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             Cast_T(tmp_minimum_value));
                    PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::String::Get__Time().c_str());
                    tmp_maximum_value = MyEA::String::Cin_Real_Number<T>(tmp_minimum_value,
                                                                                                             T(1),
                                                                                                             MyEA::String::Get__Time() + ": Maximum value: ");
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<int>(%d, %d)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             0,
                                             14,
                                             __LINE__);
                        return(false);
            }

            std::get<1>(this->_vector_hyperparameters[tmp_option]) = tmp_layer_index;
            std::get<3>(this->_vector_hyperparameters[tmp_option]) = tmp_minimum_value;
            std::get<4>(this->_vector_hyperparameters[tmp_option]) = tmp_maximum_value;
            std::get<5>(this->_vector_hyperparameters[tmp_option]) = tmp_variance;
        }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<size_t>(%zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     0_zu,
                                     this->_vector_hyperparameters.size(),
                                     __LINE__);
        }
    }
#endif

    return(false);
}

template<typename T>
bool Gaussian_Search<T>::User_Controls__OpenMP(void)
{
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, OpenMP:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Use OpenMP (%s | %s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_use_OpenMP ? "Yes" : "No",
                                 this->_is_OpenMP_initialized ? "Yes" : "No");
        PRINT_FORMAT("%s:\t[1]: Maximum threads (%.2f%%)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_percentage_maximum_thread_usage);
        PRINT_FORMAT("%s:\t[2]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());
        
        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                2u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__OpenMP(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use OpenMP: ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__OpenMP()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Maximum threads:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0%%, 100.0%%]." NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__Maximum_Thread_Usage(MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                 100_T,
                                                                                                                                 MyEA::String::Get__Time() + ": Maximum threads (percent): ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Maximum_Thread_Usage()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 2u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         2u,
                                         __LINE__);
                    break;
        }
    }

    return(false);
}

template<typename T>
bool Gaussian_Search<T>::Push_Back(int const hyper_parameter_id_received,
                                                          size_t const index_received,
                                                          T const value_received,
                                                          T const minimum_value_received,
                                                          T const maximum_value_received,
                                                          T const variance_received)
{
    if(hyper_parameter_id_received >= 14)
    {
        PRINT_FORMAT("%s: %s: ERROR: Hyperparameter id (%d) undefined. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 hyper_parameter_id_received,
                                 __LINE__);

        return(false);
    }
    else if(minimum_value_received >= maximum_value_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value (%f) can not be greater or equal to maximum value (%f). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_value_received),
                                 Cast_T(maximum_value_received),
                                 __LINE__);
        
        return(false);
    }
    else if(variance_received <= T(0))
    {
        PRINT_FORMAT("%s: %s: ERROR: Variance can not be less or equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(variance_received > maximum_value_received - minimum_value_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Variance (%f) can not be greater to than %f. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(variance_received),
                                 Cast_T(maximum_value_received - minimum_value_received),
                                 __LINE__);

        return(false);
    }
    
    this->_vector_hyperparameters.push_back(std::tuple<int, size_t, T, T, T, T>(hyper_parameter_id_received,
                                                                                                                    index_received,
                                                                                                                    value_received,
                                                                                                                    minimum_value_received,
                                                                                                                    maximum_value_received,
                                                                                                                    variance_received));

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Initialize__Hyper_Parameters(class Neural_Network *const ptr_Neural_Network_received)
{
    for(size_t tmp_hyper_parameter_index(0_zu); tmp_hyper_parameter_index != this->_vector_hyperparameters.size(); ++tmp_hyper_parameter_index)
    {
        if(this->Initialize__Hyper_Parameter(this->_vector_hyperparameters[tmp_hyper_parameter_index], ptr_Neural_Network_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__Hyper_Parameter(hyper[%zu], ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_hyper_parameter_index,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Initialize__Hyper_Parameter(std::tuple<int, size_t, T, T, T, T> &ref_hyperparameter_tuple_received, class Neural_Network *const ptr_Neural_Network_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Neural_Network_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    struct Layer const *tmp_ptr_layer_it;

    switch(std::get<0>(ref_hyperparameter_tuple_received))
    {
        case 0: // Regularization, Weight decay.
            std::get<2>(ref_hyperparameter_tuple_received) = ptr_Neural_Network_received->regularization__weight_decay;
                break;
        case 1: // Regularization, L1.
            std::get<2>(ref_hyperparameter_tuple_received) = ptr_Neural_Network_received->regularization__l1;
                break;
        case 2: // Regularization, L2.
            std::get<2>(ref_hyperparameter_tuple_received) = ptr_Neural_Network_received->regularization__l2;
                break;
        case 3: // Regularization, Max-norm constraints.
            std::get<2>(ref_hyperparameter_tuple_received) = ptr_Neural_Network_received->regularization__max_norm_constraints;
                break;
        case 4: // Normalization, average momentum.
            std::get<2>(ref_hyperparameter_tuple_received) = ptr_Neural_Network_received->normalization_momentum_average;
                break;
        case 5: // Dropout, alpha, dropout probability.
        case 8: // Dropout, bernoulli, keep probability.
        case 9: // Dropout, bernoulli-inverted, keep probability.
        case 10: // Dropout, gaussian, dropout probability.
        case 11: // Dropout, uout, dropout probability.
        case 12: // Dropout, zoneout, cell zoneout probability.
            tmp_ptr_layer_it = ptr_Neural_Network_received->ptr_array_layers + std::get<1>(ref_hyperparameter_tuple_received);

            std::get<2>(ref_hyperparameter_tuple_received) = tmp_ptr_layer_it->dropout_values[0u];
                break;
        case 6: // Dropout, alpha, a.
        case 13: // Dropout, zoneout, hidden zoneout probability.
            tmp_ptr_layer_it = ptr_Neural_Network_received->ptr_array_layers + std::get<1>(ref_hyperparameter_tuple_received);

            std::get<2>(ref_hyperparameter_tuple_received) = tmp_ptr_layer_it->dropout_values[1u];
                break;
        case 7: // Dropout, alpha, b.
            tmp_ptr_layer_it = ptr_Neural_Network_received->ptr_array_layers + std::get<1>(ref_hyperparameter_tuple_received);

            std::get<2>(ref_hyperparameter_tuple_received) = tmp_ptr_layer_it->dropout_values[2u];
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Hyper parameter id (%d) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     std::get<0>(ref_hyperparameter_tuple_received),
                                     __LINE__);
                return(false);
    }
    
    // If value underflow.
    std::get<2>(ref_hyperparameter_tuple_received) = MyEA::Math::Maximum<T>(std::get<2>(ref_hyperparameter_tuple_received), std::get<3>(ref_hyperparameter_tuple_received));
    
    // If value overflow.
    std::get<2>(ref_hyperparameter_tuple_received) = MyEA::Math::Minimum<T>(std::get<2>(ref_hyperparameter_tuple_received), std::get<4>(ref_hyperparameter_tuple_received));

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Shuffle__Hyper_Parameter(void)
{
    if(this->_vector_hyperparameters.size() == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No hyper parameter available for shuffling. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    this->_Class_Generator_Random_Int.Range(0, static_cast<int>(this->_vector_hyperparameters.size()) - 1);

    this->_ptr_selected_hyperparameter = &this->_vector_hyperparameters.at(this->_Class_Generator_Random_Int.Generate_Integer());

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Feed__Hyper_Parameter(void)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->_population_size == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Population is empty. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_ptr_array_ptr_individuals == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Array of neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_ptr_selected_hyperparameter == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Selected hyper parameter is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    size_t const tmp_population_random_size(MyEA::Math::Maximum<size_t>(1_zu, static_cast<size_t>(floor(static_cast<double>(this->_population_size) * (100.0 - this->_population_gaussian_percent) / 100.0))));
    size_t tmp_individual_index;

    T const tmp_default_value(std::get<2>(*this->_ptr_selected_hyperparameter)),
                tmp_minimum_value(std::get<3>(*this->_ptr_selected_hyperparameter)),
                tmp_maximum_value(std::get<4>(*this->_ptr_selected_hyperparameter)),
                tmp_variance(std::get<5>(*this->_ptr_selected_hyperparameter));
    T tmp_generate_value;

    std::tuple<int, size_t, T, T, T, T> tmp_hyperparameter_tuple(*this->_ptr_selected_hyperparameter);

    class Neural_Network *tmp_ptr_Neural_Network;
    
    // Initialize random generator.
    this->_Class_Generator_Random_Real.Range(tmp_minimum_value, tmp_maximum_value);

    // Exploration.
    for(tmp_individual_index = 1_zu; tmp_individual_index != tmp_population_random_size; ++tmp_individual_index)
    {
        std::get<2>(tmp_hyperparameter_tuple) = this->_Class_Generator_Random_Real.Generate_Real();

        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_individual_index,
                                     __LINE__);

            return(false);
        }
    #endif

        if(this->Feed__Hyper_Parameter(tmp_hyperparameter_tuple, tmp_ptr_Neural_Network) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Feed__Hyper_Parameter(ref, ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    
    // Initialize gaussian generator.
    this->_Class_Generator_Random_Gaussian.Range(T(0), tmp_variance);
    
    // Exploitation.
    for(; tmp_individual_index != this->_population_size; ++tmp_individual_index)
    {
        do { tmp_generate_value = tmp_default_value + this->_Class_Generator_Random_Gaussian.Generate_Gaussian(); }
        while(tmp_generate_value < tmp_minimum_value || tmp_generate_value > tmp_maximum_value);

        std::get<2>(tmp_hyperparameter_tuple) = tmp_generate_value;

        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_individuals[tmp_individual_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_individual_index,
                                     __LINE__);

            return(false);
        }
    #endif

        if(this->Feed__Hyper_Parameter(tmp_hyperparameter_tuple, tmp_ptr_Neural_Network) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Feed__Hyper_Parameter(ref, ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    
    return(true);
}

template<typename T>
#if defined(COMPILE_x86) // TODO: Compatibility x86-x64. Internal compilator error.
bool Gaussian_Search<T>::Feed__Hyper_Parameter(std::tuple<int, size_t, T, T, T, T> const ref_hyperparameter_tuple_received, class Neural_Network *const ptr_Neural_Network_received)
#elif defined(COMPILE_x64)
bool Gaussian_Search<T>::Feed__Hyper_Parameter(std::tuple<int, size_t, T, T, T, T> const &ref_hyperparameter_tuple_received, class Neural_Network *const ptr_Neural_Network_received)
#endif
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Neural_Network_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Neural networks is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    struct Layer const *tmp_ptr_layer_it;

    switch(std::get<0>(ref_hyperparameter_tuple_received))
    {
        case 0: // Regularization, weight decay.
            if(ptr_Neural_Network_received->Set__Regularization__Weight_Decay(std::get<2>(ref_hyperparameter_tuple_received)) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Weight_Decay(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        case 1: // Regularization, L1.
            if(ptr_Neural_Network_received->Set__Regularization__L1(std::get<2>(ref_hyperparameter_tuple_received)) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__L1(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        case 2: // Regularization, L2.
            if(ptr_Neural_Network_received->Set__Regularization__L2(std::get<2>(ref_hyperparameter_tuple_received)) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__L2(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        case 3: // Regularization, max-norm constraints.
            if(ptr_Neural_Network_received->Set__Regularization__Max_Norm_Constraints(std::get<2>(ref_hyperparameter_tuple_received)) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Max_Norm_Constraints(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        case 4: // Normalization, average momentum.
            if(ptr_Neural_Network_received->Set__Normalization_Momentum_Average(std::get<2>(ref_hyperparameter_tuple_received)) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_Momentum_Average(%f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        case 5: // Dropout, alpha, dropout probability.
            tmp_ptr_layer_it = ptr_Neural_Network_received->ptr_array_layers + std::get<1>(ref_hyperparameter_tuple_received);

            if(ptr_Neural_Network_received->Set__Dropout(std::get<1>(ref_hyperparameter_tuple_received),
                                                                                MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA,
                                                                                std::array<T, 3_zu>{std::get<2>(ref_hyperparameter_tuple_received), tmp_ptr_layer_it->dropout_values[1u], tmp_ptr_layer_it->dropout_values[2u]}.data()) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         std::get<1>(ref_hyperparameter_tuple_received),
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         Cast_T(tmp_ptr_layer_it->dropout_values[1u]),
                                         Cast_T(tmp_ptr_layer_it->dropout_values[2u]),
                                         __LINE__);

                return(false);
            }
                break;
        case 6: // Dropout, alpha, a.
            tmp_ptr_layer_it = ptr_Neural_Network_received->ptr_array_layers + std::get<1>(ref_hyperparameter_tuple_received);

            if(ptr_Neural_Network_received->Set__Dropout(std::get<1>(ref_hyperparameter_tuple_received),
                                                                                MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA,
                                                                                std::array<T, 3_zu>{tmp_ptr_layer_it->dropout_values[0u], std::get<2>(ref_hyperparameter_tuple_received), tmp_ptr_layer_it->dropout_values[2u]}.data()) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         std::get<1>(ref_hyperparameter_tuple_received),
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA,
                                         Cast_T(tmp_ptr_layer_it->dropout_values[0u]),
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         Cast_T(tmp_ptr_layer_it->dropout_values[2u]),
                                         __LINE__);

                return(false);
            }
                break;
        case 7: // Dropout, alpha, b.
            tmp_ptr_layer_it = ptr_Neural_Network_received->ptr_array_layers + std::get<1>(ref_hyperparameter_tuple_received);

            if(ptr_Neural_Network_received->Set__Dropout(std::get<1>(ref_hyperparameter_tuple_received),
                                                                                MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA,
                                                                                std::array<T, 3_zu>{tmp_ptr_layer_it->dropout_values[0u], tmp_ptr_layer_it->dropout_values[1u], std::get<2>(ref_hyperparameter_tuple_received)}.data()) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f, %f, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         std::get<1>(ref_hyperparameter_tuple_received),
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA,
                                         Cast_T(tmp_ptr_layer_it->dropout_values[0u]),
                                         Cast_T(tmp_ptr_layer_it->dropout_values[1u]),
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        case 8: // Dropout, bernoulli, keep probability.
            if(ptr_Neural_Network_received->Set__Dropout(std::get<1>(ref_hyperparameter_tuple_received),
                                                                                MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI,
                                                                                std::array<T, 1_zu>{std::get<2>(ref_hyperparameter_tuple_received)}.data()) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         std::get<1>(ref_hyperparameter_tuple_received),
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        case 9: // Dropout, bernoulli-inverted, keep probability.
            if(ptr_Neural_Network_received->Set__Dropout(std::get<1>(ref_hyperparameter_tuple_received),
                                                                                MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED,
                                                                                std::array<T, 1_zu>{std::get<2>(ref_hyperparameter_tuple_received)}.data()) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         std::get<1>(ref_hyperparameter_tuple_received),
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        case 10: // Dropout, gaussian, dropout probability.
            if(ptr_Neural_Network_received->Set__Dropout(std::get<1>(ref_hyperparameter_tuple_received),
                                                                                MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN,
                                                                                std::array<T, 1_zu>{std::get<2>(ref_hyperparameter_tuple_received)}.data()) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         std::get<1>(ref_hyperparameter_tuple_received),
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        case 11: // Dropout, uout, dropout probability.
            if(ptr_Neural_Network_received->Set__Dropout(std::get<1>(ref_hyperparameter_tuple_received),
                                                                                MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT,
                                                                                std::array<T, 1_zu>{std::get<2>(ref_hyperparameter_tuple_received)}.data()) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         std::get<1>(ref_hyperparameter_tuple_received),
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        case 12: // Dropout, zoneout, cell zoneout probability.
            tmp_ptr_layer_it = ptr_Neural_Network_received->ptr_array_layers + std::get<1>(ref_hyperparameter_tuple_received);

            if(ptr_Neural_Network_received->Set__Dropout(std::get<1>(ref_hyperparameter_tuple_received),
                                                                                MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT,
                                                                                std::array<T, 2_zu>{std::get<2>(ref_hyperparameter_tuple_received), tmp_ptr_layer_it->dropout_values[1u]}.data()) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         std::get<1>(ref_hyperparameter_tuple_received),
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT,
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         Cast_T(tmp_ptr_layer_it->dropout_values[1u]),
                                         __LINE__);

                return(false);
            }
                break;
        case 13: // Dropout, zoneout, hidden zoneout probability.
            tmp_ptr_layer_it = ptr_Neural_Network_received->ptr_array_layers + std::get<1>(ref_hyperparameter_tuple_received);

            if(ptr_Neural_Network_received->Set__Dropout(std::get<1>(ref_hyperparameter_tuple_received),
                                                                                MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT,
                                                                                std::array<T, 2_zu>{tmp_ptr_layer_it->dropout_values[0u], std::get<2>(ref_hyperparameter_tuple_received)}.data()) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f, %f)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         std::get<1>(ref_hyperparameter_tuple_received),
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT,
                                         Cast_T(tmp_ptr_layer_it->dropout_values[0u]),
                                         Cast_T(std::get<2>(ref_hyperparameter_tuple_received)),
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Hyper parameter id (%d) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     std::get<0>(ref_hyperparameter_tuple_received),
                                     __LINE__);
                return(false);
    }

    return(true);
}

template<typename T>
bool Gaussian_Search<T>::Deinitialize__OpenMP(void)
{
    if(this->_is_OpenMP_initialized)
    {
        this->Deallocate__Dataset_Manager();

        this->_cache_number_threads = this->_number_threads = 0_zu;

        this->_is_OpenMP_initialized = false;
    }

    return(true);
}

template<typename T>
void Gaussian_Search<T>::Deallocate__Dataset_Manager(void)
{
    SAFE_DELETE_ARRAY(this->p_ptr_array_dataset_manager);
    SAFE_DELETE_ARRAY(this->p_ptr_array_ptr_dataset_manager);
}

template<typename T>
void Gaussian_Search<T>::Deallocate__Population(void)
{
    SAFE_DELETE_ARRAY(this->p_ptr_array_individuals);
    SAFE_DELETE_ARRAY(this->p_ptr_array_ptr_individuals);
}

template<typename T>
void Gaussian_Search<T>::Deallocate(void)
{
    this->Deallocate__Dataset_Manager();
    this->Deallocate__Population();
}

template<typename T>
Gaussian_Search<T>::~Gaussian_Search(void) { this->Deallocate(); }

// template initialization declaration.
template class Gaussian_Search<T_>;