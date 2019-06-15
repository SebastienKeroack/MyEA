#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::Initialize__OpenMP(void)
{
    if(this->is_OpenMP_initialized == false)
    {
        this->is_OpenMP_initialized = true;

        omp_set_dynamic(0);
    }
}

bool Neural_Network::Set__OpenMP(bool const use_openmp_received)
{
    if((this->use_OpenMP == false && use_openmp_received)
      ||
      (this->use_OpenMP && use_openmp_received && this->is_OpenMP_initialized == false))
    { this->Initialize__OpenMP(); }
    else if((this->use_OpenMP && use_openmp_received == false)
              ||
              (this->use_OpenMP == false && use_openmp_received == false && this->is_OpenMP_initialized))
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

    this->use_OpenMP = use_openmp_received;

    return(true);
}

bool Neural_Network::Set__Maximum_Thread_Usage(double const percentage_maximum_thread_usage_received)
{
    if(this->percentage_maximum_thread_usage == percentage_maximum_thread_usage_received) { return(true); }

    this->percentage_maximum_thread_usage = percentage_maximum_thread_usage_received;

    if(this->Update__Thread_Size(this->cache_number_threads) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 this->cache_number_threads,
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::Deinitialize__OpenMP(void)
{
    if(this->is_OpenMP_initialized)
    {
        if(this->Reallocate__Thread(1_zu) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Thread(1)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        this->cache_number_threads = this->number_threads = 1_zu;

        this->is_OpenMP_initialized = false;
    }

    return(true);
}
