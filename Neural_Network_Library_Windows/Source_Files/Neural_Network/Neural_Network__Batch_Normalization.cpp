#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

bool Neural_Network::Set__Normalization_Momentum_Average(T_ const momentum_average_received)
{
    if(this->normalization_momentum_average == momentum_average_received) { return(true); }

    this->normalization_momentum_average = momentum_average_received;

#if defined(COMPILE_CUDA)
    if(this->is_device_initialized)
    { this->ptr_device_Neural_Network->Set__Normalization_Momentum_Average(momentum_average_received); }
#endif

    return(true);
}

bool Neural_Network::Set__Normalization_Epsilon(T_ const epsilon_received)
{
    if(this->normalization_epsilon == epsilon_received) { return(true); }

    this->normalization_epsilon = epsilon_received;

#if defined(COMPILE_CUDA)
    if(this->is_device_initialized)
    { this->ptr_device_Neural_Network->Set__Normalization_Epsilon(epsilon_received); }
#endif

    return(true);
}
