#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

bool Neural_Network::Set__Batch_Renormalization_r_Correction_Maximum(T_ const r_correction_maximum_received)
{
    if(this->batch_renormalization_r_correction_maximum == r_correction_maximum_received) { return(true); }

    this->batch_renormalization_r_correction_maximum = r_correction_maximum_received;

#if defined(COMPILE_CUDA)
    if(this->is_device_initialized)
    { this->ptr_device_Neural_Network->Set__Batch_Renormalization_r_Correction_Maximum(r_correction_maximum_received); }
#endif

    return(true);
}

bool Neural_Network::Set__Batch_Renormalization_d_Correction_Maximum(T_ const d_correction_maximum_received)
{
    if(this->batch_renormalization_d_correction_maximum == d_correction_maximum_received) { return(true); }

    this->batch_renormalization_d_correction_maximum = d_correction_maximum_received;

#if defined(COMPILE_CUDA)
    if(this->is_device_initialized)
    { this->ptr_device_Neural_Network->Set__Batch_Renormalization_d_Correction_Maximum(d_correction_maximum_received); }
#endif

    return(true);
}
