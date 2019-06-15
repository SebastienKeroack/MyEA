#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

bool Neural_Network::Set__Accurancy_Variance(T_ const accurancy_variance_received)
{
    if(this->accuracy_variance == accurancy_variance_received) { return(true); }
    else if(accurancy_variance_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Accuracy variance (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(accurancy_variance_received),
                                 __LINE__);

        return(false);
    }
    else if(accurancy_variance_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Accuracy variance (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(accurancy_variance_received),
                                 __LINE__);

        return(false);
    }

    this->accuracy_variance = accurancy_variance_received;
    
#if defined(COMPILE_CUDA)
    if(this->is_device_initialized)
    { this->ptr_device_Neural_Network->Set__Accurancy_Variance(accurancy_variance_received); }
#endif

    return(true);
}

bool Neural_Network::Set__Number_Time_Delays(size_t const time_delays_received)
{
    if(this->number_time_delays == time_delays_received) { return(true); }
    else if(time_delays_received > this->number_recurrent_depth)
    {
        PRINT_FORMAT("%s: %s: ERROR: Time delays (%zu) bigger than recurrent depth (%zu). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 time_delays_received,
                                 this->number_recurrent_depth,
                                 __LINE__);

        return(false);
    }

    this->number_time_delays = time_delays_received;
    
#if defined(COMPILE_CUDA)
    if(this->is_device_initialized)
    { this->ptr_device_Neural_Network->Set__Number_Time_Delays(time_delays_received); }
#endif

    return(true);
}

void Neural_Network::Set__Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, T_ const accurancy_received)
{
    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: this->accuracy_training = accurancy_received; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: this->accuracy_validating = accurancy_received; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: this->accuracy_testing = accurancy_received; break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Accuracy type (%u) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     type_dataset_received);
                break;
    }
}

T_ Neural_Network::Get__Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received) const
{
    T_ tmp_accurancy;

    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: tmp_accurancy = this->accuracy_training; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: tmp_accurancy = this->accuracy_validating; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: tmp_accurancy = this->accuracy_testing; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE:
            switch(this->type_accuracy_function)
            {
                case MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_R: tmp_accurancy = MyEA::Math::Clip<T_>(this->ptr_array_accuracy_values[0u][0u], -1_T, 1_T); break; // Floating-precession clip.
                default: tmp_accurancy = this->number_accuracy_trial == 0_zu ? 0_T : this->ptr_array_accuracy_values[0u][0u] / static_cast<T_>(this->number_accuracy_trial) * 100_T; break;
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Accuracy type (%u) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     type_dataset_received);

            tmp_accurancy = 0_T;
                break;
    }

    return(tmp_accurancy);
}
