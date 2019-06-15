#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

void Neural_Network::Reset__Global_Loss(void)
{
    this->loss_training = (std::numeric_limits<ST_>::max)();
    this->loss_validating = (std::numeric_limits<ST_>::max)();
    this->loss_testing = (std::numeric_limits<ST_>::max)();

    this->accuracy_training = 0_T;
    this->accuracy_validating = 0_T;
    this->accuracy_testing = 0_T;
}

void Neural_Network::Reset__Loss(void)
{
#if defined(COMPILE_CUDA)
    if(this->is_device_initialized)
    { this->ptr_device_Neural_Network->Reset__Loss(); }
    else
#endif
    {
        this->number_accuracy_trial = 0_zu;

        if(this->ptr_array_number_bit_fail != nullptr)
        {
            memset(this->ptr_array_number_bit_fail,
                        0,
                        this->number_threads * sizeof(size_t));
        }
        
        if(this->ptr_array_number_loss != nullptr)
        {
            memset(this->ptr_array_number_loss,
                        0,
                        this->number_threads * sizeof(size_t));
        }
        
        if(this->ptr_array_loss_values != nullptr)
        {
            MEMSET(this->ptr_array_loss_values,
                        0,
                        this->number_threads * sizeof(T_));
        }
        
        if(this->ptr_array_accuracy_values[0u] != nullptr)
        {
            MEMSET(this->ptr_array_accuracy_values[0u],
                        0,
                        this->number_threads * sizeof(T_));
        }
        
        if(this->ptr_array_accuracy_values[1u] != nullptr)
        {
            MEMSET(this->ptr_array_accuracy_values[1u],
                        0,
                        this->number_threads * sizeof(T_));
        }
        
        if(this->ptr_array_accuracy_values[2u] != nullptr)
        {
            MEMSET(this->ptr_array_accuracy_values[2u],
                        0,
                        this->number_threads * sizeof(T_));
        }
        
        if(this->ptr_array_accuracy_values[3u] != nullptr)
        {
            MEMSET(this->ptr_array_accuracy_values[3u],
                        0,
                        this->number_threads * sizeof(T_));
        }
        
        if(this->ptr_array_accuracy_values[4u] != nullptr)
        {
            MEMSET(this->ptr_array_accuracy_values[4u],
                        0,
                        this->number_threads * sizeof(T_));
        }
    }
}
    
T_ Neural_Network::Get__Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received) const
{
    T_ tmp_loss;

    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: tmp_loss = this->loss_training; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: tmp_loss = this->loss_validating; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: tmp_loss = this->loss_testing; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE:
            switch(this->type_loss_function)
            {
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_ME: tmp_loss = this->Get__ME(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L1: tmp_loss = this->Get__Loss_L1(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MAE: tmp_loss = this->Get__MAE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L2: tmp_loss = this->Get__Loss_L2(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MSE: tmp_loss = this->Get__MSE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_RMSE: tmp_loss = this->Get__RMSE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MAPE: tmp_loss = this->Get__MAPE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_SMAPE: tmp_loss = this->Get__SMAPE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MASE_SEASONAL: tmp_loss = this->Get__MASE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MASE_NON_SEASONAL: tmp_loss = this->Get__MASE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_CROSS_ENTROPY: tmp_loss = this->Get__ACE(); break;
                case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_BIT: tmp_loss = this->Get__BITFAIL(); break;
                default: tmp_loss = 1_T; break;
            }
                break;
        default: tmp_loss = 1_T; break;
    }

    return(tmp_loss);
}
    
T_ Neural_Network::Get__ME(void) const // https://en.wikipedia.org/wiki/Mean_absolute_error
{
    if(*this->ptr_array_number_loss != 0_zu)
    { return(*this->ptr_array_loss_values / static_cast<T_>(*this->ptr_array_number_loss)); }
    else
    { return(1_T); }
}
    
T_ Neural_Network::Get__Loss_L1(void) const
{ return(*this->ptr_array_loss_values); }
    
T_ Neural_Network::Get__MAE(void) const // https://en.wikipedia.org/wiki/Mean_absolute_error
{
    if(*this->ptr_array_number_loss != 0_zu)
    { return(*this->ptr_array_loss_values / static_cast<T_>(*this->ptr_array_number_loss)); }
    else
    { return(1_T); }
}
    
T_ Neural_Network::Get__Loss_L2(void) const
{ return(*this->ptr_array_loss_values); }
    
T_ Neural_Network::Get__MSE(void) const // https://en.wikipedia.org/wiki/Mean_squared_error
{
    if(*this->ptr_array_number_loss != 0_zu)
    { return(1_T / static_cast<T_>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values); }
    else
    { return(1_T); }
}
    
T_ Neural_Network::Get__RMSE(void) const // https://en.wikipedia.org/wiki/Root-mean-square_deviation
{
    if(*this->ptr_array_number_loss != 0_zu)
    { return(sqrt(1_T / static_cast<T_>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values)); }
    else
    { return(1_T); }
}
    
T_ Neural_Network::Get__MAPE(void) const // https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
{
    if(*this->ptr_array_number_loss != 0_zu)
    { return(1_T / static_cast<T_>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values); }
    else
    { return(1_T); }
}
    
T_ Neural_Network::Get__SMAPE(void) const // https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
{
    if(*this->ptr_array_number_loss != 0_zu)
    { return(1_T / static_cast<T_>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values); }
    else
    { return(1_T); }
}

T_ Neural_Network::Get__MASE(void) const // https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
{
    // Non seasonal time series
    //if(*this->ptr_array_number_loss != 0_zu
    //   &&
    //   this->mean_absolute_error_denominator != 0.0f
    //   &&
    //   *this->ptr_array_number_loss > 1u)
    //{ return(*this->ptr_array_loss_values / ((static_cast<T_>(*this->ptr_array_number_loss) / static_cast<T_>(*this->ptr_array_number_loss - 1u)) * this->mean_absolute_error_denominator)); }
    //{ return(1_T / *this->ptr_array_number_loss * (*this->ptr_array_loss_values / ((1_T / static_cast<T_>(*this->ptr_array_number_loss - 1u)) * this->mean_absolute_error_denominator))); }
    //{ return(1_T / this->number_recurrent_depth * (*this->ptr_array_loss_values / ((1_T / static_cast<T_>(this->number_recurrent_depth - 1_zu)) * this->mean_absolute_error_denominator))); }
    /*else*/    { return(1_T); }
}

T_ Neural_Network::Get__ACE(void) const // https://en.wikipedia.org/wiki/Cross_entropy
{
    if(*this->ptr_array_number_loss != 0_zu)
    { return(*this->ptr_array_loss_values / static_cast<T_>(*this->ptr_array_number_loss / this->Get__Output_Size())); }
    else
    { return((std::numeric_limits<ST_>::max)()); }
}

T_ Neural_Network::Get__BITFAIL(void) const // link
{ return(static_cast<T_>(*this->ptr_array_number_bit_fail)); }
    
void Neural_Network::Set__Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, T_ const loss_received)
{
    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: this->loss_training = loss_received; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: this->loss_validating = loss_received; break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: this->loss_testing = loss_received; break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Loss type (%u) is not managed in the switch." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     type_dataset_received);
                break;
    }
}
