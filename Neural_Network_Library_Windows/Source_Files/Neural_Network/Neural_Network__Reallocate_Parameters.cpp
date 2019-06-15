#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

bool Neural_Network::Reallocate__Parameter(size_t const number_parameters_received)
{
    if(this->total_parameters_allocated != 0u)
    {
        // Connection index.
        if(this->ptr_array_ptr_connections != nullptr)
        {
            void **tmp_ptr_array_ptr_connections(Memory::reallocate_pointers_array_cpp<void*>(this->ptr_array_ptr_connections,
                                                                                                                                       number_parameters_received,
                                                                                                                                       this->total_parameters_allocated,
                                                                                                                                       true));
            if(tmp_ptr_array_ptr_connections == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_pointers_array_cpp<%zu>(ptr, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         sizeof(void*),
                                         number_parameters_received,
                                         this->total_parameters_allocated,
                                         __LINE__);

                return(false);
            }
            this->ptr_array_ptr_connections = tmp_ptr_array_ptr_connections;
        }
        // |END| Connection index. |END|

        // Parameters.
        if(this->ptr_array_parameters != nullptr)
        {
            T_ *tmp_ptr_array_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_parameters,
                                                                                                    number_parameters_received,
                                                                                                    this->total_parameters_allocated));
            if(tmp_ptr_array_parameters == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         sizeof(T_),
                                         number_parameters_received,
                                         this->total_parameters_allocated,
                                         __LINE__);

                return(false);
            }
            this->ptr_array_parameters = tmp_ptr_array_parameters;
            
            if(this->Reallocate__Parameter__Optimizer(number_parameters_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Parameter__Optimizer(%zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         number_parameters_received,
                                         __LINE__);

                return(false);
            }
            else if(this->Use__Regularization_Parameter() && this->Reallocate__Parameter__Regularization(number_parameters_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Parameter__Regularization(%zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         number_parameters_received,
                                         __LINE__);

                return(false);
            }
            
            if(this->Use__Normalization()) { this->Reset__Parameter__Normalized_Unit(); }
        }
        // |END| Parameters. |END|

        // Derivates parameters.
        if(this->ptr_array_derivatives_parameters != nullptr)
        {
            T_ *tmp_ptr_array_derivatives_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_derivatives_parameters,
                                                                                                                    this->number_threads * number_parameters_received,
                                                                                                                    this->number_threads * this->total_parameters_allocated));
            if(tmp_ptr_array_derivatives_parameters == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         sizeof(T_),
                                         number_parameters_received,
                                         this->total_parameters_allocated,
                                         __LINE__);

                return(false);
            }
            this->ptr_array_derivatives_parameters = tmp_ptr_array_derivatives_parameters;

            if(this->Use__Normalization()) { this->Reset__Derivative_Parameter__Normalized_Unit(); }
        }
        // |END| Derivates parameters. |END|

        this->total_parameters = number_parameters_received;
        this->total_parameters_allocated = number_parameters_received;
    }

    return(true);
}

bool Neural_Network::Reallocate__Parameter__Regularization(size_t const number_parameters_received)
{
    if(this->ptr_array_mask_regularized_parameters != nullptr)
    {
        // Mask regularization parameters.
        T_ *tmp_ptr_array_mask_rergularization_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_mask_regularized_parameters,
                                                                                                                                number_parameters_received,
                                                                                                                                this->total_parameters_allocated));
        if(tmp_ptr_array_mask_rergularization_parameters == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_parameters_received,
                                     this->total_parameters_allocated,
                                     __LINE__);

            return(false);
        }
        MEMSET(tmp_ptr_array_mask_rergularization_parameters + this->total_weights_allocated,
                       0,
                       (number_parameters_received - this->total_weights_allocated) * sizeof(T_));

        this->ptr_array_mask_regularized_parameters = tmp_ptr_array_mask_rergularization_parameters;
        // |END| Mask regularization parameters. |END|
    }

    return(true);
}

bool Neural_Network::Reallocate__Parameter__Optimizer(size_t const number_parameters_received)
{
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD: return(this->Reallocate__Parameter__Gradient_Descent(number_parameters_received));
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus: return(this->Reallocate__Parameter__iRPROP_minus(number_parameters_received));
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: return(this->Reallocate__Parameter__iRPROP_plus(number_parameters_received));
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: return(this->Reallocate__Parameter__Adam(number_parameters_received));
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: return(this->Reallocate__Parameter__AMSGrad(number_parameters_received));
        default: return(true);
    }
}

bool Neural_Network::Reallocate__Parameter__Gradient_Descent(size_t const number_parameters_received)
{
    if(this->learning_momentum != 0_T
      &&
      this->ptr_array_previous_delta_parameters != nullptr)
    {
        // Previous delta parameters.
        T_ *tmp_ptr_array_previous_delta_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_previous_delta_parameters,
                                                                                                                      number_parameters_received,
                                                                                                                      this->total_parameters_allocated));
        if(tmp_ptr_array_previous_delta_parameters == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_parameters_received,
                                     this->total_parameters_allocated,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_previous_delta_parameters = tmp_ptr_array_previous_delta_parameters;
        // |END| Previous delta parameters. |END|
    }

    return(true);
}

bool Neural_Network::Reallocate__Parameter__iRPROP_minus(size_t const number_parameters_received)
{
    if(this->ptr_array_previous_steps != nullptr)
    {
        T_ *tmp_ptr_array_previous_steps(Memory::reallocate_cpp<T_>(this->ptr_array_previous_steps,
                                                                                                     number_parameters_received,
                                                                                                     this->total_parameters_allocated));
        if(tmp_ptr_array_previous_steps == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_parameters_received,
                                     this->total_parameters_allocated,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_previous_steps = tmp_ptr_array_previous_steps;
        
        Memory::Fill<T_>(this->ptr_array_previous_steps + this->total_weights_allocated,
                                  this->ptr_array_previous_steps + number_parameters_received,
                                  this->rprop_delta_zero);
    }
    
    if(this->ptr_array_previous_derivatives_parameters != nullptr)
    {
        T_ *tmp_ptr_array_previous_derivatives_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_previous_derivatives_parameters,
                                                                                                                              number_parameters_received,
                                                                                                                              this->total_parameters_allocated));
        if(tmp_ptr_array_previous_derivatives_parameters == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_parameters_received,
                                     this->total_parameters_allocated,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_previous_derivatives_parameters = tmp_ptr_array_previous_derivatives_parameters;
    }

    return(true);
}

bool Neural_Network::Reallocate__Parameter__iRPROP_plus(size_t const number_parameters_received)
{
    if(this->Reallocate__Parameter__iRPROP_minus(number_parameters_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Parameter__iRPROP_minus()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->ptr_array_previous_delta_parameters != nullptr)
    {
        T_ *tmp_ptr_array_previous_delta_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_previous_delta_parameters,
                                                                                                                      number_parameters_received,
                                                                                                                      this->total_parameters_allocated));
        if(tmp_ptr_array_previous_delta_parameters == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_parameters_received,
                                     this->total_parameters_allocated,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_previous_delta_parameters = tmp_ptr_array_previous_delta_parameters;
    }

    return(true);
}

bool Neural_Network::Reallocate__Parameter__Adam(size_t const number_parameters_received)
{
    if(this->ptr_array_previous_biased_first_moment != nullptr)
    {
        T_ *tmp_ptr_array_previous_biased_first_moment(Memory::reallocate_cpp<T_>(this->ptr_array_previous_biased_first_moment,
                                                                                                                           number_parameters_received,
                                                                                                                           this->total_parameters_allocated));
        if(tmp_ptr_array_previous_biased_first_moment == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_parameters_received,
                                     this->total_parameters_allocated,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_previous_biased_first_moment = tmp_ptr_array_previous_biased_first_moment;
    }
    
    if(this->ptr_array_previous_biased_second_moment != nullptr)
    {
        T_ *tmp_ptr_array_previous_biased_second_moment(Memory::reallocate_cpp<T_>(this->ptr_array_previous_biased_second_moment,
                                                                                                                                number_parameters_received,
                                                                                                                                this->total_parameters_allocated));
        if(tmp_ptr_array_previous_biased_second_moment == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_parameters_received,
                                     this->total_parameters_allocated,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_previous_biased_second_moment = tmp_ptr_array_previous_biased_second_moment;
    }

    return(true);
}

bool Neural_Network::Reallocate__Parameter__AMSGrad(size_t const number_parameters_received)
{
    if(this->Reallocate__Parameter__Adam(number_parameters_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Parameter__Adam()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->ptr_array_previous_biased_second_moment_hat != nullptr)
    {
        T_ *tmp_ptr_array_previous_biased_second_moment_hat(Memory::reallocate_cpp<T_>(this->ptr_array_previous_biased_second_moment_hat,
                                                                                                                                      number_parameters_received,
                                                                                                                                      this->total_parameters_allocated));
        if(tmp_ptr_array_previous_biased_second_moment_hat == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(T_),
                                     number_parameters_received,
                                     this->total_parameters_allocated,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_previous_biased_second_moment_hat = tmp_ptr_array_previous_biased_second_moment_hat;
    }

    return(true);
}