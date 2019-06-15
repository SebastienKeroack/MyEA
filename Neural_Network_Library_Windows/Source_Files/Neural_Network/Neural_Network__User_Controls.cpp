#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <iostream>
#include <array>

bool Neural_Network::User_Controls__Weights_Initializer(void)
{
    struct Weights_Initializer tmp_Weights_Initializer;

    if(tmp_Weights_Initializer.Input_Initialize() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Input_Initialize()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                    
        return(false);
    }

    if(tmp_Weights_Initializer.Output_Initialize(this) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Output_Initialize()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                    
        return(false);
    }

    return(true);
}

bool Neural_Network::User_Controls__Optimizer_Function_Initializer(void)
{
    struct Optimizer_Function_Initializer tmp_Optimizer_Function_Initializer;
    
    if(tmp_Optimizer_Function_Initializer.Input_Initialize() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Input_Initialize()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                    
        return(false);
    }

    if(tmp_Optimizer_Function_Initializer.Output_Initialize(this) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Output_Initialize()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                    
        return(false);
    }

    return(true);
}

bool Neural_Network::User_Controls__Loss_Function_Initializer(void)
{
    struct Loss_Function_Initializer tmp_Loss_Function_Initializer;
    
    if(tmp_Loss_Function_Initializer.Input_Initialize() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Input_Initialize()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                    
        return(false);
    }

    tmp_Loss_Function_Initializer.Output_Initialize(this);

    return(true);
}

bool Neural_Network::User_Controls__Accuracy_Function_Initializer(void)
{
    struct Accuracy_Function_Initializer tmp_Accuracy_Function_Initializer;
    
    if(tmp_Accuracy_Function_Initializer.Input_Initialize() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Input_Initialize()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                    
        return(false);
    }

    tmp_Accuracy_Function_Initializer.Output_Initialize(this);

    return(true);
}

bool Neural_Network::User_Controls__Optimizer_Function(void)
{
    if(this->Usable_Warm_Restarts())
    {
        while(true)
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: User controls, optimizer function %s:" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
            PRINT_FORMAT("%s:\t[0]: Modify optimizer function hyper-parameters." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\t[1]: Modify warm restarts." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\t[2]: Change optimizer function." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\t[3]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

            switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                    3u,
                                                                                    MyEA::String::Get__Time() + ": Option: "))
            {
                case 0u:
                    switch(this->type_optimizer_function)
                    {
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
                            if(this->User_Controls__Optimizer__Gradient_Descent() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer__Gradient_Descent()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
                            if(this->User_Controls__Optimizer__iRPROP() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer__iRPROP()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad:
                            if(this->User_Controls__Optimizer__Adam() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer__Adam()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM:
                            if(this->User_Controls__Optimizer__NosAdam() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer__NosAdam()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
                            if(this->User_Controls__Optimizer__AdaBound() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer__AdaBound()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        default:
                            PRINT_FORMAT("%s: ERROR: Unknow type optimizer function (%u | %s)." NEW_LINE,
                                                     __FUNCTION__,
                                                     this->type_optimizer_function,
                                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
                                break;
                    }
                        break;
                case 1u:
                    if(this->User_Controls__Warm_Restarts() == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Warm_Restarts()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                        break;
                case 2u:
                    if(this->User_Controls__Optimizer_Function_Initializer() == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer_Function_Initializer()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                        break;
                case 3u: return(true);
                default:
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             0u,
                                             3u,
                                             __LINE__);
                        break;
            }
        }
    }
    else
    {
        while(true)
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: User controls, optimizer function %s:" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
            PRINT_FORMAT("%s:\t[0]: Modify optimizer function hyper-parameters." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\t[1]: Change optimizer function." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\t[2]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

            switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                    2u,
                                                                                    MyEA::String::Get__Time() + ": Option: "))
            {
                case 0u:
                    switch(this->type_optimizer_function)
                    {
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
                            if(this->User_Controls__Optimizer__Gradient_Descent() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer__Gradient_Descent()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
                            if(this->User_Controls__Optimizer__iRPROP() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer__iRPROP()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad:
                            if(this->User_Controls__Optimizer__Adam() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer__Adam()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM:
                            if(this->User_Controls__Optimizer__NosAdam() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer__NosAdam()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
                        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
                            if(this->User_Controls__Optimizer__AdaBound() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer__AdaBound()\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        default:
                            PRINT_FORMAT("%s: ERROR: Unknow type optimizer function (%u | %s)." NEW_LINE,
                                                     __FUNCTION__,
                                                     this->type_optimizer_function,
                                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
                                break;
                    }
                        break;
                case 1u:
                    if(this->User_Controls__Optimizer_Function_Initializer() == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer_Function_Initializer()\" function. At line %d." NEW_LINE,
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
    }

    return(false);
}

bool Neural_Network::User_Controls__Optimizer__Gradient_Descent(void)
{
#if defined(COMPILE_CUDA)
    bool tmp_parameters_has_change(false);
#endif

    T_ tmp_hyper_paramater;

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, %s:" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
        PRINT_FORMAT("%s:\t[0]: Modify learning rate (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->learning_rate));
        PRINT_FORMAT("%s:\t[1]: Modify learning momentum (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->learning_momentum));
        PRINT_FORMAT("%s:\t[2]: Use Nesterov (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->use_Nesterov ? "Yes" : "No");
        PRINT_FORMAT("%s:\t[3]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                3u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Learning rate." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.01." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->learning_rate;
            #endif

                this->learning_rate = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning rate: ");
            
            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->learning_rate) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Learning momentum." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.9." NEW_LINE, MyEA::String::Get__Time().c_str());

                tmp_hyper_paramater = this->learning_momentum;

                this->learning_momentum = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning rate: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->learning_momentum) { tmp_parameters_has_change = true; }
            #endif

                if(tmp_hyper_paramater == 0_T)
                {
                    if(this->Allocate__Parameter__Gradient_Descent() == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter__Gradient_Descent()\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                }
                else if(this->learning_momentum == 0_T)
                { this->Deallocate__Parameter__Gradient_Descent(); }
                    break;
            case 2u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->learning_momentum != 0_T)
                { this->use_Nesterov = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use Nesterov?"); }
                else
                { PRINT_FORMAT("%s: WARNING: Can not use Nesterov without momentum." NEW_LINE, MyEA::String::Get__Time().c_str()); }
                    break;
            case 3u:
            #if defined(COMPILE_CUDA)
                if(this->is_device_initialized && tmp_parameters_has_change)
                { this->ptr_device_Neural_Network->Copy__Gradient_Descent_Parameters(this); }
            #endif
                    return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         3u,
                                         __LINE__);
                    break;
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__Optimizer__iRPROP(void)
{
#if defined(COMPILE_CUDA)
    bool tmp_parameters_has_change(false);

    T_ tmp_hyper_paramater;
#endif

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, %s:" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
        PRINT_FORMAT("%s:\t[0]: Modify increase factor (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->rprop_increase_factor));
        PRINT_FORMAT("%s:\t[1]: Modify decrease factor (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->rprop_decrease_factor));
        PRINT_FORMAT("%s:\t[2]: Modify delta maximum (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->rprop_delta_max));
        PRINT_FORMAT("%s:\t[3]: Modify delta minimum (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->rprop_delta_min));
        PRINT_FORMAT("%s:\t[4]: Modify delta zero (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->rprop_delta_zero));
        PRINT_FORMAT("%s:\t[5]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                5u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Increase factor." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1.2." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->rprop_increase_factor;
            #endif

                this->rprop_increase_factor = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Increase factor: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->rprop_increase_factor) { tmp_parameters_has_change = true; }
            #endif
                break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Decrease factor." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->rprop_decrease_factor;
            #endif

                this->rprop_decrease_factor = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Decrease factor: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->rprop_decrease_factor) { tmp_parameters_has_change = true; }
            #endif
                break;
            case 2u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Delta maximum." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=50.0." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->rprop_delta_max;
            #endif

                this->rprop_delta_max = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Delta maximum: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->rprop_delta_max) { tmp_parameters_has_change = true; }
            #endif
                break;
            case 3u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Delta minimum." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1e-6." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->rprop_delta_min;
            #endif

                this->rprop_delta_min = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Delta minimum: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->rprop_delta_min) { tmp_parameters_has_change = true; }
            #endif
                break;
            case 4u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Delta zero." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.1." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->rprop_delta_zero;
            #endif

                this->rprop_delta_zero = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Delta zero: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->rprop_delta_zero) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 5u:
            #if defined(COMPILE_CUDA)
                if(this->is_device_initialized && tmp_parameters_has_change)
                { this->ptr_device_Neural_Network->Copy__RPROP_minus_Parameters(this); }
            #endif
                    return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         5u,
                                         __LINE__);
                    break;
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__Optimizer__AdaBound(void)
{
#if defined(COMPILE_CUDA)
    bool tmp_parameters_has_change(false);

    T_ tmp_hyper_paramater;
#endif

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, %s:" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
        PRINT_FORMAT("%s:\t[0]: Modify learning rate (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_learning_rate));
        PRINT_FORMAT("%s:\t[1]: Modify learning rate, final (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->learning_rate_final));
        PRINT_FORMAT("%s:\t[2]: Modify beta1 (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_beta1));
        PRINT_FORMAT("%s:\t[3]: Modify beta2 (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_beta2));
        PRINT_FORMAT("%s:\t[4]: Modify epsilon (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_epsilon));
        PRINT_FORMAT("%s:\t[5]: Bias correction (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->use_adam_bias_correction ? "true" : "false");
        PRINT_FORMAT("%s:\t[6]: Modify gamma (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->learning_gamma));
        PRINT_FORMAT("%s:\t[7]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                7u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Learning rate." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.001." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_learning_rate;
            #endif

                this->adam_learning_rate = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning rate: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_learning_rate) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Learning rate, final." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.1." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->learning_rate_final;
            #endif

                this->learning_rate_final = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning rate, final: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->learning_rate_final) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 2u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Beta1." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.9." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_beta1;
            #endif

                this->adam_beta1 = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                      1_T - 1e-7_T,
                                                                                                      MyEA::String::Get__Time() + ": Beta1: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_beta1) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 3u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Beta2." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.999." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_beta2;
            #endif

                this->adam_beta2 = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                      1_T - 1e-7_T,
                                                                                                      MyEA::String::Get__Time() + ": Beta2: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_beta2) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 4u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Epsilon." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1e-8." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_epsilon;
            #endif

                this->adam_epsilon = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Epsilon: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_epsilon) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 5u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Bias correction." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=true." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = static_cast<T_>(this->use_adam_bias_correction);
            #endif

                this->use_adam_bias_correction = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Bias correction: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != static_cast<T_>(this->use_adam_bias_correction)) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 6u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Gamma." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1e-3." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->learning_gamma;
            #endif

                this->learning_gamma = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                            1_T - 1e-7_T,
                                                                                                            MyEA::String::Get__Time() + ": Gamma: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->learning_gamma) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 7u:
            #if defined(COMPILE_CUDA)
                if(this->is_device_initialized && tmp_parameters_has_change)
                { this->ptr_device_Neural_Network->Copy__AdaBound_Parameters(this); }
            #endif
                    return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         7u,
                                         __LINE__);
                    break;
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__Optimizer__Adam(void)
{
#if defined(COMPILE_CUDA)
    bool tmp_parameters_has_change(false);

    T_ tmp_hyper_paramater;
#endif

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, %s:" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
        PRINT_FORMAT("%s:\t[0]: Modify learning rate (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_learning_rate));
        PRINT_FORMAT("%s:\t[1]: Modify beta1 (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_beta1));
        PRINT_FORMAT("%s:\t[2]: Modify beta2 (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_beta2));
        PRINT_FORMAT("%s:\t[3]: Modify epsilon (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_epsilon));
        PRINT_FORMAT("%s:\t[4]: Bias correction (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->use_adam_bias_correction ? "true" : "false");
        PRINT_FORMAT("%s:\t[5]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                5u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Learning rate." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.001." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_learning_rate;
            #endif

                this->adam_learning_rate = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning rate: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_learning_rate) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Beta1." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.9." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_beta1;
            #endif

                this->adam_beta1 = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                      1_T - 1e-7_T,
                                                                                                      MyEA::String::Get__Time() + ": Beta1: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_beta1) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 2u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Beta2." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.999." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_beta2;
            #endif

                this->adam_beta2 = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                      1_T - 1e-7_T,
                                                                                                      MyEA::String::Get__Time() + ": Beta2: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_beta2) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 3u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Epsilon." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1e-8." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_epsilon;
            #endif

                this->adam_epsilon = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Epsilon: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_epsilon) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 4u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Bias correction." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=true." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = static_cast<T_>(this->use_adam_bias_correction);
            #endif

                this->use_adam_bias_correction = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Bias correction: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != static_cast<T_>(this->use_adam_bias_correction)) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 5u:
            #if defined(COMPILE_CUDA)
                if(this->is_device_initialized && tmp_parameters_has_change)
                { this->ptr_device_Neural_Network->Copy__Adam_Parameters(this); }
            #endif
                    return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         5u,
                                         __LINE__);
                    break;
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__Optimizer__NosAdam(void)
{
#if defined(COMPILE_CUDA)
    bool tmp_parameters_has_change(false);

    T_ tmp_hyper_paramater;
#endif

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, %s:" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
        PRINT_FORMAT("%s:\t[0]: Modify learning rate (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_learning_rate));
        PRINT_FORMAT("%s:\t[1]: Modify beta1 (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_beta1));
        PRINT_FORMAT("%s:\t[2]: Modify beta2 (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_beta2));
        PRINT_FORMAT("%s:\t[3]: Modify epsilon (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_epsilon));
        PRINT_FORMAT("%s:\t[4]: Bias correction (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->use_adam_bias_correction ? "true" : "false");
        PRINT_FORMAT("%s:\t[5]: Modify gamma (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->adam_gamma));
        PRINT_FORMAT("%s:\t[6]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                6u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Learning rate." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.001." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_learning_rate;
            #endif

                this->adam_learning_rate = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Learning rate: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_learning_rate) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Beta1." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.9." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_beta1;
            #endif

                this->adam_beta1 = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                      1_T - 1e-7_T,
                                                                                                      MyEA::String::Get__Time() + ": Beta1: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_beta1) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 2u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Beta2." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, 0.99...9]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.999." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_beta2;
            #endif

                this->adam_beta2 = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                      1_T - 1e-7_T,
                                                                                                      MyEA::String::Get__Time() + ": Beta2: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_beta2) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 3u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Epsilon." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1e-8." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_epsilon;
            #endif

                this->adam_epsilon = MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Epsilon: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_epsilon) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 4u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Bias correction." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=true." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = static_cast<T_>(this->use_adam_bias_correction);
            #endif

                this->use_adam_bias_correction = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Bias correction: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != static_cast<T_>(this->use_adam_bias_correction)) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 5u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Gamma." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[1e-7, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.1." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->adam_gamma;
            #endif

                this->adam_gamma = MyEA::String::Cin_Real_Number<T_>(1.0e-7_T, MyEA::String::Get__Time() + ": Gamma: ");

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->adam_gamma) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 6u:
            #if defined(COMPILE_CUDA)
                if(this->is_device_initialized && tmp_parameters_has_change)
                { this->ptr_device_Neural_Network->Copy__Adam_Parameters(this); }
            #endif
                    return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         6u,
                                         __LINE__);
                    break;
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__Warm_Restarts(void)
{
#if defined(COMPILE_CUDA)
    bool tmp_parameters_has_change(false);

    T_ tmp_hyper_paramater;
#endif

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, warm restarts:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Use warm restarts (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->use_Warm_Restarts ? "Yes" : "No");
        PRINT_FORMAT("%s:\t[1]: Modify learning rate, decay (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->warm_restarts_decay_learning_rate));
        PRINT_FORMAT("%s:\t[2]: Modify maximum learning rate (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->warm_restarts_initial_maximum_learning_rate));
        PRINT_FORMAT("%s:\t[3]: Modify minimum learning rate (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->warm_restarts_minimum_learning_rate));
        PRINT_FORMAT("%s:\t[4]: Modify initial Ti (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->warm_restarts_initial_T_i));
        PRINT_FORMAT("%s:\t[5]: Modify warm restarts multiplier (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->warm_restarts_multiplier));
        PRINT_FORMAT("%s:\t[6]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                6u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = static_cast<T_>(this->use_Warm_Restarts);
            #endif

                this->use_Warm_Restarts = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use warm restarts: ");
                
            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != static_cast<T_>(this->use_Warm_Restarts)) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Learning rate, decay:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[1e-5, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.95." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->warm_restarts_decay_learning_rate;
            #endif

                this->warm_restarts_decay_learning_rate = MyEA::String::Cin_Real_Number<T_>(1.0e-5_T,
                                                                                                                                        1_T,
                                                                                                                                        MyEA::String::Get__Time() + ": Learning rate, decay: ");
                
            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->warm_restarts_decay_learning_rate) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 2u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Maximum learning rate:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->warm_restarts_initial_maximum_learning_rate;
            #endif

                this->warm_restarts_initial_maximum_learning_rate = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                             1_T,
                                                                                                                                             MyEA::String::Get__Time() + ": Maximum learning rate: ");
                
            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->warm_restarts_initial_maximum_learning_rate) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 3u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Minimum learning rate:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, %f]." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(this->warm_restarts_initial_maximum_learning_rate));
                PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->warm_restarts_minimum_learning_rate;
            #endif
                
                this->warm_restarts_minimum_learning_rate = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                            this->warm_restarts_initial_maximum_learning_rate,
                                                                                                                                            MyEA::String::Get__Time() + ": Minimum learning rate: ");
                if(this->warm_restarts_minimum_learning_rate == 0_T) { this->warm_restarts_minimum_learning_rate = this->warm_restarts_initial_maximum_learning_rate / 1.0e+7_T; }
                
            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->warm_restarts_minimum_learning_rate) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 4u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Initial Ti:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->warm_restarts_initial_T_i;
            #endif

                this->warm_restarts_initial_T_i = static_cast<T_>(MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Initial Ti: "));
                
            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->warm_restarts_initial_T_i) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 5u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Warm restarts multiplier:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=2." NEW_LINE, MyEA::String::Get__Time().c_str());

            #if defined(COMPILE_CUDA)
                tmp_hyper_paramater = this->warm_restarts_multiplier;
            #endif

                this->warm_restarts_multiplier = static_cast<T_>(MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Warm restarts multiplier: "));

            #if defined(COMPILE_CUDA)
                if(tmp_hyper_paramater != this->warm_restarts_multiplier) { tmp_parameters_has_change = true; }
            #endif
                    break;
            case 6u:
            #if defined(COMPILE_CUDA)
                if(this->is_device_initialized && tmp_parameters_has_change)
                { this->ptr_device_Neural_Network->Copy__Warm_Restarts_Parameters(this); }
            #endif
                    return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         6u,
                                         __LINE__);
                    break;
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__Accuracy_Variance(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Accuracy variance (%f)." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             Cast_T(this->accuracy_variance));
    PRINT_FORMAT("%s:\tRange[0.0, 1.0]." NEW_LINE, MyEA::String::Get__Time().c_str());
    if(this->Set__Accurancy_Variance(MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                              1_T,
                                                                                                              MyEA::String::Get__Time() + ": Accuracy variance: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Accurancy_Variance()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::User_Controls__Time_Delays(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Time delays (%zu)." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             this->number_time_delays);
    PRINT_FORMAT("%s:\tRange[0, %zu]." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             this->number_recurrent_depth - 1_zu);
    if(this->Set__Number_Time_Delays(MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                this->number_recurrent_depth - 1_zu,
                                                                                                MyEA::String::Get__Time() + ": Time delays: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Number_Time_Delays()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::User_Controls__Clip_Gradient(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Clip gradient:" NEW_LINE, MyEA::String::Get__Time().c_str());
    this->Set__Clip_Gradient(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use clip gradient?"));

    if(this->Use__Clip_Gradient())
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Clip gradient:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0 , inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tdefault=1.0." NEW_LINE, MyEA::String::Get__Time().c_str());
        if(this->Set__Clip_Gradient(MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Clip gradient: ")) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Clip_Gradient()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

bool Neural_Network::User_Controls__Max_Norm_Constaints(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Max-norm constraints:" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[0, inf]. Off = 0." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=4.0." NEW_LINE, MyEA::String::Get__Time().c_str());
    if(this->Set__Regularization__Max_Norm_Constraints(MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Max-norm constraints: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Max_Norm_Constraints()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::User_Controls__L1_Regularization(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: L1:" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[0.0, 1.0]. Off = 0." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=0.0." NEW_LINE, MyEA::String::Get__Time().c_str());
    if(this->Set__Regularization__L1(MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                        1_T,
                                                                                                        MyEA::String::Get__Time() + ": L1: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__L1()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::User_Controls__L2_Regularization(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: L2:" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[0.0, 1.0]. Off = 0." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=1e-5." NEW_LINE, MyEA::String::Get__Time().c_str());
    if(this->Set__Regularization__L2(MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                          1_T,
                                                                                                          MyEA::String::Get__Time() + ": L2: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__L2()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::User_Controls__SRIP_Regularization(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: SRIP:" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[0.0, 1.0]. Off = 0." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=1e-5." NEW_LINE, MyEA::String::Get__Time().c_str());
    if(this->Set__Regularization__SRIP(MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                1_T,
                                                                                                                MyEA::String::Get__Time() + ": SRIP: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__SRIP()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

bool Neural_Network::User_Controls__Maximum__Batch_Size(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Maximum batch size:" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[1, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=8192." NEW_LINE, MyEA::String::Get__Time().c_str());
    if(this->Set__Maximum__Batch_Size(MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": Maximum batch size: ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Maximum__Batch_Size()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    return(true);
}

bool Neural_Network::User_Controls__OpenMP(void)
{
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, OpenMP:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Use OpenMP (%s | %s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->use_OpenMP ? "Yes" : "No",
                                 this->is_OpenMP_initialized ? "Yes" : "No");
        PRINT_FORMAT("%s:\t[1]: Maximum threads (%.2f%%)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->percentage_maximum_thread_usage);
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

bool Neural_Network::User_Controls__Regularization(void)
{
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, regularization:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Modify max-norm constraints (%.9f)." NEW_LINE, MyEA::String::Get__Time().c_str(), Cast_T(this->regularization__max_norm_constraints));
        PRINT_FORMAT("%s:\t[1]: Modify L1 (%.9f)." NEW_LINE, MyEA::String::Get__Time().c_str(), Cast_T(this->regularization__l1));
        PRINT_FORMAT("%s:\t[2]: Modify L2 (%.9f)." NEW_LINE, MyEA::String::Get__Time().c_str(), Cast_T(this->regularization__l2));
        PRINT_FORMAT("%s:\t[3]: Modify SRIP (%.9f)." NEW_LINE, MyEA::String::Get__Time().c_str(), Cast_T(this->regularization__srip));
        PRINT_FORMAT("%s:\t[4]: Modify weight decay (%.9f)." NEW_LINE, MyEA::String::Get__Time().c_str(), Cast_T(this->regularization__weight_decay));
        PRINT_FORMAT("%s:\t[5]: Modify dropout." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[6]: Modify normalization." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[7]: Modify tied parameter." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[8]: Modify k-sparse." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[9]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                9u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                if(this->User_Controls__Max_Norm_Constaints() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Max_Norm_Constaints()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                if(this->User_Controls__L1_Regularization() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__L1_Regularization()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 2u:
                if(this->User_Controls__L2_Regularization() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__L2_Regularization()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 3u:
                if(this->User_Controls__SRIP_Regularization() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__SRIP_Regularization()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 4u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Weight decay:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0.0, 1.0]. Off = 0." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1e-5." NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__Regularization__Weight_Decay(MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                        1_T,
                                                                                                                                        MyEA::String::Get__Time() + ": Weight decay: ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Weight_Decay()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 5u:
                if(this->User_Controls__Dropout() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Dropout()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 6u:
                if(this->User_Controls__Normalization() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Normalization()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 7u:
                if(this->User_Controls__Tied__Parameter() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Tied__Parameter()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 8u:
                if(this->User_Controls__K_Sparse() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__K_Sparse()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 9u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         9u,
                                         __LINE__);
                    break;
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__Dropout(void)
{
    long long int tmp_option;

    unsigned int tmp_type_dropout_layer_index;
    
    size_t const tmp_option_end(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
                                               ?
                                               this->total_layers / 2_zu + 1_zu
                                               :
                                               this->total_layers - 1_zu);
    size_t tmp_layer_index;
    
    T_ tmp_hyper_parameters[3u] = {0};
    
    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT tmp_type_layer_dropout;

    struct Layer *tmp_ptr_layer_it;

    std::string tmp_layer_name;

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, Layer dropout:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[-1]: All." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Input layer: (%f, %f, %f), %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->ptr_array_layers[0u].dropout_values[0u]),
                                 Cast_T(this->ptr_array_layers[0u].dropout_values[1u]),
                                 Cast_T(this->ptr_array_layers[0u].dropout_values[2u]),
                                 MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[this->ptr_array_layers[0u].type_dropout].c_str());
        for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_option_end; ++tmp_layer_index)
        {
            PRINT_FORMAT("%s:\t[%zu]: Hidden layer[%zu]: (%f, %f, %f), %s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_layer_index,
                                     tmp_layer_index - 1_zu,
                                     Cast_T(this->ptr_array_layers[tmp_layer_index].dropout_values[0u]),
                                     Cast_T(this->ptr_array_layers[tmp_layer_index].dropout_values[1u]),
                                     Cast_T(this->ptr_array_layers[tmp_layer_index].dropout_values[2u]),
                                     MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[this->ptr_array_layers[tmp_layer_index].type_dropout].c_str());
        }
        PRINT_FORMAT("%s:\t[%zu]: Quit." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_option_end);

        tmp_option = MyEA::String::Cin_Number<long long int>(-1ll,
                                                                                          static_cast<long long int>(tmp_option_end),
                                                                                          MyEA::String::Get__Time() + ": Option: ");

        if(tmp_option < static_cast<long long int>(tmp_option_end))
        {
            tmp_ptr_layer_it = this->ptr_array_layers + tmp_option;
            
            tmp_layer_name = tmp_option == 0ll ? "Input" : "Hidden[" + std::to_string(tmp_option - 1ll) + "]";
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Dropout layer:" NEW_LINE, MyEA::String::Get__Time().c_str());
            for(tmp_type_dropout_layer_index = 0u; tmp_type_dropout_layer_index != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_LENGTH; ++tmp_type_dropout_layer_index)
            {
                PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_type_dropout_layer_index,
                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT>(tmp_type_dropout_layer_index)].c_str());
            }
            PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI].c_str());
            
            switch((tmp_type_layer_dropout = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT>(MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                                                                                                                                             MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_LENGTH - 1u,
                                                                                                                                                                                                                             MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, type: "))))
            {
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE:
                    tmp_hyper_parameters[0u] = 0_T;
                    tmp_hyper_parameters[1u] = 0_T;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Alpha dropout: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA)
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability (" + std::to_string(Cast_T(tmp_ptr_layer_it->dropout_values[0u])) + "): ");
                    }
                    else
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability: ");
                    }

                    tmp_hyper_parameters[1u] = 0_T;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Dropout bernoulli: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI)
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, retention probability (" + std::to_string(Cast_T(tmp_ptr_layer_it->dropout_values[0u])) + "): ");
                    }
                    else
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, retention probability: ");
                    }

                    tmp_hyper_parameters[1u] = 0_T;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Dropout bernoulli inverted: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED)
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, retention probability (" + std::to_string(Cast_T(tmp_ptr_layer_it->dropout_values[0u])) + "): ");
                    }
                    else
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, retention probability: ");
                    }

                    tmp_hyper_parameters[1u] = 0_T;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Dropout gaussian: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN)
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability (" + std::to_string(Cast_T(tmp_ptr_layer_it->dropout_values[0u])) + "): ");
                    }
                    else
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability: ");
                    }

                    tmp_hyper_parameters[1u] = 0_T;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Dropout ShakeDrop: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP)
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability (" + std::to_string(Cast_T(tmp_ptr_layer_it->dropout_values[0u])) + "): ");
                    }
                    else
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability: ");
                    }

                    tmp_hyper_parameters[1u] = 0_T;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Dropout Uout: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT)
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability (" + std::to_string(Cast_T(tmp_ptr_layer_it->dropout_values[0u])) + "): ");
                    }
                    else
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, dropout probability: ");
                    }

                    tmp_hyper_parameters[1u] = 0_T;
                        break;
                case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT:
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Zoneout cell: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.5." NEW_LINE, MyEA::String::Get__Time().c_str());

                    if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT)
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, zoneout cell probability (" + std::to_string(Cast_T(tmp_ptr_layer_it->dropout_values[0u])) + "): ");
                    }
                    else
                    {
                        tmp_hyper_parameters[0u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, zoneout cell probability: ");
                    }
                    
                    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s: Zoneout hidden: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tRange[0.0, 1.0]: " NEW_LINE, MyEA::String::Get__Time().c_str());
                    PRINT_FORMAT("%s:\tdefault=0.05." NEW_LINE, MyEA::String::Get__Time().c_str());

                    if(tmp_ptr_layer_it->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT)
                    {
                        tmp_hyper_parameters[1u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, zoneout hidden probability (" + std::to_string(Cast_T(tmp_ptr_layer_it->dropout_values[1u])) + "): ");
                    }
                    else
                    {
                        tmp_hyper_parameters[1u] = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                           1_T,
                                                                                                                           MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, zoneout hidden probability: ");
                    }
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Type dropout layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_type_layer_dropout,
                                             MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_type_layer_dropout].c_str(),
                                             __LINE__);
                        return(false);
            }
            
            if(tmp_option != -1ll)
            {
                if(this->Set__Dropout(tmp_ptr_layer_it,
                                               tmp_type_layer_dropout,
                                               tmp_hyper_parameters) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(ptr, %u, %f, %f)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_type_layer_dropout,
                                             Cast_T(tmp_hyper_parameters[0u]),
                                             Cast_T(tmp_hyper_parameters[1u]),
                                             __LINE__);

                    return(false);
                }

                if(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
                  &&
                  tmp_type_layer_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE
                  &&
                  tmp_option != 0ll)
                { tmp_ptr_layer_it->use_coded_dropout = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Pre-training: Use dropout inside the coded layer?"); }
            }
            else
            {
                switch((tmp_type_layer_dropout = static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT>(MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                                                                                                                                                 MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_LENGTH - 1u,
                                                                                                                                                                                                                                 MyEA::String::Get__Time() + ": " + tmp_layer_name + " layer, type: "))))
                {
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE:
                        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != this->ptr_last_layer - 1; ++tmp_ptr_layer_it)
                        {
                            if(this->Set__Dropout(tmp_ptr_layer_it,
                                                           tmp_type_layer_dropout,
                                                           tmp_hyper_parameters) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(ptr, %u)\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         tmp_type_layer_dropout,
                                                         __LINE__);

                                return(false);
                            }
                        }
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA:
                        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != this->ptr_last_layer - 1; ++tmp_ptr_layer_it)
                        {
                            if((
                                    tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED
                                    &&
                                    *tmp_ptr_layer_it->ptr_array_AF_units->ptr_type_activation_function == MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SELU
                              )
                              ||
                              (
                                    tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT
                                    &&
                                    *tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units->ptr_type_activation_function == MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SELU
                              ))
                            {
                                if(this->Set__Dropout(tmp_ptr_layer_it,
                                                               tmp_type_layer_dropout,
                                                               tmp_hyper_parameters) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(ptr, %u, %f, %f)\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             tmp_type_layer_dropout,
                                                             Cast_T(tmp_hyper_parameters[0u]),
                                                             Cast_T(tmp_hyper_parameters[1u]),
                                                             __LINE__);

                                    return(false);
                                }
                            }
                        }
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != this->ptr_last_layer - 1; ++tmp_ptr_layer_it)
                        {
                            if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED
                              ||
                              tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
                            {
                                if(this->Set__Dropout(tmp_ptr_layer_it,
                                                               tmp_type_layer_dropout,
                                                               tmp_hyper_parameters) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(ptr, %u, %f, %f)\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             tmp_type_layer_dropout,
                                                             Cast_T(tmp_hyper_parameters[0u]),
                                                             Cast_T(tmp_hyper_parameters[1u]),
                                                             __LINE__);

                                    return(false);
                                }
                            }
                        }
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP:
                        tmp_hyper_parameters[1u] = 0_T;
                        tmp_hyper_parameters[2u] = 0_T;

                        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != this->ptr_last_layer - 1; ++tmp_ptr_layer_it)
                        {
                            if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
                            {
                                ++tmp_hyper_parameters[1u];
                            }
                        }

                        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != this->ptr_last_layer - 1; ++tmp_ptr_layer_it)
                        {
                            if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
                            {
                                if(this->Set__Dropout(tmp_ptr_layer_it,
                                                               tmp_type_layer_dropout,
                                                               std::array<T_, 1_zu>{1_T - ( ((++tmp_hyper_parameters[2u]) / tmp_hyper_parameters[1u]) * (1_T - tmp_hyper_parameters[0u]) )}.data()) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(ptr, %u, %f)\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             tmp_type_layer_dropout,
                                                             Cast_T(1_T - ( (tmp_hyper_parameters[2u] / tmp_hyper_parameters[1u]) * (1_T - tmp_hyper_parameters[0u]) )),
                                                             __LINE__);

                                    return(false);
                                }
                            }
                        }
                            break;
                    case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT:
                        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != this->ptr_last_layer - 1; ++tmp_ptr_layer_it)
                        {
                            if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM)
                            {
                                if(this->Set__Dropout(tmp_ptr_layer_it,
                                                               tmp_type_layer_dropout,
                                                               tmp_hyper_parameters) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(ptr, %u, %f, %f)\" function. At line %d." NEW_LINE,
                                                             MyEA::String::Get__Time().c_str(),
                                                             __FUNCTION__,
                                                             tmp_type_layer_dropout,
                                                             Cast_T(tmp_hyper_parameters[0u]),
                                                             Cast_T(tmp_hyper_parameters[1u]),
                                                             __LINE__);

                                    return(false);
                                }
                            }
                        }
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Type dropout layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 tmp_type_layer_dropout,
                                                 MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[tmp_type_layer_dropout].c_str(),
                                                 __LINE__);
                            return(false);
                }
            }
        }
        else if(tmp_option == static_cast<long long int>(tmp_option_end)) { return(true); }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<long long int>(%lld, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     -1ll,
                                     tmp_option_end,
                                     __LINE__);
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__Normalization(void)
{
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, normalization:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Modify momentum average (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->normalization_momentum_average));
        PRINT_FORMAT("%s:\t[1]: Modify epsilon (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->normalization_epsilon));
        PRINT_FORMAT("%s:\t[2]: Modify r correction maximum (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->batch_renormalization_r_correction_maximum));
        PRINT_FORMAT("%s:\t[3]: Modify d correction maximum (%.9f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->batch_renormalization_d_correction_maximum));
        PRINT_FORMAT("%s:\t[4]: Modify normalization (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->Use__Normalization() ? "Yes" : "No");
        PRINT_FORMAT("%s:\t[5]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                5u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Momentum average:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0.999." NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__Normalization_Momentum_Average(MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                               1_T,
                                                                                                                                               MyEA::String::Get__Time() + ": Momentum average: ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_Momentum_Average()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Epsilon:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1e-5." NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__Normalization_Epsilon(MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Epsilon: ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Normalization_Epsilon()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 2u:
                PRINT_FORMAT("%s: r correction maximum:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__Batch_Renormalization_r_Correction_Maximum(MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": r correction maximum: ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Batch_Renormalization_r_Correction_Maximum()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 3u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: d correction maximum:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__Batch_Renormalization_d_Correction_Maximum(MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": d correction maximum: ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Batch_Renormalization_d_Correction_Maximum()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 4u:
                if(this->User_Controls__Normalization_Layer() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Normalization_Layer()\" function. At line %d." NEW_LINE,
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
                    break;
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__Normalization_Layer(void)
{
#if defined(COMPILE_CUDA)
    bool tmp_parameters_has_change(false);

    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION tmp_type_normalization_parameter;
#endif
    
    unsigned int tmp_type_normalization_layer_index;
    
    size_t const tmp_option_end(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
                                               ?
                                               this->total_layers / 2_zu + 1_zu
                                               :
                                               this->total_layers - 1_zu);
    size_t tmp_option,
              tmp_layer_index;
    
    struct Layer *tmp_ptr_layer_it;

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, Layer normalization:" NEW_LINE, MyEA::String::Get__Time().c_str());
        for(tmp_layer_index = 1_zu; tmp_layer_index != tmp_option_end; ++tmp_layer_index)
        {
            PRINT_FORMAT("%s:\t[%zu]: Hidden layer[%zu]: %s, %s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_layer_index - 1_zu,
                                     tmp_layer_index - 1_zu,
                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[this->ptr_array_layers[tmp_layer_index].type_normalization].c_str(),
                                     this->ptr_array_layers[tmp_layer_index].use_layer_normalization_before_activation ? "true" : "false");
        }
        PRINT_FORMAT("%s:\t[%zu]: Quit." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_option_end - 1_zu);
        
        tmp_option = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                 tmp_option_end - 1_zu,
                                                                                 MyEA::String::Get__Time() + ": Option: ") + 1_zu;

        if(tmp_option < tmp_option_end)
        {
            tmp_ptr_layer_it = this->ptr_array_layers + tmp_option;
            
        #if defined(COMPILE_CUDA)
            tmp_type_normalization_parameter = tmp_ptr_layer_it->type_normalization;
        #endif
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Layer normalization:" NEW_LINE, MyEA::String::Get__Time().c_str());
            for(tmp_type_normalization_layer_index = 0u; tmp_type_normalization_layer_index != MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_LENGTH; ++tmp_type_normalization_layer_index)
            {
                PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_type_normalization_layer_index,
                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION>(tmp_type_normalization_layer_index)].c_str());
            }
            PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION_NAMES[MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION].c_str());
            
            if(this->Set__Layer_Normalization(tmp_ptr_layer_it, static_cast<enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION>(MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                                                                                                                                                                               MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_LENGTH - 1u,
                                                                                                                                                                                                                                                               MyEA::String::Get__Time() + ": Hidden layer " + std::to_string(static_cast<size_t>(tmp_ptr_layer_it - this->ptr_array_layers)) + ", type: "))) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED
              ||
              tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT
              ||
              tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT)
            { tmp_ptr_layer_it->use_layer_normalization_before_activation = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use normalization before activation?"); }

        #if defined(COMPILE_CUDA)
            if(tmp_type_normalization_parameter != tmp_ptr_layer_it->type_normalization) { tmp_parameters_has_change = true; }
        #endif
        }
        else if(tmp_option == tmp_option_end)
        {
        #if defined(COMPILE_CUDA)
            if(this->is_device_initialized && tmp_parameters_has_change)
            { this->ptr_device_Neural_Network->Copy__Normalization(this); }
        #endif

            return(true);
        }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<size_t>(%zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     0_zu,
                                     tmp_option_end - 1_zu,
                                     __LINE__);
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__Tied__Parameter(void)
{
    size_t const tmp_option_end(this->total_layers / 2_zu + 1_zu);
    size_t tmp_option;

    struct Layer *tmp_ptr_layer_it;

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, Tied parameter:" NEW_LINE, MyEA::String::Get__Time().c_str());
        for(size_t tmp_layer_index(1_zu); tmp_layer_index != tmp_option_end; ++tmp_layer_index)
        {
            PRINT_FORMAT("%s:\t[%zu]: Hidden layer[%zu]: %s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_layer_index - 1_zu,
                                     tmp_layer_index - 1_zu,
                                     this->ptr_array_layers[tmp_layer_index].Use__Tied_Parameter() ? "true" : "false");
        }
        PRINT_FORMAT("%s:\t[%zu]: Quit." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_option_end - 1_zu);

        tmp_option = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                 tmp_option_end - 1_zu,
                                                                                 MyEA::String::Get__Time() + ": Option: ") + 1_zu;

        if(tmp_option < tmp_option_end)
        {
            tmp_ptr_layer_it = this->ptr_array_layers + tmp_option;

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Tied parameter:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER ? "true" : "false");
            if(this->Set__Tied_Parameter(tmp_ptr_layer_it, tmp_ptr_layer_it->Use__Tied_Parameter() == false) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Alpha_Sparsity(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                continue;
            }
        }
        else if(tmp_option == tmp_option_end) { return(true); }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<size_t>(%zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     0_zu,
                                     tmp_option_end - 1_zu,
                                     __LINE__);
        }
    }

    return(false);
}

bool Neural_Network::User_Controls__K_Sparse(void)
{
    size_t const tmp_option_end(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
                                               ?
                                               this->total_layers / 2_zu + 1_zu
                                               :
                                               this->total_layers - 1_zu);
    size_t tmp_option,
              tmp_layer_size;

    struct Layer *tmp_ptr_layer_it;

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, k-Sparse:" NEW_LINE, MyEA::String::Get__Time().c_str());
        for(size_t tmp_layer_index(1_zu); tmp_layer_index != tmp_option_end; ++tmp_layer_index)
        {
            PRINT_FORMAT("%s:\t[%zu]: Hidden layer[%zu]: %zu, %f." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_layer_index - 1_zu,
                                     tmp_layer_index - 1_zu,
                                     this->ptr_array_layers[tmp_layer_index].k_sparsity,
                                     Cast_T(this->ptr_array_layers[tmp_layer_index].alpha_sparsity));
        }
        PRINT_FORMAT("%s:\t[%zu]: Quit." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_option_end - 1_zu);

        tmp_option = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                 tmp_option_end - 1_zu,
                                                                                 MyEA::String::Get__Time() + ": Option: ") + 1_zu;

        if(tmp_option < tmp_option_end)
        {
            tmp_ptr_layer_it = this->ptr_array_layers + tmp_option;

            tmp_layer_size = *tmp_ptr_layer_it->ptr_number_outputs;

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: k-Sparse:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[0, %zu]." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_layer_size);
            PRINT_FORMAT("%s:\tdefault=%zu." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     tmp_layer_size / 4_zu == 0_zu ? 1_zu : tmp_layer_size / 4_zu);
            if(this->Set__K_Sparsity(tmp_ptr_layer_it, MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                                               tmp_layer_size,
                                                                                                                               MyEA::String::Get__Time() + ": k-sparse (" + std::to_string(tmp_ptr_layer_it->k_sparsity) + "): ")) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__K_Sparsity(ptr)\" function. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            __LINE__);

                continue;
            }

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Alpha k-sparse:" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[1, inf]." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tdefault=2." NEW_LINE, MyEA::String::Get__Time().c_str());
            if(this->Set__Alpha_Sparsity(tmp_ptr_layer_it, MyEA::String::Cin_Real_Number<T_>(0_T, MyEA::String::Get__Time() + ": Alpha k-sparse (" + MyEA::String::To_string<T_, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(tmp_ptr_layer_it->alpha_sparsity) + "): ")) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Alpha_Sparsity(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                continue;
            }
        }
        else if(tmp_option == tmp_option_end) { return(true); }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<size_t>(%zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     0_zu,
                                     tmp_option_end - 1_zu,
                                     __LINE__);
        }
    }

    return(false);
}

bool Neural_Network::User_Controls(void)
{
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Clear training arrays." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[1]: Reset global loss." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[2]: Weights initializer." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[3]: Optimizer function (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
        PRINT_FORMAT("%s:\t[4]: Loss function (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS_NAMES[this->type_loss_function].c_str());
        PRINT_FORMAT("%s:\t[5]: Accuracy function (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS_NAMES[this->type_accuracy_function].c_str());
        PRINT_FORMAT("%s:\t[6]: Clip gradient (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->Use__Clip_Gradient() ? "true" : "false");
        PRINT_FORMAT("%s:\t[7]: Regularization." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[8]: Modify accuracy variance (%f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->accuracy_variance));
        PRINT_FORMAT("%s:\t[9]: Modify time delays." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[10]: OpenMP." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[11]: Batch size (%zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->maximum_batch_size);
        PRINT_FORMAT("%s:\t[12]: Print information." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[13]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                13u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u: this->Clear_Training_Arrays(); break;
            case 1u: this->Reset__Global_Loss(); break;
            case 2u:
                if(this->User_Controls__Weights_Initializer() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Weights_Initializer()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 3u:
                if(this->User_Controls__Optimizer_Function() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimizer_Function()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 4u:
                if(this->User_Controls__Loss_Function_Initializer() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Loss_Function_Initializer()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 5u:
                if(this->User_Controls__Accuracy_Function_Initializer() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Accuracy_Function_Initializer()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 6u:
                if(this->User_Controls__Clip_Gradient() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Clip_Gradient()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 7u:
                if(this->User_Controls__Regularization() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Regularization()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 8u:
                if(this->User_Controls__Accuracy_Variance() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Accuracy_Variance()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 9u:
                if(this->User_Controls__Time_Delays() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Time_Delays()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 10u:
                if(this->User_Controls__OpenMP() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__OpenMP()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 11u:
                if(this->User_Controls__Maximum__Batch_Size() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Maximum__Batch_Size()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 12u:
                PRINT_FORMAT("%s" NEW_LINE, this->Get__Parameters().c_str());
            #if defined(COMPILE_UI)
                this->plot_gradient = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to plot gradient?");
            #endif
                    break;
            case 13u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         12u,
                                         __LINE__);
                    break;
        }
    }

    return(false);
}
