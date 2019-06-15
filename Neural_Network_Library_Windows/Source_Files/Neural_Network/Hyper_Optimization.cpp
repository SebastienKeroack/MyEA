#include "stdafx.hpp"

#if defined(COMPILE_CUDA)
    #include <CUDA/CUDA_Dataset_Manager.cuh>
#endif

#include <Neural_Network/Dataset_Manager.hpp>

#include <iostream>

template<typename T>
Hyperparameter_Optimization<T>::Hyperparameter_Optimization(void) { }

template<typename T>
void Hyperparameter_Optimization<T>::Reset(void) { this->p_optimization_iterations_since_hyper_optimization = 0_zu; }

template<typename T>
bool Hyperparameter_Optimization<T>::Optimize(class Dataset_Manager<T> *const ptr_Dataset_Manager_received, class Neural_Network *const ptr_Neural_Network_received)
{
    switch(this->_type_hyperparameter_optimization)
    {
        case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_NONE: return(true);
        case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_GAUSSIAN_SEARCH:
            if(this->_ptr_Gaussian_Search->Optimize(this->p_number_hyper_optimization_iterations,
                                                                         ptr_Dataset_Manager_received,
                                                                         ptr_Neural_Network_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Optimize(%zu, ptr, ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         this->p_number_hyper_optimization_iterations,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Hyperparameter optimization type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->_type_hyperparameter_optimization,
                                     ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION_NAMES[this->_type_hyperparameter_optimization].c_str(),
                                     __LINE__);
                return(false);
    }

    this->_evaluation_require = true;
    
    return(true);
}

template<typename T>
bool Hyperparameter_Optimization<T>::Evaluation(void)
{
    switch(this->_type_hyperparameter_optimization)
    {
        case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_NONE: return(true);
        case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_GAUSSIAN_SEARCH:
            if(this->_ptr_Gaussian_Search->Evaluation() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Evaluation()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Hyperparameter optimization type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->_type_hyperparameter_optimization,
                                     ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION_NAMES[this->_type_hyperparameter_optimization].c_str(),
                                     __LINE__);
                return(false);
    }

    this->_evaluation_require = false;
    
    return(true);
}

template<typename T>
bool Hyperparameter_Optimization<T>::Evaluation(class Dataset_Manager<T> *const ptr_Dataset_Manager_received)
{
    switch(this->_type_hyperparameter_optimization)
    {
        case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_NONE: return(true);
        case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_GAUSSIAN_SEARCH:
            if(this->_ptr_Gaussian_Search->Evaluation(ptr_Dataset_Manager_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Evaluation(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Hyperparameter optimization type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->_type_hyperparameter_optimization,
                                     ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION_NAMES[this->_type_hyperparameter_optimization].c_str(),
                                     __LINE__);
                return(false);
    }

    this->_evaluation_require = false;
    
    return(true);
}

template<typename T>
bool Hyperparameter_Optimization<T>::Set__Hyperparameter_Optimization(enum ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION const type_hyper_optimization_received)
{
    // Deallocate.
    switch(this->_type_hyperparameter_optimization)
    {
        case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_NONE: break;
        case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_GAUSSIAN_SEARCH: this->Deallocate__Gaussian_Search(); break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Hyperparameter optimization type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->_type_hyperparameter_optimization,
                                     ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION_NAMES[this->_type_hyperparameter_optimization].c_str(),
                                     __LINE__);
                return(false);
    }
    
    // Allocate.
    switch(type_hyper_optimization_received)
    {
        case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_NONE: break;
        case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_GAUSSIAN_SEARCH:
            if(this->Allocate__Gaussian_Search() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Gaussian_Search()\" function. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Hyperparameter optimization type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     type_hyper_optimization_received,
                                     ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION_NAMES[type_hyper_optimization_received].c_str(),
                                     __LINE__);
                return(false);
    }
    
    this->_type_hyperparameter_optimization = type_hyper_optimization_received;

    return(true);
}

template<typename T>
bool Hyperparameter_Optimization<T>::Set__Number_Hyperparameter_Optimization_Iterations(size_t const number_hyper_optimization_iterations_received)
{
    if(number_hyper_optimization_iterations_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of hyperparameter optimization iterations can not be zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    this->p_number_hyper_optimization_iterations = number_hyper_optimization_iterations_received;

    return(true);
}

template<typename T>
bool Hyperparameter_Optimization<T>::Set__Number_Hyperparameter_Optimization_Iterations_Delay(size_t const number_hyper_optimization_iterations_delay_received)
{
    if(number_hyper_optimization_iterations_delay_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of hyperparameter optimization iterations delay can not be zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    this->p_number_hyper_optimization_iterations_delay = number_hyper_optimization_iterations_delay_received;

    return(true);
}

template<typename T>
bool Hyperparameter_Optimization<T>::Get__Evaluation_Require(void) const { return(this->_evaluation_require); }

template<typename T>
bool Hyperparameter_Optimization<T>::User_Controls__Change__Hyperparameter_Optimization(void)
{
#if defined(COMPILE_UINPUT)
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: User controls, hyperparameter optimization type." NEW_LINE, MyEA::String::Get__Time().c_str());
    for(unsigned int tmp_hyperparameter_optimization_index(0u); tmp_hyperparameter_optimization_index != ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_LENGTH; ++tmp_hyperparameter_optimization_index)
    {
        PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_hyperparameter_optimization_index,
                                 ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION_NAMES[static_cast<enum ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION>(tmp_hyperparameter_optimization_index)].c_str());
    }
    PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION_NAMES[ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_GAUSSIAN_SEARCH].c_str());
    
    if(this->Set__Hyperparameter_Optimization(static_cast<enum ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION>(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                                                                                                                                                       ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_LENGTH - 1u,
                                                                                                                                                                                                                       MyEA::String::Get__Time() + ": Type: "))) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Hyperparameter_Optimization()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    return(true);
}

template<typename T>
bool Hyperparameter_Optimization<T>::User_Controls(void)
{
#if defined(COMPILE_UINPUT)
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Number hyperparameter optimization iteration(s) (%zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->p_number_hyper_optimization_iterations);
        PRINT_FORMAT("%s:\t[1]: Number hyperparameter optimization iteration(s) delay (%zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->p_number_hyper_optimization_iterations_delay);
        PRINT_FORMAT("%s:\t[2]: Change hyperparameter optimization." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[3]: Hyperparameter optimization (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION_NAMES[this->_type_hyperparameter_optimization].c_str());
        PRINT_FORMAT("%s:\t[4]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());
        
        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                4u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Number hyperparameter optimization iteration(s):" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[1, 8]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=10." NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__Number_Hyperparameter_Optimization_Iterations(MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Iteration(s): ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Number_Hyperparameter_Optimization_Iterations()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Number hyperparameter optimization iteration(s) delay:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[1, 8]." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=25." NEW_LINE, MyEA::String::Get__Time().c_str());
                if(this->Set__Number_Hyperparameter_Optimization_Iterations_Delay(MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Iteration(s) delay: ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Number_Hyperparameter_Optimization_Iterations_Delay()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 2u:
                if(this->User_Controls__Change__Hyperparameter_Optimization() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Change__Hyperparameter_Optimization()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 3u:
                switch(this->_type_hyperparameter_optimization)
                {
                    case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_NONE: break;
                    case ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_GAUSSIAN_SEARCH:
                        if(this->_ptr_Gaussian_Search->User_Controls() == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls()\" function. At line %d." NEW_LINE,
                                                        MyEA::String::Get__Time().c_str(),
                                                        __FUNCTION__,
                                                        __LINE__);

                            return(false);
                        }
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Hyperparameter optimization type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 this->_type_hyperparameter_optimization,
                                                 ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION_NAMES[this->_type_hyperparameter_optimization].c_str(),
                                                 __LINE__);
                            return(false);
                }
                    break;
            case 4u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         4u,
                                         __LINE__);
                    return(false);
        }
    }
#endif

    return(true);
}

template<typename T>
bool Hyperparameter_Optimization<T>::Allocate__Gaussian_Search(void)
{
    if(this->_ptr_Gaussian_Search == nullptr)
    {
        if((this->_ptr_Gaussian_Search = new class Gaussian_Search<T>) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     sizeof(class Gaussian_Search<T>),
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
T Hyperparameter_Optimization<T>::Optimization(class Dataset_Manager<T> *const ptr_Dataset_Manager_received, class Neural_Network *const ptr_Neural_Network_received)
{
    if(++this->p_optimization_iterations_since_hyper_optimization >= this->p_number_hyper_optimization_iterations_delay)
    {
        this->p_optimization_iterations_since_hyper_optimization = 0_zu;

        if(this->Optimize(ptr_Dataset_Manager_received, ptr_Neural_Network_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Optimize(ptr, ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return((std::numeric_limits<ST_>::max)());
        }
        
        return(Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
    }
    else
    {
    #if defined(COMPILE_CUDA)
        if(ptr_Neural_Network_received->Use__CUDA())
        { return(ptr_Dataset_Manager_received->Get__CUDA()->Training(ptr_Neural_Network_received)); }
        else
    #endif
        { return(ptr_Dataset_Manager_received->Training(ptr_Neural_Network_received)); }
    }

    return(true);
}

template<typename T>
ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION Hyperparameter_Optimization<T>::Get__Hyperparameter_Optimization(void) const { return(this->_type_hyperparameter_optimization); }

template<typename T>
void Hyperparameter_Optimization<T>::Deallocate__Gaussian_Search(void) { SAFE_DELETE(this->_ptr_Gaussian_Search); }

template<typename T>
bool Hyperparameter_Optimization<T>::Deallocate(void)
{
    this->Deallocate__Gaussian_Search();

    return(true);
}

template<typename T>
Hyperparameter_Optimization<T>::~Hyperparameter_Optimization(void) { this->Deallocate(); }

// template initialization declaration.
template class Hyperparameter_Optimization<T_>;