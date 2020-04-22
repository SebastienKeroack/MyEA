#include "stdafx.hpp"

#include <Neural_Network/Dataset_Manager.hpp>

#include <iostream>
#include <array>

template<typename T>
Dataset_Cross_Validation_Hyperparameter_Optimization<T>::Dataset_Cross_Validation_Hyperparameter_Optimization(void) : Dataset_Cross_Validation<T>(),
                                                                                                                                                                                      Hyperparameter_Optimization<T>()
{ this->p_type_dataset_process = MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION_RANDOM_SEARCH; }

template<typename T>
T Dataset_Cross_Validation_Hyperparameter_Optimization<T>::Training_OpenMP(class Neural_Network *const ptr_Neural_Network_received)
{
    //if(++this->p_iteration < this->p_number_iterations_per_search) { return(this->Dataset_Cross_Validation<T>::Training_OpenMP(ptr_Neural_Network_received)); }
    //else { this->p_iteration = 0_zu; }

    if(this->use_shuffle) { this->Shuffle(); }

    /*
    size_t tmp_fold_index,
              tmp_sub_fold_index;
    
    class Neural_Network *tmp_ptr_Neural_Network;

    if(this->Update__Size__Population(this->Get__Number_Batch()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Size__Population(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Batch(),
                                 __LINE__);

        return(false);
    }
    else if(this->Update(ptr_Neural_Network_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Update__Thread_Size(this->Get__Number_Examples()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    else if(this->Update__Batch_Size(this->Get__Number_Examples()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    else if(this->Initialize__Hyper_Parameters(ptr_Neural_Network_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__Hyper_Parameters(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Shuffle__Hyper_Parameter() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shuffle__Hyper_Parameter()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Feed__Hyper_Parameter() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Feed__Hyper_Parameter()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    for(tmp_fold_index = 0_zu; tmp_fold_index != this->number_k_fold; ++tmp_fold_index)
    {
        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_Neural_Networks[tmp_fold_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_fold_index,
                                     __LINE__);

            return(false);
        }
    #endif

        if(this->Increment_Fold(tmp_fold_index))
        {
            for(tmp_sub_fold_index = 0_zu; tmp_sub_fold_index != this->number_k_sub_fold; ++tmp_sub_fold_index)
            {
                if(this->Increment_Sub_Fold(tmp_sub_fold_index))
                {
                    this->Train_Epoch_OpenMP(tmp_ptr_Neural_Network);

                    tmp_ptr_Neural_Network->Update_Parameter__OpenMP(this->Get__Number_Examples(), this->Dataset<T>::Get__Number_Examples());
                }
                else
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Increment_Sub_Fold(%zu)\" function. At line %d." NEW_LINE,
                                                MyEA::Time::Date_Time_Now().c_str(),
                                                __FUNCTION__,
                                                tmp_sub_fold_index,
                                                __LINE__);

                    return(false);
                }
            }
            
            this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_validation;
            this->ptr_array_outputs_array_k_sub_fold = this->ptr_array_outputs_array_validation;
            this->number_examples = this->number_examples_validating;

            this->Test_Epoch_OpenMP(tmp_ptr_Neural_Network);
            this->Measure_Accuracy(this->Get__Number_Examples(),
                                                 this->Get__Input_Array(),
                                                 this->Get__Output_Array(),
                                                 tmp_ptr_Neural_Network);
        }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Increment_Fold(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_fold_index,
                                     __LINE__);

            return(false);
        }
    }
    */

    this->Dataset_Cross_Validation<T>::Reset();
    
    if(this->Evaluation() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Evaluation()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    ptr_Neural_Network_received->epoch_time_step += 1_T;

    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));

    return(Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
}

template<typename T>
T Dataset_Cross_Validation_Hyperparameter_Optimization<T>::Training_Loop(class Neural_Network *const ptr_Neural_Network_received)
{
    //if(++this->p_iteration < this->p_number_iterations_per_search) { return(this->Dataset_Cross_Validation<T>::Training_Loop(ptr_Neural_Network_received)); }
    //else { this->p_iteration = 0_zu; }

    if(this->use_shuffle) { this->Shuffle(); }

    /*
    size_t tmp_fold_index,
              tmp_sub_fold_index;

    class Neural_Network *tmp_ptr_Neural_Network;

    if(this->Update__Size__Population(this->Get__Number_Batch()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Size__Population(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Batch(),
                                 __LINE__);

        return(false);
    }
    else if(this->Update(ptr_Neural_Network_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Update__Batch_Size(this->Get__Number_Examples()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    else if(this->Initialize__Hyper_Parameters(ptr_Neural_Network_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__Hyper_Parameters(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Shuffle__Hyper_Parameter() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shuffle__Hyper_Parameter()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Feed__Hyper_Parameter() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Feed__Hyper_Parameter()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    for(tmp_fold_index = 0_zu; tmp_fold_index != this->number_k_fold; ++tmp_fold_index)
    {
        tmp_ptr_Neural_Network = this->p_ptr_array_ptr_Neural_Networks[tmp_fold_index];
        
    #if defined(_DEBUG) || defined(COMPILE_DEBUG)
        if(tmp_ptr_Neural_Network == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Neural network #%zu is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_fold_index,
                                     __LINE__);

            return(false);
        }
    #endif

        if(this->Increment_Fold(tmp_fold_index))
        {
            for(tmp_sub_fold_index = 0_zu; tmp_sub_fold_index != this->number_k_sub_fold; ++tmp_sub_fold_index)
            {
                if(this->Increment_Sub_Fold(tmp_sub_fold_index))
                {
                    this->Train_Epoch_Loop(tmp_ptr_Neural_Network);

                    if(tmp_ptr_Neural_Network->Is_Online_Training() == false) { tmp_ptr_Neural_Network->Update_Parameter__Loop(this->Get__Number_Examples(), this->Dataset<T>::Get__Number_Examples()); }
                }
                else
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Increment_Sub_Fold(%zu)\" function. At line %d." NEW_LINE,
                                                MyEA::Time::Date_Time_Now().c_str(),
                                                __FUNCTION__,
                                                tmp_sub_fold_index,
                                                __LINE__);

                    return(false);
                }
            }
            
            this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_validation;
            this->ptr_array_outputs_array_k_sub_fold = this->ptr_array_outputs_array_validation;
            this->number_examples = this->number_examples_validating;

            this->Test_Epoch_Loop(tmp_ptr_Neural_Network);
            this->Measure_Accuracy(this->Get__Number_Examples(),
                                                 this->Get__Input_Array(),
                                                 this->Get__Output_Array(),
                                                 tmp_ptr_Neural_Network);
        }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Increment_Fold(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_fold_index,
                                     __LINE__);

            return(false);
        }
    }
    */

    this->Dataset_Cross_Validation<T>::Reset();
    
    if(this->Evaluation() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Evaluation()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    ptr_Neural_Network_received->epoch_time_step += 1_T;

    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));

    return(Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
}

template<typename T>
bool Dataset_Cross_Validation_Hyperparameter_Optimization<T>::Deallocate(void)
{
    if(this->Dataset_Cross_Validation<T>::Deallocate() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Dataset_Cross_Validation<T>::Deallocate()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(this->Hyperparameter_Optimization<T>::Deallocate() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Hyperparameter_Optimization<T>::Deallocate()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    return(true);
}

template<typename T>
Dataset_Cross_Validation_Hyperparameter_Optimization<T>::~Dataset_Cross_Validation_Hyperparameter_Optimization(void) { this->Deallocate(); }

// template initialization declaration.
template class Dataset_Cross_Validation_Hyperparameter_Optimization<T_>;
