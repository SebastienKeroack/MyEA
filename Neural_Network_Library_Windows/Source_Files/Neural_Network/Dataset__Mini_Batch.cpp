#include "stdafx.hpp"

#include <Neural_Network/Dataset_Manager.hpp>

#include <iostream>

template<typename T>
Dataset_Mini_Batch<T>::Dataset_Mini_Batch(void) : Dataset<T>()
{
    this->p_type_dataset_process = MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH;
}

template<typename T>
Dataset_Mini_Batch<T>::Dataset_Mini_Batch(bool const use_shuffle_received,
                                                                   size_t const desired_number_examples_per_mini_batch_received,
                                                                   size_t const number_mini_batch_maximum_received,
                                                                   class Dataset<T> &ref_Dataset_received) : Dataset<T>(ref_Dataset_received)
{
    this->Initialize(use_shuffle_received,
                         desired_number_examples_per_mini_batch_received,
                         number_mini_batch_maximum_received);

    this->p_type_dataset_process = MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH;
}

template<typename T>
void Dataset_Mini_Batch<T>::Shuffle(void)
{
    size_t tmp_swap,
              i;
    size_t tmp_randomize_index;
    
    for(i = this->p_number_examples; i-- != this->p_start_index;)
    {
        this->Generator_Random.Range(this->p_start_index, i);

        tmp_randomize_index = this->Generator_Random();

        // Store the index to swap from the remaining index at "tmp_randomize_index"
        tmp_swap = this->ptr_array_stochastic_index[tmp_randomize_index];

        // Get remaining index starting at index "i"
        // And store it to the remaining index at "tmp_randomize_index"
        this->ptr_array_stochastic_index[tmp_randomize_index] = this->ptr_array_stochastic_index[i];

        // Store the swapped index at the index "i"
        this->ptr_array_stochastic_index[i] = tmp_swap;
    }
}

template<typename T>
void Dataset_Mini_Batch<T>::Set__Use__Shuffle(bool const use_shuffle_received)
{
    this->use_shuffle = use_shuffle_received;

    if(use_shuffle_received == false)
    {
        for(size_t tmp_index(0_zu); tmp_index != this->p_number_examples; ++tmp_index)
        {
            this->ptr_array_stochastic_index[tmp_index] = tmp_index;
        }
    }
}

template<typename T>
void Dataset_Mini_Batch<T>::Reset(void)
{
    this->number_examples = this->number_examples_last_iteration;
}

template<typename T>
bool Dataset_Mini_Batch<T>::Initialize(void)
{
    this->Dataset<T>::Initialize();

    this->p_type_dataset_process = MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH;

    return(true);
}

template<typename T>
bool Dataset_Mini_Batch<T>::Initialize(bool const use_shuffle_received,
                                                         size_t const desired_number_examples_per_mini_batch_received,
                                                         size_t const number_mini_batch_maximum_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(desired_number_examples_per_mini_batch_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Desired number data per mini-batch equal zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(desired_number_examples_per_mini_batch_received > this->p_number_examples)
    {
        PRINT_FORMAT("%s: %s: ERROR: Desired number data per mini-batch (%zu) greater than total number of data (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 desired_number_examples_per_mini_batch_received,
                                 this->p_number_examples,
                                 __LINE__);

        return(false);
    }

    if(this->Set__Desired_Data_Per_Batch(desired_number_examples_per_mini_batch_received, number_mini_batch_maximum_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Desired_Data_Per_Batch(%zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 desired_number_examples_per_mini_batch_received,
                                 number_mini_batch_maximum_received,
                                 __LINE__);

        return(false);
    }

    this->ptr_array_stochastic_index = new size_t[this->p_number_examples];
    if(this->ptr_array_stochastic_index == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_examples * sizeof(size_t),
                                 __LINE__);

        return(false);
    }
    
    this->Set__Use__Shuffle(use_shuffle_received);

    if(use_shuffle_received)
    {
        for(size_t i(0); i != this->p_number_examples; ++i)
        {
            this->ptr_array_stochastic_index[i] = i;
        }
    }

    this->Generator_Random.Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));

    return(true);
}

template<typename T>
bool Dataset_Mini_Batch<T>::Set__Desired_Data_Per_Batch(size_t const desired_number_examples_per_mini_batch_received, size_t const number_mini_batch_maximum_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(desired_number_examples_per_mini_batch_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Desired number data per mini-batch equal zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(desired_number_examples_per_mini_batch_received > this->Dataset<T>::Get__Number_Examples())
    {
        PRINT_FORMAT("%s: %s: ERROR: Desired number data per mini-batch (%zu) greater than total number of data (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 desired_number_examples_per_mini_batch_received,
                                 this->Dataset<T>::Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    
    double const tmp_number_mini_batch__real(static_cast<double>(this->Dataset<T>::Get__Number_Examples()) / static_cast<double>(desired_number_examples_per_mini_batch_received));
    size_t tmp_number_mini_batch(static_cast<size_t>(tmp_number_mini_batch__real));
    
    if(number_mini_batch_maximum_received != 0_zu) { tmp_number_mini_batch = tmp_number_mini_batch > number_mini_batch_maximum_received ? number_mini_batch_maximum_received : tmp_number_mini_batch; }
    
    if(tmp_number_mini_batch <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Invalid number of mini-batch. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->number_mini_batch == tmp_number_mini_batch) { return(true); }
    else { this->number_mini_batch = tmp_number_mini_batch; }

    this->number_examples_per_iteration = desired_number_examples_per_mini_batch_received;
    this->number_examples = this->number_examples_per_iteration + static_cast<size_t>((tmp_number_mini_batch__real - static_cast<double>(this->number_mini_batch)) * static_cast<double>(this->number_examples_per_iteration));
    
    if(this->ptr_array_inputs_array_stochastic == nullptr)
    {
        if((this->ptr_array_inputs_array_stochastic = new T const *[this->number_examples]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_examples * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        this->ptr_array_inputs_array_stochastic = MyEA::Memory::Cpp::Reallocate_PtOfPt<T const *, false>(this->ptr_array_inputs_array_stochastic,
                                                                                                                                                this->number_examples,
                                                                                                                                                this->number_examples_last_iteration);
        if(this->ptr_array_inputs_array_stochastic == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_examples * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }
    
    if(this->ptr_array_outputs_array_stochastic == nullptr)
    {
        if((this->ptr_array_outputs_array_stochastic = new T const *[this->number_examples]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_examples * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        this->ptr_array_outputs_array_stochastic = MyEA::Memory::Cpp::Reallocate_PtOfPt<T const *, false>(this->ptr_array_outputs_array_stochastic,
                                                                                                                                                  this->number_examples,
                                                                                                                                                  this->number_examples_last_iteration);
        if(this->ptr_array_outputs_array_stochastic == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->number_examples * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }

    this->number_examples_last_iteration = this->number_examples;
    
    return(true);
}

template<typename T>
bool Dataset_Mini_Batch<T>::Increment_Mini_Batch(size_t const mini_batch_iteration_received)
{
    size_t const tmp_data_per_mini_batch(mini_batch_iteration_received + 1_zu != this->number_mini_batch ? this->number_examples_per_iteration : this->number_examples_last_iteration);
    size_t tmp_last_element_start_index,
              tmp_last_element_end_index,
              tmp_shift_index,
              tmp_index;

    tmp_last_element_start_index = mini_batch_iteration_received * this->number_examples_per_iteration;
    tmp_last_element_end_index = tmp_last_element_start_index + tmp_data_per_mini_batch;

    // Index global inputs to local inputs.
    for(tmp_index = 0_zu,
        tmp_shift_index = tmp_last_element_start_index; tmp_shift_index != tmp_last_element_end_index; ++tmp_shift_index,
                                                                                                                                                        ++tmp_index)
    {
        this->ptr_array_inputs_array_stochastic[tmp_index] = this->p_ptr_array_inputs_array[this->ptr_array_stochastic_index[tmp_shift_index + this->p_start_index]];
        this->ptr_array_outputs_array_stochastic[tmp_index] = this->p_ptr_array_outputs_array[this->ptr_array_stochastic_index[tmp_shift_index + this->p_start_index]];
    }
    // |END| Index global inputs to local inputs. |END|

    this->number_examples = tmp_data_per_mini_batch;

    return(true);
}

template<typename T>
bool Dataset_Mini_Batch<T>::Get__Use__Shuffle(void) const { return(this->use_shuffle); }

template<typename T>
bool Dataset_Mini_Batch<T>::Deallocate(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_inputs_array_stochastic);
    SAFE_DELETE_ARRAY(this->ptr_array_outputs_array_stochastic);

    SAFE_DELETE_ARRAY(this->ptr_array_stochastic_index);

    if(this->Dataset<T>::Deallocate() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Deallocate()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    return(true);
}

template<typename T>
size_t Dataset_Mini_Batch<T>::Get__Number_Examples(void) const { return(this->number_examples); }

template<typename T>
size_t Dataset_Mini_Batch<T>::Get__Number_Batch(void) const { return(this->number_mini_batch); }

template<typename T>
size_t Dataset_Mini_Batch<T>::Get__Number_Examples_Per_Batch(void) const { return(this->number_examples_per_iteration); }

template<typename T>
size_t Dataset_Mini_Batch<T>::Get__Number_Examples_Last_Batch(void) const { return(this->number_examples_last_iteration); }

template<typename T>
T Dataset_Mini_Batch<T>::Training_OpenMP(class Neural_Network *const ptr_Neural_Network_received)
{
    T tmp_summation_loss(0),
       tmp_summation_accurancy(0);

    if(this->use_shuffle) { this->Shuffle(); }

    for(size_t tmp_mini_batch_index(0_zu); tmp_mini_batch_index != this->number_mini_batch; ++tmp_mini_batch_index)
    {
        if(this->Increment_Mini_Batch(tmp_mini_batch_index))
        {
            this->Train_Epoch_OpenMP(ptr_Neural_Network_received);

            tmp_summation_loss += ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);
            tmp_summation_accurancy += this->Measure_Accuracy(this->Get__Number_Examples(),
                                                                                               this->Get__Input_Array(),
                                                                                               this->Get__Output_Array(),
                                                                                               ptr_Neural_Network_received);

            ptr_Neural_Network_received->Update_Parameter__OpenMP(this->Get__Number_Examples(), this->Dataset<T>::Get__Number_Examples());
        }
        else
        {
            PRINT_FORMAT("%s: ERROR: 'Increment_Mini_Batch' Fail." NEW_LINE, __FUNCTION__);

            return(false);
        }
    }

    this->Reset();
    
    ptr_Neural_Network_received->epoch_time_step += 1_T;

    tmp_summation_loss /= static_cast<T>(this->number_mini_batch);
    tmp_summation_accurancy /= static_cast<T>(this->number_mini_batch);

    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_loss);
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_accurancy);

    return(Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
}

template<typename T>
T Dataset_Mini_Batch<T>::Training_Loop(class Neural_Network *const ptr_Neural_Network_received)
{
    T tmp_summation_loss(0),
       tmp_summation_accurancy(0);

    if(this->use_shuffle) { this->Shuffle(); }

    for(size_t tmp_mini_batch_index(0_zu); tmp_mini_batch_index != this->number_mini_batch; ++tmp_mini_batch_index)
    {
        if(this->Increment_Mini_Batch(tmp_mini_batch_index))
        {
            this->Train_Epoch_Loop(ptr_Neural_Network_received);

            tmp_summation_loss += ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);
            tmp_summation_accurancy += this->Measure_Accuracy(this->Get__Number_Examples(),
                                                                                               this->Get__Input_Array(),
                                                                                               this->Get__Output_Array(),
                                                                                               ptr_Neural_Network_received);

            ptr_Neural_Network_received->Update_Parameter__Loop(this->Get__Number_Examples(), this->Dataset<T>::Get__Number_Examples());
        }
        else
        {
            PRINT_FORMAT("%s: ERROR: 'Increment_Mini_Batch' Fail." NEW_LINE, __FUNCTION__);

            return(false);
        }
    }

    this->Reset();

    ptr_Neural_Network_received->epoch_time_step += 1_T;

    tmp_summation_loss /= static_cast<T>(this->number_mini_batch);
    tmp_summation_accurancy /= static_cast<T>(this->number_mini_batch);
    
    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_loss);
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_accurancy);

    return(Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
}

template<typename T>
T Dataset_Mini_Batch<T>::Get__Input_At(size_t const index_received, size_t const sub_index_received) const { return(this->ptr_array_inputs_array_stochastic[index_received][sub_index_received]); }

template<typename T>
T Dataset_Mini_Batch<T>::Get__Output_At(size_t const index_received, size_t const sub_index_received) const { return(this->ptr_array_outputs_array_stochastic[index_received][sub_index_received]); }

template<typename T>
T const *const Dataset_Mini_Batch<T>::Get__Input_At(size_t const index_received) const { return(this->ptr_array_inputs_array_stochastic[index_received]); }

template<typename T>
T const *const Dataset_Mini_Batch<T>::Get__Output_At(size_t const index_received) const { return(this->ptr_array_outputs_array_stochastic[index_received]); }

template<typename T>
T const *const *const Dataset_Mini_Batch<T>::Get__Input_Array(void) const { return(this->ptr_array_inputs_array_stochastic); }

template<typename T>
T const *const *const Dataset_Mini_Batch<T>::Get__Output_Array(void) const { return(this->ptr_array_outputs_array_stochastic); }

template<typename T>
Dataset_Mini_Batch<T>::~Dataset_Mini_Batch(void)
{ this->Deallocate(); }

// template initialization declaration.
template class Dataset_Mini_Batch<T_>;
