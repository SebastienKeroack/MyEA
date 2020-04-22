#include "stdafx.hpp"

#include <Neural_Network/Dataset_Manager.hpp>

#include <iostream>

template<typename T>
Dataset_Cross_Validation<T>::Dataset_Cross_Validation(void) : Dataset<T>()
{
    this->p_type_dataset_process = MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION;
}

template<typename T>
void Dataset_Cross_Validation<T>::Shuffle(void)
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
void Dataset_Cross_Validation<T>::Set__Use__Shuffle(bool const use_shuffle_received)
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
void Dataset_Cross_Validation<T>::Reset(void)
{
    this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold;

    this->ptr_array_outputs_array_k_sub_fold = this->ptr_array_outputs_array_k_fold;

    this->number_examples = this->number_examples_validating;
}

template<typename T>
bool Dataset_Cross_Validation<T>::Initialize(void)
{
    this->Dataset<T>::Initialize();

    this->p_type_dataset_process = MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION;

    return(true);
}

template<typename T>
bool Dataset_Cross_Validation<T>::Initialize__Fold(bool const use_shuffle_received,
                                                                           size_t const number_k_fold_received,
                                                                           size_t const number_k_sub_fold_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(number_k_fold_received < 2_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: K-fold must be at least 2. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(this->Set__Desired_K_Fold(number_k_fold_received, number_k_sub_fold_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Desired_Data_Per_Batch(%zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 number_k_fold_received,
                                 number_k_sub_fold_received,
                                 __LINE__);

        return(false);
    }

    if((this->ptr_array_stochastic_index = new size_t[this->p_number_examples]) == nullptr)
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
        for(size_t i(0_zu); i != this->p_number_examples; ++i)
        {
            this->ptr_array_stochastic_index[i] = i;
        }
    }

    this->Generator_Random.Seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    
    return(true);
}

template<typename T>
bool Dataset_Cross_Validation<T>::Set__Desired_K_Fold(size_t const number_k_fold_received, size_t const number_k_sub_fold_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(number_k_fold_received < 2_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: K-fold must be at least 2. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(number_k_fold_received > this->Dataset<T>::Get__Number_Examples())
    {
        PRINT_FORMAT("%s: %s: ERROR: K-fold (%zu) must not be greater than total number of data (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 number_k_fold_received,
                                 this->Dataset<T>::Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    
    this->number_k_fold = number_k_fold_received;
    this->number_examples_per_fold = this->Dataset<T>::Get__Number_Examples() / number_k_fold_received;

    if(this->ptr_array_inputs_array_k_fold == nullptr)
    {
        if((this->ptr_array_inputs_array_k_fold = new T const *[(number_k_fold_received - 1_zu) * this->number_examples_per_fold]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     (number_k_fold_received - 1_zu) * this->number_examples_per_fold * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        this->ptr_array_inputs_array_k_fold = MyEA::Memory::Cpp::Reallocate_PtOfPt<T const *, false>(this->ptr_array_inputs_array_k_fold,
                                                                                                                                         (number_k_fold_received - 1_zu) * this->number_examples_per_fold,
                                                                                                                                         this->number_examples_training);
        if(this->ptr_array_inputs_array_k_fold == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     (number_k_fold_received - 1_zu) * this->number_examples_per_fold * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }
    this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold;
    
    if(this->ptr_array_outputs_array_k_fold == nullptr)
    {
        if((this->ptr_array_outputs_array_k_fold = new T const *[(number_k_fold_received - 1_zu) * this->number_examples_per_fold]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     (number_k_fold_received - 1_zu) * this->number_examples_per_fold * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        this->ptr_array_outputs_array_k_fold = MyEA::Memory::Cpp::Reallocate_PtOfPt<T const *, false>(this->ptr_array_outputs_array_k_fold,
                                                                                                                                           (number_k_fold_received - 1_zu) * this->number_examples_per_fold,
                                                                                                                                           this->number_examples_training);
        if(this->ptr_array_outputs_array_k_fold == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     (number_k_fold_received - 1_zu) * this->number_examples_per_fold * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }
    this->ptr_array_outputs_array_k_sub_fold = this->ptr_array_outputs_array_k_fold;
    
    this->number_examples_training = (number_k_fold_received - 1_zu) * this->number_examples_per_fold;
    
    if(this->ptr_array_inputs_array_validation == nullptr)
    {
        if((this->ptr_array_inputs_array_validation = new T const *[this->Dataset<T>::Get__Number_Examples() - this->number_examples_training]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->Dataset<T>::Get__Number_Examples() - this->number_examples_training * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        this->ptr_array_inputs_array_validation = MyEA::Memory::Cpp::Reallocate_PtOfPt<T const *, false>(this->ptr_array_inputs_array_validation,
                                                                                                                                              this->Dataset<T>::Get__Number_Examples() - this->number_examples_training,
                                                                                                                                              this->number_examples_validating);
        if(this->ptr_array_inputs_array_validation == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->Dataset<T>::Get__Number_Examples() - this->number_examples_training * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }

    if(this->ptr_array_outputs_array_validation == nullptr)
    {
        if((this->ptr_array_outputs_array_validation = new T const *[this->Dataset<T>::Get__Number_Examples() - this->number_examples_training]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->Dataset<T>::Get__Number_Examples() - this->number_examples_training * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        this->ptr_array_outputs_array_validation = MyEA::Memory::Cpp::Reallocate_PtOfPt<T const *, false>(this->ptr_array_outputs_array_validation,
                                                                                                                                                this->Dataset<T>::Get__Number_Examples() - this->number_examples_training,
                                                                                                                                                this->number_examples_validating);
        if(this->ptr_array_outputs_array_validation == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->Dataset<T>::Get__Number_Examples() - this->number_examples_training * sizeof(T const *),
                                     __LINE__);

            return(false);
        }
    }

    this->number_examples = this->number_examples_validating = this->Dataset<T>::Get__Number_Examples() - this->number_examples_training;

    if(number_k_sub_fold_received > this->number_examples_training)
    {
        PRINT_FORMAT("%s: ERROR: K-sub-fold (%zu) > (%zu) amount of data from the training set." NEW_LINE,
                                  __FUNCTION__,
                                  number_k_sub_fold_received,
                                  this->number_examples_training);

        return(false);
    }
    
    this->number_k_sub_fold = number_k_sub_fold_received == 0_zu ? 1_zu : number_k_sub_fold_received;

    // 8 / 2 = 4
    // 31383 / 240 = 130.7625
    double const tmp_number_examples_per_sub_fold(static_cast<double>(this->number_examples_training) / static_cast<double>(this->number_k_sub_fold));

    // 4
    // 130
    this->number_examples_per_sub_iteration = static_cast<size_t>(tmp_number_examples_per_sub_fold);

    // 4 + (4 - 4) * 2 = 0
    // 130 + (130.7625 - 130) * 240 = 183
    this->number_examples_last_sub_iteration = this->number_examples_per_sub_iteration + static_cast<size_t>((tmp_number_examples_per_sub_fold - static_cast<double>(this->number_examples_per_sub_iteration)) * static_cast<double>(this->number_k_sub_fold));
    
    return(true);
}

template<typename T>
bool Dataset_Cross_Validation<T>::Increment_Fold(size_t const fold_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(fold_received >= this->number_k_fold) { return(false); }

    size_t const tmp_number_examples_training_per_fold(this->number_examples_per_fold),
                       tmp_number_examples_validating(this->number_examples_validating),
                       tmp_validating_index_start(fold_received * tmp_number_examples_training_per_fold),
                       *tmp_ptr_array_stochastic_index(this->ptr_array_stochastic_index);
    size_t tmp_example_index;

    if(tmp_validating_index_start == 0_zu) // First iteration.
    {
        // Validation sample.
        // (0, 1, 2)   [3, 4, 5   6, 7, 8   9, 10, 11]
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples_validating; ++tmp_example_index)
        {
            this->ptr_array_inputs_array_validation[tmp_example_index] = this->p_ptr_array_inputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];

            this->ptr_array_outputs_array_validation[tmp_example_index] = this->p_ptr_array_outputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];
        }
        // |END| Validation sample. |END|

        // Training sample.
        tmp_ptr_array_stochastic_index += tmp_number_examples_validating;

        // (0, 1, 2)   [3, 4, 5   6, 7, 8   9, 10, 11]
        for(tmp_example_index = 0_zu; tmp_example_index != this->number_examples_training; ++tmp_example_index)
        {
            this->ptr_array_inputs_array_k_fold[tmp_example_index] = this->p_ptr_array_inputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];

            this->ptr_array_outputs_array_k_fold[tmp_example_index] = this->p_ptr_array_outputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];
        }
        // |END| Training sample. |END|
    }
    else if(tmp_validating_index_start == this->number_examples_training) // Last iteration.
    {
        // Training sample.
        // [0, 1, 2   3, 4, 5   6, 7, 8]   (9, 10, 11)
        for(tmp_example_index = 0_zu; tmp_example_index != this->number_examples_training; ++tmp_example_index)
        {
            this->ptr_array_inputs_array_k_fold[tmp_example_index] = this->p_ptr_array_inputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];

            this->ptr_array_outputs_array_k_fold[tmp_example_index] = this->p_ptr_array_outputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];
        }
        // |END| Training sample. |END|

        // Validation sample.
        tmp_ptr_array_stochastic_index += this->number_examples_training;

        // [0, 1, 2   3, 4, 5   6, 7, 8]   (9, 10, 11)
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples_validating; ++tmp_example_index)
        {
            this->ptr_array_inputs_array_validation[tmp_example_index] = this->p_ptr_array_inputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];

            this->ptr_array_outputs_array_validation[tmp_example_index] = this->p_ptr_array_outputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];
        }
        // |END| Validation sample. |END|
    }
    else // The remaining iterations.
    {
        // Training sample.
        // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_validating_index_start; ++tmp_example_index)
        {
            this->ptr_array_inputs_array_k_fold[tmp_example_index] = this->p_ptr_array_inputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];

            this->ptr_array_outputs_array_k_fold[tmp_example_index] = this->p_ptr_array_outputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];
        }

        // Validation sample.
        tmp_ptr_array_stochastic_index += tmp_validating_index_start;

        // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples_validating; ++tmp_example_index)
        {
            this->ptr_array_inputs_array_validation[tmp_example_index] = this->p_ptr_array_inputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];

            this->ptr_array_outputs_array_validation[tmp_example_index] = this->p_ptr_array_outputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];
        }
        // |END| Validation sample. |END|

        // Training sample.
        tmp_ptr_array_stochastic_index = this->ptr_array_stochastic_index + tmp_number_examples_validating;

        // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
        for(tmp_example_index = tmp_validating_index_start; tmp_example_index != this->number_examples_training; ++tmp_example_index)
        {
            this->ptr_array_inputs_array_k_fold[tmp_example_index] = this->p_ptr_array_inputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];

            this->ptr_array_outputs_array_k_fold[tmp_example_index] = this->p_ptr_array_outputs_array[tmp_ptr_array_stochastic_index[tmp_example_index + this->p_start_index]];
        }
        // |END| Training sample. |END|
    }

    return(true);
}

template<typename T>
bool Dataset_Cross_Validation<T>::Increment_Sub_Fold(size_t const sub_fold_received)
{
    if(this->number_k_sub_fold == 1_zu) { return(true); }
    else if(sub_fold_received >= this->number_k_sub_fold) { return(false); }

    size_t const tmp_data_per_sub_fold(sub_fold_received + 1_zu != this->number_k_sub_fold ? this->number_examples_per_sub_iteration : this->number_examples_last_sub_iteration);
    
    this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold + sub_fold_received * this->number_examples_per_sub_iteration;

    this->ptr_array_outputs_array_k_sub_fold = this->ptr_array_outputs_array_k_fold + sub_fold_received * this->number_examples_per_sub_iteration;

    this->number_examples = tmp_data_per_sub_fold;

    return(true);
}

template<typename T>
bool Dataset_Cross_Validation<T>::Get__Use__Shuffle(void) const { return(this->use_shuffle); }

template<typename T>
bool Dataset_Cross_Validation<T>::Deallocate(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_inputs_array_k_fold);
    SAFE_DELETE_ARRAY(this->ptr_array_outputs_array_k_fold);
    SAFE_DELETE_ARRAY(this->ptr_array_inputs_array_validation);
    SAFE_DELETE_ARRAY(this->ptr_array_outputs_array_validation);
    
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
size_t Dataset_Cross_Validation<T>::Get__Number_Examples(void) const { return(this->number_examples); }

template<typename T>
size_t Dataset_Cross_Validation<T>::Get__Number_Batch(void) const { return(this->number_k_fold); }

template<typename T>
size_t Dataset_Cross_Validation<T>::Get__Number_Sub_Batch(void) const { return(this->number_k_sub_fold); }

template<typename T>
size_t Dataset_Cross_Validation<T>::Get__Number_Examples_Training(void) const { return(this->number_examples_training); }

template<typename T>
size_t Dataset_Cross_Validation<T>::Get__Number_Examples_Validating(void) const { return(this->number_examples_validating); }

template<typename T>
size_t Dataset_Cross_Validation<T>::Get__Number_Examples_Per_Fold(void) const { return(this->number_examples_per_fold); }

template<typename T>
size_t Dataset_Cross_Validation<T>::Get__Number_Examples_Per_Sub_Iteration(void) const { return(this->number_examples_per_sub_iteration); }

template<typename T>
size_t Dataset_Cross_Validation<T>::Get__Number_Examples_Last_Sub_Iteration(void) const { return(this->number_examples_last_sub_iteration); }

template<typename T>
T Dataset_Cross_Validation<T>::Get__Input_At(size_t const index_received, size_t const sub_index_received) const { return(this->ptr_array_inputs_array_k_sub_fold[index_received][sub_index_received]); }

template<typename T>
T Dataset_Cross_Validation<T>::Get__Output_At(size_t const index_received, size_t const sub_index_received) const { return(this->ptr_array_outputs_array_k_sub_fold[index_received][sub_index_received]); }

template<typename T>
T const *const Dataset_Cross_Validation<T>::Get__Input_At(size_t const index_received) const { return(this->ptr_array_inputs_array_k_sub_fold[index_received]); }

template<typename T>
T const *const Dataset_Cross_Validation<T>::Get__Output_At(size_t const index_received) const { return(this->ptr_array_outputs_array_k_sub_fold[index_received]); }

template<typename T>
T const *const *const Dataset_Cross_Validation<T>::Get__Input_Array(void) const { return(this->ptr_array_inputs_array_k_sub_fold); }

template<typename T>
T const *const *const Dataset_Cross_Validation<T>::Get__Output_Array(void) const { return(this->ptr_array_outputs_array_k_sub_fold); }

template<typename T>
T Dataset_Cross_Validation<T>::Training_OpenMP(class Neural_Network *const ptr_Neural_Network_received)
{
    T tmp_summation_loss(0),
       tmp_summation_accurancy(0);

    if(this->use_shuffle) { this->Shuffle(); }

    size_t tmp_fold_index,
              tmp_sub_fold_index;

    for(tmp_fold_index = 0_zu; tmp_fold_index != this->number_k_fold; ++tmp_fold_index)
    {
        if(this->Increment_Fold(tmp_fold_index))
        {
            for(tmp_sub_fold_index = 0_zu; tmp_sub_fold_index != this->number_k_sub_fold; ++tmp_sub_fold_index)
            {
                if(this->Increment_Sub_Fold(tmp_sub_fold_index))
                {
                    this->Train_Epoch_OpenMP(ptr_Neural_Network_received);

                    ptr_Neural_Network_received->Update_Parameter__OpenMP(this->Get__Number_Examples(), this->Dataset<T>::Get__Number_Examples());
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

            tmp_summation_loss += this->Test_Epoch_OpenMP(ptr_Neural_Network_received);
            tmp_summation_accurancy += this->Measure_Accuracy(this->Get__Number_Examples(),
                                                                                               this->Get__Input_Array(),
                                                                                               this->Get__Output_Array(),
                                                                                               ptr_Neural_Network_received);
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

    this->Reset();
    
    ptr_Neural_Network_received->epoch_time_step += 1_T;

    tmp_summation_loss /= static_cast<T>(this->number_k_fold);
    tmp_summation_accurancy /= static_cast<T>(this->number_k_fold);

    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_loss);
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_accurancy);

    return(Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
}

template<typename T>
T Dataset_Cross_Validation<T>::Training_Loop(class Neural_Network *const ptr_Neural_Network_received)
{
    T tmp_summation_loss(0),
       tmp_summation_accurancy(0);

    if(this->use_shuffle) { this->Shuffle(); }
    
    size_t tmp_fold_index,
              tmp_sub_fold_index;

    for(tmp_fold_index = 0_zu; tmp_fold_index != this->number_k_fold; ++tmp_fold_index)
    {
        if(this->Increment_Fold(tmp_fold_index))
        {
            for(tmp_sub_fold_index = 0_zu; tmp_sub_fold_index != this->number_k_sub_fold; ++tmp_sub_fold_index)
            {
                if(this->Increment_Sub_Fold(tmp_sub_fold_index))
                {
                    this->Train_Epoch_Loop(ptr_Neural_Network_received);

                    ptr_Neural_Network_received->Update_Parameter__Loop(this->Get__Number_Examples(), this->Dataset<T>::Get__Number_Examples());
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

            tmp_summation_loss += this->Test_Epoch_Loop(ptr_Neural_Network_received);
            tmp_summation_accurancy += this->Measure_Accuracy(this->Get__Number_Examples(),
                                                                                               this->Get__Input_Array(),
                                                                                               this->Get__Output_Array(),
                                                                                               ptr_Neural_Network_received);
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

    this->Reset();
    
    ptr_Neural_Network_received->epoch_time_step += 1_T;

    tmp_summation_loss /= static_cast<T>(this->number_k_fold);
    tmp_summation_accurancy /= static_cast<T>(this->number_k_fold);

    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_loss);
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_accurancy);

    return(Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
}

template<typename T>
T Dataset_Cross_Validation<T>::Test_Epoch_OpenMP(class Neural_Network *const ptr_Neural_Network_received)
{
    size_t const tmp_number_examples(this->Get__Number_Examples()),
                       tmp_maximum_batch_size(ptr_Neural_Network_received->batch_size),
                       tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_index(0_zu),
              tmp_batch_size(0_zu);
    
    ptr_Neural_Network_received->Reset__Loss();
    
    ptr_Neural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_INFERENCE;
    
    #pragma omp parallel private(tmp_batch_index, tmp_batch_size)
    for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
    {
        tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;
        
        ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);
        
        ptr_Neural_Network_received->Compute__Loss(tmp_batch_size, this->Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);
    }
    
    ptr_Neural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING;
    
    ptr_Neural_Network_received->number_accuracy_trial = tmp_number_examples * (this->Get__Number_Recurrent_Depth() - ptr_Neural_Network_received->number_time_delays) * (ptr_Neural_Network_received->type_accuracy_function == MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY ? 1_zu : ptr_Neural_Network_received->Get__Output_Size());

    ptr_Neural_Network_received->Merge__Post__Training();

    return(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
}

template<typename T>
T Dataset_Cross_Validation<T>::Test_Epoch_Loop(class Neural_Network *const ptr_Neural_Network_received)
{
    size_t const tmp_number_examples(this->Get__Number_Examples()),
                       tmp_maximum_batch_size(ptr_Neural_Network_received->batch_size),
                       tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_size,
              tmp_batch_index;
    
    ptr_Neural_Network_received->Reset__Loss();

    ptr_Neural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_INFERENCE;
    
    for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
    {
        tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;

        ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);
        
        ptr_Neural_Network_received->Compute__Loss(tmp_batch_size, this->Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);
    }
    
    ptr_Neural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING;
    
    ptr_Neural_Network_received->number_accuracy_trial = tmp_number_examples * (this->Get__Number_Recurrent_Depth() - ptr_Neural_Network_received->number_time_delays) * (ptr_Neural_Network_received->type_accuracy_function == MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY ? 1_zu : ptr_Neural_Network_received->Get__Output_Size());

    return(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
}

template<typename T>
Dataset_Cross_Validation<T>::~Dataset_Cross_Validation(void) { this->Deallocate(); }

// template initialization declaration.
template class Dataset_Cross_Validation<T_>;
