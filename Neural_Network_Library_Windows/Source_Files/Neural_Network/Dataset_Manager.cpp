#include "stdafx.hpp"

#include <Tools/Shutdown_Block.hpp>
#include<Files/File.hpp>

#include <Neural_Network/Dataset_Manager.hpp>

#if defined(COMPILE_UI)
    #include <Enums/Enum_Type_Chart.hpp>

    #include <Form.hpp>
#endif // COMPILE_UI

#if defined(COMPILE_UINPUT)
    #include <Tools/Key_Logger.hpp>
#endif // COMPILE_UINPUT

#if defined(COMPILE_CUDA)
    #include <CUDA/CUDA_Dataset_Manager.cuh>
#endif // COMPILE_CUDA

#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>

static bool Experimental_Static_Reset = false; // Experimental.

static size_t Experimental_Static_Minutes = 2_zu; // Experimental.

static std::chrono::system_clock::time_point Experimental_Static_Time_Point = std::chrono::system_clock::now(); // Experimental.

static bool Experimental_Static_Update_BN = false; // Experimental.

template<typename T>
Dataset_Manager<T>::Dataset_Manager(void) : Dataset<T>(),
                                                                     Hyperparameter_Optimization<T>()
{ }

template<typename T>
Dataset_Manager<T>::Dataset_Manager(enum MyEA::Common::ENUM_TYPE_DATASET_FILE const type_data_file_read_received, std::string const &ref_path_received) : Dataset<T>(type_data_file_read_received, ref_path_received),
                                                                                                                                                                                                                                                        Hyperparameter_Optimization<T>()
{ }

template<typename T>
void Dataset_Manager<T>::Testing_On_Storage(class Neural_Network *const ptr_Neural_Network_received)
{
#if defined(COMPILE_COUT)
    #if defined(COMPILE_WINDOWS)
        std::chrono::steady_clock::time_point tmp_time_start,
                                                                tmp_time_end;
    #elif defined(COMPILE_LINUX)
        std::chrono::_V2::system_clock::time_point tmp_time_start,
                                                                        tmp_time_end;
    #endif // COMPILE_WINDOWS || COMPILE_LINUX
#endif // COMPILE_COUT
    
    // Training set.
#if defined(COMPILE_COUT)
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Testing on %zu example(s) from the training set." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Dataset<T>::Get__Number_Examples());

    tmp_time_start = std::chrono::high_resolution_clock::now();
#endif // COMPILE_COUT

#if defined(COMPILE_CUDA)
    if(ptr_Neural_Network_received->Use__CUDA())
    { this->Get__CUDA()->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ptr_Neural_Network_received); }
    else
#endif
    { this->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ptr_Neural_Network_received); }

#if defined(COMPILE_COUT)
    tmp_time_end = std::chrono::high_resolution_clock::now();
    
    PRINT_FORMAT("%s: %.1f example(s) per second." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count() == 0ll ? 0.0 : static_cast<double>(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Dataset<T>::Get__Number_Examples()) / (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count())  / 1e+9));
    PRINT_FORMAT("%s: Loss at training: %.9f" NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
    PRINT_FORMAT("%s: Accuracy at training: %.5f" NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             Cast_T(ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
#endif
    // |END| Training set. |END|
    
    // Validating set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING)
    {
    #if defined(COMPILE_COUT)
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Testing on %zu example(s) from the validation set." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Dataset<T>::Get__Number_Examples());

        tmp_time_start = std::chrono::high_resolution_clock::now();
    #endif

    #if defined(COMPILE_CUDA)
        if(ptr_Neural_Network_received->Use__CUDA())
        { this->Get__CUDA()->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, ptr_Neural_Network_received); }
        else
    #endif
        { this->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, ptr_Neural_Network_received); }

    #if defined(COMPILE_COUT)
        tmp_time_end = std::chrono::high_resolution_clock::now();
    
        PRINT_FORMAT("%s: %.1f example(s) per second." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count() == 0ll ? 0.0 : static_cast<double>(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Dataset<T>::Get__Number_Examples()) / (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count())  / 1e+9));
        PRINT_FORMAT("%s: Loss at validating: %.9f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
        PRINT_FORMAT("%s: Accuracy at validating: %.5f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
    #endif
    }
    else
    {
        ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
        ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, Cast_T(ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
    }
    // |END| Validating set. |END|

    // Testing set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
    {
    #if defined(COMPILE_COUT)
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Testing on %zu example(s) from the testing set." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T>::Get__Number_Examples());

        tmp_time_start = std::chrono::high_resolution_clock::now();
    #endif

    #if defined(COMPILE_CUDA)
        if(ptr_Neural_Network_received->Use__CUDA())
        { this->Get__CUDA()->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, ptr_Neural_Network_received); }
        else
    #endif
        { this->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, ptr_Neural_Network_received); }

    #if defined(COMPILE_COUT)
        tmp_time_end = std::chrono::high_resolution_clock::now();
        
        PRINT_FORMAT("%s: %.1f example(s) per second." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count() == 0ll ? 0.0 : static_cast<double>(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T>::Get__Number_Examples()) / (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count())  / 1e+9));
        PRINT_FORMAT("%s: Loss at testing: %.9f" NEW_LINE,
                                MyEA::String::Get__Time().c_str(),
                                Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
        PRINT_FORMAT("%s: Accuracy at testing: %.5f" NEW_LINE,
                                MyEA::String::Get__Time().c_str(),
                                Cast_T(ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
    #endif
    }
    else
    {
        ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
        ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, Cast_T(ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
    }
    // |END| Testing set. |END|
}

template<typename T>
void Dataset_Manager<T>::Set__Evaluation(enum MyEA::Common::ENUM_TYPE_DATASET const type_evaluation_received) { this->_type_evaluation = type_evaluation_received; }

template<typename T>
void Dataset_Manager<T>::Set__Desired_Optimization_Time_Between_Reports(double const desired_optimization_time_between_reports_received) { this->_desired_optimization_time_between_reports = desired_optimization_time_between_reports_received; }

#if defined(COMPILE_UI)
template<typename T>
void Dataset_Manager<T>::Set__Plot__Loss(bool const use_plot_received) { this->_use_plot_loss = use_plot_received; }

template<typename T>
void Dataset_Manager<T>::Set__Plot__Accuracy(bool const use_plot_received) { this->_use_plot_accuracy = use_plot_received; }

template<typename T>
void Dataset_Manager<T>::Set__Plot__Output(bool const use_plot_received) { this->_use_plot_output = use_plot_received; }

template<typename T>
void Dataset_Manager<T>::Set__Plot__Output__Is_Image(bool const is_plot_image_received) { this->_is_plot_output_image = is_plot_image_received; }

template<typename T>
void Dataset_Manager<T>::Set__Maximum_Ploted_Examples(size_t const number_examples_received) { this->_maximum_ploted_examples = number_examples_received; }

template<typename T>
bool Dataset_Manager<T>::Set__Time_Delay_Ploted(size_t const time_delay_received)
{
    if(time_delay_received >= this->Get__Number_Recurrent_Depth())
    {
        PRINT_FORMAT("%s: %s: ERROR: Time delay (%zu) cannot be equal or greater than recurrent depth (%zu). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 time_delay_received,
                                 this->Get__Number_Recurrent_Depth(),
                                 __LINE__);
        
        return(false);
    }

    this->_time_delay_ploted = time_delay_received;

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Plot__Dataset_Manager(int const input_index_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->_use_plot_output == false) { return(true); }
    else if(type_input_received >= ENUM_TYPE_INPUT::TYPE_INPUT_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    else if(input_index_received <= -1) { return(this->Plot__Dataset_Manager(type_input_received)); }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_example_index,
              tmp_number_examples,
              tmp_time_step_index,
              tmp_shift_index;
    
    if(static_cast<size_t>(input_index_received) >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%d) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }
    
    class Dataset<T> const *tmp_ptr_Dataset(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
    
    T const *const *tmp_ptr_array_inputs_array(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() : tmp_ptr_Dataset->Dataset<T>::Get__Output_Array());

    MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT);
    
    // Training set.
    tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

    for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
    {
        for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                        0u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                        static_cast<double>(tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) + (tmp_time_step_index - this->_time_delay_ploted)),
                                                                                                        tmp_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size + input_index_received]);
        }
    }

    tmp_shift_index = tmp_number_examples * (this->p_number_recurrent_depth - this->_time_delay_ploted);
    // |END| Training set. |END|

    // Validating set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING)
    {
        tmp_ptr_Dataset = this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION);
        
        tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

        tmp_ptr_array_inputs_array = type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() : tmp_ptr_Dataset->Dataset<T>::Get__Output_Array();

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                            0u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                            static_cast<double>(tmp_shift_index + tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) + (tmp_time_step_index - this->_time_delay_ploted)),
                                                                                                            tmp_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size + input_index_received]);
            }
        }

        tmp_shift_index += tmp_number_examples * (this->p_number_recurrent_depth - this->_time_delay_ploted);
    }
    // |END| Validating set. |END|

    // Testing set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
    {
        tmp_ptr_Dataset = this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING);
        
        tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

        tmp_ptr_array_inputs_array = type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() : tmp_ptr_Dataset->Dataset<T>::Get__Output_Array();

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                            0u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                            static_cast<double>(tmp_shift_index + tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) + (tmp_time_step_index - this->_time_delay_ploted)),
                                                                                                            tmp_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size + input_index_received]);
            }
        }
    }
    // |END| Testing set. |END|
    
    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Plot__Dataset_Manager(enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->_use_plot_output == false) { return(true); }

    if(type_input_received >= ENUM_TYPE_INPUT::TYPE_INPUT_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_example_index,
              tmp_number_examples,
              tmp_time_step_index,
              tmp_input_index,
              tmp_shift_index;
    
    class Dataset<T> const *tmp_ptr_Dataset(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
    
    T const *const *tmp_ptr_array_inputs_array(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() : tmp_ptr_Dataset->Dataset<T>::Get__Output_Array());

    MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT);

    // Training set.
    tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

    for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
    {
        for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
            {
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                            0u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                            static_cast<double>(tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_input_size + (tmp_time_step_index - this->_time_delay_ploted) * tmp_input_size + tmp_input_index),
                                                                                                            tmp_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size + tmp_input_index]);
            }
        }
    }

    tmp_shift_index = tmp_number_examples * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_input_size;
    // |END| Training set. |END|

    // Validating set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING)
    {
        tmp_ptr_Dataset = this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION);
        
        tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

        tmp_ptr_array_inputs_array = type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() : tmp_ptr_Dataset->Dataset<T>::Get__Output_Array();

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
                {
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                                0u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                                static_cast<double>(tmp_shift_index + tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_input_size + (tmp_time_step_index - this->_time_delay_ploted) * tmp_input_size + tmp_input_index),
                                                                                                                tmp_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size + tmp_input_index]);
                }
            }
        }

        tmp_shift_index += tmp_number_examples * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_input_size;
    }
    // |END| Validating set. |END|

    // Testing set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
    {
        tmp_ptr_Dataset = this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING);
        
        tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

        tmp_ptr_array_inputs_array = type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() : tmp_ptr_Dataset->Dataset<T>::Get__Output_Array();

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
                {
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                                0u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                                static_cast<double>(tmp_shift_index + tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_input_size + (tmp_time_step_index - this->_time_delay_ploted) * tmp_input_size + tmp_input_index),
                                                                                                                tmp_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size + tmp_input_index]);
                }
            }
        }
    }
    // |END| Testing set. |END|
    
    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Plot__Dataset_Manager(void)
{
    if(this->_use_plot_output == false) { return(true); }

    size_t tmp_example_index,
              tmp_number_examples,
              tmp_time_step_index,
              tmp_output_index,
              tmp_shift_index;

    class Dataset<T> const *tmp_ptr_Dataset(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
    
    MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT);

    // Training set.
    tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

    for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
    {
        for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            for(tmp_output_index = 0_zu; tmp_output_index != this->p_number_outputs; ++tmp_output_index)
            {
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                            0u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                            static_cast<double>(tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * this->p_number_outputs + (tmp_time_step_index - this->_time_delay_ploted) * this->p_number_outputs + tmp_output_index),
                                                                                                            tmp_ptr_Dataset->Dataset<T>::Get__Output_At(tmp_example_index)[tmp_time_step_index * this->p_number_outputs + tmp_output_index]);
            }
        }
    }

    tmp_shift_index = tmp_number_examples * (this->p_number_recurrent_depth - this->_time_delay_ploted) * this->p_number_outputs;
    // |END| Training set. |END|

    // Validating set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING)
    {
        tmp_ptr_Dataset = this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION);
        
        tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_output_index = 0_zu; tmp_output_index != this->p_number_outputs; ++tmp_output_index)
                {
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                                0u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                                static_cast<double>(tmp_shift_index + tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * this->p_number_outputs + (tmp_time_step_index - this->_time_delay_ploted) * this->p_number_outputs + tmp_output_index),
                                                                                                                tmp_ptr_Dataset->Dataset<T>::Get__Output_At(tmp_example_index)[tmp_time_step_index * this->p_number_outputs + tmp_output_index]);
                }
            }
        }

        tmp_shift_index += tmp_number_examples * (this->p_number_recurrent_depth - this->_time_delay_ploted) * this->p_number_outputs;
    }
    // |END| Validating set. |END|

    // Testing set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
    {
        tmp_ptr_Dataset = this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING);
        
        tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_output_index = 0_zu; tmp_output_index != this->p_number_outputs; ++tmp_output_index)
                {
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                                0u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                                static_cast<double>(tmp_shift_index + tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * this->p_number_outputs + (tmp_time_step_index - this->_time_delay_ploted) * this->p_number_outputs + tmp_output_index),
                                                                                                                tmp_ptr_Dataset->Dataset<T>::Get__Output_At(tmp_example_index)[tmp_time_step_index * this->p_number_outputs + tmp_output_index]);
                }
            }
        }
    }
    // |END| Testing set. |END|
    
    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Plot__Dataset_Manager__Pre_Training(class Neural_Network *const ptr_Neural_Network_received)
{
    if(this->_use_plot_output == false) { return(true); }

    if(ptr_Neural_Network_received->pre_training_level == 0_zu) { return(this->Plot__Dataset_Manager()); }

    size_t tmp_example_index,
              tmp_number_examples,
              tmp_time_step_index,
              tmp_output_index,
              tmp_output_size,
              tmp_shift_index;

    class Dataset<T> const *tmp_ptr_Dataset(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
    
    MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT);

    // Training set.
    tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

    tmp_output_size = ptr_Neural_Network_received->Get__Output_Size();

    for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
    {
        ptr_Neural_Network_received->Forward_Pass__Pre_Training(1_zu, tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() + tmp_example_index);

        for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            for(tmp_output_index = 0_zu; tmp_output_index != tmp_output_size; ++tmp_output_index)
            {
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                            0u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                            static_cast<double>(tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_output_size + (tmp_time_step_index - this->_time_delay_ploted) * tmp_output_size + tmp_output_index),
                                                                                                            ptr_Neural_Network_received->Get__Outputs(ptr_Neural_Network_received->ptr_array_layers + (ptr_Neural_Network_received->pre_training_level - 1_zu),
                                                                                                                                                                                0_zu,
                                                                                                                                                                                tmp_time_step_index)[tmp_output_index]);
            }
        }
    }

    tmp_shift_index = tmp_number_examples * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_output_size;
    // |END| Training set. |END|

    // Validating set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING)
    {
        tmp_ptr_Dataset = this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION);
        
        tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

        tmp_output_size = ptr_Neural_Network_received->Get__Output_Size();

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            ptr_Neural_Network_received->Forward_Pass__Pre_Training(1_zu, tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() + tmp_example_index);

            for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_output_index = 0_zu; tmp_output_index != tmp_output_size; ++tmp_output_index)
                {
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                                0u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                                static_cast<double>(tmp_shift_index + tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_output_size + (tmp_time_step_index - this->_time_delay_ploted) * tmp_output_size + tmp_output_index),
                                                                                                                ptr_Neural_Network_received->Get__Outputs(ptr_Neural_Network_received->ptr_array_layers + (ptr_Neural_Network_received->pre_training_level - 1_zu),
                                                                                                                                                                                    0_zu,
                                                                                                                                                                                    tmp_time_step_index)[tmp_output_index]);
                }
            }
        }

        tmp_shift_index += tmp_number_examples * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_output_size;
    }
    // |END| Validating set. |END|

    // Testing set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
    {
        tmp_ptr_Dataset = this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING);
        
        tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

        tmp_output_size = ptr_Neural_Network_received->Get__Output_Size();

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            ptr_Neural_Network_received->Forward_Pass__Pre_Training(1_zu, tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() + tmp_example_index);

            for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_output_index = 0_zu; tmp_output_index != tmp_output_size; ++tmp_output_index)
                {
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                                0u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                                static_cast<double>(tmp_shift_index + tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_output_size + (tmp_time_step_index - this->_time_delay_ploted) * tmp_output_size + tmp_output_index),
                                                                                                                ptr_Neural_Network_received->Get__Outputs(ptr_Neural_Network_received->ptr_array_layers + (ptr_Neural_Network_received->pre_training_level - 1_zu),
                                                                                                                                                                                    0_zu,
                                                                                                                                                                                    tmp_time_step_index)[tmp_output_index]);
                }
            }
        }
    }
    // |END| Testing set. |END|
    
    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Plot__Neural_Network(class Neural_Network *const ptr_Neural_Network_received)
{
    if(this->_use_plot_output == false) { return(true); }

    size_t const tmp_output_size(ptr_Neural_Network_received->Get__Output_Size());
    size_t tmp_example_index,
              tmp_number_examples,
              tmp_time_step_index,
              tmp_output_index,
              tmp_shift_index;

    class Dataset<T> const *tmp_ptr_Dataset(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
    
    // Training set.
    tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

    for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
    {
        ptr_Neural_Network_received->Forward_Pass(1_zu, tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() + tmp_example_index);
        
        for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            for(tmp_output_index = 0_zu; tmp_output_index != tmp_output_size; ++tmp_output_index)
            {
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                            1u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                            static_cast<double>(tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_output_size + (tmp_time_step_index - this->_time_delay_ploted) * tmp_output_size + tmp_output_index),
                                                                                                            ptr_Neural_Network_received->Get__Outputs(0_zu, tmp_time_step_index)[tmp_output_index]);
            }
        }
    }

    tmp_shift_index = tmp_number_examples * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_output_size;
    // |END| Training set. |END|

    // Validating set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING)
    {
        tmp_ptr_Dataset = this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION);
        
        tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            ptr_Neural_Network_received->Forward_Pass(1_zu, tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() + tmp_example_index);
            
            for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_output_index = 0_zu; tmp_output_index != tmp_output_size; ++tmp_output_index)
                {
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                                1u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                                static_cast<double>(tmp_shift_index + tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_output_size + (tmp_time_step_index - this->_time_delay_ploted) * tmp_output_size + tmp_output_index),
                                                                                                                ptr_Neural_Network_received->Get__Outputs(0_zu, tmp_time_step_index)[tmp_output_index]);
                }
            }
        }

        tmp_shift_index += tmp_number_examples * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_output_size;
    }
    // |END| Validating set. |END|

    // Testing set.
    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
    {
        tmp_ptr_Dataset = this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING);
        
        tmp_number_examples = this->_maximum_ploted_examples == 0_zu ? tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples() : MyEA::Math::Minimum<size_t>(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples(), this->_maximum_ploted_examples);

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            ptr_Neural_Network_received->Forward_Pass(1_zu, tmp_ptr_Dataset->Dataset<T>::Get__Input_Array() + tmp_example_index);
            
            for(tmp_time_step_index = this->_time_delay_ploted; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_output_index = 0_zu; tmp_output_index != tmp_output_size; ++tmp_output_index)
                {
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT,
                                                                                                                1u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                                static_cast<double>(tmp_shift_index + tmp_example_index * (this->p_number_recurrent_depth - this->_time_delay_ploted) * tmp_output_size + (tmp_time_step_index - this->_time_delay_ploted) * tmp_output_size + tmp_output_index),
                                                                                                                ptr_Neural_Network_received->Get__Outputs(0_zu, tmp_time_step_index)[tmp_output_index]);
                }
            }
        }
    }
    // |END| Testing set. |END|
    
    MyEA::Form::API__Form__Neural_Network__Chart_Rescale(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT);

    return(true);
}

template<typename T>
size_t Dataset_Manager<T>::Get__Maximum_Ploted_Examples(void) const { return(this->_maximum_ploted_examples); }

template<typename T>
size_t Dataset_Manager<T>::Get__Time_Delay_Ploted(void) const { return(this->_time_delay_ploted); }

template<typename T>
bool Dataset_Manager<T>::Get__Plot__Loss(void) const { return(this->_use_plot_loss); }

template<typename T>
bool Dataset_Manager<T>::Get__Plot__Accuracy(void) const { return(this->_use_plot_accuracy); }

template<typename T>
bool Dataset_Manager<T>::Get__Plot__Output(void) const { return(this->_use_plot_output); }

template<typename T>
bool Dataset_Manager<T>::User_Controls__Set__Maximum_Ploted_Example(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Maximum ploted example(s)." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tRange[1, %zu]." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             this->Get__Number_Examples());
    
    this->Set__Maximum_Ploted_Examples(MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                                    this->Get__Number_Examples(),
                                                                                                                    MyEA::String::Get__Time() + ": Maximum example(s): "));
    
    return(true);
}

template<typename T>
bool Dataset_Manager<T>::User_Controls__Set__Time_Delay_Ploted(void)
{
    if(this->Get__Number_Recurrent_Depth() > 1_zu)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Time delay ploted." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\tRange[0, %zu]." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->Get__Number_Recurrent_Depth() - 1_zu);
        
        if(this->Set__Time_Delay_Ploted(MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                             this->Get__Number_Recurrent_Depth() - 1_zu,
                                                                                                             MyEA::String::Get__Time() + ": Time delay: ")) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Time_Delay_Ploted()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);
        
            return(false);
        }
    }
    
    return(true);
}
#endif

template<typename T>
bool Dataset_Manager<T>::User_Controls(void)
{
#if defined(COMPILE_UI) && defined(COMPILE_UINPUT)
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Optimization, processing parameters." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[1]: Change evaluation type (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_DATASET_NAMES[this->_type_evaluation].c_str());
        PRINT_FORMAT("%s:\t[2]: Change metric comparison (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_use_metric_loss ? "Loss" : "Accuracy");
        PRINT_FORMAT("%s:\t[3]: Maximum example(s) (%zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_maximum_examples);
        PRINT_FORMAT("%s:\t[4]: Desired optimization time between reports (%f seconds)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_desired_optimization_time_between_reports);
        PRINT_FORMAT("%s:\t[5]: Minimum dataset out loss accepted (%f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_minimum_loss_holdout_accepted);
        PRINT_FORMAT("%s:\t[6]: Dataset in equal or less dataset out accepted (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_dataset_in_equal_less_holdout_accepted ? "true" : "false");
        PRINT_FORMAT("%s:\t[7]: Hyperparameter optimization." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[8]: Use chart datapoint training." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[9]: Use dataset plot (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_use_plot_output ? "true" : "false");
        PRINT_FORMAT("%s:\t[10]: Time delay ploted (%zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_time_delay_ploted);
        PRINT_FORMAT("%s:\t[11]: Maximum ploted example(s) (%zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_maximum_ploted_examples);
        PRINT_FORMAT("%s:\t[12]: Change chart total means." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[13]: Plot dataset." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[14]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                14u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                if(this->User_Controls__Optimization_Processing_Parameters() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimization_Processing_Parameters()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                if(this->User_Controls__Type_Evaluation() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Type_Evaluation()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }

                if((Experimental_Static_Reset = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Experimental reset (" + (Experimental_Static_Reset ? "true" : "false") + "): ")))
                {
                    Experimental_Static_Minutes = MyEA::String::Cin_Number<size_t>(1_zu, MyEA::String::Get__Time() + ": Experimental minutes (" + std::to_string(Experimental_Static_Minutes) + "): "); // Experimental.

                    Experimental_Static_Time_Point = std::chrono::system_clock::now(); // Experimental.
                }

                Experimental_Static_Update_BN = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Experimental update BN (" + (Experimental_Static_Update_BN ? "true" : "false") + "): "); // Experimental.
                    break;
            case 2u:
                if(this->User_Controls__Type_Metric() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Type_Metric()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 3u:
                if(this->User_Controls__Set__Maximum_Data() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Set__Maximum_Data()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 4u: this->_desired_optimization_time_between_reports = MyEA::String::Cin_Real_Number<double>(0.0, MyEA::String::Get__Time() + ": Desired optimization time between reports (seconds): "); break;
            case 5u: this->_minimum_loss_holdout_accepted = MyEA::String::Cin_Real_Number<T>(T(0), MyEA::String::Get__Time() + ": Minimum dataset out loss accepted: "); break;
            case 6u: this->_dataset_in_equal_less_holdout_accepted = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Dataset in equal or less dataset out accepted?"); break;
            case 7u:
                if(this->Hyperparameter_Optimization<T>::User_Controls() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Hyperparameter_Optimization<T>::User_Controls()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 8u: MyEA::Form::API__Form__Neural_Network__Chart_Use_Datapoint_Training(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use chart datapoint training?")); break;
            case 9u: this->Set__Plot__Output(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to plot the dataset?")); break;
           case 10u:
                if(this->User_Controls__Set__Time_Delay_Ploted() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Set__Time_Delay_Ploted()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
             case 11u:
                if(this->User_Controls__Set__Maximum_Ploted_Example() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Set__Maximum_Ploted_Example()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 12u:
                MyEA::Form::API__Form__Neural_Network__Chart_Total_Means(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                             100'000u,
                                                                                                                                                                             MyEA::String::Get__Time() + ": Chart total means: "));
                    break;
            case 13u:
                if(this->Plot__Dataset_Manager() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Dataset_Manager()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 14u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         14u,
                                         __LINE__);
                    break;
        }
    }
#elif defined(COMPILE_UINPUT)
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Optimization, processing parameters." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[1]: Change evaluation type (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_DATASET_NAMES[this->_type_evaluation].c_str());
        PRINT_FORMAT("%s:\t[2]: Change metric comparison (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_use_metric_loss ? "Loss" : "Accuracy");
        PRINT_FORMAT("%s:\t[3]: Maximum example(s) (%zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_maximum_examples);
        PRINT_FORMAT("%s:\t[4]: Desired optimization time between reports (%f seconds)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_desired_optimization_time_between_reports);
        PRINT_FORMAT("%s:\t[5]: Minimum dataset out loss accepted (%f)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(this->_minimum_loss_holdout_accepted));
        PRINT_FORMAT("%s:\t[6]: Dataset in equal or less dataset out accepted (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 this->_dataset_in_equal_less_holdout_accepted ? "true" : "false");
        PRINT_FORMAT("%s:\t[7]: Hyperparameter optimization." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[8]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                8u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                if(this->User_Controls__Optimization_Processing_Parameters() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimization_Processing_Parameters()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                if(this->User_Controls__Type_Evaluation() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Type_Evaluation()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 2u:
                if(this->User_Controls__Type_Metric() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Type_Metric()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 3u:
                if(this->User_Controls__Set__Maximum_Data() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Set__Maximum_Data()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 4u: this->Set__Desired_Optimization_Time_Between_Reports(MyEA::String::Cin_Real_Number<double>(0.0, MyEA::String::Get__Time() + ": Desired optimization time between reports (seconds): ")); break;
            case 5u: this->_minimum_loss_holdout_accepted = MyEA::String::Cin_Real_Number<T>(T(0), MyEA::String::Get__Time() + ": Minimum dataset out loss accepted: "); break;
            case 6u: this->_dataset_in_equal_less_holdout_accepted = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Dataset in equal or less dataset out accepted?"); break;
            case 7u:
                if(this->Hyperparameter_Optimization<T>::User_Controls() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Hyperparameter_Optimization<T>::User_Controls()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 8u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         8u,
                                         __LINE__);
                    break;
        }
    }
#endif // COMPILE_UI || COMPILE_UINPUT

    return(false);
}

template<typename T>
bool Dataset_Manager<T>::User_Controls__Set__Maximum_Data(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Maximum example(s)." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=%zu || 0." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             this->p_number_examples);
    
    if(this->Set__Maximum_Data(MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                    this->p_number_examples,
                                                                                                    MyEA::String::Get__Time() + ": Maximum example(s): ")) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Maximum_Data()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    return(true);
}

template<typename T>
bool Dataset_Manager<T>::User_Controls__Type_Evaluation(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Type evaluation." NEW_LINE, MyEA::String::Get__Time().c_str());
    for(unsigned int tmp_type_dataset_index(1u); tmp_type_dataset_index != MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_LENGTH; ++tmp_type_dataset_index)
    {
        PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_type_dataset_index,
                                 MyEA::Common::ENUM_TYPE_DATASET_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(tmp_type_dataset_index)].c_str());
    }
    PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             MyEA::Common::ENUM_TYPE_DATASET_NAMES[MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING].c_str());
    
    this->Set__Evaluation(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                              MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_LENGTH - 1u,
                                                                                                                                                                                              MyEA::String::Get__Time() + ": Type: ")));
    
    return(true);
}

template<typename T>
bool Dataset_Manager<T>::User_Controls__Type_Metric(void)
{
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Type metric." NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s:\tdefault=Loss." NEW_LINE, MyEA::String::Get__Time().c_str());
    
    this->_use_metric_loss = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": (Yes=Loss, No=Accuracy): ");
    
    return(true);
}

template<typename T>
bool Dataset_Manager<T>::User_Controls__Optimization_Processing_Parameters(void)
{
#if defined(COMPILE_UINPUT)
    switch(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Type_Dataset_Process())
    {
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH:
            if(this->User_Controls__Optimization_Processing_Parameters__Batch() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimization_Processing_Parameters__Batch()\" function. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH:
            if(this->User_Controls__Optimization_Processing_Parameters__Mini_Batch() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimization_Processing_Parameters__Mini_Batch()\" function. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION:
            if(this->User_Controls__Optimization_Processing_Parameters__Cross_Validation() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimization_Processing_Parameters__Cross_Validation()\" function. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION_RANDOM_SEARCH:
            if(this->User_Controls__Optimization_Processing_Parameters__Cross_Validation__Gaussian_Search() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimization_Processing_Parameters__Cross_Validation__Gaussian_Search()\" function. At line %d." NEW_LINE,
                                            MyEA::String::Get__Time().c_str(),
                                            __FUNCTION__,
                                            __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset process type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Type_Dataset_Process(),
                                     MyEA::Common::ENUM_TYPE_DATASET_PROCESS_NAMES[this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Type_Dataset_Process()].c_str(),
                                     __LINE__);
                return(false);
    }
#endif

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::User_Controls__Optimization_Processing_Parameters__Batch(void)
{
#if defined(COMPILE_UINPUT)
    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, %s:" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_DATASET_PROCESS_NAMES[MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH].c_str());
        PRINT_FORMAT("%s:\t[0]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                0u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         0u,
                                         __LINE__);
                    return(false);
        }
    }
#endif

    return(false);
}

template<typename T>
bool Dataset_Manager<T>::User_Controls__Optimization_Processing_Parameters__Mini_Batch(void)
{
#if defined(COMPILE_UINPUT)
    class Dataset_Mini_Batch<T> *tmp_ptr_Dataset_Mini_Batch(dynamic_cast<class Dataset_Mini_Batch<T> *>(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, %s:" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_DATASET_PROCESS_NAMES[MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH].c_str());
        PRINT_FORMAT("%s:\t[0]: Modify number of mini-batch (%zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_Dataset_Mini_Batch->Get__Number_Batch());
        PRINT_FORMAT("%s:\t[1]: Use shuffle (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_Dataset_Mini_Batch->Get__Use__Shuffle() ? "true" : "false");
        PRINT_FORMAT("%s:\t[2]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());
        
        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                2u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                PRINT_FORMAT("%s: Desired-examples per batch:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[1, %zu]." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_ptr_Dataset_Mini_Batch->Dataset<T>::Get__Number_Examples());
                if(tmp_ptr_Dataset_Mini_Batch->Set__Desired_Data_Per_Batch(MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                                                                                   tmp_ptr_Dataset_Mini_Batch->Dataset<T>::Get__Number_Examples(),
                                                                                                                                                                   MyEA::String::Get__Time() + ": Desired-examples per batch: ")) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Desired_Data_Per_Batch()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                PRINT_FORMAT("%s: Shuffle:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=Yes." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_ptr_Dataset_Mini_Batch->Set__Use__Shuffle(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use shuffle: "));
                    break;
            case 2u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         2u,
                                         __LINE__);
                    return(false);
        }
    }
#endif

    return(false);
}

template<typename T>
bool Dataset_Manager<T>::User_Controls__Optimization_Processing_Parameters__Cross_Validation(void)
{
#if defined(COMPILE_UINPUT)
    class Dataset_Cross_Validation<T> *tmp_ptr_Dataset_Cross_Validation(dynamic_cast<class Dataset_Cross_Validation<T> *>(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));

    size_t tmp_number_k_folds,
              tmp_number_k_sub_folds,
              tmp_number_examples_training;

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, %s:" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_DATASET_PROCESS_NAMES[MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION].c_str());
        PRINT_FORMAT("%s:\t[0]: Modify number of K-Fold (%zu, %zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_Dataset_Cross_Validation->Get__Number_Batch(),
                                 tmp_ptr_Dataset_Cross_Validation->Get__Number_Sub_Batch());
        PRINT_FORMAT("%s:\t[1]: Use shuffle (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_Dataset_Cross_Validation->Get__Use__Shuffle() ? "true" : "false");
        PRINT_FORMAT("%s:\t[2]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());
        
        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                2u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                // K-fold.
                PRINT_FORMAT("%s: K-fold:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[2, %zu]." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_ptr_Dataset_Cross_Validation->Dataset<T>::Get__Number_Examples());
                tmp_number_k_folds = MyEA::String::Cin_Number<size_t>(2_zu,
                                                                                                       tmp_ptr_Dataset_Cross_Validation->Dataset<T>::Get__Number_Examples(),
                                                                                                       MyEA::String::Get__Time() + ": K-fold: ");
                // |END| K-fold. |END|
                
                // K-sub-fold.
                tmp_number_examples_training = (tmp_number_k_folds - 1_zu) * (tmp_ptr_Dataset_Cross_Validation->Dataset<T>::Get__Number_Examples() / tmp_number_k_folds);

                PRINT_FORMAT("%s: K-sub-fold:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0, %zu]." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_number_examples_training);
                PRINT_FORMAT("%s:\tdefault=%zu." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_number_k_folds - 1_zu);
                tmp_number_k_sub_folds = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                              tmp_number_examples_training,
                                                                                                              MyEA::String::Get__Time() + ": K-sub-fold: ");
                // |END| K-sub-fold. |END|

                if(tmp_ptr_Dataset_Cross_Validation->Set__Desired_K_Fold(tmp_number_k_folds, tmp_number_k_sub_folds) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Desired_Data_Per_Batch(%zu, %zu)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_number_k_folds,
                                             tmp_number_k_sub_folds,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                PRINT_FORMAT("%s: Shuffle:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=Yes." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_ptr_Dataset_Cross_Validation->Set__Use__Shuffle(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use shuffle: "));
                    break;
            case 2u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         2u,
                                         __LINE__);
                    return(false);
        }
    }
#endif

    return(false);
}

template<typename T>
bool Dataset_Manager<T>::User_Controls__Optimization_Processing_Parameters__Cross_Validation__Gaussian_Search(void)
{
#if defined(COMPILE_UINPUT)
    class Dataset_Cross_Validation_Hyperparameter_Optimization<T> *tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization(dynamic_cast<class Dataset_Cross_Validation_Hyperparameter_Optimization<T> *>(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));

    size_t tmp_number_k_folds,
              tmp_number_k_sub_folds,
              tmp_number_examples_training;

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, %s:" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 MyEA::Common::ENUM_TYPE_DATASET_PROCESS_NAMES[MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION].c_str());
        PRINT_FORMAT("%s:\t[0]: Modify number of K-Fold (%zu, %zu)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->Get__Number_Batch(),
                                 tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->Get__Number_Sub_Batch());
        PRINT_FORMAT("%s:\t[1]: Use shuffle (%s)." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->Get__Use__Shuffle() ? "true" : "false");
        PRINT_FORMAT("%s:\t[2]: Hyperparameter optimization." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[3]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());
        
        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                3u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                // K-fold.
                PRINT_FORMAT("%s: K-fold:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[2, %zu]." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->Dataset<T>::Get__Number_Examples());
                tmp_number_k_folds = MyEA::String::Cin_Number<size_t>(2_zu,
                                                                                                       tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->Dataset<T>::Get__Number_Examples(),
                                                                                                       MyEA::String::Get__Time() + ": K-fold: ");
                // |END| K-fold. |END|
                
                // K-sub-fold.
                tmp_number_examples_training = (tmp_number_k_folds - 1_zu) * (tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->Dataset<T>::Get__Number_Examples() / tmp_number_k_folds);

                PRINT_FORMAT("%s: K-sub-fold:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tRange[0, %zu]." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_number_examples_training);
                PRINT_FORMAT("%s:\tdefault=%zu." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_number_k_folds - 1_zu);
                tmp_number_k_sub_folds = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                              tmp_number_examples_training,
                                                                                                              MyEA::String::Get__Time() + ": K-sub-fold: ");
                // |END| K-sub-fold. |END|

                if(tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->Set__Desired_K_Fold(tmp_number_k_folds, tmp_number_k_sub_folds) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Desired_Data_Per_Batch(%zu, %zu)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             tmp_number_k_folds,
                                             tmp_number_k_sub_folds,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                PRINT_FORMAT("%s: Shuffle:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s:\tdefault=Yes." NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->Set__Use__Shuffle(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use shuffle: "));
                    break;
            case 2u:
                if(tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->User_Controls() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Push_Back()\" function. At line %d." NEW_LINE,
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
                    return(false);
        }
    }
#endif

    return(false);
}

template<typename T>
bool Dataset_Manager<T>::User_Controls__Optimization(class Neural_Network *&ptr_trainer_Neural_Network_received, class Neural_Network *&ptr_trained_Neural_Network_received)
{
#if defined(COMPILE_UINPUT)
#if defined(COMPILE_UI)
    // Cache value.
    bool tmp_use_plot(this->_use_plot_output);
#endif // COMPILE_UI

    while(true)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: User controls, optimization:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Trainer controls." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[1]: Trained controls." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[2]: Transfer learning." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[3]: Dataset controls." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[4]: Quit." NEW_LINE, MyEA::String::Get__Time().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                4u,
                                                                                MyEA::String::Get__Time() + ": Option: "))
        {
            case 0u:
                if(ptr_trainer_Neural_Network_received->User_Controls() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 1u:
                if(ptr_trained_Neural_Network_received->User_Controls() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 2u:
                if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Transfer to trained: "))
                {
                    if(ptr_trained_Neural_Network_received->Update(*ptr_trainer_Neural_Network_received, true) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update(ptr, true)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                }
                else if(MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Transfer to trainer: "))
                {
                    if(ptr_trainer_Neural_Network_received->Update(*ptr_trained_Neural_Network_received, true) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update(ptr, true)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);

                        return(false);
                    }
                }
                    break;
            case 3u:
                if(this->User_Controls() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls()\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
                    break;
            case 4u: // Quit.
            #if defined(COMPILE_UI)
                // Check if the cached parameter have change.
                if(tmp_use_plot == false
                  &&
                  this->_use_plot_output
                  &&
                  this->Plot__Neural_Network(ptr_trained_Neural_Network_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Neural_Network(ptr)\" function. At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                __LINE__);

                    return(false);
                }
            #endif // COMPILE_UI
                    return(true);
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
#endif // COMPILE_UINPUT

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Assign_Shutdown_Block(class Shutdown_Block &ref_Shutdown_Block_received)
{
    if(this->Allocate__Shutdown_Boolean() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Shutdown_Boolean()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                
        return(false);
    }
    else if(ref_Shutdown_Block_received.Push_Back(this->_ptr_shutdown_boolean) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Push_Back(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Get__On_Shutdown(void) const { return(this->_ptr_shutdown_boolean != nullptr && this->_ptr_shutdown_boolean->load()); }

template<typename T>
bool Dataset_Manager<T>::Get__Dataset_In_Equal_Less_Holdout_Accepted(void) const { return(this->_dataset_in_equal_less_holdout_accepted); }

template<typename T>
bool Dataset_Manager<T>::Use__Metric_Loss(void) const { return(this->_use_metric_loss); }

template<typename T>
T Dataset_Manager<T>::Get__Minimum_Loss_Holdout_Accepted(void) const { return(this->_minimum_loss_holdout_accepted); }

template<typename T>
MyEA::Common::ENUM_TYPE_DATASET Dataset_Manager<T>::Get__Type_Dataset_Evaluation(void) const { return(this->_type_evaluation); }

template<typename T>
void Dataset_Manager<T>::Optimization(struct MyEA::Common::While_Condition const &ref_while_condition_received,
                                                           bool const save_neural_network_trainer_received,
                                                           bool const save_neural_network_trained_received,
                                                           T const desired_loss_received,
                                                           std::string const &ref_path_net_trainer_neural_network_received,
                                                           std::string const &ref_path_nn_trainer_neural_network_received,
                                                           std::string const &ref_path_net_trained_neural_network_received,
                                                           std::string const &ref_path_nn_trained_neural_network_received,
                                                           class Neural_Network *&ptr_trainer_Neural_Network_received,
                                                           class Neural_Network *&ptr_trained_Neural_Network_received)
{
    bool tmp_upgrade_trigger_trained(false),
           tmp_report(false),
           tmp_while(true);
    
    unsigned long long tmp_total_epoch(1ull),
                                tmp_current_epoch(0ull),
                                tmp_number_epochs_between_report(1ull);

    T tmp_loss_training(ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)),
       tmp_loss_validating(ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));

#if defined(COMPILE_COUT)
    #if defined(COMPILE_WINDOWS)
        std::chrono::steady_clock::time_point tmp_time_start,
                                                                tmp_time_end;
    #elif defined(COMPILE_LINUX)
        std::chrono::_V2::system_clock::time_point tmp_time_start,
                                                                        tmp_time_end;
    #endif
#endif
    
#if defined(COMPILE_WINDOWS)
    std::chrono::steady_clock::time_point tmp_time_start_report,
                                                            tmp_time_end_report;
#elif defined(COMPILE_LINUX)
    std::chrono::_V2::system_clock::time_point tmp_time_start_report,
                                                                    tmp_time_end_report;
#endif

#if defined(COMPILE_UINPUT)
    class Key_Logger tmp_Key_Logger;
#endif

    // Check if we reach the desired error.
    if(ptr_trained_Neural_Network_received->Get__Loss(this->_type_evaluation) > desired_loss_received)
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

        tmp_time_start_report = std::chrono::high_resolution_clock::now();

        do
        {
            tmp_report = ++tmp_current_epoch % tmp_number_epochs_between_report == 0ull;
            
            if(tmp_report && tmp_upgrade_trigger_trained)
            {
                tmp_upgrade_trigger_trained = false;

            #if defined(COMPILE_UI)
                if(this->Plot__Neural_Network(ptr_trained_Neural_Network_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Neural_Network(ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
            #endif

                if(save_neural_network_trained_received)
                {
                    if(ref_path_net_trained_neural_network_received.empty() == false && ptr_trained_Neural_Network_received->Save_Dimension_Parameters(ref_path_net_trained_neural_network_received) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save_Dimension_Parameters(%s)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 ref_path_net_trained_neural_network_received.c_str(),
                                                 __LINE__);
                    }
                    
                    if(ref_path_nn_trained_neural_network_received.empty() == false && ptr_trained_Neural_Network_received->Save_General_Parameters(ref_path_nn_trained_neural_network_received) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save_General_Parameters(%s)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 ref_path_nn_trained_neural_network_received.c_str(),
                                                 __LINE__);
                    }
                }
            }

            if(this->Get__On_Shutdown()) { break; }

        #if defined(COMPILE_UI)
            if(MyEA::Form::API__Form__Neural_Network__Get_Signal_Training_Stop())
            {
                PRINT_FORMAT("%s: A signal for stopping the training has been triggered from the user interface." NEW_LINE, MyEA::String::Get__Time().c_str());
                
                std::this_thread::sleep_for(std::chrono::seconds(1));

                break;
            }

            if(tmp_report && MyEA::Form::API__Form__Neural_Network__Get_Signal_Training_Menu())
            {
                if(this->User_Controls__Optimization(ptr_trainer_Neural_Network_received, ptr_trained_Neural_Network_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimization(ptr, ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }

                MyEA::Form::API__Form__Neural_Network__Reset_Signal_Training_Menu();
            }
        #endif

        #if defined(COMPILE_UINPUT)
            #if defined(COMPILE_WINDOWS)
                if(tmp_Key_Logger.Trigger_Key(0x51))
                {
                    PRINT_FORMAT("%s: A signal for stopping the training has been triggered from the user input." NEW_LINE, MyEA::String::Get__Time().c_str());
                    
                    std::this_thread::sleep_for(std::chrono::seconds(1));

                    break;
                }
            #elif defined(COMPILE_LINUX)
                tmp_Key_Logger.Collect_Keys_Pressed();

                if(tmp_Key_Logger.Trigger_Key('q'))
                {
                    PRINT_FORMAT("%s: A signal for stopping the training has been triggered from the user input." NEW_LINE, MyEA::String::Get__Time().c_str());

                    tmp_Key_Logger.Clear_Keys_Pressed();
                    
                    std::this_thread::sleep_for(std::chrono::seconds(1));

                    break;
                }
            #endif
            
            if(tmp_report)
            {
            #if defined(COMPILE_WINDOWS)
                if(tmp_Key_Logger.Trigger_Key(0x4D))
                {
                    if(this->User_Controls__Optimization(ptr_trainer_Neural_Network_received, ptr_trained_Neural_Network_received) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimization(ptr, ptr)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);
                    }
                }
            #elif defined(COMPILE_LINUX)
                if(tmp_Key_Logger.Trigger_Key('m'))
                {
                    tmp_Key_Logger.Clear_Keys_Pressed();

                    if(this->User_Controls__Optimization(ptr_trainer_Neural_Network_received, ptr_trained_Neural_Network_received) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Optimization(ptr, ptr)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 __LINE__);
                    }
                }
            #endif
            }
        #endif
            
            // Training.
        #if defined(COMPILE_COUT)
            if(tmp_report)
            {
                PRINT_FORMAT("%s: #=========================================================#" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Number of epochs between reports: %llu" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_number_epochs_between_report);

                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: [TRAINER]: Train on %zu example(s) from the training set." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Dataset<T>::Get__Number_Examples());
                
                tmp_time_start = std::chrono::high_resolution_clock::now();
            }
        #endif

            this->Optimize(ptr_trainer_Neural_Network_received);
            
            if(Experimental_Static_Update_BN) // Experimental.
            {
                this->Type_Update_Batch_Normalization(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ptr_trainer_Neural_Network_received);
            }

        #if defined(COMPILE_COUT)
            if(tmp_report)
            {
                tmp_time_end = std::chrono::high_resolution_clock::now();

                PRINT_FORMAT("%s: [TRAINER]: %.1f example(s) per second." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count() == 0ll ? 0.0 : static_cast<double>(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Dataset<T>::Get__Number_Examples()) / (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count())  / 1e9));
    
                PRINT_FORMAT("%s: [TRAINER]: Validate on %zu example(s) from the validation set." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Dataset<T>::Get__Number_Examples());
                
                tmp_time_start = std::chrono::high_resolution_clock::now();
            }
        #endif

            this->Evaluate(ptr_trainer_Neural_Network_received);
            
        #if defined(COMPILE_COUT)
            if(tmp_report)
            {
                tmp_time_end = std::chrono::high_resolution_clock::now();

                PRINT_FORMAT("%s: [TRAINER]: %.1f example(s) per second." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count() == 0ll ? 0.0 : static_cast<double>(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)->Dataset<T>::Get__Number_Examples()) / (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end - tmp_time_start).count())  / 1e9));
            }
        #endif

            if(ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING) + ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)
                                <
              tmp_loss_training + tmp_loss_validating)
            {
                tmp_loss_training = ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING);
                tmp_loss_validating = ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION);

                // TODO: Copy of trainer for refreshing. Faster. Need more memory. Skip saving.
                if(save_neural_network_trainer_received)
                {
                    if(ref_path_net_trainer_neural_network_received.empty() == false && ptr_trainer_Neural_Network_received->Save_Dimension_Parameters(ref_path_net_trainer_neural_network_received) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save_Dimension_Parameters(%s)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 ref_path_net_trainer_neural_network_received.c_str(),
                                                 __LINE__);
                    }
                    
                    if(ref_path_nn_trainer_neural_network_received.empty() == false && ptr_trainer_Neural_Network_received->Save_General_Parameters(ref_path_nn_trainer_neural_network_received) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save_General_Parameters(%s)\" function. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 ref_path_nn_trainer_neural_network_received.c_str(),
                                                 __LINE__);
                    }
                }
            }
            // |END| Training. |END|

            // Testing.
            if(this->_type_evaluation == MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING) 
            {
            #if defined(COMPILE_COUT)
                this->Optimization__Testing(tmp_report,
                                                         tmp_time_start,
                                                         tmp_time_end,
                                                         ptr_trainer_Neural_Network_received);
            #else
                this->Optimization__Testing(tmp_report, ptr_trainer_Neural_Network_received);
            #endif
            }
            // |END| Testing. |END|

            // Compare.
            if(ptr_trained_Neural_Network_received->Compare(this->_use_metric_loss,
                                                                                     this->_dataset_in_equal_less_holdout_accepted,
                                                                                     this->_type_evaluation,
                                                                                     this->_minimum_loss_holdout_accepted,
                                                                                     ptr_trainer_Neural_Network_received))
            {
                tmp_upgrade_trigger_trained = true;
                
                if(this->_type_evaluation != MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING) 
                {
                #if defined(COMPILE_COUT)
                    this->Optimization__Testing(false,
                                                             tmp_time_start,
                                                             tmp_time_end,
                                                             ptr_trainer_Neural_Network_received);
                #else
                    this->Optimization__Testing(false, ptr_trainer_Neural_Network_received);
                #endif
                }

                if(ptr_trained_Neural_Network_received->Update(*ptr_trainer_Neural_Network_received, true) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update(ptr, true)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
            }
            // |END| Compare. |END|
            
        #if defined(COMPILE_COUT) || defined(COMPILE_UI)
            if(tmp_report)
            {
            #if defined(COMPILE_COUT)
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Epochs %llu end." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         tmp_total_epoch);
                PRINT_FORMAT("%s: Loss at:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: [TRAINER]: Training:   %.9f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
                PRINT_FORMAT("%s: [TRAINED]: Training:   %.9f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
                PRINT_FORMAT("%s: [TRAINER]: Validating: %.9f" NEW_LINE, 
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
                PRINT_FORMAT("%s: [TRAINED]: Validating: %.9f" NEW_LINE, 
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
                PRINT_FORMAT("%s: [TRAINER]: Testing:    %.9f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
                PRINT_FORMAT("%s: [TRAINED]: Testing:    %.9f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
                PRINT_FORMAT("%s: Desired loss:          %.9f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(desired_loss_received));
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: Accuracy at:" NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s: [TRAINER]: Training:   %.5f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
                PRINT_FORMAT("%s: [TRAINED]: Training:   %.5f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
                PRINT_FORMAT("%s: [TRAINER]: Validating: %.5f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
                PRINT_FORMAT("%s: [TRAINED]: Validating: %.5f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
                PRINT_FORMAT("%s: [TRAINER]: Testing:    %.5f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
                PRINT_FORMAT("%s: [TRAINED]: Testing:    %.5f" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         Cast_T(ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            #endif

            #if defined(COMPILE_UI)
                if(this->_use_plot_loss)
                {
                    // Trainer training datapoint.
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                                0u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                                static_cast<double>(tmp_total_epoch),
                                                                                                                ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
                    
                    // Trained training datapoint.
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                                1u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                                static_cast<double>(tmp_total_epoch),
                                                                                                                ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
                    
                    // Trainer validating datapoint.
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                                0u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                                static_cast<double>(tmp_total_epoch),
                                                                                                                ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));

                    // Trained validating datapoint.
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                                1u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                                static_cast<double>(tmp_total_epoch),
                                                                                                                ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));
                    
                    MyEA::Form::API__Form__Neural_Network__Chart_Loss_Diff(0u,
                                                                                                              MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                              static_cast<double>(tmp_total_epoch));

                    // Testing datapoint.
                    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
                    {
                        // Trainer testing datapoint.
                        MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                                    0u,
                                                                                                                    MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                                    static_cast<double>(tmp_total_epoch),
                                                                                                                    ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));

                        // Trained testing datapoint.
                        MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                                    1u,
                                                                                                                    MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                                    static_cast<double>(tmp_total_epoch),
                                                                                                                    ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
                        
                        MyEA::Form::API__Form__Neural_Network__Chart_Loss_Diff(0u,
                                                                                                                  MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                                  static_cast<double>(tmp_total_epoch));
                    }
                }

                if(this->_use_plot_accuracy)
                {
                    // Trainer training datapoint.
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                                0u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                                static_cast<double>(tmp_total_epoch),
                                                                                                                ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
                    
                    // Trained training datapoint.
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                                1u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                                static_cast<double>(tmp_total_epoch),
                                                                                                                ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
                    
                    // Trainer validating datapoint.
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                                0u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                                static_cast<double>(tmp_total_epoch),
                                                                                                                ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));

                    // Trained validating datapoint.
                    MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                                1u,
                                                                                                                MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                                static_cast<double>(tmp_total_epoch),
                                                                                                                ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));

                    // Testing datapoint.
                    if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
                    {
                        // Trainer testing datapoint.
                        MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                                    0u,
                                                                                                                    MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                                    static_cast<double>(tmp_total_epoch),
                                                                                                                    ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));

                        // Trained testing datapoint.
                        MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                                    1u,
                                                                                                                    MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                                    static_cast<double>(tmp_total_epoch),
                                                                                                                    ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
                    }
                }
            #endif
            }
        #endif

            // Check if we reach the desired error.
            if(ptr_trained_Neural_Network_received->Get__Loss(this->_type_evaluation) <= desired_loss_received)
            {
                PRINT_FORMAT("%s: Desired error reach." NEW_LINE, MyEA::String::Get__Time().c_str());
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

                break;
            }

        #if defined(COMPILE_CUDA)
            if(tmp_total_epoch == 1ull
              &&
              ptr_trainer_Neural_Network_received->Use__CUDA())
            { ptr_trainer_Neural_Network_received->ptr_device_Neural_Network->Set__Limit_Device_Runtime_Pending_Launch_Count(); }
        #endif
            
            if(tmp_report)
            {
                tmp_time_end_report = std::chrono::high_resolution_clock::now();

                PRINT_FORMAT("%s: Total time performance: %s" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         MyEA::String::Get__Time_Elapse(static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end_report - tmp_time_start_report).count()) / 1e+9).c_str());
                PRINT_FORMAT("%s:" NEW_LINE, MyEA::String::Get__Time().c_str());
                
                switch(ref_while_condition_received.type_while_condition)
                {
                    case MyEA::Common::ENUM_TYPE_WHILE_CONDITION::TYPE_WHILE_CONDITION_EXPIRATION:
                        // X = ceil( Max(0, Min(D, Exp - Now)) / (Te - Ts) / to_secs / N) )
                        tmp_number_epochs_between_report = static_cast<unsigned long long>(ceil(MyEA::Math::Maximum<double>(0.0, MyEA::Math::Minimum<double>(this->_desired_optimization_time_between_reports, static_cast<double>(std::chrono::system_clock::to_time_t(ref_while_condition_received.expiration) - std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())))) / (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end_report - tmp_time_start_report).count()) / 1e+9 / static_cast<double>(tmp_current_epoch))));
                        tmp_while = tmp_number_epochs_between_report != 0ull;
                            break;
                    default:
                        // X = Max(1, D / (Te - Ts) / to_secs / N)
                        tmp_number_epochs_between_report = MyEA::Math::Maximum<unsigned long long>(1ull, static_cast<unsigned long long>(this->_desired_optimization_time_between_reports / (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tmp_time_end_report - tmp_time_start_report).count()) / 1e+9 / static_cast<double>(tmp_current_epoch))));
                            break;
                }

                tmp_current_epoch = 0ull;

            #if defined(_DEBUG) || defined(COMPILE_DEBUG) && defined(COMPILE_COUT)
                //PAUSE_TERMINAL();
            #endif

                tmp_time_start_report = std::chrono::high_resolution_clock::now();
            }

            if(Experimental_Static_Reset && std::chrono::system_clock::now() >= Experimental_Static_Time_Point) // Experimental.
            {
                ptr_trainer_Neural_Network_received->Initialization__Glorot__Gaussian();

                Experimental_Static_Time_Point = std::chrono::system_clock::now() + std::chrono::minutes(Experimental_Static_Minutes);
            }

            switch(ref_while_condition_received.type_while_condition)
            {
                case MyEA::Common::ENUM_TYPE_WHILE_CONDITION::TYPE_WHILE_CONDITION_INFINITY:
                case MyEA::Common::ENUM_TYPE_WHILE_CONDITION::TYPE_WHILE_CONDITION_EXPIRATION: break;
                case MyEA::Common::ENUM_TYPE_WHILE_CONDITION::TYPE_WHILE_CONDITION_ITERATION: tmp_while = tmp_total_epoch < ref_while_condition_received.maximum_iterations; break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: While condition type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             ref_while_condition_received.type_while_condition,
                                             MyEA::Common::ENUM_TYPE_WHILE_CONDITION_NAMES[ref_while_condition_received.type_while_condition].c_str(),
                                             __LINE__);
                    tmp_while = false;
                        break;
            }

            if(tmp_while) { ++tmp_total_epoch; }
        } while(tmp_while);
        
        if(tmp_upgrade_trigger_trained)
        {
            tmp_upgrade_trigger_trained = false;
            
        #if defined(COMPILE_UI)
            if(this->Plot__Neural_Network(ptr_trained_Neural_Network_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Neural_Network(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);
            }
        #endif

            if(save_neural_network_trained_received)
            {
                if(ref_path_net_trained_neural_network_received.empty() == false && ptr_trained_Neural_Network_received->Save_Dimension_Parameters(ref_path_net_trained_neural_network_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save_Dimension_Parameters(%s)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             ref_path_net_trained_neural_network_received.c_str(),
                                             __LINE__);
                }
                
                if(ref_path_nn_trained_neural_network_received.empty() == false && ptr_trained_Neural_Network_received->Save_General_Parameters(ref_path_nn_trained_neural_network_received) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save_General_Parameters(%s)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             ref_path_nn_trained_neural_network_received.c_str(),
                                             __LINE__);
                }
            }
        }

    #if defined(COMPILE_COUT)
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Epochs %llu end." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_total_epoch);
        PRINT_FORMAT("%s: [TRAINER]: Training:   %.9f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
        PRINT_FORMAT("%s: [TRAINED]: Training:   %.9f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
        PRINT_FORMAT("%s: [TRAINER]: Validating: %.9f" NEW_LINE, 
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
        PRINT_FORMAT("%s: [TRAINED]: Validating: %.9f" NEW_LINE, 
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
        PRINT_FORMAT("%s: [TRAINER]: Testing:    %.9f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
        PRINT_FORMAT("%s: [TRAINED]: Testing:    %.9f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
        PRINT_FORMAT("%s: Desired loss:          %.9f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(desired_loss_received));
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Accuracy at:" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: [TRAINER]: Training:   %.5f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
        PRINT_FORMAT("%s: [TRAINED]: Training:   %.5f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
        PRINT_FORMAT("%s: [TRAINER]: Validating: %.5f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
        PRINT_FORMAT("%s: [TRAINED]: Validating: %.5f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
        PRINT_FORMAT("%s: [TRAINER]: Testing:    %.5f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
        PRINT_FORMAT("%s: [TRAINED]: Testing:    %.5f" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 Cast_T(ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)));
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: #=========================================================#" NEW_LINE, MyEA::String::Get__Time().c_str());
    #endif

    #if defined(COMPILE_UI)
        if(this->_use_plot_loss)
        {
            // Trainer training datapoint.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                        0u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                        static_cast<double>(tmp_total_epoch),
                                                                                                        ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
            
            // Trained training datapoint.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                        1u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                        static_cast<double>(tmp_total_epoch),
                                                                                                        ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
            
            // Trainer validating datapoint.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                        0u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                        static_cast<double>(tmp_total_epoch),
                                                                                                        ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));

            // Trained validating datapoint.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                        1u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                        static_cast<double>(tmp_total_epoch),
                                                                                                        ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));
            
            MyEA::Form::API__Form__Neural_Network__Chart_Loss_Diff(0u,
                                                                                                      MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                      static_cast<double>(tmp_total_epoch));

            // Testing datapoint.
            if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
            {
                // Trainer testing datapoint.
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                            0u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                            static_cast<double>(tmp_total_epoch),
                                                                                                            ptr_trainer_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));

                // Trained testing datapoint.
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS,
                                                                                                            1u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                            static_cast<double>(tmp_total_epoch),
                                                                                                            ptr_trained_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
            
                MyEA::Form::API__Form__Neural_Network__Chart_Loss_Diff(0u,
                                                                                                          MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                          static_cast<double>(tmp_total_epoch));
            }
        }
        
        if(this->_use_plot_accuracy)
        {
            // Trainer training datapoint.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                        0u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                        static_cast<double>(tmp_total_epoch),
                                                                                                        ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
            
            // Trained training datapoint.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                        1u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                        static_cast<double>(tmp_total_epoch),
                                                                                                        ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
            
            // Trainer validating datapoint.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                        0u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                        static_cast<double>(tmp_total_epoch),
                                                                                                        ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));

            // Trained validating datapoint.
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                        1u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                                                                                        static_cast<double>(tmp_total_epoch),
                                                                                                        ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));
            
            // Testing datapoint.
            if(this->Get__Type_Storage() >= MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
            {
                // Trainer testing datapoint.
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                            0u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                            static_cast<double>(tmp_total_epoch),
                                                                                                            ptr_trainer_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));

                // Trained testing datapoint.
                MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY,
                                                                                                            1u,
                                                                                                            MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                                                                                            static_cast<double>(tmp_total_epoch),
                                                                                                            ptr_trained_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
            }
        }
    #endif

    #if defined(_DEBUG) || defined(COMPILE_DEBUG) && defined(COMPILE_COUT)
        //PAUSE_TERMINAL();
    #endif
    }
}

template<typename T>
#if defined(COMPILE_COUT)
    #if defined(COMPILE_WINDOWS)
void Dataset_Manager<T>::Optimization__Testing(bool const report_received,
                                                                         std::chrono::steady_clock::time_point &ref_time_start_received,
                                                                         std::chrono::steady_clock::time_point &ref_time_end_received,
                                                                         class Neural_Network *&ptr_trainer_Neural_Network_received)
    #elif defined(COMPILE_LINUX)
void Dataset_Manager<T>::Optimization__Testing(bool const report_received,
                                                                         std::chrono::_V2::system_clock::time_point &ref_time_start_received,
                                                                         std::chrono::_V2::system_clock::time_point &ref_time_end_received,
                                                                         class Neural_Network *&ptr_trainer_Neural_Network_received)
    #endif
#else
void Dataset_Manager<T>::Optimization__Testing(bool const report_received, class Neural_Network *&ptr_trainer_Neural_Network_received)
#endif
{
#if defined(COMPILE_COUT)
    if(report_received)
    {
        PRINT_FORMAT("%s: [TRAINER]: Testing on %zu example(s) from the testing set." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T>::Get__Number_Examples());
        
        ref_time_start_received = std::chrono::high_resolution_clock::now();
    }
#endif
    
#if defined(COMPILE_CUDA)
    if(ptr_trainer_Neural_Network_received->Use__CUDA()) { this->Get__CUDA()->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, ptr_trainer_Neural_Network_received); }
    else
#endif
    { this->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, ptr_trainer_Neural_Network_received); }
    
#if defined(COMPILE_COUT)
    if(report_received)
    {
        ref_time_end_received = std::chrono::high_resolution_clock::now();

        PRINT_FORMAT("%s: [TRAINER]: %.1f example(s) per second." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(ref_time_end_received - ref_time_start_received).count() == 0ll ? 0.0 : static_cast<double>(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)->Dataset<T>::Get__Number_Examples()) / (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(ref_time_end_received - ref_time_start_received).count())  / 1e9));
    }
#endif
}

template<typename T>
void Dataset_Manager<T>::Deallocate__Storage(void)
{
    if(this->_ptr_array_ptr_Dataset != nullptr)
    {
        if(this->_reference == false)
        {
            switch(this->_type_storage_data)
            {
                case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING: SAFE_DELETE(this->_ptr_array_ptr_Dataset[0u]); break;
                case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
                    SAFE_DELETE(this->_ptr_array_ptr_Dataset[0u]);
                    SAFE_DELETE(this->_ptr_array_ptr_Dataset[1u]);
                        break;
                case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
                    SAFE_DELETE(this->_ptr_array_ptr_Dataset[0u]);
                    SAFE_DELETE(this->_ptr_array_ptr_Dataset[1u]);
                    SAFE_DELETE(this->_ptr_array_ptr_Dataset[2u]);
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Dataset storage type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             this->_type_storage_data,
                                             MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[this->_type_storage_data].c_str(),
                                             __LINE__);
                        break;
            }
        }

        delete[](this->_ptr_array_ptr_Dataset);
        this->_ptr_array_ptr_Dataset = nullptr;
    }

    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE;
}

template<typename T>
void Dataset_Manager<T>::Deallocate__Shutdown_Boolean(void) { SAFE_DELETE(this->_ptr_shutdown_boolean); }

#if defined(COMPILE_CUDA)
template<typename T>
void Dataset_Manager<T>::Deallocate_CUDA(void)
{
    if(this->_ptr_CUDA_Dataset_Manager != NULL)
    {
        this->_ptr_CUDA_Dataset_Manager->Deallocate();

        CUDA__Safe_Call(cudaFree(this->_ptr_CUDA_Dataset_Manager));
    }
}

template<typename T>
bool Dataset_Manager<T>::Initialize__CUDA(void)
{
    if(this->_ptr_CUDA_Dataset_Manager == NULL)
    {
        CUDA__Safe_Call(cudaMalloc((void**)&this->_ptr_CUDA_Dataset_Manager, sizeof(class CUDA_Dataset_Manager<T>)));
        
        if(this->_ptr_CUDA_Dataset_Manager->Initialize() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        this->_ptr_CUDA_Dataset_Manager->Copy(this);

        switch(this->_type_storage_data)
        {
            case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
                if(this->Initialize_Dataset_CUDA(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_CUDA(%u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                             __LINE__);

                    return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
                if(this->Initialize_Dataset_CUDA(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_CUDA(%u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                             __LINE__);

                    return(false);
                }
            
                if(this->Initialize_Dataset_CUDA(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_CUDA(%u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                             __LINE__);

                    return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
                if(this->Initialize_Dataset_CUDA(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_CUDA(%u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                             __LINE__);

                    return(false);
                }
            
                if(this->Initialize_Dataset_CUDA(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_CUDA(%u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                             __LINE__);

                    return(false);
                }

                if(this->Initialize_Dataset_CUDA(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_CUDA(%u)\" function. At line %d." NEW_LINE,
                                             MyEA::String::Get__Time().c_str(),
                                             __FUNCTION__,
                                             MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                             __LINE__);

                    return(false);
                }
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Dataset manager type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         this->_type_storage_data,
                                         MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[this->_type_storage_data].c_str(),
                                         __LINE__);
                    return(false);
        }
    }

    return(true);
}
#endif

template<typename T>
bool Dataset_Manager<T>::Set__Maximum_Data(size_t const number_examples_received)
{
    if(this->_maximum_examples < number_examples_received || number_examples_received == 0_zu)
    {
        this->_maximum_examples = number_examples_received;

        return(true);
    }
    else { return(false); }
}

template<typename T>
bool Dataset_Manager<T>::Allocate__Shutdown_Boolean(void)
{
    std::atomic<bool> *tmp_ptr_shutdown_boolean(new std::atomic<bool>);

    if(tmp_ptr_shutdown_boolean == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 sizeof(std::atomic<bool>),
                                 __LINE__);

        return(false);
    }

    this->_ptr_shutdown_boolean = tmp_ptr_shutdown_boolean;
    
    this->_ptr_shutdown_boolean->store(false);

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Reallocate_Internal_Storage(void)
{
    class Dataset<T> const *const tmp_ptr_source_TrainingSet(this->_ptr_array_ptr_Dataset[0u]);
    class Dataset<T> *tmp_ptr_TrainingSet(nullptr),
                               *tmp_ptr_ValidatingSet(nullptr),
                               *tmp_ptr_TestingSet(nullptr);
    
    enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const tmp_type_dataset_process(tmp_ptr_source_TrainingSet->Get__Type_Dataset_Process());

    switch(this->_type_storage_data)
    {
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
            if((tmp_ptr_TrainingSet = this->Allocate__Dataset(tmp_type_dataset_process, MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_type_dataset_process,
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
            if((tmp_ptr_TrainingSet = this->Allocate__Dataset(tmp_type_dataset_process, MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_type_dataset_process,
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                         __LINE__);

                return(false);
            }
            
            if((tmp_ptr_ValidatingSet = this->Allocate__Dataset(tmp_type_dataset_process, MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_type_dataset_process,
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
            if((tmp_ptr_TrainingSet = this->Allocate__Dataset(tmp_type_dataset_process, MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_type_dataset_process,
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                         __LINE__);

                return(false);
            }
            
            if((tmp_ptr_ValidatingSet = this->Allocate__Dataset(tmp_type_dataset_process, MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_type_dataset_process,
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                         __LINE__);

                return(false);
            }
            
            if((tmp_ptr_TestingSet = this->Allocate__Dataset(tmp_type_dataset_process, MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_type_dataset_process,
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset storage type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->_type_storage_data,
                                     MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[this->_type_storage_data].c_str(),
                                     __LINE__);
                return(false);
    }
    
    struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;
    
    switch(tmp_type_dataset_process)
    {
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH: break;
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH:
            {
                class Dataset_Mini_Batch<T> const *const tmp_ptr_Dataset_Mini_Batch_Stochastic(dynamic_cast<class Dataset_Mini_Batch<T> const *>(tmp_ptr_source_TrainingSet));
                
                tmp_Dataset_Manager_Parameters.training_parameters.value_0 = static_cast<int>(tmp_ptr_Dataset_Mini_Batch_Stochastic->Get__Use__Shuffle());
                tmp_Dataset_Manager_Parameters.training_parameters.value_1 = static_cast<int>(tmp_ptr_Dataset_Mini_Batch_Stochastic->Get__Number_Examples_Per_Batch());
                tmp_Dataset_Manager_Parameters.training_parameters.value_2 = static_cast<int>(tmp_ptr_Dataset_Mini_Batch_Stochastic->Get__Number_Batch());
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION:
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION_RANDOM_SEARCH:
            {
                class Dataset_Cross_Validation<T> const *const tmp_ptr_Dataset_Cross_Validation(dynamic_cast<class Dataset_Cross_Validation<T> const *>(tmp_ptr_source_TrainingSet));
                
                tmp_Dataset_Manager_Parameters.training_parameters.value_0 = static_cast<int>(tmp_ptr_Dataset_Cross_Validation->Get__Use__Shuffle());
                tmp_Dataset_Manager_Parameters.training_parameters.value_1 = static_cast<int>(tmp_ptr_Dataset_Cross_Validation->Get__Number_Batch());
                tmp_Dataset_Manager_Parameters.training_parameters.value_2 = static_cast<int>(tmp_ptr_Dataset_Cross_Validation->Get__Number_Sub_Batch());
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset process type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_type_dataset_process,
                                        MyEA::Common::ENUM_TYPE_DATASET_PROCESS_NAMES[tmp_type_dataset_process].c_str(),
                                        __LINE__);
                return(false);
    }

    switch(this->_type_storage_data)
    {
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
            if(this->Prepare_Storage(tmp_ptr_TrainingSet) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Prepare_Storage()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
            if(this->Prepare_Storage(this->_size_dataset_training__percent,
                                                this->_size_dataset_testing__percent,
                                                tmp_ptr_TrainingSet,
                                                tmp_ptr_ValidatingSet) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Prepare_Storage()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
            if(this->Prepare_Storage(this->_size_dataset_training__percent,
                                                this->_size_dataset_validation__percent,
                                                this->_size_dataset_testing__percent,
                                                tmp_ptr_TrainingSet,
                                                tmp_ptr_ValidatingSet,
                                                tmp_ptr_TestingSet) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Prepare_Storage()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset storage type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     this->_type_storage_data,
                                     MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[this->_type_storage_data].c_str(),
                                     __LINE__);
                return(false);
    }

    if(this->Initialize_Dataset(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                        tmp_type_dataset_process,
                                        &tmp_Dataset_Manager_Parameters.training_parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset(%u, %u, ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                 this->_ptr_array_ptr_Dataset[0u]->Get__Type_Dataset_Process(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Push_Back(T const *const ptr_array_inputs_received, T const *const ptr_array_outputs_received)
{
    size_t tmp_index,
              tmp_time_step_index,
              tmp_data_input_index,
              tmp_data_output_index,
              tmp_data_time_step_index;

    T const **tmp_ptr_matrix_inputs,
                **tmp_ptr_matrix_outputs;
    T *tmp_ptr_array_inputs,
       *tmp_ptr_array_outputs;

    if(this->_maximum_examples != 0_zu && this->p_number_examples >= this->_maximum_examples)
    {
        size_t const tmp_number_examples_minus_one(this->p_number_examples - 1_zu);
        size_t tmp_input_index,
                  tmp_shift_data_input_index,
                  tmp_shift_data_output_index,
                  tmp_shift_data_time_step_index;

        // Shift index toward zero by one all inputs/outputs.
        for(tmp_index = 0_zu; tmp_index != tmp_number_examples_minus_one; ++tmp_index)
        {
            tmp_data_input_index = tmp_index * this->p_number_inputs * this->p_number_recurrent_depth;
            tmp_data_output_index = tmp_index * this->p_number_outputs * this->p_number_recurrent_depth;

            tmp_shift_data_input_index = (tmp_index + 1_zu) * this->p_number_inputs * this->p_number_recurrent_depth;
            tmp_shift_data_output_index = (tmp_index + 1_zu) * this->p_number_outputs * this->p_number_recurrent_depth;

            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                tmp_data_time_step_index = tmp_data_input_index + tmp_time_step_index * this->p_number_inputs;

                tmp_shift_data_time_step_index = tmp_shift_data_input_index + tmp_time_step_index * this->p_number_inputs;

                for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_inputs; ++tmp_input_index) { this->p_ptr_array_inputs[tmp_data_time_step_index + tmp_input_index] = this->p_ptr_array_inputs[tmp_shift_data_time_step_index + tmp_input_index]; }

                tmp_data_time_step_index = tmp_data_output_index + tmp_time_step_index * this->p_number_outputs;

                tmp_shift_data_time_step_index = tmp_shift_data_output_index + tmp_time_step_index * this->p_number_outputs;

                for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_outputs; ++tmp_input_index) { this->p_ptr_array_outputs[tmp_data_time_step_index + tmp_input_index] = this->p_ptr_array_outputs[tmp_shift_data_time_step_index + tmp_input_index]; }
            }
        }
        // |END| Shift index toward zero by one all inputs/outputs. |END|

        // Assign new inputs/outputs.
        tmp_data_input_index = tmp_number_examples_minus_one * this->p_number_inputs * this->p_number_recurrent_depth;
        tmp_data_output_index = tmp_number_examples_minus_one * this->p_number_outputs * this->p_number_recurrent_depth;

        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_data_time_step_index = tmp_data_input_index + tmp_time_step_index * this->p_number_inputs;

            for(tmp_index = 0_zu; tmp_index != this->p_number_inputs; ++tmp_index) { this->p_ptr_array_inputs[tmp_data_time_step_index + tmp_index] = ptr_array_inputs_received[tmp_time_step_index * this->p_number_inputs + tmp_index]; }
            
            tmp_data_time_step_index = tmp_data_output_index + tmp_time_step_index * this->p_number_outputs;

            for(tmp_index = 0_zu; tmp_index != this->p_number_outputs; ++tmp_index) { this->p_ptr_array_outputs[tmp_data_time_step_index + tmp_index] = ptr_array_outputs_received[tmp_time_step_index * this->p_number_outputs + tmp_index]; }
        }
        // |END| Assign new inputs/outputs. |END|
    }
    else
    {
        // TODO: Reallocate preprocessing scaler...
        // TODO: Allocate by chunk of memory. Keep tracking of the size.
        size_t const tmp_number_examples_plus_one(this->p_number_examples + 1_zu);

        // Reallocate.
        //  Inputs.
        tmp_ptr_array_inputs = Memory::reallocate_cpp<T>(this->p_ptr_array_inputs,
                                                                                    tmp_number_examples_plus_one * this->p_number_inputs * this->p_number_recurrent_depth,
                                                                                    this->p_number_examples * this->p_number_inputs * this->p_number_recurrent_depth,
                                                                                    true);
        if(tmp_ptr_array_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_number_examples_plus_one * this->p_number_inputs * this->p_number_recurrent_depth,
                                     this->p_number_examples * this->p_number_inputs * this->p_number_recurrent_depth,
                                     __LINE__);

            return(false);
        }

        this->p_ptr_array_inputs = tmp_ptr_array_inputs;

        tmp_ptr_matrix_inputs = Memory::reallocate_pointers_array_cpp<T const *>(this->p_ptr_array_inputs_array,
                                                                                                                       tmp_number_examples_plus_one,
                                                                                                                       this->p_number_examples,
                                                                                                                       false);
        if(tmp_ptr_matrix_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_number_examples_plus_one,
                                     this->p_number_examples,
                                     __LINE__);

            return(false);
        }

        this->p_ptr_array_inputs_array = tmp_ptr_matrix_inputs;
        //  |END| Inputs. |END|
        
        //  Outputs.
        tmp_ptr_array_outputs = Memory::reallocate_cpp<T>(this->p_ptr_array_outputs,
                                                                                      tmp_number_examples_plus_one * this->p_number_outputs * this->p_number_recurrent_depth,
                                                                                      this->p_number_examples * this->p_number_outputs * this->p_number_recurrent_depth,
                                                                                      true);
        if(tmp_ptr_array_outputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_number_examples_plus_one * this->p_number_outputs * this->p_number_recurrent_depth,
                                     this->p_number_examples * this->p_number_outputs * this->p_number_recurrent_depth,
                                     __LINE__);

            return(false);
        }

        this->p_ptr_array_outputs = tmp_ptr_array_outputs;

        tmp_ptr_matrix_outputs = Memory::reallocate_pointers_array_cpp<T const *>(this->p_ptr_array_outputs_array,
                                                                                                                         tmp_number_examples_plus_one,
                                                                                                                         this->p_number_examples,
                                                                                                                         false);
        if(tmp_ptr_matrix_outputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_number_examples_plus_one,
                                     this->p_number_examples,
                                     __LINE__);

            return(false);
        }

        this->p_ptr_array_outputs_array = tmp_ptr_matrix_outputs;
        //  |END| Outputs. |END|
        // |END| Reallocate. |END|

        // Assign new position.
        for(tmp_index = 0_zu; tmp_index != tmp_number_examples_plus_one; ++tmp_index)
        {
            tmp_ptr_matrix_inputs[tmp_index] = tmp_ptr_array_inputs + tmp_index * this->p_number_inputs * this->p_number_recurrent_depth;

            tmp_ptr_matrix_outputs[tmp_index] = tmp_ptr_array_outputs + tmp_index * this->p_number_outputs * this->p_number_recurrent_depth;
        }
        // |END| Assign new position. |END|

        // Assign new inputs/outputs.
        tmp_data_input_index = this->p_number_examples * this->p_number_inputs * this->p_number_recurrent_depth;
        tmp_data_output_index = this->p_number_examples * this->p_number_outputs * this->p_number_recurrent_depth;

        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_data_time_step_index = tmp_data_input_index + tmp_time_step_index * this->p_number_inputs;

            for(tmp_index = 0_zu; tmp_index != this->p_number_inputs; ++tmp_index) { tmp_ptr_array_inputs[tmp_data_time_step_index + tmp_index] = ptr_array_inputs_received[tmp_time_step_index * this->p_number_inputs + tmp_index]; }
            
            tmp_data_time_step_index = tmp_data_output_index + tmp_time_step_index * this->p_number_outputs;

            for(tmp_index = 0_zu; tmp_index != this->p_number_outputs; ++tmp_index) { tmp_ptr_array_outputs[tmp_data_time_step_index + tmp_index] = ptr_array_outputs_received[tmp_time_step_index * this->p_number_outputs + tmp_index]; }
        }
        // |END| Assign new inputs/outputs. |END|

        ++this->p_number_examples;

        this->Reallocate_Internal_Storage();
    }
    
    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Prepare_Storage(class Dataset<T> *const ptr_TrainingSet_received)
{
    if(this->Get__Number_Examples() == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number example(s) can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_TrainingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_TrainingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(this->_type_storage_data != MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE) { this->Deallocate__Storage(); }

    this->_ptr_array_ptr_Dataset = new class Dataset<T>*[1u];
    if(this->_ptr_array_ptr_Dataset == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    sizeof(class Dataset<T>*),
                                    __LINE__);

        return(false);
    }

    this->_ptr_array_ptr_Dataset[0u] = ptr_TrainingSet_received;
    ptr_TrainingSet_received->Reference(this->p_number_examples - this->p_start_index,
                                                          this->p_ptr_array_inputs_array + this->p_start_index,
                                                          this->p_ptr_array_outputs_array + this->p_start_index,
                                                          *this);

    this->_size_dataset_training__percent = 100.0;
    this->_size_dataset_validation__percent = 0.0;
    this->_size_dataset_testing__percent = 0.0;
    
    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING;

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Prepare_Storage(size_t const number_examples_training_received,
                                                                  size_t const number_examples_testing_received,
                                                                  class Dataset<T> *const ptr_TrainingSet_received,
                                                                  class Dataset<T> *const ptr_TestingSet_received)
{
    if(number_examples_training_received + number_examples_testing_received != this->Get__Number_Examples())
    {
        PRINT_FORMAT("%s: %s: ERROR: training(%zu) + testing(%zu) != total(%zu). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_examples_training_received,
                                 number_examples_testing_received,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    else if(number_examples_training_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) from the training set can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(number_examples_testing_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) from the testing set can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Get__Number_Examples() < 2_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) (%zu) is less than 2. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    else if(ptr_TrainingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_TrainingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_TestingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_TestingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->_type_storage_data != MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE) { this->Deallocate__Storage(); }

    T const **tmp_ptr_array_inputs_array(this->p_ptr_array_inputs_array + this->p_start_index),
                **tmp_ptr_array_outputs_array(this->p_ptr_array_outputs_array + this->p_start_index);

    this->_ptr_array_ptr_Dataset = new class Dataset<T>*[2u];
    if(this->_ptr_array_ptr_Dataset == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    2_zu * sizeof(class Dataset<T>*),
                                    __LINE__);

        return(false);
    }

    this->_ptr_array_ptr_Dataset[0u] = ptr_TrainingSet_received;
    ptr_TrainingSet_received->Reference(number_examples_training_received,
                                                          tmp_ptr_array_inputs_array,
                                                          tmp_ptr_array_outputs_array,
                                                          *this);

    tmp_ptr_array_inputs_array += number_examples_training_received;
    tmp_ptr_array_outputs_array += number_examples_training_received;

    this->_ptr_array_ptr_Dataset[1u] = ptr_TestingSet_received;
    ptr_TestingSet_received->Reference(number_examples_testing_received,
                                                         tmp_ptr_array_inputs_array,
                                                         tmp_ptr_array_outputs_array,
                                                         *this);

    this->_size_dataset_training__percent = 100.0 * static_cast<double>(number_examples_training_received) / static_cast<double>(this->Get__Number_Examples());
    this->_size_dataset_validation__percent = 0.0;
    this->_size_dataset_testing__percent = 100.0 * static_cast<double>(number_examples_testing_received) / static_cast<double>(this->Get__Number_Examples());
    
    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING;

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Prepare_Storage(size_t const number_examples_training_received,
                                                                  size_t const number_examples_validation_received,
                                                                  size_t const number_examples_testing_received,
                                                                  class Dataset<T> *const ptr_TrainingSet_received,
                                                                  class Dataset<T> *const ptr_ValidatingSet_received,
                                                                  class Dataset<T> *const ptr_TestingSet_received)
{
    if(number_examples_training_received + number_examples_validation_received + number_examples_testing_received != this->Get__Number_Examples())
    {
        PRINT_FORMAT("%s: %s: ERROR: training(%zu) + validating(%zu) + testing(%zu) != total(%zu). At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_examples_training_received,
                                 number_examples_validation_received,
                                 number_examples_testing_received,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    else if(number_examples_training_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) from the training set can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(number_examples_validation_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) from the validation set can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(number_examples_testing_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) from the testing set can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Get__Number_Examples() < 3_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) (%zu) is less than 3. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    else if(ptr_TrainingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_TrainingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_ValidatingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_ValidatingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_TestingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_TestingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(this->_type_storage_data != MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE) { this->Deallocate__Storage(); }
    
    T const **tmp_ptr_array_inputs_array(this->p_ptr_array_inputs_array + this->p_start_index),
                **tmp_ptr_array_outputs_array(this->p_ptr_array_outputs_array + this->p_start_index);

    this->_ptr_array_ptr_Dataset = new class Dataset<T>*[3u];
    if(this->_ptr_array_ptr_Dataset == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    3_zu * sizeof(class Dataset<T>*),
                                    __LINE__);

        return(false);
    }

    this->_ptr_array_ptr_Dataset[0u] = ptr_TrainingSet_received;
    ptr_TrainingSet_received->Reference(number_examples_training_received,
                                                          tmp_ptr_array_inputs_array,
                                                          tmp_ptr_array_outputs_array,
                                                          *this);

    tmp_ptr_array_inputs_array += number_examples_training_received;
    tmp_ptr_array_outputs_array += number_examples_training_received;

    this->_ptr_array_ptr_Dataset[1u] = ptr_ValidatingSet_received;
    ptr_ValidatingSet_received->Reference(number_examples_validation_received,
                                                            tmp_ptr_array_inputs_array,
                                                            tmp_ptr_array_outputs_array,
                                                            *this);

    tmp_ptr_array_inputs_array += number_examples_validation_received;
    tmp_ptr_array_outputs_array += number_examples_validation_received;

    this->_ptr_array_ptr_Dataset[2u] = ptr_TestingSet_received;
    ptr_TestingSet_received->Reference(number_examples_testing_received,
                                                         tmp_ptr_array_inputs_array,
                                                         tmp_ptr_array_outputs_array,
                                                         *this);

    this->_size_dataset_training__percent = 100.0 * static_cast<double>(number_examples_training_received) / static_cast<double>(this->Get__Number_Examples());
    this->_size_dataset_validation__percent = 100.0 * static_cast<double>(number_examples_validation_received) / static_cast<double>(this->Get__Number_Examples());
    this->_size_dataset_testing__percent = 100.0 * static_cast<double>(number_examples_testing_received) / static_cast<double>(this->Get__Number_Examples());
    
    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING;

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Prepare_Storage(double const number_examples_percent_training_received,
                                                                  double const number_examples_percent_testing_received,
                                                                  class Dataset<T> *const ptr_TrainingSet_received,
                                                                  class Dataset<T> *const ptr_TestingSet_received)
{
    if(number_examples_percent_training_received + number_examples_percent_testing_received != 100.0)
    {
        PRINT_FORMAT("%s: %s: ERROR: training(%f%%) + testing(%f%%) != 100.0%%. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_examples_percent_training_received,
                                 number_examples_percent_testing_received,
                                 __LINE__);

        return(false);
    }
    else if(number_examples_percent_training_received == 0.0)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) from the training set can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(number_examples_percent_testing_received == 0.0)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) from the testing set can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Get__Number_Examples() < 2_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) (%zu) is less than 2. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    else if(ptr_TrainingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_TrainingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_TestingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_TestingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->_type_storage_data != MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE) { this->Deallocate__Storage(); }

    size_t const tmp_number_examples_training(MyEA::Math::Maximum<size_t>(static_cast<size_t>(round(static_cast<double>(this->Get__Number_Examples()) * number_examples_percent_training_received / 100.0)), 1_zu)),
                       tmp_number_examples_testing(this->Get__Number_Examples() - tmp_number_examples_training);
    
    T const **tmp_ptr_array_inputs_array(this->p_ptr_array_inputs_array + this->p_start_index),
                **tmp_ptr_array_outputs_array(this->p_ptr_array_outputs_array + this->p_start_index);

    this->_ptr_array_ptr_Dataset = new class Dataset<T>*[2u];
    if(this->_ptr_array_ptr_Dataset == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    2_zu * sizeof(class Dataset<T>*),
                                    __LINE__);

        return(false);
    }

    this->_ptr_array_ptr_Dataset[0u] = ptr_TrainingSet_received;
    ptr_TrainingSet_received->Reference(tmp_number_examples_training,
                                                          tmp_ptr_array_inputs_array,
                                                          tmp_ptr_array_outputs_array,
                                                          *this);

    tmp_ptr_array_inputs_array += tmp_number_examples_training;
    tmp_ptr_array_outputs_array += tmp_number_examples_training;

    this->_ptr_array_ptr_Dataset[1u] = ptr_TestingSet_received;
    ptr_TestingSet_received->Reference(tmp_number_examples_testing,
                                                         tmp_ptr_array_inputs_array,
                                                         tmp_ptr_array_outputs_array,
                                                         *this);

    this->_size_dataset_training__percent = number_examples_percent_training_received;
    this->_size_dataset_validation__percent = 0.0;
    this->_size_dataset_testing__percent = number_examples_percent_testing_received;
    
    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING;

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Prepare_Storage(double const number_examples_percent_training_received,
                                                                  double const number_examples_percent_validation_received,
                                                                  double const number_examples_percent_testing_received,
                                                                  class Dataset<T> *const ptr_TrainingSet_received,
                                                                  class Dataset<T> *const ptr_ValidatingSet_received,
                                                                  class Dataset<T> *const ptr_TestingSet_received)
{
    if(number_examples_percent_training_received + number_examples_percent_validation_received + number_examples_percent_testing_received != 100.0)
    {
        PRINT_FORMAT("%s: %s: ERROR: training(%f%%) + validation(%f%%) + testing(%f%%) != 100.0%%. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 number_examples_percent_training_received,
                                 number_examples_percent_validation_received,
                                 number_examples_percent_testing_received,
                                 __LINE__);

        return(false);
    }
    else if(number_examples_percent_training_received == 0.0)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) from the training set can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(number_examples_percent_validation_received == 0.0)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) from the validation set can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(number_examples_percent_testing_received == 0.0)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) from the testing set can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Get__Number_Examples() < 3_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of example(s) (%zu) is less than 3. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    else if(ptr_TrainingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_TrainingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_ValidatingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_ValidatingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_TestingSet_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_TestingSet_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    if(this->_type_storage_data != MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE) { this->Deallocate__Storage(); }

    size_t const tmp_number_examples_training(MyEA::Math::Clip<size_t>(static_cast<size_t>(round(static_cast<double>(this->Get__Number_Examples()) * number_examples_percent_training_received / 100.0)),
                                                                                                             1_zu,
                                                                                                             this->Get__Number_Examples() - 2_zu)),
                       tmp_number_examples_validation(MyEA::Math::Clip<size_t>(static_cast<size_t>(round(static_cast<double>(this->Get__Number_Examples()) * number_examples_percent_validation_received / 100.0)),
                                                                                                                1_zu,
                                                                                                                this->Get__Number_Examples() - tmp_number_examples_training - 1_zu)),
                       tmp_number_examples_testing(MyEA::Math::Maximum<size_t>(this->Get__Number_Examples() - tmp_number_examples_training - tmp_number_examples_validation, 1_zu));
    
    T const **tmp_ptr_array_inputs_array(this->p_ptr_array_inputs_array + this->p_start_index),
                **tmp_ptr_array_outputs_array(this->p_ptr_array_outputs_array + this->p_start_index);

    this->_ptr_array_ptr_Dataset = new class Dataset<T>*[3u];
    if(this->_ptr_array_ptr_Dataset == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__,
                                    3_zu * sizeof(class Dataset<T>*),
                                    __LINE__);

        return(false);
    }

    this->_ptr_array_ptr_Dataset[0u] = ptr_TrainingSet_received;
    ptr_TrainingSet_received->Reference(tmp_number_examples_training,
                                                          tmp_ptr_array_inputs_array,
                                                          tmp_ptr_array_outputs_array,
                                                          *this);

    tmp_ptr_array_inputs_array += tmp_number_examples_training;
    tmp_ptr_array_outputs_array += tmp_number_examples_training;

    this->_ptr_array_ptr_Dataset[1u] = ptr_ValidatingSet_received;
    ptr_ValidatingSet_received->Reference(tmp_number_examples_validation,
                                                            tmp_ptr_array_inputs_array,
                                                            tmp_ptr_array_outputs_array,
                                                            *this);

    tmp_ptr_array_inputs_array += tmp_number_examples_validation;
    tmp_ptr_array_outputs_array += tmp_number_examples_validation;

    this->_ptr_array_ptr_Dataset[2u] = ptr_TestingSet_received;
    ptr_TestingSet_received->Reference(tmp_number_examples_testing,
                                                         tmp_ptr_array_inputs_array,
                                                         tmp_ptr_array_outputs_array,
                                                         *this);

    this->_size_dataset_training__percent = number_examples_percent_training_received;
    this->_size_dataset_validation__percent = number_examples_percent_validation_received;
    this->_size_dataset_testing__percent = number_examples_percent_testing_received;
    
    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING;

    return(true);
}

#if defined(COMPILE_CUDA)
    template<typename T>
    bool Dataset_Manager<T>::Initialize_Dataset_CUDA(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received)
    {
        Dataset<T> const *const tmp_ptr_Dataset(this->Get__Dataset_At(type_dataset_received));
        
        if(tmp_ptr_Dataset == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Get__Dataset_At(%u)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     type_dataset_received,
                                     __LINE__);

            return(false);
        }

        if(this->_ptr_CUDA_Dataset_Manager->Set__Type_Gradient_Descent(type_dataset_received, tmp_ptr_Dataset->Get__Type_Dataset_Process()) == false)
        {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"CUDA->Set__Type_Gradient_Descent(%u, %u)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     type_dataset_received,
                                     tmp_ptr_Dataset->Get__Type_Dataset_Process(),
                                     __LINE__);

            return(false);
        }

        switch(type_dataset_received)
        {
            case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
                switch(tmp_ptr_Dataset->Get__Type_Dataset_Process())
                {
                    case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH: break;
                    case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH:
                        {
                            class Dataset_Mini_Batch<T> const *const tmp_ptr_Dataset_Mini_Batch(dynamic_cast<class Dataset_Mini_Batch<T> const *const>(tmp_ptr_Dataset));

                            if(this->_ptr_CUDA_Dataset_Manager->Initialize_Mini_Batch_Stochastic_Gradient_Descent(tmp_ptr_Dataset_Mini_Batch->Get__Use__Shuffle(),
                                                                                                                                                                 tmp_ptr_Dataset_Mini_Batch->Get__Number_Examples_Per_Batch(),
                                                                                                                                                                 tmp_ptr_Dataset_Mini_Batch->Get__Number_Batch()) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Mini_Batch_Stochastic_Gradient_Descent(%s, %zu, %zu)\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         tmp_ptr_Dataset_Mini_Batch->Get__Use__Shuffle() ? "true" : "false",
                                                         tmp_ptr_Dataset_Mini_Batch->Get__Number_Examples_Per_Batch(),
                                                         tmp_ptr_Dataset_Mini_Batch->Get__Number_Batch(),
                                                         __LINE__);

                                return(false);
                            }
                        }
                            break;
                    case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION:
                        {
                            class Dataset_Cross_Validation<T> const *const tmp_ptr_Dataset_Cross_Validation(dynamic_cast<class Dataset_Cross_Validation<T> const *const>(tmp_ptr_Dataset));

                            if(this->_ptr_CUDA_Dataset_Manager->Initialize__Cross_Validation(tmp_ptr_Dataset_Cross_Validation->Get__Use__Shuffle(),
                                                                                                                              tmp_ptr_Dataset_Cross_Validation->Get__Number_Batch(),
                                                                                                                              tmp_ptr_Dataset_Cross_Validation->Get__Number_Sub_Batch()) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__Cross_Validation(%s, %zu, %zu)\" function. At line %d." NEW_LINE,
                                                         MyEA::String::Get__Time().c_str(),
                                                         __FUNCTION__,
                                                         tmp_ptr_Dataset_Cross_Validation->Get__Use__Shuffle() ? " true" : "false",
                                                         tmp_ptr_Dataset_Cross_Validation->Get__Number_Batch(),
                                                         tmp_ptr_Dataset_Cross_Validation->Get__Number_Sub_Batch(),
                                                         __LINE__);

                                return(false);
                            }
                        }
                            break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Dataset process type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                    MyEA::String::Get__Time().c_str(),
                                                    __FUNCTION__,
                                                    tmp_ptr_Dataset->Get__Type_Dataset_Process(),
                                                    MyEA::Common::ENUM_TYPE_DATASET_PROCESS_NAMES[tmp_ptr_Dataset->Get__Type_Dataset_Process()].c_str(),
                                                    __LINE__);
                            return(nullptr);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
            case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Dataset type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         type_dataset_received,
                                         MyEA::Common::ENUM_TYPE_DATASET_NAMES[type_dataset_received].c_str(),
                                         __LINE__);
                    return(false);
        }

        return(true);
    }
#endif

template<typename T>
class Dataset<T> *Dataset_Manager<T>::Allocate__Dataset(enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_dataset_process_received, enum MyEA::Common::ENUM_TYPE_DATASET const type_data_received)
{
    class Dataset<T> *tmp_ptr_Dataset(nullptr);

    switch(type_data_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
            switch(type_dataset_process_received)
            {
                case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH: // Batch.
                    if((tmp_ptr_Dataset = new class Dataset<T>) == nullptr)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 sizeof(class Dataset<T>),
                                                 __LINE__);

                        return(nullptr);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH: // Mini-batch stochastic.
                    if((tmp_ptr_Dataset = new class Dataset_Mini_Batch<T>) == nullptr)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 sizeof(class Dataset_Mini_Batch<T>),
                                                 __LINE__);

                        return(nullptr);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION: // Cross-validation
                    if((tmp_ptr_Dataset = new class Dataset_Cross_Validation<T>) == nullptr)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 sizeof(class Dataset_Cross_Validation<T>),
                                                 __LINE__);

                        return(nullptr);
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION_RANDOM_SEARCH: // Cross-validation, random search
                    if((tmp_ptr_Dataset = new class Dataset_Cross_Validation_Hyperparameter_Optimization<T>) == nullptr)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 __FUNCTION__,
                                                 sizeof(class Dataset_Cross_Validation_Hyperparameter_Optimization<T>),
                                                 __LINE__);

                        return(nullptr);
                    }
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Dataset process type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                type_dataset_process_received,
                                                MyEA::Common::ENUM_TYPE_DATASET_PROCESS_NAMES[type_dataset_process_received].c_str(),
                                                __LINE__);
                        return(nullptr);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING:
            if((tmp_ptr_Dataset = new class Dataset<T>) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         sizeof(class Dataset<T>),
                                         __LINE__);

                return(nullptr);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        type_data_received,
                                        MyEA::Common::ENUM_TYPE_DATASET_NAMES[type_data_received].c_str(),
                                        __LINE__);
                return(nullptr);
    }

    return(tmp_ptr_Dataset);
}

template<typename T>
bool Dataset_Manager<T>::Initialize_Dataset(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received,
                                                                  enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_dataset_process_received,
                                                                  struct Dataset_Parameters const *const ptr_Dataset_Parameters_received)
{
    Dataset<T> *const tmp_ptr_Dataset(this->Get__Dataset_At(type_dataset_received));
    
    if(tmp_ptr_Dataset == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Get__Dataset_At(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 type_dataset_received,
                                 __LINE__);

        return(false);
    }

    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
            switch(type_dataset_process_received)
            {
                case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH: break;
                case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH:
                    {
                        class Dataset_Mini_Batch<T> *const tmp_ptr_Dataset_Mini_Batch_Stochastic(dynamic_cast<class Dataset_Mini_Batch<T>*>(tmp_ptr_Dataset));
                        
                        bool tmp_use_shuffle;

                        size_t const tmp_number_examples(tmp_ptr_Dataset->Dataset<T>::Get__Number_Examples());
                        size_t tmp_number_desired_data_per_batch,
                                  tmp_number_maximum_batch;
                        
                        if(ptr_Dataset_Parameters_received == nullptr
                          ||
                          (ptr_Dataset_Parameters_received != nullptr && ptr_Dataset_Parameters_received->value_0 == -1))
                        {
                        #if defined(COMPILE_COUT)
                            // Shuffle.
                            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s: Shuffle:" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s:\tdefault=Yes." NEW_LINE, MyEA::String::Get__Time().c_str());
                            tmp_use_shuffle = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use shuffle: ");
                            // |END| Shuffle. |END|
                        #else
                            tmp_use_shuffle = true;
                        #endif
                        }
                        else { tmp_use_shuffle = ptr_Dataset_Parameters_received->value_0 != 0; }
                        
                        if(ptr_Dataset_Parameters_received == nullptr
                          ||
                          (ptr_Dataset_Parameters_received != nullptr && ptr_Dataset_Parameters_received->value_1 == -1))
                        {
                        #if defined(COMPILE_COUT)
                            // Desired-examples per batch.
                            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s: Desired-examples per batch:" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s:\tRange[1, %zu]." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_number_examples);
                            tmp_number_desired_data_per_batch = MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                                                           tmp_number_examples,
                                                                                                                                           MyEA::String::Get__Time() + ": Desired-examples per batch: ");
                            // |END| Desired-examples per batch. |END|
                        #else
                            if(tmp_number_examples > 1u)
                            {
                                tmp_number_desired_data_per_batch = tmp_number_examples / 128_zu;
                                tmp_number_desired_data_per_batch = MyEA::Math::Minimum<size_t>(1_zu, tmp_number_desired_data_per_batch);
                            }
                            else { tmp_number_desired_data_per_batch = 1u; }
                        #endif
                        }
                        else { tmp_number_desired_data_per_batch = static_cast<size_t>(ptr_Dataset_Parameters_received->value_1); }

                        if(ptr_Dataset_Parameters_received == nullptr
                          ||
                          (ptr_Dataset_Parameters_received != nullptr && ptr_Dataset_Parameters_received->value_2 == -1))
                        {
                        #if defined(COMPILE_COUT)
                            // Maximum sub-sample.
                            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s: Maximum sub-sample:" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s:\tRange[0, %zu]. Off = 0." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_number_examples);
                            tmp_number_maximum_batch = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                                                 tmp_number_examples,
                                                                                                                                 MyEA::String::Get__Time() + ": Maximum sub-sample: ");
                            // |END| Maximum sub-sample. |END|
                        #else
                            tmp_number_maximum_batch = 0u;
                        #endif
                        }
                        else { tmp_number_maximum_batch = static_cast<size_t>(ptr_Dataset_Parameters_received->value_2); }

                        if(tmp_ptr_Dataset_Mini_Batch_Stochastic->Initialize(tmp_use_shuffle,
                                                                                                      tmp_number_desired_data_per_batch,
                                                                                                      tmp_number_maximum_batch) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize(%s, %zu, %zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_use_shuffle ? "true" : "false",
                                                     tmp_number_desired_data_per_batch,
                                                     tmp_number_maximum_batch,
                                                     __LINE__);

                            return(false);
                        }

                        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                        PRINT_FORMAT("%s: The number of mini-batch is set to %zu." NEW_LINE,
                                                 MyEA::String::Get__Time().c_str(),
                                                 tmp_ptr_Dataset_Mini_Batch_Stochastic->Get__Number_Batch());
                    }
                        break;
                case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION:
                case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION_RANDOM_SEARCH:
                    {
                        class Dataset_Cross_Validation<T> *const tmp_ptr_Dataset_Cross_Validation(dynamic_cast<class Dataset_Cross_Validation<T> *>(tmp_ptr_Dataset));

                        bool tmp_use_shuffle;
                        
                        size_t const tmp_number_examples(tmp_ptr_Dataset_Cross_Validation->Dataset<T>::Get__Number_Examples());
                        size_t tmp_number_k_folds,
                                  tmp_number_k_sub_folds;
                        
                        if(tmp_number_examples < 2_zu)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: The number of example(s) (%zu) is less than 3. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_number_examples,
                                                     __LINE__);

                            return(false);
                        }
                        
                        if(ptr_Dataset_Parameters_received == nullptr
                          ||
                          (ptr_Dataset_Parameters_received != nullptr && ptr_Dataset_Parameters_received->value_0 == -1))
                        {
                        #if defined(COMPILE_COUT)
                            // Shuffle.
                            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s: Shuffle:" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s:\tdefault=Yes." NEW_LINE, MyEA::String::Get__Time().c_str());
                            tmp_use_shuffle = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Use shuffle: ");
                            // |END| Shuffle. |END|
                        #else
                            tmp_use_shuffle = true;
                        #endif
                        }
                        else { tmp_use_shuffle = ptr_Dataset_Parameters_received->value_0 != 0; }
                        
                        if(ptr_Dataset_Parameters_received == nullptr
                          ||
                          (ptr_Dataset_Parameters_received != nullptr && ptr_Dataset_Parameters_received->value_1 == -1))
                        {
                        #if defined(COMPILE_COUT)
                            // K-fold.
                            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s: K-fold:" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s:\tRange[2, %zu]." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_number_examples);
                            tmp_number_k_folds = MyEA::String::Cin_Number<size_t>(2_zu,
                                                                                                                   tmp_number_examples,
                                                                                                                   MyEA::String::Get__Time() + ": K-fold: ");
                            // |END| K-fold. |END|
                        #else
                            tmp_number_k_folds = tmp_number_examples / 5_zu;
                            tmp_number_k_folds = MyEA::Math::Minimum<size_t>(2_zu, tmp_number_k_folds);
                            tmp_number_k_folds = MyEA::Math::Maximum<size_t>(5_zu, tmp_number_k_folds);
                        #endif
                        }
                        else { tmp_number_k_folds = static_cast<size_t>(ptr_Dataset_Parameters_received->value_1); }
                        
                        if(ptr_Dataset_Parameters_received == nullptr
                          ||
                          (ptr_Dataset_Parameters_received != nullptr && ptr_Dataset_Parameters_received->value_2 == -1))
                        {
                        #if defined(COMPILE_COUT)
                            // K-sub-fold.
                            size_t const tmp_number_examples_training((tmp_number_k_folds - 1_zu) * (tmp_number_examples / tmp_number_k_folds));

                            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s: K-sub-fold:" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s:\tRange[0, %zu]." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_number_examples_training);
                            PRINT_FORMAT("%s:\tdefault=%zu." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     tmp_number_k_folds - 1_zu);
                            tmp_number_k_sub_folds = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                                          tmp_number_examples_training,
                                                                                                                          MyEA::String::Get__Time() + ": K-sub-fold: ");
                            // |END| K-sub-fold. |END|
                        #else
                            tmp_number_k_sub_folds = 0_zu;
                        #endif
                        }
                        else { tmp_number_k_sub_folds = static_cast<size_t>(ptr_Dataset_Parameters_received->value_2); }

                        if(tmp_ptr_Dataset_Cross_Validation->Initialize__Fold(tmp_use_shuffle,
                                                                                                      tmp_number_k_folds,
                                                                                                      tmp_number_k_sub_folds) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__Fold(%s, %zu, %zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_use_shuffle ? "true" : "false",
                                                     tmp_number_k_folds,
                                                     tmp_number_k_sub_folds,
                                                     __LINE__);

                            return(false);
                        }
                    }

                    if(type_dataset_process_received == MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION_RANDOM_SEARCH)
                    {
                        class Dataset_Cross_Validation_Hyperparameter_Optimization<T> *const tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization(dynamic_cast<class Dataset_Cross_Validation_Hyperparameter_Optimization<T> *>(tmp_ptr_Dataset));
                        
                        size_t tmp_number_hyper_optimization_iterations,
                                  tmp_number_hyper_optimization_iterations_delay;
                        
                        if(ptr_Dataset_Parameters_received == nullptr
                          ||
                          (ptr_Dataset_Parameters_received != nullptr && ptr_Dataset_Parameters_received->value_3 == -1))
                        {
                        #if defined(COMPILE_COUT)
                            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s: Number hyperparameter optimization iteration(s):" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s:\tRange[1, 8]." NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s:\tdefault=10." NEW_LINE, MyEA::String::Get__Time().c_str());
                            tmp_number_hyper_optimization_iterations = MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Iteration(s): ");
                        #else
                            tmp_number_hyper_optimization_iterations = 10_zu;
                        #endif
                        }
                        else { tmp_number_hyper_optimization_iterations = ptr_Dataset_Parameters_received->value_3 != 0; }
                        
                        if(tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->Set__Number_Hyperparameter_Optimization_Iterations(tmp_number_hyper_optimization_iterations) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Number_Hyperparameter_Optimization_Iterations(%zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_number_hyper_optimization_iterations,
                                                     __LINE__);

                            return(false);
                        }
                        
                        if(ptr_Dataset_Parameters_received == nullptr
                          ||
                          (ptr_Dataset_Parameters_received != nullptr && ptr_Dataset_Parameters_received->value_4 == -1))
                        {
                        #if defined(COMPILE_COUT)
                            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s: Number hyperparameter optimization iteration(s) delay:" NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s:\tRange[1, 8]." NEW_LINE, MyEA::String::Get__Time().c_str());
                            PRINT_FORMAT("%s:\tdefault=25." NEW_LINE, MyEA::String::Get__Time().c_str());
                            tmp_number_hyper_optimization_iterations_delay = MyEA::String::Cin_Number<size_t>(0_zu, MyEA::String::Get__Time() + ": Iteration(s) delay: ");
                        #else
                            tmp_number_hyper_optimization_iterations_delay = 25_zu;
                        #endif
                        }
                        else { tmp_number_hyper_optimization_iterations_delay = ptr_Dataset_Parameters_received->value_4 != 0; }
                        
                        if(tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->Set__Number_Hyperparameter_Optimization_Iterations_Delay(tmp_number_hyper_optimization_iterations_delay) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Number_Hyperparameter_Optimization_Iterations_Delay(%zu)\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     tmp_number_hyper_optimization_iterations_delay,
                                                     __LINE__);

                            return(false);
                        }
                        else if(tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization->User_Controls() == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls()\" function. At line %d." NEW_LINE,
                                                     MyEA::String::Get__Time().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                    }
                        break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Dataset process type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                type_dataset_process_received,
                                                MyEA::Common::ENUM_TYPE_DATASET_PROCESS_NAMES[type_dataset_process_received].c_str(),
                                                __LINE__);
                        return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     type_dataset_received,
                                     MyEA::Common::ENUM_TYPE_DATASET_NAMES[type_dataset_received].c_str(),
                                     __LINE__);
                return(false);
    }

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Preparing_Dataset_Manager(struct Dataset_Manager_Parameters const *const ptr_Dataset_Manager_Parameters_received)
{
    double tmp_percent_training_size,
              tmp_percent_validation_size,
              tmp_percent_testing_size;

    class Dataset<T> *tmp_ptr_TrainingSet(nullptr),
                               *tmp_ptr_ValidatingSet(nullptr),
                               *tmp_ptr_TestingSet(nullptr);

    // Type storage.
    unsigned int tmp_type_storage_choose;

    if(ptr_Dataset_Manager_Parameters_received == nullptr
      ||
      (ptr_Dataset_Manager_Parameters_received != nullptr && ptr_Dataset_Manager_Parameters_received->type_storage == -1))
    {
    #if defined(COMPILE_COUT)
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Type storage: " NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Training." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[1]: Training and testing." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[2]: Training, validation and testing." NEW_LINE, MyEA::String::Get__Time().c_str());

        tmp_type_storage_choose = MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                                                2u,
                                                                                                                MyEA::String::Get__Time() + ": Choose: ");
    #else
        tmp_type_storage_choose = 0u;
    #endif
    }
    else { tmp_type_storage_choose = static_cast<unsigned int>(ptr_Dataset_Manager_Parameters_received->type_storage); }

    // Type training.
    unsigned int tmp_type_training_choose;

    if(ptr_Dataset_Manager_Parameters_received == nullptr
      ||
      (ptr_Dataset_Manager_Parameters_received != nullptr && ptr_Dataset_Manager_Parameters_received->type_training == -1))
    {
    #if defined(COMPILE_COUT)
        PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s: Type training: " NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[0]: Batch." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[1]: Mini-batch." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[2]: Cross-validation." NEW_LINE, MyEA::String::Get__Time().c_str());
        PRINT_FORMAT("%s:\t[3]: Cross-validation, random search." NEW_LINE, MyEA::String::Get__Time().c_str());

        tmp_type_training_choose = MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                                                3u,
                                                                                                                MyEA::String::Get__Time() + ": Choose: ");
    #else
        tmp_type_training_choose = 0u;
    #endif
    }
    else { tmp_type_training_choose = static_cast<unsigned int>(ptr_Dataset_Manager_Parameters_received->type_training); }

    switch(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE>(tmp_type_storage_choose + 1u))
    {
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
            if((tmp_ptr_TrainingSet = this->Allocate__Dataset(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u), MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                         __LINE__);

                return(false);
            }

            if(this->Prepare_Storage(tmp_ptr_TrainingSet) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Prepare_Storage(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            if(this->Initialize_Dataset(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                                ptr_Dataset_Manager_Parameters_received == nullptr ? nullptr : &ptr_Dataset_Manager_Parameters_received->training_parameters) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset(%u, %u, ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                         static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
            if(ptr_Dataset_Manager_Parameters_received == nullptr
              ||
              (ptr_Dataset_Manager_Parameters_received != nullptr && ptr_Dataset_Manager_Parameters_received->percent_training_size == 0.0))
            {
            #if defined(COMPILE_COUT)
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_percent_training_size = MyEA::String::Cin_Real_Number<double>(100.0 - 99.9999,
                                                                                                                        99.9999,
                                                                                                                        MyEA::String::Get__Time() + ": Training size [" + std::to_string(100.0 - 99.9999) + "%, 99.9999%]: ");
            #else
                tmp_percent_training_size = 80.0;
            #endif
            }
            else { tmp_percent_training_size = ptr_Dataset_Manager_Parameters_received->percent_training_size; }

            tmp_percent_testing_size = 100.0 - tmp_percent_training_size;
            
            if((tmp_ptr_TrainingSet = this->Allocate__Dataset(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u), MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                         __LINE__);

                return(false);
            }
            
            if((tmp_ptr_TestingSet = this->Allocate__Dataset(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u), MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                         __LINE__);

                return(false);
            }

            if(this->Prepare_Storage(tmp_percent_training_size,
                                                tmp_percent_testing_size,
                                                tmp_ptr_TrainingSet,
                                                tmp_ptr_TestingSet) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Prepare_Storage(%f, %f, ptr, ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_percent_training_size,
                                         tmp_percent_testing_size,
                                         __LINE__);

                return(false);
            }

            if(this->Initialize_Dataset(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                                ptr_Dataset_Manager_Parameters_received == nullptr ? nullptr : &ptr_Dataset_Manager_Parameters_received->training_parameters) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset(%u, %u, ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                         static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
            if(ptr_Dataset_Manager_Parameters_received == nullptr
              ||
              (ptr_Dataset_Manager_Parameters_received != nullptr && ptr_Dataset_Manager_Parameters_received->percent_training_size == 0.0))
            {
            #if defined(COMPILE_COUT)
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_percent_training_size = MyEA::String::Cin_Real_Number<double>(1e-4,
                                                                                                                        100.0 - 1e-4,
                                                                                                                        MyEA::String::Get__Time() + ": Training size [1e-4%, 99.9999%]: ");
            #else
                tmp_percent_training_size = 60.0;
            #endif
            }
            else { tmp_percent_training_size = ptr_Dataset_Manager_Parameters_received->percent_training_size; }

            if(ptr_Dataset_Manager_Parameters_received == nullptr
              ||
              (ptr_Dataset_Manager_Parameters_received != nullptr && ptr_Dataset_Manager_Parameters_received->percent_validation_size == 0.0))
            {
            #if defined(COMPILE_COUT)
                PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
                tmp_percent_validation_size = MyEA::String::Cin_Real_Number<double>(1e-5,
                                                                                                                           100.0 - 1e-5 - tmp_percent_training_size,
                                                                                                                           MyEA::String::Get__Time() + ": Validation size [1e-5%, " + MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_MANIPULATOR_STRING::TYPE_MANIPULATOR_STRING_DEFAULTFLOAT>(100.0 - 1e-5 - tmp_percent_training_size) + "%]: ");
            #else
                tmp_percent_validation_size = 20.0;
            #endif
            }
            else { tmp_percent_validation_size = ptr_Dataset_Manager_Parameters_received->percent_validation_size; }

            tmp_percent_testing_size = 100.0 - tmp_percent_training_size - tmp_percent_validation_size;
            
            if((tmp_ptr_TrainingSet = this->Allocate__Dataset(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u), MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                         __LINE__);

                return(false);
            }
            
            if((tmp_ptr_ValidatingSet = this->Allocate__Dataset(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u), MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                         __LINE__);

                return(false);
            }
            
            if((tmp_ptr_TestingSet = this->Allocate__Dataset(static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u), MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING,
                                         __LINE__);

                return(false);
            }

            if(this->Prepare_Storage(tmp_percent_training_size,
                                                tmp_percent_validation_size,
                                                tmp_percent_testing_size,
                                                tmp_ptr_TrainingSet,
                                                tmp_ptr_ValidatingSet,
                                                tmp_ptr_TestingSet) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Prepare_Storage(%f, %f, %f, ptr, ptr, ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_percent_training_size,
                                         tmp_percent_validation_size,
                                         tmp_percent_testing_size,
                                         __LINE__);

                return(false);
            }

            if(this->Initialize_Dataset(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                                ptr_Dataset_Manager_Parameters_received == nullptr ? nullptr : &ptr_Dataset_Manager_Parameters_received->training_parameters) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset(%u, %u, ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                         static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS>(tmp_type_training_choose + 1u),
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset storage type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_type_storage_choose + 1u,
                                     MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE>(tmp_type_storage_choose + 1u)].c_str(),
                                     __LINE__);
                return(false);
    }

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Copy__Storage(class Dataset_Manager<T> const *const ptr_source_Dataset_Manager_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_source_Dataset_Manager_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_source_Dataset_Manager_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else
#endif
    if(ptr_source_Dataset_Manager_received->_type_storage_data == MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    {
        PRINT_FORMAT("%s: %s: ERROR: Undefined dataset storage type. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }

    class Dataset<T> const *const tmp_ptr_TrainingSet(ptr_source_Dataset_Manager_received->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));

    if(tmp_ptr_TrainingSet == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_TrainingSet\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(tmp_ptr_TrainingSet->Get__Type_Dataset_Process() == MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_NONE)
    {
        PRINT_FORMAT("%s: %s: ERROR: Undefined dataset process type. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }

    struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;

    tmp_Dataset_Manager_Parameters.type_storage = static_cast<int>(ptr_source_Dataset_Manager_received->_type_storage_data) - 1;
    tmp_Dataset_Manager_Parameters.type_training = static_cast<int>(tmp_ptr_TrainingSet->Get__Type_Dataset_Process()) - 1;
    
    tmp_Dataset_Manager_Parameters.percent_training_size = ptr_source_Dataset_Manager_received->_size_dataset_training__percent;
    tmp_Dataset_Manager_Parameters.percent_validation_size = ptr_source_Dataset_Manager_received->_size_dataset_validation__percent;
    
    switch(tmp_ptr_TrainingSet->Get__Type_Dataset_Process())
    {
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH: break;
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH:
            {
                class Dataset_Mini_Batch<T> const *const tmp_ptr_Dataset_Mini_Batch(dynamic_cast<class Dataset_Mini_Batch<T> const *const>(tmp_ptr_TrainingSet));

                tmp_Dataset_Manager_Parameters.training_parameters.value_0 = static_cast<int>(tmp_ptr_Dataset_Mini_Batch->Get__Use__Shuffle());
                tmp_Dataset_Manager_Parameters.training_parameters.value_1 = static_cast<int>(tmp_ptr_Dataset_Mini_Batch->Get__Number_Examples_Per_Batch());
                tmp_Dataset_Manager_Parameters.training_parameters.value_2 = static_cast<int>(tmp_ptr_Dataset_Mini_Batch->Get__Number_Batch());
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION:
            {
                class Dataset_Cross_Validation<T> const *const tmp_ptr_Dataset_Cross_Validation(dynamic_cast<class Dataset_Cross_Validation<T> const *const>(tmp_ptr_TrainingSet));
                
                tmp_Dataset_Manager_Parameters.training_parameters.value_0 = static_cast<int>(tmp_ptr_Dataset_Cross_Validation->Get__Use__Shuffle());
                tmp_Dataset_Manager_Parameters.training_parameters.value_1 = static_cast<int>(tmp_ptr_Dataset_Cross_Validation->Get__Number_Batch());
                tmp_Dataset_Manager_Parameters.training_parameters.value_2 = static_cast<int>(tmp_ptr_Dataset_Cross_Validation->Get__Number_Sub_Batch());
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset process type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        tmp_ptr_TrainingSet->Get__Type_Dataset_Process(),
                                        MyEA::Common::ENUM_TYPE_DATASET_PROCESS_NAMES[tmp_ptr_TrainingSet->Get__Type_Dataset_Process()].c_str(),
                                        __LINE__);
                return(false);
    }

    if(this->Preparing_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preparing_Dataset_Manager(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Reference(class Dataset_Manager<T> *const ptr_source_Dataset_Manager_received)
{
    this->Deallocate();
    
    this->Dataset<T>::Reference(ptr_source_Dataset_Manager_received->p_number_examples - ptr_source_Dataset_Manager_received->p_start_index,
                                               ptr_source_Dataset_Manager_received->p_ptr_array_inputs_array + ptr_source_Dataset_Manager_received->p_start_index,
                                               ptr_source_Dataset_Manager_received->p_ptr_array_outputs_array + ptr_source_Dataset_Manager_received->p_start_index,
                                               *ptr_source_Dataset_Manager_received);

    this->_reference = true;

    // Private.
    this->_dataset_in_equal_less_holdout_accepted = ptr_source_Dataset_Manager_received->_dataset_in_equal_less_holdout_accepted;
    this->_use_metric_loss = ptr_source_Dataset_Manager_received->_use_metric_loss;
    this->_ptr_shutdown_boolean = ptr_source_Dataset_Manager_received->_ptr_shutdown_boolean;
    
    this->_maximum_examples = ptr_source_Dataset_Manager_received->_maximum_examples;

    this->_minimum_loss_holdout_accepted = ptr_source_Dataset_Manager_received->_minimum_loss_holdout_accepted;

    if(this->Copy__Storage(ptr_source_Dataset_Manager_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Storage(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    this->_desired_optimization_time_between_reports = ptr_source_Dataset_Manager_received->_desired_optimization_time_between_reports;

    this->_type_evaluation = ptr_source_Dataset_Manager_received->_type_evaluation;
    
#if defined(COMPILE_UI)
    this->_use_plot_loss = ptr_source_Dataset_Manager_received->_use_plot_loss;
    this->_use_plot_accuracy = ptr_source_Dataset_Manager_received->_use_plot_accuracy;
    this->_use_plot_output = ptr_source_Dataset_Manager_received->_use_plot_output;

    this->_maximum_ploted_examples = ptr_source_Dataset_Manager_received->_maximum_ploted_examples;
    this->_time_delay_ploted = ptr_source_Dataset_Manager_received->_time_delay_ploted;
#endif

#if defined(COMPILE_CUDA)
    this->_ptr_CUDA_Dataset_Manager = ptr_source_Dataset_Manager_received->_ptr_CUDA_Dataset_Manager;
#endif
    // |END| Private. |END|

    return(true);
}

template<typename T>
bool Dataset_Manager<T>::Deallocate(void)
{
    this->Deallocate__Storage();
    
    if(this->_reference)
    {
        // Protected.
        this->p_ptr_array_inputs = nullptr;
        this->p_ptr_array_inputs_array = nullptr;
    
        this->p_ptr_array_outputs = nullptr;
        this->p_ptr_array_outputs_array = nullptr;
        // |END| Protected. |END|

        // Private.
        this->_ptr_shutdown_boolean = nullptr;
    
    #if defined(COMPILE_CUDA)
        this->_ptr_CUDA_Dataset_Manager = nullptr;
    #endif
        // |END| Private. |END|
    }
    else
    {
        this->Deallocate__Shutdown_Boolean();
        
    #if defined(COMPILE_CUDA)
        this->Deallocate_CUDA();
    #endif
        
        if(this->Dataset<T>::Deallocate() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Dataset<T>::Deallocate()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        
        if(this->Hyperparameter_Optimization<T>::Deallocate() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Hyperparameter_Optimization<T>::Deallocate()\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    this->_reference = false;

    return(true);
}

template<typename T>
T Dataset_Manager<T>::Training(class Neural_Network *const ptr_Neural_Network_received)
{
    class Dataset<T> *tmp_ptr_Dataset(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));

    if(tmp_ptr_Dataset == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Get__Dataset_At(%s)\" function. Pointer return is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_DATASET_NAMES[MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING].c_str(),
                                 __LINE__);

        return(T(1));
    }

    return(tmp_ptr_Dataset->Training(ptr_Neural_Network_received));
}

template<typename T>
T Dataset_Manager<T>::Optimize(class Neural_Network *const ptr_Neural_Network_received)
{
    if(this->Get__Hyperparameter_Optimization() != ENUM_TYPE_HYPERPARAMETER_OPTIMIZATION::TYPE_HYPERPARAMETER_OPTIMIZATION_NONE
      &&
      ++this->p_optimization_iterations_since_hyper_optimization >= this->p_number_hyper_optimization_iterations_delay)
    {
        this->p_optimization_iterations_since_hyper_optimization = 0_zu;

        if(this->Hyperparameter_Optimization<T>::Optimize(this, ptr_Neural_Network_received) == false)
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
        { return(this->Get__CUDA()->Training(ptr_Neural_Network_received)); }
        else
    #endif
        { return(this->Training(ptr_Neural_Network_received)); }
    }
}

template<typename T>
T Dataset_Manager<T>::Type_Testing(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, class Neural_Network *const ptr_Neural_Network_received)
{
    class Dataset<T> *tmp_ptr_Dataset(this->Get__Dataset_At(type_dataset_received));

    if(tmp_ptr_Dataset == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Get__Dataset_At(%s)\" function. Pointer return is a nullptr." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_DATASET_NAMES[type_dataset_received].c_str());

        return(T(1));
    }

    T const tmp_previous_loss(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)),
                tmp_previous_accuracy(ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)),
                tmp_loss(tmp_ptr_Dataset->Testing(ptr_Neural_Network_received)); // By default: loss_testing = tmp_loss;

    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
            ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_loss);
            ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
                break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
            ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, tmp_loss);
            ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
                break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        type_dataset_received,
                                        MyEA::Common::ENUM_TYPE_DATASET_NAMES[type_dataset_received].c_str(),
                                        __LINE__);
                return((std::numeric_limits<ST_>::max)());
    }

    // Reset testing loss/accuracy.
    if(type_dataset_received != MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)
    {
        ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_previous_loss);
        ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_previous_accuracy);
    }
    // |END| Reset testing loss/accuracy. |END|

    return(tmp_loss);
}

template<typename T>
std::pair<T, T> Dataset_Manager<T>::Type_Update_Batch_Normalization(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, class Neural_Network *const ptr_Neural_Network_received)
{
    ptr_Neural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_UPDATE_BATCH_NORMALIZATION;
    
    T const tmp_previous_loss(ptr_Neural_Network_received->Get__Loss(type_dataset_received)),
                tmp_previous_accuracy(ptr_Neural_Network_received->Get__Accuracy(type_dataset_received)),
                tmp_loss(this->Type_Testing(type_dataset_received, ptr_Neural_Network_received));
    T const tmp_accuracy(ptr_Neural_Network_received->Get__Accuracy(type_dataset_received));

    ptr_Neural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_INFERENCE;
    
    // Reset past loss/accuracy.
    if(type_dataset_received != MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)
    {
        ptr_Neural_Network_received->Set__Loss(type_dataset_received, tmp_previous_loss);
        ptr_Neural_Network_received->Set__Accuracy(type_dataset_received, tmp_previous_accuracy);
    }
    // |END| Reset past loss/accuracy. |END|

    return(std::make_pair(tmp_loss, tmp_accuracy));
}

template<typename T>
T Dataset_Manager<T>::Evaluate(class Neural_Network *const ptr_Neural_Network_received)
{
    if(this->Get__Evaluation_Require())
    {
        if(this->Evaluation(this) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Evaluate(ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);
            
            return((std::numeric_limits<ST_>::max)());
        }
        
        return(Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)));
    }
    else
    {
    #if defined(COMPILE_CUDA)
        if(ptr_Neural_Network_received->Use__CUDA())
        { return(this->Get__CUDA()->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, ptr_Neural_Network_received)); }
        else
    #endif
        { return(this->Type_Testing(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, ptr_Neural_Network_received)); }
    }
}

template<typename T>
MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE Dataset_Manager<T>::Get__Type_Storage(void) const { return(this->_type_storage_data); }

template<typename T>
Dataset<T>* Dataset_Manager<T>::Get__Dataset_At(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received) const
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(this->_ptr_array_ptr_Dataset == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"_ptr_array_ptr_Dataset\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(nullptr);
    }
#endif

    switch(this->_type_storage_data)
    {
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING: return(this->_ptr_array_ptr_Dataset[0u]);
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
            switch(type_dataset_received)
            {
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: return(this->_ptr_array_ptr_Dataset[0u]);
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: return(this->_ptr_array_ptr_Dataset[0u]);
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: return(this->_ptr_array_ptr_Dataset[1u]);
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Dataset type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                type_dataset_received,
                                                MyEA::Common::ENUM_TYPE_DATASET_NAMES[type_dataset_received].c_str(),
                                                __LINE__);
                        return(nullptr);
            }
        case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
            switch(type_dataset_received)
            {
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: return(this->_ptr_array_ptr_Dataset[0u]);
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: return(this->_ptr_array_ptr_Dataset[1u]);
                case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: return(this->_ptr_array_ptr_Dataset[2u]);
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Dataset type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                MyEA::String::Get__Time().c_str(),
                                                __FUNCTION__,
                                                type_dataset_received,
                                                MyEA::Common::ENUM_TYPE_DATASET_NAMES[type_dataset_received].c_str(),
                                                __LINE__);
                        return(nullptr);
            }
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset storage type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                        MyEA::String::Get__Time().c_str(),
                                        __FUNCTION__,
                                        this->_type_storage_data,
                                        MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[this->_type_storage_data].c_str(),
                                        __LINE__);
                return(nullptr);
    }

    return(nullptr);
}

#if defined(COMPILE_CUDA)
template<typename T>
CUDA_Dataset_Manager<T>* Dataset_Manager<T>::Get__CUDA(void) { return(this->_ptr_CUDA_Dataset_Manager); }
#endif

template<typename T>
Dataset_Manager<T>::~Dataset_Manager(void) { this->Deallocate(); }

// template initialization declaration.
template class Dataset_Manager<T_>;
