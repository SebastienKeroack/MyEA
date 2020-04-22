#include "stdafx.hpp"

#include<Files/File.hpp>
#include <Math/MODWT.hpp>

#include <Neural_Network/Dataset_Manager.hpp>

#include <fstream>
#include <iostream>
#include <omp.h>

bool Input_Dataset_File(enum MyEA::Common::ENUM_TYPE_DATASET_FILE &ref_type_dateset_file_received, std::string const &ref_path_received)
{
    unsigned int tmp_type_choose;

    std::vector<enum MyEA::Common::ENUM_TYPE_DATASET_FILE> tmp_vector_Type_Dataset_File_Available;
    
    if(MyEA::File::Path_Exist(ref_path_received + ".dataset")) { tmp_vector_Type_Dataset_File_Available.push_back(MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_DATASET); }

    if(MyEA::File::Path_Exist(ref_path_received + ".dataset-input") && MyEA::File::Path_Exist(ref_path_received + ".dataset-output")) { tmp_vector_Type_Dataset_File_Available.push_back(MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_DATASET_SPLIT); }
    
    if(MyEA::File::Path_Exist(ref_path_received + ".idx3-ubyte") && MyEA::File::Path_Exist(ref_path_received + ".idx1-ubyte")) { tmp_vector_Type_Dataset_File_Available.push_back(MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_MNIST); }
    
    if(tmp_vector_Type_Dataset_File_Available.empty())
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available in the folder. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_vector_Type_Dataset_File_Available.size() == 1_zu)
    {
        ref_type_dateset_file_received = tmp_vector_Type_Dataset_File_Available.at(0u);

        return(true);
    }
    else
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: Dataset file type available:" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

        for(size_t tmp_type_dataset_file_available_size(tmp_vector_Type_Dataset_File_Available.size()),
                      tmp_type_dataset_file_available_index(0_zu); tmp_type_dataset_file_available_index != tmp_type_dataset_file_available_size; ++tmp_type_dataset_file_available_index)
        {
            PRINT_FORMAT("%s:\t[%zu] Type: %s." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     tmp_type_dataset_file_available_index,
                                     MyEA::Common::ENUM_TYPE_DATASET_FILE_NAMES[tmp_vector_Type_Dataset_File_Available.at(tmp_type_dataset_file_available_index)].c_str());
        }

        tmp_type_choose = MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                                    static_cast<unsigned int>(tmp_vector_Type_Dataset_File_Available.size()) - 1u,
                                                                                                    MyEA::Time::Date_Time_Now() + ": Type data file: ");

        ref_type_dateset_file_received = tmp_vector_Type_Dataset_File_Available.at(tmp_type_choose);

        return(true);
    }
}

template<typename T>
bool Append_To_Dataset_File(size_t const size_inputs_received,
                                            size_t const size_outputs_received,
                                            size_t const size_recurrent_depth_received,
                                            T const *const ptr_array_inputs_received,
                                            T const *const ptr_array_outputs_received,
                                            std::string &ref_path_received)
{
    if(MyEA::File::Path_Exist(ref_path_received) == false) { MyEA::File::File_Create(ref_path_received); }
            
    if(MyEA::File::Retrieve_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Retrieve_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    std::fstream tmp_fstream(ref_path_received, std::ios::out | std::ios::binary | std::ios::in); // Open For Read / Write
    // Check if the file is open and is not empty

    if(tmp_fstream.is_open())
    {
        size_t tmp_number_examples;

        std::string tmp_line,
                        tmp_string_write;

        // Get first line
        if(tmp_fstream.peek() != std::fstream::traits_type::eof()) { getline(tmp_fstream, tmp_line); }
        else { tmp_fstream.clear(); }

        // Get the number of data in the file
        tmp_number_examples = static_cast<size_t>(strtoul(tmp_line.c_str(),
                                                                                        NULL,
                                                                                        0));

        tmp_string_write = std::to_string(tmp_number_examples + 1_zu) + " " +
                                    std::to_string(size_inputs_received) + " " +
                                    std::to_string(size_outputs_received) + " " +
                                    std::to_string(size_recurrent_depth_received);

        if(std::to_string(tmp_number_examples + 1_zu).length() != std::to_string(tmp_number_examples).length()) // Different length file.
        {
            tmp_fstream.seekg(0, std::fstream::end);
                    
            std::streampos tmp_file_size = tmp_fstream.tellg();
                    
            tmp_file_size -= tmp_string_write.length() - 1_zu;

            tmp_fstream.seekg(tmp_string_write.length() - 1_zu, std::fstream::beg);

            std::string tmp_buffer(static_cast<size_t>(tmp_file_size), ' ');
            
            tmp_fstream.read(&tmp_buffer[0], tmp_file_size);

            tmp_string_write += tmp_buffer;
        }
        else
        {
            tmp_fstream.seekg(0, std::fstream::beg);

            tmp_fstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.length()));
                    
            if(tmp_fstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"fail()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            else if(tmp_fstream.bad())
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"bad()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            tmp_fstream.flush();
                    
            if(tmp_fstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"fail()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            else if(tmp_fstream.bad())
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"bad()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }

        tmp_fstream.close();
        // ---

        // Input & Output
        std::ofstream tmp_ofstream;

        if(std::to_string(tmp_number_examples + 1_zu).length() != std::to_string(tmp_number_examples).length()) // Different length file.
        {
            // Create tempory file.
            if(MyEA::File::Write_Temporary_File(ref_path_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Write_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         ref_path_received.c_str(),
                                         __LINE__);

                return(false);
            }

            // Truncate existing file
            tmp_ofstream.open(ref_path_received, std::ios::out | std::ios::binary | std::ios::trunc);
        }
        else
        {
            tmp_ofstream.open(ref_path_received, std::ios::out | std::ios::binary | std::ios::app);

            tmp_string_write = "";
        }

        if(tmp_ofstream.is_open())
        {
            for(size_t tmp_index,
                            tmp_time_step_index(0_zu); tmp_time_step_index != size_recurrent_depth_received; ++tmp_time_step_index)
            {
                // Input
                tmp_string_write += NEW_LINE;
                for(tmp_index = 0_zu; tmp_index != size_inputs_received; ++tmp_index)
                {
                    tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_array_inputs_received[tmp_time_step_index * size_inputs_received + tmp_index], 9u);

                    if(tmp_index + 1_zu != size_inputs_received) { tmp_string_write += " "; }
                }

                // Output
                tmp_string_write += NEW_LINE;
                for(tmp_index = 0_zu; tmp_index != size_outputs_received; ++tmp_index)
                {
                    tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(ptr_array_outputs_received[tmp_time_step_index * size_outputs_received + tmp_index], 9u);

                    if(tmp_index + 1_zu != size_outputs_received) { tmp_string_write += " "; }
                }
            }

            tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.length()));
                    
            if(tmp_ofstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"fail()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            else if(tmp_ofstream.bad())
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"bad()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            tmp_ofstream.flush();
                    
            if(tmp_ofstream.fail())
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"fail()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            else if(tmp_ofstream.bad())
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"bad()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            tmp_ofstream.close();
        }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }
            
    if(MyEA::File::Delete_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Delete_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

template bool Append_To_Dataset_File<T_>(size_t const,
                                                                 size_t const,
                                                                 size_t const,
                                                                 T_ const *const,
                                                                 T_ const *const,
                                                                 std::string &);

template<typename T>
bool Time_Direction(size_t const number_outputs_received,
                              size_t const number_recurrent_depth_received,
                              T const minimum_range_received,
                              T const maximum_range_received,
                              T *const ptr_array_outputs_received)
{
    if(minimum_range_received > maximum_range_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum range (%f) can not be greater than maximum range (%f). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_range_received),
                                 Cast_T(maximum_range_received),
                                 __LINE__);
        
        return(false);
    }
    else if(number_recurrent_depth_received <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be less or equal to one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    size_t tmp_time_step_index,
              tmp_input_index;
    
    T tmp_direction;
    
    // Time step T...1.
    for(tmp_time_step_index = number_recurrent_depth_received; --tmp_time_step_index > 0;)
    {
        // Input N...1.
        for(tmp_input_index = number_outputs_received; --tmp_input_index > 0;)
        {
            tmp_direction = MyEA::Math::Sign<T>(ptr_array_outputs_received[tmp_time_step_index * number_outputs_received + tmp_input_index] - ptr_array_outputs_received[tmp_time_step_index * number_outputs_received + tmp_input_index - 1_zu]);
            
            switch(static_cast<int>(tmp_direction))
            {
                case 1: tmp_direction = maximum_range_received; break;
                case -1: tmp_direction = minimum_range_received; break;
                case 0: break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Direction (%f) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             Cast_T(tmp_direction),
                                             __LINE__);
                        break;
            }

            ptr_array_outputs_received[tmp_time_step_index * number_outputs_received + tmp_input_index] = tmp_direction;
        }

        // First input.
        tmp_direction = MyEA::Math::Sign<T>(ptr_array_outputs_received[tmp_time_step_index * number_outputs_received] - ptr_array_outputs_received[(tmp_time_step_index - 1_zu) * number_outputs_received]);
                
        switch(static_cast<int>(tmp_direction))
        {
            case 1: tmp_direction = maximum_range_received; break;
            case -1: tmp_direction = minimum_range_received; break;
            case 0: break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Direction (%f) is not managed in the switch. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         Cast_T(tmp_direction),
                                         __LINE__);
                    break;
        }

        ptr_array_outputs_received[tmp_time_step_index * number_outputs_received] = tmp_direction;
    }

    // First time step.
    //  Input N...1.
    for(tmp_input_index = number_outputs_received; --tmp_input_index > 0;)
    {
        tmp_direction = MyEA::Math::Sign<T>(ptr_array_outputs_received[tmp_input_index] - ptr_array_outputs_received[tmp_input_index - 1_zu]);
        
        switch(static_cast<int>(tmp_direction))
        {
            case 1: tmp_direction = maximum_range_received; break;
            case -1: tmp_direction = minimum_range_received; break;
            case 0: break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Direction (%f) is not managed in the switch. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         Cast_T(tmp_direction),
                                         __LINE__);
                    break;
        }

        ptr_array_outputs_received[tmp_input_index] = tmp_direction;
    }

    //  First input.
    ptr_array_outputs_received[0u] = T(0);
    // |END| First time step. |END|

    return(true);
}

template bool Time_Direction<T_>(size_t const,
                                                  size_t const,
                                                  T_ const,
                                                  T_ const,
                                                  T_ *const);

template<class T>
Scaler__Minimum_Maximum<T>& Scaler__Minimum_Maximum<T>::operator=(struct Scaler__Minimum_Maximum<T> const &ref_Scaler__Minimum_Maximum_received)
{
    if(&ref_Scaler__Minimum_Maximum_received != this) { this->Copy(ref_Scaler__Minimum_Maximum_received); }

    return(*this);
}

template<typename T>
void Scaler__Minimum_Maximum<T>::Copy(struct Scaler__Minimum_Maximum<T> const &ref_Scaler__Minimum_Maximum_received)
{
    this->start_index = ref_Scaler__Minimum_Maximum_received.start_index;
    this->end_index = ref_Scaler__Minimum_Maximum_received.end_index;

    this->minimum_value = ref_Scaler__Minimum_Maximum_received.minimum_value;
    this->maximum_value = ref_Scaler__Minimum_Maximum_received.maximum_value;
    this->minimum_range = ref_Scaler__Minimum_Maximum_received.minimum_range;
    this->maximum_range = ref_Scaler__Minimum_Maximum_received.maximum_range;
}

template<class T>
Scaler__Zero_Centered<T>& Scaler__Zero_Centered<T>::operator=(struct Scaler__Zero_Centered<T> const &ref_Scaler__Zero_Centered_received)
{
    if(&ref_Scaler__Zero_Centered_received != this) { this->Copy(ref_Scaler__Zero_Centered_received); }

    return(*this);
}

template<typename T>
void Scaler__Zero_Centered<T>::Copy(struct Scaler__Zero_Centered<T> const &ref_Scaler__Zero_Centered_received)
{
    this->start_index = ref_Scaler__Zero_Centered_received.start_index;
    this->end_index = ref_Scaler__Zero_Centered_received.end_index;

    this->multiplier = ref_Scaler__Zero_Centered_received.multiplier;
}

template<typename T>
Dataset<T>::Dataset(void) { }

template<typename T>
Dataset<T>::Dataset(enum MyEA::Common::ENUM_TYPE_DATASET_FILE const type_data_file_read_received, std::string const &ref_path_received) { this->Allocate(type_data_file_read_received, ref_path_received); }

template<class T>
// TODO: Dataset overload=
Dataset<T>& Dataset<T>::operator=(class Dataset<T> const &ref_Dataset_received)
{
    if(&ref_Dataset_received != this) { PRINT("TODO: Dataset overload=" NEW_LINE); /*this->Copy(ref_Dataset_received);*/ }

    return(*this);
}

template<typename T>
void Dataset<T>::Copy(class Dataset<T> const &ref_Dataset_received)
{
    PRINT("TODO: Dataset \"Copy\"" NEW_LINE);
    if(this->_reference == false) { this->Deallocate(); }
    
    this->_use_multi_label = ref_Dataset_received._use_multi_label;

    this->p_number_examples_allocated = this->p_number_examples = ref_Dataset_received.p_number_examples;
    this->p_number_inputs = ref_Dataset_received.p_number_inputs;
    this->p_number_outputs = ref_Dataset_received.p_number_outputs;
    this->p_number_recurrent_depth = ref_Dataset_received.p_number_recurrent_depth;

    this->p_file_buffer_size = ref_Dataset_received.p_file_buffer_size;
    this->p_file_buffer_shift_size = ref_Dataset_received.p_file_buffer_shift_size;

    this->p_ptr_array_inputs_array = ref_Dataset_received.p_ptr_array_inputs_array;
    this->p_ptr_array_outputs_array = ref_Dataset_received.p_ptr_array_outputs_array;
    
    this->p_type_dataset_process = ref_Dataset_received.p_type_dataset_process;
    this->p_type_data_file = ref_Dataset_received.p_type_data_file;

    this->_reference = true;
    
    this->p_start_index = ref_Dataset_received.p_start_index;
}

template<typename T>
void Dataset<T>::Reference(size_t const number_examples_received,
                                         T const **ptr_array_inputs_array_received,
                                         T const **ptr_array_outputs_array_received,
                                         class Dataset<T> const &ref_Dataset_received)
{
    this->Deallocate();
    
    this->_use_multi_label = ref_Dataset_received.Use__Multi_Label();

    this->p_number_examples_allocated = this->p_number_examples = number_examples_received;
    this->p_number_inputs = ref_Dataset_received.Get__Number_Inputs();
    this->p_number_outputs = ref_Dataset_received.Get__Number_Outputs();
    this->p_number_recurrent_depth = ref_Dataset_received.Get__Number_Recurrent_Depth();

    this->p_file_buffer_size = ref_Dataset_received.p_file_buffer_size;
    this->p_file_buffer_shift_size = ref_Dataset_received.p_file_buffer_shift_size;

    this->p_ptr_array_inputs_array = ptr_array_inputs_array_received;
    this->p_ptr_array_outputs_array = ptr_array_outputs_array_received;

    this->p_type_data_file = ref_Dataset_received.p_type_data_file;
    
    this->_reference = true;
    
    this->p_start_index = 0_zu;
}

template<typename T>
void Dataset<T>::Train_Epoch_OpenMP(class Neural_Network *const ptr_Neural_Network_received)
{
    if(ptr_Neural_Network_received->Use__Dropout__Bernoulli()
      ||
      ptr_Neural_Network_received->Use__Dropout__Bernoulli__Inverted()
      ||
      ptr_Neural_Network_received->Use__Dropout__Alpha())
    { ptr_Neural_Network_received->Dropout_Bernoulli(); }
    else if(ptr_Neural_Network_received->Use__Dropout__Zoneout())
    { ptr_Neural_Network_received->Dropout_Zoneout(); }

    ptr_Neural_Network_received->Reset__Loss();

    switch(ptr_Neural_Network_received->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND: 
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: 
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Train_Batch_BP_OpenMP(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
            ptr_Neural_Network_received->previous_loss_rprop = ptr_Neural_Network_received->loss_rprop;

            this->Train_Batch_BP_OpenMP(ptr_Neural_Network_received);

            ptr_Neural_Network_received->loss_rprop = MyEA::Math::Absolute<T>(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Optimizer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_Neural_Network_received->type_optimizer_function,
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[ptr_Neural_Network_received->type_optimizer_function].c_str(),
                                     __LINE__);
                break;
    }
}

template<typename T>
void Dataset<T>::Train_Epoch_Loop(class Neural_Network *const ptr_Neural_Network_received)
{
    if(ptr_Neural_Network_received->Use__Dropout__Bernoulli()
      ||
      ptr_Neural_Network_received->Use__Dropout__Bernoulli__Inverted()
      ||
      ptr_Neural_Network_received->Use__Dropout__Alpha())
    { ptr_Neural_Network_received->Dropout_Bernoulli(); }
    else if(ptr_Neural_Network_received->Use__Dropout__Zoneout())
    { ptr_Neural_Network_received->Dropout_Zoneout(); }

    ptr_Neural_Network_received->Reset__Loss();

    switch(ptr_Neural_Network_received->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND: 
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: 
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Train_Batch_BP_Loop(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
            ptr_Neural_Network_received->previous_loss_rprop = ptr_Neural_Network_received->loss_rprop;

            this->Train_Batch_BP_Loop(ptr_Neural_Network_received);
                
            ptr_Neural_Network_received->loss_rprop = MyEA::Math::Absolute<T>(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Optimizer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_Neural_Network_received->type_optimizer_function,
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[ptr_Neural_Network_received->type_optimizer_function].c_str(),
                                     __LINE__);
                break;
    }
}

template<typename T>
void Dataset<T>::Train_Batch_BP_OpenMP(class Neural_Network *const ptr_Neural_Network_received)
{
    size_t const tmp_number_examples(this->Get__Number_Examples()),
                       tmp_maximum_batch_size(ptr_Neural_Network_received->batch_size),
                       tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_index(0_zu),
              tmp_batch_size(0_zu);
    
    #pragma omp parallel private(tmp_batch_index, tmp_batch_size)
    for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
    {
        tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;
        
        ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);

        ptr_Neural_Network_received->Compute__Error(tmp_batch_size, this->Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);

        ptr_Neural_Network_received->Backward_Pass(tmp_batch_size);

        ptr_Neural_Network_received->Update_Derivative_Weight(tmp_batch_size,
                                                                                           ptr_Neural_Network_received->ptr_array_layers + 1,
                                                                                           ptr_Neural_Network_received->ptr_last_layer);
    }

    ptr_Neural_Network_received->Merge__Post__Training();

    ptr_Neural_Network_received->number_accuracy_trial = tmp_number_examples * (this->Get__Number_Recurrent_Depth() - ptr_Neural_Network_received->number_time_delays) * (ptr_Neural_Network_received->type_accuracy_function == MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY ? 1_zu : ptr_Neural_Network_received->Get__Output_Size());
}

template<typename T>
void Dataset<T>::Train_Batch_BP_Loop(class Neural_Network *const ptr_Neural_Network_received)
{
    size_t const tmp_number_examples(this->Get__Number_Examples()),
                       tmp_maximum_batch_size(ptr_Neural_Network_received->batch_size),
                       tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_size,
              tmp_batch_index;
    
    for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
    {
        tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;

        ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);

        ptr_Neural_Network_received->Compute__Error(tmp_batch_size, this->Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);

        ptr_Neural_Network_received->Backward_Pass(tmp_batch_size);

        ptr_Neural_Network_received->Update_Derivative_Weight(tmp_batch_size,
                                                                                           ptr_Neural_Network_received->ptr_array_layers + 1,
                                                                                           ptr_Neural_Network_received->ptr_last_layer);
    }

    ptr_Neural_Network_received->number_accuracy_trial = tmp_number_examples * (this->Get__Number_Recurrent_Depth() - ptr_Neural_Network_received->number_time_delays) * (ptr_Neural_Network_received->type_accuracy_function == MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY ? 1_zu : ptr_Neural_Network_received->Get__Output_Size());
}

template<typename T>
bool Dataset<T>::Initialize(void)
{
    this->p_number_examples_allocated = this->p_number_examples = 0_zu;
    this->p_number_inputs = 0_zu;
    this->p_number_outputs = 0_zu;
    this->p_number_recurrent_depth = 0_zu;

    this->p_file_buffer_size = 32_zu * KILOBYTE * KILOBYTE; // byte(s).
    this->p_file_buffer_shift_size = 256_zu * KILOBYTE; // byte(s).

    this->p_ptr_array_inputs_array = nullptr;
    this->p_ptr_array_inputs = nullptr;

    this->p_ptr_array_outputs_array = nullptr;
    this->p_ptr_array_outputs = nullptr;

    this->p_type_dataset_process = MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH;
    
    this->p_start_index = 0_zu;

    return(true);
}

template<typename T>
bool Dataset<T>::Save(std::string const &ref_path_received, bool const normalize_received)
{
    switch(this->p_type_data_file)
    {
        case MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_DATASET: return(this->Save__Dataset(ref_path_received + ".dataset", normalize_received)); break;
        case MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_DATASET_SPLIT: return(this->Save__Dataset_Split(ref_path_received)); break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset file type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_type_data_file,
                                     MyEA::Common::ENUM_TYPE_DATASET_FILE_NAMES[this->p_type_data_file].c_str(),
                                     __LINE__);
                return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Save__Dataset(std::string const &ref_path_received, bool const normalize_received)
{
    if(MyEA::File::Write_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Write_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    size_t const tmp_number_examples(normalize_received ? this->Dataset<T>::Get__Number_Examples() : this->p_number_examples);
    size_t tmp_time_step_index,
              tmp_example_index,
              tmp_index;

    T_ const *const *const tmp_ptr_array_inputs_array(normalize_received ? this->Dataset<T>::Get__Input_Array() : this->p_ptr_array_inputs_array),
                  *const *const tmp_ptr_array_outputs_array(normalize_received ? this->Dataset<T>::Get__Output_Array() : this->p_ptr_array_outputs_array);

    std::string tmp_string_write;

    std::ofstream tmp_ofstream(ref_path_received, std::ios::out | std::ios::binary | std::ios::trunc);

    if(tmp_ofstream.is_open())
    {
        // Topology
        tmp_string_write = std::to_string(tmp_number_examples) + " " +
                                    std::to_string(this->p_number_inputs) + " " +
                                    std::to_string(this->p_number_outputs) + " " +
                                    std::to_string(this->p_number_recurrent_depth);

        // Input & Output
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                // Next line.
                tmp_string_write += NEW_LINE;

                // Inputs [0...(N-1)]
                for(tmp_index = 0_zu; tmp_index != this->p_number_inputs - 1_zu; ++tmp_index) { tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * this->p_number_inputs + tmp_index], 9u) + " "; }

                // Last input
                tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * this->p_number_inputs + tmp_index], 9u);

                // Next line.
                tmp_string_write += NEW_LINE;

                // Output [0...(N-1)]
                for(tmp_index = 0_zu; tmp_index != this->p_number_outputs - 1_zu; ++tmp_index) { tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + tmp_index], 9u) + " "; }

                // Last Output
                tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + tmp_index], 9u);
            }

            if(tmp_string_write.size() >= this->p_file_buffer_size)
            {
                tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

                tmp_string_write = "";
            }
        }
        
        tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

        tmp_ofstream.flush();
        tmp_ofstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    if(MyEA::File::Delete_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Delete_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Save__Dataset_Custom(std::string const &ref_path_received) // WARNING
{
    if(MyEA::File::Write_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Write_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    size_t const tmp_number_examples(this->p_number_examples);
    size_t tmp_time_step_index,
              tmp_example_index,
              tmp_index;

    T_ const *const *const tmp_ptr_array_inputs_array(this->p_ptr_array_inputs_array),
                  *const *const tmp_ptr_array_outputs_array(this->p_ptr_array_outputs_array);

    std::string tmp_string_write;

    std::ofstream tmp_ofstream(ref_path_received, std::ios::out | std::ios::binary | std::ios::trunc);

    if(tmp_ofstream.is_open())
    {
        // Topology
        tmp_string_write = std::to_string(tmp_number_examples) + " " +
                                    std::to_string(this->p_number_inputs) + " " +
                                    std::to_string(this->p_number_outputs - 2_zu) + " " +
                                    std::to_string(this->p_number_recurrent_depth);

        // Input & Output
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                // Next line.
                tmp_string_write += NEW_LINE;

                // Inputs [0...(N-1)]
                for(tmp_index = 0_zu; tmp_index != this->p_number_inputs - 1_zu; ++tmp_index) { tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * this->p_number_inputs + tmp_index], 9u) + " "; }

                // Last input
                tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * this->p_number_inputs + tmp_index], 9u);

                // Next line.
                tmp_string_write += NEW_LINE;

                // Output [0...(N-1)]
                for(tmp_index = 2_zu; tmp_index != this->p_number_outputs - 1_zu; ++tmp_index)
                {
                    if(tmp_index == 2_zu)
                    {
                        tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + 0_zu]
                                                                                                                                                                                                                                                          +
                                                                                                                                                                                                                                                          tmp_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + 1_zu]
                                                                                                                                                                                                                                                          +
                                                                                                                                                                                                                                                          tmp_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + 2_zu], 9u) + " ";
                    }
                    else
                    {
                        tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + tmp_index], 9u) + " ";
                    }
                }

                // Last Output
                tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + tmp_index], 9u);
            }

            if(tmp_string_write.size() >= this->p_file_buffer_size)
            {
                tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

                tmp_string_write = "";
            }
        }
        
        tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

        tmp_ofstream.flush();
        tmp_ofstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    if(MyEA::File::Delete_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Delete_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Save__Dataset_Split(std::string const &ref_path_received)
{
    if(this->Save__Dataset_Split__Input(ref_path_received + ".dataset-input") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save__Dataset_Split__Input(%s.dataset-input)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(this->Save__Dataset_Split__Output(ref_path_received + ".dataset-output") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save__Dataset_Split__Output(%s.dataset-output)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Save__Dataset_Split__Input(std::string const &ref_path_received)
{
    if(MyEA::File::Write_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Write_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    size_t tmp_time_step_index,
              tmp_example_index,
              tmp_index;

    std::string tmp_string_write;

    std::ofstream tmp_ofstream(ref_path_received, std::ios::out | std::ios::binary | std::ios::trunc);

    if(tmp_ofstream.is_open())
    {
        // Topology
        tmp_string_write = std::to_string(this->p_number_examples) + " " +
                                    std::to_string(this->p_number_inputs) + " " +
                                    std::to_string(this->p_number_recurrent_depth);

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                // Next line.
                tmp_string_write += NEW_LINE;

                // Inputs [0...(N-1)]
                for(tmp_index = 0_zu; tmp_index != this->p_number_inputs - 1_zu; ++tmp_index)
                { tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->p_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * this->p_number_inputs + tmp_index], 9u) + " "; }

                // Last input
                tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->p_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * this->p_number_inputs + tmp_index], 9u);
            }

            if(tmp_string_write.size() >= this->p_file_buffer_size)
            {
                tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

                tmp_string_write = "";
            }
        }
        
        tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

        tmp_ofstream.flush();
        tmp_ofstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }
    
    if(MyEA::File::Delete_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Delete_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Save__Dataset_Split__Output(std::string const &ref_path_received)
{
    if(MyEA::File::Write_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Write_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    size_t tmp_time_step_index,
              tmp_example_index,
              tmp_index;

    std::string tmp_string_write;

    std::ofstream tmp_ofstream(ref_path_received, std::ios::out | std::ios::binary | std::ios::trunc);

    if(tmp_ofstream.is_open())
    {
        // Topology
        tmp_string_write = std::to_string(this->p_number_examples) + " " +
                                    std::to_string(this->p_number_outputs) + " " +
                                    std::to_string(this->p_number_recurrent_depth);

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                // Next line.
                tmp_string_write += NEW_LINE;

                // Output [0...(N-1)]
                for(tmp_index = 0_zu; tmp_index != this->p_number_outputs - 1_zu; ++tmp_index)
                { tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->p_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + tmp_index], 9u) + " "; }

                // Last Output
                tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->p_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + tmp_index], 9u);
            }

            if(tmp_string_write.size() >= this->p_file_buffer_size)
            {
                tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

                tmp_string_write = "";
            }
        }
        
        tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

        tmp_ofstream.flush();
        tmp_ofstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    if(MyEA::File::Delete_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Delete_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Save(class Neural_Network *const ptr_Autoencoder_received, std::string path_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Check_Topology(ptr_Autoencoder_received->number_inputs,
                                               ptr_Autoencoder_received->number_outputs,
                                               ptr_Autoencoder_received->number_recurrent_depth) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Check_Topology(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_Autoencoder_received->number_inputs,
                                 ptr_Autoencoder_received->number_outputs,
                                 ptr_Autoencoder_received->number_recurrent_depth,
                                 __LINE__);

        return(false);
    }
    else if(ptr_Autoencoder_received->type_network != MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER)
    {
        PRINT_FORMAT("%s: %s: ERROR: The neural network (%s) receive as argument need to be a %s. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[ptr_Autoencoder_received->type_network].c_str(),
                                 MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER].c_str(),
                                 __LINE__);

        return(false);
    }
    
    auto tmp_Reset_IO_Mode([tmp_use_first_layer_as_input = ptr_Autoencoder_received->use_first_layer_as_input,
                                            tmp_use_last_layer_as_output = ptr_Autoencoder_received->use_last_layer_as_output,
                                            &ptr_Autoencoder_received]() -> bool
    {
        bool tmp_succes(true);

        if(ptr_Autoencoder_received->Set__Input_Mode(tmp_use_first_layer_as_input) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Input_Mode(%s)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_use_first_layer_as_input ? "true" : "false",
                                     __LINE__);

            tmp_succes = false;
        }
        
        if(ptr_Autoencoder_received->Set__Output_Mode(tmp_use_last_layer_as_output) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Output_Mode(%s)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_use_last_layer_as_output ? "true" : "false",
                                     __LINE__);

            tmp_succes = false;
        }

        return(tmp_succes);
    });
    
    if(ptr_Autoencoder_received->Set__Input_Mode(true) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Input_Mode(true)\" function. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    __LINE__);
        
        return(false);
    }
    else if(ptr_Autoencoder_received->Set__Output_Mode(false) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Output_Mode(false)\" function. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    __LINE__);
        
        if(tmp_Reset_IO_Mode() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Reset_IO_Mode()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    else if(ptr_Autoencoder_received->Update__Batch_Size(this->p_number_examples) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_examples,
                                 __LINE__);

        return(false);
    }
    
    // By default save the dataset into .dataset extension.
    path_received += ".dataset";
    
    if(MyEA::File::Write_Temporary_File(path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Write_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 path_received.c_str(),
                                 __LINE__);
        
        if(tmp_Reset_IO_Mode() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Reset_IO_Mode()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }

    size_t const tmp_number_examples(this->p_number_examples),
                       tmp_maximum_batch_size(ptr_Autoencoder_received->batch_size),
                       tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size)))),
                       tmp_number_outputs(ptr_Autoencoder_received->Get__Output_Size());
    size_t tmp_index,
              tmp_time_step_index,
              tmp_example_index,
              tmp_batch_size,
              tmp_batch_index;
    
    T const *tmp_ptr_array_outputs;

    std::string tmp_string_write;

    std::ofstream tmp_ofstream(path_received, std::ios::out | std::ios::binary | std::ios::trunc);

    if(tmp_ofstream.is_open())
    {
        // Topology
        tmp_string_write = std::to_string(tmp_number_examples) + " " +
                                    std::to_string(tmp_number_outputs) + " " +
                                    std::to_string(tmp_number_outputs) + " " +
                                    std::to_string(this->p_number_recurrent_depth);

        // Input & Output
        for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
        {
            tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;
            
            ptr_Autoencoder_received->Forward_Pass(tmp_batch_size, this->p_ptr_array_inputs_array + tmp_batch_index * tmp_maximum_batch_size);

            for(tmp_example_index = 0_zu; tmp_example_index != tmp_batch_size; ++tmp_example_index)
            {
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
                {
                    tmp_ptr_array_outputs = ptr_Autoencoder_received->Get__Outputs(tmp_example_index, tmp_time_step_index);

                    // Input
                    tmp_string_write += NEW_LINE;
                    for(tmp_index = 0_zu; tmp_index != tmp_number_outputs; ++tmp_index)
                    {
                        tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_outputs[tmp_index], 9u);

                        if(tmp_index + 1_zu != tmp_number_outputs) { tmp_string_write += " "; }
                    }

                    // Output
                    tmp_string_write += NEW_LINE;
                    for(tmp_index = 0_zu; tmp_index != tmp_number_outputs; ++tmp_index)
                    {
                        tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_outputs[tmp_index], 9u);

                        if(tmp_index + 1_zu != tmp_number_outputs) { tmp_string_write += " "; }
                    }
                }
            }
        }

        tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

        tmp_ofstream.flush();
        tmp_ofstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 path_received.c_str(),
                                 __LINE__);
        
        if(tmp_Reset_IO_Mode() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Reset_IO_Mode()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    
    if(MyEA::File::Delete_Temporary_File(path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Delete_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 path_received.c_str(),
                                 __LINE__);
        
        if(tmp_Reset_IO_Mode() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Reset_IO_Mode()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    else if(tmp_Reset_IO_Mode() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Reset_IO_Mode()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
// TODO: Finish function. Backward or forward?
bool Dataset<T>::Shift_Entries(size_t const shift_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(shift_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Shift can not be zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(shift_received >= this->p_number_examples)
    {
        PRINT_FORMAT("%s: %s: ERROR: Shift (%zu) can not be greater or equal to the number of data (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 shift_received,
                                 this->p_number_examples,
                                 __LINE__);

        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_number_examples(this->p_number_examples),
                       tmp_number_examples_shifted(this->p_number_examples - shift_received),
                       tmp_input_size(this->p_number_inputs),
                       tmp_number_outputs(this->p_number_outputs),
                       tmp_number_recurrent_depth(this->p_number_recurrent_depth);
    size_t tmp_example_index;
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples_shifted; ++tmp_example_index)
        {
            MEMCPY(this->p_ptr_array_inputs + tmp_example_index * tmp_input_size * tmp_number_recurrent_depth,
                          this->p_ptr_array_inputs + (tmp_example_index + 1_zu) * tmp_input_size * tmp_number_recurrent_depth,
                          tmp_input_size * tmp_number_recurrent_depth * sizeof(T));
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples_shifted; ++tmp_example_index)
        {
            MEMCPY(this->p_ptr_array_outputs + tmp_example_index * tmp_number_outputs * tmp_number_recurrent_depth,
                          this->p_ptr_array_outputs + (tmp_example_index + 1_zu) * tmp_number_outputs * tmp_number_recurrent_depth,
                          tmp_number_outputs * tmp_number_recurrent_depth * sizeof(T));
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    // Inputs.
    T *tmp_ptr_array_inputs(MyEA::Memory::Cpp::Reallocate<T>(this->p_ptr_array_inputs,
                                                                                 tmp_input_size * tmp_number_examples_shifted * tmp_number_recurrent_depth,
                                                                                 tmp_input_size * tmp_number_examples * tmp_number_recurrent_depth));

    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sizeof(T),
                                 tmp_input_size * tmp_number_examples_shifted * tmp_number_recurrent_depth,
                                 tmp_input_size * tmp_number_examples * tmp_number_recurrent_depth,
                                 __LINE__);

        return(false);
    }

    this->p_ptr_array_inputs = tmp_ptr_array_inputs;
    // |END| Inputs. |END|

    // Outputs.
    T *tmp_ptr_array_outputs(MyEA::Memory::Cpp::Reallocate<T>(this->p_ptr_array_outputs,
                                                                                   tmp_number_outputs * tmp_number_examples_shifted * tmp_number_recurrent_depth,
                                                                                   tmp_number_outputs * tmp_number_examples * tmp_number_recurrent_depth));

    if(tmp_ptr_array_outputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sizeof(T),
                                 tmp_number_outputs * tmp_number_examples_shifted * tmp_number_recurrent_depth,
                                 tmp_number_outputs * tmp_number_examples * tmp_number_recurrent_depth,
                                 __LINE__);

        return(false);
    }

    this->p_ptr_array_outputs = tmp_ptr_array_outputs;
    // |END| Outputs. |END|

    this->p_number_examples = tmp_number_examples_shifted;
    
    for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
    {
        this->p_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_inputs + tmp_example_index * tmp_input_size * tmp_number_recurrent_depth;
        this->p_ptr_array_outputs_array[tmp_example_index] = tmp_ptr_array_outputs + tmp_example_index * tmp_number_outputs * tmp_number_recurrent_depth;
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Time_Direction(T const minimum_range_received,
                                                 T const maximum_range_received,
                                                 enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(minimum_range_received > maximum_range_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum range (%f) can not be greater than maximum range (%f). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_range_received),
                                 Cast_T(maximum_range_received),
                                 __LINE__);
        
        return(false);
    }
    else if(this->p_number_recurrent_depth <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be less or equal to one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_number_examples(this->p_number_examples),
                       tmp_input_size(this->p_number_inputs),
                       tmp_number_outputs(this->p_number_outputs),
                       tmp_number_recurrent_depth(this->p_number_recurrent_depth);
    size_t tmp_example_index,
              tmp_time_step_index,
              tmp_input_index;
    
    T tmp_direction;

    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            // Time step T...1.
            for(tmp_time_step_index = tmp_number_recurrent_depth; --tmp_time_step_index > 0;)
            {
                // Input N...1.
                for(tmp_input_index = tmp_input_size; --tmp_input_index > 0;)
                {
                    tmp_direction = MyEA::Math::Sign<T>(this->p_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size + tmp_input_index] - this->p_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size + tmp_input_index - 1_zu]);
                    
                    switch(static_cast<int>(tmp_direction))
                    {
                        case 1: tmp_direction = maximum_range_received; break;
                        case -1: tmp_direction = minimum_range_received; break;
                        case 0: break;
                        default:
                            PRINT_FORMAT("%s: %s: ERROR: Direction (%f) is not managed in the switch. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     Cast_T(tmp_direction),
                                                     __LINE__);
                                break;
                    }

                    this->p_ptr_array_inputs[tmp_example_index * tmp_input_size * tmp_number_recurrent_depth + tmp_time_step_index * tmp_input_size + tmp_input_index] = tmp_direction;
                }

                // First input.
                tmp_direction = MyEA::Math::Sign<T>(this->p_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size] - this->p_ptr_array_inputs_array[tmp_example_index][(tmp_time_step_index - 1_zu) * tmp_input_size]);
                
                switch(static_cast<int>(tmp_direction))
                {
                    case 1: tmp_direction = maximum_range_received; break;
                    case -1: tmp_direction = minimum_range_received; break;
                    case 0: break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Direction (%f) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 Cast_T(tmp_direction),
                                                 __LINE__);
                            break;
                }

                this->p_ptr_array_inputs[tmp_example_index * tmp_input_size * tmp_number_recurrent_depth + tmp_time_step_index * tmp_input_size] = tmp_direction;
            }

            // First time step.
            //  Input N...1.
            for(tmp_input_index = tmp_input_size; --tmp_input_index > 0;)
            {
                tmp_direction = MyEA::Math::Sign<T>(this->p_ptr_array_inputs_array[tmp_example_index][tmp_input_index] - this->p_ptr_array_inputs_array[tmp_example_index][tmp_input_index - 1_zu]);
                
                switch(static_cast<int>(tmp_direction))
                {
                    case 1: tmp_direction = maximum_range_received; break;
                    case -1: tmp_direction = minimum_range_received; break;
                    case 0: break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Direction (%f) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 Cast_T(tmp_direction),
                                                 __LINE__);
                            break;
                }

                this->p_ptr_array_inputs[tmp_example_index * tmp_input_size * tmp_number_recurrent_depth + tmp_input_index] = tmp_direction;
            }
            
            //  First input.
            this->p_ptr_array_inputs[tmp_example_index * tmp_input_size * tmp_number_recurrent_depth] = T(0);
            // |END| First time step. |END|
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            // Time step T...1.
            for(tmp_time_step_index = tmp_number_recurrent_depth; --tmp_time_step_index > 0;)
            {
                // Input N...1.
                for(tmp_input_index = tmp_number_outputs; --tmp_input_index > 0;)
                {
                    tmp_direction = MyEA::Math::Sign<T>(this->p_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * tmp_number_outputs + tmp_input_index] - this->p_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * tmp_number_outputs + tmp_input_index - 1_zu]);
                    
                    switch(static_cast<int>(tmp_direction))
                    {
                        case 1: tmp_direction = maximum_range_received; break;
                        case -1: tmp_direction = minimum_range_received; break;
                        case 0: break;
                        default:
                            PRINT_FORMAT("%s: %s: ERROR: Direction (%f) is not managed in the switch. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     Cast_T(tmp_direction),
                                                     __LINE__);
                                break;
                    }

                    this->p_ptr_array_outputs[tmp_example_index * tmp_number_outputs * tmp_number_recurrent_depth + tmp_time_step_index * tmp_number_outputs + tmp_input_index] = tmp_direction;
                }

                // First input.
                tmp_direction = MyEA::Math::Sign<T>(this->p_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * tmp_number_outputs] - this->p_ptr_array_outputs_array[tmp_example_index][(tmp_time_step_index - 1_zu) * tmp_number_outputs]);
                
                switch(static_cast<int>(tmp_direction))
                {
                    case 1: tmp_direction = maximum_range_received; break;
                    case -1: tmp_direction = minimum_range_received; break;
                    case 0: break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Direction (%f) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 Cast_T(tmp_direction),
                                                 __LINE__);
                            break;
                }

                this->p_ptr_array_outputs[tmp_example_index * tmp_number_outputs * tmp_number_recurrent_depth + tmp_time_step_index * tmp_number_outputs] = tmp_direction;
            }

            // First time step.
            //  Input N...1.
            for(tmp_input_index = tmp_number_outputs; --tmp_input_index > 0;)
            {
                tmp_direction = MyEA::Math::Sign<T>(this->p_ptr_array_outputs_array[tmp_example_index][tmp_input_index] - this->p_ptr_array_outputs_array[tmp_example_index][tmp_input_index - 1_zu]);
                
                switch(static_cast<int>(tmp_direction))
                {
                    case 1: tmp_direction = maximum_range_received; break;
                    case -1: tmp_direction = minimum_range_received; break;
                    case 0: break;
                    default:
                        PRINT_FORMAT("%s: %s: ERROR: Direction (%f) is not managed in the switch. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 Cast_T(tmp_direction),
                                                 __LINE__);
                            break;
                }

                this->p_ptr_array_outputs[tmp_example_index * tmp_number_outputs * tmp_number_recurrent_depth + tmp_input_index] = tmp_direction;
            }
            
            //  First input.
            this->p_ptr_array_outputs[tmp_example_index * tmp_number_outputs * tmp_number_recurrent_depth] = T(0);
            // |END| First time step. |END|
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Input_To_Output(enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }

    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        size_t const tmp_input_size(this->p_number_outputs);
        size_t tmp_time_step_index,
                  tmp_example_index,
                  tmp_index;

        T *tmp_ptr_array_outputs(MyEA::Memory::Cpp::Reallocate<T, false>(
            this->p_ptr_array_inputs,
            tmp_input_size * this->p_number_examples * this->p_number_recurrent_depth,
            this->p_number_inputs * this->p_number_examples * this->p_number_recurrent_depth
        ));

        if(tmp_ptr_array_outputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T),
                                     tmp_input_size * this->p_number_examples * this->p_number_recurrent_depth,
                                     this->p_number_inputs * this->p_number_examples * this->p_number_recurrent_depth,
                                     __LINE__);

            return(false);
        }

        this->p_ptr_array_inputs = tmp_ptr_array_outputs;

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        { this->p_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_outputs + tmp_example_index * tmp_input_size * this->p_number_recurrent_depth; }

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_index = 0_zu; tmp_index != tmp_input_size; ++tmp_index)
                {
                    *tmp_ptr_array_outputs++ = this->p_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size + tmp_index];
                }
            }
        }
    
        this->p_number_inputs = tmp_input_size;
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        size_t const tmp_input_size(this->p_number_inputs);
        size_t tmp_time_step_index,
                  tmp_example_index,
                  tmp_index;

        T *tmp_ptr_array_outputs(MyEA::Memory::Cpp::Reallocate<T, false>(
            this->p_ptr_array_outputs,
            tmp_input_size * this->p_number_examples * this->p_number_recurrent_depth,
            this->p_number_outputs * this->p_number_examples * this->p_number_recurrent_depth
        ));

        if(tmp_ptr_array_outputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T),
                                     tmp_input_size * this->p_number_examples * this->p_number_recurrent_depth,
                                     this->p_number_outputs * this->p_number_examples * this->p_number_recurrent_depth,
                                     __LINE__);

            return(false);
        }

        this->p_ptr_array_outputs = tmp_ptr_array_outputs;

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        { this->p_ptr_array_outputs_array[tmp_example_index] = tmp_ptr_array_outputs + tmp_example_index * tmp_input_size * this->p_number_recurrent_depth; }

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_index = 0_zu; tmp_index != tmp_input_size; ++tmp_index)
                {
                    *tmp_ptr_array_outputs++ = this->p_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_input_size + tmp_index];
                }
            }
        }
    
        this->p_number_outputs = tmp_input_size;
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Unrecurrent(void)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->p_number_recurrent_depth <= 1_zu) { return(true); }

    size_t const tmp_last_time_step_index(this->p_number_recurrent_depth - 1_zu);
    size_t tmp_example_index,
              tmp_index;

    T *tmp_ptr_array_inputs,
       *tmp_ptr_array_outputs;

    if((tmp_ptr_array_inputs = new T[this->p_number_examples * this->p_number_inputs]) == nullptr)
    {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_examples * this->p_number_inputs * sizeof(T),
                                     __LINE__);

        return(false);
    }
    else if((tmp_ptr_array_outputs = new T[this->p_number_examples * this->p_number_outputs]) == nullptr)
    {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_examples * this->p_number_outputs * sizeof(T),
                                     __LINE__);

        return(false);
    }

    for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
    {
        for(tmp_index = 0_zu; tmp_index != this->p_number_inputs; ++tmp_index)
        {
            tmp_ptr_array_inputs[tmp_example_index * this->p_number_inputs + tmp_index] = this->p_ptr_array_inputs_array[tmp_example_index][tmp_last_time_step_index * this->p_number_inputs + tmp_index];
        }

        for(tmp_index = 0_zu; tmp_index != this->p_number_outputs; ++tmp_index)
        {
            tmp_ptr_array_outputs[tmp_example_index * this->p_number_outputs + tmp_index] = this->p_ptr_array_outputs_array[tmp_example_index][tmp_last_time_step_index * this->p_number_outputs + tmp_index];
        }
    }
    
    delete[](this->p_ptr_array_inputs);
    this->p_ptr_array_inputs = tmp_ptr_array_inputs;

    delete[](this->p_ptr_array_outputs);
    this->p_ptr_array_outputs = tmp_ptr_array_outputs;

    for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
    {
        this->p_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_inputs + tmp_example_index * this->p_number_inputs;

        this->p_ptr_array_outputs_array[tmp_example_index] = tmp_ptr_array_outputs + tmp_example_index * this->p_number_outputs;
    }
    
    this->p_number_recurrent_depth = 1_zu;

    return(true);
}

template<typename T>
bool Dataset<T>::Remove(size_t const input_index_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(input_index_received >= this->p_number_inputs)
        {
            PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     input_index_received,
                                     this->p_number_inputs,
                                     __LINE__);
        
            return(false);
        }
        
        size_t const tmp_new_input_size(this->p_number_inputs - 1_zu);
        size_t tmp_example_index,
                  tmp_time_step_index,
                  tmp_shifted_index,
                  tmp_index;

        T *tmp_ptr_array_inputs;

        if((tmp_ptr_array_inputs = new T[this->p_number_examples * this->p_number_recurrent_depth * tmp_new_input_size]) == nullptr)
        {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_examples * this->p_number_recurrent_depth * tmp_new_input_size * sizeof(T),
                                         __LINE__);

            return(false);
        }

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                // Left.
                for(tmp_index = 0_zu; tmp_index != input_index_received; ++tmp_index)
                {
                    tmp_ptr_array_inputs[tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_new_input_size + tmp_index] = this->p_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * this->p_number_inputs + tmp_index];
                }

                // Right.
                for(tmp_shifted_index = input_index_received,
                    tmp_index = input_index_received + 1_zu; tmp_index != this->p_number_inputs; ++tmp_index)
                {
                    tmp_ptr_array_inputs[tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_new_input_size + tmp_shifted_index] = this->p_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * this->p_number_inputs + tmp_index];
                }
            }
        }
        
        delete[](this->p_ptr_array_inputs);
        this->p_ptr_array_inputs = tmp_ptr_array_inputs;

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            this->p_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_inputs + tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth;
        }
        
        --this->p_number_inputs;
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(input_index_received >= this->p_number_outputs)
        {
            PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     input_index_received,
                                     this->p_number_outputs,
                                     __LINE__);
        
            return(false);
        }
        
        size_t const tmp_new_input_size(this->p_number_outputs - 1_zu);
        size_t tmp_example_index,
                  tmp_time_step_index,
                  tmp_shifted_index,
                  tmp_index;

        T *tmp_ptr_array_outputs;
        
        if((tmp_ptr_array_outputs = new T[this->p_number_examples * this->p_number_recurrent_depth * tmp_new_input_size]) == nullptr)
        {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_examples * this->p_number_recurrent_depth * tmp_new_input_size * sizeof(T),
                                         __LINE__);

            return(false);
        }

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                // Left.
                for(tmp_index = 0_zu; tmp_index != input_index_received; ++tmp_index)
                {
                    tmp_ptr_array_outputs[tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_new_input_size + tmp_index] = this->p_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + tmp_index];
                }

                // Right.
                for(tmp_shifted_index = input_index_received,
                    tmp_index = input_index_received + 1_zu; tmp_index != this->p_number_outputs; ++tmp_index)
                {
                    tmp_ptr_array_outputs[tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_new_input_size + tmp_shifted_index] = this->p_ptr_array_outputs_array[tmp_example_index][tmp_time_step_index * this->p_number_outputs + tmp_index];
                }
            }
        }
        
        delete[](this->p_ptr_array_outputs);
        this->p_ptr_array_outputs = tmp_ptr_array_outputs;

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            this->p_ptr_array_outputs_array[tmp_example_index] = tmp_ptr_array_outputs + tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth;
        }
        
        --this->p_number_outputs;
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Allocate(enum MyEA::Common::ENUM_TYPE_DATASET_FILE const type_data_file_read_received, std::string const &ref_path_received)
{
    switch(type_data_file_read_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_DATASET:
            if(this->Allocate__Dataset(ref_path_received + ".dataset") == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset(%s + \".dataset\")\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         ref_path_received.c_str(),
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_DATASET_SPLIT:
            if(this->Allocate__Dataset_Split(ref_path_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset_Split(%s)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         ref_path_received.c_str(),
                                         __LINE__);

                return(false);
            }
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_MNIST:
            if(this->Allocate__MNIST(ref_path_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__MNIST(%s)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         ref_path_received.c_str(),
                                         __LINE__);

                return(false);
            }
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dataset file type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     type_data_file_read_received,
                                     MyEA::Common::ENUM_TYPE_DATASET_FILE_NAMES[type_data_file_read_received].c_str(),
                                     __LINE__);
                return(false);
    }

    this->p_type_data_file = type_data_file_read_received;

    this->Check_Use__Label();

    this->_reference = false;

    return(true);
}

template<typename T>
void Dataset<T>::Check_Use__Label(void)
{
    if(this->Get__Number_Outputs() == 1_zu) { this->_use_multi_label = false; }
    else
    {
        size_t tmp_numbre_labels(0_zu);

        for(size_t tmp_output_index(0_zu); tmp_output_index != this->p_number_outputs; ++tmp_output_index)
        {
            if(this->p_ptr_array_outputs_array[0u][tmp_output_index] != T(0)) { ++tmp_numbre_labels; }
        }

        if(tmp_numbre_labels > 1_zu) { this->_use_multi_label = true; }
    }
}

template<typename T>
void Dataset<T>::Compute__Start_Index(void)
{
    size_t tmp_start_index(0_zu);
    
    if(this->p_ptr_input_array_coefficient_matrix_size != nullptr)
    {
        size_t tmp_J_level,
                  tmp_j_index,
                  tmp_input_index,
                  tmp_circularity_end;

        for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_inputs; ++tmp_input_index)
        {
            tmp_J_level = this->p_ptr_input_array_coefficient_matrix_size[tmp_input_index] / this->p_number_examples;

            if(tmp_J_level > 1_zu)
            {
                --tmp_J_level;

                tmp_circularity_end = 1_zu;

                for(tmp_j_index = 1_zu; tmp_j_index != tmp_J_level; ++tmp_j_index) { tmp_circularity_end = 2_zu * tmp_circularity_end + 1_zu; }

                tmp_start_index = MyEA::Math::Maximum<size_t>(tmp_start_index, tmp_circularity_end);
            }
        }
    }

    if(this->p_ptr_output_array_coefficient_matrix_size != nullptr)
    {
        size_t tmp_J_level,
                  tmp_j_index,
                  tmp_input_index,
                  tmp_circularity_end;

        for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_outputs; ++tmp_input_index)
        {
            tmp_J_level = this->p_ptr_output_array_coefficient_matrix_size[tmp_input_index] / this->p_number_examples;

            if(tmp_J_level > 1_zu)
            {
                --tmp_J_level;

                tmp_circularity_end = 1_zu;

                for(tmp_j_index = 1_zu; tmp_j_index != tmp_J_level; ++tmp_j_index) { tmp_circularity_end = 2_zu * tmp_circularity_end + 1_zu; }

                tmp_start_index = MyEA::Math::Maximum<size_t>(tmp_start_index, tmp_circularity_end);
            }
        }
    }

    this->p_start_index = tmp_start_index;
}

template<typename T>
bool Dataset<T>::Set__Type_Data_File(enum MyEA::Common::ENUM_TYPE_DATASET_FILE const type_dataset_file_received)
{
    if(type_dataset_file_received >= MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: Dataset file type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_dataset_file_received,
                                 MyEA::Common::ENUM_TYPE_DATASET_FILE_NAMES[type_dataset_file_received].c_str(),
                                 __LINE__);

        return(false);
    }

    this->p_type_data_file = type_dataset_file_received;

    return(true);
}

template<typename T>
bool Dataset<T>::Allocate__Dataset(std::string const &ref_path_received)
{
    char *tmp_ptr_array_buffers(nullptr),
           *tmp_ptr_last_buffer(nullptr);

    size_t tmp_block_size,
              tmp_number_examples,
              tmp_input_size,
              tmp_number_outputs,
              tmp_number_recurrent_depth,
              tmp_index,
              tmp_example_index,
              tmp_time_step_index;

    T const **tmp_ptr_array_inputs_array(nullptr),
                **tmp_ptr_array_outputs_array(nullptr);
    T *tmp_ptr_array_inputs,
       *tmp_ptr_array_outputs;
    
    double tmp_output;

    std::vector<char> tmp_vector_buffers;

    std::ifstream tmp_ifstream(ref_path_received, std::ios::in | std::ios::binary);
    
    if(tmp_ifstream.is_open())
    {
        if(tmp_ifstream.eof())
        {
            PRINT_FORMAT("%s: %s: ERROR: File \"%s\" is empty. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }
        
        if(MyEA::String::Read_Stream_Block_And_Parse_Number<size_t>(tmp_ptr_array_buffers,
                                                                                                          tmp_ptr_last_buffer,
                                                                                                          tmp_block_size,
                                                                                                          this->p_file_buffer_size,
                                                                                                          this->p_file_buffer_shift_size,
                                                                                                          tmp_number_examples,
                                                                                                          tmp_vector_buffers,
                                                                                                          tmp_ifstream,
                                                                                                          '\n') == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading the number of examples. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_block_size,
                                     this->p_file_buffer_size,
                                     this->p_file_buffer_shift_size,
                                     __LINE__);

            return(false);
        }
        else if(MyEA::String::Read_Stream_Block_And_Parse_Number<size_t>(tmp_ptr_array_buffers,
                                                                                                                 tmp_ptr_last_buffer,
                                                                                                                 tmp_block_size,
                                                                                                                 this->p_file_buffer_size,
                                                                                                                 this->p_file_buffer_shift_size,
                                                                                                                 tmp_input_size,
                                                                                                                 tmp_vector_buffers,
                                                                                                                 tmp_ifstream,
                                                                                                                 '\n') == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading the number of inputs. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_block_size,
                                     this->p_file_buffer_size,
                                     this->p_file_buffer_shift_size,
                                     __LINE__);

            return(false);
        }
        else if(MyEA::String::Read_Stream_Block_And_Parse_Number<size_t>(tmp_ptr_array_buffers,
                                                                                                                 tmp_ptr_last_buffer,
                                                                                                                 tmp_block_size,
                                                                                                                 this->p_file_buffer_size,
                                                                                                                 this->p_file_buffer_shift_size,
                                                                                                                 tmp_number_outputs,
                                                                                                                 tmp_vector_buffers,
                                                                                                                 tmp_ifstream,
                                                                                                                 '\n') == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading the number of outputs. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_block_size,
                                     this->p_file_buffer_size,
                                     this->p_file_buffer_shift_size,
                                     __LINE__);

            return(false);
        }
        else if(MyEA::String::Read_Stream_Block_And_Parse_Number<size_t>(tmp_ptr_array_buffers,
                                                                                                                 tmp_ptr_last_buffer,
                                                                                                                 tmp_block_size,
                                                                                                                 this->p_file_buffer_size,
                                                                                                                 this->p_file_buffer_shift_size,
                                                                                                                 tmp_number_recurrent_depth,
                                                                                                                 tmp_vector_buffers,
                                                                                                                 tmp_ifstream,
                                                                                                                 '\n') == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading the number of recurrent depth. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_block_size,
                                     this->p_file_buffer_size,
                                     this->p_file_buffer_shift_size,
                                     __LINE__);

            return(false);
        }

        if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not read topology. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        tmp_ptr_array_inputs_array = new T const *[tmp_number_examples];
        if(tmp_ptr_array_inputs_array == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_number_examples * sizeof(T const *),
                                     __LINE__);

            return(false);
        }

        tmp_ptr_array_outputs_array = new T const *[tmp_number_examples];
        if(tmp_ptr_array_outputs_array == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_number_examples * sizeof(T const *),
                                     __LINE__);

            delete[](tmp_ptr_array_inputs_array);

            return(false);
        }

        tmp_ptr_array_inputs = new T[tmp_input_size * tmp_number_examples * tmp_number_recurrent_depth];
        if(tmp_ptr_array_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_size * tmp_number_examples * sizeof(T),
                                     __LINE__);

            delete[](tmp_ptr_array_inputs_array);
            delete[](tmp_ptr_array_outputs_array);

            return(false);
        }

        tmp_ptr_array_outputs = new T[tmp_number_outputs * tmp_number_examples * tmp_number_recurrent_depth];
        if(tmp_ptr_array_outputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_number_outputs * tmp_number_examples * sizeof(T),
                                     __LINE__);

            delete[](tmp_ptr_array_inputs_array);
            delete[](tmp_ptr_array_outputs_array);
            delete[](tmp_ptr_array_inputs);

            return(false);
        }

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            tmp_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_inputs + tmp_example_index * tmp_input_size * tmp_number_recurrent_depth;

            tmp_ptr_array_outputs_array[tmp_example_index] = tmp_ptr_array_outputs + tmp_example_index * tmp_number_outputs * tmp_number_recurrent_depth;
        }

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != tmp_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_index = 0_zu; tmp_index != tmp_input_size; ++tmp_index,
                                                                                                   ++tmp_ptr_array_inputs)
                {
                    if(MyEA::String::Read_Stream_Block_And_Parse_Number<double>(tmp_ptr_array_buffers,
                                                                                                                               tmp_ptr_last_buffer,
                                                                                                                               tmp_block_size,
                                                                                                                               this->p_file_buffer_size,
                                                                                                                               this->p_file_buffer_shift_size,
                                                                                                                               tmp_output,
                                                                                                                               tmp_vector_buffers,
                                                                                                                               tmp_ifstream,
                                                                                                                               '\n') == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading data %zu at %zu input. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_block_size,
                                                 this->p_file_buffer_size,
                                                 this->p_file_buffer_shift_size,
                                                 tmp_example_index,
                                                 tmp_index,
                                                 __LINE__);
                        
                        delete[](tmp_ptr_array_inputs_array[0u]);
                        delete[](tmp_ptr_array_inputs_array);
                        delete[](tmp_ptr_array_outputs_array[0u]);
                        delete[](tmp_ptr_array_outputs_array);

                        return(false);
                    }

                    *tmp_ptr_array_inputs = static_cast<T>(tmp_output);
                }

                for(tmp_index = 0_zu; tmp_index != tmp_number_outputs; ++tmp_index,
                                                                                                     ++tmp_ptr_array_outputs)
                {
                    if(MyEA::String::Read_Stream_Block_And_Parse_Number<double>(tmp_ptr_array_buffers,
                                                                                                                               tmp_ptr_last_buffer,
                                                                                                                               tmp_block_size,
                                                                                                                               this->p_file_buffer_size,
                                                                                                                               this->p_file_buffer_shift_size,
                                                                                                                               tmp_output,
                                                                                                                               tmp_vector_buffers,
                                                                                                                               tmp_ifstream,
                                                                                                                               '\n') == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading data %zu at %zu output. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_block_size,
                                                 this->p_file_buffer_size,
                                                 this->p_file_buffer_shift_size,
                                                 tmp_example_index,
                                                 tmp_index,
                                                 __LINE__);
                        
                        delete[](tmp_ptr_array_inputs_array[0u]);
                        delete[](tmp_ptr_array_inputs_array);
                        delete[](tmp_ptr_array_outputs_array[0u]);
                        delete[](tmp_ptr_array_outputs_array);

                        return(false);
                    }

                    *tmp_ptr_array_outputs = static_cast<T>(tmp_output);
                }
            }
        }
        
        if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Logical error on i/o operation \"%s\". At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }

        tmp_ifstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    this->p_number_examples_allocated = this->p_number_examples = tmp_number_examples;
    this->p_number_inputs = tmp_input_size;
    this->p_number_outputs = tmp_number_outputs;
    this->p_number_recurrent_depth = tmp_number_recurrent_depth;

    this->p_ptr_array_inputs_array = tmp_ptr_array_inputs_array;
    this->p_ptr_array_inputs = tmp_ptr_array_inputs - tmp_number_examples * tmp_input_size * tmp_number_recurrent_depth;
    this->p_ptr_array_outputs_array = tmp_ptr_array_outputs_array;
    this->p_ptr_array_outputs = tmp_ptr_array_outputs - tmp_number_examples * tmp_number_outputs * tmp_number_recurrent_depth;

    return(true);
}

template<typename T>
bool Dataset<T>::Allocate__Dataset_Split(std::string const &ref_path_received)
{
    if(this->Allocate__Dataset_Split__Input(ref_path_received + ".dataset-input") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset_Split__Input(%s.dataset-input)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(this->Allocate__Dataset_Split__Output(ref_path_received + ".dataset-output") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Dataset_Split__Output(%s.dataset-output)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Allocate__Dataset_Split__Input(std::string const &ref_path_received)
{
    char *tmp_ptr_array_buffers(nullptr),
           *tmp_ptr_last_buffer(nullptr);

    size_t tmp_block_size,
              tmp_number_examples,
              tmp_input_size,
              tmp_number_recurrent_depth,
              tmp_index,
              tmp_example_index,
              tmp_time_step_index;

    T const **tmp_ptr_array_inputs_array(nullptr);
    T *tmp_ptr_array_inputs;
    
    double tmp_output;

    std::vector<char> tmp_vector_buffers;

    std::ifstream tmp_ifstream(ref_path_received, std::ios::in | std::ios::binary);
    
    if(tmp_ifstream.is_open())
    {
        if(tmp_ifstream.eof())
        {
            PRINT_FORMAT("%s: %s: ERROR: File \"%s\" is empty. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }
        
        if(MyEA::String::Read_Stream_Block_And_Parse_Number<size_t>(tmp_ptr_array_buffers,
                                                                                                          tmp_ptr_last_buffer,
                                                                                                          tmp_block_size,
                                                                                                          this->p_file_buffer_size,
                                                                                                          this->p_file_buffer_shift_size,
                                                                                                          tmp_number_examples,
                                                                                                          tmp_vector_buffers,
                                                                                                          tmp_ifstream,
                                                                                                          '\n') == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading the number of examples. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_block_size,
                                     this->p_file_buffer_size,
                                     this->p_file_buffer_shift_size,
                                     __LINE__);

            return(false);
        }
        else if(MyEA::String::Read_Stream_Block_And_Parse_Number<size_t>(tmp_ptr_array_buffers,
                                                                                                                 tmp_ptr_last_buffer,
                                                                                                                 tmp_block_size,
                                                                                                                 this->p_file_buffer_size,
                                                                                                                 this->p_file_buffer_shift_size,
                                                                                                                 tmp_input_size,
                                                                                                                 tmp_vector_buffers,
                                                                                                                 tmp_ifstream,
                                                                                                                 '\n') == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading the number of inputs. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_block_size,
                                     this->p_file_buffer_size,
                                     this->p_file_buffer_shift_size,
                                     __LINE__);

            return(false);
        }
        else if(MyEA::String::Read_Stream_Block_And_Parse_Number<size_t>(tmp_ptr_array_buffers,
                                                                                                                 tmp_ptr_last_buffer,
                                                                                                                 tmp_block_size,
                                                                                                                 this->p_file_buffer_size,
                                                                                                                 this->p_file_buffer_shift_size,
                                                                                                                 tmp_number_recurrent_depth,
                                                                                                                 tmp_vector_buffers,
                                                                                                                 tmp_ifstream,
                                                                                                                 '\n') == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading the number of recurrent depth. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_block_size,
                                     this->p_file_buffer_size,
                                     this->p_file_buffer_shift_size,
                                     __LINE__);

            return(false);
        }

        tmp_ptr_array_inputs_array = new T const *[tmp_number_examples];
        if(tmp_ptr_array_inputs_array == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_number_examples * sizeof(T const *),
                                     __LINE__);

            return(false);
        }

        tmp_ptr_array_inputs = new T[tmp_input_size * tmp_number_examples * tmp_number_recurrent_depth];
        if(tmp_ptr_array_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_size * tmp_number_examples * sizeof(T),
                                     __LINE__);

            delete[](tmp_ptr_array_inputs_array);

            return(false);
        }

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        { tmp_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_inputs + tmp_example_index * tmp_input_size * tmp_number_recurrent_depth; }
        
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != tmp_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_index = 0_zu; tmp_index != tmp_input_size; ++tmp_index,
                                                                                                   ++tmp_ptr_array_inputs)
                {
                    if(MyEA::String::Read_Stream_Block_And_Parse_Number<double>(tmp_ptr_array_buffers,
                                                                                                                               tmp_ptr_last_buffer,
                                                                                                                               tmp_block_size,
                                                                                                                               this->p_file_buffer_size,
                                                                                                                               this->p_file_buffer_shift_size,
                                                                                                                               tmp_output,
                                                                                                                               tmp_vector_buffers,
                                                                                                                               tmp_ifstream,
                                                                                                                               '\n') == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading data %zu at %zu input. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_block_size,
                                                 this->p_file_buffer_size,
                                                 this->p_file_buffer_shift_size,
                                                 tmp_example_index,
                                                 tmp_index,
                                                 __LINE__);
                        
                        delete[](tmp_ptr_array_inputs_array[0u]);
                        delete[](tmp_ptr_array_inputs_array);

                        return(false);
                    }

                    *tmp_ptr_array_inputs = static_cast<T>(tmp_output);
                }
            }
        }
        
        if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Logical error on i/o operation \"%s\". At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }

        tmp_ifstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    this->p_number_examples_allocated = this->p_number_examples = tmp_number_examples;
    this->p_number_inputs = tmp_input_size;
    this->p_number_recurrent_depth = tmp_number_recurrent_depth;

    this->p_ptr_array_inputs_array = tmp_ptr_array_inputs_array;
    this->p_ptr_array_inputs = tmp_ptr_array_inputs - tmp_number_examples * tmp_input_size * tmp_number_recurrent_depth;

    return(true);
}

template<typename T>
bool Dataset<T>::Allocate__Dataset_Split__Output(std::string const &ref_path_received)
{
    char *tmp_ptr_array_buffers(nullptr),
           *tmp_ptr_last_buffer(nullptr);

    size_t tmp_block_size,
              tmp_number_examples,
              tmp_number_outputs,
              tmp_number_recurrent_depth,
              tmp_index,
              tmp_example_index,
              tmp_time_step_index;

    T const **tmp_ptr_array_outputs_array(nullptr);
    T *tmp_ptr_array_outputs;
    
    double tmp_output;

    std::vector<char> tmp_vector_buffers;

    std::ifstream tmp_ifstream(ref_path_received, std::ios::in | std::ios::binary);
    
    if(tmp_ifstream.is_open())
    {
        if(tmp_ifstream.eof())
        {
            PRINT_FORMAT("%s: %s: ERROR: File \"%s\" is empty. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }
        
        if(MyEA::String::Read_Stream_Block_And_Parse_Number<size_t>(tmp_ptr_array_buffers,
                                                                                                          tmp_ptr_last_buffer,
                                                                                                          tmp_block_size,
                                                                                                          this->p_file_buffer_size,
                                                                                                          this->p_file_buffer_shift_size,
                                                                                                          tmp_number_examples,
                                                                                                          tmp_vector_buffers,
                                                                                                          tmp_ifstream,
                                                                                                          '\n') == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading the number of examples. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_block_size,
                                     this->p_file_buffer_size,
                                     this->p_file_buffer_shift_size,
                                     __LINE__);

            return(false);
        }
        else if(MyEA::String::Read_Stream_Block_And_Parse_Number<size_t>(tmp_ptr_array_buffers,
                                                                                                                 tmp_ptr_last_buffer,
                                                                                                                 tmp_block_size,
                                                                                                                 this->p_file_buffer_size,
                                                                                                                 this->p_file_buffer_shift_size,
                                                                                                                 tmp_number_outputs,
                                                                                                                 tmp_vector_buffers,
                                                                                                                 tmp_ifstream,
                                                                                                                 '\n') == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading the number of outputs. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_block_size,
                                     this->p_file_buffer_size,
                                     this->p_file_buffer_shift_size,
                                     __LINE__);

            return(false);
        }
        else if(MyEA::String::Read_Stream_Block_And_Parse_Number<size_t>(tmp_ptr_array_buffers,
                                                                                                                 tmp_ptr_last_buffer,
                                                                                                                 tmp_block_size,
                                                                                                                 this->p_file_buffer_size,
                                                                                                                 this->p_file_buffer_shift_size,
                                                                                                                 tmp_number_recurrent_depth,
                                                                                                                 tmp_vector_buffers,
                                                                                                                 tmp_ifstream,
                                                                                                                 '\n') == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading the number of recurrent depth. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_block_size,
                                     this->p_file_buffer_size,
                                     this->p_file_buffer_shift_size,
                                     __LINE__);

            return(false);
        }

        tmp_ptr_array_outputs_array = new T const *[tmp_number_examples];
        if(tmp_ptr_array_outputs_array == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_number_examples * sizeof(T const *),
                                     __LINE__);

            return(false);
        }

        tmp_ptr_array_outputs = new T[tmp_number_outputs * tmp_number_examples * tmp_number_recurrent_depth];
        if(tmp_ptr_array_outputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_number_outputs * tmp_number_examples * sizeof(T),
                                     __LINE__);

            delete[](tmp_ptr_array_outputs_array);

            return(false);
        }

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        { tmp_ptr_array_outputs_array[tmp_example_index] = tmp_ptr_array_outputs + tmp_example_index * tmp_number_outputs * tmp_number_recurrent_depth; }

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != tmp_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_index = 0_zu; tmp_index != tmp_number_outputs; ++tmp_index,
                                                                                                     ++tmp_ptr_array_outputs)
                {
                    if(MyEA::String::Read_Stream_Block_And_Parse_Number<double>(tmp_ptr_array_buffers,
                                                                                                                               tmp_ptr_last_buffer,
                                                                                                                               tmp_block_size,
                                                                                                                               this->p_file_buffer_size,
                                                                                                                               this->p_file_buffer_shift_size,
                                                                                                                               tmp_output,
                                                                                                                               tmp_vector_buffers,
                                                                                                                               tmp_ifstream,
                                                                                                                               '\n') == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Read_Stream_Block_And_Parse_Number(ptr, ptr, %zu, %zu, %zu, ptr, vector, ifstream, '\\n')\" function, while reading data %zu at %zu output. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 tmp_block_size,
                                                 this->p_file_buffer_size,
                                                 this->p_file_buffer_shift_size,
                                                 tmp_example_index,
                                                 tmp_index,
                                                 __LINE__);
                        
                        delete[](tmp_ptr_array_outputs_array[0u]);
                        delete[](tmp_ptr_array_outputs_array);

                        return(false);
                    }

                    *tmp_ptr_array_outputs = static_cast<T>(tmp_output);
                }
            }
        }
        
        if(tmp_ifstream.fail())
        {
            PRINT_FORMAT("%s: %s: ERROR: Logical error on i/o operation \"%s\". At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ref_path_received.c_str(),
                                     __LINE__);

            return(false);
        }

        tmp_ifstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    this->p_number_examples_allocated = this->p_number_examples = tmp_number_examples;
    this->p_number_outputs = tmp_number_outputs;
    this->p_number_recurrent_depth = tmp_number_recurrent_depth;

    this->p_ptr_array_outputs_array = tmp_ptr_array_outputs_array;
    this->p_ptr_array_outputs = tmp_ptr_array_outputs - tmp_number_examples * tmp_number_outputs * tmp_number_recurrent_depth;

    return(true);
}

template<typename T>
bool Dataset<T>::Allocate__MNIST(std::string const &ref_path_received)
{
    int tmp_input_size,
         tmp_number_outputs,
         tmp_number_images,
         tmp_number_labels,
         tmp_number_rows,
         tmp_number_columns,
         tmp_index,
         tmp_example_index,
         tmp_magic_number;

    unsigned char tmp_input;

    T const **tmp_ptr_array_inputs_array(nullptr),
                **tmp_ptr_array_outputs_array(nullptr);
    T *tmp_ptr_array_inputs,
       *tmp_ptr_array_outputs;
    
    std::string const tmp_path_images(ref_path_received + ".idx3-ubyte"),
                            tmp_path_label(ref_path_received + ".idx1-ubyte");

    if(MyEA::File::Path_Exist(tmp_path_images) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: Could not find the following path \"%s\". At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_path_images.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(MyEA::File::Path_Exist(tmp_path_label) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: Could not find the following path \"%s\". At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_path_label.c_str(),
                                 __LINE__);

        return(false);
    }

    std::ifstream tmp_ifstream_images(tmp_path_images, std::ios::in | std::ios::binary),
                       tmp_ifstream_labels(tmp_path_label, std::ios::in | std::ios::binary);
    
    if(tmp_ifstream_images.is_open() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_path_images.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_ifstream_images.eof())
    {
        PRINT_FORMAT("%s: %s: ERROR: File \"%s\" is empty. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_path_images.c_str(),
                                 __LINE__);

        return(false);
    }

    if(tmp_ifstream_labels.is_open() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_path_label.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_ifstream_labels.eof())
    {
        PRINT_FORMAT("%s: %s: ERROR: File \"%s\" is empty. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_path_label.c_str(),
                                 __LINE__);

        return(false);
    }

    // MNIST image file.
    tmp_ifstream_images.read((char *)&tmp_magic_number, sizeof(int));
    tmp_magic_number = MyEA::Math::Reverse_Integer<int>(tmp_magic_number);

    if(tmp_magic_number != 2051)
    {
        PRINT_FORMAT("%s: %s: ERROR: Invalid MNIST image file! Magic number equal %d. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_magic_number,
                                 __LINE__);

        return(false);
    }
    
    tmp_ifstream_images.read((char *)&tmp_number_images, sizeof(int));
    tmp_number_images = MyEA::Math::Reverse_Integer<int>(tmp_number_images);
    
    tmp_ifstream_images.read((char *)&tmp_number_rows, sizeof(int));
    tmp_number_rows = MyEA::Math::Reverse_Integer<int>(tmp_number_rows);
    
    tmp_ifstream_images.read((char *)&tmp_number_columns, sizeof(int));
    tmp_number_columns = MyEA::Math::Reverse_Integer<int>(tmp_number_columns);

    tmp_input_size = tmp_number_rows * tmp_number_columns;

    tmp_ptr_array_inputs_array = new T const *[tmp_number_images];
    if(tmp_ptr_array_inputs_array == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 static_cast<size_t>(tmp_number_images) * sizeof(T const *),
                                 __LINE__);

        return(false);
    }

    tmp_ptr_array_inputs = new T[tmp_number_images * tmp_input_size];
    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 static_cast<size_t>(tmp_number_images * tmp_input_size) * sizeof(T),
                                 __LINE__);

        delete[](tmp_ptr_array_inputs_array);

        return(false);
    }

    for(tmp_example_index = 0; tmp_example_index != tmp_number_images; ++tmp_example_index)
    {
        tmp_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_inputs;

        for(tmp_index = 0; tmp_index != tmp_input_size; ++tmp_index,
                                                                                      ++tmp_ptr_array_inputs)
        {
            tmp_ifstream_images.read((char *)&tmp_input, sizeof(unsigned char));

            *tmp_ptr_array_inputs = static_cast<T>(tmp_input) / static_cast<T>(255.0);
        }
    }

    tmp_ifstream_images.close();
    // |END| MNIST image file. |END|

    // MNIST label file.
    tmp_ifstream_labels.read((char *)&tmp_magic_number, sizeof(int));
    tmp_magic_number = MyEA::Math::Reverse_Integer<int>(tmp_magic_number);

    if(tmp_magic_number != 2049)
    {
        PRINT_FORMAT("%s: %s: ERROR: Invalid MNIST image file! Magic number equal %d. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_magic_number,
                                 __LINE__);

        delete[](tmp_ptr_array_inputs_array[0u]);
        delete[](tmp_ptr_array_inputs_array);

        return(false);
    }
    
    tmp_ifstream_labels.read((char *)&tmp_number_labels, sizeof(int));
    tmp_number_labels = MyEA::Math::Reverse_Integer<int>(tmp_number_labels);

    if(tmp_number_images != tmp_number_labels)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of images (%d) differs with the number of labels (%d). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_number_images,
                                 tmp_number_labels,
                                 __LINE__);

        delete[](tmp_ptr_array_inputs_array[0u]);
        delete[](tmp_ptr_array_inputs_array);

        return(false);
    }

    tmp_number_outputs = 10;

    tmp_ptr_array_outputs_array = new T const *[tmp_number_labels];
    if(tmp_ptr_array_outputs_array == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 static_cast<size_t>(tmp_number_labels) * sizeof(T const *),
                                 __LINE__);
                            
        delete[](tmp_ptr_array_inputs_array[0u]);
        delete[](tmp_ptr_array_inputs_array);

        return(false);
    }

    tmp_ptr_array_outputs = new T[tmp_number_labels * tmp_number_outputs];
    if(tmp_ptr_array_outputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 static_cast<size_t>(tmp_number_labels * tmp_number_outputs) * sizeof(T),
                                 __LINE__);
                            
        delete[](tmp_ptr_array_inputs_array[0u]);
        delete[](tmp_ptr_array_inputs_array);
        delete[](tmp_ptr_array_outputs_array);

        return(false);
    }

    MEMSET(tmp_ptr_array_outputs,
                 0,
                 static_cast<size_t>(tmp_number_labels * tmp_number_outputs) * sizeof(T));

    for(tmp_example_index = 0; tmp_example_index != tmp_number_labels; ++tmp_example_index)
    {
        tmp_ptr_array_outputs_array[tmp_example_index] = tmp_ptr_array_outputs;

        tmp_ifstream_labels.read((char *)&tmp_input, sizeof(unsigned char));

        tmp_ptr_array_outputs[tmp_input] = T(1);

        tmp_ptr_array_outputs += tmp_number_outputs;
    }

    tmp_ifstream_labels.close();
    // |END| MNIST label file. |END|

    this->p_number_examples_allocated = this->p_number_examples = static_cast<size_t>(tmp_number_images);
    this->p_number_inputs = static_cast<size_t>(tmp_input_size);
    this->p_number_outputs = static_cast<size_t>(tmp_number_outputs);
    this->p_number_recurrent_depth = 1_zu;

    this->p_ptr_array_inputs_array = tmp_ptr_array_inputs_array;
    this->p_ptr_array_inputs = tmp_ptr_array_inputs - tmp_number_images * tmp_input_size;
    this->p_ptr_array_outputs_array = tmp_ptr_array_outputs_array;
    this->p_ptr_array_outputs = tmp_ptr_array_outputs - tmp_number_images * tmp_number_outputs;

    return(true);
}

template<typename T>
bool Dataset<T>::Remove_Duplicate(void)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    size_t tmp_new_number_examples(0_zu),
              tmp_example_index,
              tmp_data_check_index,
              tmp_input_index(0_zu),
              tmp_time_step_index;

    T *tmp_ptr_array_inputs,
       *tmp_ptr_array_outputs;
    
    tmp_ptr_array_inputs = new T[this->p_number_examples * this->p_number_inputs * this->p_number_recurrent_depth];
    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    this->p_number_examples * this->p_number_inputs * this->p_number_recurrent_depth * sizeof(T),
                                    __LINE__);

        return(false);
    }
    
    tmp_ptr_array_outputs = new T[this->p_number_examples * this->p_number_outputs * this->p_number_recurrent_depth];
    if(tmp_ptr_array_outputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    this->p_number_examples * this->p_number_outputs * this->p_number_recurrent_depth * sizeof(T),
                                    __LINE__);

        delete[](tmp_ptr_array_inputs);

        return(false);
    }

    for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
    {
        for(tmp_data_check_index = tmp_example_index + 1_zu; tmp_data_check_index < this->p_number_examples; ++tmp_data_check_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                // Check inputs duplications
                for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_inputs; ++tmp_input_index)
                {
                    if(this->p_ptr_array_inputs_array[tmp_example_index][tmp_input_index] != this->p_ptr_array_inputs_array[tmp_data_check_index][this->p_number_inputs * tmp_time_step_index + tmp_input_index])
                    { break; }
                }

                // If duplicate.
                if(tmp_input_index == this->p_number_inputs) { break; }
            }

            // If duplicate.
            if(tmp_input_index == this->p_number_inputs) { break; }
        }

        // If not duplicate.
        if(tmp_input_index != this->p_number_inputs)
        {
            // Store current inputs to a tempory array of inputs
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_inputs; ++tmp_input_index)
                { tmp_ptr_array_inputs[tmp_input_index] = this->p_ptr_array_inputs_array[tmp_example_index][this->p_number_inputs * tmp_time_step_index + tmp_input_index]; }
            }
            tmp_ptr_array_inputs += this->p_number_inputs * this->p_number_recurrent_depth;

            // Store current outputs to a tempory array of outputs
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_outputs; ++tmp_input_index)
                { tmp_ptr_array_outputs[tmp_input_index] = this->p_ptr_array_outputs_array[tmp_example_index][this->p_number_outputs * tmp_time_step_index + tmp_input_index]; }
            }
            tmp_ptr_array_outputs += this->p_number_outputs * this->p_number_recurrent_depth;

            // Increment nData
            ++tmp_new_number_examples;
        }
    }

    // Reset pointer position to begining
    tmp_ptr_array_inputs -= tmp_new_number_examples * this->p_number_inputs * this->p_number_recurrent_depth;
    tmp_ptr_array_outputs -= tmp_new_number_examples * this->p_number_outputs * this->p_number_recurrent_depth;

    if(this->p_number_examples != tmp_new_number_examples)
    {
        SAFE_DELETE_ARRAY(this->p_ptr_array_inputs);

        SAFE_DELETE_ARRAY(this->p_ptr_array_inputs_array);

        SAFE_DELETE_ARRAY(this->p_ptr_array_outputs);

        SAFE_DELETE_ARRAY(this->p_ptr_array_outputs_array);

        // Alloc
        this->p_ptr_array_inputs_array = new T const *[tmp_new_number_examples];
        if(this->p_ptr_array_inputs_array == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_new_number_examples * sizeof(T const *),
                                     __LINE__);
            
            return(false);
        }
        this->p_ptr_array_inputs = tmp_ptr_array_inputs;

        this->p_ptr_array_outputs_array = new T const *[tmp_new_number_examples];
        if(this->p_ptr_array_outputs_array == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_new_number_examples * sizeof(T const *),
                                     __LINE__);
            
            return(false);
        }
        this->p_ptr_array_outputs = tmp_ptr_array_outputs;

        // Assign new data
        for(tmp_example_index = 0_zu; tmp_example_index != tmp_new_number_examples; ++tmp_example_index)
        {
            this->p_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_inputs;
            tmp_ptr_array_inputs += this->p_number_inputs * this->p_number_recurrent_depth;

            this->p_ptr_array_outputs_array[tmp_example_index] = tmp_ptr_array_outputs;
            tmp_ptr_array_outputs += this->p_number_outputs * this->p_number_recurrent_depth;
        }

        this->p_number_examples_allocated = this->p_number_examples = tmp_new_number_examples;
    }
    else
    {
        delete[](tmp_ptr_array_inputs);
        delete[](tmp_ptr_array_outputs);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Spliting_Dataset(size_t const desired_data_per_file_received, std::string const &ref_path_file_received)
{
    size_t const tmp_number_files_to_create(static_cast<size_t>(ceil(static_cast<double>(this->p_number_examples) / static_cast<double>(desired_data_per_file_received))));

    if(tmp_number_files_to_create == 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not generate only one file. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ref_path_file_received.find(".dataset") == std::string::npos)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not find \".dataset\" in the path \"%s\". At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_file_received.c_str(),
                                 __LINE__);

        return(false);
    }

    std::string tmp_path,
                    tmp_string_write;

    std::ofstream tmp_ofstream;

    T const **tmp_ptr_array_inputs(this->p_ptr_array_inputs_array),
                **tmp_ptr_array_outputs(this->p_ptr_array_outputs_array);

    for(size_t tmp_data_per_file,
                  tmp_example_index,
                  tmp_time_step_index,
                  tmp_index,
                  tmp_file_index_shift(0_zu),
                  tmp_file_index(0_zu); tmp_file_index != tmp_number_files_to_create; ++tmp_file_index)
    {
        tmp_data_per_file = tmp_file_index + 1_zu != tmp_number_files_to_create ? desired_data_per_file_received : MyEA::Math::Minimum<size_t>(desired_data_per_file_received, this->p_number_examples - desired_data_per_file_received * tmp_file_index);

        tmp_path = ref_path_file_received;
        tmp_path.erase(tmp_path.end() - 8, tmp_path.end()); // ".dataset"
        tmp_path += "_" + std::to_string(tmp_file_index_shift++) + ".dataset";

        while(MyEA::File::Path_Exist(tmp_path))
        {
            tmp_path.erase(tmp_path.end() - (9 + std::to_string(tmp_file_index_shift).length()), tmp_path.end());
            tmp_path += "_" + std::to_string(tmp_file_index_shift++) + ".dataset";
        }

        tmp_ofstream.open(tmp_path, std::ios::out | std::ios::binary | std::ios::trunc);

        if(tmp_ofstream.is_open())
        {
            // Topology
            tmp_string_write = std::to_string(tmp_data_per_file) + " " +
                                        std::to_string(this->p_number_inputs) + " " +
                                        std::to_string(this->p_number_outputs) + " " +
                                        std::to_string(this->p_number_recurrent_depth);

            // Input & Output
            for(tmp_example_index = 0_zu; tmp_example_index != tmp_data_per_file; ++tmp_example_index)
            {
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
                {
                    // Input
                    tmp_string_write += NEW_LINE;
                    for(tmp_index = 0_zu; tmp_index != this->p_number_inputs; ++tmp_index)
                    {
                        tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_inputs[tmp_example_index][tmp_time_step_index * this->p_number_inputs + tmp_index], 9u);

                        if(tmp_index + 1_zu != this->p_number_inputs) { tmp_string_write += " "; }
                    }

                    // Output
                    tmp_string_write += NEW_LINE;
                    for(tmp_index = 0_zu; tmp_index != this->p_number_outputs; ++tmp_index)
                    {
                        tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(tmp_ptr_array_outputs[tmp_example_index][tmp_time_step_index * this->p_number_outputs + tmp_index], 9u);

                        if(tmp_index + 1_zu != this->p_number_outputs) { tmp_string_write += " "; }
                    }
                }
            }

            tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

            tmp_ofstream.flush();
            tmp_ofstream.close();
        }
        else
        {
            PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_path.c_str(),
                                     __LINE__);

            return(false);
        }

        tmp_ptr_array_inputs += tmp_data_per_file;
        tmp_ptr_array_outputs += tmp_data_per_file;
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Simulate_Classification_Trading_Session(class Neural_Network *const ptr_Neural_Network_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Check_Topology(ptr_Neural_Network_received->number_inputs,
                                               ptr_Neural_Network_received->number_outputs,
                                               ptr_Neural_Network_received->number_recurrent_depth) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Check_Topology(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_Neural_Network_received->number_inputs,
                                 ptr_Neural_Network_received->number_outputs,
                                 ptr_Neural_Network_received->number_recurrent_depth,
                                 __LINE__);

        return(false);
    }
    else if(ptr_Neural_Network_received->Update__Batch_Size(this->Get__Number_Examples()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return(false);
    }
    
    size_t const tmp_number_outputs(this->Get__Number_Outputs()),
                       tmp_timed_index(this->p_number_recurrent_depth - 1_zu),
                       tmp_number_examples(this->Get__Number_Examples()),
                       tmp_maximum_batch_size(ptr_Neural_Network_received->batch_size),
                       tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_size,
              tmp_batch_index,
              tmp_example_index,
              tmp_output_index,
              tmp_number_same_sign(0_zu);

    T tmp_desired_output,
       tmp_output;
    
    for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
    {
        tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;
        
        ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_batch_size; ++tmp_example_index)
        {
            tmp_desired_output = 0_T;
            tmp_output = 0_T;

            for(tmp_output_index = 0_zu; tmp_output_index != tmp_number_outputs; ++tmp_output_index)
            {
                tmp_desired_output += this->Get__Output_At(tmp_batch_index * tmp_maximum_batch_size + tmp_example_index, tmp_timed_index * tmp_number_outputs + tmp_output_index);
                
                tmp_output += ptr_Neural_Network_received->Get__Outputs(tmp_example_index, tmp_timed_index)[tmp_output_index];

                /*
                T_ const tmp_diff(this->Get__Output_At(tmp_batch_index * tmp_maximum_batch_size + tmp_example_index, tmp_timed_index * tmp_number_outputs + tmp_output_index)
                                                -
                                          ptr_Neural_Network_received->Get__Outputs(tmp_example_index, tmp_timed_index)[tmp_output_index]);
                if(tmp_output_index == 0 && abs(tmp_diff) > 0.06)
                {
                    PRINT_FORMAT("[%zu | %zu][%zu]: %f" NEW_LINE,
                                             tmp_batch_index * tmp_maximum_batch_size + tmp_example_index,
                                             tmp_example_index,
                                             tmp_output_index,
                                             tmp_diff);
                }
                */
            }

            tmp_number_same_sign += static_cast<size_t>(MyEA::Math::Sign<T_>(tmp_output) == MyEA::Math::Sign<T_>(tmp_desired_output));
        }
    }

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Report total trades: %zu" NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_number_examples);
    PRINT_FORMAT("%s: \tSucces: %f%%" NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             static_cast<double>(tmp_number_same_sign) / static_cast<double>(tmp_number_examples) * 100.0);
    
    return(true);
}

template<typename T>
bool Dataset<T>::Replace_Entries(class Dataset<T> const *const ptr_source_Dataset_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_source_Dataset_received->p_number_examples != this->p_number_examples)
    {
        PRINT_FORMAT("%s: %s: ERROR: Source number data (%zu) differ from destination number data (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_source_Dataset_received->p_number_examples,
                                 this->p_number_examples,
                                 __LINE__);

        return(false);
    }
    else if(ptr_source_Dataset_received->Get__Number_Recurrent_Depth() != this->p_number_recurrent_depth)
    {
        PRINT_FORMAT("%s: %s: ERROR: Source recurrent depth (%zu) differ from destination recurrent depth (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_source_Dataset_received->Get__Number_Recurrent_Depth(),
                                 this->p_number_recurrent_depth,
                                 __LINE__);

        return(false);
    }
    else if(type_input_received > ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? ptr_source_Dataset_received->Get__Number_Inputs() : ptr_source_Dataset_received->Get__Number_Outputs());

    T const *const tmp_ptr_array_source_inputs(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? ptr_source_Dataset_received->Get__Input_At(0_zu) : ptr_source_Dataset_received->Get__Output_At(0_zu));
    
    if(tmp_ptr_array_source_inputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_array_source_inputs\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    T const **tmp_ptr_array_inputs_array;
    T *tmp_ptr_array_inputs;

    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        tmp_ptr_array_inputs = MyEA::Memory::Cpp::Reallocate<T, false>(
            this->p_ptr_array_inputs,
            this->p_number_examples * this->p_number_recurrent_depth * tmp_input_size,
            this->p_number_examples * this->p_number_recurrent_depth * this->p_number_inputs
        );
        
        if(tmp_ptr_array_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T),
                                     this->p_number_examples * this->p_number_recurrent_depth * tmp_input_size,
                                     this->p_number_examples * this->p_number_recurrent_depth * this->p_number_inputs,
                                     __LINE__);

            this->p_ptr_array_inputs = nullptr;

            return(false);
        }

        this->p_ptr_array_inputs = tmp_ptr_array_inputs;
        
        tmp_ptr_array_inputs_array = this->p_ptr_array_inputs_array;

        this->p_number_inputs = tmp_input_size;
    }
    else
    {
        tmp_ptr_array_inputs = MyEA::Memory::Cpp::Reallocate<T, false>(this->p_ptr_array_outputs,
                                                                                    this->p_number_examples * this->p_number_recurrent_depth * tmp_input_size,
                                                                                    this->p_number_examples * this->p_number_recurrent_depth * this->p_number_outputs
                                                                                    );
        
        if(tmp_ptr_array_inputs == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     sizeof(T),
                                     this->p_number_examples * this->p_number_recurrent_depth * tmp_input_size,
                                     this->p_number_examples * this->p_number_recurrent_depth * this->p_number_outputs,
                                     __LINE__);
            
            this->p_ptr_array_outputs = nullptr;

            return(false);
        }

        this->p_ptr_array_outputs = tmp_ptr_array_inputs;
        
        tmp_ptr_array_inputs_array = this->p_ptr_array_outputs_array;

        this->p_number_outputs = tmp_input_size;
    }
    
    MEMCPY(tmp_ptr_array_inputs,
                   tmp_ptr_array_source_inputs,
                   this->p_number_examples * this->p_number_recurrent_depth * tmp_input_size * sizeof(T));

    for(size_t tmp_example_index(0_zu); tmp_example_index != this->p_number_examples; ++tmp_example_index)
    { tmp_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_inputs + tmp_example_index * tmp_input_size * this->p_number_recurrent_depth; }
    
    // TODO: Deep copy inputs/outputs.
    // ...
    // ...
    // ...

    return(true);
}

template<typename T>
bool Dataset<T>::Replace_Entries(class Dataset<T> const *const ptr_Autoencoder_Dataset_received, class Neural_Network *const ptr_Autoencoder_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_examples != ptr_Autoencoder_Dataset_received->p_number_examples)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of data (%zu) differ from the number of data received as argument (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_examples,
                                 ptr_Autoencoder_Dataset_received->p_number_examples,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_recurrent_depth != ptr_Autoencoder_Dataset_received->p_number_recurrent_depth)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of recurrent depth (%zu) differ from the number of recurrent depth received as argument (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_recurrent_depth,
                                 ptr_Autoencoder_Dataset_received->p_number_recurrent_depth,
                                 __LINE__);

        return(false);
    }
    else if(ptr_Autoencoder_Dataset_received->Check_Topology(ptr_Autoencoder_received->number_inputs,
                                                                                            ptr_Autoencoder_received->number_outputs,
                                                                                            ptr_Autoencoder_received->number_recurrent_depth) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Check_Topology(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_Autoencoder_received->number_inputs,
                                 ptr_Autoencoder_received->number_outputs,
                                 ptr_Autoencoder_received->number_recurrent_depth,
                                 __LINE__);

        return(false);
    }
    else if(ptr_Autoencoder_received->type_network != MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER)
    {
        PRINT_FORMAT("%s: %s: ERROR: The neural network (%s) receive as argument need to be a %s. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[ptr_Autoencoder_received->type_network].c_str(),
                                 MyEA::Common::ENUM_TYPE_NETWORKS_NAMES[MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER].c_str(),
                                 __LINE__);

        return(false);
    }

    auto tmp_Reset_IO_Mode([tmp_use_first_layer_as_input = ptr_Autoencoder_received->use_first_layer_as_input,
                                            tmp_use_last_layer_as_output = ptr_Autoencoder_received->use_last_layer_as_output,
                                            &ptr_Autoencoder_received]() -> bool
    {
        bool tmp_succes(true);

        if(ptr_Autoencoder_received->Set__Input_Mode(tmp_use_first_layer_as_input) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Input_Mode(%s)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_use_first_layer_as_input ? "true" : "false",
                                     __LINE__);

            tmp_succes = false;
        }
        
        if(ptr_Autoencoder_received->Set__Output_Mode(tmp_use_last_layer_as_output) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Output_Mode(%s)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_use_last_layer_as_output ? "true" : "false",
                                     __LINE__);

            tmp_succes = false;
        }

        return(tmp_succes);
    });
    
    if(ptr_Autoencoder_received->Set__Input_Mode(true) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Input_Mode(true)\" function. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    __LINE__);
        
        return(false);
    }
    else if(ptr_Autoencoder_received->Set__Output_Mode(false) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Output_Mode(false)\" function. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    __LINE__);
        
        if(tmp_Reset_IO_Mode() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Reset_IO_Mode()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }
    else if(this->p_number_inputs != ptr_Autoencoder_received->Get__Output_Size())
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of input(s) (%zu) differ from the number of output(s) from the autoencoder (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_inputs,
                                 ptr_Autoencoder_received->Get__Output_Size(),
                                 __LINE__);

        if(tmp_Reset_IO_Mode() == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Reset_IO_Mode()\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        return(false);
    }

    size_t const tmp_number_examples(this->p_number_examples),
                       tmp_maximum_batch_size(ptr_Autoencoder_received->batch_size),
                       tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_input_index,
              tmp_time_step_index,
              tmp_example_index,
              tmp_batch_size,
              tmp_batch_index;

    T const *tmp_ptr_array_outputs;
    
    for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
    {
        tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;
        
        ptr_Autoencoder_received->Forward_Pass(tmp_batch_size, ptr_Autoencoder_Dataset_received->p_ptr_array_inputs_array + tmp_batch_index * tmp_maximum_batch_size);

        for(tmp_example_index = 0_zu; tmp_example_index != tmp_batch_size; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
            {
                tmp_ptr_array_outputs = ptr_Autoencoder_received->Get__Outputs(tmp_example_index, tmp_time_step_index);

                for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_inputs; ++tmp_input_index)
                {
                    this->p_ptr_array_inputs[tmp_example_index * this->p_number_inputs * this->p_number_recurrent_depth + tmp_time_step_index * this->p_number_inputs + tmp_input_index] = tmp_ptr_array_outputs[tmp_input_index];
                }
            }
        }
    }
    
    if(tmp_Reset_IO_Mode() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"tmp_Reset_IO_Mode()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Concat(class Dataset<T> const *const ptr_source_Dataset_received)
{
    if(ptr_source_Dataset_received->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available from the source. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Check_Topology(ptr_source_Dataset_received->Get__Number_Inputs(),
                                              ptr_source_Dataset_received->Get__Number_Outputs(),
                                              ptr_source_Dataset_received->Get__Number_Recurrent_Depth()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Check_Topology(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_source_Dataset_received->Get__Number_Inputs(),
                                 ptr_source_Dataset_received->Get__Number_Outputs(),
                                 ptr_source_Dataset_received->Get__Number_Recurrent_Depth(),
                                 __LINE__);

        return(false);
    }
    else if(ptr_source_Dataset_received->Get__Number_Recurrent_Depth() != this->p_number_recurrent_depth)
    {
        PRINT_FORMAT("%s: %s: ERROR: Source recurrent depth (%zu) differ from destination recurrent depth (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_source_Dataset_received->Get__Number_Recurrent_Depth(),
                                 this->p_number_recurrent_depth,
                                 __LINE__);

        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_concat_number_examples(this->p_number_examples + ptr_source_Dataset_received->p_number_examples);

    // Array inputs.
    T *tmp_ptr_array_inputs(MyEA::Memory::Cpp::Reallocate<T>(this->p_ptr_array_inputs,
                                                                                 tmp_concat_number_examples * this->p_number_recurrent_depth * this->p_number_inputs,
                                                                                 this->p_number_examples * this->p_number_recurrent_depth * this->p_number_inputs));

    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    sizeof(T),
                                    this->p_number_examples * this->p_number_recurrent_depth * this->p_number_inputs,
                                    tmp_concat_number_examples * this->p_number_recurrent_depth * this->p_number_inputs,
                                    __LINE__);

        this->p_ptr_array_inputs = nullptr;

        return(false);
    }

    this->p_ptr_array_inputs = tmp_ptr_array_inputs;
    
    T const **tmp_ptr_array_inputs_array(MyEA::Memory::Cpp::Reallocate_PtOfPt<T const *, false>(this->p_ptr_array_inputs_array,
                                                                                                                tmp_concat_number_examples,
                                                                                                                this->p_number_examples));

    if(tmp_ptr_array_inputs_array == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    sizeof(T),
                                    this->p_number_examples * this->p_number_recurrent_depth * this->p_number_inputs,
                                    tmp_concat_number_examples * this->p_number_recurrent_depth * this->p_number_inputs,
                                    __LINE__);

        this->p_ptr_array_inputs_array = nullptr;

        return(false);
    }

    this->p_ptr_array_inputs_array = tmp_ptr_array_inputs_array;
    
    MEMCPY(tmp_ptr_array_inputs + this->p_number_examples * this->p_number_recurrent_depth * this->p_number_inputs,
                   ptr_source_Dataset_received->p_ptr_array_inputs,
                   (tmp_concat_number_examples - this->p_number_examples) * this->p_number_recurrent_depth * this->p_number_inputs * sizeof(T));
    // |END| Array inputs. |END|

    // Array outputs.
    T *tmp_ptr_array_outputs(MyEA::Memory::Cpp::Reallocate<T>(this->p_ptr_array_outputs,
                                                                                   tmp_concat_number_examples * this->p_number_recurrent_depth * this->p_number_outputs,
                                                                                   this->p_number_examples * this->p_number_recurrent_depth * this->p_number_outputs));
        
    if(tmp_ptr_array_outputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    sizeof(T),
                                    this->p_number_examples * this->p_number_recurrent_depth * this->p_number_outputs,
                                    tmp_concat_number_examples * this->p_number_recurrent_depth * this->p_number_outputs,
                                    __LINE__);
            
        this->p_ptr_array_outputs = nullptr;

        return(false);
    }

    this->p_ptr_array_outputs = tmp_ptr_array_outputs;
    
    T const **tmp_ptr_array_outputs_array(MyEA::Memory::Cpp::Reallocate_PtOfPt<T const *, false>(this->p_ptr_array_outputs_array,
                                                                                                                  tmp_concat_number_examples,
                                                                                                                  this->p_number_examples));
        
    if(tmp_ptr_array_outputs_array == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                    MyEA::Time::Date_Time_Now().c_str(),
                                    __FUNCTION__,
                                    sizeof(T),
                                    this->p_number_examples * this->p_number_recurrent_depth * this->p_number_outputs,
                                    tmp_concat_number_examples * this->p_number_recurrent_depth * this->p_number_outputs,
                                    __LINE__);
            
        this->p_ptr_array_outputs_array = nullptr;

        return(false);
    }

    this->p_ptr_array_outputs_array = tmp_ptr_array_outputs_array;
    
    MEMCPY(tmp_ptr_array_outputs + this->p_number_examples * this->p_number_recurrent_depth * this->p_number_outputs,
                  ptr_source_Dataset_received->p_ptr_array_outputs,
                  (tmp_concat_number_examples - this->p_number_examples) * this->p_number_recurrent_depth * this->p_number_outputs * sizeof(T));
    //|END| Array outputs. |END|

    this->p_number_examples = tmp_concat_number_examples;
    
    for(size_t tmp_example_index(0_zu); tmp_example_index != tmp_concat_number_examples; ++tmp_example_index)
    {
        tmp_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_inputs + tmp_example_index * this->p_number_inputs * this->p_number_recurrent_depth;

        tmp_ptr_array_outputs_array[tmp_example_index] = tmp_ptr_array_outputs + tmp_example_index * this->p_number_outputs * this->p_number_recurrent_depth;
    }
    
    return(true);
}

template<typename T>
bool Dataset<T>::Save__Sequential_Input(size_t const number_recurrent_depth_received, std::string const &ref_path_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(number_recurrent_depth_received <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth (%zu) need to be greater or equal 2. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 number_recurrent_depth_received,
                                 __LINE__);

        return(false);
    }
    else if(number_recurrent_depth_received > this->p_number_inputs)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth (%zu) greater than the number of inputs (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 number_recurrent_depth_received,
                                 this->p_number_inputs,
                                 __LINE__);

        return(false);
    }
    
    if(MyEA::File::Write_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Write_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }
    
    size_t const tmp_number_inputs_per_time_step(static_cast<size_t>(this->p_number_inputs / number_recurrent_depth_received));
    size_t tmp_time_step_index,
              tmp_example_index,
              tmp_index;

    std::string tmp_string_write;

    std::ofstream tmp_ofstream(ref_path_received, std::ios::out | std::ios::binary | std::ios::trunc);

    if(tmp_ofstream.is_open())
    {
        // Topology
        tmp_string_write = std::to_string(this->p_number_examples) + " " +
                                    std::to_string(tmp_number_inputs_per_time_step) + " " +
                                    std::to_string(this->p_number_outputs) + " " +
                                    std::to_string(number_recurrent_depth_received);

        // Input & Output
        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            for(tmp_time_step_index = 0_zu; tmp_time_step_index != number_recurrent_depth_received; ++tmp_time_step_index)
            {
                // Input
                tmp_string_write += NEW_LINE;
                for(tmp_index = 0_zu; tmp_index != tmp_number_inputs_per_time_step; ++tmp_index)
                {
                    tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->p_ptr_array_inputs_array[tmp_example_index][tmp_time_step_index * tmp_number_inputs_per_time_step + tmp_index], 9u);
                    
                    if(tmp_index + 1_zu != tmp_number_inputs_per_time_step) { tmp_string_write += " "; }
                }

                // Output
                tmp_string_write += NEW_LINE;
                for(tmp_index = 0_zu; tmp_index != this->p_number_outputs; ++tmp_index)
                {
                    tmp_string_write += MyEA::String::To_string<T, MyEA::String::ENUM_TYPE_STRING_FORMAT::DEFAULTFLOAT>(this->p_ptr_array_outputs_array[tmp_example_index][(tmp_time_step_index % this->p_number_recurrent_depth) * this->p_number_outputs + tmp_index], 9u);

                    if(tmp_index + 1_zu != this->p_number_outputs) { tmp_string_write += " "; }
                }
            }
        }
        
        tmp_ofstream.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size() * sizeof(char)));

        tmp_ofstream.flush();
        tmp_ofstream.close();
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: The file %s can not be opened. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }
    
    if(MyEA::File::Delete_Temporary_File(ref_path_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Delete_Temporary_File(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_path_received.c_str(),
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Minimum_Maximum(size_t const data_start_index_received,
                                                                                size_t const data_end_index_received,
                                                                                T const minimum_value_received,
                                                                                T const maximum_value_received,
                                                                                T const minimum_range_received,
                                                                                T const maximum_range_received,
                                                                                enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(data_start_index_received > data_end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_start_index_received,
                                 data_end_index_received,
                                 __LINE__);
        
        return(false);
    }
    else if(minimum_value_received > maximum_value_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value (%f) can not be greater than maximum value (%f). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_value_received),
                                 Cast_T(maximum_value_received),
                                 __LINE__);
        
        return(false);
    }
    else if(minimum_range_received > maximum_range_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum range (%f) can not be greater than maximum range (%f). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_range_received),
                                 Cast_T(maximum_range_received),
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(minimum_value_received == minimum_range_received
             ||
             maximum_value_received == maximum_range_received) { return(true); }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__minimum_maximum == nullptr && (this->_ptr_input_array_scaler__minimum_maximum = new struct Scaler__Minimum_Maximum<T>[this->p_number_inputs]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_inputs * sizeof(struct Scaler__Minimum_Maximum<T>),
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__minimum_maximum == nullptr && (this->_ptr_output_array_scaler__minimum_maximum = new struct Scaler__Minimum_Maximum<T>[this->p_number_outputs]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_outputs * sizeof(struct Scaler__Minimum_Maximum<T>),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        if(this->Preprocessing__Minimum_Maximum(data_start_index_received,
                                                                         data_end_index_received,
                                                                         tmp_input_index,
                                                                         minimum_value_received,
                                                                         maximum_value_received,
                                                                         minimum_range_received,
                                                                         maximum_range_received,
                                                                         type_input_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum(%zu, %zu, %zu, %f, %f, %f, %f, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     data_start_index_received,
                                     data_end_index_received,
                                     tmp_input_index,
                                     Cast_T(minimum_value_received),
                                     Cast_T(maximum_value_received),
                                     Cast_T(minimum_range_received),
                                     Cast_T(maximum_range_received),
                                     type_input_received,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Minimum_Maximum(size_t const data_start_index_received,
                                                                                size_t const data_end_index_received,
                                                                                size_t const input_index_received,
                                                                                T const minimum_value_received,
                                                                                T const maximum_value_received,
                                                                                T const minimum_range_received,
                                                                                T const maximum_range_received,
                                                                                enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(data_start_index_received > data_end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_start_index_received,
                                 data_end_index_received,
                                 __LINE__);
        
        return(false);
    }
    else if(minimum_value_received > maximum_value_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum value (%f) can not be greater than maximum value (%f). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_value_received),
                                 Cast_T(maximum_value_received),
                                 __LINE__);
        
        return(false);
    }
    else if(minimum_range_received > maximum_range_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Minimum range (%f) can not be greater than maximum range (%f). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(minimum_range_received),
                                 Cast_T(maximum_range_received),
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(minimum_value_received == minimum_range_received
             ||
             maximum_value_received == maximum_range_received) { return(true); }
    
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__minimum_maximum == nullptr && (this->_ptr_input_array_scaler__minimum_maximum = new struct Scaler__Minimum_Maximum<T>[this->p_number_inputs]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_inputs * sizeof(struct Scaler__Minimum_Maximum<T>),
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__minimum_maximum == nullptr && (this->_ptr_output_array_scaler__minimum_maximum = new struct Scaler__Minimum_Maximum<T>[this->p_number_outputs]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_outputs * sizeof(struct Scaler__Minimum_Maximum<T>),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_example_index,
              tmp_time_step_index;

    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }
    
    struct Scaler__Minimum_Maximum<T> *const tmp_ptr_scaler__minimum_maximum(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? &this->_ptr_input_array_scaler__minimum_maximum[input_index_received] : &this->_ptr_output_array_scaler__minimum_maximum[input_index_received]);
    
    if(tmp_ptr_scaler__minimum_maximum == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    T *const tmp_ptr_array_inputs(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_array_inputs : this->p_ptr_array_outputs);

    for(tmp_example_index = data_start_index_received; tmp_example_index != data_end_index_received; ++tmp_example_index)
    {
        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_ptr_array_inputs[tmp_example_index * tmp_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_input_size + input_index_received] = (((tmp_ptr_array_inputs[tmp_example_index * tmp_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_input_size + input_index_received] - minimum_value_received) * (maximum_range_received - minimum_range_received)) / (maximum_value_received - minimum_value_received)) + minimum_range_received;
        }
    }

    tmp_ptr_scaler__minimum_maximum->minimum_value = minimum_value_received;
    tmp_ptr_scaler__minimum_maximum->maximum_value = maximum_value_received;

    tmp_ptr_scaler__minimum_maximum->minimum_range = minimum_range_received;
    tmp_ptr_scaler__minimum_maximum->maximum_range = maximum_range_received;
    
    tmp_ptr_scaler__minimum_maximum->start_index = data_start_index_received;
    tmp_ptr_scaler__minimum_maximum->end_index = data_end_index_received;
    
    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Minimum_Maximum(T *const ptr_array_inputs_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_input_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_output_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        if(this->Preprocessing__Minimum_Maximum(tmp_input_index,
                                                                         ptr_array_inputs_received,
                                                                         type_input_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum(%zu, ptr, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_index,
                                     type_input_received,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Minimum_Maximum(size_t const input_index_received,
                                                                                T *const ptr_array_inputs_received,
                                                                                enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_input_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_output_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_time_step_index;

    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }
    
    struct Scaler__Minimum_Maximum<T> *const tmp_ptr_scaler__minimum_maximum(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? &this->_ptr_input_array_scaler__minimum_maximum[input_index_received] : &this->_ptr_output_array_scaler__minimum_maximum[input_index_received]);
    
    if(tmp_ptr_scaler__minimum_maximum == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    T const tmp_minimum_value(tmp_ptr_scaler__minimum_maximum->minimum_value),
                tmp_maximum_value(tmp_ptr_scaler__minimum_maximum->maximum_value),
                tmp_minimum_range(tmp_ptr_scaler__minimum_maximum->minimum_range),
                tmp_maximum_range(tmp_ptr_scaler__minimum_maximum->maximum_range);
    
    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
    { ptr_array_inputs_received[tmp_time_step_index * tmp_input_size + input_index_received] = (((ptr_array_inputs_received[tmp_time_step_index * tmp_input_size + input_index_received] - tmp_minimum_value) * (tmp_maximum_range - tmp_minimum_range)) / (tmp_maximum_value - tmp_minimum_value)) + tmp_minimum_range; }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Minimum_Maximum_Inverse(enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_input_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_output_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        if(this->Preprocessing__Minimum_Maximum_Inverse(tmp_input_index, type_input_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse(%zu, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_index,
                                     type_input_received,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Minimum_Maximum_Inverse(size_t const input_index_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_input_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_output_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_example_index,
              tmp_time_step_index;

    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }
    
    struct Scaler__Minimum_Maximum<T> *const tmp_ptr_scaler__minimum_maximum(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? &this->_ptr_input_array_scaler__minimum_maximum[input_index_received] : &this->_ptr_output_array_scaler__minimum_maximum[input_index_received]);
    
    if(tmp_ptr_scaler__minimum_maximum == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_scaler__minimum_maximum->start_index == tmp_ptr_scaler__minimum_maximum->end_index) { return(true); }
    
    T const tmp_minimum_value(tmp_ptr_scaler__minimum_maximum->minimum_range),
                tmp_maximum_value(tmp_ptr_scaler__minimum_maximum->maximum_range),
                tmp_minimum_range(tmp_ptr_scaler__minimum_maximum->minimum_value),
                tmp_maximum_range(tmp_ptr_scaler__minimum_maximum->maximum_value);
    
    if(tmp_minimum_value == tmp_minimum_range
      ||
      tmp_maximum_value == tmp_maximum_range) { return(true); }
    
    size_t const tmp_data_end_index(tmp_ptr_scaler__minimum_maximum->end_index);

    T *const tmp_ptr_array_inputs(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_array_inputs : this->p_ptr_array_outputs);

    for(tmp_example_index = tmp_ptr_scaler__minimum_maximum->start_index; tmp_example_index != tmp_data_end_index; ++tmp_example_index)
    {
        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_ptr_array_inputs[tmp_example_index * tmp_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_input_size + input_index_received] = (((tmp_ptr_array_inputs[tmp_example_index * tmp_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_input_size + input_index_received] - tmp_minimum_value) * (tmp_maximum_range - tmp_minimum_range)) / (tmp_maximum_value - tmp_minimum_value)) + tmp_minimum_range;
        }
    }
    
    tmp_ptr_scaler__minimum_maximum->start_index = 0_zu;
    tmp_ptr_scaler__minimum_maximum->end_index = 0_zu;
    
    tmp_ptr_scaler__minimum_maximum->minimum_value = T(0);
    tmp_ptr_scaler__minimum_maximum->maximum_value = T(1);

    tmp_ptr_scaler__minimum_maximum->minimum_range = T(0);
    tmp_ptr_scaler__minimum_maximum->maximum_range = T(1);
    
    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Minimum_Maximum_Inverse(T *const ptr_array_inputs_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_input_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_output_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        if(this->Preprocessing__Minimum_Maximum_Inverse(tmp_input_index,
                                                                                     ptr_array_inputs_received,
                                                                                     type_input_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse(%zu, ptr, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_index,
                                     type_input_received,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Minimum_Maximum_Inverse(size_t const input_index_received,
                                                                                            T *const ptr_array_inputs_received,
                                                                                            enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_input_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__minimum_maximum == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_output_array_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_time_step_index;

    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }
    
    struct Scaler__Minimum_Maximum<T> *const tmp_ptr_scaler__minimum_maximum(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? &this->_ptr_input_array_scaler__minimum_maximum[input_index_received] : &this->_ptr_output_array_scaler__minimum_maximum[input_index_received]);
    
    if(tmp_ptr_scaler__minimum_maximum == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_scaler__minimum_maximum\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_scaler__minimum_maximum->start_index == tmp_ptr_scaler__minimum_maximum->end_index) { return(true); }
    
    T const tmp_minimum_value(tmp_ptr_scaler__minimum_maximum->minimum_range),
                tmp_maximum_value(tmp_ptr_scaler__minimum_maximum->maximum_range),
                tmp_minimum_range(tmp_ptr_scaler__minimum_maximum->minimum_value),
                tmp_maximum_range(tmp_ptr_scaler__minimum_maximum->maximum_value);
    
    if(tmp_minimum_value == tmp_minimum_range
      ||
      tmp_maximum_value == tmp_maximum_range) { return(true); }

    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index) { ptr_array_inputs_received[tmp_time_step_index * tmp_input_size + input_index_received] = (((ptr_array_inputs_received[tmp_time_step_index * tmp_input_size + input_index_received] - tmp_minimum_value) * (tmp_maximum_range - tmp_minimum_range)) / (tmp_maximum_value - tmp_minimum_value)) + tmp_minimum_range; }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Zero_Centered(size_t const data_start_index_received,
                                                                        size_t const data_end_index_received,
                                                                        T const multiplier_received,
                                                                        enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(data_start_index_received > data_end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_start_index_received,
                                 data_end_index_received,
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__zero_centered == nullptr && (this->_ptr_input_array_scaler__zero_centered = new struct Scaler__Zero_Centered<T>[this->p_number_inputs]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_inputs * sizeof(struct Scaler__Zero_Centered<T>),
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__zero_centered == nullptr && (this->_ptr_output_array_scaler__zero_centered = new struct Scaler__Zero_Centered<T>[this->p_number_outputs]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_outputs * sizeof(struct Scaler__Zero_Centered<T>),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        if(this->Preprocessing__Zero_Centered(data_start_index_received,
                                                                 data_end_index_received,
                                                                 tmp_input_index,
                                                                 multiplier_received,
                                                                 type_input_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered(%zu, %zu, %zu, %f, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     data_start_index_received,
                                     data_end_index_received,
                                     tmp_input_index,
                                     Cast_T(multiplier_received),
                                     type_input_received,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Zero_Centered(size_t const data_start_index_received,
                                                                        size_t const data_end_index_received,
                                                                        size_t const input_index_received,
                                                                        T const multiplier_received,
                                                                        enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(data_start_index_received > data_end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_start_index_received,
                                 data_end_index_received,
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__zero_centered == nullptr && (this->_ptr_input_array_scaler__zero_centered = new struct Scaler__Zero_Centered<T>[this->p_number_inputs]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_inputs * sizeof(struct Scaler__Zero_Centered<T>),
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__zero_centered == nullptr && (this->_ptr_output_array_scaler__zero_centered = new struct Scaler__Zero_Centered<T>[this->p_number_outputs]) == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_outputs * sizeof(struct Scaler__Zero_Centered<T>),
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_example_index,
              tmp_time_step_index;

    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }
    
    struct Scaler__Zero_Centered<T> *const tmp_ptr_scaler__zero_centered(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? &this->_ptr_input_array_scaler__zero_centered[input_index_received] : &this->_ptr_output_array_scaler__zero_centered[input_index_received]);
    
    if(tmp_ptr_scaler__zero_centered == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    T *const tmp_ptr_array_inputs(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_array_inputs : this->p_ptr_array_outputs);

    for(tmp_example_index = data_start_index_received; tmp_example_index != data_end_index_received; ++tmp_example_index)
    {
        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_ptr_array_inputs[tmp_example_index * tmp_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_input_size + input_index_received] *= multiplier_received;
        }
    }

    tmp_ptr_scaler__zero_centered->start_index = data_start_index_received;
    tmp_ptr_scaler__zero_centered->end_index = data_end_index_received;
    
    tmp_ptr_scaler__zero_centered->multiplier = multiplier_received;

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Zero_Centered(T *const ptr_array_inputs_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__zero_centered == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_input_array_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__zero_centered == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_output_array_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        if(this->Preprocessing__Zero_Centered(tmp_input_index,
                                                                 ptr_array_inputs_received,
                                                                 type_input_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered(%zu, ptr, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_index,
                                     type_input_received,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Zero_Centered(size_t const input_index_received,
                                                                        T *const ptr_array_inputs_received,
                                                                        enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__zero_centered == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_input_array_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__zero_centered == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_output_array_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_time_step_index;

    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }
    
    struct Scaler__Zero_Centered<T> *const tmp_ptr_scaler__zero_centered(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? &this->_ptr_input_array_scaler__zero_centered[input_index_received] : &this->_ptr_output_array_scaler__zero_centered[input_index_received]);
    
    if(tmp_ptr_scaler__zero_centered == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_scaler__zero_centered->start_index == tmp_ptr_scaler__zero_centered->end_index) { return(true); }
    
    T const tmp_multiplier(tmp_ptr_scaler__zero_centered->multiplier);
    
    if(tmp_multiplier == T(1)) { return(true); }

    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
    { ptr_array_inputs_received[tmp_time_step_index * tmp_input_size + input_index_received] *= tmp_multiplier; }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Zero_Centered_Inverse(enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__zero_centered == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_input_array_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__zero_centered == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_output_array_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        if(this->Preprocessing__Zero_Centered_Inverse(tmp_input_index, type_input_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum_Inverse(%zu, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_index,
                                     type_input_received,
                                     __LINE__);

            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Zero_Centered_Inverse(size_t const input_index_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->_ptr_input_array_scaler__zero_centered == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_input_array_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->_ptr_output_array_scaler__zero_centered == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"_ptr_output_array_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_example_index,
              tmp_time_step_index;

    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }
    
    struct Scaler__Zero_Centered<T> *const tmp_ptr_scaler__zero_centered(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? &this->_ptr_input_array_scaler__zero_centered[input_index_received] : &this->_ptr_output_array_scaler__zero_centered[input_index_received]);
    
    if(tmp_ptr_scaler__zero_centered == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_scaler__zero_centered\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_scaler__zero_centered->start_index == tmp_ptr_scaler__zero_centered->end_index) { return(true); }
    
    T const tmp_multiplier(T(1) / tmp_ptr_scaler__zero_centered->multiplier);
    
    if(tmp_multiplier == T(1)) { return(true); }

    size_t const tmp_data_end_index(tmp_ptr_scaler__zero_centered->end_index);

    T *const tmp_ptr_array_inputs(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_array_inputs : this->p_ptr_array_outputs);

    for(tmp_example_index = tmp_ptr_scaler__zero_centered->start_index; tmp_example_index != tmp_data_end_index; ++tmp_example_index)
    {
        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_ptr_array_inputs[tmp_example_index * tmp_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_input_size + input_index_received] *= tmp_multiplier;
        }
    }
    
    tmp_ptr_scaler__zero_centered->start_index = 0_zu;
    tmp_ptr_scaler__zero_centered->end_index = 0_zu;
    
    tmp_ptr_scaler__zero_centered->multiplier = T(1);
    
    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__MODWT(size_t const desired_J_level_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No enought data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_recurrent_depth == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(desired_J_level_received == 0_zu) { return(true); }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->p_ptr_input_coefficient_matrix == nullptr)
        {
            if((this->p_ptr_input_coefficient_matrix = new T*[this->p_number_inputs * this->p_number_recurrent_depth]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_inputs * sizeof(T*),
                                         __LINE__);

                return(false);
            }
            MyEA::Memory::Cpp::Fill_Nullptr<T*>(this->p_ptr_input_coefficient_matrix, this->p_ptr_input_coefficient_matrix + this->p_number_inputs * this->p_number_recurrent_depth);
        }
        
        if(this->p_ptr_input_array_coefficient_matrix_size == nullptr)
        {
            if((this->p_ptr_input_array_coefficient_matrix_size = new size_t[this->p_number_inputs]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_inputs * sizeof(size_t),
                                         __LINE__);

                return(false);
            }
            memset(this->p_ptr_input_array_coefficient_matrix_size,
                         0,
                         this->p_number_inputs * sizeof(size_t));
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->p_ptr_output_coefficient_matrix == nullptr)
        {
            if((this->p_ptr_output_coefficient_matrix = new T*[this->p_number_outputs * this->p_number_recurrent_depth]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_outputs * sizeof(T*),
                                         __LINE__);

                return(false);
            }
            MyEA::Memory::Cpp::Fill_Nullptr<T*>(this->p_ptr_output_coefficient_matrix, this->p_ptr_output_coefficient_matrix + this->p_number_outputs * this->p_number_recurrent_depth);
        }
        
        if(this->p_ptr_output_array_coefficient_matrix_size == nullptr)
        {
            if((this->p_ptr_output_array_coefficient_matrix_size = new size_t[this->p_number_outputs]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_outputs * sizeof(size_t),
                                         __LINE__);

                return(false);
            }
            memset(this->p_ptr_output_array_coefficient_matrix_size,
                         0,
                         this->p_number_outputs * sizeof(size_t));
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        if(this->Preprocessing__MODWT(tmp_input_index,
                                                         desired_J_level_received,
                                                         type_input_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT(%zu, %zu, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_index,
                                     desired_J_level_received,
                                     type_input_received,
                                     __LINE__);
            
            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__MODWT(size_t const input_index_received,
                                                                size_t const desired_J_level_received,
                                                                enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No enought data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_recurrent_depth == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(desired_J_level_received == 0_zu) { return(true); }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->p_ptr_input_coefficient_matrix == nullptr)
        {
            if((this->p_ptr_input_coefficient_matrix = new T*[this->p_number_inputs * this->p_number_recurrent_depth]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_inputs * sizeof(T*),
                                         __LINE__);

                return(false);
            }
            MyEA::Memory::Cpp::Fill_Nullptr<T*>(this->p_ptr_input_coefficient_matrix, this->p_ptr_input_coefficient_matrix + this->p_number_inputs * this->p_number_recurrent_depth);
        }

        if(this->p_ptr_input_array_coefficient_matrix_size == nullptr)
        {
            if((this->p_ptr_input_array_coefficient_matrix_size = new size_t[this->p_number_inputs]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_inputs * sizeof(size_t),
                                         __LINE__);

                return(false);
            }
            memset(this->p_ptr_input_array_coefficient_matrix_size,
                         0,
                         this->p_number_inputs * sizeof(size_t));
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->p_ptr_output_coefficient_matrix == nullptr)
        {
            if((this->p_ptr_output_coefficient_matrix = new T*[this->p_number_outputs * this->p_number_recurrent_depth]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_outputs * sizeof(T*),
                                         __LINE__);

                return(false);
            }
            MyEA::Memory::Cpp::Fill_Nullptr<T*>(this->p_ptr_output_coefficient_matrix, this->p_ptr_output_coefficient_matrix + this->p_number_outputs * this->p_number_recurrent_depth);
        }
        
        if(this->p_ptr_output_array_coefficient_matrix_size == nullptr)
        {
            if((this->p_ptr_output_array_coefficient_matrix_size = new size_t[this->p_number_outputs]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_outputs * sizeof(size_t),
                                         __LINE__);

                return(false);
            }
            memset(this->p_ptr_output_array_coefficient_matrix_size,
                         0,
                         this->p_number_outputs * sizeof(size_t));
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    
    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_J_level(MyEA::Math::Minimum<size_t>(this->MODWT__J_Level_Maximum(), desired_J_level_received));
    size_t tmp_coefficient_matrix_size,
              tmp_example_index,
              tmp_time_step_index;

    T *const tmp_ptr_array_inputs(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_array_inputs : this->p_ptr_array_outputs),
       *tmp_ptr_array_inputs_preproced,
       *tmp_ptr_array_smooth_coefficients;
    
    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_array_inputs\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if((tmp_ptr_array_inputs_preproced = new T[this->p_number_examples]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_examples * sizeof(T),
                                 __LINE__);

        return(false);
    }
    
    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
    {
        T *&tmp_ptr_coefficient_matrix(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_coefficient_matrix[input_index_received * this->p_number_recurrent_depth + tmp_time_step_index] : this->p_ptr_output_coefficient_matrix[input_index_received * this->p_number_recurrent_depth + tmp_time_step_index]);
        
        tmp_coefficient_matrix_size = type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_array_coefficient_matrix_size[input_index_received] : this->p_ptr_output_array_coefficient_matrix_size[input_index_received];

        // Get timed input.
        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            tmp_ptr_array_inputs_preproced[tmp_example_index] = tmp_ptr_array_inputs[tmp_example_index * tmp_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_input_size + input_index_received];
        }

        if(MODWT<T>(this->p_number_examples,
                              tmp_coefficient_matrix_size,
                              tmp_ptr_array_inputs_preproced,
                              tmp_ptr_coefficient_matrix,
                              tmp_J_level) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"MODWT(%zu, %zu, ptr, ptr, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_examples,
                                     tmp_coefficient_matrix_size,
                                     tmp_J_level,
                                     __LINE__);

            return(false);
        }

        // Set timed input.
        tmp_ptr_array_smooth_coefficients = tmp_ptr_coefficient_matrix + tmp_J_level * this->p_number_examples;

        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            tmp_ptr_array_inputs[tmp_example_index * tmp_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_input_size + input_index_received] = tmp_ptr_array_smooth_coefficients[tmp_example_index];
        }
    }
    
    delete[](tmp_ptr_array_inputs_preproced);

    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) { this->p_ptr_input_array_coefficient_matrix_size[input_index_received] = tmp_coefficient_matrix_size; }
    else { this->p_ptr_output_array_coefficient_matrix_size[input_index_received] = tmp_coefficient_matrix_size; }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__MODWT(size_t const input_index_received,
                                                                T *const ptr_array_inputs_received,
                                                                enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_recurrent_depth == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->p_ptr_input_coefficient_matrix == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_input_coefficient_matrix\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        if(this->p_ptr_input_array_coefficient_matrix_size == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_input_array_coefficient_matrix_size\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->p_ptr_output_coefficient_matrix == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_output_coefficient_matrix\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        if(this->p_ptr_output_array_coefficient_matrix_size == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_output_array_coefficient_matrix_size\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    
    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }
    
    size_t const tmp_batch_size(this->p_number_examples + 1_zu),
                       tmp_input_coefficient_matrix_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_array_coefficient_matrix_size[input_index_received] : this->p_ptr_output_array_coefficient_matrix_size[input_index_received]);
    size_t tmp_J_level(tmp_input_coefficient_matrix_size / this->p_number_examples),
              tmp_coefficient_matrix_size,
              tmp_example_index,
              tmp_time_step_index,
              tmp_j_index;

    T const *tmp_ptr_source_coefficient_matrix;
    T *tmp_ptr_coefficient_matrix,
       *tmp_ptr_array_inputs_preproced,
       *tmp_ptr_array_smooth_coefficients;
    
    // Valid input index.
    if(tmp_J_level == 0_zu) { return(true); }
    else { --tmp_J_level; }
    // |END| Valid input index. |END|

    if((tmp_ptr_array_inputs_preproced = new T[tmp_batch_size]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_batch_size * sizeof(T),
                                 __LINE__);

        return(false);
    }
    
    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
    {
        tmp_ptr_source_coefficient_matrix = (type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_coefficient_matrix[input_index_received * this->p_number_recurrent_depth + tmp_time_step_index] : this->p_ptr_output_coefficient_matrix[input_index_received * this->p_number_recurrent_depth + tmp_time_step_index]);
        tmp_ptr_coefficient_matrix = nullptr;

        tmp_coefficient_matrix_size = 0_zu;

        // Get timed input from dataset.
        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            tmp_ptr_array_inputs_preproced[tmp_example_index] = tmp_ptr_source_coefficient_matrix[tmp_example_index];
        }
        for(tmp_j_index = 1_zu; tmp_j_index != tmp_J_level + 1_zu; ++tmp_j_index)
        {
            for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
            {
                tmp_ptr_array_inputs_preproced[tmp_example_index] += tmp_ptr_source_coefficient_matrix[tmp_j_index * this->p_number_examples + tmp_example_index];
            }
        }
        // |END| Get timed input from dataset. |END|

        // Get timed input from arguments.
        tmp_ptr_array_inputs_preproced[tmp_example_index] = ptr_array_inputs_received[tmp_time_step_index * tmp_input_size + input_index_received];

        if(MODWT<T>(tmp_batch_size,
                              tmp_coefficient_matrix_size,
                              tmp_ptr_array_inputs_preproced,
                              tmp_ptr_coefficient_matrix,
                              tmp_J_level) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"MODWT(%zu, %zu, ptr, ptr, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_batch_size,
                                     tmp_coefficient_matrix_size,
                                     tmp_J_level,
                                     __LINE__);

            return(false);
        }
        
        // Shift array for continious access.
        tmp_ptr_array_smooth_coefficients = tmp_ptr_coefficient_matrix + tmp_J_level * tmp_batch_size;

        // Set timed input from arguments.
        ptr_array_inputs_received[tmp_time_step_index * tmp_input_size + input_index_received] = tmp_ptr_array_smooth_coefficients[tmp_example_index];

        delete[](tmp_ptr_coefficient_matrix);
    }
    
    // Delete tempory inputs storage.
    delete[](tmp_ptr_array_inputs_preproced);
    // |END| Delete tempory inputs storage. |END|

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__MODWT_Inverse(enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No enought data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_recurrent_depth == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->p_ptr_input_coefficient_matrix == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_input_coefficient_matrix\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        
        if(this->p_ptr_input_array_coefficient_matrix_size == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_input_array_coefficient_matrix_size\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->p_ptr_output_coefficient_matrix == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_output_coefficient_matrix\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        
        if(this->p_ptr_output_array_coefficient_matrix_size == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_output_array_coefficient_matrix_size\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    for(tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        if(this->Preprocessing__MODWT_Inverse(tmp_input_index, type_input_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT_Inverse(%zu, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_index,
                                     type_input_received,
                                     __LINE__);
            
            return(false);
        }
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__MODWT_Inverse(size_t const input_index_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No enought data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_recurrent_depth == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->p_ptr_input_coefficient_matrix == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_input_coefficient_matrix\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        
        if(this->p_ptr_input_array_coefficient_matrix_size == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_input_array_coefficient_matrix_size\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->p_ptr_output_coefficient_matrix == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_output_coefficient_matrix\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        
        if(this->p_ptr_output_array_coefficient_matrix_size == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_output_array_coefficient_matrix_size\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    
    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }

    size_t const tmp_coefficient_matrix_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_array_coefficient_matrix_size[input_index_received] : this->p_ptr_output_array_coefficient_matrix_size[input_index_received]);
    size_t tmp_example_index,
              tmp_time_step_index;

    T *tmp_ptr_array_inputs(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_array_inputs : this->p_ptr_array_outputs),
       *tmp_ptr_array_inputs_inverse;
    
    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_array_inputs\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if((tmp_ptr_array_inputs_inverse = new T[this->p_number_examples]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_examples * sizeof(T),
                                 __LINE__);

        return(false);
    }
    
    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
    {
        T *&tmp_ptr_coefficient_matrix(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_coefficient_matrix[input_index_received * this->p_number_recurrent_depth + tmp_time_step_index] : this->p_ptr_output_coefficient_matrix[input_index_received * this->p_number_recurrent_depth + tmp_time_step_index]);
        
        if(MODWT_Inverse<T>(tmp_coefficient_matrix_size,
                                          this->p_number_examples,
                                          tmp_ptr_coefficient_matrix,
                                          tmp_ptr_array_inputs_inverse) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"MODWT_Inverse(%zu, %zu, ptr, ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_examples,
                                     type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_array_coefficient_matrix_size[input_index_received] : this->p_ptr_output_array_coefficient_matrix_size[input_index_received],
                                     __LINE__);
            
            return(false);
        }

        // Set timed input.
        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            tmp_ptr_array_inputs[tmp_example_index * tmp_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_input_size + input_index_received] = tmp_ptr_array_inputs_inverse[tmp_example_index];
        }

        SAFE_DELETE_ARRAY(tmp_ptr_coefficient_matrix)
    }

    delete[](tmp_ptr_array_inputs_inverse);
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) { this->p_ptr_input_array_coefficient_matrix_size[input_index_received] = 0_zu; }
    else { this->p_ptr_output_array_coefficient_matrix_size[input_index_received] = 0_zu; }

    return(true);
}

template<typename T>
bool Shift(size_t const start_index_received,
              size_t const end_index_received,
              size_t const shift_received,
              T *const ptr_array_received)
{
    for(size_t tmp_input_index(start_index_received); tmp_input_index != end_index_received; --tmp_input_index) { ptr_array_received[tmp_input_index + shift_received] = ptr_array_received[tmp_input_index]; }
    
    return(true);
}

template<typename T>
bool Shift(size_t const row_start_index_received,
              size_t const row_end_index_received,
              size_t const columns_size_received,
              size_t const shift_received,
              T *const ptr_array_received)
{
    for(size_t tmp_row(row_start_index_received),
                  tmp_column; tmp_row != row_end_index_received; --tmp_row)
    {
        for(tmp_column = 0_zu; tmp_column != columns_size_received; ++tmp_column)
        {
            ptr_array_received[(tmp_row + shift_received) * columns_size_received + tmp_column] = ptr_array_received[tmp_row * columns_size_received + tmp_column];
        }
    }
    
    return(true);
}

template<typename T>
bool Dataset<T>::Shift_Arrays(size_t const input_index_received,
                                             size_t const shift_size_received,
                                             enum ENUM_TYPE_INPUT const type_input_received)
{
    size_t const tmp_new_input_size((type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs) + shift_size_received);
    size_t tmp_example_index,
              tmp_time_step_index,
              tmp_input_index;
    
    T *tmp_ptr_array_inputs;

    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->p_ptr_input_coefficient_matrix != nullptr)
        {
            this->p_ptr_input_coefficient_matrix = MyEA::Memory::Cpp::Reallocate_PtOfPt<T*, true>(this->p_ptr_input_coefficient_matrix,
                                                                                                                                    tmp_new_input_size * this->p_number_recurrent_depth,
                                                                                                                                    this->p_number_inputs * this->p_number_recurrent_depth);
            if(this->p_ptr_input_coefficient_matrix == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_pointers_array_cpp<%zu>(ptr, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         sizeof(T*),
                                         tmp_new_input_size * this->p_number_recurrent_depth,
                                         this->p_number_inputs * this->p_number_recurrent_depth,
                                         __LINE__);

                return(false);
            }
            else if(Shift<T*>(this->p_number_inputs - 1_zu,
                                    input_index_received,
                                    this->p_number_recurrent_depth,
                                    shift_size_received,
                                    this->p_ptr_input_coefficient_matrix) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shift(%zu, %zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_inputs - 1_zu,
                                         input_index_received,
                                         this->p_number_recurrent_depth,
                                         shift_size_received,
                                         __LINE__);
            
                return(false);
            }
            MyEA::Memory::Cpp::Fill_Nullptr<T*>(this->p_ptr_input_coefficient_matrix + (input_index_received + 1_zu) * this->p_number_recurrent_depth,
                                                 this->p_ptr_input_coefficient_matrix + (input_index_received + 1_zu + shift_size_received) * this->p_number_recurrent_depth);
        }

        if(this->p_ptr_input_array_coefficient_matrix_size != nullptr)
        {
            this->p_ptr_input_array_coefficient_matrix_size = MyEA::Memory::Cpp::Reallocate<size_t, true>(this->p_ptr_input_array_coefficient_matrix_size,
                                                                                                                                     tmp_new_input_size,
                                                                                                                                     this->p_number_inputs);
            if(this->p_ptr_input_array_coefficient_matrix_size == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         sizeof(size_t),
                                         tmp_new_input_size,
                                         this->p_number_inputs,
                                         __LINE__);

                return(false);
            }
            else if(Shift<size_t>(this->p_number_inputs - 1_zu,
                                          input_index_received,
                                          shift_size_received,
                                          this->p_ptr_input_array_coefficient_matrix_size) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shift(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_inputs - 1_zu,
                                         input_index_received,
                                         shift_size_received,
                                         __LINE__);
            
                return(false);
            }
            memset(this->p_ptr_input_array_coefficient_matrix_size + (input_index_received + 1_zu),
                         0,
                         shift_size_received * sizeof(size_t));
        }

        if(this->p_ptr_array_inputs != nullptr)
        {
            if((tmp_ptr_array_inputs = new T[this->p_number_examples * this->p_number_recurrent_depth * tmp_new_input_size]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_examples * this->p_number_recurrent_depth * tmp_new_input_size * sizeof(T),
                                         __LINE__);

                return(false);
            }
            MEMSET(tmp_ptr_array_inputs,
                           0,
                           this->p_number_examples * this->p_number_recurrent_depth * tmp_new_input_size * sizeof(T));
            for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
            {
                this->p_ptr_array_inputs_array[tmp_example_index] = tmp_ptr_array_inputs + tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth;

                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
                {
                    // Left inputs.
                    for(tmp_input_index = 0_zu; tmp_input_index != input_index_received + 1_zu; ++tmp_input_index)
                    {
                        tmp_ptr_array_inputs[tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_new_input_size + tmp_input_index] = this->p_ptr_array_inputs[tmp_example_index * this->p_number_inputs * this->p_number_recurrent_depth + tmp_time_step_index * this->p_number_inputs + tmp_input_index];
                    }

                    // Right inputs.
                    for(tmp_input_index = this->p_number_inputs - 1_zu; tmp_input_index != input_index_received; --tmp_input_index)
                    {
                        tmp_ptr_array_inputs[tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_new_input_size + tmp_input_index + shift_size_received] = this->p_ptr_array_inputs[tmp_example_index * this->p_number_inputs * this->p_number_recurrent_depth + tmp_time_step_index * this->p_number_inputs + tmp_input_index];
                    }
                }
            }
            delete[](this->p_ptr_array_inputs);
            this->p_ptr_array_inputs = tmp_ptr_array_inputs;
        }
        
        if(this->_ptr_input_array_scaler__zero_centered != nullptr)
        {
            this->_ptr_input_array_scaler__zero_centered = MyEA::Memory::Cpp::Reallocate_Objects<struct Scaler__Zero_Centered<T>, true>(this->_ptr_input_array_scaler__zero_centered,
                                                                                                                                                                                       tmp_new_input_size,
                                                                                                                                                                                       this->p_number_inputs);
            if(this->_ptr_input_array_scaler__zero_centered == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         sizeof(struct Scaler__Zero_Centered<T>),
                                         tmp_new_input_size,
                                         this->p_number_inputs,
                                         __LINE__);

                return(false);
            }
            else if(Shift<struct Scaler__Zero_Centered<T>>(this->p_number_inputs - 1_zu,
                                                                                  input_index_received,
                                                                                  shift_size_received,
                                                                                  this->_ptr_input_array_scaler__zero_centered) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shift(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_inputs - 1_zu,
                                         input_index_received,
                                         shift_size_received,
                                         __LINE__);
            
                return(false);
            }
            MyEA::Memory::Fill<struct Scaler__Zero_Centered<T>>(this->_ptr_input_array_scaler__zero_centered + (input_index_received + 1_zu),
                                                                                    this->_ptr_input_array_scaler__zero_centered + (input_index_received + 1_zu + shift_size_received),
                                                                                    Scaler__Zero_Centered<T>());
        }
        
        if(this->_ptr_input_array_scaler__minimum_maximum != nullptr)
        {
            this->_ptr_input_array_scaler__minimum_maximum = MyEA::Memory::Cpp::Reallocate_Objects<struct Scaler__Minimum_Maximum<T>, true>(this->_ptr_input_array_scaler__minimum_maximum,
                                                                                                                                                                                                        tmp_new_input_size,
                                                                                                                                                                                                        this->p_number_inputs);
            if(this->_ptr_input_array_scaler__minimum_maximum == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         sizeof(struct Scaler__Minimum_Maximum<T>),
                                         tmp_new_input_size,
                                         this->p_number_inputs,
                                         __LINE__);

                return(false);
            }
            else if(Shift<struct Scaler__Minimum_Maximum<T>>(this->p_number_inputs - 1_zu,
                                                                                          input_index_received,
                                                                                          shift_size_received,
                                                                                          this->_ptr_input_array_scaler__minimum_maximum) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shift(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_inputs - 1_zu,
                                         input_index_received,
                                         shift_size_received,
                                         __LINE__);
            
                return(false);
            }
            MyEA::Memory::Fill<struct Scaler__Minimum_Maximum<T>>(this->_ptr_input_array_scaler__minimum_maximum + (input_index_received + 1_zu),
                                                                                            this->_ptr_input_array_scaler__minimum_maximum + (input_index_received + 1_zu + shift_size_received),
                                                                                            Scaler__Minimum_Maximum<T>());
        }

        this->p_number_inputs += shift_size_received;
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->p_ptr_output_coefficient_matrix != nullptr)
        {
            this->p_ptr_output_coefficient_matrix = MyEA::Memory::Cpp::Reallocate_PtOfPt<T*, true>(this->p_ptr_output_coefficient_matrix,
                                                                                                                                      tmp_new_input_size * this->p_number_recurrent_depth,
                                                                                                                                      this->p_number_outputs * this->p_number_recurrent_depth);
            if(this->p_ptr_output_coefficient_matrix == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_pointers_array_cpp<%zu>(ptr, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         sizeof(T*),
                                         tmp_new_input_size * this->p_number_recurrent_depth,
                                         this->p_number_outputs * this->p_number_recurrent_depth,
                                         __LINE__);

                return(false);
            }
            else if(Shift<T*>(this->p_number_outputs - 1_zu,
                                    input_index_received,
                                    this->p_number_recurrent_depth,
                                    shift_size_received,
                                    this->p_ptr_output_coefficient_matrix) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shift(%zu, %zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_outputs - 1_zu,
                                         input_index_received,
                                         this->p_number_recurrent_depth,
                                         shift_size_received,
                                         __LINE__);
            
                return(false);
            }
            MyEA::Memory::Cpp::Fill_Nullptr<T*>(this->p_ptr_output_coefficient_matrix + (input_index_received + 1_zu) * this->p_number_recurrent_depth,
                                                 this->p_ptr_output_coefficient_matrix + (input_index_received + 1_zu + shift_size_received) * this->p_number_recurrent_depth);
        }
        
        if(this->p_ptr_output_array_coefficient_matrix_size != nullptr)
        {
            this->p_ptr_output_array_coefficient_matrix_size = MyEA::Memory::Cpp::Reallocate<size_t, true>(this->p_ptr_output_array_coefficient_matrix_size,
                                                                                                                                       tmp_new_input_size,
                                                                                                                                       this->p_number_outputs);
            if(this->p_ptr_output_array_coefficient_matrix_size == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         sizeof(size_t),
                                         tmp_new_input_size,
                                         this->p_number_outputs,
                                         __LINE__);

                return(false);
            }
            else if(Shift<size_t>(this->p_number_outputs - 1_zu,
                                          input_index_received,
                                          shift_size_received,
                                          this->p_ptr_output_array_coefficient_matrix_size) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shift(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_outputs - 1_zu,
                                         input_index_received,
                                         shift_size_received,
                                         __LINE__);
            
                return(false);
            }
            memset(this->p_ptr_output_array_coefficient_matrix_size + (input_index_received + 1_zu),
                         0,
                         shift_size_received * sizeof(size_t));
        }
        
        if(this->p_ptr_array_outputs != nullptr)
        {
            if((tmp_ptr_array_inputs = new T[this->p_number_examples * this->p_number_recurrent_depth * tmp_new_input_size]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_examples * this->p_number_recurrent_depth * tmp_new_input_size * sizeof(T),
                                         __LINE__);

                return(false);
            }
            MEMSET(tmp_ptr_array_inputs,
                           0,
                           this->p_number_examples * this->p_number_recurrent_depth * tmp_new_input_size * sizeof(T));
            for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
            {
                this->p_ptr_array_outputs_array[tmp_example_index] = tmp_ptr_array_inputs + tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth;

                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
                {
                    // Left inputs.
                    for(tmp_input_index = 0_zu; tmp_input_index != input_index_received + 1_zu; ++tmp_input_index)
                    {
                        tmp_ptr_array_inputs[tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_new_input_size + tmp_input_index] = this->p_ptr_array_outputs[tmp_example_index * this->p_number_outputs * this->p_number_recurrent_depth + tmp_time_step_index * this->p_number_outputs + tmp_input_index];
                    }

                    // Right inputs.
                    for(tmp_input_index = this->p_number_outputs - 1_zu; tmp_input_index != input_index_received; --tmp_input_index)
                    {
                        tmp_ptr_array_inputs[tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_new_input_size + tmp_input_index + shift_size_received] = this->p_ptr_array_outputs[tmp_example_index * this->p_number_outputs * this->p_number_recurrent_depth + tmp_time_step_index * this->p_number_outputs + tmp_input_index];
                    }
                }
            }
            delete[](this->p_ptr_array_outputs);
            this->p_ptr_array_outputs = tmp_ptr_array_inputs;
        }
        
        if(this->_ptr_output_array_scaler__zero_centered != nullptr)
        {
            this->_ptr_output_array_scaler__zero_centered = MyEA::Memory::Cpp::Reallocate_Objects<struct Scaler__Zero_Centered<T>, true>(this->_ptr_output_array_scaler__zero_centered,
                                                                                                                                                                                         tmp_new_input_size,
                                                                                                                                                                                         this->p_number_outputs);
            if(this->_ptr_output_array_scaler__zero_centered == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         sizeof(struct Scaler__Zero_Centered<T>),
                                         tmp_new_input_size,
                                         this->p_number_outputs,
                                         __LINE__);

                return(false);
            }
            else if(Shift<struct Scaler__Zero_Centered<T>>(this->p_number_outputs - 1_zu,
                                                                                  input_index_received,
                                                                                  shift_size_received,
                                                                                  this->_ptr_output_array_scaler__zero_centered) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shift(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_outputs - 1_zu,
                                         input_index_received,
                                         shift_size_received,
                                         __LINE__);
            
                return(false);
            }
            MyEA::Memory::Fill<struct Scaler__Zero_Centered<T>>(this->_ptr_output_array_scaler__zero_centered + (input_index_received + 1_zu),
                                                                                    this->_ptr_output_array_scaler__zero_centered + (input_index_received + 1_zu + shift_size_received),
                                                                                    Scaler__Zero_Centered<T>());
        }
        
        if(this->_ptr_output_array_scaler__minimum_maximum != nullptr)
        {
            this->_ptr_output_array_scaler__minimum_maximum = MyEA::Memory::Cpp::Reallocate_Objects<struct Scaler__Minimum_Maximum<T>, true>(this->_ptr_output_array_scaler__minimum_maximum,
                                                                                                                                                                                                          tmp_new_input_size,
                                                                                                                                                                                                          this->p_number_inputs);
            if(this->_ptr_output_array_scaler__minimum_maximum == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"reallocate_objects_cpp<%zu>(ptr, %zu, %zu, true)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         sizeof(struct Scaler__Minimum_Maximum<T>),
                                         tmp_new_input_size,
                                         this->p_number_inputs,
                                         __LINE__);

                return(false);
            }
            else if(Shift<struct Scaler__Minimum_Maximum<T>>(this->p_number_outputs - 1_zu,
                                                                                          input_index_received,
                                                                                          shift_size_received,
                                                                                          this->_ptr_output_array_scaler__minimum_maximum) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shift(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         this->p_number_outputs - 1_zu,
                                         input_index_received,
                                         shift_size_received,
                                         __LINE__);
            
                return(false);
            }
            MyEA::Memory::Fill<struct Scaler__Minimum_Maximum<T>>(this->_ptr_output_array_scaler__minimum_maximum + (input_index_received + 1_zu),
                                                                                            this->_ptr_output_array_scaler__minimum_maximum + (input_index_received + 1_zu + shift_size_received),
                                                                                            Scaler__Minimum_Maximum<T>());
        }

        this->p_number_outputs += shift_size_received;
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Merge__MODWT(size_t const desired_J_level_received, enum ENUM_TYPE_INPUT const type_input_received)
{
    if(this->p_number_examples <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No enought data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_recurrent_depth == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(desired_J_level_received == 0_zu) { return(true); }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index,
              tmp_shift_index;
    
    for(tmp_shift_index = 0_zu,
        tmp_input_index = 0_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        if(this->Preprocessing__Merge__MODWT(tmp_input_index + tmp_shift_index,
                                                                     desired_J_level_received,
                                                                     type_input_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT(%zu, %zu, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_index + tmp_shift_index,
                                     desired_J_level_received,
                                     type_input_received,
                                     __LINE__);
            
            return(false);
        }

        tmp_shift_index += desired_J_level_received;
    }

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Merge__MODWT(size_t const input_index_received,
                                                                            size_t const desired_J_level_received,
                                                                            enum ENUM_TYPE_INPUT const type_input_received)
{
    // Valid dataset.
    if(this->p_number_examples <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No enought data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_recurrent_depth == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(desired_J_level_received == 0_zu) { return(true); }
    // |END| Valid dataset. |END|
    
    // Variables.
    size_t const tmp_J_level(MyEA::Math::Minimum<size_t>(this->MODWT__J_Level_Maximum(), desired_J_level_received));
    size_t tmp_coefficient_matrix_size,
              tmp_example_index,
              tmp_time_step_index,
              tmp_j_index;
    // |END| Variables. |END|
    
    // Valid input index.
    size_t const tmp_new_input_size((type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs) + tmp_J_level);
    
    if(input_index_received >= tmp_new_input_size - tmp_J_level)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_new_input_size - tmp_J_level,
                                 __LINE__);
        
        return(false);
    }
    // |END| Valid input index. |END|

    // Allocate || Reallocate.
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->Shift_Arrays(input_index_received,
                                      tmp_J_level,
                                      ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shift_Arrays(%zu, %zu, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     input_index_received,
                                     tmp_J_level,
                                     ENUM_TYPE_INPUT::TYPE_INPUT_INPUT,
                                     __LINE__);

            return(false);
        }

        // Allocate.
        if(this->p_ptr_input_coefficient_matrix == nullptr)
        {
            if((this->p_ptr_input_coefficient_matrix = new T*[tmp_new_input_size * this->p_number_recurrent_depth]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_new_input_size * sizeof(T*),
                                         __LINE__);

                return(false);
            }
            MyEA::Memory::Cpp::Fill_Nullptr<T*>(this->p_ptr_input_coefficient_matrix, this->p_ptr_input_coefficient_matrix + tmp_new_input_size * this->p_number_recurrent_depth);
        }

        // Allocate.
        if(this->p_ptr_input_array_coefficient_matrix_size == nullptr)
        {
            if((this->p_ptr_input_array_coefficient_matrix_size = new size_t[tmp_new_input_size]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_new_input_size * sizeof(size_t),
                                         __LINE__);

                return(false);
            }
            memset(this->p_ptr_input_array_coefficient_matrix_size,
                         0,
                         tmp_new_input_size * sizeof(size_t));
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->Shift_Arrays(input_index_received,
                                      tmp_J_level,
                                      ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Shift_Arrays(%zu, %zu, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     input_index_received,
                                     tmp_J_level,
                                     ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT,
                                     __LINE__);

            return(false);
        }

        // Allocate.
        if(this->p_ptr_output_coefficient_matrix == nullptr)
        {
            if((this->p_ptr_output_coefficient_matrix = new T*[tmp_new_input_size * this->p_number_recurrent_depth]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_new_input_size * sizeof(T*),
                                         __LINE__);

                return(false);
            }
            MyEA::Memory::Cpp::Fill_Nullptr<T*>(this->p_ptr_output_coefficient_matrix, this->p_ptr_output_coefficient_matrix + tmp_new_input_size * this->p_number_recurrent_depth);
        }
        
        // Allocate.
        if(this->p_ptr_output_array_coefficient_matrix_size == nullptr)
        {
            if((this->p_ptr_output_array_coefficient_matrix_size = new size_t[tmp_new_input_size]) == nullptr)
            {
                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_new_input_size * sizeof(size_t),
                                         __LINE__);

                return(false);
            }
            memset(this->p_ptr_output_array_coefficient_matrix_size,
                         0,
                         tmp_new_input_size * sizeof(size_t));
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    // |END| Allocate || Reallocate. |END|
    
    // Allocate tempory inputs storage.
    T *const tmp_ptr_array_inputs(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_array_inputs : this->p_ptr_array_outputs),
       *tmp_ptr_array_inputs_preproced;
    
    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_array_inputs\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    if((tmp_ptr_array_inputs_preproced = new T[this->p_number_examples]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_examples * sizeof(T),
                                 __LINE__);

        return(false);
    }
    // |END| Allocate tempory inputs storage. |END|
    
    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
    {
        T *&tmp_ptr_coefficient_matrix(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_coefficient_matrix[input_index_received * this->p_number_recurrent_depth + tmp_time_step_index] : this->p_ptr_output_coefficient_matrix[input_index_received * this->p_number_recurrent_depth + tmp_time_step_index]);
        
        tmp_coefficient_matrix_size = type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_array_coefficient_matrix_size[input_index_received] : this->p_ptr_output_array_coefficient_matrix_size[input_index_received];

        // Get timed input.
        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            tmp_ptr_array_inputs_preproced[tmp_example_index] = tmp_ptr_array_inputs[tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_new_input_size + input_index_received];
        }

        if(MODWT<T>(this->p_number_examples,
                              tmp_coefficient_matrix_size,
                              tmp_ptr_array_inputs_preproced,
                              tmp_ptr_coefficient_matrix,
                              tmp_J_level) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"MODWT(%zu, %zu, ptr, ptr, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->p_number_examples,
                                     tmp_coefficient_matrix_size,
                                     tmp_J_level,
                                     __LINE__);

            return(false);
        }

        // Set timed input.
        for(tmp_j_index = 0_zu; tmp_j_index != tmp_J_level + 1_zu; ++tmp_j_index)
        {
            for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
            {
                tmp_ptr_array_inputs[tmp_example_index * tmp_new_input_size * this->p_number_recurrent_depth + tmp_time_step_index * tmp_new_input_size + input_index_received + tmp_j_index] = tmp_ptr_coefficient_matrix[tmp_j_index * this->p_number_examples + tmp_example_index];
            }
        }
    }
    
    // Delete tempory inputs storage.
    delete[](tmp_ptr_array_inputs_preproced);
    // |END| Delete tempory inputs storage. |END|

    // Store the matrix size.
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) { this->p_ptr_input_array_coefficient_matrix_size[input_index_received] = tmp_coefficient_matrix_size; }
    else { this->p_ptr_output_array_coefficient_matrix_size[input_index_received] = tmp_coefficient_matrix_size; }
    // |END| Store the matrix size. |END|

    this->Compute__Start_Index();

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Merge__MODWT(size_t const input_index_received,
                                                                            size_t const input_size_received,
                                                                            T *&ptr_array_inputs_received,
                                                                            enum ENUM_TYPE_INPUT const type_input_received)
{
    // Valid dataset.
    if(this->p_number_examples <= 1_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No enought data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_recurrent_depth == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Recurrent depth can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(this->_reference)
    {
        PRINT_FORMAT("%s: %s: ERROR: The dataset is allocate as refence. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    
    if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT)
    {
        if(this->p_ptr_input_coefficient_matrix == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_input_coefficient_matrix\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        if(this->p_ptr_input_array_coefficient_matrix_size == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_input_array_coefficient_matrix_size\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        if(this->p_ptr_output_coefficient_matrix == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_output_coefficient_matrix\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }

        if(this->p_ptr_output_array_coefficient_matrix_size == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: \"p_ptr_output_array_coefficient_matrix_size\" is a nullptr. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(false);
    }
    // |END| Valid dataset. |END|
    
    // Variables.
    size_t const tmp_batch_size(this->p_number_examples + 1_zu),
                       tmp_input_coefficient_matrix_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_array_coefficient_matrix_size[input_index_received] : this->p_ptr_output_array_coefficient_matrix_size[input_index_received]);
    size_t tmp_J_level(tmp_input_coefficient_matrix_size / this->p_number_examples),
              tmp_coefficient_matrix_size,
              tmp_example_index,
              tmp_time_step_index,
              tmp_input_index,
              tmp_j_index;
    // |END| Variables. |END|
    
    // Valid input index.
    if(tmp_J_level == 0_zu) { return(true); }
    else { --tmp_J_level; }

    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    
    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(false);
    }
    else if(input_index_received >= input_size_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 input_size_received,
                                 __LINE__);
        
        return(false);
    }
    // |END| Valid input index. |END|

    // Reallocate.
    size_t const tmp_new_input_size(input_size_received + tmp_J_level);

    T *tmp_ptr_array_inputs;
    if((tmp_ptr_array_inputs = new T[this->p_number_recurrent_depth * tmp_new_input_size]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_recurrent_depth * tmp_new_input_size * sizeof(T),
                                 __LINE__);

        return(false);
    }
    
    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
    {
        // Left inputs.
        for(tmp_input_index = 0_zu; tmp_input_index != input_index_received + 1_zu; ++tmp_input_index)
        {
            tmp_ptr_array_inputs[tmp_time_step_index * tmp_new_input_size + tmp_input_index] = ptr_array_inputs_received[tmp_time_step_index * input_size_received + tmp_input_index];
        }

        // Right inputs.
        for(tmp_input_index = input_size_received - 1_zu; tmp_input_index != input_index_received; --tmp_input_index)
        {
            tmp_ptr_array_inputs[tmp_time_step_index * tmp_new_input_size + tmp_input_index + tmp_J_level] = ptr_array_inputs_received[tmp_time_step_index * input_size_received + tmp_input_index];
        }
    }

    delete[](ptr_array_inputs_received);
    ptr_array_inputs_received = tmp_ptr_array_inputs;
    // |END| Reallocate. |END|
    
    // Allocate tempory inputs storage.
    T const *tmp_ptr_source_coefficient_matrix;
    T *tmp_ptr_array_inputs_preproced,
       *tmp_ptr_coefficient_matrix;
    
    if((tmp_ptr_array_inputs_preproced = new T[tmp_batch_size]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_batch_size * sizeof(T),
                                 __LINE__);

        return(false);
    }
    // |END| Allocate tempory inputs storage. |END|
    
    for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
    {
        tmp_ptr_source_coefficient_matrix = (type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_ptr_input_coefficient_matrix[input_index_received * this->p_number_recurrent_depth + tmp_time_step_index] : this->p_ptr_output_coefficient_matrix[input_index_received * this->p_number_recurrent_depth + tmp_time_step_index]);
        tmp_ptr_coefficient_matrix = nullptr;
        
        tmp_coefficient_matrix_size = 0_zu;
        
        // Get timed input from dataset.
        for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
        {
            tmp_ptr_array_inputs_preproced[tmp_example_index] = tmp_ptr_source_coefficient_matrix[tmp_example_index];
        }
        for(tmp_j_index = 1_zu; tmp_j_index != tmp_J_level + 1_zu; ++tmp_j_index)
        {
            for(tmp_example_index = 0_zu; tmp_example_index != this->p_number_examples; ++tmp_example_index)
            {
                tmp_ptr_array_inputs_preproced[tmp_example_index] += tmp_ptr_source_coefficient_matrix[tmp_j_index * this->p_number_examples + tmp_example_index];
            }
        }
        // |END| Get timed input from dataset. |END|

        // Get timed input from arguments.
        tmp_ptr_array_inputs_preproced[tmp_example_index] = tmp_ptr_array_inputs[tmp_time_step_index * tmp_new_input_size + input_index_received];

        if(MODWT<T>(tmp_batch_size,
                              tmp_coefficient_matrix_size,
                              tmp_ptr_array_inputs_preproced,
                              tmp_ptr_coefficient_matrix,
                              tmp_J_level) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"MODWT(%zu, %zu, ptr, ptr, %zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_batch_size,
                                     tmp_coefficient_matrix_size,
                                     tmp_J_level,
                                     __LINE__);

            return(false);
        }

        // Set timed input into arguments.
        for(tmp_j_index = 0_zu; tmp_j_index != tmp_J_level + 1_zu; ++tmp_j_index)
        {
            tmp_ptr_array_inputs[tmp_time_step_index * tmp_new_input_size + input_index_received + tmp_j_index] = tmp_ptr_coefficient_matrix[tmp_j_index * tmp_batch_size + tmp_example_index];
        }

        delete[](tmp_ptr_coefficient_matrix);
    }
    
    // Delete tempory inputs storage.
    delete[](tmp_ptr_array_inputs_preproced);
    // |END| Delete tempory inputs storage. |END|

    return(true);
}

template<typename T>
bool Dataset<T>::Preprocessing__Sequence_Window(size_t const sequence_window_received,
                                                                               size_t const sequence_horizon_received,
                                                                               T *&ptr_array_inputs_received)
{
    if(sequence_window_received <= 1_zu) { return(true); }
    else if(sequence_horizon_received == 0_zu) { return(true); }

    T const *const tmp_ptr_array_previous_inputs(this->p_ptr_array_inputs + this->p_number_examples * this->p_number_inputs - ((sequence_window_received - 1_zu) + (sequence_horizon_received - 1_zu)) * this->p_number_inputs);
    T *tmp_ptr_array_inputs;

    if((tmp_ptr_array_inputs = new T[sequence_window_received * this->p_number_inputs]) == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 sequence_window_received * this->p_number_inputs * sizeof(T),
                                 __LINE__);

        return(false);
    }
    
    for(size_t tmp_index(0_zu); tmp_index != (sequence_window_received - 1_zu) * this->p_number_inputs; ++tmp_index)
    {
        tmp_ptr_array_inputs[tmp_index] = tmp_ptr_array_previous_inputs[tmp_index];
    }

    MEMCPY(tmp_ptr_array_inputs + (sequence_window_received - 1_zu) * this->p_number_inputs,
                    ptr_array_inputs_received,
                    this->p_number_inputs * sizeof(T));
    
    delete[](ptr_array_inputs_received);

    ptr_array_inputs_received = tmp_ptr_array_inputs;

    return(true);
}

template<typename T>
bool Dataset<T>::Check_Topology(size_t const number_inputs_received,
                                                   size_t const number_outputs_received,
                                                   size_t const number_recurrent_depth_received) const
{
    if(this->p_number_inputs != number_inputs_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of inputs (%zu) differ from the number of inputs received as argument (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_inputs,
                                 number_inputs_received,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_outputs != number_outputs_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of outputs (%zu) differ from the number of outputs received as argument (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_outputs,
                                 number_outputs_received,
                                 __LINE__);

        return(false);
    }
    else if(this->p_number_recurrent_depth != number_recurrent_depth_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of recurrent depth (%zu) differ from the number of recurrent depth received as argument (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->p_number_recurrent_depth,
                                 number_recurrent_depth_received,
                                 __LINE__);

        return(false);
    }
    
    return(true);
}

template<typename T>
bool Dataset<T>::Get__Reference(void) const { return(this->_reference); }

template<typename T>
bool Dataset<T>::Use__Multi_Label(void) const { return(this->_use_multi_label); }

template<typename T>
bool Dataset<T>::Deallocate(void)
{
    if(this->_reference)
    {
        this->p_ptr_array_inputs = nullptr;
        this->p_ptr_array_inputs_array = nullptr;

        this->p_ptr_array_outputs = nullptr;
        this->p_ptr_array_outputs_array = nullptr;
        
        this->p_ptr_input_coefficient_matrix = nullptr;
        this->p_ptr_output_coefficient_matrix = nullptr;

        this->p_ptr_input_array_coefficient_matrix_size = nullptr;
        this->p_ptr_output_array_coefficient_matrix_size = nullptr;

        this->_ptr_input_array_scaler__minimum_maximum = nullptr;
        this->_ptr_output_array_scaler__minimum_maximum = nullptr;

        this->_ptr_input_array_scaler__zero_centered = nullptr;
        this->_ptr_output_array_scaler__zero_centered = nullptr;
    }
    else
    {
        size_t tmp_input_index;

        SAFE_DELETE_ARRAY(this->p_ptr_array_inputs);
        SAFE_DELETE_ARRAY(this->p_ptr_array_inputs_array);

        SAFE_DELETE_ARRAY(this->p_ptr_array_outputs);
        SAFE_DELETE_ARRAY(this->p_ptr_array_outputs_array);

        if(this->p_ptr_input_coefficient_matrix != nullptr)
        {
            for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_inputs * this->p_number_recurrent_depth; ++tmp_input_index)
            { SAFE_DELETE_ARRAY(this->p_ptr_input_coefficient_matrix[tmp_input_index]); }

            SAFE_DELETE_ARRAY(this->p_ptr_input_coefficient_matrix);
        }

        if(this->p_ptr_output_coefficient_matrix != nullptr)
        {
            for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_outputs * this->p_number_recurrent_depth; ++tmp_input_index)
            { SAFE_DELETE_ARRAY(this->p_ptr_output_coefficient_matrix[tmp_input_index]); }
            
            SAFE_DELETE_ARRAY(this->p_ptr_output_coefficient_matrix);
        }

        SAFE_DELETE_ARRAY(this->p_ptr_input_array_coefficient_matrix_size);
        SAFE_DELETE_ARRAY(this->p_ptr_output_array_coefficient_matrix_size);

        SAFE_DELETE_ARRAY(this->_ptr_input_array_scaler__minimum_maximum);
        SAFE_DELETE_ARRAY(this->_ptr_output_array_scaler__minimum_maximum);

        SAFE_DELETE_ARRAY(this->_ptr_input_array_scaler__zero_centered);
        SAFE_DELETE_ARRAY(this->_ptr_output_array_scaler__zero_centered);
    }

    return(true);
}

template<typename T>
size_t Dataset<T>::Get__Identical_Outputs(std::vector<T> const &ref_vector_identical_outputs_received)
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(0);
    }
    else if(ref_vector_identical_outputs_received.empty())
    {
        PRINT_FORMAT("%s: %s: ERROR: No entries to compare. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(0);
    }
    else if(ref_vector_identical_outputs_received.size() != this->p_number_outputs)
    {
        PRINT_FORMAT("%s: %s: ERROR: The number of entries (%zu) differs from the number of dataset outputs (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ref_vector_identical_outputs_received.size(),
                                 this->p_number_outputs,
                                 __LINE__);

        return(0);
    }

    size_t tmp_total_identical_entries(0);

    for(size_t tmp_index,
                  tmp_example_index(0); tmp_example_index != this->p_number_examples; ++tmp_example_index)
    {
        for(tmp_index = 0_zu; tmp_index != this->p_number_outputs; ++tmp_index)
        {
            if(this->p_ptr_array_outputs_array[tmp_example_index][tmp_index] == ref_vector_identical_outputs_received[tmp_index])
            { ++tmp_total_identical_entries; }
        }
    }

    return(tmp_total_identical_entries);
}

template<typename T>
size_t Dataset<T>::Get__Number_Examples(void) const { return(this->p_number_examples - this->p_start_index); }

template<typename T>
size_t Dataset<T>::Get__Number_Batch(void) const { return(1_zu); }

template<typename T>
size_t Dataset<T>::Get__Number_Inputs(void) const { return(this->p_number_inputs); }

template<typename T>
size_t Dataset<T>::Get__Number_Outputs(void) const { return(this->p_number_outputs); }

template<typename T>
size_t Dataset<T>::Get__Number_Recurrent_Depth(void) const { return(this->p_number_recurrent_depth); }

template<typename T>
size_t Dataset<T>::MODWT__J_Level_Maximum(void) const { return(static_cast<size_t>(floor(log(static_cast<double>(this->p_number_examples)) / log(2.0)))); }

template<typename T>
T Dataset<T>::Training(class Neural_Network *const ptr_Neural_Network_received)
{
    if(this->Check_Topology(ptr_Neural_Network_received->number_inputs,
                                        ptr_Neural_Network_received->number_outputs,
                                        ptr_Neural_Network_received->number_recurrent_depth) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Check_Topology(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_Neural_Network_received->number_inputs,
                                 ptr_Neural_Network_received->number_outputs,
                                 ptr_Neural_Network_received->number_recurrent_depth,
                                 __LINE__);

        return((std::numeric_limits<ST_>::max)());
    }

    // Initialize training.
    if(ptr_Neural_Network_received->ptr_array_derivatives_parameters == nullptr) { ptr_Neural_Network_received->Clear_Training_Arrays(); }

    T tmp_loss;
    
    ptr_Neural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING;
    
    if(ptr_Neural_Network_received->Set__Multi_Label(this->Use__Multi_Label()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Multi_Label(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->Use__Multi_Label() ? "true" : "false",
                                 __LINE__);

        return((std::numeric_limits<ST_>::max)());
    }
    else if(ptr_Neural_Network_received->Update__Batch_Size(this->Get__Number_Examples()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return((std::numeric_limits<ST_>::max)());
    }
    else if(ptr_Neural_Network_received->Initialized__Weight() == false && ptr_Neural_Network_received->Initialize__Weight(this) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__Weight(self)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return((std::numeric_limits<ST_>::max)());
    }

    if(ptr_Neural_Network_received->use_OpenMP && ptr_Neural_Network_received->is_OpenMP_initialized)
    {
        if(ptr_Neural_Network_received->Update__Thread_Size(this->Get__Number_Examples()) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->Get__Number_Examples(),
                                     __LINE__);

            return((std::numeric_limits<ST_>::max)());
        }
        
        omp_set_num_threads(static_cast<int>(ptr_Neural_Network_received->number_threads));
        
        tmp_loss = this->Training_OpenMP(ptr_Neural_Network_received);
    }
    else
    { tmp_loss = this->Training_Loop(ptr_Neural_Network_received); }

    ptr_Neural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_INFERENCE;

    return(tmp_loss);
}

template<typename T>
T Dataset<T>::Training_OpenMP(class Neural_Network *const ptr_Neural_Network_received)
{
    this->Train_Epoch_OpenMP(ptr_Neural_Network_received);

    ptr_Neural_Network_received->Update_Parameter__OpenMP(this->Get__Number_Examples(), this->Get__Number_Examples());
    
    ptr_Neural_Network_received->epoch_time_step += 1_T;

    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, this->Measure_Accuracy(this->Get__Number_Examples(),
                                                                                                                                                                                                                        this->Get__Input_Array(),
                                                                                                                                                                                                                        this->Get__Output_Array(),
                                                                                                                                                                                                                        ptr_Neural_Network_received));

    return(Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
}

template<typename T>
T Dataset<T>::Training_Loop(class Neural_Network *const ptr_Neural_Network_received)
{
    this->Train_Epoch_Loop(ptr_Neural_Network_received);

    ptr_Neural_Network_received->Update_Parameter__Loop(this->Get__Number_Examples(), this->Get__Number_Examples());
    
    ptr_Neural_Network_received->epoch_time_step += 1_T;

    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, this->Measure_Accuracy(this->Get__Number_Examples(),
                                                                                                                                                                                                                        this->Get__Input_Array(),
                                                                                                                                                                                                                        this->Get__Output_Array(),
                                                                                                                                                                                                                        ptr_Neural_Network_received));

    return(Cast_T(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)));
}

template<typename T>
T Dataset<T>::Measure_Accuracy(size_t const batch_size_received,
                                                   T const *const *const ptr_array_inputs_received,
                                                   T const *const *const ptr_array_desired_outputs_received,
                                                   class Neural_Network *const ptr_Neural_Network_received)
{
    if(this->Check_Topology(ptr_Neural_Network_received->number_inputs,
                                        ptr_Neural_Network_received->number_outputs,
                                        ptr_Neural_Network_received->number_recurrent_depth) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Check_Topology(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_Neural_Network_received->number_inputs,
                                 ptr_Neural_Network_received->number_outputs,
                                 ptr_Neural_Network_received->number_recurrent_depth,
                                 __LINE__);
        
        return((std::numeric_limits<ST_>::max)());
    }
    
    if(ptr_Neural_Network_received->type_accuracy_function == MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_R)
    {
        size_t const tmp_output_size(ptr_Neural_Network_received->Get__Output_Size()),
                           tmp_maximum_batch_size(ptr_Neural_Network_received->batch_size),
                           tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(batch_size_received) / static_cast<double>(tmp_maximum_batch_size))));
        size_t tmp_batch_index(0_zu),
                  tmp_batch_size(0_zu);
        
        // Mean.
        *ptr_Neural_Network_received->ptr_array_accuracy_values[0u] /= static_cast<T_>(batch_size_received * (this->Get__Number_Recurrent_Depth() - ptr_Neural_Network_received->number_time_delays) * tmp_output_size);
        *ptr_Neural_Network_received->ptr_array_accuracy_values[1u] /= static_cast<T_>(batch_size_received * (this->Get__Number_Recurrent_Depth() - ptr_Neural_Network_received->number_time_delays) * tmp_output_size);

        if(ptr_Neural_Network_received->use_OpenMP && ptr_Neural_Network_received->is_OpenMP_initialized)
        {
            omp_set_num_threads(static_cast<int>(ptr_Neural_Network_received->number_threads));
            
            #pragma omp parallel private(tmp_batch_index, tmp_batch_size)
            for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
            {
                tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : batch_size_received - tmp_batch_index * tmp_maximum_batch_size;
                
                ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, ptr_array_inputs_received + tmp_batch_index * tmp_maximum_batch_size);
                
                ptr_Neural_Network_received->Compute__Accuracy__R(tmp_batch_size, ptr_array_desired_outputs_received + tmp_batch_index * tmp_maximum_batch_size);
            }

            ptr_Neural_Network_received->Merge__Accuracy__R();
        }
        else
        {
            for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
            {
                tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : batch_size_received - tmp_batch_index * tmp_maximum_batch_size;
                
                ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, ptr_array_inputs_received + tmp_batch_index * tmp_maximum_batch_size);
                
                ptr_Neural_Network_received->Compute__Accuracy__R(tmp_batch_size, ptr_array_desired_outputs_received + tmp_batch_index * tmp_maximum_batch_size);
            }
        }

        // R = numerator / (sqrt(denominator_desired) * sqrt(denominator_predicted)).
        *ptr_Neural_Network_received->ptr_array_accuracy_values[0u] = *ptr_Neural_Network_received->ptr_array_accuracy_values[3u] * *ptr_Neural_Network_received->ptr_array_accuracy_values[4u] == T(0) ? T(0) : *ptr_Neural_Network_received->ptr_array_accuracy_values[2u] / (sqrt(*ptr_Neural_Network_received->ptr_array_accuracy_values[3u]) * sqrt(*ptr_Neural_Network_received->ptr_array_accuracy_values[4u]));
    }

    return(ptr_Neural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
}

template<typename T>
T Dataset<T>::Testing(class Neural_Network *const ptr_Neural_Network_received)
{
    if(this->Check_Topology(ptr_Neural_Network_received->number_inputs,
                                        ptr_Neural_Network_received->number_outputs,
                                        ptr_Neural_Network_received->number_recurrent_depth) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Check_Topology(%zu, %zu, %zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 ptr_Neural_Network_received->number_inputs,
                                 ptr_Neural_Network_received->number_outputs,
                                 ptr_Neural_Network_received->number_recurrent_depth,
                                 __LINE__);

        return((std::numeric_limits<ST_>::max)());
    }
    
    ptr_Neural_Network_received->Reset__Loss();
    
    if(ptr_Neural_Network_received->Set__Multi_Label(this->Use__Multi_Label()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Multi_Label(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->Use__Multi_Label() ? "true" : "false",
                                 __LINE__);

        return((std::numeric_limits<ST_>::max)());
    }
    else if(ptr_Neural_Network_received->Update__Batch_Size(this->Dataset<T>::Get__Number_Examples()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 this->Dataset<T>::Get__Number_Examples(),
                                 __LINE__);

        return((std::numeric_limits<ST_>::max)());
    }
    else if(ptr_Neural_Network_received->Initialized__Weight() == false && ptr_Neural_Network_received->Initialize__Weight(this) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__Weight(self)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return((std::numeric_limits<ST_>::max)());
    }

    if(ptr_Neural_Network_received->use_OpenMP && ptr_Neural_Network_received->is_OpenMP_initialized)
    {
        if(ptr_Neural_Network_received->Update__Thread_Size(this->Dataset<T>::Get__Number_Examples()) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->Dataset<T>::Get__Number_Examples(),
                                     __LINE__);

            return((std::numeric_limits<ST_>::max)());
        }
        
        omp_set_num_threads(static_cast<int>(ptr_Neural_Network_received->number_threads));
        
        return(this->Testing_OpenMP(ptr_Neural_Network_received));
    }
    else
    { return(this->Testing_Loop(ptr_Neural_Network_received)); }
}

template<typename T>
T Dataset<T>::Testing_OpenMP(class Neural_Network *const ptr_Neural_Network_received)
{
    size_t const tmp_number_examples(this->Dataset<T>::Get__Number_Examples()),
                       tmp_maximum_batch_size(ptr_Neural_Network_received->batch_size),
                       tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_index(0_zu),
              tmp_batch_size(0_zu);
    
    #pragma omp parallel private(tmp_batch_index, tmp_batch_size)
    for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
    {
        tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;
        
        ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, this->Dataset<T>::Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);
        
        ptr_Neural_Network_received->Compute__Loss(tmp_batch_size, this->Dataset<T>::Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);
    }
    
    ptr_Neural_Network_received->Merge__Post__Training();
    
    ptr_Neural_Network_received->number_accuracy_trial = tmp_number_examples * (this->Get__Number_Recurrent_Depth() - ptr_Neural_Network_received->number_time_delays) * (ptr_Neural_Network_received->type_accuracy_function == MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY ? 1_zu : ptr_Neural_Network_received->Get__Output_Size());

    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, this->Measure_Accuracy(tmp_number_examples,
                                                                                                                                                                                                                       this->Dataset<T>::Get__Input_Array(),
                                                                                                                                                                                                                       this->Dataset<T>::Get__Output_Array(),
                                                                                                                                                                                                                       ptr_Neural_Network_received));

    return(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
}

#if defined(COMPILE_AUTODIFF)
void Set__Output__Gradient(size_t const batch_size_received,
                                         size_t const total_batch_size_received,
                                         size_t const number_time_delay_received,
                                         size_t const number_recurrent_depth_received,
                                         size_t const number_outputs_received,
                                         T_ const *const *const ptr_array_outputs_array_received,
                                         T_ *const ptr_array_outputs_received)
{
    size_t tmp_example_index,
              tmp_output_index,
              tmp_time_step_index;

    T_ tmp_predicted,
        tmp_target,
        tmp_diff;

    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        for(tmp_time_step_index = number_time_delay_received; tmp_time_step_index != number_recurrent_depth_received; ++tmp_time_step_index)
        {
            PRINT_FORMAT("O-DATA[%zu], TIME[%zu]:" NEW_LINE, tmp_example_index, tmp_time_step_index);
            for(tmp_output_index = 0_zu; tmp_output_index != number_outputs_received; ++tmp_output_index)
            {
                tmp_predicted = ptr_array_outputs_received[tmp_example_index * number_outputs_received + total_batch_size_received * number_outputs_received * tmp_time_step_index + tmp_output_index];
                
                tmp_target = ptr_array_outputs_array_received[tmp_example_index][tmp_time_step_index * number_outputs_received + tmp_output_index];
                
                tmp_diff = tmp_predicted - tmp_target;
                
                PRINT_FORMAT("%f ", Cast_T(tmp_diff));

                ptr_array_outputs_received[tmp_example_index * number_outputs_received + total_batch_size_received * number_outputs_received * tmp_time_step_index + tmp_output_index].set_gradient(tmp_diff.value());  
            }
            PRINT_FORMAT(NEW_LINE);
        }
    }
}
#endif

template<typename T>
void Dataset<T>::Adept__Gradient(class Neural_Network *const ptr_Neural_Network_received)
{
    size_t tmp_batch_size(this->Dataset<T>::Get__Number_Examples());
    //size_t tmp_batch_size(MyEA::Math::Minimum<size_t>(4_zu, this->Dataset<T>::Get__Number_Examples());

    if(ptr_Neural_Network_received->Update__Batch_Size(tmp_batch_size) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_batch_size,
                                 __LINE__);

        return;
    }
    else if(ptr_Neural_Network_received->Initialized__Weight() == false && ptr_Neural_Network_received->Initialize__Weight(this) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize__Weight(self)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return;
    }
    
    ptr_Neural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING;
    
#if defined(COMPILE_AUTODIFF)
    adept::active_stack()->new_recording();
#endif

    if(ptr_Neural_Network_received->use_OpenMP && ptr_Neural_Network_received->is_OpenMP_initialized)
    {
        if(ptr_Neural_Network_received->Update__Thread_Size(tmp_batch_size) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_batch_size,
                                     __LINE__);

            return;
        }
        
        omp_set_num_threads(static_cast<int>(ptr_Neural_Network_received->number_threads));
        
        #pragma omp parallel
        ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, this->Dataset<T>::Get__Input_Array());
    }
    else
    {
        ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, this->Dataset<T>::Get__Input_Array());
    }

#if defined(COMPILE_AUTODIFF)
    Set__Output__Gradient(tmp_batch_size,
                                      ptr_Neural_Network_received->batch_size,
                                      ptr_Neural_Network_received->number_time_delays,
                                      ptr_Neural_Network_received->number_recurrent_depth,
                                      *(ptr_Neural_Network_received->ptr_last_layer - 1)->ptr_number_outputs,
                                      this->p_ptr_array_outputs_array,
                                      (ptr_Neural_Network_received->ptr_last_layer - 1)->ptr_array_outputs);

    adept::active_stack()->reverse();
#endif
    
    if(ptr_Neural_Network_received->use_OpenMP && ptr_Neural_Network_received->is_OpenMP_initialized)
    {
        #pragma omp parallel
        {
            ptr_Neural_Network_received->Compute__Error(tmp_batch_size, this->Dataset<T>::Get__Output_Array());

            ptr_Neural_Network_received->Backward_Pass(tmp_batch_size);
            
            ptr_Neural_Network_received->Update_Derivative_Weight(tmp_batch_size,
                                                                                               ptr_Neural_Network_received->ptr_array_layers + 1,
                                                                                               ptr_Neural_Network_received->ptr_last_layer);
        }

        ptr_Neural_Network_received->Merge_Derivatives_Parameters(0_zu, ptr_Neural_Network_received->total_parameters);
    }
    else
    {
        ptr_Neural_Network_received->Compute__Error(tmp_batch_size, this->Dataset<T>::Get__Output_Array());

        ptr_Neural_Network_received->Backward_Pass(tmp_batch_size);
        
        ptr_Neural_Network_received->Update_Derivative_Weight(tmp_batch_size,
                                                                                           ptr_Neural_Network_received->ptr_array_layers + 1,
                                                                                           ptr_Neural_Network_received->ptr_last_layer);
    }

#if defined(COMPILE_AUTODIFF)
    PRINT_FORMAT("Autodiff dx/dw: " NEW_LINE);
    for(size_t w = 0_zu; w != ptr_Neural_Network_received->total_parameters; ++w)
    {
        PRINT_FORMAT("%.8f ", ptr_Neural_Network_received->ptr_array_parameters[w].get_gradient());
        
        if(w != 0_zu && (w + 1) % 20 == 0_zu) { PRINT_FORMAT(NEW_LINE); }
    }
    PRINT_FORMAT(NEW_LINE);
#endif

    PRINT_FORMAT("Handcoded dx/dw: " NEW_LINE);
    for(size_t w = 0_zu; w != ptr_Neural_Network_received->total_parameters; ++w)
    {
        PRINT_FORMAT("%.8f ", Cast_T(ptr_Neural_Network_received->ptr_array_derivatives_parameters[w]));
        
        if(w != 0_zu && (w + 1) % 20 == 0_zu) { PRINT_FORMAT(NEW_LINE); }
    }
    PRINT_FORMAT(NEW_LINE);

    ptr_Neural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_INFERENCE;
    
    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, this->Measure_Accuracy(tmp_batch_size,
                                                                                                                                                                                                                       this->Get__Input_Array(),
                                                                                                                                                                                                                       this->Get__Output_Array(),
                                                                                                                                                                                                                       ptr_Neural_Network_received));
}

template<typename T>
T Dataset<T>::Testing_Loop(class Neural_Network *const ptr_Neural_Network_received)
{
    size_t const tmp_number_examples(this->Dataset<T>::Get__Number_Examples()),
                       tmp_maximum_batch_size(ptr_Neural_Network_received->batch_size),
                       tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_size,
              tmp_batch_index;
    
    for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
    {
        tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;

        ptr_Neural_Network_received->Forward_Pass(tmp_batch_size, this->Dataset<T>::Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);
        
        ptr_Neural_Network_received->Compute__Loss(tmp_batch_size, this->Dataset<T>::Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);
    }
    
    ptr_Neural_Network_received->number_accuracy_trial = tmp_number_examples * (this->Get__Number_Recurrent_Depth() - ptr_Neural_Network_received->number_time_delays) * (ptr_Neural_Network_received->type_accuracy_function == MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_CROSS_ENTROPY ? 1_zu : ptr_Neural_Network_received->Get__Output_Size());
    
    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, this->Measure_Accuracy(tmp_number_examples,
                                                                                                                                                                                                                       this->Dataset<T>::Get__Input_Array(),
                                                                                                                                                                                                                       this->Dataset<T>::Get__Output_Array(),
                                                                                                                                                                                                                       ptr_Neural_Network_received));

    return(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
}

template<typename T>
MyEA::Common::ENUM_TYPE_DATASET_PROCESS Dataset<T>::Get__Type_Dataset_Process(void) const { return(this->p_type_dataset_process); }

template<typename T>
T Dataset<T>::Get__Minimum_Input(size_t const data_start_index_received,
                                                      size_t const data_end_index_received,
                                                      size_t const input_index_received,
                                                      enum ENUM_TYPE_INPUT const type_input_received) const
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(T(0));
    }
    else if(data_start_index_received > data_end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_start_index_received,
                                 data_end_index_received,
                                 __LINE__);
        
        return(T(0));
    }
    else if(data_end_index_received > this->Dataset<T>::Get__Number_Examples())
    {
        PRINT_FORMAT("%s: %s: ERROR: End index (%zu) can not be greater than total examples (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_end_index_received,
                                 this->Dataset<T>::Get__Number_Examples(),
                                 __LINE__);
        
        return(T(0));
    }
    else if(type_input_received > ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(T(0));
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_example_index,
              tmp_time_step_index;

    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(T(0));
    }
    
    T const *const *const tmp_ptr_array_inputs(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->Dataset<T>::Get__Input_Array() : this->Dataset<T>::Get__Output_Array());

    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_array_inputs\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(T(0));
    }

    T tmp_minimum_value((std::numeric_limits<ST_>::max)()),
       tmp_input_value;
    
    for(tmp_example_index = data_start_index_received; tmp_example_index != data_end_index_received; ++tmp_example_index)
    {
        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_input_value = tmp_ptr_array_inputs[tmp_example_index][tmp_time_step_index * tmp_input_size + input_index_received];
            
            if(tmp_input_value < tmp_minimum_value) { tmp_minimum_value = tmp_input_value; }
        }
    }

    return(tmp_minimum_value);
}

template<typename T>
T Dataset<T>::Get__Minimum_Input(size_t const data_start_index_received,
                                                      size_t const data_end_index_received,
                                                      enum ENUM_TYPE_INPUT const type_input_received) const
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(T(0));
    }
    else if(data_start_index_received > data_end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_start_index_received,
                                 data_end_index_received,
                                 __LINE__);
        
        return(T(0));
    }
    else if(data_end_index_received > this->Dataset<T>::Get__Number_Examples())
    {
        PRINT_FORMAT("%s: %s: ERROR: End index (%zu) can not be greater than total examples (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_end_index_received,
                                 this->Dataset<T>::Get__Number_Examples(),
                                 __LINE__);
        
        return(T(0));
    }
    else if(type_input_received > ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(T(0));
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    T tmp_value,
       tmp_minimum_value(this->Get__Minimum_Input(data_start_index_received,
                                                                             data_end_index_received,
                                                                             0_zu,
                                                                             type_input_received));
    
    for(tmp_input_index = 1_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        tmp_value = this->Get__Minimum_Input(data_start_index_received,
                                                                  data_end_index_received,
                                                                  tmp_input_index,
                                                                  type_input_received);

        if(tmp_value < tmp_minimum_value) { tmp_minimum_value = tmp_value; }
    }

    return(tmp_minimum_value);
}

template<typename T>
T Dataset<T>::Get__Maximum_Input(size_t const data_start_index_received,
                                                       size_t const data_end_index_received,
                                                       size_t const input_index_received,
                                                       enum ENUM_TYPE_INPUT const type_input_received) const
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(T(0));
    }
    else if(data_start_index_received > data_end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_start_index_received,
                                 data_end_index_received,
                                 __LINE__);
        
        return(T(0));
    }
    else if(data_end_index_received > this->Dataset<T>::Get__Number_Examples())
    {
        PRINT_FORMAT("%s: %s: ERROR: End index (%zu) can not be greater than total examples (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_end_index_received,
                                 this->Dataset<T>::Get__Number_Examples(),
                                 __LINE__);
        
        return(T(0));
    }
    else if(type_input_received > ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(T(0));
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_example_index,
              tmp_time_step_index;

    if(input_index_received >= tmp_input_size)
    {
        PRINT_FORMAT("%s: %s: ERROR: Input index (%zu) overflow (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 input_index_received,
                                 tmp_input_size,
                                 __LINE__);
        
        return(T(0));
    }
    
    T const *const *const tmp_ptr_array_inputs(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->Dataset<T>::Get__Input_Array() : this->Dataset<T>::Get__Output_Array());

    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"tmp_ptr_array_inputs\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(T(0));
    }

    T tmp_maximum_value(-(std::numeric_limits<ST_>::max)()),
       tmp_input_value;
    
    for(tmp_example_index = data_start_index_received; tmp_example_index != data_end_index_received; ++tmp_example_index)
    {
        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->p_number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_input_value = tmp_ptr_array_inputs[tmp_example_index][tmp_time_step_index * tmp_input_size + input_index_received];
            
            if(tmp_input_value > tmp_maximum_value) { tmp_maximum_value = tmp_input_value; }
        }
    }

    return(tmp_maximum_value);
}

template<typename T>
T Dataset<T>::Get__Maximum_Input(size_t const data_start_index_received,
                                                       size_t const data_end_index_received,
                                                       enum ENUM_TYPE_INPUT const type_input_received) const
{
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: No data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(T(0));
    }
    else if(data_start_index_received > data_end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not be greater than end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_start_index_received,
                                 data_end_index_received,
                                 __LINE__);
        
        return(T(0));
    }
    else if(data_end_index_received > this->Dataset<T>::Get__Number_Examples())
    {
        PRINT_FORMAT("%s: %s: ERROR: End index (%zu) can not be greater than total examples (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 data_end_index_received,
                                 this->Dataset<T>::Get__Number_Examples(),
                                 __LINE__);
        
        return(T(0));
    }
    else if(type_input_received > ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT)
    {
        PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 type_input_received,
                                 __LINE__);
        
        return(T(0));
    }
    
    size_t const tmp_input_size(type_input_received == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? this->p_number_inputs : this->p_number_outputs);
    size_t tmp_input_index;

    T tmp_value,
       tmp_maximum_value(this->Get__Maximum_Input(data_start_index_received,
                                                                               data_end_index_received,
                                                                               0_zu,
                                                                               type_input_received));
    
    for(tmp_input_index = 1_zu; tmp_input_index != tmp_input_size; ++tmp_input_index)
    {
        tmp_value = this->Get__Maximum_Input(data_start_index_received,
                                                                    data_end_index_received,
                                                                    tmp_input_index,
                                                                    type_input_received);

        if(tmp_value > tmp_maximum_value) { tmp_maximum_value = tmp_value; }
    }

    return(tmp_maximum_value);
}

template<typename T>
T Dataset<T>::Get__Input_At(size_t const index_received, size_t const sub_index_received) const { return(this->p_ptr_array_inputs_array[index_received + this->p_start_index][sub_index_received]); }

template<typename T>
T Dataset<T>::Get__Output_At(size_t const index_received, size_t const sub_index_received) const { return(this->p_ptr_array_outputs_array[index_received + this->p_start_index][sub_index_received]); }

template<typename T>
T const *const Dataset<T>::Get__Input_At(size_t const index_received) const { return(this->p_ptr_array_inputs_array[index_received + this->p_start_index]); }

template<typename T>
T const *const Dataset<T>::Get__Output_At(size_t const index_received) const { return(this->p_ptr_array_outputs_array[index_received + this->p_start_index]); }

template<typename T>
T const *const *const Dataset<T>::Get__Input_Array(void) const { return(this->p_ptr_array_inputs_array + this->p_start_index); }

template<typename T>
T const *const *const Dataset<T>::Get__Output_Array(void) const { return(this->p_ptr_array_outputs_array + this->p_start_index); }

template<typename T>
size_t Dataset<T>::Get__Sizeof(void)
{
    size_t tmp_total_size_t(0);

    tmp_total_size_t += sizeof(class Dataset<T>); // this
    
    if(this->_reference == false)
    {
        size_t tmp_input_index;

        if(this->p_ptr_array_inputs_array != nullptr) { tmp_total_size_t += this->p_number_examples * sizeof(T*); }
        if(this->p_ptr_array_outputs_array != nullptr) { tmp_total_size_t += this->p_number_examples * sizeof(T*); }

        if(this->p_ptr_array_inputs != nullptr) { tmp_total_size_t += this->p_number_examples * this->p_number_inputs * this->p_number_recurrent_depth * sizeof(T); }
        if(this->p_ptr_array_outputs != nullptr) { tmp_total_size_t += this->p_number_examples * this->p_number_outputs * this->p_number_recurrent_depth * sizeof(T); }

        if(this->_ptr_input_array_scaler__minimum_maximum != nullptr) { tmp_total_size_t += this->p_number_inputs * sizeof(this->_ptr_input_array_scaler__minimum_maximum); }
        if(this->_ptr_output_array_scaler__minimum_maximum != nullptr) { tmp_total_size_t += this->p_number_outputs * sizeof(this->_ptr_output_array_scaler__minimum_maximum); }
        
        if(this->_ptr_input_array_scaler__zero_centered != nullptr) { tmp_total_size_t += this->p_number_inputs * sizeof(this->_ptr_input_array_scaler__zero_centered); }
        if(this->_ptr_output_array_scaler__zero_centered != nullptr) { tmp_total_size_t += this->p_number_outputs * sizeof(this->_ptr_output_array_scaler__zero_centered); }
        
        if(this->p_ptr_input_coefficient_matrix != nullptr)
        {
            tmp_total_size_t += this->p_number_inputs * this->p_number_recurrent_depth * sizeof(T*);
        
            if(this->p_ptr_input_array_coefficient_matrix_size != nullptr)
            {
                for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_inputs; ++tmp_input_index)
                { tmp_total_size_t += this->p_ptr_input_array_coefficient_matrix_size[tmp_input_index] * this->p_number_recurrent_depth * sizeof(T); }
            }
        }
    
        if(this->p_ptr_input_array_coefficient_matrix_size != nullptr) { tmp_total_size_t += this->p_number_inputs * sizeof(size_t); }

        if(this->p_ptr_output_coefficient_matrix != nullptr)
        {
            tmp_total_size_t += this->p_number_outputs * this->p_number_recurrent_depth * sizeof(T*);
        
            if(this->p_ptr_output_array_coefficient_matrix_size != nullptr)
            {
                for(tmp_input_index = 0_zu; tmp_input_index != this->p_number_outputs; ++tmp_input_index)
                { tmp_total_size_t += this->p_ptr_output_array_coefficient_matrix_size[tmp_input_index] * this->p_number_recurrent_depth * sizeof(T); }
            }
        }

        if(this->p_ptr_output_array_coefficient_matrix_size != nullptr) { tmp_total_size_t += this->p_number_outputs * sizeof(size_t); }
    }

    return(tmp_total_size_t);
}

template<typename T>
struct Scaler__Minimum_Maximum<T> *const Dataset<T>::Get__Scalar__Minimum_Maximum(enum ENUM_TYPE_INPUT const type_input_received) const
{
    switch(type_input_received)
    {
        case ENUM_TYPE_INPUT::TYPE_INPUT_INPUT: return(this->_ptr_input_array_scaler__minimum_maximum);
        case ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT: return(this->_ptr_output_array_scaler__minimum_maximum);
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     type_input_received,
                                     __LINE__);
                return(nullptr);
    }
}

template<typename T>
struct Scaler__Zero_Centered<T> *const Dataset<T>::Get__Scalar__Zero_Centered(enum ENUM_TYPE_INPUT const type_input_received) const
{
    switch(type_input_received)
    {
        case ENUM_TYPE_INPUT::TYPE_INPUT_INPUT: return(this->_ptr_input_array_scaler__zero_centered);
        case ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT: return(this->_ptr_output_array_scaler__zero_centered);
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type input (%u) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     type_input_received,
                                     __LINE__);
                return(nullptr);
    }
}

template<typename T>
Dataset<T>::~Dataset(void) { this->Deallocate(); }

// template initialization declaration.
template class Dataset<T_>;