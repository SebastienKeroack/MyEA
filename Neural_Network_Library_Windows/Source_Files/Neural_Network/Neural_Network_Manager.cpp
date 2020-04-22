#include "stdafx.hpp"

#if defined(COMPILE_LINUX)
    #include <unistd.h> // getlogin
#endif

#if defined(COMPILE_CUDA)
    #include <Tools/CUDA_Configuration.cuh>
#endif

#if defined(COMPILE_COUT)
    #include <Strings/Animation_Waiting.hpp>
#endif

#if defined(COMPILE_UI)
    #include <Enums/Enum_Type_Chart.hpp>

    #include <Form.hpp>
#endif

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <Time/Time.hpp>
#include <Capturing/Shutdown/Shutdown.hpp>
#include <Files/File.hpp>

#include <fstream>
#include <iostream>
#include <array>

namespace MyEA
{
    namespace Neural_Network
    {
        Neural_Network_Manager::Neural_Network_Manager(void)
        { }

        bool Neural_Network_Manager::Initialize_Path(std::string const &ref_class_name_received, std::string const &ref_neural_network_name_received)
        {
            std::string const tmp_directory_name("MyEA");
            std::string tmp_root_drive(MyEA::File::Get__Full_Path(tmp_directory_name)),
                           tmp_path;
            
            if(tmp_root_drive.empty())
            {
                std::vector<std::pair<std::string, std::string>> const tmp_drives(MyEA::File::Get__List_Drives());

            #if defined(COMPILE_COUT)
                size_t tmp_drive_index,
                          tmp_drive_option;

                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                PRINT_FORMAT("%s: Path available:" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                
            #if defined(COMPILE_WINDOWS)
                for(tmp_drive_index = 0_zu; tmp_drive_index != tmp_drives.size(); ++tmp_drive_index)
                {
                    PRINT_FORMAT("%s: [%zu]: %s" NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             tmp_drive_index,
                                             tmp_drives[tmp_drive_index].first.c_str());
                }
            #elif defined(COMPILE_LINUX)
                for(tmp_drive_index = 0_zu; tmp_drive_index != tmp_drives.size(); ++tmp_drive_index)
                {
                    PRINT_FORMAT("%s: [%zu]: %s, %s" NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             tmp_drive_index,
                                             tmp_drives[tmp_drive_index].first.c_str(),
                                             tmp_drives[tmp_drive_index].second.c_str());
                }
            #endif // COMPILE_WINDOWS || COMPILE_LINUX

                if((tmp_drive_option = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                                     tmp_drives.size() - 1_zu,
                                                                                                     MyEA::Time::Date_Time_Now() + ": Drive: ")) >= tmp_drives.size())
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<size_t>(%zu, %zu)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             0_zu,
                                             tmp_drives.size() - 1_zu,
                                             __LINE__);

                    return(false);
                }
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                
                tmp_path = tmp_drives[tmp_drive_option].second + ESCAPE_FILE + tmp_directory_name;
            #else // COMPILE_COUT
                return(false);
            #endif // COMPILE_COUT
            }
            else { tmp_path = tmp_root_drive; }

            this->_path_root = tmp_path;
            
            if(MyEA::File::Path_Exist(tmp_path) == false)
            { MyEA::File::Directory_Create(tmp_path); }

            if(MyEA::File::Path_Exist(tmp_path + ESCAPE_FILE + "LOG") == false)
            { MyEA::File::Directory_Create(tmp_path + ESCAPE_FILE + "LOG"); }

            if(MyEA::File::Path_Exist(tmp_path + ESCAPE_FILE + "ERROR") == false)
            { MyEA::File::Directory_Create(tmp_path + ESCAPE_FILE + "ERROR"); }

            if(MyEA::File::Path_Exist(tmp_path + ESCAPE_FILE + "DEBUG") == false)
            { MyEA::File::Directory_Create(tmp_path + ESCAPE_FILE + "DEBUG"); }

            // Drive://MyEA//Class_Name
            if(MyEA::File::Directory_Create(tmp_path + ESCAPE_FILE +
                                                            ref_class_name_received) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Directory_Create(" + tmp_path + ESCAPE_FILE +
                                        ref_class_name_received + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            // Drive://MyEA//Class_Name//Model
            if(MyEA::File::Directory_Create(tmp_path + ESCAPE_FILE +
                                                            ref_class_name_received + ESCAPE_FILE +
                                                                "Model") == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Directory_Create(" + tmp_path + ESCAPE_FILE +
                                                            ref_class_name_received + ESCAPE_FILE +
                                                                "Model" + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            // Drive://MyEA//Class_Name//Model//Trainer
            if(MyEA::File::Directory_Create(tmp_path + ESCAPE_FILE +
                                                            ref_class_name_received + ESCAPE_FILE +
                                                                "Model" + ESCAPE_FILE +
                                                                    "Trainer") == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Directory_Create(" + tmp_path + ESCAPE_FILE +
                                                            ref_class_name_received + ESCAPE_FILE +
                                                                "Model" + ESCAPE_FILE +
                                                                    "Trainer" + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            // Drive://MyEA//Class_Name//Model//Trained
            if(MyEA::File::Directory_Create(tmp_path + ESCAPE_FILE +
                                                            ref_class_name_received + ESCAPE_FILE +
                                                                "Model" + ESCAPE_FILE +
                                                                    "Trained") == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Directory_Create(" + tmp_path + ESCAPE_FILE +
                                                            ref_class_name_received + ESCAPE_FILE +
                                                                "Model" + ESCAPE_FILE +
                                                                    "Trained" + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            // Drive://MyEA//Class_Name//DATASET
            if(MyEA::File::Directory_Create(tmp_path + ESCAPE_FILE +
                                                            ref_class_name_received + ESCAPE_FILE +
                                                                "Dataset") == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Directory_Create(" + tmp_path + ESCAPE_FILE +
                                                            ref_class_name_received + ESCAPE_FILE +
                                                                "Dataset" + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            /* Drive://MyEA//Class_Name//Model//Trainer//Model_Name */
            this->_path_model_trainer = tmp_path + ESCAPE_FILE +
                                                        ref_class_name_received + ESCAPE_FILE +
                                                            "Model" + ESCAPE_FILE +
                                                                "Trainer" + ESCAPE_FILE +
                                                                    ref_neural_network_name_received;
            
            /* Drive://MyEA//Class_Name//Model//Trained//Model_Name */
            this->_path_model_trained = tmp_path + ESCAPE_FILE +
                                                        ref_class_name_received + ESCAPE_FILE +
                                                            "Model" + ESCAPE_FILE +
                                                                "Trained" + ESCAPE_FILE +
                                                                    ref_neural_network_name_received;
            
            /* Drive://MyEA//Class_Name//DATASET//Model_Name */
            this->_path_dataset = tmp_path + ESCAPE_FILE +
                                                    ref_class_name_received + ESCAPE_FILE +
                                                        "Dataset" + ESCAPE_FILE +
                                                            ref_neural_network_name_received;

            /* Drive://MyEA//Class_Name//DATASET//Model_Name_history.dataset */
            this->_path_dataset_history = tmp_path + ESCAPE_FILE +
                                                            ref_class_name_received + ESCAPE_FILE +
                                                                "Dataset" + ESCAPE_FILE +
                                                                    ref_neural_network_name_received + "_history.dataset";
            
            return(true);
        }
        
        bool Neural_Network_Manager::Initialize_Dataset_Manager(struct Dataset_Manager_Parameters const *const ptr_Dataset_Manager_Parameters_received)
        {
            if(this->_ptr_Dataset_Manager != nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"Dataset_Manager<T>\" is not a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

        #if defined(COMPILE_COUT)
            enum MyEA::Common::ENUM_TYPE_DATASET_FILE tmp_type_dataset_file;

            if(Input_Dataset_File(tmp_type_dataset_file, this->_path_dataset) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Input_Dataset_File(" + this->_path_dataset + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            PRINT_FORMAT("%s: Loading from %s... ",
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     this->_path_dataset.c_str());
            class MyEA::String::Animation_Waiting tmp_Animation_Waiting;
            tmp_Animation_Waiting.Print_While_Async();
            
            if((this->_ptr_Dataset_Manager = new class Dataset_Manager<T_>(tmp_type_dataset_file, this->_path_dataset)) == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Can not allocate " + std::to_string(sizeof(class Neural_Network)) + " bytes. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            tmp_Animation_Waiting.Join();
            PRINT_FORMAT(NEW_LINE);
        #else
            if(MyEA::File::Path_Exist(this->_path_dataset + ".dataset") == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Could not find the following path " + this->_path_dataset + ".dataset. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }
            
            if((this->_ptr_Dataset_Manager = new class Dataset_Manager<T_>(MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_DATASET, this->_path_dataset)) == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Can not allocate " + std::to_string(sizeof(class Neural_Network)) + " bytes. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }
        #endif
            
            if(this->_ptr_Dataset_Manager->Set__Maximum_Data(this->_ptr_Dataset_Manager->Get__Number_Examples()) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Maximum_Data(" + std::to_string(this->_ptr_Dataset_Manager->Get__Number_Examples()) + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                SAFE_DELETE(this->_ptr_Dataset_Manager);

                return(false);
            }
            else if(this->_ptr_Dataset_Manager->Preparing_Dataset_Manager(ptr_Dataset_Manager_Parameters_received) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Preparing_Dataset_Manager(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                SAFE_DELETE(this->_ptr_Dataset_Manager);

                return(false);
            }
            
        #if defined(COMPILE_CUDA)
            if(this->_use_CUDA && this->_ptr_Dataset_Manager->Initialize__CUDA() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Initialize__CUDA()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                SAFE_DELETE(this->_ptr_Dataset_Manager);

                return(false);
            }
        #endif

            return(true);
        }
        
        bool Neural_Network_Manager::Create_Neural_Network(size_t const maximum_allowable_host_memory_bytes_received)
        {
            if(this->_ptr_trainer_Neural_Network != nullptr) { this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER); }
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to use template?"))
            {
                // Neural network initializer.
                struct Neural_Network_Initializer tmp_Neural_Network_Initializer;
                
                if(tmp_Neural_Network_Initializer.Template_Initialize() == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Template_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                    return(false);
                }

                if((this->_ptr_trainer_Neural_Network = tmp_Neural_Network_Initializer.Output_Initialize(maximum_allowable_host_memory_bytes_received)) == nullptr)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Output_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                    return(false);
                }
                // |END| Neural network initializer. |END|

                size_t tmp_layer_index,
                          tmp_residual_index(0_zu),
                          tmp_total_residual_blocks(0_zu);
                
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                PRINT_FORMAT("%s: ShakeDrop, dropout probability." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                PRINT_FORMAT("%s:\tRange[0, %f]." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         Cast_T(1_T - 1e-7_T));
                T_ const tmp_shakedrop_dropout_probability(MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                          1_T - 1e-7_T,
                                                                                                                                          MyEA::Time::Date_Time_Now() + ": \tDropout probability: "));
                
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                PRINT_FORMAT("%s: Activation functions:" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                for(unsigned int tmp_activation_function_index(1u); tmp_activation_function_index != MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH; ++tmp_activation_function_index)
                {
                    PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             tmp_activation_function_index,
                                             MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(tmp_activation_function_index)].c_str());
                }
                PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION_NAME[MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LEAKY_RELU].c_str());
                
                enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION tmp_type_activation_function_hidden,
                                                                                                              tmp_type_activation_function_output;
                
                if((tmp_type_activation_function_hidden = static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                                                            MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                                                                                                                                                                                                                                            MyEA::Time::Date_Time_Now() + ": Hidden layer, activation function: "))) >= MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             1u,
                                             MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                             __LINE__);

                    return(false);
                }
                
                if((tmp_type_activation_function_output = static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION>(MyEA::String::Cin_Number<unsigned int>(1u,
                                                                                                                                                                                                                                                            MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                                                                                                                                                                                                                                            MyEA::Time::Date_Time_Now() + ": Output layer, activation function: "))) >= MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             1u,
                                             MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LENGTH - 1u,
                                             __LINE__);

                    return(false);
                }

                for(tmp_layer_index = 1_zu; tmp_layer_index != this->_ptr_trainer_Neural_Network->total_layers - 1_zu; ++tmp_layer_index)
                {
                    if(this->_ptr_trainer_Neural_Network->ptr_array_layers[tmp_layer_index].type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { ++tmp_total_residual_blocks; }
                }

                for(tmp_layer_index = 1_zu; tmp_layer_index != this->_ptr_trainer_Neural_Network->total_layers - 1_zu; ++tmp_layer_index)
                {
                    switch(this->_ptr_trainer_Neural_Network->ptr_array_layers[tmp_layer_index].type_layer)
                    {
                        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
                        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                            if(this->_ptr_trainer_Neural_Network->Set__Layer_Activation_Function(tmp_layer_index, tmp_type_activation_function_hidden) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function(%zu, %u)\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         tmp_layer_index,
                                                         tmp_type_activation_function_hidden,
                                                         __LINE__);

                                return(false);
                            }
                            else if(this->_ptr_trainer_Neural_Network->Set__Layer_Normalization(tmp_layer_index, MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization(%zu, %u)\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         tmp_layer_index,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                            if(this->_ptr_trainer_Neural_Network->Set__Layer_Normalization(tmp_layer_index, MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization(%zu, %u)\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         tmp_layer_index,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION,
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL:
                            if(this->_ptr_trainer_Neural_Network->Set__Layer_Normalization(tmp_layer_index, MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Normalization(%zu, %u)\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         tmp_layer_index,
                                                         MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION,
                                                         __LINE__);

                                return(false);
                            }
                            else if(tmp_shakedrop_dropout_probability != 0_T
                                      &&
                                      this->_ptr_trainer_Neural_Network->Set__Dropout(tmp_layer_index,
                                                                                                              MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP,
                                                                                                              std::array<T_, 1_zu>{1_T - ( (static_cast<T_>(++tmp_residual_index) / static_cast<T_>(tmp_total_residual_blocks)) * (1_T - tmp_shakedrop_dropout_probability) )}.data()) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout(%zu, %u, %f)\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         tmp_layer_index,
                                                         MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP,
                                                         Cast_T(1_T - ( (static_cast<T_>(tmp_residual_index) / static_cast<T_>(tmp_total_residual_blocks)) * (1_T - tmp_shakedrop_dropout_probability) )),
                                                         __LINE__);

                                return(false);
                            }
                                break;
                        default:
                            PRINT_FORMAT("%s: %s: ERROR: Layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     this->_ptr_trainer_Neural_Network->ptr_array_layers[tmp_layer_index].type_layer,
                                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[this->_ptr_trainer_Neural_Network->ptr_array_layers[tmp_layer_index].type_layer].c_str(),
                                                     __LINE__);
                                return(false);
                    }
                }

                // Output layer.
                if(this->_ptr_trainer_Neural_Network->Set__Layer_Activation_Function(tmp_layer_index, tmp_type_activation_function_output) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Layer_Activation_Function(%zu, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             tmp_layer_index,
                                             tmp_type_activation_function_output,
                                             __LINE__);

                    return(false);
                }
                // |END| Output layer. |END|
            }
            else
            {
                // Neural network initializer.
                struct Neural_Network_Initializer tmp_Neural_Network_Initializer;
                
                if(tmp_Neural_Network_Initializer.Input_Initialize() == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Input_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                    return(false);
                }

                if((this->_ptr_trainer_Neural_Network = tmp_Neural_Network_Initializer.Output_Initialize(maximum_allowable_host_memory_bytes_received)) == nullptr)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Output_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                    return(false);
                }
                // |END| Neural network initializer. |END|

                // Activation functions.
                struct Activation_Function_Initializer tmp_Activation_Function_Initializer;
                
                if(tmp_Activation_Function_Initializer.Input_Initialize(this->_ptr_trainer_Neural_Network->total_layers, this->_ptr_trainer_Neural_Network->type_network) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Input_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                    this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                    return(false);
                }

                if(tmp_Activation_Function_Initializer.Output_Initialize(this->_ptr_trainer_Neural_Network) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Output_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                    this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                    return(false);
                }
                // |END| Activation functions. |END|
                
                // Steepness.
                struct Activation_Steepness_Initializer tmp_Activation_Steepness_Initializer;
                
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to use activation steepness?"))
                {
                    if(tmp_Activation_Steepness_Initializer.Input_Initialize(this->_ptr_trainer_Neural_Network->total_layers, this->_ptr_trainer_Neural_Network->type_network) == false)
                    {
                        this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                                ": ERROR: An error has been triggered from the \"Input_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                        this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                        return(false);
                    }

                    if(tmp_Activation_Steepness_Initializer.Output_Initialize(this->_ptr_trainer_Neural_Network) == false)
                    {
                        this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                                ": ERROR: An error has been triggered from the \"Output_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                        this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                        return(false);
                    }
                }
                // |END| Steepness. |END|
                
                // Dropout.
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to use dropout?"))
                {
                    struct Dropout_Initializer tmp_Dropout_Initializer;
                
                    if(tmp_Dropout_Initializer.Input_Initialize(this->_ptr_trainer_Neural_Network->total_layers, this->_ptr_trainer_Neural_Network->type_network) == false)
                    {
                        this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                                ": ERROR: An error has been triggered from the \"Input_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                        this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                        return(false);
                    }

                    if(tmp_Dropout_Initializer.Output_Initialize(this->_ptr_trainer_Neural_Network) == false)
                    {
                        this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                                ": ERROR: An error has been triggered from the \"Output_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                        this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                        return(false);
                    }
                }
                // |END| Dropout. |END|
                
                // Normalization.
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to use normalization?"))
                {
                    struct Normalization_Initializer tmp_Normalization_Initializer;
                
                    if(tmp_Normalization_Initializer.Input_Initialize(this->_ptr_trainer_Neural_Network->total_layers,
                                                                                      this->_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING) == nullptr ? 1_zu : this->_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)->Get__Number_Batch(),
                                                                                      this->_ptr_trainer_Neural_Network->type_network) == false)
                    {
                        this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                                ": ERROR: An error has been triggered from the \"Input_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                        this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                        return(false);
                    }

                    if(tmp_Normalization_Initializer.Output_Initialize(this->_ptr_trainer_Neural_Network) == false)
                    {
                        this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                                ": ERROR: An error has been triggered from the \"Output_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                        this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                        return(false);
                    }
                }
                // |END| Normalization. |END|
                
                // Tied parameter.
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to use tied parameter?")
                   &&
                   this->_ptr_trainer_Neural_Network->User_Controls__Tied__Parameter() == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"User_Controls__Tied__Parameter()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                    this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                    return(false);
                }
                // |END| Tied parameter. |END|
                
                // k-Sparse.
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to use k-Sparse?")
                   &&
                   this->_ptr_trainer_Neural_Network->User_Controls__K_Sparse() == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"User_Controls__K_Sparse()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                    this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                    return(false);
                }
                // |END| k-Sparse. |END|
            }

            // Loss function.
            struct Loss_Function_Initializer tmp_Loss_Function_Initializer;

            if(tmp_Loss_Function_Initializer.Input_Initialize() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Input_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }

            tmp_Loss_Function_Initializer.Output_Initialize(this->_ptr_trainer_Neural_Network);
            // |END| Loss function. |END|
            
            // Accuracy function.
            struct Accuracy_Function_Initializer tmp_Accuracy_Function_Initializer;

            if(tmp_Accuracy_Function_Initializer.Input_Initialize() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Input_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }

            tmp_Accuracy_Function_Initializer.Output_Initialize(this->_ptr_trainer_Neural_Network);
            // |END| Accuracy function. |END|
            
            // Accuracy variance.
            if(this->_ptr_trainer_Neural_Network->type_accuracy_function == MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_DISTANCE
              &&
              this->_ptr_trainer_Neural_Network->User_Controls__Accuracy_Variance() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"User_Controls__Accuracy_Variance()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Accuracy variance. |END|
            
            // Optimizer function.
            struct Optimizer_Function_Initializer tmp_Optimizer_Function_Initializer;

            if(tmp_Optimizer_Function_Initializer.Input_Initialize() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Input_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            
            if(tmp_Optimizer_Function_Initializer.Output_Initialize(this->_ptr_trainer_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Output_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Optimizer function. |END|
            
            // Warm restarts.
            struct Warm_Restarts_Initializer tmp_Warm_Restarts_Initializer;

            tmp_Warm_Restarts_Initializer.Input_Initialize();
            
            if(this->_ptr_trainer_Neural_Network->Usable_Warm_Restarts()
              &&
              tmp_Warm_Restarts_Initializer.Output_Initialize(this->_ptr_trainer_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Output_Initialize()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Warm restarts. |END|

            // Clip gradient.
            if(this->_ptr_trainer_Neural_Network->User_Controls__Clip_Gradient() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"User_Controls__Clip_Gradient()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Clip gradient. |END|

            // Regularization Max-norm.
            if(this->_ptr_trainer_Neural_Network->User_Controls__Max_Norm_Constaints() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"User_Controls__Max_Norm_Constaints()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Regularization Max-norm. |END|

            // Regularization L1.
            if(this->_ptr_trainer_Neural_Network->User_Controls__L1_Regularization() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"User_Controls__L1_Regularization()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Regularization L1. |END|

            // Regularization L2.
            if(this->_ptr_trainer_Neural_Network->User_Controls__L2_Regularization() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"User_Controls__L2_Regularization()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Regularization L2. |END|
            
            // Regularization SRIP.
            if(this->_ptr_trainer_Neural_Network->User_Controls__SRIP_Regularization() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"User_Controls__SRIP_Regularization()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Regularization SRIP. |END|
            
            // Weights initializer.
            struct Weights_Initializer tmp_Weights_Initializer;
            
            if(tmp_Weights_Initializer.Input_Initialize() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Output_Initialize(ptr)\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            
            if(tmp_Weights_Initializer.Output_Initialize(this->_ptr_trainer_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Output_Initialize(ptr)\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Weights initializer. |END|

            // Batch size.
            if(this->_ptr_trainer_Neural_Network->User_Controls__Maximum__Batch_Size() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"User_Controls__Maximum__Batch_Size()\" function, at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Batch size. |END|

            // OpenMP.
            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to use OpenMP?"))
            {
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                PRINT_FORMAT("%s: Maximum threads:" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                PRINT_FORMAT("%s:\tRange[0.0%%, 100.0%%]." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                this->_ptr_trainer_Neural_Network->percentage_maximum_thread_usage = MyEA::String::Cin_Real_Number<T_>(0_T,
                                                                                                                                                                                     100_T,
                                                                                                                                                                                     MyEA::Time::Date_Time_Now() + ": Maximum threads (percent): ");

                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                PRINT_FORMAT("%s: Initialize OpenMP." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(this->_ptr_trainer_Neural_Network->Set__OpenMP(true) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__OpenMP(true)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
            }
            // |END| OpenMP. |END|
            
            // CUDA.
        #if defined(COMPILE_CUDA)
            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to use CUDA?"))
            {
                int tmp_index_device(-1);

                size_t tmp_maximum_device_memory_allocate_bytes(0_zu);

                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                this->Set__Use__CUDA(CUDA__Input__Use__CUDA(tmp_index_device, tmp_maximum_device_memory_allocate_bytes));
                
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                PRINT_FORMAT("%s: Initialize CUDA." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(this->_ptr_trainer_Neural_Network->Set__CUDA(this->_use_CUDA, tmp_maximum_device_memory_allocate_bytes) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__CUDA(%s, %zu)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             this->_use_CUDA ? "true" : "false",
                                             tmp_maximum_device_memory_allocate_bytes,
                                             __LINE__);

                    return(false);
                }
            }
        #endif
            // |END| CUDA. |END|
            
            // Copy trainer neural networks parameters to competitor neural network general parameters.
            if((this->_ptr_competitor_Neural_Network = new class Neural_Network) == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Can not allocate " + std::to_string(sizeof(class Neural_Network)) + " bytes at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            PRINT_FORMAT("%s: Copy dimension and hyper-parameters into a neural network called competitor." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            if(this->_ptr_competitor_Neural_Network->Copy(*this->_ptr_trainer_Neural_Network, true) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy(ptr, true)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            if(this->_ptr_competitor_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Copy(ptr, true)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Copy trainer neural networks parameters to competitor neural network general parameters. |END|

            // Copy trainer neural networks parameters to trained neural network general parameters.
            if(this->_ptr_trained_Neural_Network != nullptr) { this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED); }

            if((this->_ptr_trained_Neural_Network = new class Neural_Network) == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Can not allocate " + std::to_string(sizeof(class Neural_Network)) + " bytes at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            PRINT_FORMAT("%s: Copy dimension and hyper-parameters into a neural network called trained." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            if(this->_ptr_trained_Neural_Network->Copy(*this->_ptr_trainer_Neural_Network, true) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy(ptr, true)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            if(this->_ptr_trained_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Copy(ptr, true)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER);

                return(false);
            }
            // |END| Copy trainer neural networks parameters to trained neural network general parameters. |END|

            return(true);
        }
        
        bool Neural_Network_Manager::Allocate__Shutdown_Boolean(void)
        {
            std::atomic<bool> *tmp_ptr_shutdown_boolean(new std::atomic<bool>);

            if(tmp_ptr_shutdown_boolean == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Can not allocate " + std::to_string(sizeof(std::atomic<bool>)) + " bytes at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            this->_ptr_shutdown_boolean = tmp_ptr_shutdown_boolean;

            this->_ptr_shutdown_boolean->store(false);

            return(true);
        }
        
        bool Neural_Network_Manager::Write_File(enum MyEA::Common::ENUM_TYPE_FILE_LOG const type_file_log_received, std::string const &log_received) const
        {
            std::string tmp_path_root(this->_path_root);

            PRINT_FORMAT("%s", log_received.c_str());

            if(tmp_path_root.empty())
            {
                PRINT_FORMAT("%s: %s: ERROR: The root of the path is indefinite. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            switch(type_file_log_received)
            {
                case MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_LOG: tmp_path_root += std::string(ESCAPE_FILE) + "LOG" + ESCAPE_FILE + "Log_" + MyEA::Time::Date_Now() + ".log"; break;
                case MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR: tmp_path_root += std::string(ESCAPE_FILE) + "ERROR" + ESCAPE_FILE + "Error_" + MyEA::Time::Date_Now() + ".log"; break;
                case MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_DEBUG: tmp_path_root += std::string(ESCAPE_FILE) + "DEBUG" + ESCAPE_FILE + "Debug_" + MyEA::Time::Date_Now() + ".log"; break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Type file log (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             type_file_log_received,
                                             MyEA::Common::ENUM_TYPE_FILE_LOG_NAMES[type_file_log_received].c_str(),
                                             __LINE__);
                        return(false);
            }

            return(MyEA::File::Write_File(tmp_path_root, log_received, std::ios::out | std::ios::app | std::ios::binary));
        }

    #if defined(COMPILE_CUDA)    
        void Neural_Network_Manager::Set__Use__CUDA(bool const use_CUDA_received) { this->_use_CUDA = use_CUDA_received; }
    #endif
        
        void Neural_Network_Manager::Set__Auto_Save_Dataset(bool const auto_save_received) { this->_auto_save_dataset = auto_save_received; }
        
        void Neural_Network_Manager::Set__Optimization_Auto_Save_Trainer(bool const auto_save_received) { this->_optimization_auto_save_trainer = auto_save_received; }

        void Neural_Network_Manager::Set__Optimization_Auto_Save_Competitor(bool const auto_save_received) { this->_optimization_auto_save_competitor = auto_save_received; }
        
        void Neural_Network_Manager::Set__Optimization_Auto_Save_Trained(bool const auto_save_received) { this->_optimization_auto_save_trained = auto_save_received; }

        void Neural_Network_Manager::Set__Comparison_Expiration(size_t const expiration_seconds_received) { this->_expiration_seconds = expiration_seconds_received; }

        bool Neural_Network_Manager::Set__Output_Mode(bool const use_last_layer_as_output_received, enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received)
        {
            class Neural_Network *tmp_ptr_Neural_Network;
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: tmp_ptr_Neural_Network = this->_ptr_trainer_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: tmp_ptr_Neural_Network = this->_ptr_trained_Neural_Network; break;
                default:
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ + ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                        return(false);
            }

            return(tmp_ptr_Neural_Network->Set__Output_Mode(use_last_layer_as_output_received));
        }

        bool Neural_Network_Manager::Set__While_Condition_Optimization(struct MyEA::Common::While_Condition &ref_while_condition_received)
        {
            if(ref_while_condition_received.type_while_condition == MyEA::Common::ENUM_TYPE_WHILE_CONDITION::TYPE_WHILE_CONDITION_LENGTH)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: While condition type not set. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            this->_While_Condition_Optimization = ref_while_condition_received;

            return(true);
        }
        
        bool Neural_Network_Manager::Set__Number_Inputs(size_t const number_inputs_received)
        {
            if(number_inputs_received == 0_zu)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Number of inputs cannot be equal to zero. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            this->_number_inputs = number_inputs_received;

            return(true);
        }

        bool Neural_Network_Manager::Set__Number_Outputs(size_t const number_outputs_received)
        {
            if(number_outputs_received == 0_zu)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Number of outputs cannot be equal to zero. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            this->_number_outputs = number_outputs_received;

            return(true);
        }
        
        bool Neural_Network_Manager::Set__Number_Recurrent_Depth(size_t const number_recurrent_depth_received)
        {
            if(number_recurrent_depth_received == 0_zu)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Number of time predictions cannot be equal to zero. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            this->_recurrent_depth = number_recurrent_depth_received;

            return(true);
        }
        
        bool Neural_Network_Manager::Set__Desired_Loss(T_ const desired_loss_received)
        {
            if(desired_loss_received < 0.0f)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: The desired loss can not be less than zero. At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            this->_desired_loss = desired_loss_received;

            return(true);
        }
        
        bool Neural_Network_Manager::Get__On_Shutdown(void) const { return(this->_ptr_shutdown_boolean != nullptr && this->_ptr_shutdown_boolean->load()); }

        bool Neural_Network_Manager::Get__Require_Testing(void) const { return(this->_require_testing); }
        
        // TODO: Merge to Dataset class.
        bool Neural_Network_Manager::Get__Is_Output_Symmetric(void) const { return(false); }
        
        bool Neural_Network_Manager::Get__Path_Neural_Network_Exist(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received) const
        { return(MyEA::File::Path_Exist(this->Get__Path_Neural_Network(type_neural_network_use_received, "net")) && MyEA::File::Path_Exist(this->Get__Path_Neural_Network(type_neural_network_use_received, "nn"))); }

        size_t Neural_Network_Manager::Get__Number_Inputs(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received) const
        {
            class Neural_Network *tmp_ptr_Neural_Network;
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: tmp_ptr_Neural_Network = this->_ptr_trainer_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: tmp_ptr_Neural_Network = this->_ptr_trained_Neural_Network; break;
                default:
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ + ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                        return(0_zu);
            }

            if(tmp_ptr_Neural_Network != nullptr) { return(tmp_ptr_Neural_Network->number_inputs); }
            else { return(this->_number_inputs); }
        }

        size_t Neural_Network_Manager::Get__Number_Outputs(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received) const
        {
            class Neural_Network *tmp_ptr_Neural_Network;
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: tmp_ptr_Neural_Network = this->_ptr_trainer_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: tmp_ptr_Neural_Network = this->_ptr_trained_Neural_Network; break;
                default:
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ + ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                        return(0_zu);
            }

            if(tmp_ptr_Neural_Network != nullptr) { return(tmp_ptr_Neural_Network->Get__Output_Size()); }
            else { return(this->_number_outputs); }
        }
        
        size_t Neural_Network_Manager::Get__Number_Recurrent_Depth(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received) const
        {
            class Neural_Network *tmp_ptr_Neural_Network;
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: tmp_ptr_Neural_Network = this->_ptr_trainer_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: tmp_ptr_Neural_Network = this->_ptr_trained_Neural_Network; break;
                default:
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ + ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                        return(0_zu);
            }

            if(tmp_ptr_Neural_Network != nullptr) { return(tmp_ptr_Neural_Network->number_recurrent_depth); }
            else { return(this->_recurrent_depth); }
        }

        T_ Neural_Network_Manager::Get__Loss(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received, enum MyEA::Common::ENUM_TYPE_DATASET const type_loss_received) const
        {
            class Neural_Network *tmp_ptr_Neural_Network;
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: tmp_ptr_Neural_Network = this->_ptr_trainer_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: tmp_ptr_Neural_Network = this->_ptr_trained_Neural_Network; break;
                default:
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ + ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                        return(1.0f);
            }

            if(tmp_ptr_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ + ": ERROR: Neural network is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(1.0f);
            }
            else { return(tmp_ptr_Neural_Network->Get__Loss(type_loss_received)); }
        }
        
        T_ Neural_Network_Manager::Get__Accuracy(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received, enum MyEA::Common::ENUM_TYPE_DATASET const type_accuracy_received) const
        {
            class Neural_Network *tmp_ptr_Neural_Network;
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: tmp_ptr_Neural_Network = this->_ptr_trainer_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: tmp_ptr_Neural_Network = this->_ptr_trained_Neural_Network; break;
                default:
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ + ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                        return(0.0f);
            }

            if(tmp_ptr_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ + ": ERROR: Neural network is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(0.0f);
            }
            else { return(tmp_ptr_Neural_Network->Get__Accuracy(type_accuracy_received)); }
        }

        T_ const *const Neural_Network_Manager::Get__Output(size_t const time_step_index_received, enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received) const
        {
            class Neural_Network *tmp_ptr_Neural_Network;
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: tmp_ptr_Neural_Network = this->_ptr_trainer_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: tmp_ptr_Neural_Network = this->_ptr_trained_Neural_Network; break;
                default:
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ + ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                        return(nullptr);
            }

            if(tmp_ptr_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Neural network is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(nullptr);
            }
            else if(time_step_index_received >= tmp_ptr_Neural_Network->number_recurrent_depth)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Time step receive as arguments is out of range. (" + std::to_string(time_step_index_received) + " >= " + std::to_string(tmp_ptr_Neural_Network->number_recurrent_depth) + ")" NEW_LINE);
                
                return(nullptr);
            }
            else { return(tmp_ptr_Neural_Network->Get__Outputs(0_zu, time_step_index_received)); }
        }

        std::string Neural_Network_Manager::Get__Path_Neural_Network(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received, std::string const path_postfix_received) const
        {
            std::string tmp_path;
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: tmp_path = this->_path_model_trainer + "." + path_postfix_received; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: tmp_path = this->_path_model_trained + "." + path_postfix_received; break;
                default:
                    tmp_path = "";
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ + ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                        break;
            }

            return(tmp_path);
        }
        
        std::string Neural_Network_Manager::Get__Path_Dataset_Manager(void) const { return(this->_path_dataset); }
        
        std::string Neural_Network_Manager::Get__Path_Dataset_Manager_History(void) const { return(this->_path_dataset_history); }
        
        class Dataset_Manager<T_> *Neural_Network_Manager::Get__Dataset_Manager(void) { return(this->_ptr_Dataset_Manager); }
        
        class Neural_Network *Neural_Network_Manager::Get__Neural_Network(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received)
        {
            class Neural_Network *tmp_ptr_Neural_Network;
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: tmp_ptr_Neural_Network = this->_ptr_trainer_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: tmp_ptr_Neural_Network = this->_ptr_trained_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_COMPETITOR: tmp_ptr_Neural_Network = this->_ptr_competitor_Neural_Network; break;
                default:
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                        return(nullptr);
            }

            return(tmp_ptr_Neural_Network);
        }
        
        bool Neural_Network_Manager::Append_To_Dataset_History(T_ const *const ptr_array_inputs_received, T_ const *const ptr_array_outputs_received)
        {
            return(Append_To_Dataset_File(this->Get__Number_Inputs(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER),
                                                          this->Get__Number_Outputs(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER),
                                                          this->Get__Number_Recurrent_Depth(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER),
                                                          ptr_array_inputs_received,
                                                          ptr_array_outputs_received,
                                                          this->_path_dataset_history));
        }

        bool Neural_Network_Manager::Append_To_Dataset(T_ const *const ptr_array_inputs_received, T_ const *const ptr_array_outputs_received)
        {
            if(ptr_array_inputs_received == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Array of inputs receive as arguments is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(ptr_array_outputs_received == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Array of outputs receive as arguments is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->_ptr_Dataset_Manager == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"Dataset_Manager<T>\" is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->Get__On_Shutdown())
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_LOG, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": INFO: On shutdown. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            if(this->_ptr_Dataset_Manager->Push_Back(ptr_array_inputs_received, ptr_array_outputs_received) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Push_Back(ptr, ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            if(this->_auto_save_dataset && this->_ptr_Dataset_Manager->Save(this->_path_dataset, true) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Save(" + this->_path_dataset + ", true)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            if(this->_ptr_trainer_Neural_Network != nullptr) { this->_ptr_trainer_Neural_Network->Clear_Training_Arrays(); }
            
            this->_require_testing = true;

            return(true);
        }
        
        bool Neural_Network_Manager::Check_Expiration(void)
        {
            if(std::chrono::system_clock::now() >= this->_competitor_expiration)
            {
                // Weights_Initializer.
                // TODO: Genetic algorithm.
                struct Weights_Initializer tmp_Weights_Initializer;

                // TODO: Hyperparameter weights initializer.
                tmp_Weights_Initializer.type_weights_initializer = MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_ORTHOGONAL;
                
                if(tmp_Weights_Initializer.Output_Initialize(this->_ptr_trainer_Neural_Network) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Output_Initialize(ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    return(false);
                }
                // |END| Weights_Initializer. |END|
                
                this->_ptr_trainer_Neural_Network->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, (std::numeric_limits<ST_>::max)());
                this->_ptr_trainer_Neural_Network->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, (std::numeric_limits<ST_>::max)());
                this->_ptr_trainer_Neural_Network->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, (std::numeric_limits<ST_>::max)());
                
                this->_ptr_trainer_Neural_Network->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, 0_T);
                this->_ptr_trainer_Neural_Network->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, 0_T);
                this->_ptr_trainer_Neural_Network->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, 0_T);

                this->_ptr_competitor_Neural_Network->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, (std::numeric_limits<ST_>::max)());
                this->_ptr_competitor_Neural_Network->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, (std::numeric_limits<ST_>::max)());
                this->_ptr_competitor_Neural_Network->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, (std::numeric_limits<ST_>::max)());
                
                this->_ptr_competitor_Neural_Network->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, 0_T);
                this->_ptr_competitor_Neural_Network->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, 0_T);
                this->_ptr_competitor_Neural_Network->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, 0_T);

                this->_competitor_expiration = std::chrono::system_clock::now() + std::chrono::seconds(this->_expiration_seconds);

                return(true);
            }

            return(false);
        }

        bool Neural_Network_Manager::Testing(void)
        {
            if(this->_ptr_Dataset_Manager == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"Dataset_Manager<T>\" is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            if(this->_ptr_trainer_Neural_Network != nullptr && this->Testing(this->_ptr_trainer_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            if(this->_ptr_competitor_Neural_Network != nullptr && this->Testing(this->_ptr_competitor_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            if(this->_ptr_trained_Neural_Network != nullptr && this->Testing(this->_ptr_trained_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            this->_require_testing = false;

            return(true);
        }
        
        bool Neural_Network_Manager::Testing__Pre_Training(void)
        {
            if(this->_ptr_Dataset_Manager == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"Dataset_Manager<T>\" is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            if(this->_ptr_trainer_Neural_Network != nullptr && this->Testing__Pre_Training(this->_ptr_trainer_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            if(this->_ptr_competitor_Neural_Network != nullptr && this->Testing__Pre_Training(this->_ptr_competitor_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            if(this->_ptr_trained_Neural_Network != nullptr && this->Testing__Pre_Training(this->_ptr_trained_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            this->_require_testing = false;

            return(true);
        }
        
        bool Neural_Network_Manager::Testing(class Neural_Network *const ptr_neural_network_received)
        {
            if(this->_ptr_Dataset_Manager == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"Dataset_Manager<T>\" is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(ptr_neural_network_received == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"ptr_neural_network_received\" is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            this->_ptr_Dataset_Manager->Testing_On_Storage(ptr_neural_network_received);

            return(true);
        }
        
        bool Neural_Network_Manager::Testing__Pre_Training(class Neural_Network *const ptr_neural_network_received)
        {
            if(this->_ptr_Dataset_Manager == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"Dataset_Manager<T>\" is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(ptr_neural_network_received == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"ptr_neural_network_received\" is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            if(ptr_neural_network_received->Set__Pre_Training_Level(1_zu) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            this->_ptr_Dataset_Manager->Testing_On_Storage(ptr_neural_network_received);
            
            if(ptr_neural_network_received->Set__Pre_Training_Level(0_zu) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            return(true);
        }
        
        bool Neural_Network_Manager::Testing_If_Require(void)
        {
            if(this->Get__Require_Testing() && this->Testing() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Testing()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                return(false);
            }

            return(true);
        }

        bool Neural_Network_Manager::Testing_If_Require__Pre_Training(void)
        {
            if(this->Get__Require_Testing())
            {
                if(this->Testing__Pre_Training() == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Testing()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }

                this->_require_testing = false;
            }

            return(true);
        }

        bool Neural_Network_Manager::Compare_Trained(void)
        {
            bool const tmp_updated(this->_ptr_trained_Neural_Network->Compare(this->_ptr_Dataset_Manager->Use__Metric_Loss(),
                                                                                                                   this->_ptr_Dataset_Manager->Get__Dataset_In_Equal_Less_Holdout_Accepted(),
                                                                                                                   this->_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(),
                                                                                                                   this->_ptr_Dataset_Manager->Get__Minimum_Loss_Holdout_Accepted(),
                                                                                                                   this->_ptr_competitor_Neural_Network));

            if(tmp_updated)
            {
                if(this->_ptr_trained_Neural_Network->Update(*this->_ptr_competitor_Neural_Network, true) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                                ": ERROR: An error has been triggered from the \"Update(ptr, true)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }
                
                if(this->_optimization_auto_save_trained && this->Save_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Save_Neural_Network(" + MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED] + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                    return(false);
                }

                this->_competitor_expiration = std::chrono::system_clock::now() + std::chrono::seconds(this->_expiration_seconds);
            }

            return(tmp_updated);
        }

        bool Neural_Network_Manager::Compare_Trained__Pre_Training(void)
        {
            // Enable last pre-training mode.
            if(this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level((this->_ptr_competitor_Neural_Network->total_layers - 3_zu) / 2_zu + 1_zu) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->_ptr_trained_Neural_Network->Set__Pre_Training_Level((this->_ptr_trained_Neural_Network->total_layers - 3_zu) / 2_zu + 1_zu) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            // Evaluation.
            if(this->Testing(this->_ptr_competitor_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                return(false);
            }
            else if(this->Testing(this->_ptr_trained_Neural_Network) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                return(false);
            }
            
            // Compare.
            bool const tmp_updated(this->_ptr_trained_Neural_Network->Compare(this->_ptr_Dataset_Manager->Use__Metric_Loss(),
                                                                                                                  this->_ptr_Dataset_Manager->Get__Dataset_In_Equal_Less_Holdout_Accepted(),
                                                                                                                  this->_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(),
                                                                                                                  this->_ptr_Dataset_Manager->Get__Minimum_Loss_Holdout_Accepted(),
                                                                                                                  this->_ptr_competitor_Neural_Network));
            
            // Disable pre-training mode.
            if(this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(0_zu) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->_ptr_trained_Neural_Network->Set__Pre_Training_Level(0_zu) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            if(tmp_updated)
            {
                if(this->_ptr_trained_Neural_Network->Update(*this->_ptr_competitor_Neural_Network, true) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                                ": ERROR: An error has been triggered from the \"Update(ptr, true)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }
                
                if(this->_optimization_auto_save_trained && this->Save_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Save_Neural_Network(" + MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE_NAMES[MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED] + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                    return(false);
                }

                this->_competitor_expiration = std::chrono::system_clock::now() + std::chrono::seconds(this->_expiration_seconds);
            }

            return(tmp_updated);
        }
        
        bool Neural_Network_Manager::Pre_Training(void)
        {
            if(this->_ptr_trainer_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Trainer neural network is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->_ptr_competitor_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Competitor neural network is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->_ptr_Dataset_Manager == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"Dataset_Manager<T>\" is not initialized. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->Get__On_Shutdown())
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_LOG, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": INFO: On shutdown. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            this->Check_Expiration();

            std::chrono::system_clock::time_point const tmp_expiration(this->_While_Condition_Optimization.expiration);

            // Loop through each pre-training level.
            for(size_t tmp_optimization_time_level,
                          tmp_pre_training_end((this->_ptr_trainer_Neural_Network->total_layers - 3_zu) / 2_zu + 2_zu),
                          tmp_pre_training_level(1_zu); tmp_pre_training_level != tmp_pre_training_end && this->Get__On_Shutdown() == false; ++tmp_pre_training_level)
            {
                tmp_optimization_time_level = static_cast<size_t>(std::chrono::duration_cast<std::chrono::seconds>(tmp_expiration - std::chrono::system_clock::now()).count()) / (tmp_pre_training_end - tmp_pre_training_level);
                this->_While_Condition_Optimization.expiration = std::chrono::system_clock::now() + std::chrono::seconds(tmp_optimization_time_level);

                if(this->_ptr_trainer_Neural_Network->Set__Pre_Training_Level(tmp_pre_training_level) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }
                else if(this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(tmp_pre_training_level) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }

            #if defined(COMPILE_UI)
                MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS);

                MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY);

                if(this->_ptr_Dataset_Manager->Plot__Dataset_Manager__Pre_Training(this->_ptr_trainer_Neural_Network) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Dataset_Manager__Pre_Training(ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
            #endif
                
                if(this->Testing(this->_ptr_trainer_Neural_Network) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }
                else if(this->Testing(this->_ptr_competitor_Neural_Network) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }

                // If is not the first pre-training level. Train the neural network based on the previously best parameters found.
                if(tmp_pre_training_level != 1_zu && this->_ptr_trainer_Neural_Network->Update(*this->_ptr_competitor_Neural_Network, true) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Update(ptr, true)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                }
                
                // Reset training array(s).
                this->_ptr_trainer_Neural_Network->Clear_Training_Arrays();

                // If use hyperparameter optimization. Reset...
                this->_ptr_Dataset_Manager->Reset();

                this->_ptr_Dataset_Manager->Optimization(this->_While_Condition_Optimization,
                                                                               this->_optimization_auto_save_trainer,
                                                                               false,
                                                                               this->_desired_loss,
                                                                               this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER, "net"),
                                                                               this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER, "nn"),
                                                                               "",
                                                                               "",
                                                                               this->_ptr_trainer_Neural_Network,
                                                                               this->_ptr_competitor_Neural_Network);
            }
            
            // Disable pre-training mode.
            if(this->_ptr_trainer_Neural_Network->Set__Pre_Training_Level(0_zu) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(0_zu) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            return(true);
        }
        
        bool Neural_Network_Manager::Pre_Training(std::vector<size_t> const &ref_vector_epochs_per_pre_training_level_received)
        {
            if(ref_vector_epochs_per_pre_training_level_received.empty())
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Vector epochs can not be empty. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->_ptr_trainer_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Trainer neural network is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->_ptr_competitor_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Competitor neural network is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->_ptr_Dataset_Manager == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"Dataset_Manager<T>\" is not initialized. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->Get__On_Shutdown())
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_LOG, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": INFO: On shutdown. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            this->Check_Expiration();

            struct MyEA::Common::While_Condition tmp_while_condition;

            tmp_while_condition.type_while_condition = MyEA::Common::ENUM_TYPE_WHILE_CONDITION::TYPE_WHILE_CONDITION_ITERATION;

            // Loop through each pre-training level.
            for(size_t tmp_pre_training_end((this->_ptr_trainer_Neural_Network->total_layers - 3_zu) / 2_zu + 2_zu),
                          tmp_pre_training_level(1_zu); tmp_pre_training_level != tmp_pre_training_end && this->Get__On_Shutdown() == false; ++tmp_pre_training_level)
            {
                if(this->_ptr_trainer_Neural_Network->Set__Pre_Training_Level(tmp_pre_training_level) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }
                else if(this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(tmp_pre_training_level) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }

            #if defined(COMPILE_UI)
                MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_LOSS);

                MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_ACCURACY);

                if(this->_ptr_Dataset_Manager->Plot__Dataset_Manager__Pre_Training(this->_ptr_trainer_Neural_Network) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Dataset_Manager__Pre_Training(ptr)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
            #endif
                
                if(this->Testing(this->_ptr_trainer_Neural_Network) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }
                else if(this->Testing(this->_ptr_competitor_Neural_Network) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Testing(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }

                // Reset training array(s).
                this->_ptr_trainer_Neural_Network->Clear_Training_Arrays();

                // If is not the first pre-training level. Train the neural network based on the previously best parameters found.
                if(tmp_pre_training_level != 1_zu && this->_ptr_trainer_Neural_Network->Update(*this->_ptr_competitor_Neural_Network, true) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Update(ptr, true)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                }

                tmp_while_condition.maximum_iterations = this->_ptr_trainer_Neural_Network->pre_training_level <= ref_vector_epochs_per_pre_training_level_received.size() ? ref_vector_epochs_per_pre_training_level_received.at(this->_ptr_trainer_Neural_Network->pre_training_level - 1_zu) : ref_vector_epochs_per_pre_training_level_received.at(ref_vector_epochs_per_pre_training_level_received.size() - 1_zu);
                
                // If use hyperparameter optimization. Reset...
                this->_ptr_Dataset_Manager->Reset();

                this->_ptr_Dataset_Manager->Optimization(tmp_while_condition,
                                                                               this->_optimization_auto_save_trainer,
                                                                               false,
                                                                               this->_desired_loss,
                                                                               this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER, "net"),
                                                                               this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER, "nn"),
                                                                               "",
                                                                               "",
                                                                               this->_ptr_trainer_Neural_Network,
                                                                               this->_ptr_competitor_Neural_Network);
            }
            
            // Disable pre-training mode.
            if(this->_ptr_trainer_Neural_Network->Set__Pre_Training_Level(0_zu) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(0_zu) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Set__Pre_Training_Level()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            return(true);
        }
        
        T_ Neural_Network_Manager::Optimization(void)
        {
            if(this->_ptr_trainer_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Trainer neural network is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return((std::numeric_limits<ST_>::max)());
            }
            else if(this->_ptr_competitor_Neural_Network == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Competitor neural network is a nullptr. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return((std::numeric_limits<ST_>::max)());
            }
            else if(this->_ptr_Dataset_Manager == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Host pointer \"Dataset_Manager<T>\" is not initialized. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return((std::numeric_limits<ST_>::max)());
            }
            else if(this->Get__On_Shutdown())
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_LOG, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": INFO: On shutdown. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return((std::numeric_limits<ST_>::max)());
            }
            
            this->Check_Expiration();

            // If use hyperparameter optimization. Reset...
            this->_ptr_Dataset_Manager->Reset();

            this->_ptr_Dataset_Manager->Optimization(this->_While_Condition_Optimization,
                                                                           this->_optimization_auto_save_trainer,
                                                                           this->_optimization_auto_save_competitor,
                                                                           this->_desired_loss,
                                                                           this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER, "net"),
                                                                           this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER, "nn"),
                                                                           this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED, "net"),
                                                                           this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED, "nn"),
                                                                           this->_ptr_trainer_Neural_Network,
                                                                           this->_ptr_competitor_Neural_Network);
            
            return(this->_ptr_trainer_Neural_Network->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
        }
        
    #if defined(COMPILE_CUDA)
        bool Neural_Network_Manager::Load_Neural_Network(enum Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received,
                                                                                       size_t const maximum_allowable_host_memory_bytes_received,
                                                                                       size_t const maximum_allowable_device_memory_bytes_received,
                                                                                       bool const copy_to_competitor_received)
    #else
        bool Neural_Network_Manager::Load_Neural_Network(enum Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received,
                                                                                       size_t const maximum_allowable_host_memory_bytes_received,
                                                                                       bool const copy_to_competitor_received)
    #endif
        {
            class Neural_Network *tmp_ptr_Neural_Network;
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: tmp_ptr_Neural_Network = this->_ptr_trainer_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: tmp_ptr_Neural_Network = this->_ptr_trained_Neural_Network; break;
                default:
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                        return(false);
            }

            if(tmp_ptr_Neural_Network != nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: The neural network of type (" + std::to_string(type_neural_network_use_received) + ") is already loaded. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            std::string const tmp_path_net(this->Get__Path_Neural_Network(type_neural_network_use_received, "net"));

            if(MyEA::File::Path_Exist(tmp_path_net) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Could not find the following path " + tmp_path_net + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            std::string const tmp_path_nn(this->Get__Path_Neural_Network(type_neural_network_use_received, "nn"));

            if(MyEA::File::Path_Exist(tmp_path_nn) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Could not find the following path " + tmp_path_nn + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            
            if(sizeof(class Neural_Network) > maximum_allowable_host_memory_bytes_received)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Can not allocate " + std::to_string(sizeof(class Neural_Network)) + " bytes at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            if((tmp_ptr_Neural_Network = new class Neural_Network) == nullptr)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: Can not allocate " + std::to_string(sizeof(class Neural_Network)) + " bytes at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                return(false);
            }

            if(tmp_ptr_Neural_Network->Load(tmp_path_net,
                                                             tmp_path_nn,
                                                             maximum_allowable_host_memory_bytes_received) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Load(" + tmp_path_net + ", " + tmp_path_nn + ", " + std::to_string(maximum_allowable_host_memory_bytes_received) + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                SAFE_DELETE(tmp_ptr_Neural_Network);

                return(false);
            }
            
            switch(type_neural_network_use_received)
            {
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER: this->_ptr_trainer_Neural_Network = tmp_ptr_Neural_Network; break;
                case MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED: this->_ptr_trained_Neural_Network = tmp_ptr_Neural_Network; break;
                default:
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: Type neural network to use (" + std::to_string(type_neural_network_use_received) + ") is not managed in the switch. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    SAFE_DELETE(tmp_ptr_Neural_Network);
                        return(false);
            }

            this->_number_inputs = tmp_ptr_Neural_Network->number_inputs;

            this->_number_outputs = tmp_ptr_Neural_Network->number_outputs;

            this->_recurrent_depth = tmp_ptr_Neural_Network->number_recurrent_depth;

            if(copy_to_competitor_received)
            {
                if(this->_ptr_competitor_Neural_Network != nullptr) { SAFE_DELETE(this->_ptr_competitor_Neural_Network); }

                if((this->_ptr_competitor_Neural_Network = new class Neural_Network) == nullptr)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: Can not allocate " + std::to_string(sizeof(class Neural_Network)) + " bytes at line " + std::to_string(__LINE__) + ". At line " + std::to_string(__LINE__) + "." NEW_LINE);

                    return(false);
                }

                if(this->_ptr_competitor_Neural_Network->Copy(*tmp_ptr_Neural_Network, true) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy(ptr, true)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }
            }

            if(tmp_ptr_Neural_Network->Set__OpenMP(tmp_ptr_Neural_Network->use_OpenMP) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__OpenMP(%s)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_Neural_Network->use_OpenMP ? "true" : "false",
                                         __LINE__);

                return(false);
            }
            
        #if defined(COMPILE_CUDA)
            if(tmp_ptr_Neural_Network->Set__CUDA(tmp_ptr_Neural_Network->use_CUDA, maximum_allowable_device_memory_bytes_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__CUDA(%s, %zu)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         tmp_ptr_Neural_Network->use_CUDA ? "true" : "false",
                                         maximum_allowable_device_memory_bytes_received,
                                         __LINE__);

                return(false);
            }
        #endif

            return(true);
        }

        bool Neural_Network_Manager::Save_Neural_Network(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received)
        {
            std::string tmp_path_net,
                            tmp_path_nn;

            if(type_neural_network_use_received == MyEA::Common::TYPE_NEURAL_NETWORK_ALL || type_neural_network_use_received == MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER)
            {
                tmp_path_net = this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER, "net");

                if(this->_ptr_trainer_Neural_Network->Save_Dimension_Parameters(tmp_path_net.c_str()) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Save_Dimension_Parameters(" + tmp_path_net + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }

                tmp_path_nn = this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER, "nn");

                if(this->_ptr_trainer_Neural_Network->Save_General_Parameters(tmp_path_nn.c_str()) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Save_General_Parameters(" + tmp_path_nn + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }
            }
            
            if(type_neural_network_use_received == MyEA::Common::TYPE_NEURAL_NETWORK_ALL || type_neural_network_use_received == MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED)
            {
                tmp_path_net = this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED, "net");

                if(this->_ptr_trained_Neural_Network->Save_Dimension_Parameters(tmp_path_net.c_str()) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Save_Dimension_Parameters(" + tmp_path_net + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }
                
                tmp_path_nn = this->Get__Path_Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED, "nn");

                if(this->_ptr_trained_Neural_Network->Save_General_Parameters(tmp_path_nn.c_str()) == false)
                {
                    this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                            ": ERROR: An error has been triggered from the \"Save_General_Parameters(" + tmp_path_nn + ")\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                    
                    return(false);
                }
            }

            return(true);
        }
        
        bool Neural_Network_Manager::Assign_Shutdown_Block(class MyEA::Capturing::Shutdown &shutdown_module)
        {
            if(this->Allocate__Shutdown_Boolean() == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Allocate__Shutdown_Boolean()\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }
            else if(shutdown_module.Push_Back(this->_ptr_shutdown_boolean) == false)
            {
                this->Write_File(MyEA::Common::ENUM_TYPE_FILE_LOG::TYPE_FILE_ERROR, MyEA::Time::Date_Time_Now() + ": " + __FUNCTION__ +
                                        ": ERROR: An error has been triggered from the \"Push_Back(ptr)\" function. At line " + std::to_string(__LINE__) + "." NEW_LINE);
                
                return(false);
            }

            return(true);
        }

        void Neural_Network_Manager::Deallocate__Neural_Network(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received)
        {
            // Trainer.
            if((type_neural_network_use_received == MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_ALL || type_neural_network_use_received == MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER)
               &&
               this->_ptr_trainer_Neural_Network != nullptr)
            { SAFE_DELETE(this->_ptr_trainer_Neural_Network); }

            // Competitor.
            if((type_neural_network_use_received == MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_ALL || type_neural_network_use_received == MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINER)
               &&
               this->_ptr_competitor_Neural_Network != nullptr)
            { SAFE_DELETE(this->_ptr_competitor_Neural_Network); }
            
            // Trained.
            if((type_neural_network_use_received == MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_ALL || type_neural_network_use_received == MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED)
               &&
               this->_ptr_trained_Neural_Network != nullptr)
            { SAFE_DELETE(this->_ptr_trained_Neural_Network); }
        }

        void Neural_Network_Manager::Deallocate__Shutdown_Boolean(void) { SAFE_DELETE(this->_ptr_shutdown_boolean); }

        void Neural_Network_Manager::Deallocate__Dataset_Manager(void) { SAFE_DELETE(this->_ptr_Dataset_Manager); }
        
        Neural_Network_Manager::~Neural_Network_Manager(void)
        {
            this->Deallocate__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_ALL);

            this->Deallocate__Shutdown_Boolean();

            this->Deallocate__Dataset_Manager();
        }
    }
}