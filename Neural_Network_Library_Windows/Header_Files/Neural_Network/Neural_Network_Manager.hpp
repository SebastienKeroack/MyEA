#pragma once

#include <Configuration/Configuration.hpp>
#include <Tools/While_Condition.hpp>
#include <Enums/Enum_Type_File_Log.hpp>
#include <Enums/Enum_Type_Neural_Network_Use.hpp>
#include <Neural_Network/Data.hpp>

#include <string>
#include <chrono>
#include <atomic>

namespace MyEA
{
    namespace Neural_Network
    {
        class Neural_Network_Manager
        {
            public:
                Neural_Network_Manager(void);
                ~Neural_Network_Manager(void);
                
                void Set__Auto_Save_Dataset(bool const auto_save_received);
                void Set__Optimization_Auto_Save_Trainer(bool const auto_save_received);
                void Set__Optimization_Auto_Save_Competitor(bool const auto_save_received);
                void Set__Optimization_Auto_Save_Trained(bool const auto_save_received);
                void Set__Comparison_Expiration(size_t const expiration_seconds_received);
                void Deallocate__Neural_Network(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received);
                void Deallocate__Shutdown_Boolean(void);
                void Deallocate__Dataset_Manager(void);

                bool Set__Output_Mode(bool const use_last_layer_as_output_received, enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received);
                bool Set__While_Condition_Optimization(struct MyEA::Common::While_Condition &ref_while_condition_received);
                bool Set__Number_Inputs(size_t const number_inputs_received);
                bool Set__Number_Outputs(size_t const number_outputs_received);
                bool Set__Number_Recurrent_Depth(size_t const number_recurrent_depth_received);
                bool Set__Desired_Loss(T_ const desired_loss_received);

                // [     GET      ]
                bool Get__On_Shutdown(void) const;
                bool Get__Require_Testing(void) const;
                bool Get__Is_Output_Symmetric(void) const;
                bool Get__Path_Neural_Network_Exist(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received) const;

                size_t Get__Number_Inputs(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received) const;
                size_t Get__Number_Outputs(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received) const;
                size_t Get__Number_Recurrent_Depth(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received) const;
                
                T_ Get__Loss(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received, enum MyEA::Common::ENUM_TYPE_DATASET const type_loss_received) const;
                T_ Get__Accuracy(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received, enum MyEA::Common::ENUM_TYPE_DATASET const type_accuracy_received) const;

                T_ const *const Get__Output(size_t const time_step_index_received, enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received) const;

                std::string Get__Path_Neural_Network(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received, std::string const path_postfix_received = "net") const;
                std::string Get__Path_Dataset_Manager(void) const;
                std::string Get__Path_Dataset_Manager_History(void) const;

                class Dataset_Manager<T_> *Get__Dataset_Manager(void);

                class Neural_Network *Get__Neural_Network(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received);
                // ----- GET -----

                bool Initialize_Path(std::string const &ref_class_name_received, std::string const &ref_neural_network_name_received);
                bool Initialize_Dataset_Manager(struct Dataset_Manager_Parameters const *const ptr_Dataset_Manager_Parameters_received = nullptr);
                bool Create_Neural_Network(size_t const maximum_allowable_host_memory_bytes_received);
                bool Allocate__Shutdown_Boolean(void);
                bool Write_File(enum MyEA::Common::ENUM_TYPE_FILE_LOG const type_file_log_received, std::string const &log_received) const;
                bool Append_To_Dataset_History(T_ const *const ptr_array_inputs_received, T_ const *const ptr_array_outputs_received);
                bool Append_To_Dataset(T_ const *const ptr_array_input_received, T_ const *const ptr_array_output_received);
                bool Check_Expiration(void);
                bool Testing(void);
                bool Testing__Pre_Training(void);
                bool Testing(class Neural_Network *const ptr_neural_network_received);
                bool Testing__Pre_Training(class Neural_Network *const ptr_neural_network_received);
                bool Pre_Training(void);
                bool Pre_Training(std::vector<size_t> const &ref_vector_epochs_per_pre_training_level_received);
            #if defined(COMPILE_CUDA) == false
                bool Load_Neural_Network(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received,
                                                        size_t const maximum_allowable_host_memory_bytes_received,
                                                        bool const copy_to_competitor_received);
            #endif
                bool Save_Neural_Network(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received);
                bool Assign_Shutdown_Block(class MyEA::Capturing::Shutdown &shutdown_module);
                bool Testing_If_Require(void);
                bool Testing_If_Require__Pre_Training(void);
                bool Compare_Trained(void);
                bool Compare_Trained__Pre_Training(void);

            #if defined(COMPILE_CUDA)    
                void Set__Use__CUDA(bool const use_CUDA_received);

                bool Load_Neural_Network(enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE const type_neural_network_use_received,
                                                       size_t const maximum_allowable_host_memory_bytes_received,
                                                       size_t const maximum_allowable_device_memory_bytes_received,
                                                       bool const copy_to_competitor_received);
            #endif

                T_ Optimization(void);

            private:
                bool _require_testing = true;
                bool _auto_save_dataset = false;
                bool _optimization_auto_save_trainer = false;
                bool _optimization_auto_save_competitor = false;
                bool _optimization_auto_save_trained = false;
                std::atomic<bool> *_ptr_shutdown_boolean = nullptr;

                size_t _number_inputs = 0;
                size_t _number_outputs = 0;
                size_t _recurrent_depth = 1;
                size_t _expiration_seconds = 24_zu * 60_zu * 60_zu;

                T_ _desired_loss = 0.0;

                std::string _path_root = "";
                std::string _path_model_trained = "";
                std::string _path_model_trainer = "";
                std::string _path_dataset = "";
                std::string _path_dataset_history = "";
                
                struct MyEA::Common::While_Condition _While_Condition_Optimization;
                
                struct Scaler__Minimum_Maximum<T_> _scaler__minimum_maximum;

                class Neural_Network *_ptr_trainer_Neural_Network = nullptr;
                class Neural_Network *_ptr_competitor_Neural_Network = nullptr;
                class Neural_Network *_ptr_trained_Neural_Network = nullptr;

                class Dataset_Manager<T_> *_ptr_Dataset_Manager = nullptr;

                std::chrono::system_clock::time_point _competitor_expiration = std::chrono::system_clock::now() + std::chrono::seconds(this->_expiration_seconds);

            #if defined(COMPILE_CUDA)
                bool _use_CUDA = false;
            #endif
        };
    }
}