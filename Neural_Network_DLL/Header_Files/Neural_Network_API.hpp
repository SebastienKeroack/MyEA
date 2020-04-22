/* Copyright 2020 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Configuration/Configuration.hpp>

#include <vector>
#include <thread>

#include <Neural_Network/Neural_Network_Manager.hpp>

namespace MyEA
{
    namespace Neural_Network
    {
        class Threaded_Neural_Network_Manager
        {
            public:
                Threaded_Neural_Network_Manager(bool const is_type_position_long_received, enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicator_received);

                class Neural_Network_Manager neural_network_manager;

                std::thread optimizer_thread;
        };

        extern std::vector<class Threaded_Neural_Network_Manager *> *global_ptr_vector_ptr_Threaded_Neural_Network_Manager;

        DLL_API bool API__Neural_Network__Is_Loaded(void);
        
        DLL_API bool API__Neural_Network__Initialize(bool const is_type_position_long_received,
                                                                                                     unsigned int const type_indicator_received,
                                                                                                     unsigned int const time_frames_received);
        
        DLL_API bool API__Neural_Network__Initialize_Dataset_Manager(bool const is_type_position_long_received, unsigned int const type_indicator_received);
        
        DLL_API bool API__Neural_Network__Deinitialize(bool const is_type_position_long_received, unsigned int const type_indicator_received);
        
        /* MODWT: Preprocess the input(s) array with the past datapoint in the dataset.
                          The array should not be present inside the dataset!
                          Should be call sequentialy w.r.t dataset order. */
        DLL_API bool API__Neural_Network__Forward_Pass(bool const is_type_position_long_received,
                                                                                                              unsigned int const type_indicator_received,
                                                                                                              unsigned int const type_neural_network_use_received,
                                                                                                              T_ *const ptr_array_inputs_received);
        
        DLL_API bool API__Neural_Network__Append_To_Dataset_File(bool const is_type_position_long_received,
                                                                                                                             unsigned int const type_indicator_received,
                                                                                                                             T_ *ptr_array_inputs_received,
                                                                                                                             T_ *ptr_array_outputs_received);
        
        DLL_API bool API__Neural_Network__Append_To_Dataset(bool const is_type_position_long_received,
                                                                                                                       unsigned int const type_indicator_received,
                                                                                                                       T_ *const ptr_array_inputs_received,
                                                                                                                       T_ *const ptr_array_outputs_received);
        
        DLL_API bool API__Neural_Network__Write_File(bool const is_type_position_long_received,
                                                                                                        unsigned int const type_indicator_received,
                                                                                                        unsigned int const type_file_log_received,
                                                                                                        wchar_t const *const log_received);
        
        DLL_API bool API__Neural_Network__Load_Neural_Network(bool const is_type_position_long_received,
                                                                                                                         unsigned int const type_indicator_received,
                                                                                                                         unsigned int const type_neural_network_use_received);
        
        DLL_API bool API__Neural_Network__Save_Neural_Network(bool const is_type_position_long_received,
                                                                                                                         unsigned int const type_indicator_received,
                                                                                                                         unsigned int const type_neural_network_use_received);
        
        DLL_API bool API__Neural_Network__Join(bool const is_type_position_long_received, unsigned int const type_indicator_received);
        
        DLL_API bool API__Neural_Network__Optimization(bool const is_type_position_long_received, unsigned int const type_indicator_received);
        
        DLL_API int API__Neural_Network__Set__Output_Mode(bool const is_type_position_long_received,
                                                                                                                   unsigned int const type_indicator_received,
                                                                                                                   unsigned int const type_neural_network_use_received,
                                                                                                                   bool const use_last_layer_as_output_received);
        
        DLL_API int API__Neural_Network__Set__Number_Inputs(bool const is_type_position_long_received,
                                                                                                                     unsigned int const type_indicator_received,
                                                                                                                     unsigned int const number_inputs_received);
        
        DLL_API int API__Neural_Network__Set__Number_Outputs(bool const is_type_position_long_received,
                                                                                                                        unsigned int const type_indicator_received,
                                                                                                                        unsigned int const number_outputs_received);
        
        DLL_API int API__Neural_Network__Set__Number_Time_Predictions(bool const is_type_position_long_received,
                                                                                                                                     unsigned int const type_indicator_received,
                                                                                                                                     unsigned int const number_recurrent_depth_received);

        DLL_API int API__Neural_Network__Get__Is_Output_Symmetric(bool const is_type_position_long_received, unsigned int const type_indicator_received);
        
        DLL_API int API__Neural_Network__Get__Path_Neural_Network_Exist(bool const is_type_position_long_received,
                                                                                                                                        unsigned int const type_indicator_received,
                                                                                                                                        unsigned int const type_neural_network_use_received);
        
        DLL_API int API__Neural_Network__Get__Number_Inputs(bool const is_type_position_long_received, unsigned int const type_indicator_received);
        
        DLL_API int API__Neural_Network__Get__Number_Outputs(bool const is_type_position_long_received,
                                                                                                                        unsigned int const type_indicator_received,
                                                                                                                        unsigned int const type_neural_network_use_received);
        
        DLL_API int API__Neural_Network__Get__Number_Time_Predictions(bool const is_type_position_long_received, unsigned int const type_indicator_received);
        
        DLL_API float API__Neural_Network__Get__Loss(bool const is_type_position_long_received,
                                                                                                         unsigned int const type_indicator_received,
                                                                                                         unsigned int const type_neural_network_use_received,
                                                                                                         unsigned int const type_loss_received);
        
        DLL_API float API__Neural_Network__Get__Accuracy(bool const is_type_position_long_received,
                                                                                                                unsigned int const type_indicator_received,
                                                                                                                unsigned int const type_neural_network_use_received,
                                                                                                                unsigned int const type_accuracy_received);
        
        DLL_API T_ API__Neural_Network__Get__Output(bool const is_type_position_long_received,
                                                                                                         unsigned int const type_indicator_received,
                                                                                                         unsigned int const type_neural_network_use_received,
                                                                                                         unsigned int const output_index_received,
                                                                                                         unsigned int const time_step_received);

        DLL_API bool API__Neural_Network__Allocate(void);
    }
}